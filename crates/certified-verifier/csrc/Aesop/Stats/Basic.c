// Lean compiler output
// Module: Aesop.Stats.Basic
// Imports: public import Init public meta import Init public import Aesop.Rule.Name public import Aesop.Tracing public import Aesop.Options.Public
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
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_IO_monoNanosNow___boxed(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
extern lean_object* l_Lean_KVMap_instValueBool;
extern lean_object* lp_aesop_Aesop_aesop_collectStats;
lean_object* l_Lean_Option_get___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_KVMap_instValueString;
extern lean_object* lp_aesop_Aesop_aesop_stats_file;
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_TraceOption_stats;
lean_object* lp_aesop_Aesop_TraceOption_isEnabled___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* lp_aesop_Aesop_Nanos_printAsMillis(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
extern lean_object* lp_aesop_Aesop_instInhabitedRuleName_default;
lean_object* lean_nat_add(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson(lean_object*);
lean_object* l_Lean_JsonNumber_fromNat(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Json_mkObj(lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_get_size(lean_object*);
uint64_t lp_aesop_Aesop_instHashableDisplayRuleName_hash(lean_object*);
uint64_t lean_uint64_shift_right(uint64_t, uint64_t);
uint64_t lean_uint64_xor(uint64_t, uint64_t);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_land(size_t, size_t);
uint8_t lp_aesop_Aesop_instBEqDisplayRuleName_beq(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_instOrdNanos_ord(lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofFormat(lean_object*);
lean_object* l_ReaderT_instMonadLift___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName(lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_instOrdDisplayRuleName_ord(lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
lean_object* lean_io_get_num_heartbeats();
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_toArray___redArg(lean_object*);
extern lean_object* l_Lean_trace_profiler;
lean_object* l_Lean_PersistentArray_append___redArg(lean_object*, lean_object*);
double lean_float_sub(double, double);
uint8_t lean_float_decLt(double, double);
extern lean_object* l_Lean_trace_profiler_useHeartbeats;
extern lean_object* l_Lean_trace_profiler_threshold;
double lean_float_div(double, double);
lean_object* lean_io_mono_nanos_now();
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_modifyThe(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_instInhabitedDisplayRuleName_default;
lean_object* l_StateRefT_x27_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_instInhabitedForwardInstantiationStats_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_instInhabitedForwardInstantiationStats_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedForwardInstantiationStats_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedForwardInstantiationStats_default = (const lean_object*)&lp_aesop_Aesop_instInhabitedForwardInstantiationStats_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedForwardInstantiationStats = (const lean_object*)&lp_aesop_Aesop_instInhabitedForwardInstantiationStats_default___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Aesop_instToJsonForwardInstantiationStats_toJson_spec__0(lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "matches"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__0_value;
static const lean_string_object lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hyps"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__1 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__1_value;
static const lean_array_object lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__2 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToJsonForwardInstantiationStats___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToJsonForwardInstantiationStats___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardInstantiationStats___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToJsonForwardInstantiationStats = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardInstantiationStats___closed__0_value;
static const lean_array_object lp_aesop_Aesop_instInhabitedForwardClusterStateStats_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_instInhabitedForwardClusterStateStats_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedForwardClusterStateStats_default___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_instInhabitedForwardClusterStateStats_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_instInhabitedForwardClusterStateStats_default___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_instInhabitedForwardClusterStateStats_default___closed__1 = (const lean_object*)&lp_aesop_Aesop_instInhabitedForwardClusterStateStats_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedForwardClusterStateStats_default = (const lean_object*)&lp_aesop_Aesop_instInhabitedForwardClusterStateStats_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedForwardClusterStateStats = (const lean_object*)&lp_aesop_Aesop_instInhabitedForwardClusterStateStats_default___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardClusterStateStats_toJson_spec__0_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardClusterStateStats_toJson_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Array_toJson___at___00Aesop_instToJsonForwardClusterStateStats_toJson_spec__0(lean_object*);
static const lean_string_object lp_aesop_Aesop_instToJsonForwardClusterStateStats_toJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "slots"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardClusterStateStats_toJson___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardClusterStateStats_toJson___closed__0_value;
static const lean_string_object lp_aesop_Aesop_instToJsonForwardClusterStateStats_toJson___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "instantiationStats"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardClusterStateStats_toJson___closed__1 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardClusterStateStats_toJson___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonForwardClusterStateStats_toJson(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToJsonForwardClusterStateStats___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instToJsonForwardClusterStateStats_toJson, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToJsonForwardClusterStateStats___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardClusterStateStats___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToJsonForwardClusterStateStats = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardClusterStateStats___closed__0_value;
static const lean_array_object lp_aesop_Aesop_instInhabitedForwardRuleStateStats_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_instInhabitedForwardRuleStateStats_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedForwardRuleStateStats_default___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedForwardRuleStateStats_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedForwardRuleStateStats_default___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedForwardRuleStateStats_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedForwardRuleStateStats;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardRuleStateStats_toJson_spec__0_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardRuleStateStats_toJson_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Array_toJson___at___00Aesop_instToJsonForwardRuleStateStats_toJson_spec__0(lean_object*);
static const lean_string_object lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "ruleName"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__0_value;
static const lean_string_object lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "clusterStateStats"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__1 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__1_value;
static const lean_string_object lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__2 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__2_value;
static const lean_string_object lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__3 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__3_value;
static const lean_string_object lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__4 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__4_value;
static const lean_string_object lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__5 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__5_value;
static const lean_string_object lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__6 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__6_value;
static const lean_string_object lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__7 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__7_value;
static const lean_string_object lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__8 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__8_value;
static const lean_string_object lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__9 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__9_value;
static const lean_string_object lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__10 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__10_value;
static const lean_string_object lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__11 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__11_value;
static const lean_string_object lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__12 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__12_value;
static const lean_string_object lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__13 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__13_value;
static const lean_string_object lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__14 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__14_value;
static const lean_string_object lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__15 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__15_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToJsonForwardRuleStateStats___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardRuleStateStats___closed__0_value;
static const lean_array_object lp_aesop_Aesop_instInhabitedForwardStateStats_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_instInhabitedForwardStateStats_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedForwardStateStats_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedForwardStateStats_default = (const lean_object*)&lp_aesop_Aesop_instInhabitedForwardStateStats_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedForwardStateStats = (const lean_object*)&lp_aesop_Aesop_instInhabitedForwardStateStats_default___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardStateStats_toJson_spec__0_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardStateStats_toJson_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Array_toJson___at___00Aesop_instToJsonForwardStateStats_toJson_spec__0(lean_object*);
static const lean_string_object lp_aesop_Aesop_instToJsonForwardStateStats_toJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "ruleStateStats"};
static const lean_object* lp_aesop_Aesop_instToJsonForwardStateStats_toJson___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardStateStats_toJson___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonForwardStateStats_toJson(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToJsonForwardStateStats___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instToJsonForwardStateStats_toJson, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToJsonForwardStateStats___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardStateStats___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToJsonForwardStateStats = (const lean_object*)&lp_aesop_Aesop_instToJsonForwardStateStats___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_preNorm_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_preNorm_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_preNorm_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_preNorm_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_postNorm_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_postNorm_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_postNorm_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_postNorm_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedGoalKind_default;
LEAN_EXPORT uint8_t lp_aesop_Aesop_instInhabitedGoalKind;
static const lean_string_object lp_aesop_Aesop_instToJsonGoalKind_toJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "preNorm"};
static const lean_object* lp_aesop_Aesop_instToJsonGoalKind_toJson___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonGoalKind_toJson___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_instToJsonGoalKind_toJson___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instToJsonGoalKind_toJson___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_instToJsonGoalKind_toJson___closed__1 = (const lean_object*)&lp_aesop_Aesop_instToJsonGoalKind_toJson___closed__1_value;
static const lean_string_object lp_aesop_Aesop_instToJsonGoalKind_toJson___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "postNorm"};
static const lean_object* lp_aesop_Aesop_instToJsonGoalKind_toJson___closed__2 = (const lean_object*)&lp_aesop_Aesop_instToJsonGoalKind_toJson___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_instToJsonGoalKind_toJson___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_instToJsonGoalKind_toJson___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_instToJsonGoalKind_toJson___closed__3 = (const lean_object*)&lp_aesop_Aesop_instToJsonGoalKind_toJson___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonGoalKind_toJson(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonGoalKind_toJson___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToJsonGoalKind___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instToJsonGoalKind_toJson___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToJsonGoalKind___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonGoalKind___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToJsonGoalKind = (const lean_object*)&lp_aesop_Aesop_instToJsonGoalKind___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_instInhabitedGoalStats_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 8, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_instInhabitedForwardStateStats_default___closed__0_value),LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_instInhabitedGoalStats_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedGoalStats_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedGoalStats_default = (const lean_object*)&lp_aesop_Aesop_instInhabitedGoalStats_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedGoalStats = (const lean_object*)&lp_aesop_Aesop_instInhabitedGoalStats_default___closed__0_value;
static const lean_string_object lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "goalId"};
static const lean_object* lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__0_value;
static const lean_string_object lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "goalKind"};
static const lean_object* lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__1 = (const lean_object*)&lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__1_value;
static const lean_string_object lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "lctxSize"};
static const lean_object* lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__2 = (const lean_object*)&lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__2_value;
static const lean_string_object lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "depth"};
static const lean_object* lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__3 = (const lean_object*)&lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__3_value;
static const lean_string_object lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "forwardStateStats"};
static const lean_object* lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__4 = (const lean_object*)&lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__4_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonGoalStats_toJson(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToJsonGoalStats___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instToJsonGoalStats_toJson, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToJsonGoalStats___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonGoalStats___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToJsonGoalStats = (const lean_object*)&lp_aesop_Aesop_instToJsonGoalStats___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_instInhabitedRuleStats_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_instInhabitedRuleStats_default___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRuleStats_default;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instInhabitedRuleStats;
static const lean_string_object lp_aesop_Aesop_instToJsonRuleStats_toJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "rule"};
static const lean_object* lp_aesop_Aesop_instToJsonRuleStats_toJson___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonRuleStats_toJson___closed__0_value;
static const lean_string_object lp_aesop_Aesop_instToJsonRuleStats_toJson___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "elapsed"};
static const lean_object* lp_aesop_Aesop_instToJsonRuleStats_toJson___closed__1 = (const lean_object*)&lp_aesop_Aesop_instToJsonRuleStats_toJson___closed__1_value;
static const lean_string_object lp_aesop_Aesop_instToJsonRuleStats_toJson___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "successful"};
static const lean_object* lp_aesop_Aesop_instToJsonRuleStats_toJson___closed__2 = (const lean_object*)&lp_aesop_Aesop_instToJsonRuleStats_toJson___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonRuleStats_toJson(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToJsonRuleStats___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instToJsonRuleStats_toJson, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToJsonRuleStats___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonRuleStats___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToJsonRuleStats = (const lean_object*)&lp_aesop_Aesop_instToJsonRuleStats___closed__0_value;
static const lean_string_object lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ("};
static const lean_object* lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__0_value;
static const lean_string_object lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__1 = (const lean_object*)&lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__1_value;
static const lean_string_object lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "["};
static const lean_object* lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__2 = (const lean_object*)&lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__2_value;
static const lean_string_object lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "] apply rule "};
static const lean_object* lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__3 = (const lean_object*)&lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__3_value;
static const lean_string_object lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "<norm simp>"};
static const lean_object* lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__4 = (const lean_object*)&lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__4_value;
static const lean_string_object lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "<norm unfold>"};
static const lean_object* lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__5 = (const lean_object*)&lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__5_value;
static const lean_string_object lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "failed"};
static const lean_object* lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__6 = (const lean_object*)&lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__6_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleStats_instToString___lam__0(lean_object*);
static const lean_closure_object lp_aesop_Aesop_RuleStats_instToString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_RuleStats_instToString___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_RuleStats_instToString___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleStats_instToString___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_RuleStats_instToString = (const lean_object*)&lp_aesop_Aesop_RuleStats_instToString___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_static_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_static_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_static_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_static_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_dynamic_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_dynamic_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_dynamic_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_dynamic_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_ScriptGenerated_instInhabitedMethod_default;
LEAN_EXPORT uint8_t lp_aesop_Aesop_ScriptGenerated_instInhabitedMethod;
static const lean_string_object lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "static"};
static const lean_object* lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__0 = (const lean_object*)&lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__1 = (const lean_object*)&lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__1_value;
static const lean_string_object lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "dynamic"};
static const lean_object* lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__2 = (const lean_object*)&lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__3 = (const lean_object*)&lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_ScriptGenerated_instToJsonMethod___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_ScriptGenerated_instToJsonMethod___closed__0 = (const lean_object*)&lp_aesop_Aesop_ScriptGenerated_instToJsonMethod___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_ScriptGenerated_instToJsonMethod = (const lean_object*)&lp_aesop_Aesop_ScriptGenerated_instToJsonMethod___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToStringMethod___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToStringMethod___lam__0___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToStringMethod___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instToStringMethod___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToStringMethod___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToStringMethod___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToStringMethod = (const lean_object*)&lp_aesop_Aesop_instToStringMethod___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_instInhabitedScriptGenerated_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*0 + 8, .m_other = 0, .m_tag = 0}, .m_objs = {LEAN_SCALAR_PTR_LITERAL(0, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_aesop_Aesop_instInhabitedScriptGenerated_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedScriptGenerated_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedScriptGenerated_default = (const lean_object*)&lp_aesop_Aesop_instInhabitedScriptGenerated_default___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedScriptGenerated = (const lean_object*)&lp_aesop_Aesop_instInhabitedScriptGenerated_default___closed__0_value;
static const lean_string_object lp_aesop_Aesop_instToJsonScriptGenerated_toJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "method"};
static const lean_object* lp_aesop_Aesop_instToJsonScriptGenerated_toJson___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonScriptGenerated_toJson___closed__0_value;
static const lean_string_object lp_aesop_Aesop_instToJsonScriptGenerated_toJson___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "perfect"};
static const lean_object* lp_aesop_Aesop_instToJsonScriptGenerated_toJson___closed__1 = (const lean_object*)&lp_aesop_Aesop_instToJsonScriptGenerated_toJson___closed__1_value;
static const lean_string_object lp_aesop_Aesop_instToJsonScriptGenerated_toJson___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "hasMVar"};
static const lean_object* lp_aesop_Aesop_instToJsonScriptGenerated_toJson___closed__2 = (const lean_object*)&lp_aesop_Aesop_instToJsonScriptGenerated_toJson___closed__2_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonScriptGenerated_toJson(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonScriptGenerated_toJson___boxed(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToJsonScriptGenerated___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instToJsonScriptGenerated_toJson___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToJsonScriptGenerated___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonScriptGenerated___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToJsonScriptGenerated = (const lean_object*)&lp_aesop_Aesop_instToJsonScriptGenerated___closed__0_value;
static const lean_string_object lp_aesop_Aesop_ScriptGenerated_toString___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = " structuring"};
static const lean_object* lp_aesop_Aesop_ScriptGenerated_toString___closed__0 = (const lean_object*)&lp_aesop_Aesop_ScriptGenerated_toString___closed__0_value;
static const lean_string_object lp_aesop_Aesop_ScriptGenerated_toString___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "with "};
static const lean_object* lp_aesop_Aesop_ScriptGenerated_toString___closed__1 = (const lean_object*)&lp_aesop_Aesop_ScriptGenerated_toString___closed__1_value;
static const lean_string_object lp_aesop_Aesop_ScriptGenerated_toString___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = " "};
static const lean_object* lp_aesop_Aesop_ScriptGenerated_toString___closed__2 = (const lean_object*)&lp_aesop_Aesop_ScriptGenerated_toString___closed__2_value;
static const lean_string_object lp_aesop_Aesop_ScriptGenerated_toString___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "imperfect"};
static const lean_object* lp_aesop_Aesop_ScriptGenerated_toString___closed__3 = (const lean_object*)&lp_aesop_Aesop_ScriptGenerated_toString___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_toString(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_toString___boxed(lean_object*);
static const lean_array_object lp_aesop_Aesop_instInhabitedStats_default___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_instInhabitedStats_default___closed__0 = (const lean_object*)&lp_aesop_Aesop_instInhabitedStats_default___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_instInhabitedStats_default___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*10 + 0, .m_other = 10, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_instInhabitedStats_default___closed__0_value),((lean_object*)&lp_aesop_Aesop_instInhabitedStats_default___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_instInhabitedStats_default___closed__1 = (const lean_object*)&lp_aesop_Aesop_instInhabitedStats_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedStats_default = (const lean_object*)&lp_aesop_Aesop_instInhabitedStats_default___closed__1_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instInhabitedStats = (const lean_object*)&lp_aesop_Aesop_instInhabitedStats_default___closed__1_value;
static const lean_array_object lp_aesop_Aesop_Stats_empty___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_Stats_empty___closed__0 = (const lean_object*)&lp_aesop_Aesop_Stats_empty___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_Stats_empty___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*10 + 0, .m_other = 10, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Stats_empty___closed__0_value),((lean_object*)&lp_aesop_Aesop_Stats_empty___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_Stats_empty___closed__1 = (const lean_object*)&lp_aesop_Aesop_Stats_empty___closed__1_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Stats_empty = (const lean_object*)&lp_aesop_Aesop_Stats_empty___closed__1_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_Stats_instEmptyCollection = (const lean_object*)&lp_aesop_Aesop_Stats_empty___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_RuleStatsTotals_empty___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_RuleStatsTotals_empty___closed__0 = (const lean_object*)&lp_aesop_Aesop_RuleStatsTotals_empty___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_RuleStatsTotals_empty = (const lean_object*)&lp_aesop_Aesop_RuleStatsTotals_empty___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_RuleStatsTotals_instEmptyCollection = (const lean_object*)&lp_aesop_Aesop_RuleStatsTotals_empty___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleStatsTotals_compareByTotalElapsed___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleStatsTotals_compareByTotalElapsed___lam__0___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleStatsTotals_compareByTotalElapsed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleStatsTotals_compareByTotalElapsed___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1_spec__2_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_ruleStatsTotals_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_ruleStatsTotals_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stats_ruleStatsTotals(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stats_ruleStatsTotals___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1_spec__2_spec__5(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___redArg___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortRuleStatsTotals(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg___closed__0;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_Stats_trace_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_Stats_trace_spec__4___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stats_trace___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stats_trace___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Stats_trace_spec__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Stats_trace_spec__6___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__7(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__7___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__7___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__8(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__8___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__9(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__9___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__0;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__1;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__2;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__3;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__4;
static lean_once_cell_t lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__5;
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__6_spec__7(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__6_spec__7___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__0;
static const lean_string_object lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "<exception thrown while producing trace node message>"};
static const lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__1 = (const lean_object*)&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__1_value;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__2;
static lean_once_cell_t lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__3;
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1___closed__0 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1___closed__0_value;
static const lean_array_object lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1___closed__1 = (const lean_object*)&lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__8(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__0;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__1;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " / "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__2_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__3;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "] "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__4 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__4_value;
static lean_once_cell_t lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__5;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2(lean_object*, uint8_t, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Stats_trace_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Stats_trace_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_Stats_trace___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_Stats_trace___closed__0;
static const lean_string_object lp_aesop_Aesop_Stats_trace___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_aesop_Aesop_Stats_trace___closed__1 = (const lean_object*)&lp_aesop_Aesop_Stats_trace___closed__1_value;
static const lean_ctor_object lp_aesop_Aesop_Stats_trace___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_Stats_trace___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_aesop_Aesop_Stats_trace___closed__2 = (const lean_object*)&lp_aesop_Aesop_Stats_trace___closed__2_value;
static const lean_string_object lp_aesop_Aesop_Stats_trace___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "Forward state updates: "};
static const lean_object* lp_aesop_Aesop_Stats_trace___closed__3 = (const lean_object*)&lp_aesop_Aesop_Stats_trace___closed__3_value;
static lean_once_cell_t lp_aesop_Aesop_Stats_trace___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Stats_trace___closed__4;
static lean_once_cell_t lp_aesop_Aesop_Stats_trace___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Stats_trace___closed__5;
static lean_once_cell_t lp_aesop_Aesop_Stats_trace___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Stats_trace___closed__6;
static const lean_string_object lp_aesop_Aesop_Stats_trace___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Rule applications: "};
static const lean_object* lp_aesop_Aesop_Stats_trace___closed__7 = (const lean_object*)&lp_aesop_Aesop_Stats_trace___closed__7_value;
static lean_once_cell_t lp_aesop_Aesop_Stats_trace___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Stats_trace___closed__8;
static const lean_string_object lp_aesop_Aesop_Stats_trace___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 31, .m_capacity = 31, .m_length = 30, .m_data = " [total / successful / failed]"};
static const lean_object* lp_aesop_Aesop_Stats_trace___closed__9 = (const lean_object*)&lp_aesop_Aesop_Stats_trace___closed__9_value;
static lean_once_cell_t lp_aesop_Aesop_Stats_trace___closed__10_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Stats_trace___closed__10;
static const lean_string_object lp_aesop_Aesop_Stats_trace___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "Rule selection: "};
static const lean_object* lp_aesop_Aesop_Stats_trace___closed__11 = (const lean_object*)&lp_aesop_Aesop_Stats_trace___closed__11_value;
static lean_once_cell_t lp_aesop_Aesop_Stats_trace___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Stats_trace___closed__12;
static const lean_string_object lp_aesop_Aesop_Stats_trace___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Search: "};
static const lean_object* lp_aesop_Aesop_Stats_trace___closed__13 = (const lean_object*)&lp_aesop_Aesop_Stats_trace___closed__13_value;
static lean_once_cell_t lp_aesop_Aesop_Stats_trace___closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Stats_trace___closed__14;
static const lean_string_object lp_aesop_Aesop_Stats_trace___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Total: "};
static const lean_object* lp_aesop_Aesop_Stats_trace___closed__15 = (const lean_object*)&lp_aesop_Aesop_Stats_trace___closed__15_value;
static lean_once_cell_t lp_aesop_Aesop_Stats_trace___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Stats_trace___closed__16;
static const lean_string_object lp_aesop_Aesop_Stats_trace___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "Configuration parsing: "};
static const lean_object* lp_aesop_Aesop_Stats_trace___closed__17 = (const lean_object*)&lp_aesop_Aesop_Stats_trace___closed__17_value;
static lean_once_cell_t lp_aesop_Aesop_Stats_trace___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Stats_trace___closed__18;
static const lean_string_object lp_aesop_Aesop_Stats_trace___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 24, .m_capacity = 24, .m_length = 23, .m_data = "Rule set construction: "};
static const lean_object* lp_aesop_Aesop_Stats_trace___closed__19 = (const lean_object*)&lp_aesop_Aesop_Stats_trace___closed__19_value;
static lean_once_cell_t lp_aesop_Aesop_Stats_trace___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Stats_trace___closed__20;
static const lean_string_object lp_aesop_Aesop_Stats_trace___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Script generation: "};
static const lean_object* lp_aesop_Aesop_Stats_trace___closed__21 = (const lean_object*)&lp_aesop_Aesop_Stats_trace___closed__21_value;
static lean_once_cell_t lp_aesop_Aesop_Stats_trace___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Stats_trace___closed__22;
static const lean_string_object lp_aesop_Aesop_Stats_trace___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "Script generated: "};
static const lean_object* lp_aesop_Aesop_Stats_trace___closed__23 = (const lean_object*)&lp_aesop_Aesop_Stats_trace___closed__23_value;
static lean_once_cell_t lp_aesop_Aesop_Stats_trace___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Stats_trace___closed__24;
static const lean_string_object lp_aesop_Aesop_Stats_trace___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "no"};
static const lean_object* lp_aesop_Aesop_Stats_trace___closed__25 = (const lean_object*)&lp_aesop_Aesop_Stats_trace___closed__25_value;
static const lean_ctor_object lp_aesop_Aesop_Stats_trace___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_Stats_trace___closed__25_value)}};
static const lean_object* lp_aesop_Aesop_Stats_trace___closed__26 = (const lean_object*)&lp_aesop_Aesop_Stats_trace___closed__26_value;
static lean_once_cell_t lp_aesop_Aesop_Stats_trace___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_Stats_trace___closed__27;
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stats_trace(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stats_trace___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Stats_trace_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Stats_trace_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsReaderT___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsReaderT___redArg___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsReaderT___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsReaderT(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsOfMonadOptionsOfMonadStateOfStats___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsOfMonadOptionsOfMonadStateOfStats___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsOfMonadOptionsOfMonadStateOfStats(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_enableStatsCollection___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsCollection___redArg___lam__0___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_enableStatsCollection___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_enableStatsCollection___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsCollection___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsCollection(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsTracing___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsTracing(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsFile___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsFile___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsFile___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsFile(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStats___redArg___lam__1(lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStats___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStats___redArg___lam__0(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStats___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStats___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStats___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStats___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStats(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_profiling___redArg___lam__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_IO_monoNanosNow___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_profiling___redArg___lam__6___closed__0 = (const lean_object*)&lp_aesop_Aesop_profiling___redArg___lam__6___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__9(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRuleSelection___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRuleSelection___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRuleSelection___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRuleSelection___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRuleSelection(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRule___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRule___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRule___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingForwardState___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingForwardState___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingForwardState___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingForwardState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingForwardState(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyStatsIfEnabled___redArg___lam__0(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyStatsIfEnabled___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyStatsIfEnabled___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyStatsIfEnabled(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptGenerated___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptGenerated___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptGenerated(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Aesop_instToJsonForwardInstantiationStats_toJson_spec__0(lean_object* v_a_5_, lean_object* v_a_6_){
_start:
{
if (lean_obj_tag(v_a_5_) == 0)
{
lean_object* v___x_7_; 
v___x_7_ = lean_array_to_list(v_a_6_);
return v___x_7_;
}
else
{
lean_object* v_head_8_; lean_object* v_tail_9_; lean_object* v___x_10_; 
v_head_8_ = lean_ctor_get(v_a_5_, 0);
lean_inc(v_head_8_);
v_tail_9_ = lean_ctor_get(v_a_5_, 1);
lean_inc(v_tail_9_);
lean_dec_ref_known(v_a_5_, 2);
v___x_10_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_6_, v_head_8_);
v_a_5_ = v_tail_9_;
v_a_6_ = v___x_10_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson(lean_object* v_x_16_){
_start:
{
lean_object* v_matches_17_; lean_object* v_hyps_18_; lean_object* v___x_20_; uint8_t v_isShared_21_; uint8_t v_isSharedCheck_40_; 
v_matches_17_ = lean_ctor_get(v_x_16_, 0);
v_hyps_18_ = lean_ctor_get(v_x_16_, 1);
v_isSharedCheck_40_ = !lean_is_exclusive(v_x_16_);
if (v_isSharedCheck_40_ == 0)
{
v___x_20_ = v_x_16_;
v_isShared_21_ = v_isSharedCheck_40_;
goto v_resetjp_19_;
}
else
{
lean_inc(v_hyps_18_);
lean_inc(v_matches_17_);
lean_dec(v_x_16_);
v___x_20_ = lean_box(0);
v_isShared_21_ = v_isSharedCheck_40_;
goto v_resetjp_19_;
}
v_resetjp_19_:
{
lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_26_; 
v___x_22_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__0));
v___x_23_ = l_Lean_JsonNumber_fromNat(v_matches_17_);
v___x_24_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_24_, 0, v___x_23_);
if (v_isShared_21_ == 0)
{
lean_ctor_set(v___x_20_, 1, v___x_24_);
lean_ctor_set(v___x_20_, 0, v___x_22_);
v___x_26_ = v___x_20_;
goto v_reusejp_25_;
}
else
{
lean_object* v_reuseFailAlloc_39_; 
v_reuseFailAlloc_39_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_39_, 0, v___x_22_);
lean_ctor_set(v_reuseFailAlloc_39_, 1, v___x_24_);
v___x_26_ = v_reuseFailAlloc_39_;
goto v_reusejp_25_;
}
v_reusejp_25_:
{
lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_27_ = lean_box(0);
v___x_28_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_28_, 0, v___x_26_);
lean_ctor_set(v___x_28_, 1, v___x_27_);
v___x_29_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__1));
v___x_30_ = l_Lean_JsonNumber_fromNat(v_hyps_18_);
v___x_31_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_31_, 0, v___x_30_);
v___x_32_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_32_, 0, v___x_29_);
lean_ctor_set(v___x_32_, 1, v___x_31_);
v___x_33_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_33_, 0, v___x_32_);
lean_ctor_set(v___x_33_, 1, v___x_27_);
v___x_34_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_34_, 0, v___x_33_);
lean_ctor_set(v___x_34_, 1, v___x_27_);
v___x_35_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_35_, 0, v___x_28_);
lean_ctor_set(v___x_35_, 1, v___x_34_);
v___x_36_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__2));
v___x_37_ = lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Aesop_instToJsonForwardInstantiationStats_toJson_spec__0(v___x_35_, v___x_36_);
v___x_38_ = l_Lean_Json_mkObj(v___x_37_);
lean_dec(v___x_37_);
return v___x_38_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardClusterStateStats_toJson_spec__0_spec__0(size_t v_sz_50_, size_t v_i_51_, lean_object* v_bs_52_){
_start:
{
uint8_t v___x_53_; 
v___x_53_ = lean_usize_dec_lt(v_i_51_, v_sz_50_);
if (v___x_53_ == 0)
{
return v_bs_52_;
}
else
{
lean_object* v_v_54_; lean_object* v___x_55_; lean_object* v_bs_x27_56_; lean_object* v___x_57_; size_t v___x_58_; size_t v___x_59_; lean_object* v___x_60_; 
v_v_54_ = lean_array_uget(v_bs_52_, v_i_51_);
v___x_55_ = lean_unsigned_to_nat(0u);
v_bs_x27_56_ = lean_array_uset(v_bs_52_, v_i_51_, v___x_55_);
v___x_57_ = lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson(v_v_54_);
v___x_58_ = ((size_t)1ULL);
v___x_59_ = lean_usize_add(v_i_51_, v___x_58_);
v___x_60_ = lean_array_uset(v_bs_x27_56_, v_i_51_, v___x_57_);
v_i_51_ = v___x_59_;
v_bs_52_ = v___x_60_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardClusterStateStats_toJson_spec__0_spec__0___boxed(lean_object* v_sz_62_, lean_object* v_i_63_, lean_object* v_bs_64_){
_start:
{
size_t v_sz_boxed_65_; size_t v_i_boxed_66_; lean_object* v_res_67_; 
v_sz_boxed_65_ = lean_unbox_usize(v_sz_62_);
lean_dec(v_sz_62_);
v_i_boxed_66_ = lean_unbox_usize(v_i_63_);
lean_dec(v_i_63_);
v_res_67_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardClusterStateStats_toJson_spec__0_spec__0(v_sz_boxed_65_, v_i_boxed_66_, v_bs_64_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Array_toJson___at___00Aesop_instToJsonForwardClusterStateStats_toJson_spec__0(lean_object* v_a_68_){
_start:
{
size_t v_sz_69_; size_t v___x_70_; lean_object* v___x_71_; lean_object* v___x_72_; 
v_sz_69_ = lean_array_size(v_a_68_);
v___x_70_ = ((size_t)0ULL);
v___x_71_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardClusterStateStats_toJson_spec__0_spec__0(v_sz_69_, v___x_70_, v_a_68_);
v___x_72_ = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(v___x_72_, 0, v___x_71_);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonForwardClusterStateStats_toJson(lean_object* v_x_75_){
_start:
{
lean_object* v_slots_76_; lean_object* v_instantiationStats_77_; lean_object* v___x_79_; uint8_t v_isShared_80_; uint8_t v_isSharedCheck_98_; 
v_slots_76_ = lean_ctor_get(v_x_75_, 0);
v_instantiationStats_77_ = lean_ctor_get(v_x_75_, 1);
v_isSharedCheck_98_ = !lean_is_exclusive(v_x_75_);
if (v_isSharedCheck_98_ == 0)
{
v___x_79_ = v_x_75_;
v_isShared_80_ = v_isSharedCheck_98_;
goto v_resetjp_78_;
}
else
{
lean_inc(v_instantiationStats_77_);
lean_inc(v_slots_76_);
lean_dec(v_x_75_);
v___x_79_ = lean_box(0);
v_isShared_80_ = v_isSharedCheck_98_;
goto v_resetjp_78_;
}
v_resetjp_78_:
{
lean_object* v___x_81_; lean_object* v___x_82_; lean_object* v___x_83_; lean_object* v___x_85_; 
v___x_81_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardClusterStateStats_toJson___closed__0));
v___x_82_ = l_Lean_JsonNumber_fromNat(v_slots_76_);
v___x_83_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_83_, 0, v___x_82_);
if (v_isShared_80_ == 0)
{
lean_ctor_set(v___x_79_, 1, v___x_83_);
lean_ctor_set(v___x_79_, 0, v___x_81_);
v___x_85_ = v___x_79_;
goto v_reusejp_84_;
}
else
{
lean_object* v_reuseFailAlloc_97_; 
v_reuseFailAlloc_97_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_97_, 0, v___x_81_);
lean_ctor_set(v_reuseFailAlloc_97_, 1, v___x_83_);
v___x_85_ = v_reuseFailAlloc_97_;
goto v_reusejp_84_;
}
v_reusejp_84_:
{
lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_86_ = lean_box(0);
v___x_87_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_87_, 0, v___x_85_);
lean_ctor_set(v___x_87_, 1, v___x_86_);
v___x_88_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardClusterStateStats_toJson___closed__1));
v___x_89_ = lp_aesop_Lean_Array_toJson___at___00Aesop_instToJsonForwardClusterStateStats_toJson_spec__0(v_instantiationStats_77_);
v___x_90_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_90_, 0, v___x_88_);
lean_ctor_set(v___x_90_, 1, v___x_89_);
v___x_91_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_91_, 0, v___x_90_);
lean_ctor_set(v___x_91_, 1, v___x_86_);
v___x_92_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_92_, 0, v___x_91_);
lean_ctor_set(v___x_92_, 1, v___x_86_);
v___x_93_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_93_, 0, v___x_87_);
lean_ctor_set(v___x_93_, 1, v___x_92_);
v___x_94_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__2));
v___x_95_ = lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Aesop_instToJsonForwardInstantiationStats_toJson_spec__0(v___x_93_, v___x_94_);
v___x_96_ = l_Lean_Json_mkObj(v___x_95_);
lean_dec(v___x_95_);
return v___x_96_;
}
}
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardRuleStateStats_default___closed__1(void){
_start:
{
lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_103_ = ((lean_object*)(lp_aesop_Aesop_instInhabitedForwardRuleStateStats_default___closed__0));
v___x_104_ = lp_aesop_Aesop_instInhabitedRuleName_default;
v___x_105_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_105_, 0, v___x_104_);
lean_ctor_set(v___x_105_, 1, v___x_103_);
return v___x_105_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardRuleStateStats_default(void){
_start:
{
lean_object* v___x_106_; 
v___x_106_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedForwardRuleStateStats_default___closed__1, &lp_aesop_Aesop_instInhabitedForwardRuleStateStats_default___closed__1_once, _init_lp_aesop_Aesop_instInhabitedForwardRuleStateStats_default___closed__1);
return v___x_106_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedForwardRuleStateStats(void){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lp_aesop_Aesop_instInhabitedForwardRuleStateStats_default;
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardRuleStateStats_toJson_spec__0_spec__0(size_t v_sz_108_, size_t v_i_109_, lean_object* v_bs_110_){
_start:
{
uint8_t v___x_111_; 
v___x_111_ = lean_usize_dec_lt(v_i_109_, v_sz_108_);
if (v___x_111_ == 0)
{
return v_bs_110_;
}
else
{
lean_object* v_v_112_; lean_object* v___x_113_; lean_object* v_bs_x27_114_; lean_object* v___x_115_; size_t v___x_116_; size_t v___x_117_; lean_object* v___x_118_; 
v_v_112_ = lean_array_uget(v_bs_110_, v_i_109_);
v___x_113_ = lean_unsigned_to_nat(0u);
v_bs_x27_114_ = lean_array_uset(v_bs_110_, v_i_109_, v___x_113_);
v___x_115_ = lp_aesop_Aesop_instToJsonForwardClusterStateStats_toJson(v_v_112_);
v___x_116_ = ((size_t)1ULL);
v___x_117_ = lean_usize_add(v_i_109_, v___x_116_);
v___x_118_ = lean_array_uset(v_bs_x27_114_, v_i_109_, v___x_115_);
v_i_109_ = v___x_117_;
v_bs_110_ = v___x_118_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardRuleStateStats_toJson_spec__0_spec__0___boxed(lean_object* v_sz_120_, lean_object* v_i_121_, lean_object* v_bs_122_){
_start:
{
size_t v_sz_boxed_123_; size_t v_i_boxed_124_; lean_object* v_res_125_; 
v_sz_boxed_123_ = lean_unbox_usize(v_sz_120_);
lean_dec(v_sz_120_);
v_i_boxed_124_ = lean_unbox_usize(v_i_121_);
lean_dec(v_i_121_);
v_res_125_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardRuleStateStats_toJson_spec__0_spec__0(v_sz_boxed_123_, v_i_boxed_124_, v_bs_122_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Array_toJson___at___00Aesop_instToJsonForwardRuleStateStats_toJson_spec__0(lean_object* v_a_126_){
_start:
{
size_t v_sz_127_; size_t v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; 
v_sz_127_ = lean_array_size(v_a_126_);
v___x_128_ = ((size_t)0ULL);
v___x_129_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardRuleStateStats_toJson_spec__0_spec__0(v_sz_127_, v___x_128_, v_a_126_);
v___x_130_ = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(v___x_130_, 0, v___x_129_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson(lean_object* v_x_147_){
_start:
{
lean_object* v_ruleName_148_; lean_object* v_clusterStateStats_149_; lean_object* v___x_151_; uint8_t v_isShared_152_; uint8_t v_isSharedCheck_206_; 
v_ruleName_148_ = lean_ctor_get(v_x_147_, 0);
v_clusterStateStats_149_ = lean_ctor_get(v_x_147_, 1);
v_isSharedCheck_206_ = !lean_is_exclusive(v_x_147_);
if (v_isSharedCheck_206_ == 0)
{
v___x_151_ = v_x_147_;
v_isShared_152_ = v_isSharedCheck_206_;
goto v_resetjp_150_;
}
else
{
lean_inc(v_clusterStateStats_149_);
lean_inc(v_ruleName_148_);
lean_dec(v_x_147_);
v___x_151_ = lean_box(0);
v_isShared_152_ = v_isSharedCheck_206_;
goto v_resetjp_150_;
}
v_resetjp_150_:
{
lean_object* v_name_153_; uint8_t v_builder_154_; uint8_t v_phase_155_; uint8_t v_scope_156_; lean_object* v___x_157_; lean_object* v___y_159_; lean_object* v___y_160_; lean_object* v___y_161_; lean_object* v___y_184_; lean_object* v___y_185_; lean_object* v___y_186_; lean_object* v___y_192_; 
v_name_153_ = lean_ctor_get(v_ruleName_148_, 0);
lean_inc(v_name_153_);
v_builder_154_ = lean_ctor_get_uint8(v_ruleName_148_, sizeof(void*)*1 + 8);
v_phase_155_ = lean_ctor_get_uint8(v_ruleName_148_, sizeof(void*)*1 + 9);
v_scope_156_ = lean_ctor_get_uint8(v_ruleName_148_, sizeof(void*)*1 + 10);
lean_dec_ref(v_ruleName_148_);
v___x_157_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__0));
switch(v_phase_155_)
{
case 0:
{
lean_object* v___x_203_; 
v___x_203_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__13));
v___y_192_ = v___x_203_;
goto v___jp_191_;
}
case 1:
{
lean_object* v___x_204_; 
v___x_204_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__14));
v___y_192_ = v___x_204_;
goto v___jp_191_;
}
default: 
{
lean_object* v___x_205_; 
v___x_205_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__15));
v___y_192_ = v___x_205_;
goto v___jp_191_;
}
}
v___jp_158_:
{
lean_object* v___x_162_; lean_object* v___x_163_; uint8_t v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_170_; 
v___x_162_ = lean_string_append(v___y_159_, v___y_161_);
v___x_163_ = lean_string_append(v___x_162_, v___y_160_);
v___x_164_ = 1;
lean_inc(v_name_153_);
v___x_165_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_153_, v___x_164_);
v___x_166_ = lean_string_append(v___x_163_, v___x_165_);
lean_dec_ref(v___x_165_);
v___x_167_ = lean_alloc_ctor(0, 2, 3);
lean_ctor_set(v___x_167_, 0, v_name_153_);
lean_ctor_set(v___x_167_, 1, v___x_166_);
lean_ctor_set_uint8(v___x_167_, sizeof(void*)*2, v_builder_154_);
lean_ctor_set_uint8(v___x_167_, sizeof(void*)*2 + 1, v_phase_155_);
lean_ctor_set_uint8(v___x_167_, sizeof(void*)*2 + 2, v_scope_156_);
v___x_168_ = lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson(v___x_167_);
if (v_isShared_152_ == 0)
{
lean_ctor_set(v___x_151_, 1, v___x_168_);
lean_ctor_set(v___x_151_, 0, v___x_157_);
v___x_170_ = v___x_151_;
goto v_reusejp_169_;
}
else
{
lean_object* v_reuseFailAlloc_182_; 
v_reuseFailAlloc_182_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_182_, 0, v___x_157_);
lean_ctor_set(v_reuseFailAlloc_182_, 1, v___x_168_);
v___x_170_ = v_reuseFailAlloc_182_;
goto v_reusejp_169_;
}
v_reusejp_169_:
{
lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; 
v___x_171_ = lean_box(0);
v___x_172_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_172_, 0, v___x_170_);
lean_ctor_set(v___x_172_, 1, v___x_171_);
v___x_173_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__1));
v___x_174_ = lp_aesop_Lean_Array_toJson___at___00Aesop_instToJsonForwardRuleStateStats_toJson_spec__0(v_clusterStateStats_149_);
v___x_175_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_175_, 0, v___x_173_);
lean_ctor_set(v___x_175_, 1, v___x_174_);
v___x_176_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_176_, 0, v___x_175_);
lean_ctor_set(v___x_176_, 1, v___x_171_);
v___x_177_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_177_, 0, v___x_176_);
lean_ctor_set(v___x_177_, 1, v___x_171_);
v___x_178_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_178_, 0, v___x_172_);
lean_ctor_set(v___x_178_, 1, v___x_177_);
v___x_179_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__2));
v___x_180_ = lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Aesop_instToJsonForwardInstantiationStats_toJson_spec__0(v___x_178_, v___x_179_);
v___x_181_ = l_Lean_Json_mkObj(v___x_180_);
lean_dec(v___x_180_);
return v___x_181_;
}
}
v___jp_183_:
{
lean_object* v___x_187_; lean_object* v___x_188_; 
v___x_187_ = lean_string_append(v___y_185_, v___y_186_);
v___x_188_ = lean_string_append(v___x_187_, v___y_184_);
if (v_scope_156_ == 0)
{
lean_object* v___x_189_; 
v___x_189_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__2));
v___y_159_ = v___x_188_;
v___y_160_ = v___y_184_;
v___y_161_ = v___x_189_;
goto v___jp_158_;
}
else
{
lean_object* v___x_190_; 
v___x_190_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__3));
v___y_159_ = v___x_188_;
v___y_160_ = v___y_184_;
v___y_161_ = v___x_190_;
goto v___jp_158_;
}
}
v___jp_191_:
{
lean_object* v___x_193_; lean_object* v___x_194_; 
v___x_193_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__4));
lean_inc_ref(v___y_192_);
v___x_194_ = lean_string_append(v___y_192_, v___x_193_);
switch(v_builder_154_)
{
case 0:
{
lean_object* v___x_195_; 
v___x_195_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__5));
v___y_184_ = v___x_193_;
v___y_185_ = v___x_194_;
v___y_186_ = v___x_195_;
goto v___jp_183_;
}
case 1:
{
lean_object* v___x_196_; 
v___x_196_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__6));
v___y_184_ = v___x_193_;
v___y_185_ = v___x_194_;
v___y_186_ = v___x_196_;
goto v___jp_183_;
}
case 2:
{
lean_object* v___x_197_; 
v___x_197_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__7));
v___y_184_ = v___x_193_;
v___y_185_ = v___x_194_;
v___y_186_ = v___x_197_;
goto v___jp_183_;
}
case 3:
{
lean_object* v___x_198_; 
v___x_198_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__8));
v___y_184_ = v___x_193_;
v___y_185_ = v___x_194_;
v___y_186_ = v___x_198_;
goto v___jp_183_;
}
case 4:
{
lean_object* v___x_199_; 
v___x_199_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__9));
v___y_184_ = v___x_193_;
v___y_185_ = v___x_194_;
v___y_186_ = v___x_199_;
goto v___jp_183_;
}
case 5:
{
lean_object* v___x_200_; 
v___x_200_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__10));
v___y_184_ = v___x_193_;
v___y_185_ = v___x_194_;
v___y_186_ = v___x_200_;
goto v___jp_183_;
}
case 6:
{
lean_object* v___x_201_; 
v___x_201_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__11));
v___y_184_ = v___x_193_;
v___y_185_ = v___x_194_;
v___y_186_ = v___x_201_;
goto v___jp_183_;
}
default: 
{
lean_object* v___x_202_; 
v___x_202_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__12));
v___y_184_ = v___x_193_;
v___y_185_ = v___x_194_;
v___y_186_ = v___x_202_;
goto v___jp_183_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardStateStats_toJson_spec__0_spec__0(size_t v_sz_213_, size_t v_i_214_, lean_object* v_bs_215_){
_start:
{
uint8_t v___x_216_; 
v___x_216_ = lean_usize_dec_lt(v_i_214_, v_sz_213_);
if (v___x_216_ == 0)
{
return v_bs_215_;
}
else
{
lean_object* v_v_217_; lean_object* v___x_218_; lean_object* v_bs_x27_219_; lean_object* v___x_220_; size_t v___x_221_; size_t v___x_222_; lean_object* v___x_223_; 
v_v_217_ = lean_array_uget(v_bs_215_, v_i_214_);
v___x_218_ = lean_unsigned_to_nat(0u);
v_bs_x27_219_ = lean_array_uset(v_bs_215_, v_i_214_, v___x_218_);
v___x_220_ = lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson(v_v_217_);
v___x_221_ = ((size_t)1ULL);
v___x_222_ = lean_usize_add(v_i_214_, v___x_221_);
v___x_223_ = lean_array_uset(v_bs_x27_219_, v_i_214_, v___x_220_);
v_i_214_ = v___x_222_;
v_bs_215_ = v___x_223_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardStateStats_toJson_spec__0_spec__0___boxed(lean_object* v_sz_225_, lean_object* v_i_226_, lean_object* v_bs_227_){
_start:
{
size_t v_sz_boxed_228_; size_t v_i_boxed_229_; lean_object* v_res_230_; 
v_sz_boxed_228_ = lean_unbox_usize(v_sz_225_);
lean_dec(v_sz_225_);
v_i_boxed_229_ = lean_unbox_usize(v_i_226_);
lean_dec(v_i_226_);
v_res_230_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardStateStats_toJson_spec__0_spec__0(v_sz_boxed_228_, v_i_boxed_229_, v_bs_227_);
return v_res_230_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Array_toJson___at___00Aesop_instToJsonForwardStateStats_toJson_spec__0(lean_object* v_a_231_){
_start:
{
size_t v_sz_232_; size_t v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; 
v_sz_232_ = lean_array_size(v_a_231_);
v___x_233_ = ((size_t)0ULL);
v___x_234_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonForwardStateStats_toJson_spec__0_spec__0(v_sz_232_, v___x_233_, v_a_231_);
v___x_235_ = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(v___x_235_, 0, v___x_234_);
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonForwardStateStats_toJson(lean_object* v_x_237_){
_start:
{
lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; 
v___x_238_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardStateStats_toJson___closed__0));
v___x_239_ = lp_aesop_Lean_Array_toJson___at___00Aesop_instToJsonForwardStateStats_toJson_spec__0(v_x_237_);
v___x_240_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_240_, 0, v___x_238_);
lean_ctor_set(v___x_240_, 1, v___x_239_);
v___x_241_ = lean_box(0);
v___x_242_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_242_, 0, v___x_240_);
lean_ctor_set(v___x_242_, 1, v___x_241_);
v___x_243_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_243_, 0, v___x_242_);
lean_ctor_set(v___x_243_, 1, v___x_241_);
v___x_244_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__2));
v___x_245_ = lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Aesop_instToJsonForwardInstantiationStats_toJson_spec__0(v___x_243_, v___x_244_);
v___x_246_ = l_Lean_Json_mkObj(v___x_245_);
lean_dec(v___x_245_);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_ctorIdx(uint8_t v_x_249_){
_start:
{
if (v_x_249_ == 0)
{
lean_object* v___x_250_; 
v___x_250_ = lean_unsigned_to_nat(0u);
return v___x_250_;
}
else
{
lean_object* v___x_251_; 
v___x_251_ = lean_unsigned_to_nat(1u);
return v___x_251_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_ctorIdx___boxed(lean_object* v_x_252_){
_start:
{
uint8_t v_x_boxed_253_; lean_object* v_res_254_; 
v_x_boxed_253_ = lean_unbox(v_x_252_);
v_res_254_ = lp_aesop_Aesop_GoalKind_ctorIdx(v_x_boxed_253_);
return v_res_254_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_ctorElim___redArg(lean_object* v_k_255_){
_start:
{
lean_inc(v_k_255_);
return v_k_255_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_ctorElim___redArg___boxed(lean_object* v_k_256_){
_start:
{
lean_object* v_res_257_; 
v_res_257_ = lp_aesop_Aesop_GoalKind_ctorElim___redArg(v_k_256_);
lean_dec(v_k_256_);
return v_res_257_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_ctorElim(lean_object* v_motive_258_, lean_object* v_ctorIdx_259_, uint8_t v_t_260_, lean_object* v_h_261_, lean_object* v_k_262_){
_start:
{
lean_inc(v_k_262_);
return v_k_262_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_ctorElim___boxed(lean_object* v_motive_263_, lean_object* v_ctorIdx_264_, lean_object* v_t_265_, lean_object* v_h_266_, lean_object* v_k_267_){
_start:
{
uint8_t v_t_boxed_268_; lean_object* v_res_269_; 
v_t_boxed_268_ = lean_unbox(v_t_265_);
v_res_269_ = lp_aesop_Aesop_GoalKind_ctorElim(v_motive_263_, v_ctorIdx_264_, v_t_boxed_268_, v_h_266_, v_k_267_);
lean_dec(v_k_267_);
lean_dec(v_ctorIdx_264_);
return v_res_269_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_preNorm_elim___redArg(lean_object* v_preNorm_270_){
_start:
{
lean_inc(v_preNorm_270_);
return v_preNorm_270_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_preNorm_elim___redArg___boxed(lean_object* v_preNorm_271_){
_start:
{
lean_object* v_res_272_; 
v_res_272_ = lp_aesop_Aesop_GoalKind_preNorm_elim___redArg(v_preNorm_271_);
lean_dec(v_preNorm_271_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_preNorm_elim(lean_object* v_motive_273_, uint8_t v_t_274_, lean_object* v_h_275_, lean_object* v_preNorm_276_){
_start:
{
lean_inc(v_preNorm_276_);
return v_preNorm_276_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_preNorm_elim___boxed(lean_object* v_motive_277_, lean_object* v_t_278_, lean_object* v_h_279_, lean_object* v_preNorm_280_){
_start:
{
uint8_t v_t_boxed_281_; lean_object* v_res_282_; 
v_t_boxed_281_ = lean_unbox(v_t_278_);
v_res_282_ = lp_aesop_Aesop_GoalKind_preNorm_elim(v_motive_277_, v_t_boxed_281_, v_h_279_, v_preNorm_280_);
lean_dec(v_preNorm_280_);
return v_res_282_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_postNorm_elim___redArg(lean_object* v_postNorm_283_){
_start:
{
lean_inc(v_postNorm_283_);
return v_postNorm_283_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_postNorm_elim___redArg___boxed(lean_object* v_postNorm_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_aesop_Aesop_GoalKind_postNorm_elim___redArg(v_postNorm_284_);
lean_dec(v_postNorm_284_);
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_postNorm_elim(lean_object* v_motive_286_, uint8_t v_t_287_, lean_object* v_h_288_, lean_object* v_postNorm_289_){
_start:
{
lean_inc(v_postNorm_289_);
return v_postNorm_289_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_GoalKind_postNorm_elim___boxed(lean_object* v_motive_290_, lean_object* v_t_291_, lean_object* v_h_292_, lean_object* v_postNorm_293_){
_start:
{
uint8_t v_t_boxed_294_; lean_object* v_res_295_; 
v_t_boxed_294_ = lean_unbox(v_t_291_);
v_res_295_ = lp_aesop_Aesop_GoalKind_postNorm_elim(v_motive_290_, v_t_boxed_294_, v_h_292_, v_postNorm_293_);
lean_dec(v_postNorm_293_);
return v_res_295_;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedGoalKind_default(void){
_start:
{
uint8_t v___x_296_; 
v___x_296_ = 0;
return v___x_296_;
}
}
static uint8_t _init_lp_aesop_Aesop_instInhabitedGoalKind(void){
_start:
{
uint8_t v___x_297_; 
v___x_297_ = 0;
return v___x_297_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonGoalKind_toJson(uint8_t v_x_304_){
_start:
{
if (v_x_304_ == 0)
{
lean_object* v___x_305_; 
v___x_305_ = ((lean_object*)(lp_aesop_Aesop_instToJsonGoalKind_toJson___closed__1));
return v___x_305_;
}
else
{
lean_object* v___x_306_; 
v___x_306_ = ((lean_object*)(lp_aesop_Aesop_instToJsonGoalKind_toJson___closed__3));
return v___x_306_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonGoalKind_toJson___boxed(lean_object* v_x_307_){
_start:
{
uint8_t v_x_46__boxed_308_; lean_object* v_res_309_; 
v_x_46__boxed_308_ = lean_unbox(v_x_307_);
v_res_309_ = lp_aesop_Aesop_instToJsonGoalKind_toJson(v_x_46__boxed_308_);
return v_res_309_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonGoalStats_toJson(lean_object* v_x_323_){
_start:
{
lean_object* v_goalId_324_; uint8_t v_goalKind_325_; lean_object* v_lctxSize_326_; lean_object* v_depth_327_; lean_object* v_forwardStateStats_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; 
v_goalId_324_ = lean_ctor_get(v_x_323_, 0);
lean_inc(v_goalId_324_);
v_goalKind_325_ = lean_ctor_get_uint8(v_x_323_, sizeof(void*)*4);
v_lctxSize_326_ = lean_ctor_get(v_x_323_, 1);
lean_inc(v_lctxSize_326_);
v_depth_327_ = lean_ctor_get(v_x_323_, 2);
lean_inc(v_depth_327_);
v_forwardStateStats_328_ = lean_ctor_get(v_x_323_, 3);
lean_inc_ref(v_forwardStateStats_328_);
lean_dec_ref(v_x_323_);
v___x_329_ = ((lean_object*)(lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__0));
v___x_330_ = l_Lean_JsonNumber_fromNat(v_goalId_324_);
v___x_331_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_331_, 0, v___x_330_);
v___x_332_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_332_, 0, v___x_329_);
lean_ctor_set(v___x_332_, 1, v___x_331_);
v___x_333_ = lean_box(0);
v___x_334_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_334_, 0, v___x_332_);
lean_ctor_set(v___x_334_, 1, v___x_333_);
v___x_335_ = ((lean_object*)(lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__1));
v___x_336_ = lp_aesop_Aesop_instToJsonGoalKind_toJson(v_goalKind_325_);
v___x_337_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_337_, 0, v___x_335_);
lean_ctor_set(v___x_337_, 1, v___x_336_);
v___x_338_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_338_, 0, v___x_337_);
lean_ctor_set(v___x_338_, 1, v___x_333_);
v___x_339_ = ((lean_object*)(lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__2));
v___x_340_ = l_Lean_JsonNumber_fromNat(v_lctxSize_326_);
v___x_341_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_341_, 0, v___x_340_);
v___x_342_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_342_, 0, v___x_339_);
lean_ctor_set(v___x_342_, 1, v___x_341_);
v___x_343_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_343_, 0, v___x_342_);
lean_ctor_set(v___x_343_, 1, v___x_333_);
v___x_344_ = ((lean_object*)(lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__3));
v___x_345_ = l_Lean_JsonNumber_fromNat(v_depth_327_);
v___x_346_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_346_, 0, v___x_345_);
v___x_347_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_347_, 0, v___x_344_);
lean_ctor_set(v___x_347_, 1, v___x_346_);
v___x_348_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_348_, 0, v___x_347_);
lean_ctor_set(v___x_348_, 1, v___x_333_);
v___x_349_ = ((lean_object*)(lp_aesop_Aesop_instToJsonGoalStats_toJson___closed__4));
v___x_350_ = lp_aesop_Aesop_instToJsonForwardStateStats_toJson(v_forwardStateStats_328_);
v___x_351_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_351_, 0, v___x_349_);
lean_ctor_set(v___x_351_, 1, v___x_350_);
v___x_352_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_352_, 0, v___x_351_);
lean_ctor_set(v___x_352_, 1, v___x_333_);
v___x_353_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_353_, 0, v___x_352_);
lean_ctor_set(v___x_353_, 1, v___x_333_);
v___x_354_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_354_, 0, v___x_348_);
lean_ctor_set(v___x_354_, 1, v___x_353_);
v___x_355_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_355_, 0, v___x_343_);
lean_ctor_set(v___x_355_, 1, v___x_354_);
v___x_356_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_356_, 0, v___x_338_);
lean_ctor_set(v___x_356_, 1, v___x_355_);
v___x_357_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_357_, 0, v___x_334_);
lean_ctor_set(v___x_357_, 1, v___x_356_);
v___x_358_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__2));
v___x_359_ = lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Aesop_instToJsonForwardInstantiationStats_toJson_spec__0(v___x_357_, v___x_358_);
v___x_360_ = l_Lean_Json_mkObj(v___x_359_);
lean_dec(v___x_359_);
return v___x_360_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRuleStats_default___closed__0(void){
_start:
{
uint8_t v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; 
v___x_363_ = 0;
v___x_364_ = lean_unsigned_to_nat(0u);
v___x_365_ = lp_aesop_Aesop_instInhabitedDisplayRuleName_default;
v___x_366_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v___x_366_, 0, v___x_365_);
lean_ctor_set(v___x_366_, 1, v___x_364_);
lean_ctor_set_uint8(v___x_366_, sizeof(void*)*2, v___x_363_);
return v___x_366_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRuleStats_default(void){
_start:
{
lean_object* v___x_367_; 
v___x_367_ = lean_obj_once(&lp_aesop_Aesop_instInhabitedRuleStats_default___closed__0, &lp_aesop_Aesop_instInhabitedRuleStats_default___closed__0_once, _init_lp_aesop_Aesop_instInhabitedRuleStats_default___closed__0);
return v___x_367_;
}
}
static lean_object* _init_lp_aesop_Aesop_instInhabitedRuleStats(void){
_start:
{
lean_object* v___x_368_; 
v___x_368_ = lp_aesop_Aesop_instInhabitedRuleStats_default;
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonRuleStats_toJson(lean_object* v_x_372_){
_start:
{
lean_object* v_rule_373_; lean_object* v_elapsed_374_; uint8_t v_successful_375_; lean_object* v___x_376_; lean_object* v_name_377_; uint8_t v_builder_378_; uint8_t v_phase_379_; uint8_t v_scope_380_; lean_object* v___x_381_; lean_object* v___y_383_; lean_object* v___y_384_; lean_object* v___y_385_; lean_object* v___y_412_; lean_object* v___y_413_; lean_object* v___y_414_; lean_object* v___y_420_; 
v_rule_373_ = lean_ctor_get(v_x_372_, 0);
lean_inc(v_rule_373_);
v_elapsed_374_ = lean_ctor_get(v_x_372_, 1);
lean_inc(v_elapsed_374_);
v_successful_375_ = lean_ctor_get_uint8(v_x_372_, sizeof(void*)*2);
lean_dec_ref(v_x_372_);
v___x_376_ = lp_aesop___private_Aesop_Rule_Name_0__Aesop_DisplayRuleName_toJsonRuleName(v_rule_373_);
lean_dec(v_rule_373_);
v_name_377_ = lean_ctor_get(v___x_376_, 0);
lean_inc(v_name_377_);
v_builder_378_ = lean_ctor_get_uint8(v___x_376_, sizeof(void*)*1 + 8);
v_phase_379_ = lean_ctor_get_uint8(v___x_376_, sizeof(void*)*1 + 9);
v_scope_380_ = lean_ctor_get_uint8(v___x_376_, sizeof(void*)*1 + 10);
lean_dec_ref(v___x_376_);
v___x_381_ = ((lean_object*)(lp_aesop_Aesop_instToJsonRuleStats_toJson___closed__0));
switch(v_phase_379_)
{
case 0:
{
lean_object* v___x_431_; 
v___x_431_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__13));
v___y_420_ = v___x_431_;
goto v___jp_419_;
}
case 1:
{
lean_object* v___x_432_; 
v___x_432_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__14));
v___y_420_ = v___x_432_;
goto v___jp_419_;
}
default: 
{
lean_object* v___x_433_; 
v___x_433_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__15));
v___y_420_ = v___x_433_;
goto v___jp_419_;
}
}
v___jp_382_:
{
uint8_t v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; lean_object* v___x_408_; lean_object* v___x_409_; lean_object* v___x_410_; 
v___x_386_ = 1;
lean_inc(v_name_377_);
v___x_387_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_377_, v___x_386_);
v___x_388_ = lean_string_append(v___y_383_, v___y_385_);
v___x_389_ = lean_string_append(v___x_388_, v___y_384_);
v___x_390_ = lean_string_append(v___x_389_, v___x_387_);
lean_dec_ref(v___x_387_);
v___x_391_ = lean_alloc_ctor(0, 2, 3);
lean_ctor_set(v___x_391_, 0, v_name_377_);
lean_ctor_set(v___x_391_, 1, v___x_390_);
lean_ctor_set_uint8(v___x_391_, sizeof(void*)*2, v_builder_378_);
lean_ctor_set_uint8(v___x_391_, sizeof(void*)*2 + 1, v_phase_379_);
lean_ctor_set_uint8(v___x_391_, sizeof(void*)*2 + 2, v_scope_380_);
v___x_392_ = lp_aesop___private_Aesop_Rule_Name_0__Aesop_RuleName_instToJsonJson_toJson(v___x_391_);
v___x_393_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_393_, 0, v___x_381_);
lean_ctor_set(v___x_393_, 1, v___x_392_);
v___x_394_ = lean_box(0);
v___x_395_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_395_, 0, v___x_393_);
lean_ctor_set(v___x_395_, 1, v___x_394_);
v___x_396_ = ((lean_object*)(lp_aesop_Aesop_instToJsonRuleStats_toJson___closed__1));
v___x_397_ = l_Lean_JsonNumber_fromNat(v_elapsed_374_);
v___x_398_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_398_, 0, v___x_397_);
v___x_399_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_399_, 0, v___x_396_);
lean_ctor_set(v___x_399_, 1, v___x_398_);
v___x_400_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_400_, 0, v___x_399_);
lean_ctor_set(v___x_400_, 1, v___x_394_);
v___x_401_ = ((lean_object*)(lp_aesop_Aesop_instToJsonRuleStats_toJson___closed__2));
v___x_402_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_402_, 0, v_successful_375_);
v___x_403_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_403_, 0, v___x_401_);
lean_ctor_set(v___x_403_, 1, v___x_402_);
v___x_404_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_404_, 0, v___x_403_);
lean_ctor_set(v___x_404_, 1, v___x_394_);
v___x_405_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_405_, 0, v___x_404_);
lean_ctor_set(v___x_405_, 1, v___x_394_);
v___x_406_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_406_, 0, v___x_400_);
lean_ctor_set(v___x_406_, 1, v___x_405_);
v___x_407_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_407_, 0, v___x_395_);
lean_ctor_set(v___x_407_, 1, v___x_406_);
v___x_408_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__2));
v___x_409_ = lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Aesop_instToJsonForwardInstantiationStats_toJson_spec__0(v___x_407_, v___x_408_);
v___x_410_ = l_Lean_Json_mkObj(v___x_409_);
lean_dec(v___x_409_);
return v___x_410_;
}
v___jp_411_:
{
lean_object* v___x_415_; lean_object* v___x_416_; 
v___x_415_ = lean_string_append(v___y_412_, v___y_414_);
v___x_416_ = lean_string_append(v___x_415_, v___y_413_);
if (v_scope_380_ == 0)
{
lean_object* v___x_417_; 
v___x_417_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__2));
v___y_383_ = v___x_416_;
v___y_384_ = v___y_413_;
v___y_385_ = v___x_417_;
goto v___jp_382_;
}
else
{
lean_object* v___x_418_; 
v___x_418_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__3));
v___y_383_ = v___x_416_;
v___y_384_ = v___y_413_;
v___y_385_ = v___x_418_;
goto v___jp_382_;
}
}
v___jp_419_:
{
lean_object* v___x_421_; lean_object* v___x_422_; 
v___x_421_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__4));
lean_inc_ref(v___y_420_);
v___x_422_ = lean_string_append(v___y_420_, v___x_421_);
switch(v_builder_378_)
{
case 0:
{
lean_object* v___x_423_; 
v___x_423_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__5));
v___y_412_ = v___x_422_;
v___y_413_ = v___x_421_;
v___y_414_ = v___x_423_;
goto v___jp_411_;
}
case 1:
{
lean_object* v___x_424_; 
v___x_424_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__6));
v___y_412_ = v___x_422_;
v___y_413_ = v___x_421_;
v___y_414_ = v___x_424_;
goto v___jp_411_;
}
case 2:
{
lean_object* v___x_425_; 
v___x_425_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__7));
v___y_412_ = v___x_422_;
v___y_413_ = v___x_421_;
v___y_414_ = v___x_425_;
goto v___jp_411_;
}
case 3:
{
lean_object* v___x_426_; 
v___x_426_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__8));
v___y_412_ = v___x_422_;
v___y_413_ = v___x_421_;
v___y_414_ = v___x_426_;
goto v___jp_411_;
}
case 4:
{
lean_object* v___x_427_; 
v___x_427_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__9));
v___y_412_ = v___x_422_;
v___y_413_ = v___x_421_;
v___y_414_ = v___x_427_;
goto v___jp_411_;
}
case 5:
{
lean_object* v___x_428_; 
v___x_428_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__10));
v___y_412_ = v___x_422_;
v___y_413_ = v___x_421_;
v___y_414_ = v___x_428_;
goto v___jp_411_;
}
case 6:
{
lean_object* v___x_429_; 
v___x_429_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__11));
v___y_412_ = v___x_422_;
v___y_413_ = v___x_421_;
v___y_414_ = v___x_429_;
goto v___jp_411_;
}
default: 
{
lean_object* v___x_430_; 
v___x_430_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__12));
v___y_412_ = v___x_422_;
v___y_413_ = v___x_421_;
v___y_414_ = v___x_430_;
goto v___jp_411_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleStats_instToString___lam__0(lean_object* v_rp_443_){
_start:
{
lean_object* v___y_445_; lean_object* v___y_446_; lean_object* v___y_447_; lean_object* v___y_455_; lean_object* v_name_456_; lean_object* v___y_457_; lean_object* v___y_458_; lean_object* v___y_459_; lean_object* v___y_460_; lean_object* v___y_467_; lean_object* v___y_468_; lean_object* v_name_469_; uint8_t v_scope_470_; lean_object* v___y_471_; lean_object* v___y_472_; lean_object* v___y_473_; lean_object* v___y_479_; lean_object* v_name_480_; uint8_t v_builder_481_; uint8_t v_scope_482_; lean_object* v___y_483_; lean_object* v___y_484_; lean_object* v_rule_495_; lean_object* v_elapsed_496_; uint8_t v_successful_497_; lean_object* v___y_499_; 
v_rule_495_ = lean_ctor_get(v_rp_443_, 0);
lean_inc(v_rule_495_);
v_elapsed_496_ = lean_ctor_get(v_rp_443_, 1);
lean_inc(v_elapsed_496_);
v_successful_497_ = lean_ctor_get_uint8(v_rp_443_, sizeof(void*)*2);
lean_dec_ref(v_rp_443_);
if (v_successful_497_ == 0)
{
lean_object* v___x_521_; 
v___x_521_ = ((lean_object*)(lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__6));
v___y_499_ = v___x_521_;
goto v___jp_498_;
}
else
{
lean_object* v___x_522_; 
v___x_522_ = ((lean_object*)(lp_aesop_Aesop_instToJsonRuleStats_toJson___closed__2));
v___y_499_ = v___x_522_;
goto v___jp_498_;
}
v___jp_444_:
{
lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; 
v___x_448_ = lean_string_append(v___y_446_, v___y_447_);
lean_dec_ref(v___y_447_);
v___x_449_ = ((lean_object*)(lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__0));
v___x_450_ = lean_string_append(v___x_448_, v___x_449_);
v___x_451_ = lean_string_append(v___x_450_, v___y_445_);
v___x_452_ = ((lean_object*)(lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__1));
v___x_453_ = lean_string_append(v___x_451_, v___x_452_);
return v___x_453_;
}
v___jp_454_:
{
lean_object* v___x_461_; lean_object* v___x_462_; uint8_t v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; 
v___x_461_ = lean_string_append(v___y_458_, v___y_460_);
v___x_462_ = lean_string_append(v___x_461_, v___y_457_);
v___x_463_ = 1;
v___x_464_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_456_, v___x_463_);
v___x_465_ = lean_string_append(v___x_462_, v___x_464_);
lean_dec_ref(v___x_464_);
v___y_445_ = v___y_455_;
v___y_446_ = v___y_459_;
v___y_447_ = v___x_465_;
goto v___jp_444_;
}
v___jp_466_:
{
lean_object* v___x_474_; lean_object* v___x_475_; 
v___x_474_ = lean_string_append(v___y_468_, v___y_473_);
v___x_475_ = lean_string_append(v___x_474_, v___y_471_);
if (v_scope_470_ == 0)
{
lean_object* v___x_476_; 
v___x_476_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__2));
v___y_455_ = v___y_467_;
v_name_456_ = v_name_469_;
v___y_457_ = v___y_471_;
v___y_458_ = v___x_475_;
v___y_459_ = v___y_472_;
v___y_460_ = v___x_476_;
goto v___jp_454_;
}
else
{
lean_object* v___x_477_; 
v___x_477_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__3));
v___y_455_ = v___y_467_;
v_name_456_ = v_name_469_;
v___y_457_ = v___y_471_;
v___y_458_ = v___x_475_;
v___y_459_ = v___y_472_;
v___y_460_ = v___x_477_;
goto v___jp_454_;
}
}
v___jp_478_:
{
lean_object* v___x_485_; lean_object* v___x_486_; 
v___x_485_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__4));
lean_inc_ref(v___y_484_);
v___x_486_ = lean_string_append(v___y_484_, v___x_485_);
switch(v_builder_481_)
{
case 0:
{
lean_object* v___x_487_; 
v___x_487_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__5));
v___y_467_ = v___y_479_;
v___y_468_ = v___x_486_;
v_name_469_ = v_name_480_;
v_scope_470_ = v_scope_482_;
v___y_471_ = v___x_485_;
v___y_472_ = v___y_483_;
v___y_473_ = v___x_487_;
goto v___jp_466_;
}
case 1:
{
lean_object* v___x_488_; 
v___x_488_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__6));
v___y_467_ = v___y_479_;
v___y_468_ = v___x_486_;
v_name_469_ = v_name_480_;
v_scope_470_ = v_scope_482_;
v___y_471_ = v___x_485_;
v___y_472_ = v___y_483_;
v___y_473_ = v___x_488_;
goto v___jp_466_;
}
case 2:
{
lean_object* v___x_489_; 
v___x_489_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__7));
v___y_467_ = v___y_479_;
v___y_468_ = v___x_486_;
v_name_469_ = v_name_480_;
v_scope_470_ = v_scope_482_;
v___y_471_ = v___x_485_;
v___y_472_ = v___y_483_;
v___y_473_ = v___x_489_;
goto v___jp_466_;
}
case 3:
{
lean_object* v___x_490_; 
v___x_490_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__8));
v___y_467_ = v___y_479_;
v___y_468_ = v___x_486_;
v_name_469_ = v_name_480_;
v_scope_470_ = v_scope_482_;
v___y_471_ = v___x_485_;
v___y_472_ = v___y_483_;
v___y_473_ = v___x_490_;
goto v___jp_466_;
}
case 4:
{
lean_object* v___x_491_; 
v___x_491_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__9));
v___y_467_ = v___y_479_;
v___y_468_ = v___x_486_;
v_name_469_ = v_name_480_;
v_scope_470_ = v_scope_482_;
v___y_471_ = v___x_485_;
v___y_472_ = v___y_483_;
v___y_473_ = v___x_491_;
goto v___jp_466_;
}
case 5:
{
lean_object* v___x_492_; 
v___x_492_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__10));
v___y_467_ = v___y_479_;
v___y_468_ = v___x_486_;
v_name_469_ = v_name_480_;
v_scope_470_ = v_scope_482_;
v___y_471_ = v___x_485_;
v___y_472_ = v___y_483_;
v___y_473_ = v___x_492_;
goto v___jp_466_;
}
case 6:
{
lean_object* v___x_493_; 
v___x_493_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__11));
v___y_467_ = v___y_479_;
v___y_468_ = v___x_486_;
v_name_469_ = v_name_480_;
v_scope_470_ = v_scope_482_;
v___y_471_ = v___x_485_;
v___y_472_ = v___y_483_;
v___y_473_ = v___x_493_;
goto v___jp_466_;
}
default: 
{
lean_object* v___x_494_; 
v___x_494_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__12));
v___y_467_ = v___y_479_;
v___y_468_ = v___x_486_;
v_name_469_ = v_name_480_;
v_scope_470_ = v_scope_482_;
v___y_471_ = v___x_485_;
v___y_472_ = v___y_483_;
v___y_473_ = v___x_494_;
goto v___jp_466_;
}
}
}
v___jp_498_:
{
lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___x_504_; 
v___x_500_ = ((lean_object*)(lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__2));
v___x_501_ = lp_aesop_Aesop_Nanos_printAsMillis(v_elapsed_496_);
v___x_502_ = lean_string_append(v___x_500_, v___x_501_);
lean_dec_ref(v___x_501_);
v___x_503_ = ((lean_object*)(lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__3));
v___x_504_ = lean_string_append(v___x_502_, v___x_503_);
switch(lean_obj_tag(v_rule_495_))
{
case 0:
{
lean_object* v_n_505_; uint8_t v_phase_506_; 
v_n_505_ = lean_ctor_get(v_rule_495_, 0);
lean_inc_ref(v_n_505_);
lean_dec_ref_known(v_rule_495_, 1);
v_phase_506_ = lean_ctor_get_uint8(v_n_505_, sizeof(void*)*1 + 9);
switch(v_phase_506_)
{
case 0:
{
lean_object* v_name_507_; uint8_t v_builder_508_; uint8_t v_scope_509_; lean_object* v___x_510_; 
v_name_507_ = lean_ctor_get(v_n_505_, 0);
lean_inc(v_name_507_);
v_builder_508_ = lean_ctor_get_uint8(v_n_505_, sizeof(void*)*1 + 8);
v_scope_509_ = lean_ctor_get_uint8(v_n_505_, sizeof(void*)*1 + 10);
lean_dec_ref(v_n_505_);
v___x_510_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__13));
v___y_479_ = v___y_499_;
v_name_480_ = v_name_507_;
v_builder_481_ = v_builder_508_;
v_scope_482_ = v_scope_509_;
v___y_483_ = v___x_504_;
v___y_484_ = v___x_510_;
goto v___jp_478_;
}
case 1:
{
lean_object* v_name_511_; uint8_t v_builder_512_; uint8_t v_scope_513_; lean_object* v___x_514_; 
v_name_511_ = lean_ctor_get(v_n_505_, 0);
lean_inc(v_name_511_);
v_builder_512_ = lean_ctor_get_uint8(v_n_505_, sizeof(void*)*1 + 8);
v_scope_513_ = lean_ctor_get_uint8(v_n_505_, sizeof(void*)*1 + 10);
lean_dec_ref(v_n_505_);
v___x_514_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__14));
v___y_479_ = v___y_499_;
v_name_480_ = v_name_511_;
v_builder_481_ = v_builder_512_;
v_scope_482_ = v_scope_513_;
v___y_483_ = v___x_504_;
v___y_484_ = v___x_514_;
goto v___jp_478_;
}
default: 
{
lean_object* v_name_515_; uint8_t v_builder_516_; uint8_t v_scope_517_; lean_object* v___x_518_; 
v_name_515_ = lean_ctor_get(v_n_505_, 0);
lean_inc(v_name_515_);
v_builder_516_ = lean_ctor_get_uint8(v_n_505_, sizeof(void*)*1 + 8);
v_scope_517_ = lean_ctor_get_uint8(v_n_505_, sizeof(void*)*1 + 10);
lean_dec_ref(v_n_505_);
v___x_518_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__15));
v___y_479_ = v___y_499_;
v_name_480_ = v_name_515_;
v_builder_481_ = v_builder_516_;
v_scope_482_ = v_scope_517_;
v___y_483_ = v___x_504_;
v___y_484_ = v___x_518_;
goto v___jp_478_;
}
}
}
case 1:
{
lean_object* v___x_519_; 
v___x_519_ = ((lean_object*)(lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__4));
v___y_445_ = v___y_499_;
v___y_446_ = v___x_504_;
v___y_447_ = v___x_519_;
goto v___jp_444_;
}
default: 
{
lean_object* v___x_520_; 
v___x_520_ = ((lean_object*)(lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__5));
v___y_445_ = v___y_499_;
v___y_446_ = v___x_504_;
v___y_447_ = v___x_520_;
goto v___jp_444_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_ctorIdx(uint8_t v_x_525_){
_start:
{
if (v_x_525_ == 0)
{
lean_object* v___x_526_; 
v___x_526_ = lean_unsigned_to_nat(0u);
return v___x_526_;
}
else
{
lean_object* v___x_527_; 
v___x_527_ = lean_unsigned_to_nat(1u);
return v___x_527_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_ctorIdx___boxed(lean_object* v_x_528_){
_start:
{
uint8_t v_x_boxed_529_; lean_object* v_res_530_; 
v_x_boxed_529_ = lean_unbox(v_x_528_);
v_res_530_ = lp_aesop_Aesop_ScriptGenerated_Method_ctorIdx(v_x_boxed_529_);
return v_res_530_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_ctorElim___redArg(lean_object* v_k_531_){
_start:
{
lean_inc(v_k_531_);
return v_k_531_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_ctorElim___redArg___boxed(lean_object* v_k_532_){
_start:
{
lean_object* v_res_533_; 
v_res_533_ = lp_aesop_Aesop_ScriptGenerated_Method_ctorElim___redArg(v_k_532_);
lean_dec(v_k_532_);
return v_res_533_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_ctorElim(lean_object* v_motive_534_, lean_object* v_ctorIdx_535_, uint8_t v_t_536_, lean_object* v_h_537_, lean_object* v_k_538_){
_start:
{
lean_inc(v_k_538_);
return v_k_538_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_ctorElim___boxed(lean_object* v_motive_539_, lean_object* v_ctorIdx_540_, lean_object* v_t_541_, lean_object* v_h_542_, lean_object* v_k_543_){
_start:
{
uint8_t v_t_boxed_544_; lean_object* v_res_545_; 
v_t_boxed_544_ = lean_unbox(v_t_541_);
v_res_545_ = lp_aesop_Aesop_ScriptGenerated_Method_ctorElim(v_motive_539_, v_ctorIdx_540_, v_t_boxed_544_, v_h_542_, v_k_543_);
lean_dec(v_k_543_);
lean_dec(v_ctorIdx_540_);
return v_res_545_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_static_elim___redArg(lean_object* v_static_546_){
_start:
{
lean_inc(v_static_546_);
return v_static_546_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_static_elim___redArg___boxed(lean_object* v_static_547_){
_start:
{
lean_object* v_res_548_; 
v_res_548_ = lp_aesop_Aesop_ScriptGenerated_Method_static_elim___redArg(v_static_547_);
lean_dec(v_static_547_);
return v_res_548_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_static_elim(lean_object* v_motive_549_, uint8_t v_t_550_, lean_object* v_h_551_, lean_object* v_static_552_){
_start:
{
lean_inc(v_static_552_);
return v_static_552_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_static_elim___boxed(lean_object* v_motive_553_, lean_object* v_t_554_, lean_object* v_h_555_, lean_object* v_static_556_){
_start:
{
uint8_t v_t_boxed_557_; lean_object* v_res_558_; 
v_t_boxed_557_ = lean_unbox(v_t_554_);
v_res_558_ = lp_aesop_Aesop_ScriptGenerated_Method_static_elim(v_motive_553_, v_t_boxed_557_, v_h_555_, v_static_556_);
lean_dec(v_static_556_);
return v_res_558_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_dynamic_elim___redArg(lean_object* v_dynamic_559_){
_start:
{
lean_inc(v_dynamic_559_);
return v_dynamic_559_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_dynamic_elim___redArg___boxed(lean_object* v_dynamic_560_){
_start:
{
lean_object* v_res_561_; 
v_res_561_ = lp_aesop_Aesop_ScriptGenerated_Method_dynamic_elim___redArg(v_dynamic_560_);
lean_dec(v_dynamic_560_);
return v_res_561_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_dynamic_elim(lean_object* v_motive_562_, uint8_t v_t_563_, lean_object* v_h_564_, lean_object* v_dynamic_565_){
_start:
{
lean_inc(v_dynamic_565_);
return v_dynamic_565_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_Method_dynamic_elim___boxed(lean_object* v_motive_566_, lean_object* v_t_567_, lean_object* v_h_568_, lean_object* v_dynamic_569_){
_start:
{
uint8_t v_t_boxed_570_; lean_object* v_res_571_; 
v_t_boxed_570_ = lean_unbox(v_t_567_);
v_res_571_ = lp_aesop_Aesop_ScriptGenerated_Method_dynamic_elim(v_motive_566_, v_t_boxed_570_, v_h_568_, v_dynamic_569_);
lean_dec(v_dynamic_569_);
return v_res_571_;
}
}
static uint8_t _init_lp_aesop_Aesop_ScriptGenerated_instInhabitedMethod_default(void){
_start:
{
uint8_t v___x_572_; 
v___x_572_ = 0;
return v___x_572_;
}
}
static uint8_t _init_lp_aesop_Aesop_ScriptGenerated_instInhabitedMethod(void){
_start:
{
uint8_t v___x_573_; 
v___x_573_ = 0;
return v___x_573_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson(uint8_t v_x_580_){
_start:
{
if (v_x_580_ == 0)
{
lean_object* v___x_581_; 
v___x_581_ = ((lean_object*)(lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__1));
return v___x_581_;
}
else
{
lean_object* v___x_582_; 
v___x_582_ = ((lean_object*)(lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__3));
return v___x_582_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___boxed(lean_object* v_x_583_){
_start:
{
uint8_t v_x_46__boxed_584_; lean_object* v_res_585_; 
v_x_46__boxed_584_ = lean_unbox(v_x_583_);
v_res_585_ = lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson(v_x_46__boxed_584_);
return v_res_585_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToStringMethod___lam__0(uint8_t v_x_588_){
_start:
{
if (v_x_588_ == 0)
{
lean_object* v___x_589_; 
v___x_589_ = ((lean_object*)(lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__0));
return v___x_589_;
}
else
{
lean_object* v___x_590_; 
v___x_590_ = ((lean_object*)(lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__2));
return v___x_590_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToStringMethod___lam__0___boxed(lean_object* v_x_591_){
_start:
{
uint8_t v_x_24__boxed_592_; lean_object* v_res_593_; 
v_x_24__boxed_592_ = lean_unbox(v_x_591_);
v_res_593_ = lp_aesop_Aesop_instToStringMethod___lam__0(v_x_24__boxed_592_);
return v_res_593_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonScriptGenerated_toJson(lean_object* v_x_604_){
_start:
{
uint8_t v_method_605_; uint8_t v_perfect_606_; uint8_t v_hasMVar_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; 
v_method_605_ = lean_ctor_get_uint8(v_x_604_, 0);
v_perfect_606_ = lean_ctor_get_uint8(v_x_604_, 1);
v_hasMVar_607_ = lean_ctor_get_uint8(v_x_604_, 2);
v___x_608_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScriptGenerated_toJson___closed__0));
v___x_609_ = lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson(v_method_605_);
v___x_610_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_610_, 0, v___x_608_);
lean_ctor_set(v___x_610_, 1, v___x_609_);
v___x_611_ = lean_box(0);
v___x_612_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_612_, 0, v___x_610_);
lean_ctor_set(v___x_612_, 1, v___x_611_);
v___x_613_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScriptGenerated_toJson___closed__1));
v___x_614_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_614_, 0, v_perfect_606_);
v___x_615_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_615_, 0, v___x_613_);
lean_ctor_set(v___x_615_, 1, v___x_614_);
v___x_616_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_616_, 0, v___x_615_);
lean_ctor_set(v___x_616_, 1, v___x_611_);
v___x_617_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScriptGenerated_toJson___closed__2));
v___x_618_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_618_, 0, v_hasMVar_607_);
v___x_619_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_619_, 0, v___x_617_);
lean_ctor_set(v___x_619_, 1, v___x_618_);
v___x_620_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_620_, 0, v___x_619_);
lean_ctor_set(v___x_620_, 1, v___x_611_);
v___x_621_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_621_, 0, v___x_620_);
lean_ctor_set(v___x_621_, 1, v___x_611_);
v___x_622_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_622_, 0, v___x_616_);
lean_ctor_set(v___x_622_, 1, v___x_621_);
v___x_623_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_623_, 0, v___x_612_);
lean_ctor_set(v___x_623_, 1, v___x_622_);
v___x_624_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardInstantiationStats_toJson___closed__2));
v___x_625_ = lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Aesop_instToJsonForwardInstantiationStats_toJson_spec__0(v___x_623_, v___x_624_);
v___x_626_ = l_Lean_Json_mkObj(v___x_625_);
lean_dec(v___x_625_);
return v___x_626_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonScriptGenerated_toJson___boxed(lean_object* v_x_627_){
_start:
{
lean_object* v_res_628_; 
v_res_628_ = lp_aesop_Aesop_instToJsonScriptGenerated_toJson(v_x_627_);
lean_dec_ref(v_x_627_);
return v_res_628_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_toString(lean_object* v_g_635_){
_start:
{
lean_object* v___y_637_; lean_object* v___y_638_; uint8_t v_method_642_; uint8_t v_perfect_643_; lean_object* v___x_644_; lean_object* v___y_646_; 
v_method_642_ = lean_ctor_get_uint8(v_g_635_, 0);
v_perfect_643_ = lean_ctor_get_uint8(v_g_635_, 1);
v___x_644_ = ((lean_object*)(lp_aesop_Aesop_ScriptGenerated_toString___closed__1));
if (v_perfect_643_ == 0)
{
lean_object* v___x_652_; 
v___x_652_ = ((lean_object*)(lp_aesop_Aesop_ScriptGenerated_toString___closed__3));
v___y_646_ = v___x_652_;
goto v___jp_645_;
}
else
{
lean_object* v___x_653_; 
v___x_653_ = ((lean_object*)(lp_aesop_Aesop_instToJsonScriptGenerated_toJson___closed__1));
v___y_646_ = v___x_653_;
goto v___jp_645_;
}
v___jp_636_:
{
lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; 
v___x_639_ = lean_string_append(v___y_637_, v___y_638_);
v___x_640_ = ((lean_object*)(lp_aesop_Aesop_ScriptGenerated_toString___closed__0));
v___x_641_ = lean_string_append(v___x_639_, v___x_640_);
return v___x_641_;
}
v___jp_645_:
{
lean_object* v___x_647_; lean_object* v___x_648_; lean_object* v___x_649_; 
v___x_647_ = lean_string_append(v___x_644_, v___y_646_);
v___x_648_ = ((lean_object*)(lp_aesop_Aesop_ScriptGenerated_toString___closed__2));
v___x_649_ = lean_string_append(v___x_647_, v___x_648_);
if (v_method_642_ == 0)
{
lean_object* v___x_650_; 
v___x_650_ = ((lean_object*)(lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__0));
v___y_637_ = v___x_649_;
v___y_638_ = v___x_650_;
goto v___jp_636_;
}
else
{
lean_object* v___x_651_; 
v___x_651_ = ((lean_object*)(lp_aesop_Aesop_ScriptGenerated_instToJsonMethod_toJson___closed__2));
v___y_637_ = v___x_649_;
v___y_638_ = v___x_651_;
goto v___jp_636_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_ScriptGenerated_toString___boxed(lean_object* v_g_654_){
_start:
{
lean_object* v_res_655_; 
v_res_655_ = lp_aesop_Aesop_ScriptGenerated_toString(v_g_654_);
lean_dec_ref(v_g_654_);
return v_res_655_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleStatsTotals_compareByTotalElapsed___lam__0(lean_object* v_totals_676_){
_start:
{
lean_object* v_elapsedSuccessful_677_; lean_object* v_elapsedFailed_678_; lean_object* v___x_679_; 
v_elapsedSuccessful_677_ = lean_ctor_get(v_totals_676_, 2);
v_elapsedFailed_678_ = lean_ctor_get(v_totals_676_, 3);
v___x_679_ = lean_nat_add(v_elapsedSuccessful_677_, v_elapsedFailed_678_);
return v___x_679_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleStatsTotals_compareByTotalElapsed___lam__0___boxed(lean_object* v_totals_680_){
_start:
{
lean_object* v_res_681_; 
v_res_681_ = lp_aesop_Aesop_RuleStatsTotals_compareByTotalElapsed___lam__0(v_totals_680_);
lean_dec_ref(v_totals_680_);
return v_res_681_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_RuleStatsTotals_compareByTotalElapsed(lean_object* v_x_682_, lean_object* v_y_683_){
_start:
{
lean_object* v___x_684_; lean_object* v___x_685_; uint8_t v___x_686_; 
v___x_684_ = lp_aesop_Aesop_RuleStatsTotals_compareByTotalElapsed___lam__0(v_x_682_);
v___x_685_ = lp_aesop_Aesop_RuleStatsTotals_compareByTotalElapsed___lam__0(v_y_683_);
v___x_686_ = lp_aesop_Aesop_instOrdNanos_ord(v___x_684_, v___x_685_);
lean_dec(v___x_685_);
lean_dec(v___x_684_);
return v___x_686_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_RuleStatsTotals_compareByTotalElapsed___boxed(lean_object* v_x_687_, lean_object* v_y_688_){
_start:
{
uint8_t v_res_689_; lean_object* v_r_690_; 
v_res_689_ = lp_aesop_Aesop_RuleStatsTotals_compareByTotalElapsed(v_x_687_, v_y_688_);
lean_dec_ref(v_y_688_);
lean_dec_ref(v_x_687_);
v_r_690_ = lean_box(v_res_689_);
return v_r_690_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1_spec__4___redArg(lean_object* v_a_691_, lean_object* v_fallback_692_, lean_object* v_x_693_){
_start:
{
if (lean_obj_tag(v_x_693_) == 0)
{
lean_inc(v_fallback_692_);
return v_fallback_692_;
}
else
{
lean_object* v_key_694_; lean_object* v_value_695_; lean_object* v_tail_696_; uint8_t v___x_697_; 
v_key_694_ = lean_ctor_get(v_x_693_, 0);
v_value_695_ = lean_ctor_get(v_x_693_, 1);
v_tail_696_ = lean_ctor_get(v_x_693_, 2);
v___x_697_ = lp_aesop_Aesop_instBEqDisplayRuleName_beq(v_key_694_, v_a_691_);
if (v___x_697_ == 0)
{
v_x_693_ = v_tail_696_;
goto _start;
}
else
{
lean_inc(v_value_695_);
return v_value_695_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1_spec__4___redArg___boxed(lean_object* v_a_699_, lean_object* v_fallback_700_, lean_object* v_x_701_){
_start:
{
lean_object* v_res_702_; 
v_res_702_ = lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1_spec__4___redArg(v_a_699_, v_fallback_700_, v_x_701_);
lean_dec(v_x_701_);
lean_dec(v_fallback_700_);
lean_dec(v_a_699_);
return v_res_702_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1___redArg(lean_object* v_m_703_, lean_object* v_a_704_, lean_object* v_fallback_705_){
_start:
{
lean_object* v_buckets_706_; lean_object* v___x_707_; uint64_t v___x_708_; uint64_t v___x_709_; uint64_t v___x_710_; uint64_t v_fold_711_; uint64_t v___x_712_; uint64_t v___x_713_; uint64_t v___x_714_; size_t v___x_715_; size_t v___x_716_; size_t v___x_717_; size_t v___x_718_; size_t v___x_719_; lean_object* v___x_720_; lean_object* v___x_721_; 
v_buckets_706_ = lean_ctor_get(v_m_703_, 1);
v___x_707_ = lean_array_get_size(v_buckets_706_);
v___x_708_ = lp_aesop_Aesop_instHashableDisplayRuleName_hash(v_a_704_);
v___x_709_ = 32ULL;
v___x_710_ = lean_uint64_shift_right(v___x_708_, v___x_709_);
v_fold_711_ = lean_uint64_xor(v___x_708_, v___x_710_);
v___x_712_ = 16ULL;
v___x_713_ = lean_uint64_shift_right(v_fold_711_, v___x_712_);
v___x_714_ = lean_uint64_xor(v_fold_711_, v___x_713_);
v___x_715_ = lean_uint64_to_usize(v___x_714_);
v___x_716_ = lean_usize_of_nat(v___x_707_);
v___x_717_ = ((size_t)1ULL);
v___x_718_ = lean_usize_sub(v___x_716_, v___x_717_);
v___x_719_ = lean_usize_land(v___x_715_, v___x_718_);
v___x_720_ = lean_array_uget_borrowed(v_buckets_706_, v___x_719_);
v___x_721_ = lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1_spec__4___redArg(v_a_704_, v_fallback_705_, v___x_720_);
return v___x_721_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1___redArg___boxed(lean_object* v_m_722_, lean_object* v_a_723_, lean_object* v_fallback_724_){
_start:
{
lean_object* v_res_725_; 
v_res_725_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1___redArg(v_m_722_, v_a_723_, v_fallback_724_);
lean_dec(v_fallback_724_);
lean_dec(v_a_723_);
lean_dec_ref(v_m_722_);
return v_res_725_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__0___redArg(lean_object* v_a_726_, lean_object* v_x_727_){
_start:
{
if (lean_obj_tag(v_x_727_) == 0)
{
uint8_t v___x_728_; 
v___x_728_ = 0;
return v___x_728_;
}
else
{
lean_object* v_key_729_; lean_object* v_tail_730_; uint8_t v___x_731_; 
v_key_729_ = lean_ctor_get(v_x_727_, 0);
v_tail_730_ = lean_ctor_get(v_x_727_, 2);
v___x_731_ = lp_aesop_Aesop_instBEqDisplayRuleName_beq(v_key_729_, v_a_726_);
if (v___x_731_ == 0)
{
v_x_727_ = v_tail_730_;
goto _start;
}
else
{
return v___x_731_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__0___redArg___boxed(lean_object* v_a_733_, lean_object* v_x_734_){
_start:
{
uint8_t v_res_735_; lean_object* v_r_736_; 
v_res_735_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__0___redArg(v_a_733_, v_x_734_);
lean_dec(v_x_734_);
lean_dec(v_a_733_);
v_r_736_ = lean_box(v_res_735_);
return v_r_736_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1_spec__2_spec__5___redArg(lean_object* v_x_737_, lean_object* v_x_738_){
_start:
{
if (lean_obj_tag(v_x_738_) == 0)
{
return v_x_737_;
}
else
{
lean_object* v_key_739_; lean_object* v_value_740_; lean_object* v_tail_741_; lean_object* v___x_743_; uint8_t v_isShared_744_; uint8_t v_isSharedCheck_764_; 
v_key_739_ = lean_ctor_get(v_x_738_, 0);
v_value_740_ = lean_ctor_get(v_x_738_, 1);
v_tail_741_ = lean_ctor_get(v_x_738_, 2);
v_isSharedCheck_764_ = !lean_is_exclusive(v_x_738_);
if (v_isSharedCheck_764_ == 0)
{
v___x_743_ = v_x_738_;
v_isShared_744_ = v_isSharedCheck_764_;
goto v_resetjp_742_;
}
else
{
lean_inc(v_tail_741_);
lean_inc(v_value_740_);
lean_inc(v_key_739_);
lean_dec(v_x_738_);
v___x_743_ = lean_box(0);
v_isShared_744_ = v_isSharedCheck_764_;
goto v_resetjp_742_;
}
v_resetjp_742_:
{
lean_object* v___x_745_; uint64_t v___x_746_; uint64_t v___x_747_; uint64_t v___x_748_; uint64_t v_fold_749_; uint64_t v___x_750_; uint64_t v___x_751_; uint64_t v___x_752_; size_t v___x_753_; size_t v___x_754_; size_t v___x_755_; size_t v___x_756_; size_t v___x_757_; lean_object* v___x_758_; lean_object* v___x_760_; 
v___x_745_ = lean_array_get_size(v_x_737_);
v___x_746_ = lp_aesop_Aesop_instHashableDisplayRuleName_hash(v_key_739_);
v___x_747_ = 32ULL;
v___x_748_ = lean_uint64_shift_right(v___x_746_, v___x_747_);
v_fold_749_ = lean_uint64_xor(v___x_746_, v___x_748_);
v___x_750_ = 16ULL;
v___x_751_ = lean_uint64_shift_right(v_fold_749_, v___x_750_);
v___x_752_ = lean_uint64_xor(v_fold_749_, v___x_751_);
v___x_753_ = lean_uint64_to_usize(v___x_752_);
v___x_754_ = lean_usize_of_nat(v___x_745_);
v___x_755_ = ((size_t)1ULL);
v___x_756_ = lean_usize_sub(v___x_754_, v___x_755_);
v___x_757_ = lean_usize_land(v___x_753_, v___x_756_);
v___x_758_ = lean_array_uget_borrowed(v_x_737_, v___x_757_);
lean_inc(v___x_758_);
if (v_isShared_744_ == 0)
{
lean_ctor_set(v___x_743_, 2, v___x_758_);
v___x_760_ = v___x_743_;
goto v_reusejp_759_;
}
else
{
lean_object* v_reuseFailAlloc_763_; 
v_reuseFailAlloc_763_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_763_, 0, v_key_739_);
lean_ctor_set(v_reuseFailAlloc_763_, 1, v_value_740_);
lean_ctor_set(v_reuseFailAlloc_763_, 2, v___x_758_);
v___x_760_ = v_reuseFailAlloc_763_;
goto v_reusejp_759_;
}
v_reusejp_759_:
{
lean_object* v___x_761_; 
v___x_761_ = lean_array_uset(v_x_737_, v___x_757_, v___x_760_);
v_x_737_ = v___x_761_;
v_x_738_ = v_tail_741_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1_spec__2___redArg(lean_object* v_i_765_, lean_object* v_source_766_, lean_object* v_target_767_){
_start:
{
lean_object* v___x_768_; uint8_t v___x_769_; 
v___x_768_ = lean_array_get_size(v_source_766_);
v___x_769_ = lean_nat_dec_lt(v_i_765_, v___x_768_);
if (v___x_769_ == 0)
{
lean_dec_ref(v_source_766_);
lean_dec(v_i_765_);
return v_target_767_;
}
else
{
lean_object* v_es_770_; lean_object* v___x_771_; lean_object* v_source_772_; lean_object* v_target_773_; lean_object* v___x_774_; lean_object* v___x_775_; 
v_es_770_ = lean_array_fget(v_source_766_, v_i_765_);
v___x_771_ = lean_box(0);
v_source_772_ = lean_array_fset(v_source_766_, v_i_765_, v___x_771_);
v_target_773_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1_spec__2_spec__5___redArg(v_target_767_, v_es_770_);
v___x_774_ = lean_unsigned_to_nat(1u);
v___x_775_ = lean_nat_add(v_i_765_, v___x_774_);
lean_dec(v_i_765_);
v_i_765_ = v___x_775_;
v_source_766_ = v_source_772_;
v_target_767_ = v_target_773_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1___redArg(lean_object* v_data_777_){
_start:
{
lean_object* v___x_778_; lean_object* v___x_779_; lean_object* v_nbuckets_780_; lean_object* v___x_781_; lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; 
v___x_778_ = lean_array_get_size(v_data_777_);
v___x_779_ = lean_unsigned_to_nat(2u);
v_nbuckets_780_ = lean_nat_mul(v___x_778_, v___x_779_);
v___x_781_ = lean_unsigned_to_nat(0u);
v___x_782_ = lean_box(0);
v___x_783_ = lean_mk_array(v_nbuckets_780_, v___x_782_);
v___x_784_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1_spec__2___redArg(v___x_781_, v_data_777_, v___x_783_);
return v___x_784_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__2___redArg(lean_object* v_a_785_, lean_object* v_b_786_, lean_object* v_x_787_){
_start:
{
if (lean_obj_tag(v_x_787_) == 0)
{
lean_dec(v_b_786_);
lean_dec(v_a_785_);
return v_x_787_;
}
else
{
lean_object* v_key_788_; lean_object* v_value_789_; lean_object* v_tail_790_; lean_object* v___x_792_; uint8_t v_isShared_793_; uint8_t v_isSharedCheck_802_; 
v_key_788_ = lean_ctor_get(v_x_787_, 0);
v_value_789_ = lean_ctor_get(v_x_787_, 1);
v_tail_790_ = lean_ctor_get(v_x_787_, 2);
v_isSharedCheck_802_ = !lean_is_exclusive(v_x_787_);
if (v_isSharedCheck_802_ == 0)
{
v___x_792_ = v_x_787_;
v_isShared_793_ = v_isSharedCheck_802_;
goto v_resetjp_791_;
}
else
{
lean_inc(v_tail_790_);
lean_inc(v_value_789_);
lean_inc(v_key_788_);
lean_dec(v_x_787_);
v___x_792_ = lean_box(0);
v_isShared_793_ = v_isSharedCheck_802_;
goto v_resetjp_791_;
}
v_resetjp_791_:
{
uint8_t v___x_794_; 
v___x_794_ = lp_aesop_Aesop_instBEqDisplayRuleName_beq(v_key_788_, v_a_785_);
if (v___x_794_ == 0)
{
lean_object* v___x_795_; lean_object* v___x_797_; 
v___x_795_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__2___redArg(v_a_785_, v_b_786_, v_tail_790_);
if (v_isShared_793_ == 0)
{
lean_ctor_set(v___x_792_, 2, v___x_795_);
v___x_797_ = v___x_792_;
goto v_reusejp_796_;
}
else
{
lean_object* v_reuseFailAlloc_798_; 
v_reuseFailAlloc_798_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_798_, 0, v_key_788_);
lean_ctor_set(v_reuseFailAlloc_798_, 1, v_value_789_);
lean_ctor_set(v_reuseFailAlloc_798_, 2, v___x_795_);
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
lean_object* v___x_800_; 
lean_dec(v_value_789_);
lean_dec(v_key_788_);
if (v_isShared_793_ == 0)
{
lean_ctor_set(v___x_792_, 1, v_b_786_);
lean_ctor_set(v___x_792_, 0, v_a_785_);
v___x_800_ = v___x_792_;
goto v_reusejp_799_;
}
else
{
lean_object* v_reuseFailAlloc_801_; 
v_reuseFailAlloc_801_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_801_, 0, v_a_785_);
lean_ctor_set(v_reuseFailAlloc_801_, 1, v_b_786_);
lean_ctor_set(v_reuseFailAlloc_801_, 2, v_tail_790_);
v___x_800_ = v_reuseFailAlloc_801_;
goto v_reusejp_799_;
}
v_reusejp_799_:
{
return v___x_800_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0___redArg(lean_object* v_m_803_, lean_object* v_a_804_, lean_object* v_b_805_){
_start:
{
lean_object* v_size_806_; lean_object* v_buckets_807_; lean_object* v___x_809_; uint8_t v_isShared_810_; uint8_t v_isSharedCheck_850_; 
v_size_806_ = lean_ctor_get(v_m_803_, 0);
v_buckets_807_ = lean_ctor_get(v_m_803_, 1);
v_isSharedCheck_850_ = !lean_is_exclusive(v_m_803_);
if (v_isSharedCheck_850_ == 0)
{
v___x_809_ = v_m_803_;
v_isShared_810_ = v_isSharedCheck_850_;
goto v_resetjp_808_;
}
else
{
lean_inc(v_buckets_807_);
lean_inc(v_size_806_);
lean_dec(v_m_803_);
v___x_809_ = lean_box(0);
v_isShared_810_ = v_isSharedCheck_850_;
goto v_resetjp_808_;
}
v_resetjp_808_:
{
lean_object* v___x_811_; uint64_t v___x_812_; uint64_t v___x_813_; uint64_t v___x_814_; uint64_t v_fold_815_; uint64_t v___x_816_; uint64_t v___x_817_; uint64_t v___x_818_; size_t v___x_819_; size_t v___x_820_; size_t v___x_821_; size_t v___x_822_; size_t v___x_823_; lean_object* v_bkt_824_; uint8_t v___x_825_; 
v___x_811_ = lean_array_get_size(v_buckets_807_);
v___x_812_ = lp_aesop_Aesop_instHashableDisplayRuleName_hash(v_a_804_);
v___x_813_ = 32ULL;
v___x_814_ = lean_uint64_shift_right(v___x_812_, v___x_813_);
v_fold_815_ = lean_uint64_xor(v___x_812_, v___x_814_);
v___x_816_ = 16ULL;
v___x_817_ = lean_uint64_shift_right(v_fold_815_, v___x_816_);
v___x_818_ = lean_uint64_xor(v_fold_815_, v___x_817_);
v___x_819_ = lean_uint64_to_usize(v___x_818_);
v___x_820_ = lean_usize_of_nat(v___x_811_);
v___x_821_ = ((size_t)1ULL);
v___x_822_ = lean_usize_sub(v___x_820_, v___x_821_);
v___x_823_ = lean_usize_land(v___x_819_, v___x_822_);
v_bkt_824_ = lean_array_uget_borrowed(v_buckets_807_, v___x_823_);
v___x_825_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__0___redArg(v_a_804_, v_bkt_824_);
if (v___x_825_ == 0)
{
lean_object* v___x_826_; lean_object* v_size_x27_827_; lean_object* v___x_828_; lean_object* v_buckets_x27_829_; lean_object* v___x_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; uint8_t v___x_835_; 
v___x_826_ = lean_unsigned_to_nat(1u);
v_size_x27_827_ = lean_nat_add(v_size_806_, v___x_826_);
lean_dec(v_size_806_);
lean_inc(v_bkt_824_);
v___x_828_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_828_, 0, v_a_804_);
lean_ctor_set(v___x_828_, 1, v_b_805_);
lean_ctor_set(v___x_828_, 2, v_bkt_824_);
v_buckets_x27_829_ = lean_array_uset(v_buckets_807_, v___x_823_, v___x_828_);
v___x_830_ = lean_unsigned_to_nat(4u);
v___x_831_ = lean_nat_mul(v_size_x27_827_, v___x_830_);
v___x_832_ = lean_unsigned_to_nat(3u);
v___x_833_ = lean_nat_div(v___x_831_, v___x_832_);
lean_dec(v___x_831_);
v___x_834_ = lean_array_get_size(v_buckets_x27_829_);
v___x_835_ = lean_nat_dec_le(v___x_833_, v___x_834_);
lean_dec(v___x_833_);
if (v___x_835_ == 0)
{
lean_object* v_val_836_; lean_object* v___x_838_; 
v_val_836_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1___redArg(v_buckets_x27_829_);
if (v_isShared_810_ == 0)
{
lean_ctor_set(v___x_809_, 1, v_val_836_);
lean_ctor_set(v___x_809_, 0, v_size_x27_827_);
v___x_838_ = v___x_809_;
goto v_reusejp_837_;
}
else
{
lean_object* v_reuseFailAlloc_839_; 
v_reuseFailAlloc_839_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_839_, 0, v_size_x27_827_);
lean_ctor_set(v_reuseFailAlloc_839_, 1, v_val_836_);
v___x_838_ = v_reuseFailAlloc_839_;
goto v_reusejp_837_;
}
v_reusejp_837_:
{
return v___x_838_;
}
}
else
{
lean_object* v___x_841_; 
if (v_isShared_810_ == 0)
{
lean_ctor_set(v___x_809_, 1, v_buckets_x27_829_);
lean_ctor_set(v___x_809_, 0, v_size_x27_827_);
v___x_841_ = v___x_809_;
goto v_reusejp_840_;
}
else
{
lean_object* v_reuseFailAlloc_842_; 
v_reuseFailAlloc_842_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_842_, 0, v_size_x27_827_);
lean_ctor_set(v_reuseFailAlloc_842_, 1, v_buckets_x27_829_);
v___x_841_ = v_reuseFailAlloc_842_;
goto v_reusejp_840_;
}
v_reusejp_840_:
{
return v___x_841_;
}
}
}
else
{
lean_object* v___x_843_; lean_object* v_buckets_x27_844_; lean_object* v___x_845_; lean_object* v___x_846_; lean_object* v___x_848_; 
lean_inc(v_bkt_824_);
v___x_843_ = lean_box(0);
v_buckets_x27_844_ = lean_array_uset(v_buckets_807_, v___x_823_, v___x_843_);
v___x_845_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__2___redArg(v_a_804_, v_b_805_, v_bkt_824_);
v___x_846_ = lean_array_uset(v_buckets_x27_844_, v___x_823_, v___x_845_);
if (v_isShared_810_ == 0)
{
lean_ctor_set(v___x_809_, 1, v___x_846_);
v___x_848_ = v___x_809_;
goto v_reusejp_847_;
}
else
{
lean_object* v_reuseFailAlloc_849_; 
v_reuseFailAlloc_849_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_849_, 0, v_size_806_);
lean_ctor_set(v_reuseFailAlloc_849_, 1, v___x_846_);
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
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_ruleStatsTotals_spec__2(lean_object* v_as_851_, size_t v_i_852_, size_t v_stop_853_, lean_object* v_b_854_){
_start:
{
uint8_t v___x_855_; 
v___x_855_ = lean_usize_dec_eq(v_i_852_, v_stop_853_);
if (v___x_855_ == 0)
{
lean_object* v___x_856_; lean_object* v_rule_857_; lean_object* v_elapsed_858_; uint8_t v_successful_859_; lean_object* v_stats_861_; lean_object* v___x_866_; lean_object* v_stats_867_; 
v___x_856_ = lean_array_uget_borrowed(v_as_851_, v_i_852_);
v_rule_857_ = lean_ctor_get(v___x_856_, 0);
v_elapsed_858_ = lean_ctor_get(v___x_856_, 1);
v_successful_859_ = lean_ctor_get_uint8(v___x_856_, sizeof(void*)*2);
v___x_866_ = ((lean_object*)(lp_aesop_Aesop_RuleStatsTotals_empty));
v_stats_867_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1___redArg(v_b_854_, v_rule_857_, v___x_866_);
if (v_successful_859_ == 0)
{
lean_object* v_numSuccessful_868_; lean_object* v_numFailed_869_; lean_object* v_elapsedSuccessful_870_; lean_object* v_elapsedFailed_871_; lean_object* v___x_873_; uint8_t v_isShared_874_; uint8_t v_isSharedCheck_881_; 
v_numSuccessful_868_ = lean_ctor_get(v_stats_867_, 0);
v_numFailed_869_ = lean_ctor_get(v_stats_867_, 1);
v_elapsedSuccessful_870_ = lean_ctor_get(v_stats_867_, 2);
v_elapsedFailed_871_ = lean_ctor_get(v_stats_867_, 3);
v_isSharedCheck_881_ = !lean_is_exclusive(v_stats_867_);
if (v_isSharedCheck_881_ == 0)
{
v___x_873_ = v_stats_867_;
v_isShared_874_ = v_isSharedCheck_881_;
goto v_resetjp_872_;
}
else
{
lean_inc(v_elapsedFailed_871_);
lean_inc(v_elapsedSuccessful_870_);
lean_inc(v_numFailed_869_);
lean_inc(v_numSuccessful_868_);
lean_dec(v_stats_867_);
v___x_873_ = lean_box(0);
v_isShared_874_ = v_isSharedCheck_881_;
goto v_resetjp_872_;
}
v_resetjp_872_:
{
lean_object* v___x_875_; lean_object* v___x_876_; lean_object* v___x_877_; lean_object* v_stats_879_; 
v___x_875_ = lean_unsigned_to_nat(1u);
v___x_876_ = lean_nat_add(v_numFailed_869_, v___x_875_);
lean_dec(v_numFailed_869_);
v___x_877_ = lean_nat_add(v_elapsedFailed_871_, v_elapsed_858_);
lean_dec(v_elapsedFailed_871_);
if (v_isShared_874_ == 0)
{
lean_ctor_set(v___x_873_, 3, v___x_877_);
lean_ctor_set(v___x_873_, 1, v___x_876_);
v_stats_879_ = v___x_873_;
goto v_reusejp_878_;
}
else
{
lean_object* v_reuseFailAlloc_880_; 
v_reuseFailAlloc_880_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_880_, 0, v_numSuccessful_868_);
lean_ctor_set(v_reuseFailAlloc_880_, 1, v___x_876_);
lean_ctor_set(v_reuseFailAlloc_880_, 2, v_elapsedSuccessful_870_);
lean_ctor_set(v_reuseFailAlloc_880_, 3, v___x_877_);
v_stats_879_ = v_reuseFailAlloc_880_;
goto v_reusejp_878_;
}
v_reusejp_878_:
{
v_stats_861_ = v_stats_879_;
goto v___jp_860_;
}
}
}
else
{
lean_object* v_numSuccessful_882_; lean_object* v_numFailed_883_; lean_object* v_elapsedSuccessful_884_; lean_object* v_elapsedFailed_885_; lean_object* v___x_887_; uint8_t v_isShared_888_; uint8_t v_isSharedCheck_895_; 
v_numSuccessful_882_ = lean_ctor_get(v_stats_867_, 0);
v_numFailed_883_ = lean_ctor_get(v_stats_867_, 1);
v_elapsedSuccessful_884_ = lean_ctor_get(v_stats_867_, 2);
v_elapsedFailed_885_ = lean_ctor_get(v_stats_867_, 3);
v_isSharedCheck_895_ = !lean_is_exclusive(v_stats_867_);
if (v_isSharedCheck_895_ == 0)
{
v___x_887_ = v_stats_867_;
v_isShared_888_ = v_isSharedCheck_895_;
goto v_resetjp_886_;
}
else
{
lean_inc(v_elapsedFailed_885_);
lean_inc(v_elapsedSuccessful_884_);
lean_inc(v_numFailed_883_);
lean_inc(v_numSuccessful_882_);
lean_dec(v_stats_867_);
v___x_887_ = lean_box(0);
v_isShared_888_ = v_isSharedCheck_895_;
goto v_resetjp_886_;
}
v_resetjp_886_:
{
lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; lean_object* v_stats_893_; 
v___x_889_ = lean_unsigned_to_nat(1u);
v___x_890_ = lean_nat_add(v_numSuccessful_882_, v___x_889_);
lean_dec(v_numSuccessful_882_);
v___x_891_ = lean_nat_add(v_elapsedSuccessful_884_, v_elapsed_858_);
lean_dec(v_elapsedSuccessful_884_);
if (v_isShared_888_ == 0)
{
lean_ctor_set(v___x_887_, 2, v___x_891_);
lean_ctor_set(v___x_887_, 0, v___x_890_);
v_stats_893_ = v___x_887_;
goto v_reusejp_892_;
}
else
{
lean_object* v_reuseFailAlloc_894_; 
v_reuseFailAlloc_894_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_894_, 0, v___x_890_);
lean_ctor_set(v_reuseFailAlloc_894_, 1, v_numFailed_883_);
lean_ctor_set(v_reuseFailAlloc_894_, 2, v___x_891_);
lean_ctor_set(v_reuseFailAlloc_894_, 3, v_elapsedFailed_885_);
v_stats_893_ = v_reuseFailAlloc_894_;
goto v_reusejp_892_;
}
v_reusejp_892_:
{
v_stats_861_ = v_stats_893_;
goto v___jp_860_;
}
}
}
v___jp_860_:
{
lean_object* v___x_862_; size_t v___x_863_; size_t v___x_864_; 
lean_inc(v_rule_857_);
v___x_862_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0___redArg(v_b_854_, v_rule_857_, v_stats_861_);
v___x_863_ = ((size_t)1ULL);
v___x_864_ = lean_usize_add(v_i_852_, v___x_863_);
v_i_852_ = v___x_864_;
v_b_854_ = v___x_862_;
goto _start;
}
}
else
{
return v_b_854_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_ruleStatsTotals_spec__2___boxed(lean_object* v_as_896_, lean_object* v_i_897_, lean_object* v_stop_898_, lean_object* v_b_899_){
_start:
{
size_t v_i_boxed_900_; size_t v_stop_boxed_901_; lean_object* v_res_902_; 
v_i_boxed_900_ = lean_unbox_usize(v_i_897_);
lean_dec(v_i_897_);
v_stop_boxed_901_ = lean_unbox_usize(v_stop_898_);
lean_dec(v_stop_898_);
v_res_902_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_ruleStatsTotals_spec__2(v_as_896_, v_i_boxed_900_, v_stop_boxed_901_, v_b_899_);
lean_dec_ref(v_as_896_);
return v_res_902_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stats_ruleStatsTotals(lean_object* v_p_903_, lean_object* v_init_904_){
_start:
{
lean_object* v_ruleStats_905_; lean_object* v___x_906_; lean_object* v___x_907_; uint8_t v___x_908_; 
v_ruleStats_905_ = lean_ctor_get(v_p_903_, 8);
v___x_906_ = lean_unsigned_to_nat(0u);
v___x_907_ = lean_array_get_size(v_ruleStats_905_);
v___x_908_ = lean_nat_dec_lt(v___x_906_, v___x_907_);
if (v___x_908_ == 0)
{
return v_init_904_;
}
else
{
uint8_t v___x_909_; 
v___x_909_ = lean_nat_dec_le(v___x_907_, v___x_907_);
if (v___x_909_ == 0)
{
if (v___x_908_ == 0)
{
return v_init_904_;
}
else
{
size_t v___x_910_; size_t v___x_911_; lean_object* v___x_912_; 
v___x_910_ = ((size_t)0ULL);
v___x_911_ = lean_usize_of_nat(v___x_907_);
v___x_912_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_ruleStatsTotals_spec__2(v_ruleStats_905_, v___x_910_, v___x_911_, v_init_904_);
return v___x_912_;
}
}
else
{
size_t v___x_913_; size_t v___x_914_; lean_object* v___x_915_; 
v___x_913_ = ((size_t)0ULL);
v___x_914_ = lean_usize_of_nat(v___x_907_);
v___x_915_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_ruleStatsTotals_spec__2(v_ruleStats_905_, v___x_913_, v___x_914_, v_init_904_);
return v___x_915_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stats_ruleStatsTotals___boxed(lean_object* v_p_916_, lean_object* v_init_917_){
_start:
{
lean_object* v_res_918_; 
v_res_918_ = lp_aesop_Aesop_Stats_ruleStatsTotals(v_p_916_, v_init_917_);
lean_dec_ref(v_p_916_);
return v_res_918_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0(lean_object* v_00_u03b2_919_, lean_object* v_m_920_, lean_object* v_a_921_, lean_object* v_b_922_){
_start:
{
lean_object* v___x_923_; 
v___x_923_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0___redArg(v_m_920_, v_a_921_, v_b_922_);
return v___x_923_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1(lean_object* v_00_u03b2_924_, lean_object* v_m_925_, lean_object* v_a_926_, lean_object* v_fallback_927_){
_start:
{
lean_object* v___x_928_; 
v___x_928_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1___redArg(v_m_925_, v_a_926_, v_fallback_927_);
return v___x_928_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1___boxed(lean_object* v_00_u03b2_929_, lean_object* v_m_930_, lean_object* v_a_931_, lean_object* v_fallback_932_){
_start:
{
lean_object* v_res_933_; 
v_res_933_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1(v_00_u03b2_929_, v_m_930_, v_a_931_, v_fallback_932_);
lean_dec(v_fallback_932_);
lean_dec(v_a_931_);
lean_dec_ref(v_m_930_);
return v_res_933_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__0(lean_object* v_00_u03b2_934_, lean_object* v_a_935_, lean_object* v_x_936_){
_start:
{
uint8_t v___x_937_; 
v___x_937_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__0___redArg(v_a_935_, v_x_936_);
return v___x_937_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__0___boxed(lean_object* v_00_u03b2_938_, lean_object* v_a_939_, lean_object* v_x_940_){
_start:
{
uint8_t v_res_941_; lean_object* v_r_942_; 
v_res_941_ = lp_aesop_Std_DHashMap_Internal_AssocList_contains___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__0(v_00_u03b2_938_, v_a_939_, v_x_940_);
lean_dec(v_x_940_);
lean_dec(v_a_939_);
v_r_942_ = lean_box(v_res_941_);
return v_r_942_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1(lean_object* v_00_u03b2_943_, lean_object* v_data_944_){
_start:
{
lean_object* v___x_945_; 
v___x_945_ = lp_aesop_Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1___redArg(v_data_944_);
return v___x_945_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__2(lean_object* v_00_u03b2_946_, lean_object* v_a_947_, lean_object* v_b_948_, lean_object* v_x_949_){
_start:
{
lean_object* v___x_950_; 
v___x_950_ = lp_aesop_Std_DHashMap_Internal_AssocList_replace___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__2___redArg(v_a_947_, v_b_948_, v_x_949_);
return v___x_950_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1_spec__4(lean_object* v_00_u03b2_951_, lean_object* v_a_952_, lean_object* v_fallback_953_, lean_object* v_x_954_){
_start:
{
lean_object* v___x_955_; 
v___x_955_ = lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1_spec__4___redArg(v_a_952_, v_fallback_953_, v_x_954_);
return v___x_955_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1_spec__4___boxed(lean_object* v_00_u03b2_956_, lean_object* v_a_957_, lean_object* v_fallback_958_, lean_object* v_x_959_){
_start:
{
lean_object* v_res_960_; 
v_res_960_ = lp_aesop_Std_DHashMap_Internal_AssocList_getD___at___00Std_DHashMap_Internal_Raw_u2080_Const_getD___at___00Aesop_Stats_ruleStatsTotals_spec__1_spec__4(v_00_u03b2_956_, v_a_957_, v_fallback_958_, v_x_959_);
lean_dec(v_x_959_);
lean_dec(v_fallback_958_);
lean_dec(v_a_957_);
return v_res_960_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_961_, lean_object* v_i_962_, lean_object* v_source_963_, lean_object* v_target_964_){
_start:
{
lean_object* v___x_965_; 
v___x_965_ = lp_aesop___private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1_spec__2___redArg(v_i_962_, v_source_963_, v_target_964_);
return v___x_965_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1_spec__2_spec__5(lean_object* v_00_u03b2_966_, lean_object* v_x_967_, lean_object* v_x_968_){
_start:
{
lean_object* v___x_969_; 
v___x_969_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00__private_Std_Data_DHashMap_Internal_Defs_0__Std_DHashMap_Internal_Raw_u2080_expand_go___at___00Std_DHashMap_Internal_Raw_u2080_expand___at___00Std_DHashMap_Internal_Raw_u2080_insert___at___00Aesop_Stats_ruleStatsTotals_spec__0_spec__1_spec__2_spec__5___redArg(v_x_967_, v_x_968_);
return v___x_969_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0_spec__0___redArg(lean_object* v_hi_970_, lean_object* v_pivot_971_, lean_object* v_as_972_, lean_object* v_i_973_, lean_object* v_k_974_){
_start:
{
uint8_t v___y_986_; uint8_t v___x_987_; 
v___x_987_ = lean_nat_dec_lt(v_k_974_, v_hi_970_);
if (v___x_987_ == 0)
{
lean_object* v___x_988_; lean_object* v___x_989_; 
lean_dec(v_k_974_);
v___x_988_ = lean_array_fswap(v_as_972_, v_i_973_, v_hi_970_);
v___x_989_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_989_, 0, v_i_973_);
lean_ctor_set(v___x_989_, 1, v___x_988_);
return v___x_989_;
}
else
{
lean_object* v___x_990_; lean_object* v_fst_991_; lean_object* v_snd_992_; lean_object* v_fst_993_; lean_object* v_snd_994_; uint8_t v___x_995_; 
v___x_990_ = lean_array_fget_borrowed(v_as_972_, v_k_974_);
v_fst_991_ = lean_ctor_get(v___x_990_, 0);
v_snd_992_ = lean_ctor_get(v___x_990_, 1);
v_fst_993_ = lean_ctor_get(v_pivot_971_, 0);
v_snd_994_ = lean_ctor_get(v_pivot_971_, 1);
v___x_995_ = lp_aesop_Aesop_RuleStatsTotals_compareByTotalElapsed(v_snd_992_, v_snd_994_);
switch(v___x_995_)
{
case 0:
{
goto v___jp_975_;
}
case 1:
{
if (v___x_995_ == 1)
{
uint8_t v___x_996_; 
v___x_996_ = lp_aesop_Aesop_instOrdDisplayRuleName_ord(v_fst_991_, v_fst_993_);
v___y_986_ = v___x_996_;
goto v___jp_985_;
}
else
{
v___y_986_ = v___x_995_;
goto v___jp_985_;
}
}
default: 
{
goto v___jp_979_;
}
}
}
v___jp_975_:
{
lean_object* v___x_976_; lean_object* v___x_977_; 
v___x_976_ = lean_unsigned_to_nat(1u);
v___x_977_ = lean_nat_add(v_k_974_, v___x_976_);
lean_dec(v_k_974_);
v_k_974_ = v___x_977_;
goto _start;
}
v___jp_979_:
{
lean_object* v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; lean_object* v___x_983_; 
v___x_980_ = lean_array_fswap(v_as_972_, v_i_973_, v_k_974_);
v___x_981_ = lean_unsigned_to_nat(1u);
v___x_982_ = lean_nat_add(v_i_973_, v___x_981_);
lean_dec(v_i_973_);
v___x_983_ = lean_nat_add(v_k_974_, v___x_981_);
lean_dec(v_k_974_);
v_as_972_ = v___x_980_;
v_i_973_ = v___x_982_;
v_k_974_ = v___x_983_;
goto _start;
}
v___jp_985_:
{
if (v___y_986_ == 0)
{
goto v___jp_979_;
}
else
{
goto v___jp_975_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0_spec__0___redArg___boxed(lean_object* v_hi_997_, lean_object* v_pivot_998_, lean_object* v_as_999_, lean_object* v_i_1000_, lean_object* v_k_1001_){
_start:
{
lean_object* v_res_1002_; 
v_res_1002_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0_spec__0___redArg(v_hi_997_, v_pivot_998_, v_as_999_, v_i_1000_, v_k_1001_);
lean_dec_ref(v_pivot_998_);
lean_dec(v_hi_997_);
return v_res_1002_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___redArg___lam__0(uint8_t v___x_1003_, lean_object* v_x_1004_, lean_object* v_x_1005_){
_start:
{
uint8_t v___y_1007_; lean_object* v_fst_1009_; lean_object* v_snd_1010_; lean_object* v_fst_1011_; lean_object* v_snd_1012_; uint8_t v___x_1013_; 
v_fst_1009_ = lean_ctor_get(v_x_1004_, 0);
v_snd_1010_ = lean_ctor_get(v_x_1004_, 1);
v_fst_1011_ = lean_ctor_get(v_x_1005_, 0);
v_snd_1012_ = lean_ctor_get(v_x_1005_, 1);
v___x_1013_ = lp_aesop_Aesop_RuleStatsTotals_compareByTotalElapsed(v_snd_1010_, v_snd_1012_);
switch(v___x_1013_)
{
case 0:
{
uint8_t v___x_1014_; 
v___x_1014_ = 0;
return v___x_1014_;
}
case 1:
{
if (v___x_1013_ == 1)
{
uint8_t v___x_1015_; 
v___x_1015_ = lp_aesop_Aesop_instOrdDisplayRuleName_ord(v_fst_1009_, v_fst_1011_);
v___y_1007_ = v___x_1015_;
goto v___jp_1006_;
}
else
{
v___y_1007_ = v___x_1013_;
goto v___jp_1006_;
}
}
default: 
{
return v___x_1003_;
}
}
v___jp_1006_:
{
if (v___y_1007_ == 0)
{
return v___x_1003_;
}
else
{
uint8_t v___x_1008_; 
v___x_1008_ = 0;
return v___x_1008_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___redArg___lam__0___boxed(lean_object* v___x_1016_, lean_object* v_x_1017_, lean_object* v_x_1018_){
_start:
{
uint8_t v___x_365__boxed_1019_; uint8_t v_res_1020_; lean_object* v_r_1021_; 
v___x_365__boxed_1019_ = lean_unbox(v___x_1016_);
v_res_1020_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___redArg___lam__0(v___x_365__boxed_1019_, v_x_1017_, v_x_1018_);
lean_dec_ref(v_x_1018_);
lean_dec_ref(v_x_1017_);
v_r_1021_ = lean_box(v_res_1020_);
return v_r_1021_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___redArg(lean_object* v_n_1022_, lean_object* v_as_1023_, lean_object* v_lo_1024_, lean_object* v_hi_1025_){
_start:
{
lean_object* v___y_1027_; uint8_t v___x_1037_; 
v___x_1037_ = lean_nat_dec_lt(v_lo_1024_, v_hi_1025_);
if (v___x_1037_ == 0)
{
lean_dec(v_lo_1024_);
return v_as_1023_;
}
else
{
lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v_mid_1040_; lean_object* v___y_1042_; lean_object* v___y_1048_; lean_object* v___x_1053_; lean_object* v___x_1054_; uint8_t v___x_1055_; 
v___x_1038_ = lean_nat_add(v_lo_1024_, v_hi_1025_);
v___x_1039_ = lean_unsigned_to_nat(1u);
v_mid_1040_ = lean_nat_shiftr(v___x_1038_, v___x_1039_);
lean_dec(v___x_1038_);
v___x_1053_ = lean_array_fget_borrowed(v_as_1023_, v_mid_1040_);
v___x_1054_ = lean_array_fget_borrowed(v_as_1023_, v_lo_1024_);
v___x_1055_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___redArg___lam__0(v___x_1037_, v___x_1053_, v___x_1054_);
if (v___x_1055_ == 0)
{
v___y_1048_ = v_as_1023_;
goto v___jp_1047_;
}
else
{
lean_object* v___x_1056_; 
v___x_1056_ = lean_array_fswap(v_as_1023_, v_lo_1024_, v_mid_1040_);
v___y_1048_ = v___x_1056_;
goto v___jp_1047_;
}
v___jp_1041_:
{
lean_object* v___x_1043_; lean_object* v___x_1044_; uint8_t v___x_1045_; 
v___x_1043_ = lean_array_fget_borrowed(v___y_1042_, v_mid_1040_);
v___x_1044_ = lean_array_fget_borrowed(v___y_1042_, v_hi_1025_);
v___x_1045_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___redArg___lam__0(v___x_1037_, v___x_1043_, v___x_1044_);
if (v___x_1045_ == 0)
{
lean_dec(v_mid_1040_);
v___y_1027_ = v___y_1042_;
goto v___jp_1026_;
}
else
{
lean_object* v___x_1046_; 
v___x_1046_ = lean_array_fswap(v___y_1042_, v_mid_1040_, v_hi_1025_);
lean_dec(v_mid_1040_);
v___y_1027_ = v___x_1046_;
goto v___jp_1026_;
}
}
v___jp_1047_:
{
lean_object* v___x_1049_; lean_object* v___x_1050_; uint8_t v___x_1051_; 
v___x_1049_ = lean_array_fget_borrowed(v___y_1048_, v_hi_1025_);
v___x_1050_ = lean_array_fget_borrowed(v___y_1048_, v_lo_1024_);
v___x_1051_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___redArg___lam__0(v___x_1037_, v___x_1049_, v___x_1050_);
if (v___x_1051_ == 0)
{
v___y_1042_ = v___y_1048_;
goto v___jp_1041_;
}
else
{
lean_object* v___x_1052_; 
v___x_1052_ = lean_array_fswap(v___y_1048_, v_lo_1024_, v_hi_1025_);
v___y_1042_ = v___x_1052_;
goto v___jp_1041_;
}
}
}
v___jp_1026_:
{
lean_object* v_pivot_1028_; lean_object* v___x_1029_; lean_object* v_fst_1030_; lean_object* v_snd_1031_; uint8_t v___x_1032_; 
v_pivot_1028_ = lean_array_fget(v___y_1027_, v_hi_1025_);
lean_inc_n(v_lo_1024_, 2);
v___x_1029_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0_spec__0___redArg(v_hi_1025_, v_pivot_1028_, v___y_1027_, v_lo_1024_, v_lo_1024_);
lean_dec(v_pivot_1028_);
v_fst_1030_ = lean_ctor_get(v___x_1029_, 0);
lean_inc(v_fst_1030_);
v_snd_1031_ = lean_ctor_get(v___x_1029_, 1);
lean_inc(v_snd_1031_);
lean_dec_ref(v___x_1029_);
v___x_1032_ = lean_nat_dec_le(v_hi_1025_, v_fst_1030_);
if (v___x_1032_ == 0)
{
lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; 
v___x_1033_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___redArg(v_n_1022_, v_snd_1031_, v_lo_1024_, v_fst_1030_);
v___x_1034_ = lean_unsigned_to_nat(1u);
v___x_1035_ = lean_nat_add(v_fst_1030_, v___x_1034_);
lean_dec(v_fst_1030_);
v_as_1023_ = v___x_1033_;
v_lo_1024_ = v___x_1035_;
goto _start;
}
else
{
lean_dec(v_fst_1030_);
lean_dec(v_lo_1024_);
return v_snd_1031_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___redArg___boxed(lean_object* v_n_1057_, lean_object* v_as_1058_, lean_object* v_lo_1059_, lean_object* v_hi_1060_){
_start:
{
lean_object* v_res_1061_; 
v_res_1061_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___redArg(v_n_1057_, v_as_1058_, v_lo_1059_, v_hi_1060_);
lean_dec(v_hi_1060_);
lean_dec(v_n_1057_);
return v_res_1061_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortRuleStatsTotals(lean_object* v_ts_1062_){
_start:
{
lean_object* v___x_1063_; lean_object* v___x_1064_; uint8_t v___x_1065_; 
v___x_1063_ = lean_array_get_size(v_ts_1062_);
v___x_1064_ = lean_unsigned_to_nat(0u);
v___x_1065_ = lean_nat_dec_eq(v___x_1063_, v___x_1064_);
if (v___x_1065_ == 0)
{
lean_object* v___x_1066_; lean_object* v___x_1067_; lean_object* v___y_1069_; uint8_t v___x_1073_; 
v___x_1066_ = lean_unsigned_to_nat(1u);
v___x_1067_ = lean_nat_sub(v___x_1063_, v___x_1066_);
v___x_1073_ = lean_nat_dec_le(v___x_1064_, v___x_1067_);
if (v___x_1073_ == 0)
{
lean_inc(v___x_1067_);
v___y_1069_ = v___x_1067_;
goto v___jp_1068_;
}
else
{
v___y_1069_ = v___x_1064_;
goto v___jp_1068_;
}
v___jp_1068_:
{
uint8_t v___x_1070_; 
v___x_1070_ = lean_nat_dec_le(v___y_1069_, v___x_1067_);
if (v___x_1070_ == 0)
{
lean_object* v___x_1071_; 
lean_dec(v___x_1067_);
lean_inc(v___y_1069_);
v___x_1071_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___redArg(v___x_1063_, v_ts_1062_, v___y_1069_, v___y_1069_);
lean_dec(v___y_1069_);
return v___x_1071_;
}
else
{
lean_object* v___x_1072_; 
v___x_1072_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___redArg(v___x_1063_, v_ts_1062_, v___y_1069_, v___x_1067_);
lean_dec(v___x_1067_);
return v___x_1072_;
}
}
}
else
{
return v_ts_1062_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0(lean_object* v_n_1074_, lean_object* v_as_1075_, lean_object* v_lo_1076_, lean_object* v_hi_1077_, lean_object* v_w_1078_, lean_object* v_hlo_1079_, lean_object* v_hhi_1080_){
_start:
{
lean_object* v___x_1081_; 
v___x_1081_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___redArg(v_n_1074_, v_as_1075_, v_lo_1076_, v_hi_1077_);
return v___x_1081_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0___boxed(lean_object* v_n_1082_, lean_object* v_as_1083_, lean_object* v_lo_1084_, lean_object* v_hi_1085_, lean_object* v_w_1086_, lean_object* v_hlo_1087_, lean_object* v_hhi_1088_){
_start:
{
lean_object* v_res_1089_; 
v_res_1089_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0(v_n_1082_, v_as_1083_, v_lo_1084_, v_hi_1085_, v_w_1086_, v_hlo_1087_, v_hhi_1088_);
lean_dec(v_hi_1085_);
lean_dec(v_n_1082_);
return v_res_1089_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0_spec__0(lean_object* v_n_1090_, lean_object* v_lo_1091_, lean_object* v_hi_1092_, lean_object* v_hhi_1093_, lean_object* v_pivot_1094_, lean_object* v_as_1095_, lean_object* v_i_1096_, lean_object* v_k_1097_, lean_object* v_ilo_1098_, lean_object* v_ik_1099_, lean_object* v_w_1100_){
_start:
{
lean_object* v___x_1101_; 
v___x_1101_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0_spec__0___redArg(v_hi_1092_, v_pivot_1094_, v_as_1095_, v_i_1096_, v_k_1097_);
return v___x_1101_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0_spec__0___boxed(lean_object* v_n_1102_, lean_object* v_lo_1103_, lean_object* v_hi_1104_, lean_object* v_hhi_1105_, lean_object* v_pivot_1106_, lean_object* v_as_1107_, lean_object* v_i_1108_, lean_object* v_k_1109_, lean_object* v_ilo_1110_, lean_object* v_ik_1111_, lean_object* v_w_1112_){
_start:
{
lean_object* v_res_1113_; 
v_res_1113_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_sortRuleStatsTotals_spec__0_spec__0(v_n_1102_, v_lo_1103_, v_hi_1104_, v_hhi_1105_, v_pivot_1106_, v_as_1107_, v_i_1108_, v_k_1109_, v_ilo_1110_, v_ik_1111_, v_w_1112_);
lean_dec_ref(v_pivot_1106_);
lean_dec(v_hi_1104_);
lean_dec(v_lo_1103_);
lean_dec(v_n_1102_);
return v_res_1113_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; 
v___x_1114_ = lean_unsigned_to_nat(32u);
v___x_1115_ = lean_mk_empty_array_with_capacity(v___x_1114_);
v___x_1116_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1116_, 0, v___x_1115_);
return v___x_1116_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg___closed__1(void){
_start:
{
size_t v___x_1117_; lean_object* v___x_1118_; lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; 
v___x_1117_ = ((size_t)5ULL);
v___x_1118_ = lean_unsigned_to_nat(0u);
v___x_1119_ = lean_unsigned_to_nat(32u);
v___x_1120_ = lean_mk_empty_array_with_capacity(v___x_1119_);
v___x_1121_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg___closed__0);
v___x_1122_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1122_, 0, v___x_1121_);
lean_ctor_set(v___x_1122_, 1, v___x_1120_);
lean_ctor_set(v___x_1122_, 2, v___x_1118_);
lean_ctor_set(v___x_1122_, 3, v___x_1118_);
lean_ctor_set_usize(v___x_1122_, 4, v___x_1117_);
return v___x_1122_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg(lean_object* v___y_1123_){
_start:
{
lean_object* v___x_1125_; lean_object* v_traceState_1126_; lean_object* v_traces_1127_; lean_object* v___x_1128_; lean_object* v_traceState_1129_; lean_object* v_env_1130_; lean_object* v_nextMacroScope_1131_; lean_object* v_ngen_1132_; lean_object* v_auxDeclNGen_1133_; lean_object* v_cache_1134_; lean_object* v_messages_1135_; lean_object* v_infoState_1136_; lean_object* v_snapshotTasks_1137_; lean_object* v___x_1139_; uint8_t v_isShared_1140_; uint8_t v_isSharedCheck_1156_; 
v___x_1125_ = lean_st_ref_get(v___y_1123_);
v_traceState_1126_ = lean_ctor_get(v___x_1125_, 4);
lean_inc_ref(v_traceState_1126_);
lean_dec(v___x_1125_);
v_traces_1127_ = lean_ctor_get(v_traceState_1126_, 0);
lean_inc_ref(v_traces_1127_);
lean_dec_ref(v_traceState_1126_);
v___x_1128_ = lean_st_ref_take(v___y_1123_);
v_traceState_1129_ = lean_ctor_get(v___x_1128_, 4);
v_env_1130_ = lean_ctor_get(v___x_1128_, 0);
v_nextMacroScope_1131_ = lean_ctor_get(v___x_1128_, 1);
v_ngen_1132_ = lean_ctor_get(v___x_1128_, 2);
v_auxDeclNGen_1133_ = lean_ctor_get(v___x_1128_, 3);
v_cache_1134_ = lean_ctor_get(v___x_1128_, 5);
v_messages_1135_ = lean_ctor_get(v___x_1128_, 6);
v_infoState_1136_ = lean_ctor_get(v___x_1128_, 7);
v_snapshotTasks_1137_ = lean_ctor_get(v___x_1128_, 8);
v_isSharedCheck_1156_ = !lean_is_exclusive(v___x_1128_);
if (v_isSharedCheck_1156_ == 0)
{
v___x_1139_ = v___x_1128_;
v_isShared_1140_ = v_isSharedCheck_1156_;
goto v_resetjp_1138_;
}
else
{
lean_inc(v_snapshotTasks_1137_);
lean_inc(v_infoState_1136_);
lean_inc(v_messages_1135_);
lean_inc(v_cache_1134_);
lean_inc(v_traceState_1129_);
lean_inc(v_auxDeclNGen_1133_);
lean_inc(v_ngen_1132_);
lean_inc(v_nextMacroScope_1131_);
lean_inc(v_env_1130_);
lean_dec(v___x_1128_);
v___x_1139_ = lean_box(0);
v_isShared_1140_ = v_isSharedCheck_1156_;
goto v_resetjp_1138_;
}
v_resetjp_1138_:
{
uint64_t v_tid_1141_; lean_object* v___x_1143_; uint8_t v_isShared_1144_; uint8_t v_isSharedCheck_1154_; 
v_tid_1141_ = lean_ctor_get_uint64(v_traceState_1129_, sizeof(void*)*1);
v_isSharedCheck_1154_ = !lean_is_exclusive(v_traceState_1129_);
if (v_isSharedCheck_1154_ == 0)
{
lean_object* v_unused_1155_; 
v_unused_1155_ = lean_ctor_get(v_traceState_1129_, 0);
lean_dec(v_unused_1155_);
v___x_1143_ = v_traceState_1129_;
v_isShared_1144_ = v_isSharedCheck_1154_;
goto v_resetjp_1142_;
}
else
{
lean_dec(v_traceState_1129_);
v___x_1143_ = lean_box(0);
v_isShared_1144_ = v_isSharedCheck_1154_;
goto v_resetjp_1142_;
}
v_resetjp_1142_:
{
lean_object* v___x_1145_; lean_object* v___x_1147_; 
v___x_1145_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg___closed__1, &lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg___closed__1_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg___closed__1);
if (v_isShared_1144_ == 0)
{
lean_ctor_set(v___x_1143_, 0, v___x_1145_);
v___x_1147_ = v___x_1143_;
goto v_reusejp_1146_;
}
else
{
lean_object* v_reuseFailAlloc_1153_; 
v_reuseFailAlloc_1153_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1153_, 0, v___x_1145_);
lean_ctor_set_uint64(v_reuseFailAlloc_1153_, sizeof(void*)*1, v_tid_1141_);
v___x_1147_ = v_reuseFailAlloc_1153_;
goto v_reusejp_1146_;
}
v_reusejp_1146_:
{
lean_object* v___x_1149_; 
if (v_isShared_1140_ == 0)
{
lean_ctor_set(v___x_1139_, 4, v___x_1147_);
v___x_1149_ = v___x_1139_;
goto v_reusejp_1148_;
}
else
{
lean_object* v_reuseFailAlloc_1152_; 
v_reuseFailAlloc_1152_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1152_, 0, v_env_1130_);
lean_ctor_set(v_reuseFailAlloc_1152_, 1, v_nextMacroScope_1131_);
lean_ctor_set(v_reuseFailAlloc_1152_, 2, v_ngen_1132_);
lean_ctor_set(v_reuseFailAlloc_1152_, 3, v_auxDeclNGen_1133_);
lean_ctor_set(v_reuseFailAlloc_1152_, 4, v___x_1147_);
lean_ctor_set(v_reuseFailAlloc_1152_, 5, v_cache_1134_);
lean_ctor_set(v_reuseFailAlloc_1152_, 6, v_messages_1135_);
lean_ctor_set(v_reuseFailAlloc_1152_, 7, v_infoState_1136_);
lean_ctor_set(v_reuseFailAlloc_1152_, 8, v_snapshotTasks_1137_);
v___x_1149_ = v_reuseFailAlloc_1152_;
goto v_reusejp_1148_;
}
v_reusejp_1148_:
{
lean_object* v___x_1150_; lean_object* v___x_1151_; 
v___x_1150_ = lean_st_ref_set(v___y_1123_, v___x_1149_);
v___x_1151_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1151_, 0, v_traces_1127_);
return v___x_1151_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg___boxed(lean_object* v___y_1157_, lean_object* v___y_1158_){
_start:
{
lean_object* v_res_1159_; 
v_res_1159_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg(v___y_1157_);
lean_dec(v___y_1157_);
return v_res_1159_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3(lean_object* v___y_1160_, lean_object* v___y_1161_){
_start:
{
lean_object* v___x_1163_; 
v___x_1163_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg(v___y_1161_);
return v___x_1163_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___boxed(lean_object* v___y_1164_, lean_object* v___y_1165_, lean_object* v___y_1166_){
_start:
{
lean_object* v_res_1167_; 
v_res_1167_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3(v___y_1164_, v___y_1165_);
lean_dec(v___y_1165_);
lean_dec_ref(v___y_1164_);
return v_res_1167_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Option_get___at___00Aesop_Stats_trace_spec__4(lean_object* v_opts_1168_, lean_object* v_opt_1169_){
_start:
{
lean_object* v_name_1170_; lean_object* v_defValue_1171_; lean_object* v_map_1172_; lean_object* v___x_1173_; 
v_name_1170_ = lean_ctor_get(v_opt_1169_, 0);
v_defValue_1171_ = lean_ctor_get(v_opt_1169_, 1);
v_map_1172_ = lean_ctor_get(v_opts_1168_, 0);
v___x_1173_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1172_, v_name_1170_);
if (lean_obj_tag(v___x_1173_) == 0)
{
uint8_t v___x_1174_; 
v___x_1174_ = lean_unbox(v_defValue_1171_);
return v___x_1174_;
}
else
{
lean_object* v_val_1175_; 
v_val_1175_ = lean_ctor_get(v___x_1173_, 0);
lean_inc(v_val_1175_);
lean_dec_ref_known(v___x_1173_, 1);
if (lean_obj_tag(v_val_1175_) == 1)
{
uint8_t v_v_1176_; 
v_v_1176_ = lean_ctor_get_uint8(v_val_1175_, 0);
lean_dec_ref_known(v_val_1175_, 0);
return v_v_1176_;
}
else
{
uint8_t v___x_1177_; 
lean_dec(v_val_1175_);
v___x_1177_ = lean_unbox(v_defValue_1171_);
return v___x_1177_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00Aesop_Stats_trace_spec__4___boxed(lean_object* v_opts_1178_, lean_object* v_opt_1179_){
_start:
{
uint8_t v_res_1180_; lean_object* v_r_1181_; 
v_res_1180_ = lp_aesop_Lean_Option_get___at___00Aesop_Stats_trace_spec__4(v_opts_1178_, v_opt_1179_);
lean_dec_ref(v_opt_1179_);
lean_dec_ref(v_opts_1178_);
v_r_1181_ = lean_box(v_res_1180_);
return v_r_1181_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stats_trace___lam__0(lean_object* v___x_1182_, lean_object* v_x_1183_, lean_object* v___y_1184_, lean_object* v___y_1185_){
_start:
{
lean_object* v___x_1187_; 
v___x_1187_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1187_, 0, v___x_1182_);
return v___x_1187_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stats_trace___lam__0___boxed(lean_object* v___x_1188_, lean_object* v_x_1189_, lean_object* v___y_1190_, lean_object* v___y_1191_, lean_object* v___y_1192_){
_start:
{
lean_object* v_res_1193_; 
v_res_1193_ = lp_aesop_Aesop_Stats_trace___lam__0(v___x_1188_, v_x_1189_, v___y_1190_, v___y_1191_);
lean_dec(v___y_1191_);
lean_dec_ref(v___y_1190_);
lean_dec_ref(v_x_1189_);
return v_res_1193_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Stats_trace_spec__6(lean_object* v_x_1194_, lean_object* v_x_1195_){
_start:
{
if (lean_obj_tag(v_x_1195_) == 0)
{
return v_x_1194_;
}
else
{
lean_object* v_key_1196_; lean_object* v_value_1197_; lean_object* v_tail_1198_; lean_object* v___x_1199_; lean_object* v___x_1200_; 
v_key_1196_ = lean_ctor_get(v_x_1195_, 0);
v_value_1197_ = lean_ctor_get(v_x_1195_, 1);
v_tail_1198_ = lean_ctor_get(v_x_1195_, 2);
lean_inc(v_value_1197_);
lean_inc(v_key_1196_);
v___x_1199_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1199_, 0, v_key_1196_);
lean_ctor_set(v___x_1199_, 1, v_value_1197_);
v___x_1200_ = lean_array_push(v_x_1194_, v___x_1199_);
v_x_1194_ = v___x_1200_;
v_x_1195_ = v_tail_1198_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Stats_trace_spec__6___boxed(lean_object* v_x_1202_, lean_object* v_x_1203_){
_start:
{
lean_object* v_res_1204_; 
v_res_1204_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Stats_trace_spec__6(v_x_1202_, v_x_1203_);
lean_dec(v_x_1203_);
return v_res_1204_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__7(lean_object* v_as_1205_, size_t v_i_1206_, size_t v_stop_1207_, lean_object* v_b_1208_){
_start:
{
uint8_t v___x_1209_; 
v___x_1209_ = lean_usize_dec_eq(v_i_1206_, v_stop_1207_);
if (v___x_1209_ == 0)
{
lean_object* v___x_1210_; lean_object* v___x_1211_; size_t v___x_1212_; size_t v___x_1213_; 
v___x_1210_ = lean_array_uget_borrowed(v_as_1205_, v_i_1206_);
v___x_1211_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_Stats_trace_spec__6(v_b_1208_, v___x_1210_);
v___x_1212_ = ((size_t)1ULL);
v___x_1213_ = lean_usize_add(v_i_1206_, v___x_1212_);
v_i_1206_ = v___x_1213_;
v_b_1208_ = v___x_1211_;
goto _start;
}
else
{
return v_b_1208_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__7___boxed(lean_object* v_as_1215_, lean_object* v_i_1216_, lean_object* v_stop_1217_, lean_object* v_b_1218_){
_start:
{
size_t v_i_boxed_1219_; size_t v_stop_boxed_1220_; lean_object* v_res_1221_; 
v_i_boxed_1219_ = lean_unbox_usize(v_i_1216_);
lean_dec(v_i_1216_);
v_stop_boxed_1220_ = lean_unbox_usize(v_stop_1217_);
lean_dec(v_stop_1217_);
v_res_1221_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__7(v_as_1215_, v_i_boxed_1219_, v_stop_boxed_1220_, v_b_1218_);
lean_dec_ref(v_as_1215_);
return v_res_1221_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__7___redArg(lean_object* v_x_1222_){
_start:
{
if (lean_obj_tag(v_x_1222_) == 0)
{
lean_object* v_a_1224_; lean_object* v___x_1226_; uint8_t v_isShared_1227_; uint8_t v_isSharedCheck_1231_; 
v_a_1224_ = lean_ctor_get(v_x_1222_, 0);
v_isSharedCheck_1231_ = !lean_is_exclusive(v_x_1222_);
if (v_isSharedCheck_1231_ == 0)
{
v___x_1226_ = v_x_1222_;
v_isShared_1227_ = v_isSharedCheck_1231_;
goto v_resetjp_1225_;
}
else
{
lean_inc(v_a_1224_);
lean_dec(v_x_1222_);
v___x_1226_ = lean_box(0);
v_isShared_1227_ = v_isSharedCheck_1231_;
goto v_resetjp_1225_;
}
v_resetjp_1225_:
{
lean_object* v___x_1229_; 
if (v_isShared_1227_ == 0)
{
lean_ctor_set_tag(v___x_1226_, 1);
v___x_1229_ = v___x_1226_;
goto v_reusejp_1228_;
}
else
{
lean_object* v_reuseFailAlloc_1230_; 
v_reuseFailAlloc_1230_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1230_, 0, v_a_1224_);
v___x_1229_ = v_reuseFailAlloc_1230_;
goto v_reusejp_1228_;
}
v_reusejp_1228_:
{
return v___x_1229_;
}
}
}
else
{
lean_object* v_a_1232_; lean_object* v___x_1234_; uint8_t v_isShared_1235_; uint8_t v_isSharedCheck_1239_; 
v_a_1232_ = lean_ctor_get(v_x_1222_, 0);
v_isSharedCheck_1239_ = !lean_is_exclusive(v_x_1222_);
if (v_isSharedCheck_1239_ == 0)
{
v___x_1234_ = v_x_1222_;
v_isShared_1235_ = v_isSharedCheck_1239_;
goto v_resetjp_1233_;
}
else
{
lean_inc(v_a_1232_);
lean_dec(v_x_1222_);
v___x_1234_ = lean_box(0);
v_isShared_1235_ = v_isSharedCheck_1239_;
goto v_resetjp_1233_;
}
v_resetjp_1233_:
{
lean_object* v___x_1237_; 
if (v_isShared_1235_ == 0)
{
lean_ctor_set_tag(v___x_1234_, 0);
v___x_1237_ = v___x_1234_;
goto v_reusejp_1236_;
}
else
{
lean_object* v_reuseFailAlloc_1238_; 
v_reuseFailAlloc_1238_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1238_, 0, v_a_1232_);
v___x_1237_ = v_reuseFailAlloc_1238_;
goto v_reusejp_1236_;
}
v_reusejp_1236_:
{
return v___x_1237_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__7___redArg___boxed(lean_object* v_x_1240_, lean_object* v___y_1241_){
_start:
{
lean_object* v_res_1242_; 
v_res_1242_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__7___redArg(v_x_1240_);
return v_res_1242_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__8(lean_object* v_e_1243_){
_start:
{
if (lean_obj_tag(v_e_1243_) == 0)
{
uint8_t v___x_1244_; 
v___x_1244_ = 2;
return v___x_1244_;
}
else
{
uint8_t v___x_1245_; 
v___x_1245_ = 0;
return v___x_1245_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__8___boxed(lean_object* v_e_1246_){
_start:
{
uint8_t v_res_1247_; lean_object* v_r_1248_; 
v_res_1247_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__8(v_e_1246_);
lean_dec_ref(v_e_1246_);
v_r_1248_ = lean_box(v_res_1247_);
return v_r_1248_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__9(lean_object* v_opts_1249_, lean_object* v_opt_1250_){
_start:
{
lean_object* v_name_1251_; lean_object* v_defValue_1252_; lean_object* v_map_1253_; lean_object* v___x_1254_; 
v_name_1251_ = lean_ctor_get(v_opt_1250_, 0);
v_defValue_1252_ = lean_ctor_get(v_opt_1250_, 1);
v_map_1253_ = lean_ctor_get(v_opts_1249_, 0);
v___x_1254_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_1253_, v_name_1251_);
if (lean_obj_tag(v___x_1254_) == 0)
{
lean_inc(v_defValue_1252_);
return v_defValue_1252_;
}
else
{
lean_object* v_val_1255_; 
v_val_1255_ = lean_ctor_get(v___x_1254_, 0);
lean_inc(v_val_1255_);
lean_dec_ref_known(v___x_1254_, 1);
if (lean_obj_tag(v_val_1255_) == 3)
{
lean_object* v_v_1256_; 
v_v_1256_ = lean_ctor_get(v_val_1255_, 0);
lean_inc(v_v_1256_);
lean_dec_ref_known(v_val_1255_, 1);
return v_v_1256_;
}
else
{
lean_dec(v_val_1255_);
lean_inc(v_defValue_1252_);
return v_defValue_1252_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__9___boxed(lean_object* v_opts_1257_, lean_object* v_opt_1258_){
_start:
{
lean_object* v_res_1259_; 
v_res_1259_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__9(v_opts_1257_, v_opt_1258_);
lean_dec_ref(v_opt_1258_);
lean_dec_ref(v_opts_1257_);
return v_res_1259_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__0(void){
_start:
{
lean_object* v___x_1260_; 
v___x_1260_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_1260_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__1(void){
_start:
{
lean_object* v___x_1261_; lean_object* v___x_1262_; 
v___x_1261_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__0, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__0_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__0);
v___x_1262_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1262_, 0, v___x_1261_);
return v___x_1262_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__2(void){
_start:
{
lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; 
v___x_1263_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__1, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__1_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__1);
v___x_1264_ = lean_unsigned_to_nat(0u);
v___x_1265_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v___x_1265_, 0, v___x_1264_);
lean_ctor_set(v___x_1265_, 1, v___x_1264_);
lean_ctor_set(v___x_1265_, 2, v___x_1264_);
lean_ctor_set(v___x_1265_, 3, v___x_1264_);
lean_ctor_set(v___x_1265_, 4, v___x_1263_);
lean_ctor_set(v___x_1265_, 5, v___x_1263_);
lean_ctor_set(v___x_1265_, 6, v___x_1263_);
lean_ctor_set(v___x_1265_, 7, v___x_1263_);
lean_ctor_set(v___x_1265_, 8, v___x_1263_);
lean_ctor_set(v___x_1265_, 9, v___x_1263_);
return v___x_1265_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__3(void){
_start:
{
lean_object* v___x_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; 
v___x_1266_ = lean_unsigned_to_nat(32u);
v___x_1267_ = lean_mk_empty_array_with_capacity(v___x_1266_);
v___x_1268_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1268_, 0, v___x_1267_);
return v___x_1268_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__4(void){
_start:
{
size_t v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; 
v___x_1269_ = ((size_t)5ULL);
v___x_1270_ = lean_unsigned_to_nat(0u);
v___x_1271_ = lean_unsigned_to_nat(32u);
v___x_1272_ = lean_mk_empty_array_with_capacity(v___x_1271_);
v___x_1273_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__3, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__3_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__3);
v___x_1274_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_1274_, 0, v___x_1273_);
lean_ctor_set(v___x_1274_, 1, v___x_1272_);
lean_ctor_set(v___x_1274_, 2, v___x_1270_);
lean_ctor_set(v___x_1274_, 3, v___x_1270_);
lean_ctor_set_usize(v___x_1274_, 4, v___x_1269_);
return v___x_1274_;
}
}
static lean_object* _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__5(void){
_start:
{
lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; 
v___x_1275_ = lean_box(1);
v___x_1276_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__4, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__4_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__4);
v___x_1277_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__1, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__1_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__1);
v___x_1278_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_1278_, 0, v___x_1277_);
lean_ctor_set(v___x_1278_, 1, v___x_1276_);
lean_ctor_set(v___x_1278_, 2, v___x_1275_);
return v___x_1278_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1(lean_object* v_msgData_1279_, lean_object* v___y_1280_, lean_object* v___y_1281_){
_start:
{
lean_object* v___x_1283_; lean_object* v_env_1284_; lean_object* v_options_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; lean_object* v___x_1290_; 
v___x_1283_ = lean_st_ref_get(v___y_1281_);
v_env_1284_ = lean_ctor_get(v___x_1283_, 0);
lean_inc_ref(v_env_1284_);
lean_dec(v___x_1283_);
v_options_1285_ = lean_ctor_get(v___y_1280_, 2);
v___x_1286_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__2, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__2_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__2);
v___x_1287_ = lean_obj_once(&lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__5, &lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__5_once, _init_lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___closed__5);
lean_inc_ref(v_options_1285_);
v___x_1288_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1288_, 0, v_env_1284_);
lean_ctor_set(v___x_1288_, 1, v___x_1286_);
lean_ctor_set(v___x_1288_, 2, v___x_1287_);
lean_ctor_set(v___x_1288_, 3, v_options_1285_);
v___x_1289_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1289_, 0, v___x_1288_);
lean_ctor_set(v___x_1289_, 1, v_msgData_1279_);
v___x_1290_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1290_, 0, v___x_1289_);
return v___x_1290_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1___boxed(lean_object* v_msgData_1291_, lean_object* v___y_1292_, lean_object* v___y_1293_, lean_object* v___y_1294_){
_start:
{
lean_object* v_res_1295_; 
v_res_1295_ = lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1(v_msgData_1291_, v___y_1292_, v___y_1293_);
lean_dec(v___y_1293_);
lean_dec_ref(v___y_1292_);
return v_res_1295_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__6_spec__7(size_t v_sz_1296_, size_t v_i_1297_, lean_object* v_bs_1298_){
_start:
{
uint8_t v___x_1299_; 
v___x_1299_ = lean_usize_dec_lt(v_i_1297_, v_sz_1296_);
if (v___x_1299_ == 0)
{
return v_bs_1298_;
}
else
{
lean_object* v_v_1300_; lean_object* v_msg_1301_; lean_object* v___x_1302_; lean_object* v_bs_x27_1303_; size_t v___x_1304_; size_t v___x_1305_; lean_object* v___x_1306_; 
v_v_1300_ = lean_array_uget_borrowed(v_bs_1298_, v_i_1297_);
v_msg_1301_ = lean_ctor_get(v_v_1300_, 1);
lean_inc_ref(v_msg_1301_);
v___x_1302_ = lean_unsigned_to_nat(0u);
v_bs_x27_1303_ = lean_array_uset(v_bs_1298_, v_i_1297_, v___x_1302_);
v___x_1304_ = ((size_t)1ULL);
v___x_1305_ = lean_usize_add(v_i_1297_, v___x_1304_);
v___x_1306_ = lean_array_uset(v_bs_x27_1303_, v_i_1297_, v_msg_1301_);
v_i_1297_ = v___x_1305_;
v_bs_1298_ = v___x_1306_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__6_spec__7___boxed(lean_object* v_sz_1308_, lean_object* v_i_1309_, lean_object* v_bs_1310_){
_start:
{
size_t v_sz_boxed_1311_; size_t v_i_boxed_1312_; lean_object* v_res_1313_; 
v_sz_boxed_1311_ = lean_unbox_usize(v_sz_1308_);
lean_dec(v_sz_1308_);
v_i_boxed_1312_ = lean_unbox_usize(v_i_1309_);
lean_dec(v_i_1309_);
v_res_1313_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__6_spec__7(v_sz_boxed_1311_, v_i_boxed_1312_, v_bs_1310_);
return v_res_1313_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__6(lean_object* v_oldTraces_1314_, lean_object* v_data_1315_, lean_object* v_ref_1316_, lean_object* v_msg_1317_, lean_object* v___y_1318_, lean_object* v___y_1319_){
_start:
{
lean_object* v_fileName_1321_; lean_object* v_fileMap_1322_; lean_object* v_options_1323_; lean_object* v_currRecDepth_1324_; lean_object* v_maxRecDepth_1325_; lean_object* v_ref_1326_; lean_object* v_currNamespace_1327_; lean_object* v_openDecls_1328_; lean_object* v_initHeartbeats_1329_; lean_object* v_maxHeartbeats_1330_; lean_object* v_quotContext_1331_; lean_object* v_currMacroScope_1332_; uint8_t v_diag_1333_; lean_object* v_cancelTk_x3f_1334_; uint8_t v_suppressElabErrors_1335_; lean_object* v_inheritedTraceOptions_1336_; lean_object* v___x_1337_; lean_object* v_traceState_1338_; lean_object* v_traces_1339_; lean_object* v_ref_1340_; lean_object* v___x_1341_; lean_object* v___x_1342_; size_t v_sz_1343_; size_t v___x_1344_; lean_object* v___x_1345_; lean_object* v_msg_1346_; lean_object* v___x_1347_; lean_object* v_a_1348_; lean_object* v___x_1350_; uint8_t v_isShared_1351_; uint8_t v_isSharedCheck_1385_; 
v_fileName_1321_ = lean_ctor_get(v___y_1318_, 0);
v_fileMap_1322_ = lean_ctor_get(v___y_1318_, 1);
v_options_1323_ = lean_ctor_get(v___y_1318_, 2);
v_currRecDepth_1324_ = lean_ctor_get(v___y_1318_, 3);
v_maxRecDepth_1325_ = lean_ctor_get(v___y_1318_, 4);
v_ref_1326_ = lean_ctor_get(v___y_1318_, 5);
v_currNamespace_1327_ = lean_ctor_get(v___y_1318_, 6);
v_openDecls_1328_ = lean_ctor_get(v___y_1318_, 7);
v_initHeartbeats_1329_ = lean_ctor_get(v___y_1318_, 8);
v_maxHeartbeats_1330_ = lean_ctor_get(v___y_1318_, 9);
v_quotContext_1331_ = lean_ctor_get(v___y_1318_, 10);
v_currMacroScope_1332_ = lean_ctor_get(v___y_1318_, 11);
v_diag_1333_ = lean_ctor_get_uint8(v___y_1318_, sizeof(void*)*14);
v_cancelTk_x3f_1334_ = lean_ctor_get(v___y_1318_, 12);
v_suppressElabErrors_1335_ = lean_ctor_get_uint8(v___y_1318_, sizeof(void*)*14 + 1);
v_inheritedTraceOptions_1336_ = lean_ctor_get(v___y_1318_, 13);
v___x_1337_ = lean_st_ref_get(v___y_1319_);
v_traceState_1338_ = lean_ctor_get(v___x_1337_, 4);
lean_inc_ref(v_traceState_1338_);
lean_dec(v___x_1337_);
v_traces_1339_ = lean_ctor_get(v_traceState_1338_, 0);
lean_inc_ref(v_traces_1339_);
lean_dec_ref(v_traceState_1338_);
v_ref_1340_ = l_Lean_replaceRef(v_ref_1316_, v_ref_1326_);
lean_inc_ref(v_inheritedTraceOptions_1336_);
lean_inc(v_cancelTk_x3f_1334_);
lean_inc(v_currMacroScope_1332_);
lean_inc(v_quotContext_1331_);
lean_inc(v_maxHeartbeats_1330_);
lean_inc(v_initHeartbeats_1329_);
lean_inc(v_openDecls_1328_);
lean_inc(v_currNamespace_1327_);
lean_inc(v_maxRecDepth_1325_);
lean_inc(v_currRecDepth_1324_);
lean_inc_ref(v_options_1323_);
lean_inc_ref(v_fileMap_1322_);
lean_inc_ref(v_fileName_1321_);
v___x_1341_ = lean_alloc_ctor(0, 14, 2);
lean_ctor_set(v___x_1341_, 0, v_fileName_1321_);
lean_ctor_set(v___x_1341_, 1, v_fileMap_1322_);
lean_ctor_set(v___x_1341_, 2, v_options_1323_);
lean_ctor_set(v___x_1341_, 3, v_currRecDepth_1324_);
lean_ctor_set(v___x_1341_, 4, v_maxRecDepth_1325_);
lean_ctor_set(v___x_1341_, 5, v_ref_1340_);
lean_ctor_set(v___x_1341_, 6, v_currNamespace_1327_);
lean_ctor_set(v___x_1341_, 7, v_openDecls_1328_);
lean_ctor_set(v___x_1341_, 8, v_initHeartbeats_1329_);
lean_ctor_set(v___x_1341_, 9, v_maxHeartbeats_1330_);
lean_ctor_set(v___x_1341_, 10, v_quotContext_1331_);
lean_ctor_set(v___x_1341_, 11, v_currMacroScope_1332_);
lean_ctor_set(v___x_1341_, 12, v_cancelTk_x3f_1334_);
lean_ctor_set(v___x_1341_, 13, v_inheritedTraceOptions_1336_);
lean_ctor_set_uint8(v___x_1341_, sizeof(void*)*14, v_diag_1333_);
lean_ctor_set_uint8(v___x_1341_, sizeof(void*)*14 + 1, v_suppressElabErrors_1335_);
v___x_1342_ = l_Lean_PersistentArray_toArray___redArg(v_traces_1339_);
lean_dec_ref(v_traces_1339_);
v_sz_1343_ = lean_array_size(v___x_1342_);
v___x_1344_ = ((size_t)0ULL);
v___x_1345_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__6_spec__7(v_sz_1343_, v___x_1344_, v___x_1342_);
v_msg_1346_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v_msg_1346_, 0, v_data_1315_);
lean_ctor_set(v_msg_1346_, 1, v_msg_1317_);
lean_ctor_set(v_msg_1346_, 2, v___x_1345_);
v___x_1347_ = lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1(v_msg_1346_, v___x_1341_, v___y_1319_);
lean_dec_ref_known(v___x_1341_, 14);
v_a_1348_ = lean_ctor_get(v___x_1347_, 0);
v_isSharedCheck_1385_ = !lean_is_exclusive(v___x_1347_);
if (v_isSharedCheck_1385_ == 0)
{
v___x_1350_ = v___x_1347_;
v_isShared_1351_ = v_isSharedCheck_1385_;
goto v_resetjp_1349_;
}
else
{
lean_inc(v_a_1348_);
lean_dec(v___x_1347_);
v___x_1350_ = lean_box(0);
v_isShared_1351_ = v_isSharedCheck_1385_;
goto v_resetjp_1349_;
}
v_resetjp_1349_:
{
lean_object* v___x_1352_; lean_object* v_traceState_1353_; lean_object* v_env_1354_; lean_object* v_nextMacroScope_1355_; lean_object* v_ngen_1356_; lean_object* v_auxDeclNGen_1357_; lean_object* v_cache_1358_; lean_object* v_messages_1359_; lean_object* v_infoState_1360_; lean_object* v_snapshotTasks_1361_; lean_object* v___x_1363_; uint8_t v_isShared_1364_; uint8_t v_isSharedCheck_1384_; 
v___x_1352_ = lean_st_ref_take(v___y_1319_);
v_traceState_1353_ = lean_ctor_get(v___x_1352_, 4);
v_env_1354_ = lean_ctor_get(v___x_1352_, 0);
v_nextMacroScope_1355_ = lean_ctor_get(v___x_1352_, 1);
v_ngen_1356_ = lean_ctor_get(v___x_1352_, 2);
v_auxDeclNGen_1357_ = lean_ctor_get(v___x_1352_, 3);
v_cache_1358_ = lean_ctor_get(v___x_1352_, 5);
v_messages_1359_ = lean_ctor_get(v___x_1352_, 6);
v_infoState_1360_ = lean_ctor_get(v___x_1352_, 7);
v_snapshotTasks_1361_ = lean_ctor_get(v___x_1352_, 8);
v_isSharedCheck_1384_ = !lean_is_exclusive(v___x_1352_);
if (v_isSharedCheck_1384_ == 0)
{
v___x_1363_ = v___x_1352_;
v_isShared_1364_ = v_isSharedCheck_1384_;
goto v_resetjp_1362_;
}
else
{
lean_inc(v_snapshotTasks_1361_);
lean_inc(v_infoState_1360_);
lean_inc(v_messages_1359_);
lean_inc(v_cache_1358_);
lean_inc(v_traceState_1353_);
lean_inc(v_auxDeclNGen_1357_);
lean_inc(v_ngen_1356_);
lean_inc(v_nextMacroScope_1355_);
lean_inc(v_env_1354_);
lean_dec(v___x_1352_);
v___x_1363_ = lean_box(0);
v_isShared_1364_ = v_isSharedCheck_1384_;
goto v_resetjp_1362_;
}
v_resetjp_1362_:
{
uint64_t v_tid_1365_; lean_object* v___x_1367_; uint8_t v_isShared_1368_; uint8_t v_isSharedCheck_1382_; 
v_tid_1365_ = lean_ctor_get_uint64(v_traceState_1353_, sizeof(void*)*1);
v_isSharedCheck_1382_ = !lean_is_exclusive(v_traceState_1353_);
if (v_isSharedCheck_1382_ == 0)
{
lean_object* v_unused_1383_; 
v_unused_1383_ = lean_ctor_get(v_traceState_1353_, 0);
lean_dec(v_unused_1383_);
v___x_1367_ = v_traceState_1353_;
v_isShared_1368_ = v_isSharedCheck_1382_;
goto v_resetjp_1366_;
}
else
{
lean_dec(v_traceState_1353_);
v___x_1367_ = lean_box(0);
v_isShared_1368_ = v_isSharedCheck_1382_;
goto v_resetjp_1366_;
}
v_resetjp_1366_:
{
lean_object* v___x_1369_; lean_object* v___x_1370_; lean_object* v___x_1372_; 
v___x_1369_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1369_, 0, v_ref_1316_);
lean_ctor_set(v___x_1369_, 1, v_a_1348_);
v___x_1370_ = l_Lean_PersistentArray_push___redArg(v_oldTraces_1314_, v___x_1369_);
if (v_isShared_1368_ == 0)
{
lean_ctor_set(v___x_1367_, 0, v___x_1370_);
v___x_1372_ = v___x_1367_;
goto v_reusejp_1371_;
}
else
{
lean_object* v_reuseFailAlloc_1381_; 
v_reuseFailAlloc_1381_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1381_, 0, v___x_1370_);
lean_ctor_set_uint64(v_reuseFailAlloc_1381_, sizeof(void*)*1, v_tid_1365_);
v___x_1372_ = v_reuseFailAlloc_1381_;
goto v_reusejp_1371_;
}
v_reusejp_1371_:
{
lean_object* v___x_1374_; 
if (v_isShared_1364_ == 0)
{
lean_ctor_set(v___x_1363_, 4, v___x_1372_);
v___x_1374_ = v___x_1363_;
goto v_reusejp_1373_;
}
else
{
lean_object* v_reuseFailAlloc_1380_; 
v_reuseFailAlloc_1380_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1380_, 0, v_env_1354_);
lean_ctor_set(v_reuseFailAlloc_1380_, 1, v_nextMacroScope_1355_);
lean_ctor_set(v_reuseFailAlloc_1380_, 2, v_ngen_1356_);
lean_ctor_set(v_reuseFailAlloc_1380_, 3, v_auxDeclNGen_1357_);
lean_ctor_set(v_reuseFailAlloc_1380_, 4, v___x_1372_);
lean_ctor_set(v_reuseFailAlloc_1380_, 5, v_cache_1358_);
lean_ctor_set(v_reuseFailAlloc_1380_, 6, v_messages_1359_);
lean_ctor_set(v_reuseFailAlloc_1380_, 7, v_infoState_1360_);
lean_ctor_set(v_reuseFailAlloc_1380_, 8, v_snapshotTasks_1361_);
v___x_1374_ = v_reuseFailAlloc_1380_;
goto v_reusejp_1373_;
}
v_reusejp_1373_:
{
lean_object* v___x_1375_; lean_object* v___x_1376_; lean_object* v___x_1378_; 
v___x_1375_ = lean_st_ref_set(v___y_1319_, v___x_1374_);
v___x_1376_ = lean_box(0);
if (v_isShared_1351_ == 0)
{
lean_ctor_set(v___x_1350_, 0, v___x_1376_);
v___x_1378_ = v___x_1350_;
goto v_reusejp_1377_;
}
else
{
lean_object* v_reuseFailAlloc_1379_; 
v_reuseFailAlloc_1379_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1379_, 0, v___x_1376_);
v___x_1378_ = v_reuseFailAlloc_1379_;
goto v_reusejp_1377_;
}
v_reusejp_1377_:
{
return v___x_1378_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__6___boxed(lean_object* v_oldTraces_1386_, lean_object* v_data_1387_, lean_object* v_ref_1388_, lean_object* v_msg_1389_, lean_object* v___y_1390_, lean_object* v___y_1391_, lean_object* v___y_1392_){
_start:
{
lean_object* v_res_1393_; 
v_res_1393_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__6(v_oldTraces_1386_, v_data_1387_, v_ref_1388_, v_msg_1389_, v___y_1390_, v___y_1391_);
lean_dec(v___y_1391_);
lean_dec_ref(v___y_1390_);
return v_res_1393_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__0(void){
_start:
{
lean_object* v___x_1394_; double v___x_1395_; 
v___x_1394_ = lean_unsigned_to_nat(0u);
v___x_1395_ = lean_float_of_nat(v___x_1394_);
return v___x_1395_;
}
}
static lean_object* _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__2(void){
_start:
{
lean_object* v___x_1397_; lean_object* v___x_1398_; 
v___x_1397_ = ((lean_object*)(lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__1));
v___x_1398_ = l_Lean_stringToMessageData(v___x_1397_);
return v___x_1398_;
}
}
static double _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__3(void){
_start:
{
lean_object* v___x_1399_; double v___x_1400_; 
v___x_1399_ = lean_unsigned_to_nat(1000u);
v___x_1400_ = lean_float_of_nat(v___x_1399_);
return v___x_1400_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5(lean_object* v_cls_1401_, uint8_t v_collapsed_1402_, lean_object* v_tag_1403_, lean_object* v_opts_1404_, uint8_t v_clsEnabled_1405_, lean_object* v_oldTraces_1406_, lean_object* v_msg_1407_, lean_object* v_resStartStop_1408_, lean_object* v___y_1409_, lean_object* v___y_1410_){
_start:
{
lean_object* v_fst_1412_; lean_object* v_snd_1413_; lean_object* v___y_1415_; lean_object* v___y_1416_; lean_object* v_data_1417_; lean_object* v_fst_1420_; lean_object* v_snd_1421_; lean_object* v___x_1422_; uint8_t v___x_1423_; lean_object* v___y_1425_; lean_object* v_a_1426_; uint8_t v___y_1441_; double v___y_1472_; 
v_fst_1412_ = lean_ctor_get(v_resStartStop_1408_, 0);
lean_inc(v_fst_1412_);
v_snd_1413_ = lean_ctor_get(v_resStartStop_1408_, 1);
lean_inc(v_snd_1413_);
lean_dec_ref(v_resStartStop_1408_);
v_fst_1420_ = lean_ctor_get(v_snd_1413_, 0);
lean_inc(v_fst_1420_);
v_snd_1421_ = lean_ctor_get(v_snd_1413_, 1);
lean_inc(v_snd_1421_);
lean_dec(v_snd_1413_);
v___x_1422_ = l_Lean_trace_profiler;
v___x_1423_ = lp_aesop_Lean_Option_get___at___00Aesop_Stats_trace_spec__4(v_opts_1404_, v___x_1422_);
if (v___x_1423_ == 0)
{
v___y_1441_ = v___x_1423_;
goto v___jp_1440_;
}
else
{
lean_object* v___x_1477_; uint8_t v___x_1478_; 
v___x_1477_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1478_ = lp_aesop_Lean_Option_get___at___00Aesop_Stats_trace_spec__4(v_opts_1404_, v___x_1477_);
if (v___x_1478_ == 0)
{
lean_object* v___x_1479_; lean_object* v___x_1480_; double v___x_1481_; double v___x_1482_; double v___x_1483_; 
v___x_1479_ = l_Lean_trace_profiler_threshold;
v___x_1480_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__9(v_opts_1404_, v___x_1479_);
v___x_1481_ = lean_float_of_nat(v___x_1480_);
v___x_1482_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__3, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__3_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__3);
v___x_1483_ = lean_float_div(v___x_1481_, v___x_1482_);
v___y_1472_ = v___x_1483_;
goto v___jp_1471_;
}
else
{
lean_object* v___x_1484_; lean_object* v___x_1485_; double v___x_1486_; 
v___x_1484_ = l_Lean_trace_profiler_threshold;
v___x_1485_ = lp_aesop_Lean_Option_get___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__9(v_opts_1404_, v___x_1484_);
v___x_1486_ = lean_float_of_nat(v___x_1485_);
v___y_1472_ = v___x_1486_;
goto v___jp_1471_;
}
}
v___jp_1414_:
{
lean_object* v___x_1418_; 
lean_inc(v___y_1416_);
v___x_1418_ = lp_aesop___private_Lean_Util_Trace_0__Lean_addTraceNode___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__6(v_oldTraces_1406_, v_data_1417_, v___y_1416_, v___y_1415_, v___y_1409_, v___y_1410_);
if (lean_obj_tag(v___x_1418_) == 0)
{
lean_object* v___x_1419_; 
lean_dec_ref_known(v___x_1418_, 1);
v___x_1419_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__7___redArg(v_fst_1412_);
return v___x_1419_;
}
else
{
lean_dec(v_fst_1412_);
return v___x_1418_;
}
}
v___jp_1424_:
{
uint8_t v_result_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; double v___x_1430_; lean_object* v_data_1431_; 
v_result_1427_ = lp_aesop_Lean_Except_toTraceResult___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__8(v_fst_1412_);
v___x_1428_ = lean_box(v_result_1427_);
v___x_1429_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1429_, 0, v___x_1428_);
v___x_1430_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__0);
lean_inc_ref(v_tag_1403_);
lean_inc_ref(v___x_1429_);
lean_inc(v_cls_1401_);
v_data_1431_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1431_, 0, v_cls_1401_);
lean_ctor_set(v_data_1431_, 1, v___x_1429_);
lean_ctor_set(v_data_1431_, 2, v_tag_1403_);
lean_ctor_set_float(v_data_1431_, sizeof(void*)*3, v___x_1430_);
lean_ctor_set_float(v_data_1431_, sizeof(void*)*3 + 8, v___x_1430_);
lean_ctor_set_uint8(v_data_1431_, sizeof(void*)*3 + 16, v_collapsed_1402_);
if (v___x_1423_ == 0)
{
lean_dec_ref_known(v___x_1429_, 1);
lean_dec(v_snd_1421_);
lean_dec(v_fst_1420_);
lean_dec_ref(v_tag_1403_);
lean_dec(v_cls_1401_);
v___y_1415_ = v_a_1426_;
v___y_1416_ = v___y_1425_;
v_data_1417_ = v_data_1431_;
goto v___jp_1414_;
}
else
{
lean_object* v_data_1432_; double v___x_1433_; double v___x_1434_; 
lean_dec_ref_known(v_data_1431_, 3);
v_data_1432_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v_data_1432_, 0, v_cls_1401_);
lean_ctor_set(v_data_1432_, 1, v___x_1429_);
lean_ctor_set(v_data_1432_, 2, v_tag_1403_);
v___x_1433_ = lean_unbox_float(v_fst_1420_);
lean_dec(v_fst_1420_);
lean_ctor_set_float(v_data_1432_, sizeof(void*)*3, v___x_1433_);
v___x_1434_ = lean_unbox_float(v_snd_1421_);
lean_dec(v_snd_1421_);
lean_ctor_set_float(v_data_1432_, sizeof(void*)*3 + 8, v___x_1434_);
lean_ctor_set_uint8(v_data_1432_, sizeof(void*)*3 + 16, v_collapsed_1402_);
v___y_1415_ = v_a_1426_;
v___y_1416_ = v___y_1425_;
v_data_1417_ = v_data_1432_;
goto v___jp_1414_;
}
}
v___jp_1435_:
{
lean_object* v_ref_1436_; lean_object* v___x_1437_; 
v_ref_1436_ = lean_ctor_get(v___y_1409_, 5);
lean_inc(v___y_1410_);
lean_inc_ref(v___y_1409_);
lean_inc(v_fst_1412_);
v___x_1437_ = lean_apply_4(v_msg_1407_, v_fst_1412_, v___y_1409_, v___y_1410_, lean_box(0));
if (lean_obj_tag(v___x_1437_) == 0)
{
lean_object* v_a_1438_; 
v_a_1438_ = lean_ctor_get(v___x_1437_, 0);
lean_inc(v_a_1438_);
lean_dec_ref_known(v___x_1437_, 1);
v___y_1425_ = v_ref_1436_;
v_a_1426_ = v_a_1438_;
goto v___jp_1424_;
}
else
{
lean_object* v___x_1439_; 
lean_dec_ref_known(v___x_1437_, 1);
v___x_1439_ = lean_obj_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__2, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__2_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__2);
v___y_1425_ = v_ref_1436_;
v_a_1426_ = v___x_1439_;
goto v___jp_1424_;
}
}
v___jp_1440_:
{
if (v_clsEnabled_1405_ == 0)
{
if (v___y_1441_ == 0)
{
lean_object* v___x_1442_; lean_object* v_traceState_1443_; lean_object* v_env_1444_; lean_object* v_nextMacroScope_1445_; lean_object* v_ngen_1446_; lean_object* v_auxDeclNGen_1447_; lean_object* v_cache_1448_; lean_object* v_messages_1449_; lean_object* v_infoState_1450_; lean_object* v_snapshotTasks_1451_; lean_object* v___x_1453_; uint8_t v_isShared_1454_; uint8_t v_isSharedCheck_1470_; 
lean_dec(v_snd_1421_);
lean_dec(v_fst_1420_);
lean_dec_ref(v_msg_1407_);
lean_dec_ref(v_tag_1403_);
lean_dec(v_cls_1401_);
v___x_1442_ = lean_st_ref_take(v___y_1410_);
v_traceState_1443_ = lean_ctor_get(v___x_1442_, 4);
v_env_1444_ = lean_ctor_get(v___x_1442_, 0);
v_nextMacroScope_1445_ = lean_ctor_get(v___x_1442_, 1);
v_ngen_1446_ = lean_ctor_get(v___x_1442_, 2);
v_auxDeclNGen_1447_ = lean_ctor_get(v___x_1442_, 3);
v_cache_1448_ = lean_ctor_get(v___x_1442_, 5);
v_messages_1449_ = lean_ctor_get(v___x_1442_, 6);
v_infoState_1450_ = lean_ctor_get(v___x_1442_, 7);
v_snapshotTasks_1451_ = lean_ctor_get(v___x_1442_, 8);
v_isSharedCheck_1470_ = !lean_is_exclusive(v___x_1442_);
if (v_isSharedCheck_1470_ == 0)
{
v___x_1453_ = v___x_1442_;
v_isShared_1454_ = v_isSharedCheck_1470_;
goto v_resetjp_1452_;
}
else
{
lean_inc(v_snapshotTasks_1451_);
lean_inc(v_infoState_1450_);
lean_inc(v_messages_1449_);
lean_inc(v_cache_1448_);
lean_inc(v_traceState_1443_);
lean_inc(v_auxDeclNGen_1447_);
lean_inc(v_ngen_1446_);
lean_inc(v_nextMacroScope_1445_);
lean_inc(v_env_1444_);
lean_dec(v___x_1442_);
v___x_1453_ = lean_box(0);
v_isShared_1454_ = v_isSharedCheck_1470_;
goto v_resetjp_1452_;
}
v_resetjp_1452_:
{
uint64_t v_tid_1455_; lean_object* v_traces_1456_; lean_object* v___x_1458_; uint8_t v_isShared_1459_; uint8_t v_isSharedCheck_1469_; 
v_tid_1455_ = lean_ctor_get_uint64(v_traceState_1443_, sizeof(void*)*1);
v_traces_1456_ = lean_ctor_get(v_traceState_1443_, 0);
v_isSharedCheck_1469_ = !lean_is_exclusive(v_traceState_1443_);
if (v_isSharedCheck_1469_ == 0)
{
v___x_1458_ = v_traceState_1443_;
v_isShared_1459_ = v_isSharedCheck_1469_;
goto v_resetjp_1457_;
}
else
{
lean_inc(v_traces_1456_);
lean_dec(v_traceState_1443_);
v___x_1458_ = lean_box(0);
v_isShared_1459_ = v_isSharedCheck_1469_;
goto v_resetjp_1457_;
}
v_resetjp_1457_:
{
lean_object* v___x_1460_; lean_object* v___x_1462_; 
v___x_1460_ = l_Lean_PersistentArray_append___redArg(v_oldTraces_1406_, v_traces_1456_);
lean_dec_ref(v_traces_1456_);
if (v_isShared_1459_ == 0)
{
lean_ctor_set(v___x_1458_, 0, v___x_1460_);
v___x_1462_ = v___x_1458_;
goto v_reusejp_1461_;
}
else
{
lean_object* v_reuseFailAlloc_1468_; 
v_reuseFailAlloc_1468_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1468_, 0, v___x_1460_);
lean_ctor_set_uint64(v_reuseFailAlloc_1468_, sizeof(void*)*1, v_tid_1455_);
v___x_1462_ = v_reuseFailAlloc_1468_;
goto v_reusejp_1461_;
}
v_reusejp_1461_:
{
lean_object* v___x_1464_; 
if (v_isShared_1454_ == 0)
{
lean_ctor_set(v___x_1453_, 4, v___x_1462_);
v___x_1464_ = v___x_1453_;
goto v_reusejp_1463_;
}
else
{
lean_object* v_reuseFailAlloc_1467_; 
v_reuseFailAlloc_1467_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1467_, 0, v_env_1444_);
lean_ctor_set(v_reuseFailAlloc_1467_, 1, v_nextMacroScope_1445_);
lean_ctor_set(v_reuseFailAlloc_1467_, 2, v_ngen_1446_);
lean_ctor_set(v_reuseFailAlloc_1467_, 3, v_auxDeclNGen_1447_);
lean_ctor_set(v_reuseFailAlloc_1467_, 4, v___x_1462_);
lean_ctor_set(v_reuseFailAlloc_1467_, 5, v_cache_1448_);
lean_ctor_set(v_reuseFailAlloc_1467_, 6, v_messages_1449_);
lean_ctor_set(v_reuseFailAlloc_1467_, 7, v_infoState_1450_);
lean_ctor_set(v_reuseFailAlloc_1467_, 8, v_snapshotTasks_1451_);
v___x_1464_ = v_reuseFailAlloc_1467_;
goto v_reusejp_1463_;
}
v_reusejp_1463_:
{
lean_object* v___x_1465_; lean_object* v___x_1466_; 
v___x_1465_ = lean_st_ref_set(v___y_1410_, v___x_1464_);
v___x_1466_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__7___redArg(v_fst_1412_);
return v___x_1466_;
}
}
}
}
}
else
{
goto v___jp_1435_;
}
}
else
{
goto v___jp_1435_;
}
}
v___jp_1471_:
{
double v___x_1473_; double v___x_1474_; double v___x_1475_; uint8_t v___x_1476_; 
v___x_1473_ = lean_unbox_float(v_snd_1421_);
v___x_1474_ = lean_unbox_float(v_fst_1420_);
v___x_1475_ = lean_float_sub(v___x_1473_, v___x_1474_);
v___x_1476_ = lean_float_decLt(v___y_1472_, v___x_1475_);
v___y_1441_ = v___x_1476_;
goto v___jp_1440_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___boxed(lean_object* v_cls_1487_, lean_object* v_collapsed_1488_, lean_object* v_tag_1489_, lean_object* v_opts_1490_, lean_object* v_clsEnabled_1491_, lean_object* v_oldTraces_1492_, lean_object* v_msg_1493_, lean_object* v_resStartStop_1494_, lean_object* v___y_1495_, lean_object* v___y_1496_, lean_object* v___y_1497_){
_start:
{
uint8_t v_collapsed_boxed_1498_; uint8_t v_clsEnabled_boxed_1499_; lean_object* v_res_1500_; 
v_collapsed_boxed_1498_ = lean_unbox(v_collapsed_1488_);
v_clsEnabled_boxed_1499_ = lean_unbox(v_clsEnabled_1491_);
v_res_1500_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5(v_cls_1487_, v_collapsed_boxed_1498_, v_tag_1489_, v_opts_1490_, v_clsEnabled_boxed_1499_, v_oldTraces_1492_, v_msg_1493_, v_resStartStop_1494_, v___y_1495_, v___y_1496_);
lean_dec(v___y_1496_);
lean_dec_ref(v___y_1495_);
lean_dec_ref(v_opts_1490_);
return v_res_1500_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1(lean_object* v_cls_1504_, lean_object* v_msg_1505_, lean_object* v___y_1506_, lean_object* v___y_1507_){
_start:
{
lean_object* v_ref_1509_; lean_object* v___x_1510_; lean_object* v_a_1511_; lean_object* v___x_1513_; uint8_t v_isShared_1514_; uint8_t v_isSharedCheck_1555_; 
v_ref_1509_ = lean_ctor_get(v___y_1506_, 5);
v___x_1510_ = lp_aesop_Lean_addMessageContextPartial___at___00Lean_addTrace___at___00Aesop_Stats_trace_spec__1_spec__1(v_msg_1505_, v___y_1506_, v___y_1507_);
v_a_1511_ = lean_ctor_get(v___x_1510_, 0);
v_isSharedCheck_1555_ = !lean_is_exclusive(v___x_1510_);
if (v_isSharedCheck_1555_ == 0)
{
v___x_1513_ = v___x_1510_;
v_isShared_1514_ = v_isSharedCheck_1555_;
goto v_resetjp_1512_;
}
else
{
lean_inc(v_a_1511_);
lean_dec(v___x_1510_);
v___x_1513_ = lean_box(0);
v_isShared_1514_ = v_isSharedCheck_1555_;
goto v_resetjp_1512_;
}
v_resetjp_1512_:
{
lean_object* v___x_1515_; lean_object* v_traceState_1516_; lean_object* v_env_1517_; lean_object* v_nextMacroScope_1518_; lean_object* v_ngen_1519_; lean_object* v_auxDeclNGen_1520_; lean_object* v_cache_1521_; lean_object* v_messages_1522_; lean_object* v_infoState_1523_; lean_object* v_snapshotTasks_1524_; lean_object* v___x_1526_; uint8_t v_isShared_1527_; uint8_t v_isSharedCheck_1554_; 
v___x_1515_ = lean_st_ref_take(v___y_1507_);
v_traceState_1516_ = lean_ctor_get(v___x_1515_, 4);
v_env_1517_ = lean_ctor_get(v___x_1515_, 0);
v_nextMacroScope_1518_ = lean_ctor_get(v___x_1515_, 1);
v_ngen_1519_ = lean_ctor_get(v___x_1515_, 2);
v_auxDeclNGen_1520_ = lean_ctor_get(v___x_1515_, 3);
v_cache_1521_ = lean_ctor_get(v___x_1515_, 5);
v_messages_1522_ = lean_ctor_get(v___x_1515_, 6);
v_infoState_1523_ = lean_ctor_get(v___x_1515_, 7);
v_snapshotTasks_1524_ = lean_ctor_get(v___x_1515_, 8);
v_isSharedCheck_1554_ = !lean_is_exclusive(v___x_1515_);
if (v_isSharedCheck_1554_ == 0)
{
v___x_1526_ = v___x_1515_;
v_isShared_1527_ = v_isSharedCheck_1554_;
goto v_resetjp_1525_;
}
else
{
lean_inc(v_snapshotTasks_1524_);
lean_inc(v_infoState_1523_);
lean_inc(v_messages_1522_);
lean_inc(v_cache_1521_);
lean_inc(v_traceState_1516_);
lean_inc(v_auxDeclNGen_1520_);
lean_inc(v_ngen_1519_);
lean_inc(v_nextMacroScope_1518_);
lean_inc(v_env_1517_);
lean_dec(v___x_1515_);
v___x_1526_ = lean_box(0);
v_isShared_1527_ = v_isSharedCheck_1554_;
goto v_resetjp_1525_;
}
v_resetjp_1525_:
{
uint64_t v_tid_1528_; lean_object* v_traces_1529_; lean_object* v___x_1531_; uint8_t v_isShared_1532_; uint8_t v_isSharedCheck_1553_; 
v_tid_1528_ = lean_ctor_get_uint64(v_traceState_1516_, sizeof(void*)*1);
v_traces_1529_ = lean_ctor_get(v_traceState_1516_, 0);
v_isSharedCheck_1553_ = !lean_is_exclusive(v_traceState_1516_);
if (v_isSharedCheck_1553_ == 0)
{
v___x_1531_ = v_traceState_1516_;
v_isShared_1532_ = v_isSharedCheck_1553_;
goto v_resetjp_1530_;
}
else
{
lean_inc(v_traces_1529_);
lean_dec(v_traceState_1516_);
v___x_1531_ = lean_box(0);
v_isShared_1532_ = v_isSharedCheck_1553_;
goto v_resetjp_1530_;
}
v_resetjp_1530_:
{
lean_object* v___x_1533_; double v___x_1534_; uint8_t v___x_1535_; lean_object* v___x_1536_; lean_object* v___x_1537_; lean_object* v___x_1538_; lean_object* v___x_1539_; lean_object* v___x_1540_; lean_object* v___x_1541_; lean_object* v___x_1543_; 
v___x_1533_ = lean_box(0);
v___x_1534_ = lean_float_once(&lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__0, &lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__0_once, _init_lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5___closed__0);
v___x_1535_ = 0;
v___x_1536_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1___closed__0));
v___x_1537_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_1537_, 0, v_cls_1504_);
lean_ctor_set(v___x_1537_, 1, v___x_1533_);
lean_ctor_set(v___x_1537_, 2, v___x_1536_);
lean_ctor_set_float(v___x_1537_, sizeof(void*)*3, v___x_1534_);
lean_ctor_set_float(v___x_1537_, sizeof(void*)*3 + 8, v___x_1534_);
lean_ctor_set_uint8(v___x_1537_, sizeof(void*)*3 + 16, v___x_1535_);
v___x_1538_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1___closed__1));
v___x_1539_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_1539_, 0, v___x_1537_);
lean_ctor_set(v___x_1539_, 1, v_a_1511_);
lean_ctor_set(v___x_1539_, 2, v___x_1538_);
lean_inc(v_ref_1509_);
v___x_1540_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1540_, 0, v_ref_1509_);
lean_ctor_set(v___x_1540_, 1, v___x_1539_);
v___x_1541_ = l_Lean_PersistentArray_push___redArg(v_traces_1529_, v___x_1540_);
if (v_isShared_1532_ == 0)
{
lean_ctor_set(v___x_1531_, 0, v___x_1541_);
v___x_1543_ = v___x_1531_;
goto v_reusejp_1542_;
}
else
{
lean_object* v_reuseFailAlloc_1552_; 
v_reuseFailAlloc_1552_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_1552_, 0, v___x_1541_);
lean_ctor_set_uint64(v_reuseFailAlloc_1552_, sizeof(void*)*1, v_tid_1528_);
v___x_1543_ = v_reuseFailAlloc_1552_;
goto v_reusejp_1542_;
}
v_reusejp_1542_:
{
lean_object* v___x_1545_; 
if (v_isShared_1527_ == 0)
{
lean_ctor_set(v___x_1526_, 4, v___x_1543_);
v___x_1545_ = v___x_1526_;
goto v_reusejp_1544_;
}
else
{
lean_object* v_reuseFailAlloc_1551_; 
v_reuseFailAlloc_1551_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_1551_, 0, v_env_1517_);
lean_ctor_set(v_reuseFailAlloc_1551_, 1, v_nextMacroScope_1518_);
lean_ctor_set(v_reuseFailAlloc_1551_, 2, v_ngen_1519_);
lean_ctor_set(v_reuseFailAlloc_1551_, 3, v_auxDeclNGen_1520_);
lean_ctor_set(v_reuseFailAlloc_1551_, 4, v___x_1543_);
lean_ctor_set(v_reuseFailAlloc_1551_, 5, v_cache_1521_);
lean_ctor_set(v_reuseFailAlloc_1551_, 6, v_messages_1522_);
lean_ctor_set(v_reuseFailAlloc_1551_, 7, v_infoState_1523_);
lean_ctor_set(v_reuseFailAlloc_1551_, 8, v_snapshotTasks_1524_);
v___x_1545_ = v_reuseFailAlloc_1551_;
goto v_reusejp_1544_;
}
v_reusejp_1544_:
{
lean_object* v___x_1546_; lean_object* v___x_1547_; lean_object* v___x_1549_; 
v___x_1546_ = lean_st_ref_set(v___y_1507_, v___x_1545_);
v___x_1547_ = lean_box(0);
if (v_isShared_1514_ == 0)
{
lean_ctor_set(v___x_1513_, 0, v___x_1547_);
v___x_1549_ = v___x_1513_;
goto v_reusejp_1548_;
}
else
{
lean_object* v_reuseFailAlloc_1550_; 
v_reuseFailAlloc_1550_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1550_, 0, v___x_1547_);
v___x_1549_ = v_reuseFailAlloc_1550_;
goto v_reusejp_1548_;
}
v_reusejp_1548_:
{
return v___x_1549_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1___boxed(lean_object* v_cls_1556_, lean_object* v_msg_1557_, lean_object* v___y_1558_, lean_object* v___y_1559_, lean_object* v___y_1560_){
_start:
{
lean_object* v_res_1561_; 
v_res_1561_ = lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1(v_cls_1556_, v_msg_1557_, v___y_1558_, v___y_1559_);
lean_dec(v___y_1559_);
lean_dec_ref(v___y_1558_);
return v_res_1561_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__8(lean_object* v_as_1562_, size_t v_i_1563_, size_t v_stop_1564_, lean_object* v_b_1565_){
_start:
{
uint8_t v___x_1566_; 
v___x_1566_ = lean_usize_dec_eq(v_i_1563_, v_stop_1564_);
if (v___x_1566_ == 0)
{
lean_object* v___x_1567_; lean_object* v_elapsed_1568_; lean_object* v___x_1569_; size_t v___x_1570_; size_t v___x_1571_; 
v___x_1567_ = lean_array_uget_borrowed(v_as_1562_, v_i_1563_);
v_elapsed_1568_ = lean_ctor_get(v___x_1567_, 1);
v___x_1569_ = lean_nat_add(v_b_1565_, v_elapsed_1568_);
lean_dec(v_b_1565_);
v___x_1570_ = ((size_t)1ULL);
v___x_1571_ = lean_usize_add(v_i_1563_, v___x_1570_);
v_i_1563_ = v___x_1571_;
v_b_1565_ = v___x_1569_;
goto _start;
}
else
{
return v_b_1565_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__8___boxed(lean_object* v_as_1573_, lean_object* v_i_1574_, lean_object* v_stop_1575_, lean_object* v_b_1576_){
_start:
{
size_t v_i_boxed_1577_; size_t v_stop_boxed_1578_; lean_object* v_res_1579_; 
v_i_boxed_1577_ = lean_unbox_usize(v_i_1574_);
lean_dec(v_i_1574_);
v_stop_boxed_1578_ = lean_unbox_usize(v_stop_1575_);
lean_dec(v_stop_1575_);
v_res_1579_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__8(v_as_1573_, v_i_boxed_1577_, v_stop_boxed_1578_, v_b_1576_);
lean_dec_ref(v_as_1573_);
return v_res_1579_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__0(void){
_start:
{
lean_object* v___x_1580_; lean_object* v___x_1581_; 
v___x_1580_ = ((lean_object*)(lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__2));
v___x_1581_ = l_Lean_stringToMessageData(v___x_1580_);
return v___x_1581_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__1(void){
_start:
{
lean_object* v___x_1582_; lean_object* v___x_1583_; 
v___x_1582_ = ((lean_object*)(lp_aesop_Aesop_ScriptGenerated_toString___closed__2));
v___x_1583_ = l_Lean_stringToMessageData(v___x_1582_);
return v___x_1583_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__3(void){
_start:
{
lean_object* v___x_1585_; lean_object* v___x_1586_; 
v___x_1585_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__2));
v___x_1586_ = l_Lean_stringToMessageData(v___x_1585_);
return v___x_1586_;
}
}
static lean_object* _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__5(void){
_start:
{
lean_object* v___x_1588_; lean_object* v___x_1589_; 
v___x_1588_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__4));
v___x_1589_ = l_Lean_stringToMessageData(v___x_1588_);
return v___x_1589_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2(lean_object* v___x_1590_, uint8_t v_a_1591_, lean_object* v_as_1592_, size_t v_sz_1593_, size_t v_i_1594_, lean_object* v_b_1595_, lean_object* v___y_1596_, lean_object* v___y_1597_){
_start:
{
uint8_t v___x_1599_; 
v___x_1599_ = lean_usize_dec_lt(v_i_1594_, v_sz_1593_);
if (v___x_1599_ == 0)
{
lean_object* v___x_1600_; 
lean_dec(v___x_1590_);
v___x_1600_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1600_, 0, v_b_1595_);
return v___x_1600_;
}
else
{
lean_object* v_a_1601_; lean_object* v_snd_1602_; lean_object* v_fst_1603_; lean_object* v___x_1605_; uint8_t v_isShared_1606_; uint8_t v_isSharedCheck_1694_; 
v_a_1601_ = lean_array_uget(v_as_1592_, v_i_1594_);
v_snd_1602_ = lean_ctor_get(v_a_1601_, 1);
v_fst_1603_ = lean_ctor_get(v_a_1601_, 0);
v_isSharedCheck_1694_ = !lean_is_exclusive(v_a_1601_);
if (v_isSharedCheck_1694_ == 0)
{
v___x_1605_ = v_a_1601_;
v_isShared_1606_ = v_isSharedCheck_1694_;
goto v_resetjp_1604_;
}
else
{
lean_inc(v_snd_1602_);
lean_inc(v_fst_1603_);
lean_dec(v_a_1601_);
v___x_1605_ = lean_box(0);
v_isShared_1606_ = v_isSharedCheck_1694_;
goto v_resetjp_1604_;
}
v_resetjp_1604_:
{
lean_object* v_numSuccessful_1607_; lean_object* v_numFailed_1608_; lean_object* v_elapsedSuccessful_1609_; lean_object* v_elapsedFailed_1610_; lean_object* v___x_1611_; lean_object* v___x_1612_; lean_object* v___x_1613_; lean_object* v___x_1614_; lean_object* v___x_1615_; lean_object* v___x_1616_; lean_object* v___x_1618_; 
v_numSuccessful_1607_ = lean_ctor_get(v_snd_1602_, 0);
lean_inc(v_numSuccessful_1607_);
v_numFailed_1608_ = lean_ctor_get(v_snd_1602_, 1);
lean_inc(v_numFailed_1608_);
v_elapsedSuccessful_1609_ = lean_ctor_get(v_snd_1602_, 2);
lean_inc(v_elapsedSuccessful_1609_);
v_elapsedFailed_1610_ = lean_ctor_get(v_snd_1602_, 3);
lean_inc(v_elapsedFailed_1610_);
lean_dec(v_snd_1602_);
v___x_1611_ = lean_box(0);
v___x_1612_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__0, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__0_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__0);
v___x_1613_ = lean_nat_add(v_numSuccessful_1607_, v_numFailed_1608_);
v___x_1614_ = l_Nat_reprFast(v___x_1613_);
v___x_1615_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1615_, 0, v___x_1614_);
v___x_1616_ = l_Lean_MessageData_ofFormat(v___x_1615_);
if (v_isShared_1606_ == 0)
{
lean_ctor_set_tag(v___x_1605_, 7);
lean_ctor_set(v___x_1605_, 1, v___x_1616_);
lean_ctor_set(v___x_1605_, 0, v___x_1612_);
v___x_1618_ = v___x_1605_;
goto v_reusejp_1617_;
}
else
{
lean_object* v_reuseFailAlloc_1693_; 
v_reuseFailAlloc_1693_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1693_, 0, v___x_1612_);
lean_ctor_set(v_reuseFailAlloc_1693_, 1, v___x_1616_);
v___x_1618_ = v_reuseFailAlloc_1693_;
goto v_reusejp_1617_;
}
v_reusejp_1617_:
{
lean_object* v___x_1619_; lean_object* v___x_1620_; lean_object* v___x_1621_; lean_object* v___x_1622_; lean_object* v___x_1623_; lean_object* v___x_1624_; lean_object* v___x_1625_; lean_object* v___x_1626_; lean_object* v___x_1627_; lean_object* v___x_1628_; lean_object* v___x_1629_; lean_object* v___x_1630_; lean_object* v___x_1631_; lean_object* v___x_1632_; lean_object* v___x_1633_; lean_object* v___x_1634_; lean_object* v___x_1635_; lean_object* v___x_1636_; lean_object* v___x_1637_; lean_object* v___x_1638_; lean_object* v___x_1639_; lean_object* v___x_1640_; lean_object* v___x_1641_; lean_object* v___x_1642_; lean_object* v___x_1643_; lean_object* v___x_1644_; lean_object* v___x_1645_; lean_object* v___y_1647_; 
v___x_1619_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__1, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__1_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__1);
v___x_1620_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1620_, 0, v___x_1618_);
lean_ctor_set(v___x_1620_, 1, v___x_1619_);
v___x_1621_ = lean_nat_add(v_elapsedSuccessful_1609_, v_elapsedFailed_1610_);
v___x_1622_ = lp_aesop_Aesop_Nanos_printAsMillis(v___x_1621_);
v___x_1623_ = l_Lean_stringToMessageData(v___x_1622_);
v___x_1624_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1624_, 0, v___x_1620_);
lean_ctor_set(v___x_1624_, 1, v___x_1623_);
v___x_1625_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__3, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__3_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__3);
v___x_1626_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1626_, 0, v___x_1624_);
lean_ctor_set(v___x_1626_, 1, v___x_1625_);
v___x_1627_ = l_Nat_reprFast(v_numSuccessful_1607_);
v___x_1628_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1628_, 0, v___x_1627_);
v___x_1629_ = l_Lean_MessageData_ofFormat(v___x_1628_);
v___x_1630_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1630_, 0, v___x_1626_);
lean_ctor_set(v___x_1630_, 1, v___x_1629_);
v___x_1631_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1631_, 0, v___x_1630_);
lean_ctor_set(v___x_1631_, 1, v___x_1619_);
v___x_1632_ = lp_aesop_Aesop_Nanos_printAsMillis(v_elapsedSuccessful_1609_);
v___x_1633_ = l_Lean_stringToMessageData(v___x_1632_);
v___x_1634_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1634_, 0, v___x_1631_);
lean_ctor_set(v___x_1634_, 1, v___x_1633_);
v___x_1635_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1635_, 0, v___x_1634_);
lean_ctor_set(v___x_1635_, 1, v___x_1625_);
v___x_1636_ = l_Nat_reprFast(v_numFailed_1608_);
v___x_1637_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1637_, 0, v___x_1636_);
v___x_1638_ = l_Lean_MessageData_ofFormat(v___x_1637_);
v___x_1639_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1639_, 0, v___x_1635_);
lean_ctor_set(v___x_1639_, 1, v___x_1638_);
v___x_1640_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1640_, 0, v___x_1639_);
lean_ctor_set(v___x_1640_, 1, v___x_1619_);
v___x_1641_ = lp_aesop_Aesop_Nanos_printAsMillis(v_elapsedFailed_1610_);
v___x_1642_ = l_Lean_stringToMessageData(v___x_1641_);
v___x_1643_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1643_, 0, v___x_1640_);
lean_ctor_set(v___x_1643_, 1, v___x_1642_);
v___x_1644_ = lean_obj_once(&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__5, &lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__5_once, _init_lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___closed__5);
v___x_1645_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1645_, 0, v___x_1643_);
lean_ctor_set(v___x_1645_, 1, v___x_1644_);
switch(lean_obj_tag(v_fst_1603_))
{
case 0:
{
lean_object* v_n_1655_; lean_object* v_name_1656_; uint8_t v_builder_1657_; uint8_t v_phase_1658_; uint8_t v_scope_1659_; lean_object* v___y_1661_; lean_object* v___y_1662_; lean_object* v___y_1663_; lean_object* v___y_1669_; lean_object* v___y_1670_; lean_object* v___y_1671_; lean_object* v___y_1677_; 
v_n_1655_ = lean_ctor_get(v_fst_1603_, 0);
lean_inc_ref(v_n_1655_);
lean_dec_ref_known(v_fst_1603_, 1);
v_name_1656_ = lean_ctor_get(v_n_1655_, 0);
lean_inc(v_name_1656_);
v_builder_1657_ = lean_ctor_get_uint8(v_n_1655_, sizeof(void*)*1 + 8);
v_phase_1658_ = lean_ctor_get_uint8(v_n_1655_, sizeof(void*)*1 + 9);
v_scope_1659_ = lean_ctor_get_uint8(v_n_1655_, sizeof(void*)*1 + 10);
lean_dec_ref(v_n_1655_);
switch(v_phase_1658_)
{
case 0:
{
lean_object* v___x_1688_; 
v___x_1688_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__13));
v___y_1677_ = v___x_1688_;
goto v___jp_1676_;
}
case 1:
{
lean_object* v___x_1689_; 
v___x_1689_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__14));
v___y_1677_ = v___x_1689_;
goto v___jp_1676_;
}
default: 
{
lean_object* v___x_1690_; 
v___x_1690_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__15));
v___y_1677_ = v___x_1690_;
goto v___jp_1676_;
}
}
v___jp_1660_:
{
lean_object* v___x_1664_; lean_object* v___x_1665_; lean_object* v___x_1666_; lean_object* v___x_1667_; 
v___x_1664_ = lean_string_append(v___y_1662_, v___y_1663_);
v___x_1665_ = lean_string_append(v___x_1664_, v___y_1661_);
v___x_1666_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_1656_, v_a_1591_);
v___x_1667_ = lean_string_append(v___x_1665_, v___x_1666_);
lean_dec_ref(v___x_1666_);
v___y_1647_ = v___x_1667_;
goto v___jp_1646_;
}
v___jp_1668_:
{
lean_object* v___x_1672_; lean_object* v___x_1673_; 
v___x_1672_ = lean_string_append(v___y_1670_, v___y_1671_);
v___x_1673_ = lean_string_append(v___x_1672_, v___y_1669_);
if (v_scope_1659_ == 0)
{
lean_object* v___x_1674_; 
v___x_1674_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__2));
v___y_1661_ = v___y_1669_;
v___y_1662_ = v___x_1673_;
v___y_1663_ = v___x_1674_;
goto v___jp_1660_;
}
else
{
lean_object* v___x_1675_; 
v___x_1675_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__3));
v___y_1661_ = v___y_1669_;
v___y_1662_ = v___x_1673_;
v___y_1663_ = v___x_1675_;
goto v___jp_1660_;
}
}
v___jp_1676_:
{
lean_object* v___x_1678_; lean_object* v___x_1679_; 
v___x_1678_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__4));
lean_inc_ref(v___y_1677_);
v___x_1679_ = lean_string_append(v___y_1677_, v___x_1678_);
switch(v_builder_1657_)
{
case 0:
{
lean_object* v___x_1680_; 
v___x_1680_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__5));
v___y_1669_ = v___x_1678_;
v___y_1670_ = v___x_1679_;
v___y_1671_ = v___x_1680_;
goto v___jp_1668_;
}
case 1:
{
lean_object* v___x_1681_; 
v___x_1681_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__6));
v___y_1669_ = v___x_1678_;
v___y_1670_ = v___x_1679_;
v___y_1671_ = v___x_1681_;
goto v___jp_1668_;
}
case 2:
{
lean_object* v___x_1682_; 
v___x_1682_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__7));
v___y_1669_ = v___x_1678_;
v___y_1670_ = v___x_1679_;
v___y_1671_ = v___x_1682_;
goto v___jp_1668_;
}
case 3:
{
lean_object* v___x_1683_; 
v___x_1683_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__8));
v___y_1669_ = v___x_1678_;
v___y_1670_ = v___x_1679_;
v___y_1671_ = v___x_1683_;
goto v___jp_1668_;
}
case 4:
{
lean_object* v___x_1684_; 
v___x_1684_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__9));
v___y_1669_ = v___x_1678_;
v___y_1670_ = v___x_1679_;
v___y_1671_ = v___x_1684_;
goto v___jp_1668_;
}
case 5:
{
lean_object* v___x_1685_; 
v___x_1685_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__10));
v___y_1669_ = v___x_1678_;
v___y_1670_ = v___x_1679_;
v___y_1671_ = v___x_1685_;
goto v___jp_1668_;
}
case 6:
{
lean_object* v___x_1686_; 
v___x_1686_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__11));
v___y_1669_ = v___x_1678_;
v___y_1670_ = v___x_1679_;
v___y_1671_ = v___x_1686_;
goto v___jp_1668_;
}
default: 
{
lean_object* v___x_1687_; 
v___x_1687_ = ((lean_object*)(lp_aesop_Aesop_instToJsonForwardRuleStateStats_toJson___closed__12));
v___y_1669_ = v___x_1678_;
v___y_1670_ = v___x_1679_;
v___y_1671_ = v___x_1687_;
goto v___jp_1668_;
}
}
}
}
case 1:
{
lean_object* v___x_1691_; 
v___x_1691_ = ((lean_object*)(lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__4));
v___y_1647_ = v___x_1691_;
goto v___jp_1646_;
}
default: 
{
lean_object* v___x_1692_; 
v___x_1692_ = ((lean_object*)(lp_aesop_Aesop_RuleStats_instToString___lam__0___closed__5));
v___y_1647_ = v___x_1692_;
goto v___jp_1646_;
}
}
v___jp_1646_:
{
lean_object* v___x_1648_; lean_object* v___x_1649_; lean_object* v___x_1650_; lean_object* v___x_1651_; 
v___x_1648_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1648_, 0, v___y_1647_);
v___x_1649_ = l_Lean_MessageData_ofFormat(v___x_1648_);
v___x_1650_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_1650_, 0, v___x_1645_);
lean_ctor_set(v___x_1650_, 1, v___x_1649_);
lean_inc(v___x_1590_);
v___x_1651_ = lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1(v___x_1590_, v___x_1650_, v___y_1596_, v___y_1597_);
if (lean_obj_tag(v___x_1651_) == 0)
{
size_t v___x_1652_; size_t v___x_1653_; 
lean_dec_ref_known(v___x_1651_, 1);
v___x_1652_ = ((size_t)1ULL);
v___x_1653_ = lean_usize_add(v_i_1594_, v___x_1652_);
v_i_1594_ = v___x_1653_;
v_b_1595_ = v___x_1611_;
goto _start;
}
else
{
lean_dec(v___x_1590_);
return v___x_1651_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2___boxed(lean_object* v___x_1695_, lean_object* v_a_1696_, lean_object* v_as_1697_, lean_object* v_sz_1698_, lean_object* v_i_1699_, lean_object* v_b_1700_, lean_object* v___y_1701_, lean_object* v___y_1702_, lean_object* v___y_1703_){
_start:
{
uint8_t v_a_49984__boxed_1704_; size_t v_sz_boxed_1705_; size_t v_i_boxed_1706_; lean_object* v_res_1707_; 
v_a_49984__boxed_1704_ = lean_unbox(v_a_1696_);
v_sz_boxed_1705_ = lean_unbox_usize(v_sz_1698_);
lean_dec(v_sz_1698_);
v_i_boxed_1706_ = lean_unbox_usize(v_i_1699_);
lean_dec(v_i_1699_);
v_res_1707_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2(v___x_1695_, v_a_49984__boxed_1704_, v_as_1697_, v_sz_boxed_1705_, v_i_boxed_1706_, v_b_1700_, v___y_1701_, v___y_1702_);
lean_dec(v___y_1702_);
lean_dec_ref(v___y_1701_);
lean_dec_ref(v_as_1697_);
return v_res_1707_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Stats_trace_spec__0___redArg(lean_object* v_opt_1708_, lean_object* v___y_1709_){
_start:
{
lean_object* v_options_1711_; lean_object* v_option_1712_; uint8_t v___x_1713_; lean_object* v___x_1714_; lean_object* v___x_1715_; 
v_options_1711_ = lean_ctor_get(v___y_1709_, 2);
v_option_1712_ = lean_ctor_get(v_opt_1708_, 1);
v___x_1713_ = lp_aesop_Lean_Option_get___at___00Aesop_Stats_trace_spec__4(v_options_1711_, v_option_1712_);
v___x_1714_ = lean_box(v___x_1713_);
v___x_1715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1715_, 0, v___x_1714_);
return v___x_1715_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Stats_trace_spec__0___redArg___boxed(lean_object* v_opt_1716_, lean_object* v___y_1717_, lean_object* v___y_1718_){
_start:
{
lean_object* v_res_1719_; 
v_res_1719_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Stats_trace_spec__0___redArg(v_opt_1716_, v___y_1717_);
lean_dec_ref(v___y_1717_);
lean_dec_ref(v_opt_1716_);
return v_res_1719_;
}
}
static double _init_lp_aesop_Aesop_Stats_trace___closed__0(void){
_start:
{
lean_object* v___x_1720_; double v___x_1721_; 
v___x_1720_ = lean_unsigned_to_nat(1000000000u);
v___x_1721_ = lean_float_of_nat(v___x_1720_);
return v___x_1721_;
}
}
static lean_object* _init_lp_aesop_Aesop_Stats_trace___closed__4(void){
_start:
{
lean_object* v___x_1726_; lean_object* v___x_1727_; 
v___x_1726_ = ((lean_object*)(lp_aesop_Aesop_Stats_trace___closed__3));
v___x_1727_ = l_Lean_stringToMessageData(v___x_1726_);
return v___x_1727_;
}
}
static lean_object* _init_lp_aesop_Aesop_Stats_trace___closed__5(void){
_start:
{
lean_object* v___x_1728_; lean_object* v___x_1729_; lean_object* v___x_1730_; 
v___x_1728_ = lean_box(0);
v___x_1729_ = lean_unsigned_to_nat(16u);
v___x_1730_ = lean_mk_array(v___x_1729_, v___x_1728_);
return v___x_1730_;
}
}
static lean_object* _init_lp_aesop_Aesop_Stats_trace___closed__6(void){
_start:
{
lean_object* v___x_1731_; lean_object* v___x_1732_; lean_object* v___x_1733_; 
v___x_1731_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__5, &lp_aesop_Aesop_Stats_trace___closed__5_once, _init_lp_aesop_Aesop_Stats_trace___closed__5);
v___x_1732_ = lean_unsigned_to_nat(0u);
v___x_1733_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1733_, 0, v___x_1732_);
lean_ctor_set(v___x_1733_, 1, v___x_1731_);
return v___x_1733_;
}
}
static lean_object* _init_lp_aesop_Aesop_Stats_trace___closed__8(void){
_start:
{
lean_object* v___x_1735_; lean_object* v___x_1736_; 
v___x_1735_ = ((lean_object*)(lp_aesop_Aesop_Stats_trace___closed__7));
v___x_1736_ = l_Lean_stringToMessageData(v___x_1735_);
return v___x_1736_;
}
}
static lean_object* _init_lp_aesop_Aesop_Stats_trace___closed__10(void){
_start:
{
lean_object* v___x_1738_; lean_object* v___x_1739_; 
v___x_1738_ = ((lean_object*)(lp_aesop_Aesop_Stats_trace___closed__9));
v___x_1739_ = l_Lean_stringToMessageData(v___x_1738_);
return v___x_1739_;
}
}
static lean_object* _init_lp_aesop_Aesop_Stats_trace___closed__12(void){
_start:
{
lean_object* v___x_1741_; lean_object* v___x_1742_; 
v___x_1741_ = ((lean_object*)(lp_aesop_Aesop_Stats_trace___closed__11));
v___x_1742_ = l_Lean_stringToMessageData(v___x_1741_);
return v___x_1742_;
}
}
static lean_object* _init_lp_aesop_Aesop_Stats_trace___closed__14(void){
_start:
{
lean_object* v___x_1744_; lean_object* v___x_1745_; 
v___x_1744_ = ((lean_object*)(lp_aesop_Aesop_Stats_trace___closed__13));
v___x_1745_ = l_Lean_stringToMessageData(v___x_1744_);
return v___x_1745_;
}
}
static lean_object* _init_lp_aesop_Aesop_Stats_trace___closed__16(void){
_start:
{
lean_object* v___x_1747_; lean_object* v___x_1748_; 
v___x_1747_ = ((lean_object*)(lp_aesop_Aesop_Stats_trace___closed__15));
v___x_1748_ = l_Lean_stringToMessageData(v___x_1747_);
return v___x_1748_;
}
}
static lean_object* _init_lp_aesop_Aesop_Stats_trace___closed__18(void){
_start:
{
lean_object* v___x_1750_; lean_object* v___x_1751_; 
v___x_1750_ = ((lean_object*)(lp_aesop_Aesop_Stats_trace___closed__17));
v___x_1751_ = l_Lean_stringToMessageData(v___x_1750_);
return v___x_1751_;
}
}
static lean_object* _init_lp_aesop_Aesop_Stats_trace___closed__20(void){
_start:
{
lean_object* v___x_1753_; lean_object* v___x_1754_; 
v___x_1753_ = ((lean_object*)(lp_aesop_Aesop_Stats_trace___closed__19));
v___x_1754_ = l_Lean_stringToMessageData(v___x_1753_);
return v___x_1754_;
}
}
static lean_object* _init_lp_aesop_Aesop_Stats_trace___closed__22(void){
_start:
{
lean_object* v___x_1756_; lean_object* v___x_1757_; 
v___x_1756_ = ((lean_object*)(lp_aesop_Aesop_Stats_trace___closed__21));
v___x_1757_ = l_Lean_stringToMessageData(v___x_1756_);
return v___x_1757_;
}
}
static lean_object* _init_lp_aesop_Aesop_Stats_trace___closed__24(void){
_start:
{
lean_object* v___x_1759_; lean_object* v___x_1760_; 
v___x_1759_ = ((lean_object*)(lp_aesop_Aesop_Stats_trace___closed__23));
v___x_1760_ = l_Lean_stringToMessageData(v___x_1759_);
return v___x_1760_;
}
}
static lean_object* _init_lp_aesop_Aesop_Stats_trace___closed__27(void){
_start:
{
lean_object* v___x_1764_; lean_object* v___x_1765_; 
v___x_1764_ = ((lean_object*)(lp_aesop_Aesop_Stats_trace___closed__26));
v___x_1765_ = l_Lean_MessageData_ofFormat(v___x_1764_);
return v___x_1765_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stats_trace(lean_object* v_p_1766_, lean_object* v_opt_1767_, lean_object* v_a_1768_, lean_object* v_a_1769_){
_start:
{
lean_object* v___y_1772_; lean_object* v___y_1773_; lean_object* v___y_1774_; lean_object* v___y_1775_; uint8_t v___y_1776_; lean_object* v___y_1777_; uint8_t v___y_1778_; lean_object* v___y_1779_; lean_object* v_a_1780_; lean_object* v___y_1790_; lean_object* v___y_1791_; lean_object* v___y_1792_; lean_object* v___y_1793_; uint8_t v___y_1794_; uint8_t v___y_1795_; lean_object* v___y_1796_; lean_object* v___y_1797_; lean_object* v_a_1798_; lean_object* v___y_1801_; lean_object* v___y_1802_; lean_object* v___y_1803_; uint8_t v___y_1804_; lean_object* v___y_1805_; lean_object* v___y_1806_; uint8_t v___y_1807_; lean_object* v___y_1808_; lean_object* v_a_1809_; lean_object* v___y_1822_; lean_object* v___y_1823_; lean_object* v___y_1824_; lean_object* v___y_1825_; uint8_t v___y_1826_; uint8_t v___y_1827_; lean_object* v___y_1828_; lean_object* v___y_1829_; lean_object* v_a_1830_; lean_object* v___y_1833_; lean_object* v___y_1834_; lean_object* v___y_1835_; lean_object* v___y_1836_; size_t v___y_1837_; size_t v___y_1838_; lean_object* v___y_1839_; uint8_t v___y_1840_; uint8_t v___y_1841_; uint8_t v___y_1842_; lean_object* v___y_1843_; lean_object* v___y_1871_; lean_object* v___y_1872_; lean_object* v___y_1873_; uint8_t v___y_1874_; uint8_t v___y_1875_; lean_object* v___y_1876_; lean_object* v___y_1877_; lean_object* v___y_1878_; lean_object* v___x_1895_; lean_object* v_a_1896_; lean_object* v___x_1898_; uint8_t v_isShared_1899_; uint8_t v_isSharedCheck_2523_; 
v___x_1895_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Stats_trace_spec__0___redArg(v_opt_1767_, v_a_1768_);
v_a_1896_ = lean_ctor_get(v___x_1895_, 0);
v_isSharedCheck_2523_ = !lean_is_exclusive(v___x_1895_);
if (v_isSharedCheck_2523_ == 0)
{
v___x_1898_ = v___x_1895_;
v_isShared_1899_ = v_isSharedCheck_2523_;
goto v_resetjp_1897_;
}
else
{
lean_inc(v_a_1896_);
lean_dec(v___x_1895_);
v___x_1898_ = lean_box(0);
v_isShared_1899_ = v_isSharedCheck_2523_;
goto v_resetjp_1897_;
}
v___jp_1771_:
{
lean_object* v___x_1781_; double v___x_1782_; double v___x_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; lean_object* v___x_1786_; lean_object* v___x_1787_; lean_object* v___x_1788_; 
v___x_1781_ = lean_io_get_num_heartbeats();
v___x_1782_ = lean_float_of_nat(v___y_1774_);
v___x_1783_ = lean_float_of_nat(v___x_1781_);
v___x_1784_ = lean_box_float(v___x_1782_);
v___x_1785_ = lean_box_float(v___x_1783_);
v___x_1786_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1786_, 0, v___x_1784_);
lean_ctor_set(v___x_1786_, 1, v___x_1785_);
v___x_1787_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1787_, 0, v_a_1780_);
lean_ctor_set(v___x_1787_, 1, v___x_1786_);
lean_inc_ref(v___y_1772_);
v___x_1788_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5(v___y_1773_, v___y_1778_, v___y_1772_, v___y_1775_, v___y_1776_, v___y_1777_, v___y_1779_, v___x_1787_, v_a_1768_, v_a_1769_);
return v___x_1788_;
}
v___jp_1789_:
{
lean_object* v___x_1799_; 
v___x_1799_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1799_, 0, v_a_1798_);
v___y_1772_ = v___y_1790_;
v___y_1773_ = v___y_1792_;
v___y_1774_ = v___y_1791_;
v___y_1775_ = v___y_1793_;
v___y_1776_ = v___y_1794_;
v___y_1777_ = v___y_1796_;
v___y_1778_ = v___y_1795_;
v___y_1779_ = v___y_1797_;
v_a_1780_ = v___x_1799_;
goto v___jp_1771_;
}
v___jp_1800_:
{
lean_object* v___x_1810_; double v___x_1811_; double v___x_1812_; double v___x_1813_; double v___x_1814_; double v___x_1815_; lean_object* v___x_1816_; lean_object* v___x_1817_; lean_object* v___x_1818_; lean_object* v___x_1819_; lean_object* v___x_1820_; 
v___x_1810_ = lean_io_mono_nanos_now();
v___x_1811_ = lean_float_of_nat(v___y_1805_);
v___x_1812_ = lean_float_once(&lp_aesop_Aesop_Stats_trace___closed__0, &lp_aesop_Aesop_Stats_trace___closed__0_once, _init_lp_aesop_Aesop_Stats_trace___closed__0);
v___x_1813_ = lean_float_div(v___x_1811_, v___x_1812_);
v___x_1814_ = lean_float_of_nat(v___x_1810_);
v___x_1815_ = lean_float_div(v___x_1814_, v___x_1812_);
v___x_1816_ = lean_box_float(v___x_1813_);
v___x_1817_ = lean_box_float(v___x_1815_);
v___x_1818_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1818_, 0, v___x_1816_);
lean_ctor_set(v___x_1818_, 1, v___x_1817_);
v___x_1819_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1819_, 0, v_a_1809_);
lean_ctor_set(v___x_1819_, 1, v___x_1818_);
lean_inc_ref(v___y_1801_);
v___x_1820_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5(v___y_1802_, v___y_1807_, v___y_1801_, v___y_1803_, v___y_1804_, v___y_1806_, v___y_1808_, v___x_1819_, v_a_1768_, v_a_1769_);
return v___x_1820_;
}
v___jp_1821_:
{
lean_object* v___x_1831_; 
v___x_1831_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1831_, 0, v_a_1830_);
v___y_1801_ = v___y_1822_;
v___y_1802_ = v___y_1823_;
v___y_1803_ = v___y_1824_;
v___y_1804_ = v___y_1826_;
v___y_1805_ = v___y_1825_;
v___y_1806_ = v___y_1828_;
v___y_1807_ = v___y_1827_;
v___y_1808_ = v___y_1829_;
v_a_1809_ = v___x_1831_;
goto v___jp_1800_;
}
v___jp_1832_:
{
lean_object* v___x_1844_; lean_object* v_a_1845_; lean_object* v___x_1846_; uint8_t v___x_1847_; 
v___x_1844_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg(v_a_1769_);
v_a_1845_ = lean_ctor_get(v___x_1844_, 0);
lean_inc(v_a_1845_);
lean_dec_ref(v___x_1844_);
v___x_1846_ = l_Lean_trace_profiler_useHeartbeats;
v___x_1847_ = lp_aesop_Lean_Option_get___at___00Aesop_Stats_trace_spec__4(v___y_1836_, v___x_1846_);
if (v___x_1847_ == 0)
{
lean_object* v___x_1848_; lean_object* v___x_1849_; 
v___x_1848_ = lean_io_mono_nanos_now();
lean_inc(v___y_1835_);
v___x_1849_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2(v___y_1835_, v___y_1841_, v___y_1839_, v___y_1838_, v___y_1837_, v___y_1833_, v_a_1768_, v_a_1769_);
lean_dec_ref(v___y_1839_);
if (lean_obj_tag(v___x_1849_) == 0)
{
lean_dec_ref_known(v___x_1849_, 1);
v___y_1822_ = v___y_1834_;
v___y_1823_ = v___y_1835_;
v___y_1824_ = v___y_1836_;
v___y_1825_ = v___x_1848_;
v___y_1826_ = v___y_1840_;
v___y_1827_ = v___y_1842_;
v___y_1828_ = v_a_1845_;
v___y_1829_ = v___y_1843_;
v_a_1830_ = v___y_1833_;
goto v___jp_1821_;
}
else
{
if (lean_obj_tag(v___x_1849_) == 0)
{
lean_object* v_a_1850_; 
v_a_1850_ = lean_ctor_get(v___x_1849_, 0);
lean_inc(v_a_1850_);
lean_dec_ref_known(v___x_1849_, 1);
v___y_1822_ = v___y_1834_;
v___y_1823_ = v___y_1835_;
v___y_1824_ = v___y_1836_;
v___y_1825_ = v___x_1848_;
v___y_1826_ = v___y_1840_;
v___y_1827_ = v___y_1842_;
v___y_1828_ = v_a_1845_;
v___y_1829_ = v___y_1843_;
v_a_1830_ = v_a_1850_;
goto v___jp_1821_;
}
else
{
lean_object* v_a_1851_; lean_object* v___x_1853_; uint8_t v_isShared_1854_; uint8_t v_isSharedCheck_1858_; 
v_a_1851_ = lean_ctor_get(v___x_1849_, 0);
v_isSharedCheck_1858_ = !lean_is_exclusive(v___x_1849_);
if (v_isSharedCheck_1858_ == 0)
{
v___x_1853_ = v___x_1849_;
v_isShared_1854_ = v_isSharedCheck_1858_;
goto v_resetjp_1852_;
}
else
{
lean_inc(v_a_1851_);
lean_dec(v___x_1849_);
v___x_1853_ = lean_box(0);
v_isShared_1854_ = v_isSharedCheck_1858_;
goto v_resetjp_1852_;
}
v_resetjp_1852_:
{
lean_object* v___x_1856_; 
if (v_isShared_1854_ == 0)
{
lean_ctor_set_tag(v___x_1853_, 0);
v___x_1856_ = v___x_1853_;
goto v_reusejp_1855_;
}
else
{
lean_object* v_reuseFailAlloc_1857_; 
v_reuseFailAlloc_1857_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1857_, 0, v_a_1851_);
v___x_1856_ = v_reuseFailAlloc_1857_;
goto v_reusejp_1855_;
}
v_reusejp_1855_:
{
v___y_1801_ = v___y_1834_;
v___y_1802_ = v___y_1835_;
v___y_1803_ = v___y_1836_;
v___y_1804_ = v___y_1840_;
v___y_1805_ = v___x_1848_;
v___y_1806_ = v_a_1845_;
v___y_1807_ = v___y_1842_;
v___y_1808_ = v___y_1843_;
v_a_1809_ = v___x_1856_;
goto v___jp_1800_;
}
}
}
}
}
else
{
lean_object* v___x_1859_; lean_object* v___x_1860_; 
v___x_1859_ = lean_io_get_num_heartbeats();
lean_inc(v___y_1835_);
v___x_1860_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2(v___y_1835_, v___y_1841_, v___y_1839_, v___y_1838_, v___y_1837_, v___y_1833_, v_a_1768_, v_a_1769_);
lean_dec_ref(v___y_1839_);
if (lean_obj_tag(v___x_1860_) == 0)
{
lean_dec_ref_known(v___x_1860_, 1);
v___y_1790_ = v___y_1834_;
v___y_1791_ = v___x_1859_;
v___y_1792_ = v___y_1835_;
v___y_1793_ = v___y_1836_;
v___y_1794_ = v___y_1840_;
v___y_1795_ = v___y_1842_;
v___y_1796_ = v_a_1845_;
v___y_1797_ = v___y_1843_;
v_a_1798_ = v___y_1833_;
goto v___jp_1789_;
}
else
{
if (lean_obj_tag(v___x_1860_) == 0)
{
lean_object* v_a_1861_; 
v_a_1861_ = lean_ctor_get(v___x_1860_, 0);
lean_inc(v_a_1861_);
lean_dec_ref_known(v___x_1860_, 1);
v___y_1790_ = v___y_1834_;
v___y_1791_ = v___x_1859_;
v___y_1792_ = v___y_1835_;
v___y_1793_ = v___y_1836_;
v___y_1794_ = v___y_1840_;
v___y_1795_ = v___y_1842_;
v___y_1796_ = v_a_1845_;
v___y_1797_ = v___y_1843_;
v_a_1798_ = v_a_1861_;
goto v___jp_1789_;
}
else
{
lean_object* v_a_1862_; lean_object* v___x_1864_; uint8_t v_isShared_1865_; uint8_t v_isSharedCheck_1869_; 
v_a_1862_ = lean_ctor_get(v___x_1860_, 0);
v_isSharedCheck_1869_ = !lean_is_exclusive(v___x_1860_);
if (v_isSharedCheck_1869_ == 0)
{
v___x_1864_ = v___x_1860_;
v_isShared_1865_ = v_isSharedCheck_1869_;
goto v_resetjp_1863_;
}
else
{
lean_inc(v_a_1862_);
lean_dec(v___x_1860_);
v___x_1864_ = lean_box(0);
v_isShared_1865_ = v_isSharedCheck_1869_;
goto v_resetjp_1863_;
}
v_resetjp_1863_:
{
lean_object* v___x_1867_; 
if (v_isShared_1865_ == 0)
{
lean_ctor_set_tag(v___x_1864_, 0);
v___x_1867_ = v___x_1864_;
goto v_reusejp_1866_;
}
else
{
lean_object* v_reuseFailAlloc_1868_; 
v_reuseFailAlloc_1868_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1868_, 0, v_a_1862_);
v___x_1867_ = v_reuseFailAlloc_1868_;
goto v_reusejp_1866_;
}
v_reusejp_1866_:
{
v___y_1772_ = v___y_1834_;
v___y_1773_ = v___y_1835_;
v___y_1774_ = v___x_1859_;
v___y_1775_ = v___y_1836_;
v___y_1776_ = v___y_1840_;
v___y_1777_ = v_a_1845_;
v___y_1778_ = v___y_1842_;
v___y_1779_ = v___y_1843_;
v_a_1780_ = v___x_1867_;
goto v___jp_1771_;
}
}
}
}
}
}
v___jp_1870_:
{
lean_object* v___x_1879_; lean_object* v___x_1880_; size_t v_sz_1881_; size_t v___x_1882_; lean_object* v___x_1883_; lean_object* v___x_1884_; uint8_t v___x_1885_; 
v___x_1879_ = lp_aesop_Aesop_sortRuleStatsTotals(v___y_1878_);
v___x_1880_ = lean_box(0);
v_sz_1881_ = lean_array_size(v___x_1879_);
v___x_1882_ = ((size_t)0ULL);
v___x_1883_ = ((lean_object*)(lp_aesop_Aesop_Stats_trace___closed__2));
lean_inc(v___y_1872_);
v___x_1884_ = l_Lean_Name_append(v___x_1883_, v___y_1872_);
v___x_1885_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v___y_1877_, v___y_1873_, v___x_1884_);
lean_dec(v___x_1884_);
if (v___x_1885_ == 0)
{
if (v___y_1875_ == 0)
{
lean_object* v___x_1886_; 
lean_dec_ref(v___y_1876_);
v___x_1886_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2(v___y_1872_, v___y_1874_, v___x_1879_, v_sz_1881_, v___x_1882_, v___x_1880_, v_a_1768_, v_a_1769_);
lean_dec_ref(v___x_1879_);
if (lean_obj_tag(v___x_1886_) == 0)
{
lean_object* v___x_1888_; uint8_t v_isShared_1889_; uint8_t v_isSharedCheck_1893_; 
v_isSharedCheck_1893_ = !lean_is_exclusive(v___x_1886_);
if (v_isSharedCheck_1893_ == 0)
{
lean_object* v_unused_1894_; 
v_unused_1894_ = lean_ctor_get(v___x_1886_, 0);
lean_dec(v_unused_1894_);
v___x_1888_ = v___x_1886_;
v_isShared_1889_ = v_isSharedCheck_1893_;
goto v_resetjp_1887_;
}
else
{
lean_dec(v___x_1886_);
v___x_1888_ = lean_box(0);
v_isShared_1889_ = v_isSharedCheck_1893_;
goto v_resetjp_1887_;
}
v_resetjp_1887_:
{
lean_object* v___x_1891_; 
if (v_isShared_1889_ == 0)
{
lean_ctor_set(v___x_1888_, 0, v___x_1880_);
v___x_1891_ = v___x_1888_;
goto v_reusejp_1890_;
}
else
{
lean_object* v_reuseFailAlloc_1892_; 
v_reuseFailAlloc_1892_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1892_, 0, v___x_1880_);
v___x_1891_ = v_reuseFailAlloc_1892_;
goto v_reusejp_1890_;
}
v_reusejp_1890_:
{
return v___x_1891_;
}
}
}
else
{
return v___x_1886_;
}
}
else
{
v___y_1833_ = v___x_1880_;
v___y_1834_ = v___y_1871_;
v___y_1835_ = v___y_1872_;
v___y_1836_ = v___y_1873_;
v___y_1837_ = v___x_1882_;
v___y_1838_ = v_sz_1881_;
v___y_1839_ = v___x_1879_;
v___y_1840_ = v___x_1885_;
v___y_1841_ = v___y_1874_;
v___y_1842_ = v___y_1875_;
v___y_1843_ = v___y_1876_;
goto v___jp_1832_;
}
}
else
{
v___y_1833_ = v___x_1880_;
v___y_1834_ = v___y_1871_;
v___y_1835_ = v___y_1872_;
v___y_1836_ = v___y_1873_;
v___y_1837_ = v___x_1882_;
v___y_1838_ = v_sz_1881_;
v___y_1839_ = v___x_1879_;
v___y_1840_ = v___x_1885_;
v___y_1841_ = v___y_1874_;
v___y_1842_ = v___y_1875_;
v___y_1843_ = v___y_1876_;
goto v___jp_1832_;
}
}
v_resetjp_1897_:
{
lean_object* v___y_1901_; lean_object* v___y_1902_; uint8_t v___x_1917_; 
v___x_1917_ = lean_unbox(v_a_1896_);
if (v___x_1917_ == 0)
{
lean_object* v___x_1918_; lean_object* v___x_1920_; 
lean_dec(v_a_1896_);
lean_dec_ref(v_opt_1767_);
lean_dec_ref(v_p_1766_);
v___x_1918_ = lean_box(0);
if (v_isShared_1899_ == 0)
{
lean_ctor_set(v___x_1898_, 0, v___x_1918_);
v___x_1920_ = v___x_1898_;
goto v_reusejp_1919_;
}
else
{
lean_object* v_reuseFailAlloc_1921_; 
v_reuseFailAlloc_1921_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1921_, 0, v___x_1918_);
v___x_1920_ = v_reuseFailAlloc_1921_;
goto v_reusejp_1919_;
}
v_reusejp_1919_:
{
return v___x_1920_;
}
}
else
{
lean_object* v_total_1922_; lean_object* v_configParsing_1923_; lean_object* v_ruleSetConstruction_1924_; lean_object* v_search_1925_; lean_object* v_ruleSelection_1926_; lean_object* v_script_1927_; lean_object* v_forwardState_1928_; lean_object* v_scriptGenerated_1929_; lean_object* v_ruleStats_1930_; uint8_t v___x_1931_; lean_object* v___y_1933_; lean_object* v___y_1934_; lean_object* v___y_1935_; uint8_t v___y_1936_; lean_object* v___y_1937_; lean_object* v___y_1938_; lean_object* v___y_1939_; lean_object* v_a_1940_; lean_object* v___y_1950_; lean_object* v___y_1951_; lean_object* v___y_1952_; uint8_t v___y_1953_; lean_object* v___y_1954_; lean_object* v___y_1955_; lean_object* v___y_1956_; lean_object* v_a_1957_; lean_object* v___y_1960_; lean_object* v___y_1961_; lean_object* v___y_1962_; uint8_t v___y_1963_; lean_object* v___y_1964_; lean_object* v___y_1965_; lean_object* v___y_1966_; lean_object* v_a_1967_; lean_object* v___y_1970_; lean_object* v___y_1971_; lean_object* v___y_1972_; uint8_t v___y_1973_; lean_object* v___y_1974_; lean_object* v___y_1975_; lean_object* v___y_1976_; lean_object* v___y_1977_; lean_object* v___y_1981_; uint8_t v___y_1982_; lean_object* v___y_1983_; lean_object* v___y_1984_; lean_object* v___y_1985_; lean_object* v___y_1986_; uint8_t v___y_1987_; lean_object* v___y_1988_; lean_object* v___y_1989_; lean_object* v___y_1990_; lean_object* v___y_1991_; lean_object* v_a_1992_; lean_object* v___y_2002_; lean_object* v___y_2003_; uint8_t v___y_2004_; lean_object* v___y_2005_; lean_object* v___y_2006_; lean_object* v___y_2007_; uint8_t v___y_2008_; lean_object* v___y_2009_; lean_object* v___y_2010_; lean_object* v___y_2011_; lean_object* v___y_2012_; lean_object* v_a_2013_; lean_object* v___y_2016_; uint8_t v___y_2017_; lean_object* v___y_2018_; lean_object* v___y_2019_; lean_object* v___y_2020_; uint8_t v___y_2021_; lean_object* v___y_2022_; lean_object* v___y_2023_; lean_object* v___y_2024_; lean_object* v___y_2025_; lean_object* v___y_2026_; lean_object* v_a_2027_; lean_object* v___y_2040_; lean_object* v___y_2041_; uint8_t v___y_2042_; lean_object* v___y_2043_; lean_object* v___y_2044_; uint8_t v___y_2045_; lean_object* v___y_2046_; lean_object* v___y_2047_; lean_object* v___y_2048_; lean_object* v___y_2049_; lean_object* v___y_2050_; lean_object* v_a_2051_; uint8_t v___y_2054_; uint8_t v___y_2055_; uint8_t v___y_2056_; lean_object* v___y_2057_; lean_object* v___y_2058_; lean_object* v___y_2059_; lean_object* v___y_2060_; lean_object* v___y_2061_; lean_object* v___y_2062_; lean_object* v___y_2063_; size_t v___y_2064_; size_t v___y_2065_; lean_object* v___y_2066_; lean_object* v___y_2067_; lean_object* v___y_2094_; uint8_t v___y_2095_; lean_object* v___y_2096_; lean_object* v___y_2097_; uint8_t v___y_2098_; lean_object* v___y_2099_; lean_object* v___y_2100_; lean_object* v___y_2101_; lean_object* v___y_2102_; lean_object* v___y_2103_; lean_object* v___y_2104_; lean_object* v___y_2116_; lean_object* v___y_2117_; lean_object* v___y_2118_; uint8_t v___y_2119_; lean_object* v___y_2120_; lean_object* v___y_2121_; lean_object* v___y_2122_; lean_object* v_a_2123_; lean_object* v___y_2136_; lean_object* v___y_2137_; lean_object* v___y_2138_; uint8_t v___y_2139_; lean_object* v___y_2140_; lean_object* v___y_2141_; lean_object* v___y_2142_; lean_object* v_a_2143_; lean_object* v___y_2146_; lean_object* v___y_2147_; lean_object* v___y_2148_; uint8_t v___y_2149_; lean_object* v___y_2150_; lean_object* v___y_2151_; lean_object* v___y_2152_; lean_object* v_a_2153_; lean_object* v___y_2156_; lean_object* v___y_2157_; lean_object* v___y_2158_; uint8_t v___y_2159_; lean_object* v___y_2160_; lean_object* v___y_2161_; lean_object* v___y_2162_; lean_object* v___y_2163_; lean_object* v___y_2167_; lean_object* v___y_2168_; uint8_t v___y_2169_; lean_object* v___y_2170_; lean_object* v___y_2171_; uint8_t v___y_2172_; uint8_t v___y_2173_; lean_object* v___y_2174_; lean_object* v___y_2175_; lean_object* v___y_2176_; lean_object* v___y_2177_; lean_object* v___y_2178_; lean_object* v_a_2179_; lean_object* v___y_2192_; lean_object* v___y_2193_; uint8_t v___y_2194_; lean_object* v___y_2195_; lean_object* v___y_2196_; uint8_t v___y_2197_; uint8_t v___y_2198_; lean_object* v___y_2199_; lean_object* v___y_2200_; lean_object* v___y_2201_; lean_object* v___y_2202_; lean_object* v___y_2203_; lean_object* v_a_2204_; lean_object* v___y_2207_; lean_object* v___y_2208_; uint8_t v___y_2209_; lean_object* v___y_2210_; lean_object* v___y_2211_; uint8_t v___y_2212_; lean_object* v___y_2213_; uint8_t v___y_2214_; lean_object* v___y_2215_; lean_object* v___y_2216_; lean_object* v___y_2217_; lean_object* v___y_2218_; lean_object* v_a_2219_; lean_object* v___y_2229_; lean_object* v___y_2230_; uint8_t v___y_2231_; lean_object* v___y_2232_; lean_object* v___y_2233_; uint8_t v___y_2234_; lean_object* v___y_2235_; uint8_t v___y_2236_; lean_object* v___y_2237_; lean_object* v___y_2238_; lean_object* v___y_2239_; lean_object* v___y_2240_; lean_object* v_a_2241_; uint8_t v___y_2244_; uint8_t v___y_2245_; uint8_t v___y_2246_; lean_object* v___y_2247_; lean_object* v___y_2248_; lean_object* v___y_2249_; lean_object* v___y_2250_; lean_object* v___y_2251_; lean_object* v___y_2252_; lean_object* v___y_2253_; lean_object* v___y_2254_; size_t v___y_2255_; size_t v___y_2256_; uint8_t v___y_2257_; lean_object* v___y_2258_; lean_object* v___y_2285_; uint8_t v___y_2286_; lean_object* v___y_2287_; lean_object* v___y_2288_; uint8_t v___y_2289_; uint8_t v___y_2290_; lean_object* v___y_2291_; lean_object* v___y_2292_; lean_object* v___y_2293_; lean_object* v___y_2294_; lean_object* v___y_2295_; lean_object* v___y_2296_; lean_object* v___y_2308_; lean_object* v___y_2309_; lean_object* v___y_2310_; lean_object* v___y_2311_; uint8_t v___y_2312_; uint8_t v___y_2313_; lean_object* v___y_2314_; lean_object* v___y_2315_; lean_object* v___y_2316_; lean_object* v___y_2392_; lean_object* v___y_2393_; lean_object* v___y_2394_; lean_object* v___y_2395_; lean_object* v___y_2472_; lean_object* v___x_2513_; lean_object* v___x_2514_; uint8_t v___x_2515_; 
lean_del_object(v___x_1898_);
v_total_1922_ = lean_ctor_get(v_p_1766_, 0);
v_configParsing_1923_ = lean_ctor_get(v_p_1766_, 1);
v_ruleSetConstruction_1924_ = lean_ctor_get(v_p_1766_, 2);
v_search_1925_ = lean_ctor_get(v_p_1766_, 3);
v_ruleSelection_1926_ = lean_ctor_get(v_p_1766_, 4);
v_script_1927_ = lean_ctor_get(v_p_1766_, 5);
v_forwardState_1928_ = lean_ctor_get(v_p_1766_, 6);
v_scriptGenerated_1929_ = lean_ctor_get(v_p_1766_, 7);
lean_inc(v_scriptGenerated_1929_);
v_ruleStats_1930_ = lean_ctor_get(v_p_1766_, 8);
v___x_1931_ = 0;
v___x_2513_ = lean_unsigned_to_nat(0u);
v___x_2514_ = lean_array_get_size(v_ruleStats_1930_);
v___x_2515_ = lean_nat_dec_lt(v___x_2513_, v___x_2514_);
if (v___x_2515_ == 0)
{
v___y_2472_ = v___x_2513_;
goto v___jp_2471_;
}
else
{
uint8_t v___x_2516_; 
v___x_2516_ = lean_nat_dec_le(v___x_2514_, v___x_2514_);
if (v___x_2516_ == 0)
{
if (v___x_2515_ == 0)
{
v___y_2472_ = v___x_2513_;
goto v___jp_2471_;
}
else
{
size_t v___x_2517_; size_t v___x_2518_; lean_object* v___x_2519_; 
v___x_2517_ = ((size_t)0ULL);
v___x_2518_ = lean_usize_of_nat(v___x_2514_);
v___x_2519_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__8(v_ruleStats_1930_, v___x_2517_, v___x_2518_, v___x_2513_);
v___y_2472_ = v___x_2519_;
goto v___jp_2471_;
}
}
else
{
size_t v___x_2520_; size_t v___x_2521_; lean_object* v___x_2522_; 
v___x_2520_ = ((size_t)0ULL);
v___x_2521_ = lean_usize_of_nat(v___x_2514_);
v___x_2522_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__8(v_ruleStats_1930_, v___x_2520_, v___x_2521_, v___x_2513_);
v___y_2472_ = v___x_2522_;
goto v___jp_2471_;
}
}
v___jp_1932_:
{
lean_object* v___x_1941_; double v___x_1942_; double v___x_1943_; lean_object* v___x_1944_; lean_object* v___x_1945_; lean_object* v___x_1946_; lean_object* v___x_1947_; lean_object* v___x_1948_; 
v___x_1941_ = lean_io_get_num_heartbeats();
v___x_1942_ = lean_float_of_nat(v___y_1937_);
v___x_1943_ = lean_float_of_nat(v___x_1941_);
v___x_1944_ = lean_box_float(v___x_1942_);
v___x_1945_ = lean_box_float(v___x_1943_);
v___x_1946_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1946_, 0, v___x_1944_);
lean_ctor_set(v___x_1946_, 1, v___x_1945_);
v___x_1947_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1947_, 0, v_a_1940_);
lean_ctor_set(v___x_1947_, 1, v___x_1946_);
lean_inc_ref(v___y_1933_);
v___x_1948_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5(v___y_1934_, v___x_1931_, v___y_1933_, v___y_1935_, v___y_1936_, v___y_1938_, v___y_1939_, v___x_1947_, v_a_1768_, v_a_1769_);
return v___x_1948_;
}
v___jp_1949_:
{
lean_object* v___x_1958_; 
v___x_1958_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1958_, 0, v_a_1957_);
v___y_1933_ = v___y_1950_;
v___y_1934_ = v___y_1951_;
v___y_1935_ = v___y_1952_;
v___y_1936_ = v___y_1953_;
v___y_1937_ = v___y_1954_;
v___y_1938_ = v___y_1955_;
v___y_1939_ = v___y_1956_;
v_a_1940_ = v___x_1958_;
goto v___jp_1932_;
}
v___jp_1959_:
{
lean_object* v___x_1968_; 
v___x_1968_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1968_, 0, v_a_1967_);
v___y_1933_ = v___y_1960_;
v___y_1934_ = v___y_1961_;
v___y_1935_ = v___y_1962_;
v___y_1936_ = v___y_1963_;
v___y_1937_ = v___y_1964_;
v___y_1938_ = v___y_1965_;
v___y_1939_ = v___y_1966_;
v_a_1940_ = v___x_1968_;
goto v___jp_1932_;
}
v___jp_1969_:
{
if (lean_obj_tag(v___y_1977_) == 0)
{
lean_object* v_a_1978_; 
v_a_1978_ = lean_ctor_get(v___y_1977_, 0);
lean_inc(v_a_1978_);
lean_dec_ref_known(v___y_1977_, 1);
v___y_1950_ = v___y_1970_;
v___y_1951_ = v___y_1971_;
v___y_1952_ = v___y_1972_;
v___y_1953_ = v___y_1973_;
v___y_1954_ = v___y_1974_;
v___y_1955_ = v___y_1975_;
v___y_1956_ = v___y_1976_;
v_a_1957_ = v_a_1978_;
goto v___jp_1949_;
}
else
{
lean_object* v_a_1979_; 
v_a_1979_ = lean_ctor_get(v___y_1977_, 0);
lean_inc(v_a_1979_);
lean_dec_ref_known(v___y_1977_, 1);
v___y_1960_ = v___y_1970_;
v___y_1961_ = v___y_1971_;
v___y_1962_ = v___y_1972_;
v___y_1963_ = v___y_1973_;
v___y_1964_ = v___y_1974_;
v___y_1965_ = v___y_1975_;
v___y_1966_ = v___y_1976_;
v_a_1967_ = v_a_1979_;
goto v___jp_1959_;
}
}
v___jp_1980_:
{
lean_object* v___x_1993_; double v___x_1994_; double v___x_1995_; lean_object* v___x_1996_; lean_object* v___x_1997_; lean_object* v___x_1998_; lean_object* v___x_1999_; lean_object* v___x_2000_; 
v___x_1993_ = lean_io_get_num_heartbeats();
v___x_1994_ = lean_float_of_nat(v___y_1986_);
v___x_1995_ = lean_float_of_nat(v___x_1993_);
v___x_1996_ = lean_box_float(v___x_1994_);
v___x_1997_ = lean_box_float(v___x_1995_);
v___x_1998_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1998_, 0, v___x_1996_);
lean_ctor_set(v___x_1998_, 1, v___x_1997_);
v___x_1999_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1999_, 0, v_a_1992_);
lean_ctor_set(v___x_1999_, 1, v___x_1998_);
lean_inc_ref(v___y_1983_);
lean_inc(v___y_1984_);
v___x_2000_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5(v___y_1984_, v___x_1931_, v___y_1983_, v___y_1985_, v___y_1982_, v___y_1981_, v___y_1991_, v___x_1999_, v_a_1768_, v_a_1769_);
v___y_1970_ = v___y_1983_;
v___y_1971_ = v___y_1984_;
v___y_1972_ = v___y_1985_;
v___y_1973_ = v___y_1987_;
v___y_1974_ = v___y_1988_;
v___y_1975_ = v___y_1989_;
v___y_1976_ = v___y_1990_;
v___y_1977_ = v___x_2000_;
goto v___jp_1969_;
}
v___jp_2001_:
{
lean_object* v___x_2014_; 
v___x_2014_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2014_, 0, v_a_2013_);
v___y_1981_ = v___y_2002_;
v___y_1982_ = v___y_2004_;
v___y_1983_ = v___y_2003_;
v___y_1984_ = v___y_2005_;
v___y_1985_ = v___y_2007_;
v___y_1986_ = v___y_2006_;
v___y_1987_ = v___y_2008_;
v___y_1988_ = v___y_2009_;
v___y_1989_ = v___y_2010_;
v___y_1990_ = v___y_2011_;
v___y_1991_ = v___y_2012_;
v_a_1992_ = v___x_2014_;
goto v___jp_1980_;
}
v___jp_2015_:
{
lean_object* v___x_2028_; double v___x_2029_; double v___x_2030_; double v___x_2031_; double v___x_2032_; double v___x_2033_; lean_object* v___x_2034_; lean_object* v___x_2035_; lean_object* v___x_2036_; lean_object* v___x_2037_; lean_object* v___x_2038_; 
v___x_2028_ = lean_io_mono_nanos_now();
v___x_2029_ = lean_float_of_nat(v___y_2023_);
v___x_2030_ = lean_float_once(&lp_aesop_Aesop_Stats_trace___closed__0, &lp_aesop_Aesop_Stats_trace___closed__0_once, _init_lp_aesop_Aesop_Stats_trace___closed__0);
v___x_2031_ = lean_float_div(v___x_2029_, v___x_2030_);
v___x_2032_ = lean_float_of_nat(v___x_2028_);
v___x_2033_ = lean_float_div(v___x_2032_, v___x_2030_);
v___x_2034_ = lean_box_float(v___x_2031_);
v___x_2035_ = lean_box_float(v___x_2033_);
v___x_2036_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2036_, 0, v___x_2034_);
lean_ctor_set(v___x_2036_, 1, v___x_2035_);
v___x_2037_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2037_, 0, v_a_2027_);
lean_ctor_set(v___x_2037_, 1, v___x_2036_);
lean_inc_ref(v___y_2018_);
lean_inc(v___y_2019_);
v___x_2038_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5(v___y_2019_, v___x_1931_, v___y_2018_, v___y_2020_, v___y_2017_, v___y_2016_, v___y_2026_, v___x_2037_, v_a_1768_, v_a_1769_);
v___y_1970_ = v___y_2018_;
v___y_1971_ = v___y_2019_;
v___y_1972_ = v___y_2020_;
v___y_1973_ = v___y_2021_;
v___y_1974_ = v___y_2022_;
v___y_1975_ = v___y_2024_;
v___y_1976_ = v___y_2025_;
v___y_1977_ = v___x_2038_;
goto v___jp_1969_;
}
v___jp_2039_:
{
lean_object* v___x_2052_; 
v___x_2052_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2052_, 0, v_a_2051_);
v___y_2016_ = v___y_2040_;
v___y_2017_ = v___y_2042_;
v___y_2018_ = v___y_2041_;
v___y_2019_ = v___y_2043_;
v___y_2020_ = v___y_2044_;
v___y_2021_ = v___y_2045_;
v___y_2022_ = v___y_2046_;
v___y_2023_ = v___y_2047_;
v___y_2024_ = v___y_2048_;
v___y_2025_ = v___y_2049_;
v___y_2026_ = v___y_2050_;
v_a_2027_ = v___x_2052_;
goto v___jp_2015_;
}
v___jp_2053_:
{
lean_object* v___x_2068_; 
v___x_2068_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg(v_a_1769_);
if (v___y_2055_ == 0)
{
lean_object* v_a_2069_; lean_object* v___x_2070_; lean_object* v___x_2071_; 
v_a_2069_ = lean_ctor_get(v___x_2068_, 0);
lean_inc(v_a_2069_);
lean_dec_ref(v___x_2068_);
v___x_2070_ = lean_io_mono_nanos_now();
lean_inc(v___y_2061_);
v___x_2071_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2(v___y_2061_, v___y_2055_, v___y_2063_, v___y_2065_, v___y_2064_, v___y_2057_, v_a_1768_, v_a_1769_);
lean_dec_ref(v___y_2063_);
if (lean_obj_tag(v___x_2071_) == 0)
{
lean_dec_ref_known(v___x_2071_, 1);
v___y_2040_ = v_a_2069_;
v___y_2041_ = v___y_2060_;
v___y_2042_ = v___y_2054_;
v___y_2043_ = v___y_2061_;
v___y_2044_ = v___y_2062_;
v___y_2045_ = v___y_2056_;
v___y_2046_ = v___y_2066_;
v___y_2047_ = v___x_2070_;
v___y_2048_ = v___y_2058_;
v___y_2049_ = v___y_2059_;
v___y_2050_ = v___y_2067_;
v_a_2051_ = v___y_2057_;
goto v___jp_2039_;
}
else
{
if (lean_obj_tag(v___x_2071_) == 0)
{
lean_object* v_a_2072_; 
v_a_2072_ = lean_ctor_get(v___x_2071_, 0);
lean_inc(v_a_2072_);
lean_dec_ref_known(v___x_2071_, 1);
v___y_2040_ = v_a_2069_;
v___y_2041_ = v___y_2060_;
v___y_2042_ = v___y_2054_;
v___y_2043_ = v___y_2061_;
v___y_2044_ = v___y_2062_;
v___y_2045_ = v___y_2056_;
v___y_2046_ = v___y_2066_;
v___y_2047_ = v___x_2070_;
v___y_2048_ = v___y_2058_;
v___y_2049_ = v___y_2059_;
v___y_2050_ = v___y_2067_;
v_a_2051_ = v_a_2072_;
goto v___jp_2039_;
}
else
{
lean_object* v_a_2073_; lean_object* v___x_2075_; uint8_t v_isShared_2076_; uint8_t v_isSharedCheck_2080_; 
v_a_2073_ = lean_ctor_get(v___x_2071_, 0);
v_isSharedCheck_2080_ = !lean_is_exclusive(v___x_2071_);
if (v_isSharedCheck_2080_ == 0)
{
v___x_2075_ = v___x_2071_;
v_isShared_2076_ = v_isSharedCheck_2080_;
goto v_resetjp_2074_;
}
else
{
lean_inc(v_a_2073_);
lean_dec(v___x_2071_);
v___x_2075_ = lean_box(0);
v_isShared_2076_ = v_isSharedCheck_2080_;
goto v_resetjp_2074_;
}
v_resetjp_2074_:
{
lean_object* v___x_2078_; 
if (v_isShared_2076_ == 0)
{
lean_ctor_set_tag(v___x_2075_, 0);
v___x_2078_ = v___x_2075_;
goto v_reusejp_2077_;
}
else
{
lean_object* v_reuseFailAlloc_2079_; 
v_reuseFailAlloc_2079_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2079_, 0, v_a_2073_);
v___x_2078_ = v_reuseFailAlloc_2079_;
goto v_reusejp_2077_;
}
v_reusejp_2077_:
{
v___y_2016_ = v_a_2069_;
v___y_2017_ = v___y_2054_;
v___y_2018_ = v___y_2060_;
v___y_2019_ = v___y_2061_;
v___y_2020_ = v___y_2062_;
v___y_2021_ = v___y_2056_;
v___y_2022_ = v___y_2066_;
v___y_2023_ = v___x_2070_;
v___y_2024_ = v___y_2058_;
v___y_2025_ = v___y_2059_;
v___y_2026_ = v___y_2067_;
v_a_2027_ = v___x_2078_;
goto v___jp_2015_;
}
}
}
}
}
else
{
lean_object* v_a_2081_; lean_object* v___x_2082_; lean_object* v___x_2083_; 
v_a_2081_ = lean_ctor_get(v___x_2068_, 0);
lean_inc(v_a_2081_);
lean_dec_ref(v___x_2068_);
v___x_2082_ = lean_io_get_num_heartbeats();
lean_inc(v___y_2061_);
v___x_2083_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2(v___y_2061_, v___y_2055_, v___y_2063_, v___y_2065_, v___y_2064_, v___y_2057_, v_a_1768_, v_a_1769_);
lean_dec_ref(v___y_2063_);
if (lean_obj_tag(v___x_2083_) == 0)
{
lean_dec_ref_known(v___x_2083_, 1);
v___y_2002_ = v_a_2081_;
v___y_2003_ = v___y_2060_;
v___y_2004_ = v___y_2054_;
v___y_2005_ = v___y_2061_;
v___y_2006_ = v___x_2082_;
v___y_2007_ = v___y_2062_;
v___y_2008_ = v___y_2056_;
v___y_2009_ = v___y_2066_;
v___y_2010_ = v___y_2058_;
v___y_2011_ = v___y_2059_;
v___y_2012_ = v___y_2067_;
v_a_2013_ = v___y_2057_;
goto v___jp_2001_;
}
else
{
if (lean_obj_tag(v___x_2083_) == 0)
{
lean_object* v_a_2084_; 
v_a_2084_ = lean_ctor_get(v___x_2083_, 0);
lean_inc(v_a_2084_);
lean_dec_ref_known(v___x_2083_, 1);
v___y_2002_ = v_a_2081_;
v___y_2003_ = v___y_2060_;
v___y_2004_ = v___y_2054_;
v___y_2005_ = v___y_2061_;
v___y_2006_ = v___x_2082_;
v___y_2007_ = v___y_2062_;
v___y_2008_ = v___y_2056_;
v___y_2009_ = v___y_2066_;
v___y_2010_ = v___y_2058_;
v___y_2011_ = v___y_2059_;
v___y_2012_ = v___y_2067_;
v_a_2013_ = v_a_2084_;
goto v___jp_2001_;
}
else
{
lean_object* v_a_2085_; lean_object* v___x_2087_; uint8_t v_isShared_2088_; uint8_t v_isSharedCheck_2092_; 
v_a_2085_ = lean_ctor_get(v___x_2083_, 0);
v_isSharedCheck_2092_ = !lean_is_exclusive(v___x_2083_);
if (v_isSharedCheck_2092_ == 0)
{
v___x_2087_ = v___x_2083_;
v_isShared_2088_ = v_isSharedCheck_2092_;
goto v_resetjp_2086_;
}
else
{
lean_inc(v_a_2085_);
lean_dec(v___x_2083_);
v___x_2087_ = lean_box(0);
v_isShared_2088_ = v_isSharedCheck_2092_;
goto v_resetjp_2086_;
}
v_resetjp_2086_:
{
lean_object* v___x_2090_; 
if (v_isShared_2088_ == 0)
{
lean_ctor_set_tag(v___x_2087_, 0);
v___x_2090_ = v___x_2087_;
goto v_reusejp_2089_;
}
else
{
lean_object* v_reuseFailAlloc_2091_; 
v_reuseFailAlloc_2091_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2091_, 0, v_a_2085_);
v___x_2090_ = v_reuseFailAlloc_2091_;
goto v_reusejp_2089_;
}
v_reusejp_2089_:
{
v___y_1981_ = v_a_2081_;
v___y_1982_ = v___y_2054_;
v___y_1983_ = v___y_2060_;
v___y_1984_ = v___y_2061_;
v___y_1985_ = v___y_2062_;
v___y_1986_ = v___x_2082_;
v___y_1987_ = v___y_2056_;
v___y_1988_ = v___y_2066_;
v___y_1989_ = v___y_2058_;
v___y_1990_ = v___y_2059_;
v___y_1991_ = v___y_2067_;
v_a_1992_ = v___x_2090_;
goto v___jp_1980_;
}
}
}
}
}
}
v___jp_2093_:
{
lean_object* v___x_2105_; lean_object* v___x_2106_; size_t v_sz_2107_; size_t v___x_2108_; lean_object* v___x_2109_; lean_object* v___x_2110_; uint8_t v___x_2111_; 
v___x_2105_ = lp_aesop_Aesop_sortRuleStatsTotals(v___y_2104_);
v___x_2106_ = lean_box(0);
v_sz_2107_ = lean_array_size(v___x_2105_);
v___x_2108_ = ((size_t)0ULL);
v___x_2109_ = ((lean_object*)(lp_aesop_Aesop_Stats_trace___closed__2));
lean_inc(v___y_2096_);
v___x_2110_ = l_Lean_Name_append(v___x_2109_, v___y_2096_);
v___x_2111_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v___y_2102_, v___y_2097_, v___x_2110_);
lean_dec(v___x_2110_);
if (v___x_2111_ == 0)
{
lean_object* v___x_2112_; uint8_t v___x_2113_; 
v___x_2112_ = l_Lean_trace_profiler;
v___x_2113_ = lp_aesop_Lean_Option_get___at___00Aesop_Stats_trace_spec__4(v___y_2097_, v___x_2112_);
if (v___x_2113_ == 0)
{
lean_object* v___x_2114_; 
lean_dec_ref(v___y_2103_);
lean_inc(v___y_2096_);
v___x_2114_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2(v___y_2096_, v___y_2095_, v___x_2105_, v_sz_2107_, v___x_2108_, v___x_2106_, v_a_1768_, v_a_1769_);
lean_dec_ref(v___x_2105_);
if (lean_obj_tag(v___x_2114_) == 0)
{
lean_dec_ref_known(v___x_2114_, 1);
v___y_1950_ = v___y_2094_;
v___y_1951_ = v___y_2096_;
v___y_1952_ = v___y_2097_;
v___y_1953_ = v___y_2098_;
v___y_1954_ = v___y_2099_;
v___y_1955_ = v___y_2100_;
v___y_1956_ = v___y_2101_;
v_a_1957_ = v___x_2106_;
goto v___jp_1949_;
}
else
{
v___y_1970_ = v___y_2094_;
v___y_1971_ = v___y_2096_;
v___y_1972_ = v___y_2097_;
v___y_1973_ = v___y_2098_;
v___y_1974_ = v___y_2099_;
v___y_1975_ = v___y_2100_;
v___y_1976_ = v___y_2101_;
v___y_1977_ = v___x_2114_;
goto v___jp_1969_;
}
}
else
{
v___y_2054_ = v___x_2111_;
v___y_2055_ = v___y_2095_;
v___y_2056_ = v___y_2098_;
v___y_2057_ = v___x_2106_;
v___y_2058_ = v___y_2100_;
v___y_2059_ = v___y_2101_;
v___y_2060_ = v___y_2094_;
v___y_2061_ = v___y_2096_;
v___y_2062_ = v___y_2097_;
v___y_2063_ = v___x_2105_;
v___y_2064_ = v___x_2108_;
v___y_2065_ = v_sz_2107_;
v___y_2066_ = v___y_2099_;
v___y_2067_ = v___y_2103_;
goto v___jp_2053_;
}
}
else
{
v___y_2054_ = v___x_2111_;
v___y_2055_ = v___y_2095_;
v___y_2056_ = v___y_2098_;
v___y_2057_ = v___x_2106_;
v___y_2058_ = v___y_2100_;
v___y_2059_ = v___y_2101_;
v___y_2060_ = v___y_2094_;
v___y_2061_ = v___y_2096_;
v___y_2062_ = v___y_2097_;
v___y_2063_ = v___x_2105_;
v___y_2064_ = v___x_2108_;
v___y_2065_ = v_sz_2107_;
v___y_2066_ = v___y_2099_;
v___y_2067_ = v___y_2103_;
goto v___jp_2053_;
}
}
v___jp_2115_:
{
lean_object* v___x_2124_; double v___x_2125_; double v___x_2126_; double v___x_2127_; double v___x_2128_; double v___x_2129_; lean_object* v___x_2130_; lean_object* v___x_2131_; lean_object* v___x_2132_; lean_object* v___x_2133_; lean_object* v___x_2134_; 
v___x_2124_ = lean_io_mono_nanos_now();
v___x_2125_ = lean_float_of_nat(v___y_2122_);
v___x_2126_ = lean_float_once(&lp_aesop_Aesop_Stats_trace___closed__0, &lp_aesop_Aesop_Stats_trace___closed__0_once, _init_lp_aesop_Aesop_Stats_trace___closed__0);
v___x_2127_ = lean_float_div(v___x_2125_, v___x_2126_);
v___x_2128_ = lean_float_of_nat(v___x_2124_);
v___x_2129_ = lean_float_div(v___x_2128_, v___x_2126_);
v___x_2130_ = lean_box_float(v___x_2127_);
v___x_2131_ = lean_box_float(v___x_2129_);
v___x_2132_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2132_, 0, v___x_2130_);
lean_ctor_set(v___x_2132_, 1, v___x_2131_);
v___x_2133_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2133_, 0, v_a_2123_);
lean_ctor_set(v___x_2133_, 1, v___x_2132_);
lean_inc_ref(v___y_2116_);
v___x_2134_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5(v___y_2117_, v___x_1931_, v___y_2116_, v___y_2118_, v___y_2119_, v___y_2120_, v___y_2121_, v___x_2133_, v_a_1768_, v_a_1769_);
return v___x_2134_;
}
v___jp_2135_:
{
lean_object* v___x_2144_; 
v___x_2144_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2144_, 0, v_a_2143_);
v___y_2116_ = v___y_2136_;
v___y_2117_ = v___y_2137_;
v___y_2118_ = v___y_2138_;
v___y_2119_ = v___y_2139_;
v___y_2120_ = v___y_2140_;
v___y_2121_ = v___y_2141_;
v___y_2122_ = v___y_2142_;
v_a_2123_ = v___x_2144_;
goto v___jp_2115_;
}
v___jp_2145_:
{
lean_object* v___x_2154_; 
v___x_2154_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_2154_, 0, v_a_2153_);
v___y_2116_ = v___y_2146_;
v___y_2117_ = v___y_2147_;
v___y_2118_ = v___y_2148_;
v___y_2119_ = v___y_2149_;
v___y_2120_ = v___y_2150_;
v___y_2121_ = v___y_2151_;
v___y_2122_ = v___y_2152_;
v_a_2123_ = v___x_2154_;
goto v___jp_2115_;
}
v___jp_2155_:
{
if (lean_obj_tag(v___y_2163_) == 0)
{
lean_object* v_a_2164_; 
v_a_2164_ = lean_ctor_get(v___y_2163_, 0);
lean_inc(v_a_2164_);
lean_dec_ref_known(v___y_2163_, 1);
v___y_2136_ = v___y_2156_;
v___y_2137_ = v___y_2157_;
v___y_2138_ = v___y_2158_;
v___y_2139_ = v___y_2159_;
v___y_2140_ = v___y_2160_;
v___y_2141_ = v___y_2161_;
v___y_2142_ = v___y_2162_;
v_a_2143_ = v_a_2164_;
goto v___jp_2135_;
}
else
{
lean_object* v_a_2165_; 
v_a_2165_ = lean_ctor_get(v___y_2163_, 0);
lean_inc(v_a_2165_);
lean_dec_ref_known(v___y_2163_, 1);
v___y_2146_ = v___y_2156_;
v___y_2147_ = v___y_2157_;
v___y_2148_ = v___y_2158_;
v___y_2149_ = v___y_2159_;
v___y_2150_ = v___y_2160_;
v___y_2151_ = v___y_2161_;
v___y_2152_ = v___y_2162_;
v_a_2153_ = v_a_2165_;
goto v___jp_2145_;
}
}
v___jp_2166_:
{
lean_object* v___x_2180_; double v___x_2181_; double v___x_2182_; double v___x_2183_; double v___x_2184_; double v___x_2185_; lean_object* v___x_2186_; lean_object* v___x_2187_; lean_object* v___x_2188_; lean_object* v___x_2189_; lean_object* v___x_2190_; 
v___x_2180_ = lean_io_mono_nanos_now();
v___x_2181_ = lean_float_of_nat(v___y_2177_);
v___x_2182_ = lean_float_once(&lp_aesop_Aesop_Stats_trace___closed__0, &lp_aesop_Aesop_Stats_trace___closed__0_once, _init_lp_aesop_Aesop_Stats_trace___closed__0);
v___x_2183_ = lean_float_div(v___x_2181_, v___x_2182_);
v___x_2184_ = lean_float_of_nat(v___x_2180_);
v___x_2185_ = lean_float_div(v___x_2184_, v___x_2182_);
v___x_2186_ = lean_box_float(v___x_2183_);
v___x_2187_ = lean_box_float(v___x_2185_);
v___x_2188_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2188_, 0, v___x_2186_);
lean_ctor_set(v___x_2188_, 1, v___x_2187_);
v___x_2189_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2189_, 0, v_a_2179_);
lean_ctor_set(v___x_2189_, 1, v___x_2188_);
lean_inc_ref(v___y_2167_);
lean_inc(v___y_2170_);
v___x_2190_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5(v___y_2170_, v___y_2169_, v___y_2167_, v___y_2171_, v___y_2173_, v___y_2168_, v___y_2176_, v___x_2189_, v_a_1768_, v_a_1769_);
v___y_2156_ = v___y_2167_;
v___y_2157_ = v___y_2170_;
v___y_2158_ = v___y_2171_;
v___y_2159_ = v___y_2172_;
v___y_2160_ = v___y_2174_;
v___y_2161_ = v___y_2175_;
v___y_2162_ = v___y_2178_;
v___y_2163_ = v___x_2190_;
goto v___jp_2155_;
}
v___jp_2191_:
{
lean_object* v___x_2205_; 
v___x_2205_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2205_, 0, v_a_2204_);
v___y_2167_ = v___y_2192_;
v___y_2168_ = v___y_2195_;
v___y_2169_ = v___y_2194_;
v___y_2170_ = v___y_2193_;
v___y_2171_ = v___y_2196_;
v___y_2172_ = v___y_2197_;
v___y_2173_ = v___y_2198_;
v___y_2174_ = v___y_2199_;
v___y_2175_ = v___y_2200_;
v___y_2176_ = v___y_2202_;
v___y_2177_ = v___y_2201_;
v___y_2178_ = v___y_2203_;
v_a_2179_ = v___x_2205_;
goto v___jp_2166_;
}
v___jp_2206_:
{
lean_object* v___x_2220_; double v___x_2221_; double v___x_2222_; lean_object* v___x_2223_; lean_object* v___x_2224_; lean_object* v___x_2225_; lean_object* v___x_2226_; lean_object* v___x_2227_; 
v___x_2220_ = lean_io_get_num_heartbeats();
v___x_2221_ = lean_float_of_nat(v___y_2213_);
v___x_2222_ = lean_float_of_nat(v___x_2220_);
v___x_2223_ = lean_box_float(v___x_2221_);
v___x_2224_ = lean_box_float(v___x_2222_);
v___x_2225_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2225_, 0, v___x_2223_);
lean_ctor_set(v___x_2225_, 1, v___x_2224_);
v___x_2226_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2226_, 0, v_a_2219_);
lean_ctor_set(v___x_2226_, 1, v___x_2225_);
lean_inc_ref(v___y_2207_);
lean_inc(v___y_2210_);
v___x_2227_ = lp_aesop___private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5(v___y_2210_, v___y_2209_, v___y_2207_, v___y_2211_, v___y_2214_, v___y_2208_, v___y_2217_, v___x_2226_, v_a_1768_, v_a_1769_);
v___y_2156_ = v___y_2207_;
v___y_2157_ = v___y_2210_;
v___y_2158_ = v___y_2211_;
v___y_2159_ = v___y_2212_;
v___y_2160_ = v___y_2215_;
v___y_2161_ = v___y_2216_;
v___y_2162_ = v___y_2218_;
v___y_2163_ = v___x_2227_;
goto v___jp_2155_;
}
v___jp_2228_:
{
lean_object* v___x_2242_; 
v___x_2242_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_2242_, 0, v_a_2241_);
v___y_2207_ = v___y_2229_;
v___y_2208_ = v___y_2232_;
v___y_2209_ = v___y_2231_;
v___y_2210_ = v___y_2230_;
v___y_2211_ = v___y_2233_;
v___y_2212_ = v___y_2234_;
v___y_2213_ = v___y_2235_;
v___y_2214_ = v___y_2236_;
v___y_2215_ = v___y_2237_;
v___y_2216_ = v___y_2238_;
v___y_2217_ = v___y_2239_;
v___y_2218_ = v___y_2240_;
v_a_2219_ = v___x_2242_;
goto v___jp_2206_;
}
v___jp_2243_:
{
lean_object* v___x_2259_; 
v___x_2259_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg(v_a_1769_);
if (v___y_2244_ == 0)
{
lean_object* v_a_2260_; lean_object* v___x_2261_; lean_object* v___x_2262_; 
v_a_2260_ = lean_ctor_get(v___x_2259_, 0);
lean_inc(v_a_2260_);
lean_dec_ref(v___x_2259_);
v___x_2261_ = lean_io_mono_nanos_now();
lean_inc(v___y_2253_);
v___x_2262_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2(v___y_2253_, v___y_2257_, v___y_2247_, v___y_2255_, v___y_2256_, v___y_2258_, v_a_1768_, v_a_1769_);
lean_dec_ref(v___y_2247_);
if (lean_obj_tag(v___x_2262_) == 0)
{
lean_dec_ref_known(v___x_2262_, 1);
v___y_2192_ = v___y_2252_;
v___y_2193_ = v___y_2253_;
v___y_2194_ = v___y_2244_;
v___y_2195_ = v_a_2260_;
v___y_2196_ = v___y_2254_;
v___y_2197_ = v___y_2245_;
v___y_2198_ = v___y_2246_;
v___y_2199_ = v___y_2248_;
v___y_2200_ = v___y_2249_;
v___y_2201_ = v___x_2261_;
v___y_2202_ = v___y_2250_;
v___y_2203_ = v___y_2251_;
v_a_2204_ = v___y_2258_;
goto v___jp_2191_;
}
else
{
if (lean_obj_tag(v___x_2262_) == 0)
{
lean_object* v_a_2263_; 
v_a_2263_ = lean_ctor_get(v___x_2262_, 0);
lean_inc(v_a_2263_);
lean_dec_ref_known(v___x_2262_, 1);
v___y_2192_ = v___y_2252_;
v___y_2193_ = v___y_2253_;
v___y_2194_ = v___y_2244_;
v___y_2195_ = v_a_2260_;
v___y_2196_ = v___y_2254_;
v___y_2197_ = v___y_2245_;
v___y_2198_ = v___y_2246_;
v___y_2199_ = v___y_2248_;
v___y_2200_ = v___y_2249_;
v___y_2201_ = v___x_2261_;
v___y_2202_ = v___y_2250_;
v___y_2203_ = v___y_2251_;
v_a_2204_ = v_a_2263_;
goto v___jp_2191_;
}
else
{
lean_object* v_a_2264_; lean_object* v___x_2266_; uint8_t v_isShared_2267_; uint8_t v_isSharedCheck_2271_; 
v_a_2264_ = lean_ctor_get(v___x_2262_, 0);
v_isSharedCheck_2271_ = !lean_is_exclusive(v___x_2262_);
if (v_isSharedCheck_2271_ == 0)
{
v___x_2266_ = v___x_2262_;
v_isShared_2267_ = v_isSharedCheck_2271_;
goto v_resetjp_2265_;
}
else
{
lean_inc(v_a_2264_);
lean_dec(v___x_2262_);
v___x_2266_ = lean_box(0);
v_isShared_2267_ = v_isSharedCheck_2271_;
goto v_resetjp_2265_;
}
v_resetjp_2265_:
{
lean_object* v___x_2269_; 
if (v_isShared_2267_ == 0)
{
lean_ctor_set_tag(v___x_2266_, 0);
v___x_2269_ = v___x_2266_;
goto v_reusejp_2268_;
}
else
{
lean_object* v_reuseFailAlloc_2270_; 
v_reuseFailAlloc_2270_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2270_, 0, v_a_2264_);
v___x_2269_ = v_reuseFailAlloc_2270_;
goto v_reusejp_2268_;
}
v_reusejp_2268_:
{
v___y_2167_ = v___y_2252_;
v___y_2168_ = v_a_2260_;
v___y_2169_ = v___y_2244_;
v___y_2170_ = v___y_2253_;
v___y_2171_ = v___y_2254_;
v___y_2172_ = v___y_2245_;
v___y_2173_ = v___y_2246_;
v___y_2174_ = v___y_2248_;
v___y_2175_ = v___y_2249_;
v___y_2176_ = v___y_2250_;
v___y_2177_ = v___x_2261_;
v___y_2178_ = v___y_2251_;
v_a_2179_ = v___x_2269_;
goto v___jp_2166_;
}
}
}
}
}
else
{
lean_object* v_a_2272_; lean_object* v___x_2273_; lean_object* v___x_2274_; 
v_a_2272_ = lean_ctor_get(v___x_2259_, 0);
lean_inc(v_a_2272_);
lean_dec_ref(v___x_2259_);
v___x_2273_ = lean_io_get_num_heartbeats();
lean_inc(v___y_2253_);
v___x_2274_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2(v___y_2253_, v___y_2257_, v___y_2247_, v___y_2255_, v___y_2256_, v___y_2258_, v_a_1768_, v_a_1769_);
lean_dec_ref(v___y_2247_);
if (lean_obj_tag(v___x_2274_) == 0)
{
lean_dec_ref_known(v___x_2274_, 1);
v___y_2229_ = v___y_2252_;
v___y_2230_ = v___y_2253_;
v___y_2231_ = v___y_2244_;
v___y_2232_ = v_a_2272_;
v___y_2233_ = v___y_2254_;
v___y_2234_ = v___y_2245_;
v___y_2235_ = v___x_2273_;
v___y_2236_ = v___y_2246_;
v___y_2237_ = v___y_2248_;
v___y_2238_ = v___y_2249_;
v___y_2239_ = v___y_2250_;
v___y_2240_ = v___y_2251_;
v_a_2241_ = v___y_2258_;
goto v___jp_2228_;
}
else
{
if (lean_obj_tag(v___x_2274_) == 0)
{
lean_object* v_a_2275_; 
v_a_2275_ = lean_ctor_get(v___x_2274_, 0);
lean_inc(v_a_2275_);
lean_dec_ref_known(v___x_2274_, 1);
v___y_2229_ = v___y_2252_;
v___y_2230_ = v___y_2253_;
v___y_2231_ = v___y_2244_;
v___y_2232_ = v_a_2272_;
v___y_2233_ = v___y_2254_;
v___y_2234_ = v___y_2245_;
v___y_2235_ = v___x_2273_;
v___y_2236_ = v___y_2246_;
v___y_2237_ = v___y_2248_;
v___y_2238_ = v___y_2249_;
v___y_2239_ = v___y_2250_;
v___y_2240_ = v___y_2251_;
v_a_2241_ = v_a_2275_;
goto v___jp_2228_;
}
else
{
lean_object* v_a_2276_; lean_object* v___x_2278_; uint8_t v_isShared_2279_; uint8_t v_isSharedCheck_2283_; 
v_a_2276_ = lean_ctor_get(v___x_2274_, 0);
v_isSharedCheck_2283_ = !lean_is_exclusive(v___x_2274_);
if (v_isSharedCheck_2283_ == 0)
{
v___x_2278_ = v___x_2274_;
v_isShared_2279_ = v_isSharedCheck_2283_;
goto v_resetjp_2277_;
}
else
{
lean_inc(v_a_2276_);
lean_dec(v___x_2274_);
v___x_2278_ = lean_box(0);
v_isShared_2279_ = v_isSharedCheck_2283_;
goto v_resetjp_2277_;
}
v_resetjp_2277_:
{
lean_object* v___x_2281_; 
if (v_isShared_2279_ == 0)
{
lean_ctor_set_tag(v___x_2278_, 0);
v___x_2281_ = v___x_2278_;
goto v_reusejp_2280_;
}
else
{
lean_object* v_reuseFailAlloc_2282_; 
v_reuseFailAlloc_2282_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2282_, 0, v_a_2276_);
v___x_2281_ = v_reuseFailAlloc_2282_;
goto v_reusejp_2280_;
}
v_reusejp_2280_:
{
v___y_2207_ = v___y_2252_;
v___y_2208_ = v_a_2272_;
v___y_2209_ = v___y_2244_;
v___y_2210_ = v___y_2253_;
v___y_2211_ = v___y_2254_;
v___y_2212_ = v___y_2245_;
v___y_2213_ = v___x_2273_;
v___y_2214_ = v___y_2246_;
v___y_2215_ = v___y_2248_;
v___y_2216_ = v___y_2249_;
v___y_2217_ = v___y_2250_;
v___y_2218_ = v___y_2251_;
v_a_2219_ = v___x_2281_;
goto v___jp_2206_;
}
}
}
}
}
}
v___jp_2284_:
{
lean_object* v___x_2297_; lean_object* v___x_2298_; size_t v_sz_2299_; size_t v___x_2300_; lean_object* v___x_2301_; lean_object* v___x_2302_; uint8_t v___x_2303_; 
v___x_2297_ = lp_aesop_Aesop_sortRuleStatsTotals(v___y_2296_);
v___x_2298_ = lean_box(0);
v_sz_2299_ = lean_array_size(v___x_2297_);
v___x_2300_ = ((size_t)0ULL);
v___x_2301_ = ((lean_object*)(lp_aesop_Aesop_Stats_trace___closed__2));
lean_inc(v___y_2287_);
v___x_2302_ = l_Lean_Name_append(v___x_2301_, v___y_2287_);
v___x_2303_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v___y_2293_, v___y_2288_, v___x_2302_);
lean_dec(v___x_2302_);
if (v___x_2303_ == 0)
{
lean_object* v___x_2304_; uint8_t v___x_2305_; 
v___x_2304_ = l_Lean_trace_profiler;
v___x_2305_ = lp_aesop_Lean_Option_get___at___00Aesop_Stats_trace_spec__4(v___y_2288_, v___x_2304_);
if (v___x_2305_ == 0)
{
lean_object* v___x_2306_; 
lean_dec_ref(v___y_2294_);
lean_inc(v___y_2287_);
v___x_2306_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2(v___y_2287_, v___y_2290_, v___x_2297_, v_sz_2299_, v___x_2300_, v___x_2298_, v_a_1768_, v_a_1769_);
lean_dec_ref(v___x_2297_);
if (lean_obj_tag(v___x_2306_) == 0)
{
lean_dec_ref_known(v___x_2306_, 1);
v___y_2136_ = v___y_2285_;
v___y_2137_ = v___y_2287_;
v___y_2138_ = v___y_2288_;
v___y_2139_ = v___y_2289_;
v___y_2140_ = v___y_2291_;
v___y_2141_ = v___y_2292_;
v___y_2142_ = v___y_2295_;
v_a_2143_ = v___x_2298_;
goto v___jp_2135_;
}
else
{
v___y_2156_ = v___y_2285_;
v___y_2157_ = v___y_2287_;
v___y_2158_ = v___y_2288_;
v___y_2159_ = v___y_2289_;
v___y_2160_ = v___y_2291_;
v___y_2161_ = v___y_2292_;
v___y_2162_ = v___y_2295_;
v___y_2163_ = v___x_2306_;
goto v___jp_2155_;
}
}
else
{
v___y_2244_ = v___y_2286_;
v___y_2245_ = v___y_2289_;
v___y_2246_ = v___x_2303_;
v___y_2247_ = v___x_2297_;
v___y_2248_ = v___y_2291_;
v___y_2249_ = v___y_2292_;
v___y_2250_ = v___y_2294_;
v___y_2251_ = v___y_2295_;
v___y_2252_ = v___y_2285_;
v___y_2253_ = v___y_2287_;
v___y_2254_ = v___y_2288_;
v___y_2255_ = v_sz_2299_;
v___y_2256_ = v___x_2300_;
v___y_2257_ = v___y_2290_;
v___y_2258_ = v___x_2298_;
goto v___jp_2243_;
}
}
else
{
v___y_2244_ = v___y_2286_;
v___y_2245_ = v___y_2289_;
v___y_2246_ = v___x_2303_;
v___y_2247_ = v___x_2297_;
v___y_2248_ = v___y_2291_;
v___y_2249_ = v___y_2292_;
v___y_2250_ = v___y_2294_;
v___y_2251_ = v___y_2295_;
v___y_2252_ = v___y_2285_;
v___y_2253_ = v___y_2287_;
v___y_2254_ = v___y_2288_;
v___y_2255_ = v_sz_2299_;
v___y_2256_ = v___x_2300_;
v___y_2257_ = v___y_2290_;
v___y_2258_ = v___x_2298_;
goto v___jp_2243_;
}
}
v___jp_2307_:
{
lean_object* v___x_2317_; lean_object* v_a_2318_; lean_object* v___x_2319_; uint8_t v___x_2320_; 
v___x_2317_ = lp_aesop___private_Lean_Util_Trace_0__Lean_getResetTraces___at___00Aesop_Stats_trace_spec__3___redArg(v_a_1769_);
v_a_2318_ = lean_ctor_get(v___x_2317_, 0);
lean_inc(v_a_2318_);
lean_dec_ref(v___x_2317_);
v___x_2319_ = l_Lean_trace_profiler_useHeartbeats;
v___x_2320_ = lp_aesop_Lean_Option_get___at___00Aesop_Stats_trace_spec__4(v___y_2311_, v___x_2319_);
if (v___x_2320_ == 0)
{
lean_object* v___x_2321_; lean_object* v___x_2322_; 
v___x_2321_ = lean_io_mono_nanos_now();
lean_inc(v___y_2309_);
v___x_2322_ = lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1(v___y_2309_, v___y_2314_, v_a_1768_, v_a_1769_);
if (lean_obj_tag(v___x_2322_) == 0)
{
lean_object* v___x_2323_; lean_object* v___x_2324_; lean_object* v___x_2325_; lean_object* v___x_2326_; lean_object* v___x_2327_; 
lean_dec_ref_known(v___x_2322_, 1);
v___x_2323_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__4, &lp_aesop_Aesop_Stats_trace___closed__4_once, _init_lp_aesop_Aesop_Stats_trace___closed__4);
lean_inc(v_forwardState_1928_);
v___x_2324_ = lp_aesop_Aesop_Nanos_printAsMillis(v_forwardState_1928_);
v___x_2325_ = l_Lean_stringToMessageData(v___x_2324_);
v___x_2326_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2326_, 0, v___x_2323_);
lean_ctor_set(v___x_2326_, 1, v___x_2325_);
lean_inc(v___y_2309_);
v___x_2327_ = lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1(v___y_2309_, v___x_2326_, v_a_1768_, v_a_1769_);
if (lean_obj_tag(v___x_2327_) == 0)
{
lean_object* v___x_2328_; lean_object* v___x_2329_; lean_object* v___x_2330_; lean_object* v_size_2331_; lean_object* v_buckets_2332_; lean_object* v___x_2334_; uint8_t v_isShared_2335_; uint8_t v_isSharedCheck_2355_; 
lean_dec_ref_known(v___x_2327_, 1);
v___x_2328_ = lean_unsigned_to_nat(0u);
v___x_2329_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__6, &lp_aesop_Aesop_Stats_trace___closed__6_once, _init_lp_aesop_Aesop_Stats_trace___closed__6);
v___x_2330_ = lp_aesop_Aesop_Stats_ruleStatsTotals(v_p_1766_, v___x_2329_);
lean_dec_ref(v_p_1766_);
v_size_2331_ = lean_ctor_get(v___x_2330_, 0);
v_buckets_2332_ = lean_ctor_get(v___x_2330_, 1);
v_isSharedCheck_2355_ = !lean_is_exclusive(v___x_2330_);
if (v_isSharedCheck_2355_ == 0)
{
v___x_2334_ = v___x_2330_;
v_isShared_2335_ = v_isSharedCheck_2355_;
goto v_resetjp_2333_;
}
else
{
lean_inc(v_buckets_2332_);
lean_inc(v_size_2331_);
lean_dec(v___x_2330_);
v___x_2334_ = lean_box(0);
v_isShared_2335_ = v_isSharedCheck_2355_;
goto v_resetjp_2333_;
}
v_resetjp_2333_:
{
lean_object* v___x_2336_; lean_object* v___x_2337_; lean_object* v___x_2338_; lean_object* v___x_2340_; 
v___x_2336_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__8, &lp_aesop_Aesop_Stats_trace___closed__8_once, _init_lp_aesop_Aesop_Stats_trace___closed__8);
v___x_2337_ = lp_aesop_Aesop_Nanos_printAsMillis(v___y_2310_);
v___x_2338_ = l_Lean_stringToMessageData(v___x_2337_);
if (v_isShared_2335_ == 0)
{
lean_ctor_set_tag(v___x_2334_, 7);
lean_ctor_set(v___x_2334_, 1, v___x_2338_);
lean_ctor_set(v___x_2334_, 0, v___x_2336_);
v___x_2340_ = v___x_2334_;
goto v_reusejp_2339_;
}
else
{
lean_object* v_reuseFailAlloc_2354_; 
v_reuseFailAlloc_2354_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2354_, 0, v___x_2336_);
lean_ctor_set(v_reuseFailAlloc_2354_, 1, v___x_2338_);
v___x_2340_ = v_reuseFailAlloc_2354_;
goto v_reusejp_2339_;
}
v_reusejp_2339_:
{
lean_object* v___x_2341_; lean_object* v___x_2342_; lean_object* v___f_2343_; lean_object* v___x_2344_; lean_object* v___x_2345_; uint8_t v___x_2346_; 
v___x_2341_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__10, &lp_aesop_Aesop_Stats_trace___closed__10_once, _init_lp_aesop_Aesop_Stats_trace___closed__10);
v___x_2342_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2342_, 0, v___x_2340_);
lean_ctor_set(v___x_2342_, 1, v___x_2341_);
v___f_2343_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Stats_trace___lam__0___boxed), 5, 1);
lean_closure_set(v___f_2343_, 0, v___x_2342_);
v___x_2344_ = lean_mk_empty_array_with_capacity(v_size_2331_);
lean_dec(v_size_2331_);
v___x_2345_ = lean_array_get_size(v_buckets_2332_);
v___x_2346_ = lean_nat_dec_lt(v___x_2328_, v___x_2345_);
if (v___x_2346_ == 0)
{
lean_dec_ref(v_buckets_2332_);
v___y_2285_ = v___y_2308_;
v___y_2286_ = v___x_2320_;
v___y_2287_ = v___y_2309_;
v___y_2288_ = v___y_2311_;
v___y_2289_ = v___y_2312_;
v___y_2290_ = v___y_2313_;
v___y_2291_ = v_a_2318_;
v___y_2292_ = v___y_2316_;
v___y_2293_ = v___y_2315_;
v___y_2294_ = v___f_2343_;
v___y_2295_ = v___x_2321_;
v___y_2296_ = v___x_2344_;
goto v___jp_2284_;
}
else
{
uint8_t v___x_2347_; 
v___x_2347_ = lean_nat_dec_le(v___x_2345_, v___x_2345_);
if (v___x_2347_ == 0)
{
if (v___x_2346_ == 0)
{
lean_dec_ref(v_buckets_2332_);
v___y_2285_ = v___y_2308_;
v___y_2286_ = v___x_2320_;
v___y_2287_ = v___y_2309_;
v___y_2288_ = v___y_2311_;
v___y_2289_ = v___y_2312_;
v___y_2290_ = v___y_2313_;
v___y_2291_ = v_a_2318_;
v___y_2292_ = v___y_2316_;
v___y_2293_ = v___y_2315_;
v___y_2294_ = v___f_2343_;
v___y_2295_ = v___x_2321_;
v___y_2296_ = v___x_2344_;
goto v___jp_2284_;
}
else
{
size_t v___x_2348_; size_t v___x_2349_; lean_object* v___x_2350_; 
v___x_2348_ = ((size_t)0ULL);
v___x_2349_ = lean_usize_of_nat(v___x_2345_);
v___x_2350_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__7(v_buckets_2332_, v___x_2348_, v___x_2349_, v___x_2344_);
lean_dec_ref(v_buckets_2332_);
v___y_2285_ = v___y_2308_;
v___y_2286_ = v___x_2320_;
v___y_2287_ = v___y_2309_;
v___y_2288_ = v___y_2311_;
v___y_2289_ = v___y_2312_;
v___y_2290_ = v___y_2313_;
v___y_2291_ = v_a_2318_;
v___y_2292_ = v___y_2316_;
v___y_2293_ = v___y_2315_;
v___y_2294_ = v___f_2343_;
v___y_2295_ = v___x_2321_;
v___y_2296_ = v___x_2350_;
goto v___jp_2284_;
}
}
else
{
size_t v___x_2351_; size_t v___x_2352_; lean_object* v___x_2353_; 
v___x_2351_ = ((size_t)0ULL);
v___x_2352_ = lean_usize_of_nat(v___x_2345_);
v___x_2353_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__7(v_buckets_2332_, v___x_2351_, v___x_2352_, v___x_2344_);
lean_dec_ref(v_buckets_2332_);
v___y_2285_ = v___y_2308_;
v___y_2286_ = v___x_2320_;
v___y_2287_ = v___y_2309_;
v___y_2288_ = v___y_2311_;
v___y_2289_ = v___y_2312_;
v___y_2290_ = v___y_2313_;
v___y_2291_ = v_a_2318_;
v___y_2292_ = v___y_2316_;
v___y_2293_ = v___y_2315_;
v___y_2294_ = v___f_2343_;
v___y_2295_ = v___x_2321_;
v___y_2296_ = v___x_2353_;
goto v___jp_2284_;
}
}
}
}
}
else
{
lean_dec(v___y_2310_);
lean_dec_ref(v_p_1766_);
v___y_2156_ = v___y_2308_;
v___y_2157_ = v___y_2309_;
v___y_2158_ = v___y_2311_;
v___y_2159_ = v___y_2312_;
v___y_2160_ = v_a_2318_;
v___y_2161_ = v___y_2316_;
v___y_2162_ = v___x_2321_;
v___y_2163_ = v___x_2327_;
goto v___jp_2155_;
}
}
else
{
lean_dec(v___y_2310_);
lean_dec_ref(v_p_1766_);
v___y_2156_ = v___y_2308_;
v___y_2157_ = v___y_2309_;
v___y_2158_ = v___y_2311_;
v___y_2159_ = v___y_2312_;
v___y_2160_ = v_a_2318_;
v___y_2161_ = v___y_2316_;
v___y_2162_ = v___x_2321_;
v___y_2163_ = v___x_2322_;
goto v___jp_2155_;
}
}
else
{
lean_object* v___x_2356_; lean_object* v___x_2357_; 
v___x_2356_ = lean_io_get_num_heartbeats();
lean_inc(v___y_2309_);
v___x_2357_ = lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1(v___y_2309_, v___y_2314_, v_a_1768_, v_a_1769_);
if (lean_obj_tag(v___x_2357_) == 0)
{
lean_object* v___x_2358_; lean_object* v___x_2359_; lean_object* v___x_2360_; lean_object* v___x_2361_; lean_object* v___x_2362_; 
lean_dec_ref_known(v___x_2357_, 1);
v___x_2358_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__4, &lp_aesop_Aesop_Stats_trace___closed__4_once, _init_lp_aesop_Aesop_Stats_trace___closed__4);
lean_inc(v_forwardState_1928_);
v___x_2359_ = lp_aesop_Aesop_Nanos_printAsMillis(v_forwardState_1928_);
v___x_2360_ = l_Lean_stringToMessageData(v___x_2359_);
v___x_2361_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2361_, 0, v___x_2358_);
lean_ctor_set(v___x_2361_, 1, v___x_2360_);
lean_inc(v___y_2309_);
v___x_2362_ = lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1(v___y_2309_, v___x_2361_, v_a_1768_, v_a_1769_);
if (lean_obj_tag(v___x_2362_) == 0)
{
lean_object* v___x_2363_; lean_object* v___x_2364_; lean_object* v___x_2365_; lean_object* v_size_2366_; lean_object* v_buckets_2367_; lean_object* v___x_2369_; uint8_t v_isShared_2370_; uint8_t v_isSharedCheck_2390_; 
lean_dec_ref_known(v___x_2362_, 1);
v___x_2363_ = lean_unsigned_to_nat(0u);
v___x_2364_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__6, &lp_aesop_Aesop_Stats_trace___closed__6_once, _init_lp_aesop_Aesop_Stats_trace___closed__6);
v___x_2365_ = lp_aesop_Aesop_Stats_ruleStatsTotals(v_p_1766_, v___x_2364_);
lean_dec_ref(v_p_1766_);
v_size_2366_ = lean_ctor_get(v___x_2365_, 0);
v_buckets_2367_ = lean_ctor_get(v___x_2365_, 1);
v_isSharedCheck_2390_ = !lean_is_exclusive(v___x_2365_);
if (v_isSharedCheck_2390_ == 0)
{
v___x_2369_ = v___x_2365_;
v_isShared_2370_ = v_isSharedCheck_2390_;
goto v_resetjp_2368_;
}
else
{
lean_inc(v_buckets_2367_);
lean_inc(v_size_2366_);
lean_dec(v___x_2365_);
v___x_2369_ = lean_box(0);
v_isShared_2370_ = v_isSharedCheck_2390_;
goto v_resetjp_2368_;
}
v_resetjp_2368_:
{
lean_object* v___x_2371_; lean_object* v___x_2372_; lean_object* v___x_2373_; lean_object* v___x_2375_; 
v___x_2371_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__8, &lp_aesop_Aesop_Stats_trace___closed__8_once, _init_lp_aesop_Aesop_Stats_trace___closed__8);
v___x_2372_ = lp_aesop_Aesop_Nanos_printAsMillis(v___y_2310_);
v___x_2373_ = l_Lean_stringToMessageData(v___x_2372_);
if (v_isShared_2370_ == 0)
{
lean_ctor_set_tag(v___x_2369_, 7);
lean_ctor_set(v___x_2369_, 1, v___x_2373_);
lean_ctor_set(v___x_2369_, 0, v___x_2371_);
v___x_2375_ = v___x_2369_;
goto v_reusejp_2374_;
}
else
{
lean_object* v_reuseFailAlloc_2389_; 
v_reuseFailAlloc_2389_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2389_, 0, v___x_2371_);
lean_ctor_set(v_reuseFailAlloc_2389_, 1, v___x_2373_);
v___x_2375_ = v_reuseFailAlloc_2389_;
goto v_reusejp_2374_;
}
v_reusejp_2374_:
{
lean_object* v___x_2376_; lean_object* v___x_2377_; lean_object* v___f_2378_; lean_object* v___x_2379_; lean_object* v___x_2380_; uint8_t v___x_2381_; 
v___x_2376_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__10, &lp_aesop_Aesop_Stats_trace___closed__10_once, _init_lp_aesop_Aesop_Stats_trace___closed__10);
v___x_2377_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2377_, 0, v___x_2375_);
lean_ctor_set(v___x_2377_, 1, v___x_2376_);
v___f_2378_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Stats_trace___lam__0___boxed), 5, 1);
lean_closure_set(v___f_2378_, 0, v___x_2377_);
v___x_2379_ = lean_mk_empty_array_with_capacity(v_size_2366_);
lean_dec(v_size_2366_);
v___x_2380_ = lean_array_get_size(v_buckets_2367_);
v___x_2381_ = lean_nat_dec_lt(v___x_2363_, v___x_2380_);
if (v___x_2381_ == 0)
{
lean_dec_ref(v_buckets_2367_);
v___y_2094_ = v___y_2308_;
v___y_2095_ = v___x_2320_;
v___y_2096_ = v___y_2309_;
v___y_2097_ = v___y_2311_;
v___y_2098_ = v___y_2312_;
v___y_2099_ = v___x_2356_;
v___y_2100_ = v_a_2318_;
v___y_2101_ = v___y_2316_;
v___y_2102_ = v___y_2315_;
v___y_2103_ = v___f_2378_;
v___y_2104_ = v___x_2379_;
goto v___jp_2093_;
}
else
{
uint8_t v___x_2382_; 
v___x_2382_ = lean_nat_dec_le(v___x_2380_, v___x_2380_);
if (v___x_2382_ == 0)
{
if (v___x_2381_ == 0)
{
lean_dec_ref(v_buckets_2367_);
v___y_2094_ = v___y_2308_;
v___y_2095_ = v___x_2320_;
v___y_2096_ = v___y_2309_;
v___y_2097_ = v___y_2311_;
v___y_2098_ = v___y_2312_;
v___y_2099_ = v___x_2356_;
v___y_2100_ = v_a_2318_;
v___y_2101_ = v___y_2316_;
v___y_2102_ = v___y_2315_;
v___y_2103_ = v___f_2378_;
v___y_2104_ = v___x_2379_;
goto v___jp_2093_;
}
else
{
size_t v___x_2383_; size_t v___x_2384_; lean_object* v___x_2385_; 
v___x_2383_ = ((size_t)0ULL);
v___x_2384_ = lean_usize_of_nat(v___x_2380_);
v___x_2385_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__7(v_buckets_2367_, v___x_2383_, v___x_2384_, v___x_2379_);
lean_dec_ref(v_buckets_2367_);
v___y_2094_ = v___y_2308_;
v___y_2095_ = v___x_2320_;
v___y_2096_ = v___y_2309_;
v___y_2097_ = v___y_2311_;
v___y_2098_ = v___y_2312_;
v___y_2099_ = v___x_2356_;
v___y_2100_ = v_a_2318_;
v___y_2101_ = v___y_2316_;
v___y_2102_ = v___y_2315_;
v___y_2103_ = v___f_2378_;
v___y_2104_ = v___x_2385_;
goto v___jp_2093_;
}
}
else
{
size_t v___x_2386_; size_t v___x_2387_; lean_object* v___x_2388_; 
v___x_2386_ = ((size_t)0ULL);
v___x_2387_ = lean_usize_of_nat(v___x_2380_);
v___x_2388_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__7(v_buckets_2367_, v___x_2386_, v___x_2387_, v___x_2379_);
lean_dec_ref(v_buckets_2367_);
v___y_2094_ = v___y_2308_;
v___y_2095_ = v___x_2320_;
v___y_2096_ = v___y_2309_;
v___y_2097_ = v___y_2311_;
v___y_2098_ = v___y_2312_;
v___y_2099_ = v___x_2356_;
v___y_2100_ = v_a_2318_;
v___y_2101_ = v___y_2316_;
v___y_2102_ = v___y_2315_;
v___y_2103_ = v___f_2378_;
v___y_2104_ = v___x_2388_;
goto v___jp_2093_;
}
}
}
}
}
else
{
lean_dec(v___y_2310_);
lean_dec_ref(v_p_1766_);
v___y_1970_ = v___y_2308_;
v___y_1971_ = v___y_2309_;
v___y_1972_ = v___y_2311_;
v___y_1973_ = v___y_2312_;
v___y_1974_ = v___x_2356_;
v___y_1975_ = v_a_2318_;
v___y_1976_ = v___y_2316_;
v___y_1977_ = v___x_2362_;
goto v___jp_1969_;
}
}
else
{
lean_dec(v___y_2310_);
lean_dec_ref(v_p_1766_);
v___y_1970_ = v___y_2308_;
v___y_1971_ = v___y_2309_;
v___y_1972_ = v___y_2311_;
v___y_1973_ = v___y_2312_;
v___y_1974_ = v___x_2356_;
v___y_1975_ = v_a_2318_;
v___y_1976_ = v___y_2316_;
v___y_1977_ = v___x_2357_;
goto v___jp_1969_;
}
}
}
v___jp_2391_:
{
lean_object* v___x_2396_; lean_object* v___x_2397_; 
lean_inc_ref(v___y_2394_);
v___x_2396_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2396_, 0, v___y_2394_);
lean_ctor_set(v___x_2396_, 1, v___y_2395_);
lean_inc(v___y_2392_);
v___x_2397_ = lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1(v___y_2392_, v___x_2396_, v_a_1768_, v_a_1769_);
if (lean_obj_tag(v___x_2397_) == 0)
{
lean_object* v_options_2398_; lean_object* v_inheritedTraceOptions_2399_; uint8_t v_hasTrace_2400_; lean_object* v___x_2401_; lean_object* v___x_2402_; lean_object* v___x_2403_; lean_object* v___x_2404_; 
lean_dec_ref_known(v___x_2397_, 1);
v_options_2398_ = lean_ctor_get(v_a_1768_, 2);
v_inheritedTraceOptions_2399_ = lean_ctor_get(v_a_1768_, 13);
v_hasTrace_2400_ = lean_ctor_get_uint8(v_options_2398_, sizeof(void*)*1);
v___x_2401_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__12, &lp_aesop_Aesop_Stats_trace___closed__12_once, _init_lp_aesop_Aesop_Stats_trace___closed__12);
lean_inc(v_ruleSelection_1926_);
v___x_2402_ = lp_aesop_Aesop_Nanos_printAsMillis(v_ruleSelection_1926_);
v___x_2403_ = l_Lean_stringToMessageData(v___x_2402_);
v___x_2404_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2404_, 0, v___x_2401_);
lean_ctor_set(v___x_2404_, 1, v___x_2403_);
if (v_hasTrace_2400_ == 0)
{
lean_object* v___x_2405_; 
lean_dec(v___y_2393_);
lean_inc(v___y_2392_);
v___x_2405_ = lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1(v___y_2392_, v___x_2404_, v_a_1768_, v_a_1769_);
if (lean_obj_tag(v___x_2405_) == 0)
{
lean_object* v___x_2406_; lean_object* v___x_2407_; lean_object* v___x_2408_; lean_object* v___x_2409_; lean_object* v___x_2410_; 
lean_dec_ref_known(v___x_2405_, 1);
v___x_2406_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__4, &lp_aesop_Aesop_Stats_trace___closed__4_once, _init_lp_aesop_Aesop_Stats_trace___closed__4);
lean_inc(v_forwardState_1928_);
v___x_2407_ = lp_aesop_Aesop_Nanos_printAsMillis(v_forwardState_1928_);
v___x_2408_ = l_Lean_stringToMessageData(v___x_2407_);
v___x_2409_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2409_, 0, v___x_2406_);
lean_ctor_set(v___x_2409_, 1, v___x_2408_);
lean_inc(v___y_2392_);
v___x_2410_ = lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1(v___y_2392_, v___x_2409_, v_a_1768_, v_a_1769_);
if (lean_obj_tag(v___x_2410_) == 0)
{
lean_object* v___x_2411_; lean_object* v___x_2412_; lean_object* v___x_2413_; lean_object* v_size_2414_; lean_object* v_buckets_2415_; lean_object* v___x_2416_; lean_object* v___x_2417_; uint8_t v___x_2418_; 
lean_dec_ref_known(v___x_2410_, 1);
v___x_2411_ = lean_unsigned_to_nat(0u);
v___x_2412_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__6, &lp_aesop_Aesop_Stats_trace___closed__6_once, _init_lp_aesop_Aesop_Stats_trace___closed__6);
v___x_2413_ = lp_aesop_Aesop_Stats_ruleStatsTotals(v_p_1766_, v___x_2412_);
lean_dec_ref(v_p_1766_);
v_size_2414_ = lean_ctor_get(v___x_2413_, 0);
lean_inc(v_size_2414_);
v_buckets_2415_ = lean_ctor_get(v___x_2413_, 1);
lean_inc_ref(v_buckets_2415_);
lean_dec_ref(v___x_2413_);
v___x_2416_ = lean_mk_empty_array_with_capacity(v_size_2414_);
lean_dec(v_size_2414_);
v___x_2417_ = lean_array_get_size(v_buckets_2415_);
v___x_2418_ = lean_nat_dec_lt(v___x_2411_, v___x_2417_);
if (v___x_2418_ == 0)
{
lean_dec_ref(v_buckets_2415_);
v___y_1901_ = v___y_2392_;
v___y_1902_ = v___x_2416_;
goto v___jp_1900_;
}
else
{
uint8_t v___x_2419_; 
v___x_2419_ = lean_nat_dec_le(v___x_2417_, v___x_2417_);
if (v___x_2419_ == 0)
{
if (v___x_2418_ == 0)
{
lean_dec_ref(v_buckets_2415_);
v___y_1901_ = v___y_2392_;
v___y_1902_ = v___x_2416_;
goto v___jp_1900_;
}
else
{
size_t v___x_2420_; size_t v___x_2421_; lean_object* v___x_2422_; 
v___x_2420_ = ((size_t)0ULL);
v___x_2421_ = lean_usize_of_nat(v___x_2417_);
v___x_2422_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__7(v_buckets_2415_, v___x_2420_, v___x_2421_, v___x_2416_);
lean_dec_ref(v_buckets_2415_);
v___y_1901_ = v___y_2392_;
v___y_1902_ = v___x_2422_;
goto v___jp_1900_;
}
}
else
{
size_t v___x_2423_; size_t v___x_2424_; lean_object* v___x_2425_; 
v___x_2423_ = ((size_t)0ULL);
v___x_2424_ = lean_usize_of_nat(v___x_2417_);
v___x_2425_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__7(v_buckets_2415_, v___x_2423_, v___x_2424_, v___x_2416_);
lean_dec_ref(v_buckets_2415_);
v___y_1901_ = v___y_2392_;
v___y_1902_ = v___x_2425_;
goto v___jp_1900_;
}
}
}
else
{
lean_dec(v___y_2392_);
lean_dec(v_a_1896_);
lean_dec_ref(v_p_1766_);
return v___x_2410_;
}
}
else
{
lean_dec(v___y_2392_);
lean_dec(v_a_1896_);
lean_dec_ref(v_p_1766_);
return v___x_2405_;
}
}
else
{
lean_object* v___x_2426_; lean_object* v___x_2427_; lean_object* v___x_2428_; lean_object* v___x_2429_; lean_object* v___f_2430_; lean_object* v___x_2431_; lean_object* v___x_2432_; lean_object* v___x_2433_; uint8_t v___x_2434_; 
lean_dec(v_a_1896_);
v___x_2426_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__14, &lp_aesop_Aesop_Stats_trace___closed__14_once, _init_lp_aesop_Aesop_Stats_trace___closed__14);
lean_inc(v_search_1925_);
v___x_2427_ = lp_aesop_Aesop_Nanos_printAsMillis(v_search_1925_);
v___x_2428_ = l_Lean_stringToMessageData(v___x_2427_);
v___x_2429_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2429_, 0, v___x_2426_);
lean_ctor_set(v___x_2429_, 1, v___x_2428_);
v___f_2430_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Stats_trace___lam__0___boxed), 5, 1);
lean_closure_set(v___f_2430_, 0, v___x_2429_);
v___x_2431_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1___closed__0));
v___x_2432_ = ((lean_object*)(lp_aesop_Aesop_Stats_trace___closed__2));
lean_inc(v___y_2392_);
v___x_2433_ = l_Lean_Name_append(v___x_2432_, v___y_2392_);
v___x_2434_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_2399_, v_options_2398_, v___x_2433_);
lean_dec(v___x_2433_);
if (v___x_2434_ == 0)
{
lean_object* v___x_2435_; uint8_t v___x_2436_; 
v___x_2435_ = l_Lean_trace_profiler;
v___x_2436_ = lp_aesop_Lean_Option_get___at___00Aesop_Stats_trace_spec__4(v_options_2398_, v___x_2435_);
if (v___x_2436_ == 0)
{
lean_object* v___x_2437_; 
lean_dec_ref(v___f_2430_);
lean_inc(v___y_2392_);
v___x_2437_ = lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1(v___y_2392_, v___x_2404_, v_a_1768_, v_a_1769_);
if (lean_obj_tag(v___x_2437_) == 0)
{
lean_object* v___x_2438_; lean_object* v___x_2439_; lean_object* v___x_2440_; lean_object* v___x_2441_; lean_object* v___x_2442_; 
lean_dec_ref_known(v___x_2437_, 1);
v___x_2438_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__4, &lp_aesop_Aesop_Stats_trace___closed__4_once, _init_lp_aesop_Aesop_Stats_trace___closed__4);
lean_inc(v_forwardState_1928_);
v___x_2439_ = lp_aesop_Aesop_Nanos_printAsMillis(v_forwardState_1928_);
v___x_2440_ = l_Lean_stringToMessageData(v___x_2439_);
v___x_2441_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2441_, 0, v___x_2438_);
lean_ctor_set(v___x_2441_, 1, v___x_2440_);
lean_inc(v___y_2392_);
v___x_2442_ = lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1(v___y_2392_, v___x_2441_, v_a_1768_, v_a_1769_);
if (lean_obj_tag(v___x_2442_) == 0)
{
lean_object* v___x_2443_; lean_object* v___x_2444_; lean_object* v___x_2445_; lean_object* v_size_2446_; lean_object* v_buckets_2447_; lean_object* v___x_2449_; uint8_t v_isShared_2450_; uint8_t v_isSharedCheck_2470_; 
lean_dec_ref_known(v___x_2442_, 1);
v___x_2443_ = lean_unsigned_to_nat(0u);
v___x_2444_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__6, &lp_aesop_Aesop_Stats_trace___closed__6_once, _init_lp_aesop_Aesop_Stats_trace___closed__6);
v___x_2445_ = lp_aesop_Aesop_Stats_ruleStatsTotals(v_p_1766_, v___x_2444_);
lean_dec_ref(v_p_1766_);
v_size_2446_ = lean_ctor_get(v___x_2445_, 0);
v_buckets_2447_ = lean_ctor_get(v___x_2445_, 1);
v_isSharedCheck_2470_ = !lean_is_exclusive(v___x_2445_);
if (v_isSharedCheck_2470_ == 0)
{
v___x_2449_ = v___x_2445_;
v_isShared_2450_ = v_isSharedCheck_2470_;
goto v_resetjp_2448_;
}
else
{
lean_inc(v_buckets_2447_);
lean_inc(v_size_2446_);
lean_dec(v___x_2445_);
v___x_2449_ = lean_box(0);
v_isShared_2450_ = v_isSharedCheck_2470_;
goto v_resetjp_2448_;
}
v_resetjp_2448_:
{
lean_object* v___x_2451_; lean_object* v___x_2452_; lean_object* v___x_2453_; lean_object* v___x_2455_; 
v___x_2451_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__8, &lp_aesop_Aesop_Stats_trace___closed__8_once, _init_lp_aesop_Aesop_Stats_trace___closed__8);
v___x_2452_ = lp_aesop_Aesop_Nanos_printAsMillis(v___y_2393_);
v___x_2453_ = l_Lean_stringToMessageData(v___x_2452_);
if (v_isShared_2450_ == 0)
{
lean_ctor_set_tag(v___x_2449_, 7);
lean_ctor_set(v___x_2449_, 1, v___x_2453_);
lean_ctor_set(v___x_2449_, 0, v___x_2451_);
v___x_2455_ = v___x_2449_;
goto v_reusejp_2454_;
}
else
{
lean_object* v_reuseFailAlloc_2469_; 
v_reuseFailAlloc_2469_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2469_, 0, v___x_2451_);
lean_ctor_set(v_reuseFailAlloc_2469_, 1, v___x_2453_);
v___x_2455_ = v_reuseFailAlloc_2469_;
goto v_reusejp_2454_;
}
v_reusejp_2454_:
{
lean_object* v___x_2456_; lean_object* v___x_2457_; lean_object* v___f_2458_; lean_object* v___x_2459_; lean_object* v___x_2460_; uint8_t v___x_2461_; 
v___x_2456_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__10, &lp_aesop_Aesop_Stats_trace___closed__10_once, _init_lp_aesop_Aesop_Stats_trace___closed__10);
v___x_2457_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2457_, 0, v___x_2455_);
lean_ctor_set(v___x_2457_, 1, v___x_2456_);
v___f_2458_ = lean_alloc_closure((void*)(lp_aesop_Aesop_Stats_trace___lam__0___boxed), 5, 1);
lean_closure_set(v___f_2458_, 0, v___x_2457_);
v___x_2459_ = lean_mk_empty_array_with_capacity(v_size_2446_);
lean_dec(v_size_2446_);
v___x_2460_ = lean_array_get_size(v_buckets_2447_);
v___x_2461_ = lean_nat_dec_lt(v___x_2443_, v___x_2460_);
if (v___x_2461_ == 0)
{
lean_dec_ref(v_buckets_2447_);
v___y_1871_ = v___x_2431_;
v___y_1872_ = v___y_2392_;
v___y_1873_ = v_options_2398_;
v___y_1874_ = v_hasTrace_2400_;
v___y_1875_ = v___x_2436_;
v___y_1876_ = v___f_2458_;
v___y_1877_ = v_inheritedTraceOptions_2399_;
v___y_1878_ = v___x_2459_;
goto v___jp_1870_;
}
else
{
uint8_t v___x_2462_; 
v___x_2462_ = lean_nat_dec_le(v___x_2460_, v___x_2460_);
if (v___x_2462_ == 0)
{
if (v___x_2461_ == 0)
{
lean_dec_ref(v_buckets_2447_);
v___y_1871_ = v___x_2431_;
v___y_1872_ = v___y_2392_;
v___y_1873_ = v_options_2398_;
v___y_1874_ = v_hasTrace_2400_;
v___y_1875_ = v___x_2436_;
v___y_1876_ = v___f_2458_;
v___y_1877_ = v_inheritedTraceOptions_2399_;
v___y_1878_ = v___x_2459_;
goto v___jp_1870_;
}
else
{
size_t v___x_2463_; size_t v___x_2464_; lean_object* v___x_2465_; 
v___x_2463_ = ((size_t)0ULL);
v___x_2464_ = lean_usize_of_nat(v___x_2460_);
v___x_2465_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__7(v_buckets_2447_, v___x_2463_, v___x_2464_, v___x_2459_);
lean_dec_ref(v_buckets_2447_);
v___y_1871_ = v___x_2431_;
v___y_1872_ = v___y_2392_;
v___y_1873_ = v_options_2398_;
v___y_1874_ = v_hasTrace_2400_;
v___y_1875_ = v___x_2436_;
v___y_1876_ = v___f_2458_;
v___y_1877_ = v_inheritedTraceOptions_2399_;
v___y_1878_ = v___x_2465_;
goto v___jp_1870_;
}
}
else
{
size_t v___x_2466_; size_t v___x_2467_; lean_object* v___x_2468_; 
v___x_2466_ = ((size_t)0ULL);
v___x_2467_ = lean_usize_of_nat(v___x_2460_);
v___x_2468_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_Stats_trace_spec__7(v_buckets_2447_, v___x_2466_, v___x_2467_, v___x_2459_);
lean_dec_ref(v_buckets_2447_);
v___y_1871_ = v___x_2431_;
v___y_1872_ = v___y_2392_;
v___y_1873_ = v_options_2398_;
v___y_1874_ = v_hasTrace_2400_;
v___y_1875_ = v___x_2436_;
v___y_1876_ = v___f_2458_;
v___y_1877_ = v_inheritedTraceOptions_2399_;
v___y_1878_ = v___x_2468_;
goto v___jp_1870_;
}
}
}
}
}
else
{
lean_dec(v___y_2393_);
lean_dec(v___y_2392_);
lean_dec_ref(v_p_1766_);
return v___x_2442_;
}
}
else
{
lean_dec(v___y_2393_);
lean_dec(v___y_2392_);
lean_dec_ref(v_p_1766_);
return v___x_2437_;
}
}
else
{
v___y_2308_ = v___x_2431_;
v___y_2309_ = v___y_2392_;
v___y_2310_ = v___y_2393_;
v___y_2311_ = v_options_2398_;
v___y_2312_ = v___x_2434_;
v___y_2313_ = v_hasTrace_2400_;
v___y_2314_ = v___x_2404_;
v___y_2315_ = v_inheritedTraceOptions_2399_;
v___y_2316_ = v___f_2430_;
goto v___jp_2307_;
}
}
else
{
v___y_2308_ = v___x_2431_;
v___y_2309_ = v___y_2392_;
v___y_2310_ = v___y_2393_;
v___y_2311_ = v_options_2398_;
v___y_2312_ = v___x_2434_;
v___y_2313_ = v_hasTrace_2400_;
v___y_2314_ = v___x_2404_;
v___y_2315_ = v_inheritedTraceOptions_2399_;
v___y_2316_ = v___f_2430_;
goto v___jp_2307_;
}
}
}
else
{
lean_dec(v___y_2393_);
lean_dec(v___y_2392_);
lean_dec(v_a_1896_);
lean_dec_ref(v_p_1766_);
return v___x_2397_;
}
}
v___jp_2471_:
{
lean_object* v_traceClass_2473_; lean_object* v___x_2475_; uint8_t v_isShared_2476_; uint8_t v_isSharedCheck_2511_; 
v_traceClass_2473_ = lean_ctor_get(v_opt_1767_, 0);
v_isSharedCheck_2511_ = !lean_is_exclusive(v_opt_1767_);
if (v_isSharedCheck_2511_ == 0)
{
lean_object* v_unused_2512_; 
v_unused_2512_ = lean_ctor_get(v_opt_1767_, 1);
lean_dec(v_unused_2512_);
v___x_2475_ = v_opt_1767_;
v_isShared_2476_ = v_isSharedCheck_2511_;
goto v_resetjp_2474_;
}
else
{
lean_inc(v_traceClass_2473_);
lean_dec(v_opt_1767_);
v___x_2475_ = lean_box(0);
v_isShared_2476_ = v_isSharedCheck_2511_;
goto v_resetjp_2474_;
}
v_resetjp_2474_:
{
lean_object* v___x_2477_; lean_object* v___x_2478_; lean_object* v___x_2479_; lean_object* v___x_2481_; 
v___x_2477_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__16, &lp_aesop_Aesop_Stats_trace___closed__16_once, _init_lp_aesop_Aesop_Stats_trace___closed__16);
lean_inc(v_total_1922_);
v___x_2478_ = lp_aesop_Aesop_Nanos_printAsMillis(v_total_1922_);
v___x_2479_ = l_Lean_stringToMessageData(v___x_2478_);
if (v_isShared_2476_ == 0)
{
lean_ctor_set_tag(v___x_2475_, 7);
lean_ctor_set(v___x_2475_, 1, v___x_2479_);
lean_ctor_set(v___x_2475_, 0, v___x_2477_);
v___x_2481_ = v___x_2475_;
goto v_reusejp_2480_;
}
else
{
lean_object* v_reuseFailAlloc_2510_; 
v_reuseFailAlloc_2510_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v_reuseFailAlloc_2510_, 0, v___x_2477_);
lean_ctor_set(v_reuseFailAlloc_2510_, 1, v___x_2479_);
v___x_2481_ = v_reuseFailAlloc_2510_;
goto v_reusejp_2480_;
}
v_reusejp_2480_:
{
lean_object* v___x_2482_; 
lean_inc(v_traceClass_2473_);
v___x_2482_ = lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1(v_traceClass_2473_, v___x_2481_, v_a_1768_, v_a_1769_);
if (lean_obj_tag(v___x_2482_) == 0)
{
lean_object* v___x_2483_; lean_object* v___x_2484_; lean_object* v___x_2485_; lean_object* v___x_2486_; lean_object* v___x_2487_; 
lean_dec_ref_known(v___x_2482_, 1);
v___x_2483_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__18, &lp_aesop_Aesop_Stats_trace___closed__18_once, _init_lp_aesop_Aesop_Stats_trace___closed__18);
lean_inc(v_configParsing_1923_);
v___x_2484_ = lp_aesop_Aesop_Nanos_printAsMillis(v_configParsing_1923_);
v___x_2485_ = l_Lean_stringToMessageData(v___x_2484_);
v___x_2486_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2486_, 0, v___x_2483_);
lean_ctor_set(v___x_2486_, 1, v___x_2485_);
lean_inc(v_traceClass_2473_);
v___x_2487_ = lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1(v_traceClass_2473_, v___x_2486_, v_a_1768_, v_a_1769_);
if (lean_obj_tag(v___x_2487_) == 0)
{
lean_object* v___x_2488_; lean_object* v___x_2489_; lean_object* v___x_2490_; lean_object* v___x_2491_; lean_object* v___x_2492_; 
lean_dec_ref_known(v___x_2487_, 1);
v___x_2488_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__20, &lp_aesop_Aesop_Stats_trace___closed__20_once, _init_lp_aesop_Aesop_Stats_trace___closed__20);
lean_inc(v_ruleSetConstruction_1924_);
v___x_2489_ = lp_aesop_Aesop_Nanos_printAsMillis(v_ruleSetConstruction_1924_);
v___x_2490_ = l_Lean_stringToMessageData(v___x_2489_);
v___x_2491_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2491_, 0, v___x_2488_);
lean_ctor_set(v___x_2491_, 1, v___x_2490_);
lean_inc(v_traceClass_2473_);
v___x_2492_ = lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1(v_traceClass_2473_, v___x_2491_, v_a_1768_, v_a_1769_);
if (lean_obj_tag(v___x_2492_) == 0)
{
lean_object* v___x_2493_; lean_object* v___x_2494_; lean_object* v___x_2495_; lean_object* v___x_2496_; lean_object* v___x_2497_; 
lean_dec_ref_known(v___x_2492_, 1);
v___x_2493_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__22, &lp_aesop_Aesop_Stats_trace___closed__22_once, _init_lp_aesop_Aesop_Stats_trace___closed__22);
lean_inc(v_script_1927_);
v___x_2494_ = lp_aesop_Aesop_Nanos_printAsMillis(v_script_1927_);
v___x_2495_ = l_Lean_stringToMessageData(v___x_2494_);
v___x_2496_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_2496_, 0, v___x_2493_);
lean_ctor_set(v___x_2496_, 1, v___x_2495_);
lean_inc(v_traceClass_2473_);
v___x_2497_ = lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1(v_traceClass_2473_, v___x_2496_, v_a_1768_, v_a_1769_);
if (lean_obj_tag(v___x_2497_) == 0)
{
lean_object* v___x_2498_; 
lean_dec_ref_known(v___x_2497_, 1);
v___x_2498_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__24, &lp_aesop_Aesop_Stats_trace___closed__24_once, _init_lp_aesop_Aesop_Stats_trace___closed__24);
if (lean_obj_tag(v_scriptGenerated_1929_) == 0)
{
lean_object* v___x_2499_; 
v___x_2499_ = lean_obj_once(&lp_aesop_Aesop_Stats_trace___closed__27, &lp_aesop_Aesop_Stats_trace___closed__27_once, _init_lp_aesop_Aesop_Stats_trace___closed__27);
v___y_2392_ = v_traceClass_2473_;
v___y_2393_ = v___y_2472_;
v___y_2394_ = v___x_2498_;
v___y_2395_ = v___x_2499_;
goto v___jp_2391_;
}
else
{
lean_object* v_val_2500_; lean_object* v___x_2502_; uint8_t v_isShared_2503_; uint8_t v_isSharedCheck_2509_; 
v_val_2500_ = lean_ctor_get(v_scriptGenerated_1929_, 0);
v_isSharedCheck_2509_ = !lean_is_exclusive(v_scriptGenerated_1929_);
if (v_isSharedCheck_2509_ == 0)
{
v___x_2502_ = v_scriptGenerated_1929_;
v_isShared_2503_ = v_isSharedCheck_2509_;
goto v_resetjp_2501_;
}
else
{
lean_inc(v_val_2500_);
lean_dec(v_scriptGenerated_1929_);
v___x_2502_ = lean_box(0);
v_isShared_2503_ = v_isSharedCheck_2509_;
goto v_resetjp_2501_;
}
v_resetjp_2501_:
{
lean_object* v___x_2504_; lean_object* v___x_2506_; 
v___x_2504_ = lp_aesop_Aesop_ScriptGenerated_toString(v_val_2500_);
lean_dec(v_val_2500_);
if (v_isShared_2503_ == 0)
{
lean_ctor_set_tag(v___x_2502_, 3);
lean_ctor_set(v___x_2502_, 0, v___x_2504_);
v___x_2506_ = v___x_2502_;
goto v_reusejp_2505_;
}
else
{
lean_object* v_reuseFailAlloc_2508_; 
v_reuseFailAlloc_2508_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_2508_, 0, v___x_2504_);
v___x_2506_ = v_reuseFailAlloc_2508_;
goto v_reusejp_2505_;
}
v_reusejp_2505_:
{
lean_object* v___x_2507_; 
v___x_2507_ = l_Lean_MessageData_ofFormat(v___x_2506_);
v___y_2392_ = v_traceClass_2473_;
v___y_2393_ = v___y_2472_;
v___y_2394_ = v___x_2498_;
v___y_2395_ = v___x_2507_;
goto v___jp_2391_;
}
}
}
}
else
{
lean_dec(v_traceClass_2473_);
lean_dec(v___y_2472_);
lean_dec(v_scriptGenerated_1929_);
lean_dec(v_a_1896_);
lean_dec_ref(v_p_1766_);
return v___x_2497_;
}
}
else
{
lean_dec(v_traceClass_2473_);
lean_dec(v___y_2472_);
lean_dec(v_scriptGenerated_1929_);
lean_dec(v_a_1896_);
lean_dec_ref(v_p_1766_);
return v___x_2492_;
}
}
else
{
lean_dec(v_traceClass_2473_);
lean_dec(v___y_2472_);
lean_dec(v_scriptGenerated_1929_);
lean_dec(v_a_1896_);
lean_dec_ref(v_p_1766_);
return v___x_2487_;
}
}
else
{
lean_dec(v_traceClass_2473_);
lean_dec(v___y_2472_);
lean_dec(v_scriptGenerated_1929_);
lean_dec(v_a_1896_);
lean_dec_ref(v_p_1766_);
return v___x_2482_;
}
}
}
}
}
v___jp_1900_:
{
lean_object* v___x_1903_; lean_object* v___x_1904_; size_t v_sz_1905_; size_t v___x_1906_; uint8_t v___x_1907_; lean_object* v___x_1908_; 
v___x_1903_ = lp_aesop_Aesop_sortRuleStatsTotals(v___y_1902_);
v___x_1904_ = lean_box(0);
v_sz_1905_ = lean_array_size(v___x_1903_);
v___x_1906_ = ((size_t)0ULL);
v___x_1907_ = lean_unbox(v_a_1896_);
lean_dec(v_a_1896_);
v___x_1908_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_Stats_trace_spec__2(v___y_1901_, v___x_1907_, v___x_1903_, v_sz_1905_, v___x_1906_, v___x_1904_, v_a_1768_, v_a_1769_);
lean_dec_ref(v___x_1903_);
if (lean_obj_tag(v___x_1908_) == 0)
{
lean_object* v___x_1910_; uint8_t v_isShared_1911_; uint8_t v_isSharedCheck_1915_; 
v_isSharedCheck_1915_ = !lean_is_exclusive(v___x_1908_);
if (v_isSharedCheck_1915_ == 0)
{
lean_object* v_unused_1916_; 
v_unused_1916_ = lean_ctor_get(v___x_1908_, 0);
lean_dec(v_unused_1916_);
v___x_1910_ = v___x_1908_;
v_isShared_1911_ = v_isSharedCheck_1915_;
goto v_resetjp_1909_;
}
else
{
lean_dec(v___x_1908_);
v___x_1910_ = lean_box(0);
v_isShared_1911_ = v_isSharedCheck_1915_;
goto v_resetjp_1909_;
}
v_resetjp_1909_:
{
lean_object* v___x_1913_; 
if (v_isShared_1911_ == 0)
{
lean_ctor_set(v___x_1910_, 0, v___x_1904_);
v___x_1913_ = v___x_1910_;
goto v_reusejp_1912_;
}
else
{
lean_object* v_reuseFailAlloc_1914_; 
v_reuseFailAlloc_1914_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1914_, 0, v___x_1904_);
v___x_1913_ = v_reuseFailAlloc_1914_;
goto v_reusejp_1912_;
}
v_reusejp_1912_:
{
return v___x_1913_;
}
}
}
else
{
return v___x_1908_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Stats_trace___boxed(lean_object* v_p_2524_, lean_object* v_opt_2525_, lean_object* v_a_2526_, lean_object* v_a_2527_, lean_object* v_a_2528_){
_start:
{
lean_object* v_res_2529_; 
v_res_2529_ = lp_aesop_Aesop_Stats_trace(v_p_2524_, v_opt_2525_, v_a_2526_, v_a_2527_);
lean_dec(v_a_2527_);
lean_dec_ref(v_a_2526_);
return v_res_2529_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Stats_trace_spec__0(lean_object* v_opt_2530_, lean_object* v___y_2531_, lean_object* v___y_2532_){
_start:
{
lean_object* v___x_2534_; 
v___x_2534_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Stats_trace_spec__0___redArg(v_opt_2530_, v___y_2531_);
return v___x_2534_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Stats_trace_spec__0___boxed(lean_object* v_opt_2535_, lean_object* v___y_2536_, lean_object* v___y_2537_, lean_object* v___y_2538_){
_start:
{
lean_object* v_res_2539_; 
v_res_2539_ = lp_aesop_Aesop_TraceOption_isEnabled___at___00Aesop_Stats_trace_spec__0(v_opt_2535_, v___y_2536_, v___y_2537_);
lean_dec(v___y_2537_);
lean_dec_ref(v___y_2536_);
lean_dec_ref(v_opt_2535_);
return v_res_2539_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__7(lean_object* v_00_u03b1_2540_, lean_object* v_x_2541_, lean_object* v___y_2542_, lean_object* v___y_2543_){
_start:
{
lean_object* v___x_2545_; 
v___x_2545_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__7___redArg(v_x_2541_);
return v___x_2545_;
}
}
LEAN_EXPORT lean_object* lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__7___boxed(lean_object* v_00_u03b1_2546_, lean_object* v_x_2547_, lean_object* v___y_2548_, lean_object* v___y_2549_, lean_object* v___y_2550_){
_start:
{
lean_object* v_res_2551_; 
v_res_2551_ = lp_aesop_MonadExcept_ofExcept___at___00__private_Lean_Util_Trace_0__Lean_withTraceNode_postCallback___at___00Aesop_Stats_trace_spec__5_spec__7(v_00_u03b1_2546_, v_x_2547_, v___y_2548_, v___y_2549_);
lean_dec(v___y_2549_);
lean_dec_ref(v___y_2548_);
return v_res_2551_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg___lam__0(lean_object* v_modifyGetStats_2552_, lean_object* v_00_u03b1_2553_, lean_object* v_f_2554_, lean_object* v___y_2555_){
_start:
{
lean_object* v___x_2556_; 
v___x_2556_ = lean_apply_2(v_modifyGetStats_2552_, lean_box(0), v_f_2554_);
return v___x_2556_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg___lam__0___boxed(lean_object* v_modifyGetStats_2557_, lean_object* v_00_u03b1_2558_, lean_object* v_f_2559_, lean_object* v___y_2560_){
_start:
{
lean_object* v_res_2561_; 
v_res_2561_ = lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg___lam__0(v_modifyGetStats_2557_, v_00_u03b1_2558_, v_f_2559_, v___y_2560_);
lean_dec(v___y_2560_);
return v_res_2561_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg___lam__1(lean_object* v_modifyStats_2562_, lean_object* v_f_2563_, lean_object* v___y_2564_){
_start:
{
lean_object* v___x_2565_; 
v___x_2565_ = lean_apply_1(v_modifyStats_2562_, v_f_2563_);
return v___x_2565_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg___lam__1___boxed(lean_object* v_modifyStats_2566_, lean_object* v_f_2567_, lean_object* v___y_2568_){
_start:
{
lean_object* v_res_2569_; 
v_res_2569_ = lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg___lam__1(v_modifyStats_2566_, v_f_2567_, v___y_2568_);
lean_dec(v___y_2568_);
return v_res_2569_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(lean_object* v_inst_2570_){
_start:
{
lean_object* v_toMonadOptions_2571_; lean_object* v_modifyGetStats_2572_; lean_object* v_getStats_2573_; lean_object* v_modifyStats_2574_; lean_object* v___x_2576_; uint8_t v_isShared_2577_; uint8_t v_isSharedCheck_2585_; 
v_toMonadOptions_2571_ = lean_ctor_get(v_inst_2570_, 0);
v_modifyGetStats_2572_ = lean_ctor_get(v_inst_2570_, 1);
v_getStats_2573_ = lean_ctor_get(v_inst_2570_, 2);
v_modifyStats_2574_ = lean_ctor_get(v_inst_2570_, 3);
v_isSharedCheck_2585_ = !lean_is_exclusive(v_inst_2570_);
if (v_isSharedCheck_2585_ == 0)
{
v___x_2576_ = v_inst_2570_;
v_isShared_2577_ = v_isSharedCheck_2585_;
goto v_resetjp_2575_;
}
else
{
lean_inc(v_modifyStats_2574_);
lean_inc(v_getStats_2573_);
lean_inc(v_modifyGetStats_2572_);
lean_inc(v_toMonadOptions_2571_);
lean_dec(v_inst_2570_);
v___x_2576_ = lean_box(0);
v_isShared_2577_ = v_isSharedCheck_2585_;
goto v_resetjp_2575_;
}
v_resetjp_2575_:
{
lean_object* v___f_2578_; lean_object* v___f_2579_; lean_object* v___x_2580_; lean_object* v___x_2581_; lean_object* v___x_2583_; 
v___f_2578_ = lean_alloc_closure((void*)(lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_2578_, 0, v_modifyGetStats_2572_);
v___f_2579_ = lean_alloc_closure((void*)(lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_2579_, 0, v_modifyStats_2574_);
v___x_2580_ = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(v___x_2580_, 0, lean_box(0));
lean_closure_set(v___x_2580_, 1, lean_box(0));
lean_closure_set(v___x_2580_, 2, lean_box(0));
lean_closure_set(v___x_2580_, 3, lean_box(0));
lean_closure_set(v___x_2580_, 4, v_toMonadOptions_2571_);
v___x_2581_ = lean_alloc_closure((void*)(l_StateRefT_x27_lift___boxed), 6, 5);
lean_closure_set(v___x_2581_, 0, lean_box(0));
lean_closure_set(v___x_2581_, 1, lean_box(0));
lean_closure_set(v___x_2581_, 2, lean_box(0));
lean_closure_set(v___x_2581_, 3, lean_box(0));
lean_closure_set(v___x_2581_, 4, v_getStats_2573_);
if (v_isShared_2577_ == 0)
{
lean_ctor_set(v___x_2576_, 3, v___f_2579_);
lean_ctor_set(v___x_2576_, 2, v___x_2581_);
lean_ctor_set(v___x_2576_, 1, v___f_2578_);
lean_ctor_set(v___x_2576_, 0, v___x_2580_);
v___x_2583_ = v___x_2576_;
goto v_reusejp_2582_;
}
else
{
lean_object* v_reuseFailAlloc_2584_; 
v_reuseFailAlloc_2584_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_2584_, 0, v___x_2580_);
lean_ctor_set(v_reuseFailAlloc_2584_, 1, v___f_2578_);
lean_ctor_set(v_reuseFailAlloc_2584_, 2, v___x_2581_);
lean_ctor_set(v_reuseFailAlloc_2584_, 3, v___f_2579_);
v___x_2583_ = v_reuseFailAlloc_2584_;
goto v_reusejp_2582_;
}
v_reusejp_2582_:
{
return v___x_2583_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsStateRefT_x27(lean_object* v_m_2586_, lean_object* v_00_u03c9_2587_, lean_object* v_00_u03c3_2588_, lean_object* v_inst_2589_){
_start:
{
lean_object* v___x_2590_; 
v___x_2590_ = lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg(v_inst_2589_);
return v___x_2590_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsReaderT___redArg___lam__2(lean_object* v_getStats_2591_, lean_object* v___y_2592_){
_start:
{
lean_inc(v_getStats_2591_);
return v_getStats_2591_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsReaderT___redArg___lam__2___boxed(lean_object* v_getStats_2593_, lean_object* v___y_2594_){
_start:
{
lean_object* v_res_2595_; 
v_res_2595_ = lp_aesop_Aesop_instMonadStatsReaderT___redArg___lam__2(v_getStats_2593_, v___y_2594_);
lean_dec(v___y_2594_);
lean_dec(v_getStats_2593_);
return v_res_2595_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsReaderT___redArg(lean_object* v_inst_2596_){
_start:
{
lean_object* v_toMonadOptions_2597_; lean_object* v_modifyGetStats_2598_; lean_object* v_getStats_2599_; lean_object* v_modifyStats_2600_; lean_object* v___x_2602_; uint8_t v_isShared_2603_; uint8_t v_isSharedCheck_2611_; 
v_toMonadOptions_2597_ = lean_ctor_get(v_inst_2596_, 0);
v_modifyGetStats_2598_ = lean_ctor_get(v_inst_2596_, 1);
v_getStats_2599_ = lean_ctor_get(v_inst_2596_, 2);
v_modifyStats_2600_ = lean_ctor_get(v_inst_2596_, 3);
v_isSharedCheck_2611_ = !lean_is_exclusive(v_inst_2596_);
if (v_isSharedCheck_2611_ == 0)
{
v___x_2602_ = v_inst_2596_;
v_isShared_2603_ = v_isSharedCheck_2611_;
goto v_resetjp_2601_;
}
else
{
lean_inc(v_modifyStats_2600_);
lean_inc(v_getStats_2599_);
lean_inc(v_modifyGetStats_2598_);
lean_inc(v_toMonadOptions_2597_);
lean_dec(v_inst_2596_);
v___x_2602_ = lean_box(0);
v_isShared_2603_ = v_isSharedCheck_2611_;
goto v_resetjp_2601_;
}
v_resetjp_2601_:
{
lean_object* v___f_2604_; lean_object* v___f_2605_; lean_object* v___f_2606_; lean_object* v___x_2607_; lean_object* v___x_2609_; 
v___f_2604_ = lean_alloc_closure((void*)(lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_2604_, 0, v_modifyGetStats_2598_);
v___f_2605_ = lean_alloc_closure((void*)(lp_aesop_Aesop_instMonadStatsStateRefT_x27___redArg___lam__1___boxed), 3, 1);
lean_closure_set(v___f_2605_, 0, v_modifyStats_2600_);
v___f_2606_ = lean_alloc_closure((void*)(lp_aesop_Aesop_instMonadStatsReaderT___redArg___lam__2___boxed), 2, 1);
lean_closure_set(v___f_2606_, 0, v_getStats_2599_);
v___x_2607_ = lean_alloc_closure((void*)(l_ReaderT_instMonadLift___lam__0___boxed), 3, 2);
lean_closure_set(v___x_2607_, 0, lean_box(0));
lean_closure_set(v___x_2607_, 1, v_toMonadOptions_2597_);
if (v_isShared_2603_ == 0)
{
lean_ctor_set(v___x_2602_, 3, v___f_2605_);
lean_ctor_set(v___x_2602_, 2, v___f_2606_);
lean_ctor_set(v___x_2602_, 1, v___f_2604_);
lean_ctor_set(v___x_2602_, 0, v___x_2607_);
v___x_2609_ = v___x_2602_;
goto v_reusejp_2608_;
}
else
{
lean_object* v_reuseFailAlloc_2610_; 
v_reuseFailAlloc_2610_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v_reuseFailAlloc_2610_, 0, v___x_2607_);
lean_ctor_set(v_reuseFailAlloc_2610_, 1, v___f_2604_);
lean_ctor_set(v_reuseFailAlloc_2610_, 2, v___f_2606_);
lean_ctor_set(v_reuseFailAlloc_2610_, 3, v___f_2605_);
v___x_2609_ = v_reuseFailAlloc_2610_;
goto v_reusejp_2608_;
}
v_reusejp_2608_:
{
return v___x_2609_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsReaderT(lean_object* v_m_2612_, lean_object* v_00_u03b1_2613_, lean_object* v_inst_2614_){
_start:
{
lean_object* v___x_2615_; 
v___x_2615_ = lp_aesop_Aesop_instMonadStatsReaderT___redArg(v_inst_2614_);
return v___x_2615_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsOfMonadOptionsOfMonadStateOfStats___redArg___lam__0(lean_object* v_modifyGet_2616_, lean_object* v_00_u03b1_2617_, lean_object* v_f_2618_){
_start:
{
lean_object* v___x_2619_; 
v___x_2619_ = lean_apply_2(v_modifyGet_2616_, lean_box(0), v_f_2618_);
return v___x_2619_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsOfMonadOptionsOfMonadStateOfStats___redArg(lean_object* v_inst_2620_, lean_object* v_inst_2621_){
_start:
{
lean_object* v_get_2622_; lean_object* v_modifyGet_2623_; lean_object* v___f_2624_; lean_object* v___x_2625_; lean_object* v___x_2626_; 
v_get_2622_ = lean_ctor_get(v_inst_2621_, 0);
lean_inc(v_get_2622_);
v_modifyGet_2623_ = lean_ctor_get(v_inst_2621_, 2);
lean_inc(v_modifyGet_2623_);
v___f_2624_ = lean_alloc_closure((void*)(lp_aesop_Aesop_instMonadStatsOfMonadOptionsOfMonadStateOfStats___redArg___lam__0), 3, 1);
lean_closure_set(v___f_2624_, 0, v_modifyGet_2623_);
v___x_2625_ = lean_alloc_closure((void*)(l_modifyThe), 4, 3);
lean_closure_set(v___x_2625_, 0, lean_box(0));
lean_closure_set(v___x_2625_, 1, lean_box(0));
lean_closure_set(v___x_2625_, 2, v_inst_2621_);
v___x_2626_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2626_, 0, v_inst_2620_);
lean_ctor_set(v___x_2626_, 1, v___f_2624_);
lean_ctor_set(v___x_2626_, 2, v_get_2622_);
lean_ctor_set(v___x_2626_, 3, v___x_2625_);
return v___x_2626_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instMonadStatsOfMonadOptionsOfMonadStateOfStats(lean_object* v_m_2627_, lean_object* v_inst_2628_, lean_object* v_inst_2629_){
_start:
{
lean_object* v___x_2630_; 
v___x_2630_ = lp_aesop_Aesop_instMonadStatsOfMonadOptionsOfMonadStateOfStats___redArg(v_inst_2628_, v_inst_2629_);
return v___x_2630_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_enableStatsCollection___redArg___lam__0(lean_object* v___x_2631_, lean_object* v_opts_2632_){
_start:
{
lean_object* v___x_2633_; lean_object* v___x_2634_; uint8_t v___x_2635_; 
v___x_2633_ = lp_aesop_Aesop_aesop_collectStats;
v___x_2634_ = l_Lean_Option_get___redArg(v___x_2631_, v_opts_2632_, v___x_2633_);
v___x_2635_ = lean_unbox(v___x_2634_);
lean_dec(v___x_2634_);
return v___x_2635_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsCollection___redArg___lam__0___boxed(lean_object* v___x_2636_, lean_object* v_opts_2637_){
_start:
{
uint8_t v_res_2638_; lean_object* v_r_2639_; 
v_res_2638_ = lp_aesop_Aesop_enableStatsCollection___redArg___lam__0(v___x_2636_, v_opts_2637_);
lean_dec_ref(v_opts_2637_);
v_r_2639_ = lean_box(v_res_2638_);
return v_r_2639_;
}
}
static lean_object* _init_lp_aesop_Aesop_enableStatsCollection___redArg___closed__0(void){
_start:
{
lean_object* v___x_2640_; lean_object* v___f_2641_; 
v___x_2640_ = l_Lean_KVMap_instValueBool;
v___f_2641_ = lean_alloc_closure((void*)(lp_aesop_Aesop_enableStatsCollection___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_2641_, 0, v___x_2640_);
return v___f_2641_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsCollection___redArg(lean_object* v_inst_2642_, lean_object* v_inst_2643_){
_start:
{
lean_object* v_toApplicative_2644_; lean_object* v_toFunctor_2645_; lean_object* v_map_2646_; lean_object* v___f_2647_; lean_object* v___x_2648_; 
v_toApplicative_2644_ = lean_ctor_get(v_inst_2642_, 0);
lean_inc_ref(v_toApplicative_2644_);
lean_dec_ref(v_inst_2642_);
v_toFunctor_2645_ = lean_ctor_get(v_toApplicative_2644_, 0);
lean_inc_ref(v_toFunctor_2645_);
lean_dec_ref(v_toApplicative_2644_);
v_map_2646_ = lean_ctor_get(v_toFunctor_2645_, 0);
lean_inc(v_map_2646_);
lean_dec_ref(v_toFunctor_2645_);
v___f_2647_ = lean_obj_once(&lp_aesop_Aesop_enableStatsCollection___redArg___closed__0, &lp_aesop_Aesop_enableStatsCollection___redArg___closed__0_once, _init_lp_aesop_Aesop_enableStatsCollection___redArg___closed__0);
v___x_2648_ = lean_apply_4(v_map_2646_, lean_box(0), lean_box(0), v___f_2647_, v_inst_2643_);
return v___x_2648_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsCollection(lean_object* v_m_2649_, lean_object* v_inst_2650_, lean_object* v_inst_2651_){
_start:
{
lean_object* v_toApplicative_2652_; lean_object* v_toFunctor_2653_; lean_object* v_map_2654_; lean_object* v___f_2655_; lean_object* v___x_2656_; 
v_toApplicative_2652_ = lean_ctor_get(v_inst_2650_, 0);
lean_inc_ref(v_toApplicative_2652_);
lean_dec_ref(v_inst_2650_);
v_toFunctor_2653_ = lean_ctor_get(v_toApplicative_2652_, 0);
lean_inc_ref(v_toFunctor_2653_);
lean_dec_ref(v_toApplicative_2652_);
v_map_2654_ = lean_ctor_get(v_toFunctor_2653_, 0);
lean_inc(v_map_2654_);
lean_dec_ref(v_toFunctor_2653_);
v___f_2655_ = lean_obj_once(&lp_aesop_Aesop_enableStatsCollection___redArg___closed__0, &lp_aesop_Aesop_enableStatsCollection___redArg___closed__0_once, _init_lp_aesop_Aesop_enableStatsCollection___redArg___closed__0);
v___x_2656_ = lean_apply_4(v_map_2654_, lean_box(0), lean_box(0), v___f_2655_, v_inst_2651_);
return v___x_2656_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsTracing___redArg(lean_object* v_inst_2657_, lean_object* v_inst_2658_){
_start:
{
lean_object* v___x_2659_; lean_object* v___x_2660_; 
v___x_2659_ = lp_aesop_Aesop_TraceOption_stats;
v___x_2660_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v_inst_2657_, v_inst_2658_, v___x_2659_);
return v___x_2660_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsTracing(lean_object* v_m_2661_, lean_object* v_inst_2662_, lean_object* v_inst_2663_){
_start:
{
lean_object* v___x_2664_; lean_object* v___x_2665_; 
v___x_2664_ = lp_aesop_Aesop_TraceOption_stats;
v___x_2665_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v_inst_2662_, v_inst_2663_, v___x_2664_);
return v___x_2665_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsFile___redArg___lam__0(lean_object* v___x_2666_, lean_object* v_toPure_2667_, lean_object* v_____do__lift_2668_){
_start:
{
lean_object* v___x_2669_; lean_object* v___x_2670_; lean_object* v___x_2671_; uint8_t v___x_2672_; 
v___x_2669_ = lp_aesop_Aesop_aesop_stats_file;
v___x_2670_ = l_Lean_Option_get___redArg(v___x_2666_, v_____do__lift_2668_, v___x_2669_);
v___x_2671_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1___closed__0));
v___x_2672_ = lean_string_dec_eq(v___x_2670_, v___x_2671_);
lean_dec(v___x_2670_);
if (v___x_2672_ == 0)
{
uint8_t v___x_2673_; lean_object* v___x_2674_; lean_object* v___x_2675_; 
v___x_2673_ = 1;
v___x_2674_ = lean_box(v___x_2673_);
v___x_2675_ = lean_apply_2(v_toPure_2667_, lean_box(0), v___x_2674_);
return v___x_2675_;
}
else
{
uint8_t v___x_2676_; lean_object* v___x_2677_; lean_object* v___x_2678_; 
v___x_2676_ = 0;
v___x_2677_ = lean_box(v___x_2676_);
v___x_2678_ = lean_apply_2(v_toPure_2667_, lean_box(0), v___x_2677_);
return v___x_2678_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsFile___redArg___lam__0___boxed(lean_object* v___x_2679_, lean_object* v_toPure_2680_, lean_object* v_____do__lift_2681_){
_start:
{
lean_object* v_res_2682_; 
v_res_2682_ = lp_aesop_Aesop_enableStatsFile___redArg___lam__0(v___x_2679_, v_toPure_2680_, v_____do__lift_2681_);
lean_dec_ref(v_____do__lift_2681_);
return v_res_2682_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsFile___redArg(lean_object* v_inst_2683_, lean_object* v_inst_2684_){
_start:
{
lean_object* v___x_2685_; lean_object* v_toApplicative_2686_; lean_object* v_toBind_2687_; lean_object* v_toPure_2688_; lean_object* v___f_2689_; lean_object* v___x_2690_; 
v___x_2685_ = l_Lean_KVMap_instValueString;
v_toApplicative_2686_ = lean_ctor_get(v_inst_2683_, 0);
lean_inc_ref(v_toApplicative_2686_);
v_toBind_2687_ = lean_ctor_get(v_inst_2683_, 1);
lean_inc(v_toBind_2687_);
lean_dec_ref(v_inst_2683_);
v_toPure_2688_ = lean_ctor_get(v_toApplicative_2686_, 1);
lean_inc(v_toPure_2688_);
lean_dec_ref(v_toApplicative_2686_);
v___f_2689_ = lean_alloc_closure((void*)(lp_aesop_Aesop_enableStatsFile___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_2689_, 0, v___x_2685_);
lean_closure_set(v___f_2689_, 1, v_toPure_2688_);
v___x_2690_ = lean_apply_4(v_toBind_2687_, lean_box(0), lean_box(0), v_inst_2684_, v___f_2689_);
return v___x_2690_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStatsFile(lean_object* v_m_2691_, lean_object* v_inst_2692_, lean_object* v_inst_2693_){
_start:
{
lean_object* v___x_2694_; lean_object* v_toApplicative_2695_; lean_object* v_toBind_2696_; lean_object* v_toPure_2697_; lean_object* v___f_2698_; lean_object* v___x_2699_; 
v___x_2694_ = l_Lean_KVMap_instValueString;
v_toApplicative_2695_ = lean_ctor_get(v_inst_2692_, 0);
lean_inc_ref(v_toApplicative_2695_);
v_toBind_2696_ = lean_ctor_get(v_inst_2692_, 1);
lean_inc(v_toBind_2696_);
lean_dec_ref(v_inst_2692_);
v_toPure_2697_ = lean_ctor_get(v_toApplicative_2695_, 1);
lean_inc(v_toPure_2697_);
lean_dec_ref(v_toApplicative_2695_);
v___f_2698_ = lean_alloc_closure((void*)(lp_aesop_Aesop_enableStatsFile___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_2698_, 0, v___x_2694_);
lean_closure_set(v___f_2698_, 1, v_toPure_2697_);
v___x_2699_ = lean_apply_4(v_toBind_2696_, lean_box(0), lean_box(0), v_inst_2693_, v___f_2698_);
return v___x_2699_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStats___redArg___lam__1(lean_object* v___x_2700_, lean_object* v_toPure_2701_, uint8_t v_b_2702_, lean_object* v_____do__lift_2703_){
_start:
{
lean_object* v___x_2704_; lean_object* v___x_2705_; lean_object* v___x_2706_; uint8_t v___x_2707_; 
v___x_2704_ = lp_aesop_Aesop_aesop_stats_file;
v___x_2705_ = l_Lean_Option_get___redArg(v___x_2700_, v_____do__lift_2703_, v___x_2704_);
v___x_2706_ = ((lean_object*)(lp_aesop_Lean_addTrace___at___00Aesop_Stats_trace_spec__1___closed__0));
v___x_2707_ = lean_string_dec_eq(v___x_2705_, v___x_2706_);
lean_dec(v___x_2705_);
if (v___x_2707_ == 0)
{
uint8_t v___x_2708_; lean_object* v___x_2709_; lean_object* v___x_2710_; 
v___x_2708_ = 1;
v___x_2709_ = lean_box(v___x_2708_);
v___x_2710_ = lean_apply_2(v_toPure_2701_, lean_box(0), v___x_2709_);
return v___x_2710_;
}
else
{
lean_object* v___x_2711_; lean_object* v___x_2712_; 
v___x_2711_ = lean_box(v_b_2702_);
v___x_2712_ = lean_apply_2(v_toPure_2701_, lean_box(0), v___x_2711_);
return v___x_2712_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStats___redArg___lam__1___boxed(lean_object* v___x_2713_, lean_object* v_toPure_2714_, lean_object* v_b_2715_, lean_object* v_____do__lift_2716_){
_start:
{
uint8_t v_b_boxed_2717_; lean_object* v_res_2718_; 
v_b_boxed_2717_ = lean_unbox(v_b_2715_);
v_res_2718_ = lp_aesop_Aesop_enableStats___redArg___lam__1(v___x_2713_, v_toPure_2714_, v_b_boxed_2717_, v_____do__lift_2716_);
lean_dec_ref(v_____do__lift_2716_);
return v_res_2718_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStats___redArg___lam__0(lean_object* v_toPure_2719_, lean_object* v_toBind_2720_, lean_object* v_inst_2721_, uint8_t v_b_2722_){
_start:
{
if (v_b_2722_ == 0)
{
lean_object* v___x_2723_; lean_object* v___x_2724_; lean_object* v___f_2725_; lean_object* v___x_2726_; 
v___x_2723_ = l_Lean_KVMap_instValueString;
v___x_2724_ = lean_box(v_b_2722_);
v___f_2725_ = lean_alloc_closure((void*)(lp_aesop_Aesop_enableStats___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_2725_, 0, v___x_2723_);
lean_closure_set(v___f_2725_, 1, v_toPure_2719_);
lean_closure_set(v___f_2725_, 2, v___x_2724_);
v___x_2726_ = lean_apply_4(v_toBind_2720_, lean_box(0), lean_box(0), v_inst_2721_, v___f_2725_);
return v___x_2726_;
}
else
{
lean_object* v___x_2727_; lean_object* v___x_2728_; 
lean_dec(v_inst_2721_);
lean_dec(v_toBind_2720_);
v___x_2727_ = lean_box(v_b_2722_);
v___x_2728_ = lean_apply_2(v_toPure_2719_, lean_box(0), v___x_2727_);
return v___x_2728_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStats___redArg___lam__0___boxed(lean_object* v_toPure_2729_, lean_object* v_toBind_2730_, lean_object* v_inst_2731_, lean_object* v_b_2732_){
_start:
{
uint8_t v_b_boxed_2733_; lean_object* v_res_2734_; 
v_b_boxed_2733_ = lean_unbox(v_b_2732_);
v_res_2734_ = lp_aesop_Aesop_enableStats___redArg___lam__0(v_toPure_2729_, v_toBind_2730_, v_inst_2731_, v_b_boxed_2733_);
return v_res_2734_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStats___redArg___lam__2(lean_object* v_inst_2735_, lean_object* v_inst_2736_, lean_object* v_toBind_2737_, lean_object* v___f_2738_, lean_object* v_toPure_2739_, uint8_t v_b_2740_){
_start:
{
if (v_b_2740_ == 0)
{
lean_object* v___x_2741_; lean_object* v___x_2742_; lean_object* v___x_2743_; 
lean_dec(v_toPure_2739_);
v___x_2741_ = lp_aesop_Aesop_TraceOption_stats;
v___x_2742_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v_inst_2735_, v_inst_2736_, v___x_2741_);
v___x_2743_ = lean_apply_4(v_toBind_2737_, lean_box(0), lean_box(0), v___x_2742_, v___f_2738_);
return v___x_2743_;
}
else
{
lean_object* v___x_2744_; lean_object* v___x_2745_; 
lean_dec(v___f_2738_);
lean_dec(v_toBind_2737_);
lean_dec(v_inst_2736_);
lean_dec_ref(v_inst_2735_);
v___x_2744_ = lean_box(v_b_2740_);
v___x_2745_ = lean_apply_2(v_toPure_2739_, lean_box(0), v___x_2744_);
return v___x_2745_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStats___redArg___lam__2___boxed(lean_object* v_inst_2746_, lean_object* v_inst_2747_, lean_object* v_toBind_2748_, lean_object* v___f_2749_, lean_object* v_toPure_2750_, lean_object* v_b_2751_){
_start:
{
uint8_t v_b_boxed_2752_; lean_object* v_res_2753_; 
v_b_boxed_2752_ = lean_unbox(v_b_2751_);
v_res_2753_ = lp_aesop_Aesop_enableStats___redArg___lam__2(v_inst_2746_, v_inst_2747_, v_toBind_2748_, v___f_2749_, v_toPure_2750_, v_b_boxed_2752_);
return v_res_2753_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStats___redArg(lean_object* v_inst_2754_, lean_object* v_inst_2755_){
_start:
{
lean_object* v_toApplicative_2756_; lean_object* v_toBind_2757_; lean_object* v_toFunctor_2758_; lean_object* v_toPure_2759_; lean_object* v_map_2760_; lean_object* v___f_2761_; lean_object* v___f_2762_; lean_object* v___f_2763_; lean_object* v___x_2764_; lean_object* v___x_2765_; 
v_toApplicative_2756_ = lean_ctor_get(v_inst_2754_, 0);
v_toBind_2757_ = lean_ctor_get(v_inst_2754_, 1);
lean_inc_n(v_toBind_2757_, 3);
v_toFunctor_2758_ = lean_ctor_get(v_toApplicative_2756_, 0);
v_toPure_2759_ = lean_ctor_get(v_toApplicative_2756_, 1);
lean_inc_n(v_toPure_2759_, 2);
v_map_2760_ = lean_ctor_get(v_toFunctor_2758_, 0);
lean_inc(v_map_2760_);
v___f_2761_ = lean_obj_once(&lp_aesop_Aesop_enableStatsCollection___redArg___closed__0, &lp_aesop_Aesop_enableStatsCollection___redArg___closed__0_once, _init_lp_aesop_Aesop_enableStatsCollection___redArg___closed__0);
lean_inc_n(v_inst_2755_, 2);
v___f_2762_ = lean_alloc_closure((void*)(lp_aesop_Aesop_enableStats___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_2762_, 0, v_toPure_2759_);
lean_closure_set(v___f_2762_, 1, v_toBind_2757_);
lean_closure_set(v___f_2762_, 2, v_inst_2755_);
v___f_2763_ = lean_alloc_closure((void*)(lp_aesop_Aesop_enableStats___redArg___lam__2___boxed), 6, 5);
lean_closure_set(v___f_2763_, 0, v_inst_2754_);
lean_closure_set(v___f_2763_, 1, v_inst_2755_);
lean_closure_set(v___f_2763_, 2, v_toBind_2757_);
lean_closure_set(v___f_2763_, 3, v___f_2762_);
lean_closure_set(v___f_2763_, 4, v_toPure_2759_);
v___x_2764_ = lean_apply_4(v_map_2760_, lean_box(0), lean_box(0), v___f_2761_, v_inst_2755_);
v___x_2765_ = lean_apply_4(v_toBind_2757_, lean_box(0), lean_box(0), v___x_2764_, v___f_2763_);
return v___x_2765_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_enableStats(lean_object* v_m_2766_, lean_object* v_inst_2767_, lean_object* v_inst_2768_){
_start:
{
lean_object* v_toApplicative_2769_; lean_object* v_toBind_2770_; lean_object* v_toFunctor_2771_; lean_object* v_toPure_2772_; lean_object* v_map_2773_; lean_object* v___f_2774_; lean_object* v___f_2775_; lean_object* v___f_2776_; lean_object* v___x_2777_; lean_object* v___x_2778_; 
v_toApplicative_2769_ = lean_ctor_get(v_inst_2767_, 0);
v_toBind_2770_ = lean_ctor_get(v_inst_2767_, 1);
lean_inc_n(v_toBind_2770_, 3);
v_toFunctor_2771_ = lean_ctor_get(v_toApplicative_2769_, 0);
v_toPure_2772_ = lean_ctor_get(v_toApplicative_2769_, 1);
lean_inc_n(v_toPure_2772_, 2);
v_map_2773_ = lean_ctor_get(v_toFunctor_2771_, 0);
lean_inc(v_map_2773_);
v___f_2774_ = lean_obj_once(&lp_aesop_Aesop_enableStatsCollection___redArg___closed__0, &lp_aesop_Aesop_enableStatsCollection___redArg___closed__0_once, _init_lp_aesop_Aesop_enableStatsCollection___redArg___closed__0);
lean_inc_n(v_inst_2768_, 2);
v___f_2775_ = lean_alloc_closure((void*)(lp_aesop_Aesop_enableStats___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_2775_, 0, v_toPure_2772_);
lean_closure_set(v___f_2775_, 1, v_toBind_2770_);
lean_closure_set(v___f_2775_, 2, v_inst_2768_);
v___f_2776_ = lean_alloc_closure((void*)(lp_aesop_Aesop_enableStats___redArg___lam__2___boxed), 6, 5);
lean_closure_set(v___f_2776_, 0, v_inst_2767_);
lean_closure_set(v___f_2776_, 1, v_inst_2768_);
lean_closure_set(v___f_2776_, 2, v_toBind_2770_);
lean_closure_set(v___f_2776_, 3, v___f_2775_);
lean_closure_set(v___f_2776_, 4, v_toPure_2772_);
v___x_2777_ = lean_apply_4(v_map_2773_, lean_box(0), lean_box(0), v___f_2774_, v_inst_2768_);
v___x_2778_ = lean_apply_4(v_toBind_2770_, lean_box(0), lean_box(0), v___x_2777_, v___f_2776_);
return v___x_2778_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__0(lean_object* v_recordStats_2779_, lean_object* v_fst_2780_, lean_object* v_snd_2781_, lean_object* v_x_2782_){
_start:
{
lean_object* v___x_2783_; 
v___x_2783_ = lean_apply_3(v_recordStats_2779_, v_x_2782_, v_fst_2780_, v_snd_2781_);
return v___x_2783_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__1(lean_object* v_toPure_2784_, lean_object* v_fst_2785_, lean_object* v_____r_2786_){
_start:
{
lean_object* v___x_2787_; 
v___x_2787_ = lean_apply_2(v_toPure_2784_, lean_box(0), v_fst_2785_);
return v___x_2787_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__2(lean_object* v_recordStats_2788_, lean_object* v_toPure_2789_, lean_object* v_modifyStats_2790_, lean_object* v_toBind_2791_, lean_object* v_____x_2792_){
_start:
{
lean_object* v_fst_2793_; lean_object* v_snd_2794_; lean_object* v___f_2795_; lean_object* v___f_2796_; lean_object* v___x_2797_; lean_object* v___x_2798_; 
v_fst_2793_ = lean_ctor_get(v_____x_2792_, 0);
lean_inc_n(v_fst_2793_, 2);
v_snd_2794_ = lean_ctor_get(v_____x_2792_, 1);
lean_inc(v_snd_2794_);
lean_dec_ref(v_____x_2792_);
v___f_2795_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__0), 4, 3);
lean_closure_set(v___f_2795_, 0, v_recordStats_2788_);
lean_closure_set(v___f_2795_, 1, v_fst_2793_);
lean_closure_set(v___f_2795_, 2, v_snd_2794_);
v___f_2796_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__1), 3, 2);
lean_closure_set(v___f_2796_, 0, v_toPure_2789_);
lean_closure_set(v___f_2796_, 1, v_fst_2793_);
v___x_2797_ = lean_apply_1(v_modifyStats_2790_, v___f_2795_);
v___x_2798_ = lean_apply_4(v_toBind_2791_, lean_box(0), lean_box(0), v___x_2797_, v___f_2796_);
return v___x_2798_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__3(lean_object* v_start_2799_, lean_object* v_a_2800_, lean_object* v_toPure_2801_, lean_object* v_stop_2802_){
_start:
{
lean_object* v___x_2803_; lean_object* v___x_2804_; lean_object* v___x_2805_; 
v___x_2803_ = lean_nat_sub(v_stop_2802_, v_start_2799_);
v___x_2804_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2804_, 0, v_a_2800_);
lean_ctor_set(v___x_2804_, 1, v___x_2803_);
v___x_2805_ = lean_apply_2(v_toPure_2801_, lean_box(0), v___x_2804_);
return v___x_2805_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__3___boxed(lean_object* v_start_2806_, lean_object* v_a_2807_, lean_object* v_toPure_2808_, lean_object* v_stop_2809_){
_start:
{
lean_object* v_res_2810_; 
v_res_2810_ = lp_aesop_Aesop_profiling___redArg___lam__3(v_start_2806_, v_a_2807_, v_toPure_2808_, v_stop_2809_);
lean_dec(v_stop_2809_);
lean_dec(v_start_2806_);
return v_res_2810_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__4(lean_object* v_start_2811_, lean_object* v_toPure_2812_, lean_object* v_toBind_2813_, lean_object* v___x_2814_, lean_object* v_a_2815_){
_start:
{
lean_object* v___f_2816_; lean_object* v___x_2817_; 
v___f_2816_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__3___boxed), 4, 3);
lean_closure_set(v___f_2816_, 0, v_start_2811_);
lean_closure_set(v___f_2816_, 1, v_a_2815_);
lean_closure_set(v___f_2816_, 2, v_toPure_2812_);
v___x_2817_ = lean_apply_4(v_toBind_2813_, lean_box(0), lean_box(0), v___x_2814_, v___f_2816_);
return v___x_2817_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__5(lean_object* v_toPure_2818_, lean_object* v_toBind_2819_, lean_object* v___x_2820_, lean_object* v_x_2821_, lean_object* v_start_2822_){
_start:
{
lean_object* v___f_2823_; lean_object* v___x_2824_; 
lean_inc(v_toBind_2819_);
v___f_2823_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__4), 5, 4);
lean_closure_set(v___f_2823_, 0, v_start_2822_);
lean_closure_set(v___f_2823_, 1, v_toPure_2818_);
lean_closure_set(v___f_2823_, 2, v_toBind_2819_);
lean_closure_set(v___f_2823_, 3, v___x_2820_);
v___x_2824_ = lean_apply_4(v_toBind_2819_, lean_box(0), lean_box(0), v_x_2821_, v___f_2823_);
return v___x_2824_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__6(lean_object* v_x_2826_, lean_object* v_inst_2827_, lean_object* v_toPure_2828_, lean_object* v_toBind_2829_, lean_object* v___f_2830_, uint8_t v_____do__lift_2831_){
_start:
{
if (v_____do__lift_2831_ == 0)
{
lean_dec(v___f_2830_);
lean_dec(v_toBind_2829_);
lean_dec(v_toPure_2828_);
lean_dec(v_inst_2827_);
return v_x_2826_;
}
else
{
lean_object* v___x_2832_; lean_object* v___x_2833_; lean_object* v___f_2834_; lean_object* v___x_2835_; lean_object* v___x_2836_; 
v___x_2832_ = ((lean_object*)(lp_aesop_Aesop_profiling___redArg___lam__6___closed__0));
v___x_2833_ = lean_apply_2(v_inst_2827_, lean_box(0), v___x_2832_);
lean_inc(v___x_2833_);
lean_inc_n(v_toBind_2829_, 2);
v___f_2834_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__5), 5, 4);
lean_closure_set(v___f_2834_, 0, v_toPure_2828_);
lean_closure_set(v___f_2834_, 1, v_toBind_2829_);
lean_closure_set(v___f_2834_, 2, v___x_2833_);
lean_closure_set(v___f_2834_, 3, v_x_2826_);
v___x_2835_ = lean_apply_4(v_toBind_2829_, lean_box(0), lean_box(0), v___x_2833_, v___f_2834_);
v___x_2836_ = lean_apply_4(v_toBind_2829_, lean_box(0), lean_box(0), v___x_2835_, v___f_2830_);
return v___x_2836_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__6___boxed(lean_object* v_x_2837_, lean_object* v_inst_2838_, lean_object* v_toPure_2839_, lean_object* v_toBind_2840_, lean_object* v___f_2841_, lean_object* v_____do__lift_2842_){
_start:
{
uint8_t v_____do__lift_398__boxed_2843_; lean_object* v_res_2844_; 
v_____do__lift_398__boxed_2843_ = lean_unbox(v_____do__lift_2842_);
v_res_2844_ = lp_aesop_Aesop_profiling___redArg___lam__6(v_x_2837_, v_inst_2838_, v_toPure_2839_, v_toBind_2840_, v___f_2841_, v_____do__lift_398__boxed_2843_);
return v_res_2844_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__9(lean_object* v_toPure_2845_, lean_object* v_toBind_2846_, lean_object* v_toMonadOptions_2847_, uint8_t v_b_2848_){
_start:
{
if (v_b_2848_ == 0)
{
lean_object* v___x_2849_; lean_object* v___x_2850_; lean_object* v___f_2851_; lean_object* v___x_2852_; 
v___x_2849_ = l_Lean_KVMap_instValueString;
v___x_2850_ = lean_box(v_b_2848_);
v___f_2851_ = lean_alloc_closure((void*)(lp_aesop_Aesop_enableStats___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_2851_, 0, v___x_2849_);
lean_closure_set(v___f_2851_, 1, v_toPure_2845_);
lean_closure_set(v___f_2851_, 2, v___x_2850_);
v___x_2852_ = lean_apply_4(v_toBind_2846_, lean_box(0), lean_box(0), v_toMonadOptions_2847_, v___f_2851_);
return v___x_2852_;
}
else
{
lean_object* v___x_2853_; lean_object* v___x_2854_; 
lean_dec(v_toMonadOptions_2847_);
lean_dec(v_toBind_2846_);
v___x_2853_ = lean_box(v_b_2848_);
v___x_2854_ = lean_apply_2(v_toPure_2845_, lean_box(0), v___x_2853_);
return v___x_2854_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__9___boxed(lean_object* v_toPure_2855_, lean_object* v_toBind_2856_, lean_object* v_toMonadOptions_2857_, lean_object* v_b_2858_){
_start:
{
uint8_t v_b_boxed_2859_; lean_object* v_res_2860_; 
v_b_boxed_2859_ = lean_unbox(v_b_2858_);
v_res_2860_ = lp_aesop_Aesop_profiling___redArg___lam__9(v_toPure_2855_, v_toBind_2856_, v_toMonadOptions_2857_, v_b_boxed_2859_);
return v_res_2860_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__7(lean_object* v_inst_2861_, lean_object* v_toMonadOptions_2862_, lean_object* v_toBind_2863_, lean_object* v___f_2864_, lean_object* v_toPure_2865_, uint8_t v_b_2866_){
_start:
{
if (v_b_2866_ == 0)
{
lean_object* v___x_2867_; lean_object* v___x_2868_; lean_object* v___x_2869_; 
lean_dec(v_toPure_2865_);
v___x_2867_ = lp_aesop_Aesop_TraceOption_stats;
v___x_2868_ = lp_aesop_Aesop_TraceOption_isEnabled___redArg(v_inst_2861_, v_toMonadOptions_2862_, v___x_2867_);
v___x_2869_ = lean_apply_4(v_toBind_2863_, lean_box(0), lean_box(0), v___x_2868_, v___f_2864_);
return v___x_2869_;
}
else
{
lean_object* v___x_2870_; lean_object* v___x_2871_; 
lean_dec(v___f_2864_);
lean_dec(v_toBind_2863_);
lean_dec(v_toMonadOptions_2862_);
lean_dec_ref(v_inst_2861_);
v___x_2870_ = lean_box(v_b_2866_);
v___x_2871_ = lean_apply_2(v_toPure_2865_, lean_box(0), v___x_2870_);
return v___x_2871_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg___lam__7___boxed(lean_object* v_inst_2872_, lean_object* v_toMonadOptions_2873_, lean_object* v_toBind_2874_, lean_object* v___f_2875_, lean_object* v_toPure_2876_, lean_object* v_b_2877_){
_start:
{
uint8_t v_b_boxed_2878_; lean_object* v_res_2879_; 
v_b_boxed_2878_ = lean_unbox(v_b_2877_);
v_res_2879_ = lp_aesop_Aesop_profiling___redArg___lam__7(v_inst_2872_, v_toMonadOptions_2873_, v_toBind_2874_, v___f_2875_, v_toPure_2876_, v_b_boxed_2878_);
return v_res_2879_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling___redArg(lean_object* v_inst_2880_, lean_object* v_inst_2881_, lean_object* v_inst_2882_, lean_object* v_recordStats_2883_, lean_object* v_x_2884_){
_start:
{
lean_object* v_toApplicative_2885_; lean_object* v_toBind_2886_; lean_object* v_toMonadOptions_2887_; lean_object* v_modifyStats_2888_; lean_object* v_toFunctor_2889_; lean_object* v_toPure_2890_; lean_object* v_map_2891_; lean_object* v___f_2892_; lean_object* v___f_2893_; lean_object* v___f_2894_; lean_object* v___f_2895_; lean_object* v___f_2896_; lean_object* v___x_2897_; lean_object* v___x_2898_; lean_object* v___x_2899_; 
v_toApplicative_2885_ = lean_ctor_get(v_inst_2880_, 0);
v_toBind_2886_ = lean_ctor_get(v_inst_2880_, 1);
lean_inc_n(v_toBind_2886_, 6);
v_toMonadOptions_2887_ = lean_ctor_get(v_inst_2881_, 0);
lean_inc_n(v_toMonadOptions_2887_, 3);
v_modifyStats_2888_ = lean_ctor_get(v_inst_2881_, 3);
lean_inc(v_modifyStats_2888_);
lean_dec_ref(v_inst_2881_);
v_toFunctor_2889_ = lean_ctor_get(v_toApplicative_2885_, 0);
v_toPure_2890_ = lean_ctor_get(v_toApplicative_2885_, 1);
lean_inc_n(v_toPure_2890_, 4);
v_map_2891_ = lean_ctor_get(v_toFunctor_2889_, 0);
lean_inc(v_map_2891_);
v___f_2892_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__2), 5, 4);
lean_closure_set(v___f_2892_, 0, v_recordStats_2883_);
lean_closure_set(v___f_2892_, 1, v_toPure_2890_);
lean_closure_set(v___f_2892_, 2, v_modifyStats_2888_);
lean_closure_set(v___f_2892_, 3, v_toBind_2886_);
v___f_2893_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__6___boxed), 6, 5);
lean_closure_set(v___f_2893_, 0, v_x_2884_);
lean_closure_set(v___f_2893_, 1, v_inst_2882_);
lean_closure_set(v___f_2893_, 2, v_toPure_2890_);
lean_closure_set(v___f_2893_, 3, v_toBind_2886_);
lean_closure_set(v___f_2893_, 4, v___f_2892_);
v___f_2894_ = lean_obj_once(&lp_aesop_Aesop_enableStatsCollection___redArg___closed__0, &lp_aesop_Aesop_enableStatsCollection___redArg___closed__0_once, _init_lp_aesop_Aesop_enableStatsCollection___redArg___closed__0);
v___f_2895_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__9___boxed), 4, 3);
lean_closure_set(v___f_2895_, 0, v_toPure_2890_);
lean_closure_set(v___f_2895_, 1, v_toBind_2886_);
lean_closure_set(v___f_2895_, 2, v_toMonadOptions_2887_);
v___f_2896_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__7___boxed), 6, 5);
lean_closure_set(v___f_2896_, 0, v_inst_2880_);
lean_closure_set(v___f_2896_, 1, v_toMonadOptions_2887_);
lean_closure_set(v___f_2896_, 2, v_toBind_2886_);
lean_closure_set(v___f_2896_, 3, v___f_2895_);
lean_closure_set(v___f_2896_, 4, v_toPure_2890_);
v___x_2897_ = lean_apply_4(v_map_2891_, lean_box(0), lean_box(0), v___f_2894_, v_toMonadOptions_2887_);
v___x_2898_ = lean_apply_4(v_toBind_2886_, lean_box(0), lean_box(0), v___x_2897_, v___f_2896_);
v___x_2899_ = lean_apply_4(v_toBind_2886_, lean_box(0), lean_box(0), v___x_2898_, v___f_2893_);
return v___x_2899_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profiling(lean_object* v_m_2900_, lean_object* v_inst_2901_, lean_object* v_inst_2902_, lean_object* v_inst_2903_, lean_object* v_00_u03b1_2904_, lean_object* v_recordStats_2905_, lean_object* v_x_2906_){
_start:
{
lean_object* v_toApplicative_2907_; lean_object* v_toBind_2908_; lean_object* v_toMonadOptions_2909_; lean_object* v_modifyStats_2910_; lean_object* v_toFunctor_2911_; lean_object* v_toPure_2912_; lean_object* v_map_2913_; lean_object* v___f_2914_; lean_object* v___f_2915_; lean_object* v___f_2916_; lean_object* v___f_2917_; lean_object* v___f_2918_; lean_object* v___x_2919_; lean_object* v___x_2920_; lean_object* v___x_2921_; 
v_toApplicative_2907_ = lean_ctor_get(v_inst_2901_, 0);
v_toBind_2908_ = lean_ctor_get(v_inst_2901_, 1);
lean_inc_n(v_toBind_2908_, 6);
v_toMonadOptions_2909_ = lean_ctor_get(v_inst_2902_, 0);
lean_inc_n(v_toMonadOptions_2909_, 3);
v_modifyStats_2910_ = lean_ctor_get(v_inst_2902_, 3);
lean_inc(v_modifyStats_2910_);
lean_dec_ref(v_inst_2902_);
v_toFunctor_2911_ = lean_ctor_get(v_toApplicative_2907_, 0);
v_toPure_2912_ = lean_ctor_get(v_toApplicative_2907_, 1);
lean_inc_n(v_toPure_2912_, 4);
v_map_2913_ = lean_ctor_get(v_toFunctor_2911_, 0);
lean_inc(v_map_2913_);
v___f_2914_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__2), 5, 4);
lean_closure_set(v___f_2914_, 0, v_recordStats_2905_);
lean_closure_set(v___f_2914_, 1, v_toPure_2912_);
lean_closure_set(v___f_2914_, 2, v_modifyStats_2910_);
lean_closure_set(v___f_2914_, 3, v_toBind_2908_);
v___f_2915_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__6___boxed), 6, 5);
lean_closure_set(v___f_2915_, 0, v_x_2906_);
lean_closure_set(v___f_2915_, 1, v_inst_2903_);
lean_closure_set(v___f_2915_, 2, v_toPure_2912_);
lean_closure_set(v___f_2915_, 3, v_toBind_2908_);
lean_closure_set(v___f_2915_, 4, v___f_2914_);
v___f_2916_ = lean_obj_once(&lp_aesop_Aesop_enableStatsCollection___redArg___closed__0, &lp_aesop_Aesop_enableStatsCollection___redArg___closed__0_once, _init_lp_aesop_Aesop_enableStatsCollection___redArg___closed__0);
v___f_2917_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__9___boxed), 4, 3);
lean_closure_set(v___f_2917_, 0, v_toPure_2912_);
lean_closure_set(v___f_2917_, 1, v_toBind_2908_);
lean_closure_set(v___f_2917_, 2, v_toMonadOptions_2909_);
v___f_2918_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__7___boxed), 6, 5);
lean_closure_set(v___f_2918_, 0, v_inst_2901_);
lean_closure_set(v___f_2918_, 1, v_toMonadOptions_2909_);
lean_closure_set(v___f_2918_, 2, v_toBind_2908_);
lean_closure_set(v___f_2918_, 3, v___f_2917_);
lean_closure_set(v___f_2918_, 4, v_toPure_2912_);
v___x_2919_ = lean_apply_4(v_map_2913_, lean_box(0), lean_box(0), v___f_2916_, v_toMonadOptions_2909_);
v___x_2920_ = lean_apply_4(v_toBind_2908_, lean_box(0), lean_box(0), v___x_2919_, v___f_2918_);
v___x_2921_ = lean_apply_4(v_toBind_2908_, lean_box(0), lean_box(0), v___x_2920_, v___f_2915_);
return v___x_2921_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRuleSelection___redArg___lam__0(lean_object* v_snd_2922_, lean_object* v_x_2923_){
_start:
{
lean_object* v_total_2924_; lean_object* v_configParsing_2925_; lean_object* v_ruleSetConstruction_2926_; lean_object* v_search_2927_; lean_object* v_ruleSelection_2928_; lean_object* v_script_2929_; lean_object* v_forwardState_2930_; lean_object* v_scriptGenerated_2931_; lean_object* v_ruleStats_2932_; lean_object* v_goalStats_2933_; lean_object* v___x_2935_; uint8_t v_isShared_2936_; uint8_t v_isSharedCheck_2941_; 
v_total_2924_ = lean_ctor_get(v_x_2923_, 0);
v_configParsing_2925_ = lean_ctor_get(v_x_2923_, 1);
v_ruleSetConstruction_2926_ = lean_ctor_get(v_x_2923_, 2);
v_search_2927_ = lean_ctor_get(v_x_2923_, 3);
v_ruleSelection_2928_ = lean_ctor_get(v_x_2923_, 4);
v_script_2929_ = lean_ctor_get(v_x_2923_, 5);
v_forwardState_2930_ = lean_ctor_get(v_x_2923_, 6);
v_scriptGenerated_2931_ = lean_ctor_get(v_x_2923_, 7);
v_ruleStats_2932_ = lean_ctor_get(v_x_2923_, 8);
v_goalStats_2933_ = lean_ctor_get(v_x_2923_, 9);
v_isSharedCheck_2941_ = !lean_is_exclusive(v_x_2923_);
if (v_isSharedCheck_2941_ == 0)
{
v___x_2935_ = v_x_2923_;
v_isShared_2936_ = v_isSharedCheck_2941_;
goto v_resetjp_2934_;
}
else
{
lean_inc(v_goalStats_2933_);
lean_inc(v_ruleStats_2932_);
lean_inc(v_scriptGenerated_2931_);
lean_inc(v_forwardState_2930_);
lean_inc(v_script_2929_);
lean_inc(v_ruleSelection_2928_);
lean_inc(v_search_2927_);
lean_inc(v_ruleSetConstruction_2926_);
lean_inc(v_configParsing_2925_);
lean_inc(v_total_2924_);
lean_dec(v_x_2923_);
v___x_2935_ = lean_box(0);
v_isShared_2936_ = v_isSharedCheck_2941_;
goto v_resetjp_2934_;
}
v_resetjp_2934_:
{
lean_object* v___x_2937_; lean_object* v___x_2939_; 
v___x_2937_ = lean_nat_add(v_ruleSelection_2928_, v_snd_2922_);
lean_dec(v_ruleSelection_2928_);
if (v_isShared_2936_ == 0)
{
lean_ctor_set(v___x_2935_, 4, v___x_2937_);
v___x_2939_ = v___x_2935_;
goto v_reusejp_2938_;
}
else
{
lean_object* v_reuseFailAlloc_2940_; 
v_reuseFailAlloc_2940_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_2940_, 0, v_total_2924_);
lean_ctor_set(v_reuseFailAlloc_2940_, 1, v_configParsing_2925_);
lean_ctor_set(v_reuseFailAlloc_2940_, 2, v_ruleSetConstruction_2926_);
lean_ctor_set(v_reuseFailAlloc_2940_, 3, v_search_2927_);
lean_ctor_set(v_reuseFailAlloc_2940_, 4, v___x_2937_);
lean_ctor_set(v_reuseFailAlloc_2940_, 5, v_script_2929_);
lean_ctor_set(v_reuseFailAlloc_2940_, 6, v_forwardState_2930_);
lean_ctor_set(v_reuseFailAlloc_2940_, 7, v_scriptGenerated_2931_);
lean_ctor_set(v_reuseFailAlloc_2940_, 8, v_ruleStats_2932_);
lean_ctor_set(v_reuseFailAlloc_2940_, 9, v_goalStats_2933_);
v___x_2939_ = v_reuseFailAlloc_2940_;
goto v_reusejp_2938_;
}
v_reusejp_2938_:
{
return v___x_2939_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRuleSelection___redArg___lam__0___boxed(lean_object* v_snd_2942_, lean_object* v_x_2943_){
_start:
{
lean_object* v_res_2944_; 
v_res_2944_ = lp_aesop_Aesop_profilingRuleSelection___redArg___lam__0(v_snd_2942_, v_x_2943_);
lean_dec(v_snd_2942_);
return v_res_2944_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRuleSelection___redArg___lam__2(lean_object* v_toPure_2945_, lean_object* v_modifyStats_2946_, lean_object* v_toBind_2947_, lean_object* v_____x_2948_){
_start:
{
lean_object* v_fst_2949_; lean_object* v_snd_2950_; lean_object* v___f_2951_; lean_object* v___f_2952_; lean_object* v___x_2953_; lean_object* v___x_2954_; 
v_fst_2949_ = lean_ctor_get(v_____x_2948_, 0);
lean_inc(v_fst_2949_);
v_snd_2950_ = lean_ctor_get(v_____x_2948_, 1);
lean_inc(v_snd_2950_);
lean_dec_ref(v_____x_2948_);
v___f_2951_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profilingRuleSelection___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_2951_, 0, v_snd_2950_);
v___f_2952_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__1), 3, 2);
lean_closure_set(v___f_2952_, 0, v_toPure_2945_);
lean_closure_set(v___f_2952_, 1, v_fst_2949_);
v___x_2953_ = lean_apply_1(v_modifyStats_2946_, v___f_2951_);
v___x_2954_ = lean_apply_4(v_toBind_2947_, lean_box(0), lean_box(0), v___x_2953_, v___f_2952_);
return v___x_2954_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRuleSelection___redArg(lean_object* v_inst_2955_, lean_object* v_inst_2956_, lean_object* v_inst_2957_, lean_object* v_x_2958_){
_start:
{
lean_object* v_toApplicative_2959_; lean_object* v_toBind_2960_; lean_object* v_toMonadOptions_2961_; lean_object* v_modifyStats_2962_; lean_object* v_toFunctor_2963_; lean_object* v_toPure_2964_; lean_object* v_map_2965_; lean_object* v___f_2966_; lean_object* v___f_2967_; lean_object* v___f_2968_; lean_object* v___f_2969_; lean_object* v___f_2970_; lean_object* v___x_2971_; lean_object* v___x_2972_; lean_object* v___x_2973_; 
v_toApplicative_2959_ = lean_ctor_get(v_inst_2955_, 0);
v_toBind_2960_ = lean_ctor_get(v_inst_2955_, 1);
lean_inc_n(v_toBind_2960_, 6);
v_toMonadOptions_2961_ = lean_ctor_get(v_inst_2956_, 0);
lean_inc_n(v_toMonadOptions_2961_, 3);
v_modifyStats_2962_ = lean_ctor_get(v_inst_2956_, 3);
lean_inc(v_modifyStats_2962_);
lean_dec_ref(v_inst_2956_);
v_toFunctor_2963_ = lean_ctor_get(v_toApplicative_2959_, 0);
v_toPure_2964_ = lean_ctor_get(v_toApplicative_2959_, 1);
lean_inc_n(v_toPure_2964_, 4);
v_map_2965_ = lean_ctor_get(v_toFunctor_2963_, 0);
lean_inc(v_map_2965_);
v___f_2966_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profilingRuleSelection___redArg___lam__2), 4, 3);
lean_closure_set(v___f_2966_, 0, v_toPure_2964_);
lean_closure_set(v___f_2966_, 1, v_modifyStats_2962_);
lean_closure_set(v___f_2966_, 2, v_toBind_2960_);
v___f_2967_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__6___boxed), 6, 5);
lean_closure_set(v___f_2967_, 0, v_x_2958_);
lean_closure_set(v___f_2967_, 1, v_inst_2957_);
lean_closure_set(v___f_2967_, 2, v_toPure_2964_);
lean_closure_set(v___f_2967_, 3, v_toBind_2960_);
lean_closure_set(v___f_2967_, 4, v___f_2966_);
v___f_2968_ = lean_obj_once(&lp_aesop_Aesop_enableStatsCollection___redArg___closed__0, &lp_aesop_Aesop_enableStatsCollection___redArg___closed__0_once, _init_lp_aesop_Aesop_enableStatsCollection___redArg___closed__0);
v___f_2969_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__9___boxed), 4, 3);
lean_closure_set(v___f_2969_, 0, v_toPure_2964_);
lean_closure_set(v___f_2969_, 1, v_toBind_2960_);
lean_closure_set(v___f_2969_, 2, v_toMonadOptions_2961_);
v___f_2970_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__7___boxed), 6, 5);
lean_closure_set(v___f_2970_, 0, v_inst_2955_);
lean_closure_set(v___f_2970_, 1, v_toMonadOptions_2961_);
lean_closure_set(v___f_2970_, 2, v_toBind_2960_);
lean_closure_set(v___f_2970_, 3, v___f_2969_);
lean_closure_set(v___f_2970_, 4, v_toPure_2964_);
v___x_2971_ = lean_apply_4(v_map_2965_, lean_box(0), lean_box(0), v___f_2968_, v_toMonadOptions_2961_);
v___x_2972_ = lean_apply_4(v_toBind_2960_, lean_box(0), lean_box(0), v___x_2971_, v___f_2970_);
v___x_2973_ = lean_apply_4(v_toBind_2960_, lean_box(0), lean_box(0), v___x_2972_, v___f_2967_);
return v___x_2973_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRuleSelection(lean_object* v_m_2974_, lean_object* v_inst_2975_, lean_object* v_inst_2976_, lean_object* v_inst_2977_, lean_object* v_00_u03b1_2978_, lean_object* v_x_2979_){
_start:
{
lean_object* v_toApplicative_2980_; lean_object* v_toBind_2981_; lean_object* v_toMonadOptions_2982_; lean_object* v_modifyStats_2983_; lean_object* v_toFunctor_2984_; lean_object* v_toPure_2985_; lean_object* v_map_2986_; lean_object* v___f_2987_; lean_object* v___f_2988_; lean_object* v___f_2989_; lean_object* v___f_2990_; lean_object* v___f_2991_; lean_object* v___x_2992_; lean_object* v___x_2993_; lean_object* v___x_2994_; 
v_toApplicative_2980_ = lean_ctor_get(v_inst_2975_, 0);
v_toBind_2981_ = lean_ctor_get(v_inst_2975_, 1);
lean_inc_n(v_toBind_2981_, 6);
v_toMonadOptions_2982_ = lean_ctor_get(v_inst_2976_, 0);
lean_inc_n(v_toMonadOptions_2982_, 3);
v_modifyStats_2983_ = lean_ctor_get(v_inst_2976_, 3);
lean_inc(v_modifyStats_2983_);
lean_dec_ref(v_inst_2976_);
v_toFunctor_2984_ = lean_ctor_get(v_toApplicative_2980_, 0);
v_toPure_2985_ = lean_ctor_get(v_toApplicative_2980_, 1);
lean_inc_n(v_toPure_2985_, 4);
v_map_2986_ = lean_ctor_get(v_toFunctor_2984_, 0);
lean_inc(v_map_2986_);
v___f_2987_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profilingRuleSelection___redArg___lam__2), 4, 3);
lean_closure_set(v___f_2987_, 0, v_toPure_2985_);
lean_closure_set(v___f_2987_, 1, v_modifyStats_2983_);
lean_closure_set(v___f_2987_, 2, v_toBind_2981_);
v___f_2988_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__6___boxed), 6, 5);
lean_closure_set(v___f_2988_, 0, v_x_2979_);
lean_closure_set(v___f_2988_, 1, v_inst_2977_);
lean_closure_set(v___f_2988_, 2, v_toPure_2985_);
lean_closure_set(v___f_2988_, 3, v_toBind_2981_);
lean_closure_set(v___f_2988_, 4, v___f_2987_);
v___f_2989_ = lean_obj_once(&lp_aesop_Aesop_enableStatsCollection___redArg___closed__0, &lp_aesop_Aesop_enableStatsCollection___redArg___closed__0_once, _init_lp_aesop_Aesop_enableStatsCollection___redArg___closed__0);
v___f_2990_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__9___boxed), 4, 3);
lean_closure_set(v___f_2990_, 0, v_toPure_2985_);
lean_closure_set(v___f_2990_, 1, v_toBind_2981_);
lean_closure_set(v___f_2990_, 2, v_toMonadOptions_2982_);
v___f_2991_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__7___boxed), 6, 5);
lean_closure_set(v___f_2991_, 0, v_inst_2975_);
lean_closure_set(v___f_2991_, 1, v_toMonadOptions_2982_);
lean_closure_set(v___f_2991_, 2, v_toBind_2981_);
lean_closure_set(v___f_2991_, 3, v___f_2990_);
lean_closure_set(v___f_2991_, 4, v_toPure_2985_);
v___x_2992_ = lean_apply_4(v_map_2986_, lean_box(0), lean_box(0), v___f_2989_, v_toMonadOptions_2982_);
v___x_2993_ = lean_apply_4(v_toBind_2981_, lean_box(0), lean_box(0), v___x_2992_, v___f_2991_);
v___x_2994_ = lean_apply_4(v_toBind_2981_, lean_box(0), lean_box(0), v___x_2993_, v___f_2988_);
return v___x_2994_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRule___redArg___lam__1(lean_object* v_wasSuccessful_2995_, lean_object* v_fst_2996_, lean_object* v_rule_2997_, lean_object* v_snd_2998_, lean_object* v_x_2999_){
_start:
{
lean_object* v_total_3000_; lean_object* v_configParsing_3001_; lean_object* v_ruleSetConstruction_3002_; lean_object* v_search_3003_; lean_object* v_ruleSelection_3004_; lean_object* v_script_3005_; lean_object* v_forwardState_3006_; lean_object* v_scriptGenerated_3007_; lean_object* v_ruleStats_3008_; lean_object* v_goalStats_3009_; lean_object* v___x_3011_; uint8_t v_isShared_3012_; uint8_t v_isSharedCheck_3020_; 
v_total_3000_ = lean_ctor_get(v_x_2999_, 0);
v_configParsing_3001_ = lean_ctor_get(v_x_2999_, 1);
v_ruleSetConstruction_3002_ = lean_ctor_get(v_x_2999_, 2);
v_search_3003_ = lean_ctor_get(v_x_2999_, 3);
v_ruleSelection_3004_ = lean_ctor_get(v_x_2999_, 4);
v_script_3005_ = lean_ctor_get(v_x_2999_, 5);
v_forwardState_3006_ = lean_ctor_get(v_x_2999_, 6);
v_scriptGenerated_3007_ = lean_ctor_get(v_x_2999_, 7);
v_ruleStats_3008_ = lean_ctor_get(v_x_2999_, 8);
v_goalStats_3009_ = lean_ctor_get(v_x_2999_, 9);
v_isSharedCheck_3020_ = !lean_is_exclusive(v_x_2999_);
if (v_isSharedCheck_3020_ == 0)
{
v___x_3011_ = v_x_2999_;
v_isShared_3012_ = v_isSharedCheck_3020_;
goto v_resetjp_3010_;
}
else
{
lean_inc(v_goalStats_3009_);
lean_inc(v_ruleStats_3008_);
lean_inc(v_scriptGenerated_3007_);
lean_inc(v_forwardState_3006_);
lean_inc(v_script_3005_);
lean_inc(v_ruleSelection_3004_);
lean_inc(v_search_3003_);
lean_inc(v_ruleSetConstruction_3002_);
lean_inc(v_configParsing_3001_);
lean_inc(v_total_3000_);
lean_dec(v_x_2999_);
v___x_3011_ = lean_box(0);
v_isShared_3012_ = v_isSharedCheck_3020_;
goto v_resetjp_3010_;
}
v_resetjp_3010_:
{
lean_object* v___x_3013_; lean_object* v_rp_3014_; uint8_t v___x_3015_; lean_object* v___x_3016_; lean_object* v___x_3018_; 
v___x_3013_ = lean_apply_1(v_wasSuccessful_2995_, v_fst_2996_);
v_rp_3014_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v_rp_3014_, 0, v_rule_2997_);
lean_ctor_set(v_rp_3014_, 1, v_snd_2998_);
v___x_3015_ = lean_unbox(v___x_3013_);
lean_ctor_set_uint8(v_rp_3014_, sizeof(void*)*2, v___x_3015_);
v___x_3016_ = lean_array_push(v_ruleStats_3008_, v_rp_3014_);
if (v_isShared_3012_ == 0)
{
lean_ctor_set(v___x_3011_, 8, v___x_3016_);
v___x_3018_ = v___x_3011_;
goto v_reusejp_3017_;
}
else
{
lean_object* v_reuseFailAlloc_3019_; 
v_reuseFailAlloc_3019_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_3019_, 0, v_total_3000_);
lean_ctor_set(v_reuseFailAlloc_3019_, 1, v_configParsing_3001_);
lean_ctor_set(v_reuseFailAlloc_3019_, 2, v_ruleSetConstruction_3002_);
lean_ctor_set(v_reuseFailAlloc_3019_, 3, v_search_3003_);
lean_ctor_set(v_reuseFailAlloc_3019_, 4, v_ruleSelection_3004_);
lean_ctor_set(v_reuseFailAlloc_3019_, 5, v_script_3005_);
lean_ctor_set(v_reuseFailAlloc_3019_, 6, v_forwardState_3006_);
lean_ctor_set(v_reuseFailAlloc_3019_, 7, v_scriptGenerated_3007_);
lean_ctor_set(v_reuseFailAlloc_3019_, 8, v___x_3016_);
lean_ctor_set(v_reuseFailAlloc_3019_, 9, v_goalStats_3009_);
v___x_3018_ = v_reuseFailAlloc_3019_;
goto v_reusejp_3017_;
}
v_reusejp_3017_:
{
return v___x_3018_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRule___redArg___lam__0(lean_object* v_toPure_3021_, lean_object* v_wasSuccessful_3022_, lean_object* v_rule_3023_, lean_object* v_modifyStats_3024_, lean_object* v_toBind_3025_, lean_object* v_____x_3026_){
_start:
{
lean_object* v_fst_3027_; lean_object* v_snd_3028_; lean_object* v___f_3029_; lean_object* v___f_3030_; lean_object* v___x_3031_; lean_object* v___x_3032_; 
v_fst_3027_ = lean_ctor_get(v_____x_3026_, 0);
lean_inc_n(v_fst_3027_, 2);
v_snd_3028_ = lean_ctor_get(v_____x_3026_, 1);
lean_inc(v_snd_3028_);
lean_dec_ref(v_____x_3026_);
v___f_3029_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__1), 3, 2);
lean_closure_set(v___f_3029_, 0, v_toPure_3021_);
lean_closure_set(v___f_3029_, 1, v_fst_3027_);
v___f_3030_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profilingRule___redArg___lam__1), 5, 4);
lean_closure_set(v___f_3030_, 0, v_wasSuccessful_3022_);
lean_closure_set(v___f_3030_, 1, v_fst_3027_);
lean_closure_set(v___f_3030_, 2, v_rule_3023_);
lean_closure_set(v___f_3030_, 3, v_snd_3028_);
v___x_3031_ = lean_apply_1(v_modifyStats_3024_, v___f_3030_);
v___x_3032_ = lean_apply_4(v_toBind_3025_, lean_box(0), lean_box(0), v___x_3031_, v___f_3029_);
return v___x_3032_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRule___redArg(lean_object* v_inst_3033_, lean_object* v_inst_3034_, lean_object* v_inst_3035_, lean_object* v_rule_3036_, lean_object* v_wasSuccessful_3037_, lean_object* v_x_3038_){
_start:
{
lean_object* v_toApplicative_3039_; lean_object* v_toBind_3040_; lean_object* v_toMonadOptions_3041_; lean_object* v_modifyStats_3042_; lean_object* v_toFunctor_3043_; lean_object* v_toPure_3044_; lean_object* v_map_3045_; lean_object* v___f_3046_; lean_object* v___f_3047_; lean_object* v___f_3048_; lean_object* v___f_3049_; lean_object* v___f_3050_; lean_object* v___x_3051_; lean_object* v___x_3052_; lean_object* v___x_3053_; 
v_toApplicative_3039_ = lean_ctor_get(v_inst_3033_, 0);
v_toBind_3040_ = lean_ctor_get(v_inst_3033_, 1);
lean_inc_n(v_toBind_3040_, 6);
v_toMonadOptions_3041_ = lean_ctor_get(v_inst_3034_, 0);
lean_inc_n(v_toMonadOptions_3041_, 3);
v_modifyStats_3042_ = lean_ctor_get(v_inst_3034_, 3);
lean_inc(v_modifyStats_3042_);
lean_dec_ref(v_inst_3034_);
v_toFunctor_3043_ = lean_ctor_get(v_toApplicative_3039_, 0);
v_toPure_3044_ = lean_ctor_get(v_toApplicative_3039_, 1);
lean_inc_n(v_toPure_3044_, 4);
v_map_3045_ = lean_ctor_get(v_toFunctor_3043_, 0);
lean_inc(v_map_3045_);
v___f_3046_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profilingRule___redArg___lam__0), 6, 5);
lean_closure_set(v___f_3046_, 0, v_toPure_3044_);
lean_closure_set(v___f_3046_, 1, v_wasSuccessful_3037_);
lean_closure_set(v___f_3046_, 2, v_rule_3036_);
lean_closure_set(v___f_3046_, 3, v_modifyStats_3042_);
lean_closure_set(v___f_3046_, 4, v_toBind_3040_);
v___f_3047_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__6___boxed), 6, 5);
lean_closure_set(v___f_3047_, 0, v_x_3038_);
lean_closure_set(v___f_3047_, 1, v_inst_3035_);
lean_closure_set(v___f_3047_, 2, v_toPure_3044_);
lean_closure_set(v___f_3047_, 3, v_toBind_3040_);
lean_closure_set(v___f_3047_, 4, v___f_3046_);
v___f_3048_ = lean_obj_once(&lp_aesop_Aesop_enableStatsCollection___redArg___closed__0, &lp_aesop_Aesop_enableStatsCollection___redArg___closed__0_once, _init_lp_aesop_Aesop_enableStatsCollection___redArg___closed__0);
v___f_3049_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__9___boxed), 4, 3);
lean_closure_set(v___f_3049_, 0, v_toPure_3044_);
lean_closure_set(v___f_3049_, 1, v_toBind_3040_);
lean_closure_set(v___f_3049_, 2, v_toMonadOptions_3041_);
v___f_3050_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__7___boxed), 6, 5);
lean_closure_set(v___f_3050_, 0, v_inst_3033_);
lean_closure_set(v___f_3050_, 1, v_toMonadOptions_3041_);
lean_closure_set(v___f_3050_, 2, v_toBind_3040_);
lean_closure_set(v___f_3050_, 3, v___f_3049_);
lean_closure_set(v___f_3050_, 4, v_toPure_3044_);
v___x_3051_ = lean_apply_4(v_map_3045_, lean_box(0), lean_box(0), v___f_3048_, v_toMonadOptions_3041_);
v___x_3052_ = lean_apply_4(v_toBind_3040_, lean_box(0), lean_box(0), v___x_3051_, v___f_3050_);
v___x_3053_ = lean_apply_4(v_toBind_3040_, lean_box(0), lean_box(0), v___x_3052_, v___f_3047_);
return v___x_3053_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingRule(lean_object* v_m_3054_, lean_object* v_inst_3055_, lean_object* v_inst_3056_, lean_object* v_inst_3057_, lean_object* v_00_u03b1_3058_, lean_object* v_rule_3059_, lean_object* v_wasSuccessful_3060_, lean_object* v_x_3061_){
_start:
{
lean_object* v_toApplicative_3062_; lean_object* v_toBind_3063_; lean_object* v_toMonadOptions_3064_; lean_object* v_modifyStats_3065_; lean_object* v_toFunctor_3066_; lean_object* v_toPure_3067_; lean_object* v_map_3068_; lean_object* v___f_3069_; lean_object* v___f_3070_; lean_object* v___f_3071_; lean_object* v___f_3072_; lean_object* v___f_3073_; lean_object* v___x_3074_; lean_object* v___x_3075_; lean_object* v___x_3076_; 
v_toApplicative_3062_ = lean_ctor_get(v_inst_3055_, 0);
v_toBind_3063_ = lean_ctor_get(v_inst_3055_, 1);
lean_inc_n(v_toBind_3063_, 6);
v_toMonadOptions_3064_ = lean_ctor_get(v_inst_3056_, 0);
lean_inc_n(v_toMonadOptions_3064_, 3);
v_modifyStats_3065_ = lean_ctor_get(v_inst_3056_, 3);
lean_inc(v_modifyStats_3065_);
lean_dec_ref(v_inst_3056_);
v_toFunctor_3066_ = lean_ctor_get(v_toApplicative_3062_, 0);
v_toPure_3067_ = lean_ctor_get(v_toApplicative_3062_, 1);
lean_inc_n(v_toPure_3067_, 4);
v_map_3068_ = lean_ctor_get(v_toFunctor_3066_, 0);
lean_inc(v_map_3068_);
v___f_3069_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profilingRule___redArg___lam__0), 6, 5);
lean_closure_set(v___f_3069_, 0, v_toPure_3067_);
lean_closure_set(v___f_3069_, 1, v_wasSuccessful_3060_);
lean_closure_set(v___f_3069_, 2, v_rule_3059_);
lean_closure_set(v___f_3069_, 3, v_modifyStats_3065_);
lean_closure_set(v___f_3069_, 4, v_toBind_3063_);
v___f_3070_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__6___boxed), 6, 5);
lean_closure_set(v___f_3070_, 0, v_x_3061_);
lean_closure_set(v___f_3070_, 1, v_inst_3057_);
lean_closure_set(v___f_3070_, 2, v_toPure_3067_);
lean_closure_set(v___f_3070_, 3, v_toBind_3063_);
lean_closure_set(v___f_3070_, 4, v___f_3069_);
v___f_3071_ = lean_obj_once(&lp_aesop_Aesop_enableStatsCollection___redArg___closed__0, &lp_aesop_Aesop_enableStatsCollection___redArg___closed__0_once, _init_lp_aesop_Aesop_enableStatsCollection___redArg___closed__0);
v___f_3072_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__9___boxed), 4, 3);
lean_closure_set(v___f_3072_, 0, v_toPure_3067_);
lean_closure_set(v___f_3072_, 1, v_toBind_3063_);
lean_closure_set(v___f_3072_, 2, v_toMonadOptions_3064_);
v___f_3073_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__7___boxed), 6, 5);
lean_closure_set(v___f_3073_, 0, v_inst_3055_);
lean_closure_set(v___f_3073_, 1, v_toMonadOptions_3064_);
lean_closure_set(v___f_3073_, 2, v_toBind_3063_);
lean_closure_set(v___f_3073_, 3, v___f_3072_);
lean_closure_set(v___f_3073_, 4, v_toPure_3067_);
v___x_3074_ = lean_apply_4(v_map_3068_, lean_box(0), lean_box(0), v___f_3071_, v_toMonadOptions_3064_);
v___x_3075_ = lean_apply_4(v_toBind_3063_, lean_box(0), lean_box(0), v___x_3074_, v___f_3073_);
v___x_3076_ = lean_apply_4(v_toBind_3063_, lean_box(0), lean_box(0), v___x_3075_, v___f_3070_);
return v___x_3076_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingForwardState___redArg___lam__0(lean_object* v_snd_3077_, lean_object* v_x_3078_){
_start:
{
lean_object* v_total_3079_; lean_object* v_configParsing_3080_; lean_object* v_ruleSetConstruction_3081_; lean_object* v_search_3082_; lean_object* v_ruleSelection_3083_; lean_object* v_script_3084_; lean_object* v_forwardState_3085_; lean_object* v_scriptGenerated_3086_; lean_object* v_ruleStats_3087_; lean_object* v_goalStats_3088_; lean_object* v___x_3090_; uint8_t v_isShared_3091_; uint8_t v_isSharedCheck_3096_; 
v_total_3079_ = lean_ctor_get(v_x_3078_, 0);
v_configParsing_3080_ = lean_ctor_get(v_x_3078_, 1);
v_ruleSetConstruction_3081_ = lean_ctor_get(v_x_3078_, 2);
v_search_3082_ = lean_ctor_get(v_x_3078_, 3);
v_ruleSelection_3083_ = lean_ctor_get(v_x_3078_, 4);
v_script_3084_ = lean_ctor_get(v_x_3078_, 5);
v_forwardState_3085_ = lean_ctor_get(v_x_3078_, 6);
v_scriptGenerated_3086_ = lean_ctor_get(v_x_3078_, 7);
v_ruleStats_3087_ = lean_ctor_get(v_x_3078_, 8);
v_goalStats_3088_ = lean_ctor_get(v_x_3078_, 9);
v_isSharedCheck_3096_ = !lean_is_exclusive(v_x_3078_);
if (v_isSharedCheck_3096_ == 0)
{
v___x_3090_ = v_x_3078_;
v_isShared_3091_ = v_isSharedCheck_3096_;
goto v_resetjp_3089_;
}
else
{
lean_inc(v_goalStats_3088_);
lean_inc(v_ruleStats_3087_);
lean_inc(v_scriptGenerated_3086_);
lean_inc(v_forwardState_3085_);
lean_inc(v_script_3084_);
lean_inc(v_ruleSelection_3083_);
lean_inc(v_search_3082_);
lean_inc(v_ruleSetConstruction_3081_);
lean_inc(v_configParsing_3080_);
lean_inc(v_total_3079_);
lean_dec(v_x_3078_);
v___x_3090_ = lean_box(0);
v_isShared_3091_ = v_isSharedCheck_3096_;
goto v_resetjp_3089_;
}
v_resetjp_3089_:
{
lean_object* v___x_3092_; lean_object* v___x_3094_; 
v___x_3092_ = lean_nat_add(v_forwardState_3085_, v_snd_3077_);
lean_dec(v_forwardState_3085_);
if (v_isShared_3091_ == 0)
{
lean_ctor_set(v___x_3090_, 6, v___x_3092_);
v___x_3094_ = v___x_3090_;
goto v_reusejp_3093_;
}
else
{
lean_object* v_reuseFailAlloc_3095_; 
v_reuseFailAlloc_3095_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_3095_, 0, v_total_3079_);
lean_ctor_set(v_reuseFailAlloc_3095_, 1, v_configParsing_3080_);
lean_ctor_set(v_reuseFailAlloc_3095_, 2, v_ruleSetConstruction_3081_);
lean_ctor_set(v_reuseFailAlloc_3095_, 3, v_search_3082_);
lean_ctor_set(v_reuseFailAlloc_3095_, 4, v_ruleSelection_3083_);
lean_ctor_set(v_reuseFailAlloc_3095_, 5, v_script_3084_);
lean_ctor_set(v_reuseFailAlloc_3095_, 6, v___x_3092_);
lean_ctor_set(v_reuseFailAlloc_3095_, 7, v_scriptGenerated_3086_);
lean_ctor_set(v_reuseFailAlloc_3095_, 8, v_ruleStats_3087_);
lean_ctor_set(v_reuseFailAlloc_3095_, 9, v_goalStats_3088_);
v___x_3094_ = v_reuseFailAlloc_3095_;
goto v_reusejp_3093_;
}
v_reusejp_3093_:
{
return v___x_3094_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingForwardState___redArg___lam__0___boxed(lean_object* v_snd_3097_, lean_object* v_x_3098_){
_start:
{
lean_object* v_res_3099_; 
v_res_3099_ = lp_aesop_Aesop_profilingForwardState___redArg___lam__0(v_snd_3097_, v_x_3098_);
lean_dec(v_snd_3097_);
return v_res_3099_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingForwardState___redArg___lam__2(lean_object* v_toPure_3100_, lean_object* v_modifyStats_3101_, lean_object* v_toBind_3102_, lean_object* v_____x_3103_){
_start:
{
lean_object* v_fst_3104_; lean_object* v_snd_3105_; lean_object* v___f_3106_; lean_object* v___f_3107_; lean_object* v___x_3108_; lean_object* v___x_3109_; 
v_fst_3104_ = lean_ctor_get(v_____x_3103_, 0);
lean_inc(v_fst_3104_);
v_snd_3105_ = lean_ctor_get(v_____x_3103_, 1);
lean_inc(v_snd_3105_);
lean_dec_ref(v_____x_3103_);
v___f_3106_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profilingForwardState___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_3106_, 0, v_snd_3105_);
v___f_3107_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__1), 3, 2);
lean_closure_set(v___f_3107_, 0, v_toPure_3100_);
lean_closure_set(v___f_3107_, 1, v_fst_3104_);
v___x_3108_ = lean_apply_1(v_modifyStats_3101_, v___f_3106_);
v___x_3109_ = lean_apply_4(v_toBind_3102_, lean_box(0), lean_box(0), v___x_3108_, v___f_3107_);
return v___x_3109_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingForwardState___redArg(lean_object* v_inst_3110_, lean_object* v_inst_3111_, lean_object* v_inst_3112_, lean_object* v_x_3113_){
_start:
{
lean_object* v_toApplicative_3114_; lean_object* v_toBind_3115_; lean_object* v_toMonadOptions_3116_; lean_object* v_modifyStats_3117_; lean_object* v_toFunctor_3118_; lean_object* v_toPure_3119_; lean_object* v_map_3120_; lean_object* v___f_3121_; lean_object* v___f_3122_; lean_object* v___f_3123_; lean_object* v___f_3124_; lean_object* v___f_3125_; lean_object* v___x_3126_; lean_object* v___x_3127_; lean_object* v___x_3128_; 
v_toApplicative_3114_ = lean_ctor_get(v_inst_3110_, 0);
v_toBind_3115_ = lean_ctor_get(v_inst_3110_, 1);
lean_inc_n(v_toBind_3115_, 6);
v_toMonadOptions_3116_ = lean_ctor_get(v_inst_3111_, 0);
lean_inc_n(v_toMonadOptions_3116_, 3);
v_modifyStats_3117_ = lean_ctor_get(v_inst_3111_, 3);
lean_inc(v_modifyStats_3117_);
lean_dec_ref(v_inst_3111_);
v_toFunctor_3118_ = lean_ctor_get(v_toApplicative_3114_, 0);
v_toPure_3119_ = lean_ctor_get(v_toApplicative_3114_, 1);
lean_inc_n(v_toPure_3119_, 4);
v_map_3120_ = lean_ctor_get(v_toFunctor_3118_, 0);
lean_inc(v_map_3120_);
v___f_3121_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profilingForwardState___redArg___lam__2), 4, 3);
lean_closure_set(v___f_3121_, 0, v_toPure_3119_);
lean_closure_set(v___f_3121_, 1, v_modifyStats_3117_);
lean_closure_set(v___f_3121_, 2, v_toBind_3115_);
v___f_3122_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__6___boxed), 6, 5);
lean_closure_set(v___f_3122_, 0, v_x_3113_);
lean_closure_set(v___f_3122_, 1, v_inst_3112_);
lean_closure_set(v___f_3122_, 2, v_toPure_3119_);
lean_closure_set(v___f_3122_, 3, v_toBind_3115_);
lean_closure_set(v___f_3122_, 4, v___f_3121_);
v___f_3123_ = lean_obj_once(&lp_aesop_Aesop_enableStatsCollection___redArg___closed__0, &lp_aesop_Aesop_enableStatsCollection___redArg___closed__0_once, _init_lp_aesop_Aesop_enableStatsCollection___redArg___closed__0);
v___f_3124_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__9___boxed), 4, 3);
lean_closure_set(v___f_3124_, 0, v_toPure_3119_);
lean_closure_set(v___f_3124_, 1, v_toBind_3115_);
lean_closure_set(v___f_3124_, 2, v_toMonadOptions_3116_);
v___f_3125_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__7___boxed), 6, 5);
lean_closure_set(v___f_3125_, 0, v_inst_3110_);
lean_closure_set(v___f_3125_, 1, v_toMonadOptions_3116_);
lean_closure_set(v___f_3125_, 2, v_toBind_3115_);
lean_closure_set(v___f_3125_, 3, v___f_3124_);
lean_closure_set(v___f_3125_, 4, v_toPure_3119_);
v___x_3126_ = lean_apply_4(v_map_3120_, lean_box(0), lean_box(0), v___f_3123_, v_toMonadOptions_3116_);
v___x_3127_ = lean_apply_4(v_toBind_3115_, lean_box(0), lean_box(0), v___x_3126_, v___f_3125_);
v___x_3128_ = lean_apply_4(v_toBind_3115_, lean_box(0), lean_box(0), v___x_3127_, v___f_3122_);
return v___x_3128_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_profilingForwardState(lean_object* v_m_3129_, lean_object* v_inst_3130_, lean_object* v_inst_3131_, lean_object* v_inst_3132_, lean_object* v_00_u03b1_3133_, lean_object* v_x_3134_){
_start:
{
lean_object* v_toApplicative_3135_; lean_object* v_toBind_3136_; lean_object* v_toMonadOptions_3137_; lean_object* v_modifyStats_3138_; lean_object* v_toFunctor_3139_; lean_object* v_toPure_3140_; lean_object* v_map_3141_; lean_object* v___f_3142_; lean_object* v___f_3143_; lean_object* v___f_3144_; lean_object* v___f_3145_; lean_object* v___f_3146_; lean_object* v___x_3147_; lean_object* v___x_3148_; lean_object* v___x_3149_; 
v_toApplicative_3135_ = lean_ctor_get(v_inst_3130_, 0);
v_toBind_3136_ = lean_ctor_get(v_inst_3130_, 1);
lean_inc_n(v_toBind_3136_, 6);
v_toMonadOptions_3137_ = lean_ctor_get(v_inst_3131_, 0);
lean_inc_n(v_toMonadOptions_3137_, 3);
v_modifyStats_3138_ = lean_ctor_get(v_inst_3131_, 3);
lean_inc(v_modifyStats_3138_);
lean_dec_ref(v_inst_3131_);
v_toFunctor_3139_ = lean_ctor_get(v_toApplicative_3135_, 0);
v_toPure_3140_ = lean_ctor_get(v_toApplicative_3135_, 1);
lean_inc_n(v_toPure_3140_, 4);
v_map_3141_ = lean_ctor_get(v_toFunctor_3139_, 0);
lean_inc(v_map_3141_);
v___f_3142_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profilingForwardState___redArg___lam__2), 4, 3);
lean_closure_set(v___f_3142_, 0, v_toPure_3140_);
lean_closure_set(v___f_3142_, 1, v_modifyStats_3138_);
lean_closure_set(v___f_3142_, 2, v_toBind_3136_);
v___f_3143_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__6___boxed), 6, 5);
lean_closure_set(v___f_3143_, 0, v_x_3134_);
lean_closure_set(v___f_3143_, 1, v_inst_3132_);
lean_closure_set(v___f_3143_, 2, v_toPure_3140_);
lean_closure_set(v___f_3143_, 3, v_toBind_3136_);
lean_closure_set(v___f_3143_, 4, v___f_3142_);
v___f_3144_ = lean_obj_once(&lp_aesop_Aesop_enableStatsCollection___redArg___closed__0, &lp_aesop_Aesop_enableStatsCollection___redArg___closed__0_once, _init_lp_aesop_Aesop_enableStatsCollection___redArg___closed__0);
v___f_3145_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__9___boxed), 4, 3);
lean_closure_set(v___f_3145_, 0, v_toPure_3140_);
lean_closure_set(v___f_3145_, 1, v_toBind_3136_);
lean_closure_set(v___f_3145_, 2, v_toMonadOptions_3137_);
v___f_3146_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__7___boxed), 6, 5);
lean_closure_set(v___f_3146_, 0, v_inst_3130_);
lean_closure_set(v___f_3146_, 1, v_toMonadOptions_3137_);
lean_closure_set(v___f_3146_, 2, v_toBind_3136_);
lean_closure_set(v___f_3146_, 3, v___f_3145_);
lean_closure_set(v___f_3146_, 4, v_toPure_3140_);
v___x_3147_ = lean_apply_4(v_map_3141_, lean_box(0), lean_box(0), v___f_3144_, v_toMonadOptions_3137_);
v___x_3148_ = lean_apply_4(v_toBind_3136_, lean_box(0), lean_box(0), v___x_3147_, v___f_3146_);
v___x_3149_ = lean_apply_4(v_toBind_3136_, lean_box(0), lean_box(0), v___x_3148_, v___f_3143_);
return v___x_3149_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyStatsIfEnabled___redArg___lam__0(lean_object* v_toPure_3150_, lean_object* v_modifyStats_3151_, lean_object* v_f_3152_, uint8_t v_____do__lift_3153_){
_start:
{
if (v_____do__lift_3153_ == 0)
{
lean_object* v___x_3154_; lean_object* v___x_3155_; 
lean_dec_ref(v_f_3152_);
lean_dec(v_modifyStats_3151_);
v___x_3154_ = lean_box(0);
v___x_3155_ = lean_apply_2(v_toPure_3150_, lean_box(0), v___x_3154_);
return v___x_3155_;
}
else
{
lean_object* v___x_3156_; 
lean_dec(v_toPure_3150_);
v___x_3156_ = lean_apply_1(v_modifyStats_3151_, v_f_3152_);
return v___x_3156_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyStatsIfEnabled___redArg___lam__0___boxed(lean_object* v_toPure_3157_, lean_object* v_modifyStats_3158_, lean_object* v_f_3159_, lean_object* v_____do__lift_3160_){
_start:
{
uint8_t v_____do__lift_113__boxed_3161_; lean_object* v_res_3162_; 
v_____do__lift_113__boxed_3161_ = lean_unbox(v_____do__lift_3160_);
v_res_3162_ = lp_aesop_Aesop_modifyStatsIfEnabled___redArg___lam__0(v_toPure_3157_, v_modifyStats_3158_, v_f_3159_, v_____do__lift_113__boxed_3161_);
return v_res_3162_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyStatsIfEnabled___redArg(lean_object* v_inst_3163_, lean_object* v_inst_3164_, lean_object* v_f_3165_){
_start:
{
lean_object* v_toApplicative_3166_; lean_object* v_toBind_3167_; lean_object* v_toMonadOptions_3168_; lean_object* v_modifyStats_3169_; lean_object* v_toFunctor_3170_; lean_object* v_toPure_3171_; lean_object* v_map_3172_; lean_object* v___f_3173_; lean_object* v___f_3174_; lean_object* v___f_3175_; lean_object* v___f_3176_; lean_object* v___x_3177_; lean_object* v___x_3178_; lean_object* v___x_3179_; 
v_toApplicative_3166_ = lean_ctor_get(v_inst_3163_, 0);
v_toBind_3167_ = lean_ctor_get(v_inst_3163_, 1);
lean_inc_n(v_toBind_3167_, 4);
v_toMonadOptions_3168_ = lean_ctor_get(v_inst_3164_, 0);
lean_inc_n(v_toMonadOptions_3168_, 3);
v_modifyStats_3169_ = lean_ctor_get(v_inst_3164_, 3);
lean_inc(v_modifyStats_3169_);
lean_dec_ref(v_inst_3164_);
v_toFunctor_3170_ = lean_ctor_get(v_toApplicative_3166_, 0);
v_toPure_3171_ = lean_ctor_get(v_toApplicative_3166_, 1);
lean_inc_n(v_toPure_3171_, 3);
v_map_3172_ = lean_ctor_get(v_toFunctor_3170_, 0);
lean_inc(v_map_3172_);
v___f_3173_ = lean_alloc_closure((void*)(lp_aesop_Aesop_modifyStatsIfEnabled___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_3173_, 0, v_toPure_3171_);
lean_closure_set(v___f_3173_, 1, v_modifyStats_3169_);
lean_closure_set(v___f_3173_, 2, v_f_3165_);
v___f_3174_ = lean_obj_once(&lp_aesop_Aesop_enableStatsCollection___redArg___closed__0, &lp_aesop_Aesop_enableStatsCollection___redArg___closed__0_once, _init_lp_aesop_Aesop_enableStatsCollection___redArg___closed__0);
v___f_3175_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__9___boxed), 4, 3);
lean_closure_set(v___f_3175_, 0, v_toPure_3171_);
lean_closure_set(v___f_3175_, 1, v_toBind_3167_);
lean_closure_set(v___f_3175_, 2, v_toMonadOptions_3168_);
v___f_3176_ = lean_alloc_closure((void*)(lp_aesop_Aesop_profiling___redArg___lam__7___boxed), 6, 5);
lean_closure_set(v___f_3176_, 0, v_inst_3163_);
lean_closure_set(v___f_3176_, 1, v_toMonadOptions_3168_);
lean_closure_set(v___f_3176_, 2, v_toBind_3167_);
lean_closure_set(v___f_3176_, 3, v___f_3175_);
lean_closure_set(v___f_3176_, 4, v_toPure_3171_);
v___x_3177_ = lean_apply_4(v_map_3172_, lean_box(0), lean_box(0), v___f_3174_, v_toMonadOptions_3168_);
v___x_3178_ = lean_apply_4(v_toBind_3167_, lean_box(0), lean_box(0), v___x_3177_, v___f_3176_);
v___x_3179_ = lean_apply_4(v_toBind_3167_, lean_box(0), lean_box(0), v___x_3178_, v___f_3173_);
return v___x_3179_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_modifyStatsIfEnabled(lean_object* v_m_3180_, lean_object* v_inst_3181_, lean_object* v_inst_3182_, lean_object* v_f_3183_){
_start:
{
lean_object* v___x_3184_; 
v___x_3184_ = lp_aesop_Aesop_modifyStatsIfEnabled___redArg(v_inst_3181_, v_inst_3182_, v_f_3183_);
return v___x_3184_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptGenerated___redArg___lam__0(lean_object* v_x_3185_, lean_object* v_x_3186_){
_start:
{
lean_object* v_total_3187_; lean_object* v_configParsing_3188_; lean_object* v_ruleSetConstruction_3189_; lean_object* v_search_3190_; lean_object* v_ruleSelection_3191_; lean_object* v_script_3192_; lean_object* v_forwardState_3193_; lean_object* v_ruleStats_3194_; lean_object* v_goalStats_3195_; lean_object* v___x_3197_; uint8_t v_isShared_3198_; uint8_t v_isSharedCheck_3203_; 
v_total_3187_ = lean_ctor_get(v_x_3186_, 0);
v_configParsing_3188_ = lean_ctor_get(v_x_3186_, 1);
v_ruleSetConstruction_3189_ = lean_ctor_get(v_x_3186_, 2);
v_search_3190_ = lean_ctor_get(v_x_3186_, 3);
v_ruleSelection_3191_ = lean_ctor_get(v_x_3186_, 4);
v_script_3192_ = lean_ctor_get(v_x_3186_, 5);
v_forwardState_3193_ = lean_ctor_get(v_x_3186_, 6);
v_ruleStats_3194_ = lean_ctor_get(v_x_3186_, 8);
v_goalStats_3195_ = lean_ctor_get(v_x_3186_, 9);
v_isSharedCheck_3203_ = !lean_is_exclusive(v_x_3186_);
if (v_isSharedCheck_3203_ == 0)
{
lean_object* v_unused_3204_; 
v_unused_3204_ = lean_ctor_get(v_x_3186_, 7);
lean_dec(v_unused_3204_);
v___x_3197_ = v_x_3186_;
v_isShared_3198_ = v_isSharedCheck_3203_;
goto v_resetjp_3196_;
}
else
{
lean_inc(v_goalStats_3195_);
lean_inc(v_ruleStats_3194_);
lean_inc(v_forwardState_3193_);
lean_inc(v_script_3192_);
lean_inc(v_ruleSelection_3191_);
lean_inc(v_search_3190_);
lean_inc(v_ruleSetConstruction_3189_);
lean_inc(v_configParsing_3188_);
lean_inc(v_total_3187_);
lean_dec(v_x_3186_);
v___x_3197_ = lean_box(0);
v_isShared_3198_ = v_isSharedCheck_3203_;
goto v_resetjp_3196_;
}
v_resetjp_3196_:
{
lean_object* v___x_3199_; lean_object* v___x_3201_; 
v___x_3199_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_3199_, 0, v_x_3185_);
if (v_isShared_3198_ == 0)
{
lean_ctor_set(v___x_3197_, 7, v___x_3199_);
v___x_3201_ = v___x_3197_;
goto v_reusejp_3200_;
}
else
{
lean_object* v_reuseFailAlloc_3202_; 
v_reuseFailAlloc_3202_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_3202_, 0, v_total_3187_);
lean_ctor_set(v_reuseFailAlloc_3202_, 1, v_configParsing_3188_);
lean_ctor_set(v_reuseFailAlloc_3202_, 2, v_ruleSetConstruction_3189_);
lean_ctor_set(v_reuseFailAlloc_3202_, 3, v_search_3190_);
lean_ctor_set(v_reuseFailAlloc_3202_, 4, v_ruleSelection_3191_);
lean_ctor_set(v_reuseFailAlloc_3202_, 5, v_script_3192_);
lean_ctor_set(v_reuseFailAlloc_3202_, 6, v_forwardState_3193_);
lean_ctor_set(v_reuseFailAlloc_3202_, 7, v___x_3199_);
lean_ctor_set(v_reuseFailAlloc_3202_, 8, v_ruleStats_3194_);
lean_ctor_set(v_reuseFailAlloc_3202_, 9, v_goalStats_3195_);
v___x_3201_ = v_reuseFailAlloc_3202_;
goto v_reusejp_3200_;
}
v_reusejp_3200_:
{
return v___x_3201_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptGenerated___redArg(lean_object* v_inst_3205_, lean_object* v_inst_3206_, lean_object* v_x_3207_){
_start:
{
lean_object* v___f_3208_; lean_object* v___x_3209_; 
v___f_3208_ = lean_alloc_closure((void*)(lp_aesop_Aesop_recordScriptGenerated___redArg___lam__0), 2, 1);
lean_closure_set(v___f_3208_, 0, v_x_3207_);
v___x_3209_ = lp_aesop_Aesop_modifyStatsIfEnabled___redArg(v_inst_3205_, v_inst_3206_, v___f_3208_);
return v___x_3209_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordScriptGenerated(lean_object* v_m_3210_, lean_object* v_inst_3211_, lean_object* v_inst_3212_, lean_object* v_x_3213_){
_start:
{
lean_object* v___x_3214_; 
v___x_3214_ = lp_aesop_Aesop_recordScriptGenerated___redArg(v_inst_3211_, v_inst_3212_, v_x_3213_);
return v___x_3214_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Rule_Name(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Tracing(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Options_Public(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Stats_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Rule_Name(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Tracing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Options_Public(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_instInhabitedForwardRuleStateStats_default = _init_lp_aesop_Aesop_instInhabitedForwardRuleStateStats_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedForwardRuleStateStats_default);
lp_aesop_Aesop_instInhabitedForwardRuleStateStats = _init_lp_aesop_Aesop_instInhabitedForwardRuleStateStats();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedForwardRuleStateStats);
lp_aesop_Aesop_instInhabitedGoalKind_default = _init_lp_aesop_Aesop_instInhabitedGoalKind_default();
lp_aesop_Aesop_instInhabitedGoalKind = _init_lp_aesop_Aesop_instInhabitedGoalKind();
lp_aesop_Aesop_instInhabitedRuleStats_default = _init_lp_aesop_Aesop_instInhabitedRuleStats_default();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRuleStats_default);
lp_aesop_Aesop_instInhabitedRuleStats = _init_lp_aesop_Aesop_instInhabitedRuleStats();
lean_mark_persistent(lp_aesop_Aesop_instInhabitedRuleStats);
lp_aesop_Aesop_ScriptGenerated_instInhabitedMethod_default = _init_lp_aesop_Aesop_ScriptGenerated_instInhabitedMethod_default();
lp_aesop_Aesop_ScriptGenerated_instInhabitedMethod = _init_lp_aesop_Aesop_ScriptGenerated_instInhabitedMethod();
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Stats_Basic(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Rule_Name(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tracing(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Options_Public(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Stats_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Rule_Name(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tracing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Options_Public(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Stats_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Stats_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Stats_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
