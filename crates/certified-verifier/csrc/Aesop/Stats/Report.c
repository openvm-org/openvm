// Lean compiler output
// Module: Aesop.Stats.Report
// Imports: public import Init public meta import Init public import Aesop.Percent public import Aesop.Stats.Extension
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
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fswap(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_instOrdNanos_ord(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_shiftr(lean_object*, lean_object*);
double l_Float_ofScientific(lean_object*, uint8_t, lean_object*);
double lean_float_of_nat(lean_object*);
double lean_float_mul(double, double);
double ceil(double);
uint64_t lean_float_to_uint64(double);
lean_object* lean_uint64_to_nat(uint64_t);
lean_object* lp_aesop_Aesop_Nanos_printAsMillis(lean_object*);
lean_object* lean_nat_div(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Nat_reprFast(lean_object*);
lean_object* lean_array_to_list(lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Array_toSubarray___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_Stats_ruleStatsTotals(lean_object*, lean_object*);
lean_object* lp_aesop_Aesop_sortRuleStatsTotals(lean_object*);
lean_object* lean_string_append(lean_object*, lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* l_Std_Format_indentD(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortedPercentileD___redArg(double, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortedPercentileD___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortedPercentileD(lean_object*, double, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortedPercentileD___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_sortedMedianD___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_aesop_Aesop_sortedMedianD___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortedMedianD___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortedMedianD___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortedMedianD(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortedMedianD___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_aesop_Aesop_StatsReport_instToStringNanos___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_Nanos_printAsMillis, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_StatsReport_instToStringNanos___closed__0 = (const lean_object*)&lp_aesop_Aesop_StatsReport_instToStringNanos___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_StatsReport_instToStringNanos = (const lean_object*)&lp_aesop_Aesop_StatsReport_instToStringNanos___closed__0_value;
static const lean_string_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = " ["};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___closed__0_value)}};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___closed__1_value;
static const lean_string_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___closed__2_value;
static const lean_ctor_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___closed__2_value)}};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___closed__3_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___boxed(lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " in "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___lam__0___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___lam__0___closed__0_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___lam__0___closed__0_value)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___lam__0___closed__1 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___lam__0(lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":\n"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__0_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__0_value)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__1 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__1_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "  "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__2_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__2_value)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__3 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__3_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "total:      "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__4 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__4_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__4_value)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__5 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__5_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\n"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__6 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__6_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__6_value)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__7 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__7_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "successful: "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__8 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__8_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__8_value)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__9 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__9_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "failed:     "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__10 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__10_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__10_value)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__11 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__11_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "global"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__12 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__12_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "local"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__13 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__13_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "|"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__14 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__14_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "apply"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__15 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__15_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "cases"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__16 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__16_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "constructors"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__17 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__17_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "destruct"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__18 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__18_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "forward"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__19 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__19_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__20 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__20_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__21 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__21_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unfold"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__22 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__22_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "norm"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__23 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__23_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "safe"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__24 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__24_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "unsafe"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__25 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__25_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "<norm simp>"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__26 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__26_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "<norm unfold>"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__27 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__27_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats___closed__0_value)}};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_StatsReport_default_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_StatsReport_default_spec__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_StatsReport_default_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_StatsReport_default_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_StatsReport_default_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_StatsReport_default_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_StatsReport_default___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_StatsReport_default___closed__0;
static lean_once_cell_t lp_aesop_Aesop_StatsReport_default___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_StatsReport_default___closed__1;
static lean_once_cell_t lp_aesop_Aesop_StatsReport_default___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_StatsReport_default___closed__2;
static lean_once_cell_t lp_aesop_Aesop_StatsReport_default___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_StatsReport_default___closed__3;
static lean_once_cell_t lp_aesop_Aesop_StatsReport_default___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_StatsReport_default___closed__4;
static lean_once_cell_t lp_aesop_Aesop_StatsReport_default___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_StatsReport_default___closed__5;
static lean_once_cell_t lp_aesop_Aesop_StatsReport_default___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_StatsReport_default___closed__6;
static lean_once_cell_t lp_aesop_Aesop_StatsReport_default___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_StatsReport_default___closed__7;
static lean_once_cell_t lp_aesop_Aesop_StatsReport_default___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_StatsReport_default___closed__8;
static const lean_string_object lp_aesop_Aesop_StatsReport_default___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "Statistics for "};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__9 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__9_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_default___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__9_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__10 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__10_value;
static const lean_string_object lp_aesop_Aesop_StatsReport_default___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 102, .m_capacity = 102, .m_length = 101, .m_data = " Aesop calls in current and imported modules\nDisplaying totals and [averages]\nTotal Aesop time:      "};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__11 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__11_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_default___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__11_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__12 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__12_value;
static const lean_string_object lp_aesop_Aesop_StatsReport_default___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "\nConfig parsing:        "};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__13 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__13_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_default___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__13_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__14 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__14_value;
static const lean_string_object lp_aesop_Aesop_StatsReport_default___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "\nRule set construction: "};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__15 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__15_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_default___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__15_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__16 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__16_value;
static const lean_string_object lp_aesop_Aesop_StatsReport_default___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "\nRule selection:        "};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__17 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__17_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_default___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__17_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__18 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__18_value;
static const lean_string_object lp_aesop_Aesop_StatsReport_default___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "\nScript generation:     "};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__19 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__19_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_default___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__19_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__20 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__20_value;
static const lean_string_object lp_aesop_Aesop_StatsReport_default___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "\nSearch:                "};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__21 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__21_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_default___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__21_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__22 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__22_value;
static const lean_string_object lp_aesop_Aesop_StatsReport_default___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 25, .m_capacity = 25, .m_length = 24, .m_data = "\nForward state updates: "};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__23 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__23_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_default___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__23_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__24 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__24_value;
static const lean_string_object lp_aesop_Aesop_StatsReport_default___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "\nRules:"};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__25 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__25_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_default___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__25_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_default___closed__26 = (const lean_object*)&lp_aesop_Aesop_StatsReport_default___closed__26_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsReport_default(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsReport_default___boxed(lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__0_value)}};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__1_value;
static const lean_string_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "<none>"};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__2_value;
static const lean_ctor_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__2_value)}};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__3_value;
static const lean_string_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = " (perfect: "};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__4_value;
static const lean_ctor_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__4_value)}};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__5_value;
static const lean_string_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "false"};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__6_value;
static const lean_string_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "true"};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__7 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__7_value;
static const lean_string_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "static"};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__8 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__8_value;
static const lean_string_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "dynamic"};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__9 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__9_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated(lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___redArg___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = " (min = "};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__0 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__0_value;
static const lean_ctor_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__0_value)}};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__1 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__1_value;
static const lean_string_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = ", avg = "};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__2 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__2_value;
static const lean_ctor_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__2_value)}};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__3 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__3_value;
static const lean_string_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = ", median = "};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__4 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__4_value;
static const lean_ctor_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__4_value)}};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__5 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__5_value;
static const lean_string_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = ", 80pct = "};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__6 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__6_value;
static const lean_ctor_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__6_value)}};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__7 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__7_value;
static const lean_string_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = ", 95pct = "};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__8 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__8_value;
static const lean_ctor_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__8_value)}};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__9 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__9_value;
static const lean_string_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = ", 99pct = "};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__10 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__10_value;
static const lean_ctor_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__10_value)}};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__11 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__11_value;
static const lean_string_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = ", max = "};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__12 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__12_value;
static const lean_ctor_object lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__12_value)}};
static const lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__13 = (const lean_object*)&lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__13_value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_foldl___at___00Std_Format_joinSep___at___00Aesop_StatsReport_scriptsCore_spec__3_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Std_Format_joinSep___at___00Aesop_StatsReport_scriptsCore_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_StatsReport_scriptsCore_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_StatsReport_scriptsCore_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_StatsReport_scriptsCore_spec__1___redArg(lean_object*, lean_object*);
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__0 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__0_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__0_value)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__1 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__1_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = ": script "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__2 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__2_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__2_value)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__3 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__3_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = ", total "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__4 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__4_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__4_value)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__5 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__5_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = ", type "};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__6 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__6_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__6_value)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__7 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__7_value;
static const lean_string_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "\?:\?"};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__8 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__8_value;
static const lean_ctor_object lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__8_value)}};
static const lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__9 = (const lean_object*)&lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__9_value;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_StatsReport_scriptsCore_spec__5(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_StatsReport_scriptsCore_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = " in current and imported modules\nTotal Aesop time:         "};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__0 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__0_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__1 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__1_value;
static const lean_string_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "\nScript generation time:   "};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__2 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__2_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__2_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__3 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__3_value;
static const lean_string_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "\nScripts generated:        "};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__4 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__4_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__4_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__5 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__5_value;
static const lean_string_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "\n- Statically  structured: "};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__6 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__6_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__6_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__7 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__7_value;
static const lean_string_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 27, .m_capacity = 27, .m_length = 26, .m_data = "  - perfectly:            "};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__8 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__8_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__8_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__9 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__9_value;
static const lean_string_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 28, .m_capacity = 28, .m_length = 27, .m_data = "\n- Dynamically structured: "};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__10 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__10_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__10_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__11 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__11_value;
static const lean_string_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "\n\n"};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__12 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__12_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__12_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__13 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__13_value;
static const lean_string_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 46, .m_capacity = 46, .m_length = 45, .m_data = " Aesop calls with slowest script generation:\n"};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__14 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__14_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__14_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__15 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__15_value;
static const lean_string_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = " Aesop calls"};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__16 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__16_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__16_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__17 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__17_value;
static const lean_string_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 35, .m_capacity = 35, .m_length = 34, .m_data = " with nontrivial script generation"};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__18 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__18_value;
static const lean_ctor_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__18_value)}};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__19 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__19_value;
static const lean_array_object lp_aesop_Aesop_StatsReport_scriptsCore___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___closed__20 = (const lean_object*)&lp_aesop_Aesop_StatsReport_scriptsCore___closed__20_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsReport_scriptsCore(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_StatsReport_scriptsCore_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsReport_scripts(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsReport_scriptsNontrivial(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortedPercentileD___redArg(double v_p_1_, lean_object* v_dflt_2_, lean_object* v_xs_3_){
_start:
{
lean_object* v___y_5_; lean_object* v___x_9_; lean_object* v___x_10_; uint8_t v___x_11_; 
v___x_9_ = lean_array_get_size(v_xs_3_);
v___x_10_ = lean_unsigned_to_nat(0u);
v___x_11_ = lean_nat_dec_eq(v___x_9_, v___x_10_);
if (v___x_11_ == 0)
{
double v___x_12_; double v___x_13_; double v___x_14_; uint64_t v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; uint8_t v___x_19_; 
v___x_12_ = lean_float_of_nat(v___x_9_);
v___x_13_ = lean_float_mul(v___x_12_, v_p_1_);
v___x_14_ = ceil(v___x_13_);
v___x_15_ = lean_float_to_uint64(v___x_14_);
v___x_16_ = lean_uint64_to_nat(v___x_15_);
v___x_17_ = lean_unsigned_to_nat(1u);
v___x_18_ = lean_nat_sub(v___x_9_, v___x_17_);
v___x_19_ = lean_nat_dec_le(v___x_16_, v___x_18_);
if (v___x_19_ == 0)
{
lean_dec(v___x_16_);
v___y_5_ = v___x_18_;
goto v___jp_4_;
}
else
{
lean_dec(v___x_18_);
v___y_5_ = v___x_16_;
goto v___jp_4_;
}
}
else
{
lean_inc(v_dflt_2_);
return v_dflt_2_;
}
v___jp_4_:
{
lean_object* v___x_6_; uint8_t v___x_7_; 
v___x_6_ = lean_array_get_size(v_xs_3_);
v___x_7_ = lean_nat_dec_lt(v___y_5_, v___x_6_);
if (v___x_7_ == 0)
{
lean_dec(v___y_5_);
lean_inc(v_dflt_2_);
return v_dflt_2_;
}
else
{
lean_object* v___x_8_; 
v___x_8_ = lean_array_fget_borrowed(v_xs_3_, v___y_5_);
lean_dec(v___y_5_);
lean_inc(v___x_8_);
return v___x_8_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortedPercentileD___redArg___boxed(lean_object* v_p_20_, lean_object* v_dflt_21_, lean_object* v_xs_22_){
_start:
{
double v_p_boxed_23_; lean_object* v_res_24_; 
v_p_boxed_23_ = lean_unbox_float(v_p_20_);
lean_dec_ref(v_p_20_);
v_res_24_ = lp_aesop_Aesop_sortedPercentileD___redArg(v_p_boxed_23_, v_dflt_21_, v_xs_22_);
lean_dec_ref(v_xs_22_);
lean_dec(v_dflt_21_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortedPercentileD(lean_object* v_00_u03b1_25_, double v_p_26_, lean_object* v_dflt_27_, lean_object* v_xs_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = lp_aesop_Aesop_sortedPercentileD___redArg(v_p_26_, v_dflt_27_, v_xs_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortedPercentileD___boxed(lean_object* v_00_u03b1_30_, lean_object* v_p_31_, lean_object* v_dflt_32_, lean_object* v_xs_33_){
_start:
{
double v_p_boxed_34_; lean_object* v_res_35_; 
v_p_boxed_34_ = lean_unbox_float(v_p_31_);
lean_dec_ref(v_p_31_);
v_res_35_ = lp_aesop_Aesop_sortedPercentileD(v_00_u03b1_30_, v_p_boxed_34_, v_dflt_32_, v_xs_33_);
lean_dec_ref(v_xs_33_);
lean_dec(v_dflt_32_);
return v_res_35_;
}
}
static double _init_lp_aesop_Aesop_sortedMedianD___redArg___closed__0(void){
_start:
{
lean_object* v___x_36_; uint8_t v___x_37_; lean_object* v___x_38_; double v___x_39_; 
v___x_36_ = lean_unsigned_to_nat(1u);
v___x_37_ = 1;
v___x_38_ = lean_unsigned_to_nat(5u);
v___x_39_ = l_Float_ofScientific(v___x_38_, v___x_37_, v___x_36_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortedMedianD___redArg(lean_object* v_dflt_40_, lean_object* v_xs_41_){
_start:
{
double v___x_42_; lean_object* v___x_43_; 
v___x_42_ = lean_float_once(&lp_aesop_Aesop_sortedMedianD___redArg___closed__0, &lp_aesop_Aesop_sortedMedianD___redArg___closed__0_once, _init_lp_aesop_Aesop_sortedMedianD___redArg___closed__0);
v___x_43_ = lp_aesop_Aesop_sortedPercentileD___redArg(v___x_42_, v_dflt_40_, v_xs_41_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortedMedianD___redArg___boxed(lean_object* v_dflt_44_, lean_object* v_xs_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_aesop_Aesop_sortedMedianD___redArg(v_dflt_44_, v_xs_45_);
lean_dec_ref(v_xs_45_);
lean_dec(v_dflt_44_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortedMedianD(lean_object* v_00_u03b1_47_, lean_object* v_dflt_48_, lean_object* v_xs_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lp_aesop_Aesop_sortedMedianD___redArg(v_dflt_48_, v_xs_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_sortedMedianD___boxed(lean_object* v_00_u03b1_51_, lean_object* v_dflt_52_, lean_object* v_xs_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_aesop_Aesop_sortedMedianD(v_00_u03b1_51_, v_dflt_52_, v_xs_53_);
lean_dec_ref(v_xs_53_);
lean_dec(v_dflt_52_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime(lean_object* v_n_63_, lean_object* v_samples_64_){
_start:
{
lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___y_70_; lean_object* v___x_76_; uint8_t v___x_77_; 
lean_inc(v_n_63_);
v___x_65_ = lp_aesop_Aesop_Nanos_printAsMillis(v_n_63_);
v___x_66_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_66_, 0, v___x_65_);
v___x_67_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___closed__1));
v___x_68_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_68_, 0, v___x_66_);
lean_ctor_set(v___x_68_, 1, v___x_67_);
v___x_76_ = lean_unsigned_to_nat(0u);
v___x_77_ = lean_nat_dec_eq(v_samples_64_, v___x_76_);
if (v___x_77_ == 0)
{
lean_object* v___x_78_; 
v___x_78_ = lean_nat_div(v_n_63_, v_samples_64_);
lean_dec(v_n_63_);
v___y_70_ = v___x_78_;
goto v___jp_69_;
}
else
{
lean_dec(v_n_63_);
v___y_70_ = v___x_76_;
goto v___jp_69_;
}
v___jp_69_:
{
lean_object* v___x_71_; lean_object* v___x_72_; lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
v___x_71_ = lp_aesop_Aesop_Nanos_printAsMillis(v___y_70_);
v___x_72_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_72_, 0, v___x_71_);
v___x_73_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_73_, 0, v___x_68_);
lean_ctor_set(v___x_73_, 1, v___x_72_);
v___x_74_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___closed__3));
v___x_75_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_75_, 0, v___x_73_);
lean_ctor_set(v___x_75_, 1, v___x_74_);
return v___x_75_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime___boxed(lean_object* v_n_79_, lean_object* v_samples_80_){
_start:
{
lean_object* v_res_81_; 
v_res_81_ = lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime(v_n_79_, v_samples_80_);
lean_dec(v_samples_80_);
return v_res_81_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___lam__0(lean_object* v_n_85_, lean_object* v_samples_86_){
_start:
{
lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; 
lean_inc(v_samples_86_);
v___x_87_ = l_Nat_reprFast(v_samples_86_);
v___x_88_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_88_, 0, v___x_87_);
v___x_89_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___lam__0___closed__1));
v___x_90_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_90_, 0, v___x_88_);
lean_ctor_set(v___x_90_, 1, v___x_89_);
v___x_91_ = lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime(v_n_85_, v_samples_86_);
lean_dec(v_samples_86_);
v___x_92_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_92_, 0, v___x_90_);
lean_ctor_set(v___x_92_, 1, v___x_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0(lean_object* v_as_127_, size_t v_sz_128_, size_t v_i_129_, lean_object* v_b_130_){
_start:
{
uint8_t v___x_131_; 
v___x_131_ = lean_usize_dec_lt(v_i_129_, v_sz_128_);
if (v___x_131_ == 0)
{
return v_b_130_;
}
else
{
lean_object* v_a_132_; lean_object* v_fst_133_; lean_object* v_snd_134_; lean_object* v___x_136_; uint8_t v_isShared_137_; uint8_t v_isSharedCheck_213_; 
v_a_132_ = lean_array_uget(v_as_127_, v_i_129_);
v_fst_133_ = lean_ctor_get(v_a_132_, 0);
v_snd_134_ = lean_ctor_get(v_a_132_, 1);
v_isSharedCheck_213_ = !lean_is_exclusive(v_a_132_);
if (v_isSharedCheck_213_ == 0)
{
v___x_136_ = v_a_132_;
v_isShared_137_ = v_isSharedCheck_213_;
goto v_resetjp_135_;
}
else
{
lean_inc(v_snd_134_);
lean_inc(v_fst_133_);
lean_dec(v_a_132_);
v___x_136_ = lean_box(0);
v_isShared_137_ = v_isSharedCheck_213_;
goto v_resetjp_135_;
}
v_resetjp_135_:
{
lean_object* v___y_139_; 
switch(lean_obj_tag(v_fst_133_))
{
case 0:
{
lean_object* v_n_175_; lean_object* v_name_176_; uint8_t v_builder_177_; uint8_t v_phase_178_; uint8_t v_scope_179_; lean_object* v___y_181_; lean_object* v___y_182_; lean_object* v___y_183_; lean_object* v___y_189_; lean_object* v___y_190_; lean_object* v___y_191_; lean_object* v___y_197_; 
v_n_175_ = lean_ctor_get(v_fst_133_, 0);
lean_inc_ref(v_n_175_);
lean_dec_ref_known(v_fst_133_, 1);
v_name_176_ = lean_ctor_get(v_n_175_, 0);
lean_inc(v_name_176_);
v_builder_177_ = lean_ctor_get_uint8(v_n_175_, sizeof(void*)*1 + 8);
v_phase_178_ = lean_ctor_get_uint8(v_n_175_, sizeof(void*)*1 + 9);
v_scope_179_ = lean_ctor_get_uint8(v_n_175_, sizeof(void*)*1 + 10);
lean_dec_ref(v_n_175_);
switch(v_phase_178_)
{
case 0:
{
lean_object* v___x_208_; 
v___x_208_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__23));
v___y_197_ = v___x_208_;
goto v___jp_196_;
}
case 1:
{
lean_object* v___x_209_; 
v___x_209_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__24));
v___y_197_ = v___x_209_;
goto v___jp_196_;
}
default: 
{
lean_object* v___x_210_; 
v___x_210_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__25));
v___y_197_ = v___x_210_;
goto v___jp_196_;
}
}
v___jp_180_:
{
lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; 
v___x_184_ = lean_string_append(v___y_181_, v___y_183_);
v___x_185_ = lean_string_append(v___x_184_, v___y_182_);
v___x_186_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_name_176_, v___x_131_);
v___x_187_ = lean_string_append(v___x_185_, v___x_186_);
lean_dec_ref(v___x_186_);
v___y_139_ = v___x_187_;
goto v___jp_138_;
}
v___jp_188_:
{
lean_object* v___x_192_; lean_object* v___x_193_; 
v___x_192_ = lean_string_append(v___y_190_, v___y_191_);
v___x_193_ = lean_string_append(v___x_192_, v___y_189_);
if (v_scope_179_ == 0)
{
lean_object* v___x_194_; 
v___x_194_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__12));
v___y_181_ = v___x_193_;
v___y_182_ = v___y_189_;
v___y_183_ = v___x_194_;
goto v___jp_180_;
}
else
{
lean_object* v___x_195_; 
v___x_195_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__13));
v___y_181_ = v___x_193_;
v___y_182_ = v___y_189_;
v___y_183_ = v___x_195_;
goto v___jp_180_;
}
}
v___jp_196_:
{
lean_object* v___x_198_; lean_object* v___x_199_; 
v___x_198_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__14));
lean_inc_ref(v___y_197_);
v___x_199_ = lean_string_append(v___y_197_, v___x_198_);
switch(v_builder_177_)
{
case 0:
{
lean_object* v___x_200_; 
v___x_200_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__15));
v___y_189_ = v___x_198_;
v___y_190_ = v___x_199_;
v___y_191_ = v___x_200_;
goto v___jp_188_;
}
case 1:
{
lean_object* v___x_201_; 
v___x_201_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__16));
v___y_189_ = v___x_198_;
v___y_190_ = v___x_199_;
v___y_191_ = v___x_201_;
goto v___jp_188_;
}
case 2:
{
lean_object* v___x_202_; 
v___x_202_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__17));
v___y_189_ = v___x_198_;
v___y_190_ = v___x_199_;
v___y_191_ = v___x_202_;
goto v___jp_188_;
}
case 3:
{
lean_object* v___x_203_; 
v___x_203_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__18));
v___y_189_ = v___x_198_;
v___y_190_ = v___x_199_;
v___y_191_ = v___x_203_;
goto v___jp_188_;
}
case 4:
{
lean_object* v___x_204_; 
v___x_204_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__19));
v___y_189_ = v___x_198_;
v___y_190_ = v___x_199_;
v___y_191_ = v___x_204_;
goto v___jp_188_;
}
case 5:
{
lean_object* v___x_205_; 
v___x_205_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__20));
v___y_189_ = v___x_198_;
v___y_190_ = v___x_199_;
v___y_191_ = v___x_205_;
goto v___jp_188_;
}
case 6:
{
lean_object* v___x_206_; 
v___x_206_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__21));
v___y_189_ = v___x_198_;
v___y_190_ = v___x_199_;
v___y_191_ = v___x_206_;
goto v___jp_188_;
}
default: 
{
lean_object* v___x_207_; 
v___x_207_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__22));
v___y_189_ = v___x_198_;
v___y_190_ = v___x_199_;
v___y_191_ = v___x_207_;
goto v___jp_188_;
}
}
}
}
case 1:
{
lean_object* v___x_211_; 
v___x_211_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__26));
v___y_139_ = v___x_211_;
goto v___jp_138_;
}
default: 
{
lean_object* v___x_212_; 
v___x_212_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__27));
v___y_139_ = v___x_212_;
goto v___jp_138_;
}
}
v___jp_138_:
{
lean_object* v_numSuccessful_140_; lean_object* v_numFailed_141_; lean_object* v_elapsedSuccessful_142_; lean_object* v_elapsedFailed_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_147_; 
v_numSuccessful_140_ = lean_ctor_get(v_snd_134_, 0);
lean_inc(v_numSuccessful_140_);
v_numFailed_141_ = lean_ctor_get(v_snd_134_, 1);
lean_inc(v_numFailed_141_);
v_elapsedSuccessful_142_ = lean_ctor_get(v_snd_134_, 2);
lean_inc(v_elapsedSuccessful_142_);
v_elapsedFailed_143_ = lean_ctor_get(v_snd_134_, 3);
lean_inc(v_elapsedFailed_143_);
lean_dec(v_snd_134_);
v___x_144_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_144_, 0, v___y_139_);
v___x_145_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__1));
if (v_isShared_137_ == 0)
{
lean_ctor_set_tag(v___x_136_, 5);
lean_ctor_set(v___x_136_, 1, v___x_145_);
lean_ctor_set(v___x_136_, 0, v___x_144_);
v___x_147_ = v___x_136_;
goto v_reusejp_146_;
}
else
{
lean_object* v_reuseFailAlloc_174_; 
v_reuseFailAlloc_174_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_174_, 0, v___x_144_);
lean_ctor_set(v_reuseFailAlloc_174_, 1, v___x_145_);
v___x_147_ = v_reuseFailAlloc_174_;
goto v_reusejp_146_;
}
v_reusejp_146_:
{
lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; size_t v___x_171_; size_t v___x_172_; 
v___x_148_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__3));
v___x_149_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_149_, 0, v___x_147_);
lean_ctor_set(v___x_149_, 1, v___x_148_);
v___x_150_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__5));
v___x_151_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_151_, 0, v___x_149_);
lean_ctor_set(v___x_151_, 1, v___x_150_);
v___x_152_ = lean_nat_add(v_elapsedSuccessful_142_, v_elapsedFailed_143_);
v___x_153_ = lean_nat_add(v_numSuccessful_140_, v_numFailed_141_);
v___x_154_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___lam__0(v___x_152_, v___x_153_);
v___x_155_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_155_, 0, v___x_151_);
lean_ctor_set(v___x_155_, 1, v___x_154_);
v___x_156_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__7));
v___x_157_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_157_, 0, v___x_155_);
lean_ctor_set(v___x_157_, 1, v___x_156_);
v___x_158_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_158_, 0, v___x_157_);
lean_ctor_set(v___x_158_, 1, v___x_148_);
v___x_159_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__9));
v___x_160_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_160_, 0, v___x_158_);
lean_ctor_set(v___x_160_, 1, v___x_159_);
v___x_161_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___lam__0(v_elapsedSuccessful_142_, v_numSuccessful_140_);
v___x_162_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_162_, 0, v___x_160_);
lean_ctor_set(v___x_162_, 1, v___x_161_);
v___x_163_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_163_, 0, v___x_162_);
lean_ctor_set(v___x_163_, 1, v___x_156_);
v___x_164_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_164_, 0, v___x_163_);
lean_ctor_set(v___x_164_, 1, v___x_148_);
v___x_165_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__11));
v___x_166_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_166_, 0, v___x_164_);
lean_ctor_set(v___x_166_, 1, v___x_165_);
v___x_167_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___lam__0(v_elapsedFailed_143_, v_numFailed_141_);
v___x_168_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_168_, 0, v___x_166_);
lean_ctor_set(v___x_168_, 1, v___x_167_);
v___x_169_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_169_, 0, v___x_168_);
lean_ctor_set(v___x_169_, 1, v___x_156_);
v___x_170_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_170_, 0, v_b_130_);
lean_ctor_set(v___x_170_, 1, v___x_169_);
v___x_171_ = ((size_t)1ULL);
v___x_172_ = lean_usize_add(v_i_129_, v___x_171_);
v_i_129_ = v___x_172_;
v_b_130_ = v___x_170_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___boxed(lean_object* v_as_214_, lean_object* v_sz_215_, lean_object* v_i_216_, lean_object* v_b_217_){
_start:
{
size_t v_sz_boxed_218_; size_t v_i_boxed_219_; lean_object* v_res_220_; 
v_sz_boxed_218_ = lean_unbox_usize(v_sz_215_);
lean_dec(v_sz_215_);
v_i_boxed_219_ = lean_unbox_usize(v_i_216_);
lean_dec(v_i_216_);
v_res_220_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0(v_as_214_, v_sz_boxed_218_, v_i_boxed_219_, v_b_217_);
lean_dec_ref(v_as_214_);
return v_res_220_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats(lean_object* v_stats_224_){
_start:
{
lean_object* v_fmt_225_; size_t v_sz_226_; size_t v___x_227_; lean_object* v___x_228_; 
v_fmt_225_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats___closed__1));
v_sz_226_ = lean_array_size(v_stats_224_);
v___x_227_ = ((size_t)0ULL);
v___x_228_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0(v_stats_224_, v_sz_226_, v___x_227_, v_fmt_225_);
return v___x_228_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats___boxed(lean_object* v_stats_229_){
_start:
{
lean_object* v_res_230_; 
v_res_230_ = lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats(v_stats_229_);
lean_dec_ref(v_stats_229_);
return v_res_230_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_StatsReport_default_spec__1(lean_object* v_x_231_, lean_object* v_x_232_){
_start:
{
if (lean_obj_tag(v_x_232_) == 0)
{
return v_x_231_;
}
else
{
lean_object* v_key_233_; lean_object* v_value_234_; lean_object* v_tail_235_; lean_object* v___x_236_; lean_object* v___x_237_; 
v_key_233_ = lean_ctor_get(v_x_232_, 0);
v_value_234_ = lean_ctor_get(v_x_232_, 1);
v_tail_235_ = lean_ctor_get(v_x_232_, 2);
lean_inc(v_value_234_);
lean_inc(v_key_233_);
v___x_236_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_236_, 0, v_key_233_);
lean_ctor_set(v___x_236_, 1, v_value_234_);
v___x_237_ = lean_array_push(v_x_231_, v___x_236_);
v_x_231_ = v___x_237_;
v_x_232_ = v_tail_235_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_StatsReport_default_spec__1___boxed(lean_object* v_x_239_, lean_object* v_x_240_){
_start:
{
lean_object* v_res_241_; 
v_res_241_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_StatsReport_default_spec__1(v_x_239_, v_x_240_);
lean_dec(v_x_240_);
return v_res_241_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_StatsReport_default_spec__2(lean_object* v_as_242_, size_t v_i_243_, size_t v_stop_244_, lean_object* v_b_245_){
_start:
{
uint8_t v___x_246_; 
v___x_246_ = lean_usize_dec_eq(v_i_243_, v_stop_244_);
if (v___x_246_ == 0)
{
lean_object* v___x_247_; lean_object* v___x_248_; size_t v___x_249_; size_t v___x_250_; 
v___x_247_ = lean_array_uget_borrowed(v_as_242_, v_i_243_);
v___x_248_ = lp_aesop_Std_DHashMap_Internal_AssocList_foldlM___at___00Aesop_StatsReport_default_spec__1(v_b_245_, v___x_247_);
v___x_249_ = ((size_t)1ULL);
v___x_250_ = lean_usize_add(v_i_243_, v___x_249_);
v_i_243_ = v___x_250_;
v_b_245_ = v___x_248_;
goto _start;
}
else
{
return v_b_245_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_StatsReport_default_spec__2___boxed(lean_object* v_as_252_, lean_object* v_i_253_, lean_object* v_stop_254_, lean_object* v_b_255_){
_start:
{
size_t v_i_boxed_256_; size_t v_stop_boxed_257_; lean_object* v_res_258_; 
v_i_boxed_256_ = lean_unbox_usize(v_i_253_);
lean_dec(v_i_253_);
v_stop_boxed_257_ = lean_unbox_usize(v_stop_254_);
lean_dec(v_stop_254_);
v_res_258_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_StatsReport_default_spec__2(v_as_252_, v_i_boxed_256_, v_stop_boxed_257_, v_b_255_);
lean_dec_ref(v_as_252_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_StatsReport_default_spec__0(lean_object* v_as_259_, size_t v_sz_260_, size_t v_i_261_, lean_object* v_b_262_){
_start:
{
uint8_t v___x_263_; 
v___x_263_ = lean_usize_dec_lt(v_i_261_, v_sz_260_);
if (v___x_263_ == 0)
{
return v_b_262_;
}
else
{
lean_object* v_snd_264_; lean_object* v_snd_265_; lean_object* v_snd_266_; lean_object* v_snd_267_; lean_object* v_snd_268_; lean_object* v_snd_269_; lean_object* v_fst_270_; lean_object* v___x_272_; uint8_t v_isShared_273_; uint8_t v_isSharedCheck_351_; 
v_snd_264_ = lean_ctor_get(v_b_262_, 1);
lean_inc(v_snd_264_);
v_snd_265_ = lean_ctor_get(v_snd_264_, 1);
lean_inc(v_snd_265_);
v_snd_266_ = lean_ctor_get(v_snd_265_, 1);
lean_inc(v_snd_266_);
v_snd_267_ = lean_ctor_get(v_snd_266_, 1);
lean_inc(v_snd_267_);
v_snd_268_ = lean_ctor_get(v_snd_267_, 1);
lean_inc(v_snd_268_);
v_snd_269_ = lean_ctor_get(v_snd_268_, 1);
lean_inc(v_snd_269_);
v_fst_270_ = lean_ctor_get(v_b_262_, 0);
v_isSharedCheck_351_ = !lean_is_exclusive(v_b_262_);
if (v_isSharedCheck_351_ == 0)
{
lean_object* v_unused_352_; 
v_unused_352_ = lean_ctor_get(v_b_262_, 1);
lean_dec(v_unused_352_);
v___x_272_ = v_b_262_;
v_isShared_273_ = v_isSharedCheck_351_;
goto v_resetjp_271_;
}
else
{
lean_inc(v_fst_270_);
lean_dec(v_b_262_);
v___x_272_ = lean_box(0);
v_isShared_273_ = v_isSharedCheck_351_;
goto v_resetjp_271_;
}
v_resetjp_271_:
{
lean_object* v_fst_274_; lean_object* v___x_276_; uint8_t v_isShared_277_; uint8_t v_isSharedCheck_349_; 
v_fst_274_ = lean_ctor_get(v_snd_264_, 0);
v_isSharedCheck_349_ = !lean_is_exclusive(v_snd_264_);
if (v_isSharedCheck_349_ == 0)
{
lean_object* v_unused_350_; 
v_unused_350_ = lean_ctor_get(v_snd_264_, 1);
lean_dec(v_unused_350_);
v___x_276_ = v_snd_264_;
v_isShared_277_ = v_isSharedCheck_349_;
goto v_resetjp_275_;
}
else
{
lean_inc(v_fst_274_);
lean_dec(v_snd_264_);
v___x_276_ = lean_box(0);
v_isShared_277_ = v_isSharedCheck_349_;
goto v_resetjp_275_;
}
v_resetjp_275_:
{
lean_object* v_fst_278_; lean_object* v___x_280_; uint8_t v_isShared_281_; uint8_t v_isSharedCheck_347_; 
v_fst_278_ = lean_ctor_get(v_snd_265_, 0);
v_isSharedCheck_347_ = !lean_is_exclusive(v_snd_265_);
if (v_isSharedCheck_347_ == 0)
{
lean_object* v_unused_348_; 
v_unused_348_ = lean_ctor_get(v_snd_265_, 1);
lean_dec(v_unused_348_);
v___x_280_ = v_snd_265_;
v_isShared_281_ = v_isSharedCheck_347_;
goto v_resetjp_279_;
}
else
{
lean_inc(v_fst_278_);
lean_dec(v_snd_265_);
v___x_280_ = lean_box(0);
v_isShared_281_ = v_isSharedCheck_347_;
goto v_resetjp_279_;
}
v_resetjp_279_:
{
lean_object* v_fst_282_; lean_object* v___x_284_; uint8_t v_isShared_285_; uint8_t v_isSharedCheck_345_; 
v_fst_282_ = lean_ctor_get(v_snd_266_, 0);
v_isSharedCheck_345_ = !lean_is_exclusive(v_snd_266_);
if (v_isSharedCheck_345_ == 0)
{
lean_object* v_unused_346_; 
v_unused_346_ = lean_ctor_get(v_snd_266_, 1);
lean_dec(v_unused_346_);
v___x_284_ = v_snd_266_;
v_isShared_285_ = v_isSharedCheck_345_;
goto v_resetjp_283_;
}
else
{
lean_inc(v_fst_282_);
lean_dec(v_snd_266_);
v___x_284_ = lean_box(0);
v_isShared_285_ = v_isSharedCheck_345_;
goto v_resetjp_283_;
}
v_resetjp_283_:
{
lean_object* v_fst_286_; lean_object* v___x_288_; uint8_t v_isShared_289_; uint8_t v_isSharedCheck_343_; 
v_fst_286_ = lean_ctor_get(v_snd_267_, 0);
v_isSharedCheck_343_ = !lean_is_exclusive(v_snd_267_);
if (v_isSharedCheck_343_ == 0)
{
lean_object* v_unused_344_; 
v_unused_344_ = lean_ctor_get(v_snd_267_, 1);
lean_dec(v_unused_344_);
v___x_288_ = v_snd_267_;
v_isShared_289_ = v_isSharedCheck_343_;
goto v_resetjp_287_;
}
else
{
lean_inc(v_fst_286_);
lean_dec(v_snd_267_);
v___x_288_ = lean_box(0);
v_isShared_289_ = v_isSharedCheck_343_;
goto v_resetjp_287_;
}
v_resetjp_287_:
{
lean_object* v_fst_290_; lean_object* v___x_292_; uint8_t v_isShared_293_; uint8_t v_isSharedCheck_341_; 
v_fst_290_ = lean_ctor_get(v_snd_268_, 0);
v_isSharedCheck_341_ = !lean_is_exclusive(v_snd_268_);
if (v_isSharedCheck_341_ == 0)
{
lean_object* v_unused_342_; 
v_unused_342_ = lean_ctor_get(v_snd_268_, 1);
lean_dec(v_unused_342_);
v___x_292_ = v_snd_268_;
v_isShared_293_ = v_isSharedCheck_341_;
goto v_resetjp_291_;
}
else
{
lean_inc(v_fst_290_);
lean_dec(v_snd_268_);
v___x_292_ = lean_box(0);
v_isShared_293_ = v_isSharedCheck_341_;
goto v_resetjp_291_;
}
v_resetjp_291_:
{
lean_object* v_fst_294_; lean_object* v_snd_295_; lean_object* v___x_297_; uint8_t v_isShared_298_; uint8_t v_isSharedCheck_340_; 
v_fst_294_ = lean_ctor_get(v_snd_269_, 0);
v_snd_295_ = lean_ctor_get(v_snd_269_, 1);
v_isSharedCheck_340_ = !lean_is_exclusive(v_snd_269_);
if (v_isSharedCheck_340_ == 0)
{
v___x_297_ = v_snd_269_;
v_isShared_298_ = v_isSharedCheck_340_;
goto v_resetjp_296_;
}
else
{
lean_inc(v_snd_295_);
lean_inc(v_fst_294_);
lean_dec(v_snd_269_);
v___x_297_ = lean_box(0);
v_isShared_298_ = v_isSharedCheck_340_;
goto v_resetjp_296_;
}
v_resetjp_296_:
{
lean_object* v_a_299_; lean_object* v_stats_300_; lean_object* v_total_301_; lean_object* v_configParsing_302_; lean_object* v_ruleSetConstruction_303_; lean_object* v_search_304_; lean_object* v_ruleSelection_305_; lean_object* v_script_306_; lean_object* v_forwardState_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_317_; 
v_a_299_ = lean_array_uget_borrowed(v_as_259_, v_i_261_);
v_stats_300_ = lean_ctor_get(v_a_299_, 3);
v_total_301_ = lean_ctor_get(v_stats_300_, 0);
v_configParsing_302_ = lean_ctor_get(v_stats_300_, 1);
v_ruleSetConstruction_303_ = lean_ctor_get(v_stats_300_, 2);
v_search_304_ = lean_ctor_get(v_stats_300_, 3);
v_ruleSelection_305_ = lean_ctor_get(v_stats_300_, 4);
v_script_306_ = lean_ctor_get(v_stats_300_, 5);
v_forwardState_307_ = lean_ctor_get(v_stats_300_, 6);
v___x_308_ = lean_nat_add(v_fst_270_, v_total_301_);
lean_dec(v_fst_270_);
v___x_309_ = lean_nat_add(v_fst_274_, v_configParsing_302_);
lean_dec(v_fst_274_);
v___x_310_ = lean_nat_add(v_fst_278_, v_ruleSetConstruction_303_);
lean_dec(v_fst_278_);
v___x_311_ = lean_nat_add(v_fst_282_, v_search_304_);
lean_dec(v_fst_282_);
v___x_312_ = lean_nat_add(v_fst_286_, v_ruleSelection_305_);
lean_dec(v_fst_286_);
v___x_313_ = lean_nat_add(v_fst_290_, v_script_306_);
lean_dec(v_fst_290_);
v___x_314_ = lean_nat_add(v_fst_294_, v_forwardState_307_);
lean_dec(v_fst_294_);
v___x_315_ = lp_aesop_Aesop_Stats_ruleStatsTotals(v_stats_300_, v_snd_295_);
if (v_isShared_298_ == 0)
{
lean_ctor_set(v___x_297_, 1, v___x_315_);
lean_ctor_set(v___x_297_, 0, v___x_314_);
v___x_317_ = v___x_297_;
goto v_reusejp_316_;
}
else
{
lean_object* v_reuseFailAlloc_339_; 
v_reuseFailAlloc_339_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_339_, 0, v___x_314_);
lean_ctor_set(v_reuseFailAlloc_339_, 1, v___x_315_);
v___x_317_ = v_reuseFailAlloc_339_;
goto v_reusejp_316_;
}
v_reusejp_316_:
{
lean_object* v___x_319_; 
if (v_isShared_293_ == 0)
{
lean_ctor_set(v___x_292_, 1, v___x_317_);
lean_ctor_set(v___x_292_, 0, v___x_313_);
v___x_319_ = v___x_292_;
goto v_reusejp_318_;
}
else
{
lean_object* v_reuseFailAlloc_338_; 
v_reuseFailAlloc_338_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_338_, 0, v___x_313_);
lean_ctor_set(v_reuseFailAlloc_338_, 1, v___x_317_);
v___x_319_ = v_reuseFailAlloc_338_;
goto v_reusejp_318_;
}
v_reusejp_318_:
{
lean_object* v___x_321_; 
if (v_isShared_289_ == 0)
{
lean_ctor_set(v___x_288_, 1, v___x_319_);
lean_ctor_set(v___x_288_, 0, v___x_312_);
v___x_321_ = v___x_288_;
goto v_reusejp_320_;
}
else
{
lean_object* v_reuseFailAlloc_337_; 
v_reuseFailAlloc_337_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_337_, 0, v___x_312_);
lean_ctor_set(v_reuseFailAlloc_337_, 1, v___x_319_);
v___x_321_ = v_reuseFailAlloc_337_;
goto v_reusejp_320_;
}
v_reusejp_320_:
{
lean_object* v___x_323_; 
if (v_isShared_285_ == 0)
{
lean_ctor_set(v___x_284_, 1, v___x_321_);
lean_ctor_set(v___x_284_, 0, v___x_311_);
v___x_323_ = v___x_284_;
goto v_reusejp_322_;
}
else
{
lean_object* v_reuseFailAlloc_336_; 
v_reuseFailAlloc_336_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_336_, 0, v___x_311_);
lean_ctor_set(v_reuseFailAlloc_336_, 1, v___x_321_);
v___x_323_ = v_reuseFailAlloc_336_;
goto v_reusejp_322_;
}
v_reusejp_322_:
{
lean_object* v___x_325_; 
if (v_isShared_281_ == 0)
{
lean_ctor_set(v___x_280_, 1, v___x_323_);
lean_ctor_set(v___x_280_, 0, v___x_310_);
v___x_325_ = v___x_280_;
goto v_reusejp_324_;
}
else
{
lean_object* v_reuseFailAlloc_335_; 
v_reuseFailAlloc_335_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_335_, 0, v___x_310_);
lean_ctor_set(v_reuseFailAlloc_335_, 1, v___x_323_);
v___x_325_ = v_reuseFailAlloc_335_;
goto v_reusejp_324_;
}
v_reusejp_324_:
{
lean_object* v___x_327_; 
if (v_isShared_277_ == 0)
{
lean_ctor_set(v___x_276_, 1, v___x_325_);
lean_ctor_set(v___x_276_, 0, v___x_309_);
v___x_327_ = v___x_276_;
goto v_reusejp_326_;
}
else
{
lean_object* v_reuseFailAlloc_334_; 
v_reuseFailAlloc_334_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_334_, 0, v___x_309_);
lean_ctor_set(v_reuseFailAlloc_334_, 1, v___x_325_);
v___x_327_ = v_reuseFailAlloc_334_;
goto v_reusejp_326_;
}
v_reusejp_326_:
{
lean_object* v___x_329_; 
if (v_isShared_273_ == 0)
{
lean_ctor_set(v___x_272_, 1, v___x_327_);
lean_ctor_set(v___x_272_, 0, v___x_308_);
v___x_329_ = v___x_272_;
goto v_reusejp_328_;
}
else
{
lean_object* v_reuseFailAlloc_333_; 
v_reuseFailAlloc_333_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_333_, 0, v___x_308_);
lean_ctor_set(v_reuseFailAlloc_333_, 1, v___x_327_);
v___x_329_ = v_reuseFailAlloc_333_;
goto v_reusejp_328_;
}
v_reusejp_328_:
{
size_t v___x_330_; size_t v___x_331_; 
v___x_330_ = ((size_t)1ULL);
v___x_331_ = lean_usize_add(v_i_261_, v___x_330_);
v_i_261_ = v___x_331_;
v_b_262_ = v___x_329_;
goto _start;
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
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_StatsReport_default_spec__0___boxed(lean_object* v_as_353_, lean_object* v_sz_354_, lean_object* v_i_355_, lean_object* v_b_356_){
_start:
{
size_t v_sz_boxed_357_; size_t v_i_boxed_358_; lean_object* v_res_359_; 
v_sz_boxed_357_ = lean_unbox_usize(v_sz_354_);
lean_dec(v_sz_354_);
v_i_boxed_358_ = lean_unbox_usize(v_i_355_);
lean_dec(v_i_355_);
v_res_359_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_StatsReport_default_spec__0(v_as_353_, v_sz_boxed_357_, v_i_boxed_358_, v_b_356_);
lean_dec_ref(v_as_353_);
return v_res_359_;
}
}
static lean_object* _init_lp_aesop_Aesop_StatsReport_default___closed__0(void){
_start:
{
lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; 
v___x_360_ = lean_box(0);
v___x_361_ = lean_unsigned_to_nat(16u);
v___x_362_ = lean_mk_array(v___x_361_, v___x_360_);
return v___x_362_;
}
}
static lean_object* _init_lp_aesop_Aesop_StatsReport_default___closed__1(void){
_start:
{
lean_object* v___x_363_; lean_object* v_total_364_; lean_object* v_ruleStats_365_; 
v___x_363_ = lean_obj_once(&lp_aesop_Aesop_StatsReport_default___closed__0, &lp_aesop_Aesop_StatsReport_default___closed__0_once, _init_lp_aesop_Aesop_StatsReport_default___closed__0);
v_total_364_ = lean_unsigned_to_nat(0u);
v_ruleStats_365_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_ruleStats_365_, 0, v_total_364_);
lean_ctor_set(v_ruleStats_365_, 1, v___x_363_);
return v_ruleStats_365_;
}
}
static lean_object* _init_lp_aesop_Aesop_StatsReport_default___closed__2(void){
_start:
{
lean_object* v_ruleStats_366_; lean_object* v_total_367_; lean_object* v___x_368_; 
v_ruleStats_366_ = lean_obj_once(&lp_aesop_Aesop_StatsReport_default___closed__1, &lp_aesop_Aesop_StatsReport_default___closed__1_once, _init_lp_aesop_Aesop_StatsReport_default___closed__1);
v_total_367_ = lean_unsigned_to_nat(0u);
v___x_368_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_368_, 0, v_total_367_);
lean_ctor_set(v___x_368_, 1, v_ruleStats_366_);
return v___x_368_;
}
}
static lean_object* _init_lp_aesop_Aesop_StatsReport_default___closed__3(void){
_start:
{
lean_object* v___x_369_; lean_object* v_total_370_; lean_object* v___x_371_; 
v___x_369_ = lean_obj_once(&lp_aesop_Aesop_StatsReport_default___closed__2, &lp_aesop_Aesop_StatsReport_default___closed__2_once, _init_lp_aesop_Aesop_StatsReport_default___closed__2);
v_total_370_ = lean_unsigned_to_nat(0u);
v___x_371_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_371_, 0, v_total_370_);
lean_ctor_set(v___x_371_, 1, v___x_369_);
return v___x_371_;
}
}
static lean_object* _init_lp_aesop_Aesop_StatsReport_default___closed__4(void){
_start:
{
lean_object* v___x_372_; lean_object* v_total_373_; lean_object* v___x_374_; 
v___x_372_ = lean_obj_once(&lp_aesop_Aesop_StatsReport_default___closed__3, &lp_aesop_Aesop_StatsReport_default___closed__3_once, _init_lp_aesop_Aesop_StatsReport_default___closed__3);
v_total_373_ = lean_unsigned_to_nat(0u);
v___x_374_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_374_, 0, v_total_373_);
lean_ctor_set(v___x_374_, 1, v___x_372_);
return v___x_374_;
}
}
static lean_object* _init_lp_aesop_Aesop_StatsReport_default___closed__5(void){
_start:
{
lean_object* v___x_375_; lean_object* v_total_376_; lean_object* v___x_377_; 
v___x_375_ = lean_obj_once(&lp_aesop_Aesop_StatsReport_default___closed__4, &lp_aesop_Aesop_StatsReport_default___closed__4_once, _init_lp_aesop_Aesop_StatsReport_default___closed__4);
v_total_376_ = lean_unsigned_to_nat(0u);
v___x_377_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_377_, 0, v_total_376_);
lean_ctor_set(v___x_377_, 1, v___x_375_);
return v___x_377_;
}
}
static lean_object* _init_lp_aesop_Aesop_StatsReport_default___closed__6(void){
_start:
{
lean_object* v___x_378_; lean_object* v_total_379_; lean_object* v___x_380_; 
v___x_378_ = lean_obj_once(&lp_aesop_Aesop_StatsReport_default___closed__5, &lp_aesop_Aesop_StatsReport_default___closed__5_once, _init_lp_aesop_Aesop_StatsReport_default___closed__5);
v_total_379_ = lean_unsigned_to_nat(0u);
v___x_380_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_380_, 0, v_total_379_);
lean_ctor_set(v___x_380_, 1, v___x_378_);
return v___x_380_;
}
}
static lean_object* _init_lp_aesop_Aesop_StatsReport_default___closed__7(void){
_start:
{
lean_object* v___x_381_; lean_object* v_total_382_; lean_object* v___x_383_; 
v___x_381_ = lean_obj_once(&lp_aesop_Aesop_StatsReport_default___closed__6, &lp_aesop_Aesop_StatsReport_default___closed__6_once, _init_lp_aesop_Aesop_StatsReport_default___closed__6);
v_total_382_ = lean_unsigned_to_nat(0u);
v___x_383_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_383_, 0, v_total_382_);
lean_ctor_set(v___x_383_, 1, v___x_381_);
return v___x_383_;
}
}
static lean_object* _init_lp_aesop_Aesop_StatsReport_default___closed__8(void){
_start:
{
lean_object* v___x_384_; lean_object* v_total_385_; lean_object* v___x_386_; 
v___x_384_ = lean_obj_once(&lp_aesop_Aesop_StatsReport_default___closed__7, &lp_aesop_Aesop_StatsReport_default___closed__7_once, _init_lp_aesop_Aesop_StatsReport_default___closed__7);
v_total_385_ = lean_unsigned_to_nat(0u);
v___x_386_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_386_, 0, v_total_385_);
lean_ctor_set(v___x_386_, 1, v___x_384_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsReport_default(lean_object* v_statsArray_414_){
_start:
{
lean_object* v_total_415_; lean_object* v___x_416_; size_t v_sz_417_; size_t v___x_418_; lean_object* v___x_419_; lean_object* v_snd_420_; lean_object* v_snd_421_; lean_object* v_snd_422_; lean_object* v_snd_423_; lean_object* v_snd_424_; lean_object* v_snd_425_; lean_object* v_snd_426_; lean_object* v_fst_427_; lean_object* v___x_429_; uint8_t v_isShared_430_; uint8_t v_isSharedCheck_538_; 
v_total_415_ = lean_unsigned_to_nat(0u);
v___x_416_ = lean_obj_once(&lp_aesop_Aesop_StatsReport_default___closed__8, &lp_aesop_Aesop_StatsReport_default___closed__8_once, _init_lp_aesop_Aesop_StatsReport_default___closed__8);
v_sz_417_ = lean_array_size(v_statsArray_414_);
v___x_418_ = ((size_t)0ULL);
v___x_419_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_StatsReport_default_spec__0(v_statsArray_414_, v_sz_417_, v___x_418_, v___x_416_);
v_snd_420_ = lean_ctor_get(v___x_419_, 1);
lean_inc(v_snd_420_);
v_snd_421_ = lean_ctor_get(v_snd_420_, 1);
lean_inc(v_snd_421_);
v_snd_422_ = lean_ctor_get(v_snd_421_, 1);
lean_inc(v_snd_422_);
v_snd_423_ = lean_ctor_get(v_snd_422_, 1);
lean_inc(v_snd_423_);
v_snd_424_ = lean_ctor_get(v_snd_423_, 1);
lean_inc(v_snd_424_);
v_snd_425_ = lean_ctor_get(v_snd_424_, 1);
lean_inc(v_snd_425_);
v_snd_426_ = lean_ctor_get(v_snd_425_, 1);
lean_inc(v_snd_426_);
v_fst_427_ = lean_ctor_get(v___x_419_, 0);
v_isSharedCheck_538_ = !lean_is_exclusive(v___x_419_);
if (v_isSharedCheck_538_ == 0)
{
lean_object* v_unused_539_; 
v_unused_539_ = lean_ctor_get(v___x_419_, 1);
lean_dec(v_unused_539_);
v___x_429_ = v___x_419_;
v_isShared_430_ = v_isSharedCheck_538_;
goto v_resetjp_428_;
}
else
{
lean_inc(v_fst_427_);
lean_dec(v___x_419_);
v___x_429_ = lean_box(0);
v_isShared_430_ = v_isSharedCheck_538_;
goto v_resetjp_428_;
}
v_resetjp_428_:
{
lean_object* v_fst_431_; lean_object* v___x_433_; uint8_t v_isShared_434_; uint8_t v_isSharedCheck_536_; 
v_fst_431_ = lean_ctor_get(v_snd_420_, 0);
v_isSharedCheck_536_ = !lean_is_exclusive(v_snd_420_);
if (v_isSharedCheck_536_ == 0)
{
lean_object* v_unused_537_; 
v_unused_537_ = lean_ctor_get(v_snd_420_, 1);
lean_dec(v_unused_537_);
v___x_433_ = v_snd_420_;
v_isShared_434_ = v_isSharedCheck_536_;
goto v_resetjp_432_;
}
else
{
lean_inc(v_fst_431_);
lean_dec(v_snd_420_);
v___x_433_ = lean_box(0);
v_isShared_434_ = v_isSharedCheck_536_;
goto v_resetjp_432_;
}
v_resetjp_432_:
{
lean_object* v_fst_435_; lean_object* v___x_437_; uint8_t v_isShared_438_; uint8_t v_isSharedCheck_534_; 
v_fst_435_ = lean_ctor_get(v_snd_421_, 0);
v_isSharedCheck_534_ = !lean_is_exclusive(v_snd_421_);
if (v_isSharedCheck_534_ == 0)
{
lean_object* v_unused_535_; 
v_unused_535_ = lean_ctor_get(v_snd_421_, 1);
lean_dec(v_unused_535_);
v___x_437_ = v_snd_421_;
v_isShared_438_ = v_isSharedCheck_534_;
goto v_resetjp_436_;
}
else
{
lean_inc(v_fst_435_);
lean_dec(v_snd_421_);
v___x_437_ = lean_box(0);
v_isShared_438_ = v_isSharedCheck_534_;
goto v_resetjp_436_;
}
v_resetjp_436_:
{
lean_object* v_fst_439_; lean_object* v___x_441_; uint8_t v_isShared_442_; uint8_t v_isSharedCheck_532_; 
v_fst_439_ = lean_ctor_get(v_snd_422_, 0);
v_isSharedCheck_532_ = !lean_is_exclusive(v_snd_422_);
if (v_isSharedCheck_532_ == 0)
{
lean_object* v_unused_533_; 
v_unused_533_ = lean_ctor_get(v_snd_422_, 1);
lean_dec(v_unused_533_);
v___x_441_ = v_snd_422_;
v_isShared_442_ = v_isSharedCheck_532_;
goto v_resetjp_440_;
}
else
{
lean_inc(v_fst_439_);
lean_dec(v_snd_422_);
v___x_441_ = lean_box(0);
v_isShared_442_ = v_isSharedCheck_532_;
goto v_resetjp_440_;
}
v_resetjp_440_:
{
lean_object* v_fst_443_; lean_object* v___x_445_; uint8_t v_isShared_446_; uint8_t v_isSharedCheck_530_; 
v_fst_443_ = lean_ctor_get(v_snd_423_, 0);
v_isSharedCheck_530_ = !lean_is_exclusive(v_snd_423_);
if (v_isSharedCheck_530_ == 0)
{
lean_object* v_unused_531_; 
v_unused_531_ = lean_ctor_get(v_snd_423_, 1);
lean_dec(v_unused_531_);
v___x_445_ = v_snd_423_;
v_isShared_446_ = v_isSharedCheck_530_;
goto v_resetjp_444_;
}
else
{
lean_inc(v_fst_443_);
lean_dec(v_snd_423_);
v___x_445_ = lean_box(0);
v_isShared_446_ = v_isSharedCheck_530_;
goto v_resetjp_444_;
}
v_resetjp_444_:
{
lean_object* v_fst_447_; lean_object* v___x_449_; uint8_t v_isShared_450_; uint8_t v_isSharedCheck_528_; 
v_fst_447_ = lean_ctor_get(v_snd_424_, 0);
v_isSharedCheck_528_ = !lean_is_exclusive(v_snd_424_);
if (v_isSharedCheck_528_ == 0)
{
lean_object* v_unused_529_; 
v_unused_529_ = lean_ctor_get(v_snd_424_, 1);
lean_dec(v_unused_529_);
v___x_449_ = v_snd_424_;
v_isShared_450_ = v_isSharedCheck_528_;
goto v_resetjp_448_;
}
else
{
lean_inc(v_fst_447_);
lean_dec(v_snd_424_);
v___x_449_ = lean_box(0);
v_isShared_450_ = v_isSharedCheck_528_;
goto v_resetjp_448_;
}
v_resetjp_448_:
{
lean_object* v_fst_451_; lean_object* v___x_453_; uint8_t v_isShared_454_; uint8_t v_isSharedCheck_526_; 
v_fst_451_ = lean_ctor_get(v_snd_425_, 0);
v_isSharedCheck_526_ = !lean_is_exclusive(v_snd_425_);
if (v_isSharedCheck_526_ == 0)
{
lean_object* v_unused_527_; 
v_unused_527_ = lean_ctor_get(v_snd_425_, 1);
lean_dec(v_unused_527_);
v___x_453_ = v_snd_425_;
v_isShared_454_ = v_isSharedCheck_526_;
goto v_resetjp_452_;
}
else
{
lean_inc(v_fst_451_);
lean_dec(v_snd_425_);
v___x_453_ = lean_box(0);
v_isShared_454_ = v_isSharedCheck_526_;
goto v_resetjp_452_;
}
v_resetjp_452_:
{
lean_object* v_size_455_; lean_object* v_buckets_456_; lean_object* v___x_458_; uint8_t v_isShared_459_; uint8_t v_isSharedCheck_525_; 
v_size_455_ = lean_ctor_get(v_snd_426_, 0);
v_buckets_456_ = lean_ctor_get(v_snd_426_, 1);
v_isSharedCheck_525_ = !lean_is_exclusive(v_snd_426_);
if (v_isSharedCheck_525_ == 0)
{
v___x_458_ = v_snd_426_;
v_isShared_459_ = v_isSharedCheck_525_;
goto v_resetjp_457_;
}
else
{
lean_inc(v_buckets_456_);
lean_inc(v_size_455_);
lean_dec(v_snd_426_);
v___x_458_ = lean_box(0);
v_isShared_459_ = v_isSharedCheck_525_;
goto v_resetjp_457_;
}
v_resetjp_457_:
{
lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_465_; 
v___x_460_ = lean_array_get_size(v_statsArray_414_);
v___x_461_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_default___closed__10));
v___x_462_ = l_Nat_reprFast(v___x_460_);
v___x_463_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_463_, 0, v___x_462_);
if (v_isShared_459_ == 0)
{
lean_ctor_set_tag(v___x_458_, 5);
lean_ctor_set(v___x_458_, 1, v___x_463_);
lean_ctor_set(v___x_458_, 0, v___x_461_);
v___x_465_ = v___x_458_;
goto v_reusejp_464_;
}
else
{
lean_object* v_reuseFailAlloc_524_; 
v_reuseFailAlloc_524_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_524_, 0, v___x_461_);
lean_ctor_set(v_reuseFailAlloc_524_, 1, v___x_463_);
v___x_465_ = v_reuseFailAlloc_524_;
goto v_reusejp_464_;
}
v_reusejp_464_:
{
lean_object* v___x_466_; lean_object* v___x_468_; 
v___x_466_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_default___closed__12));
if (v_isShared_454_ == 0)
{
lean_ctor_set_tag(v___x_453_, 5);
lean_ctor_set(v___x_453_, 1, v___x_466_);
lean_ctor_set(v___x_453_, 0, v___x_465_);
v___x_468_ = v___x_453_;
goto v_reusejp_467_;
}
else
{
lean_object* v_reuseFailAlloc_523_; 
v_reuseFailAlloc_523_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_523_, 0, v___x_465_);
lean_ctor_set(v_reuseFailAlloc_523_, 1, v___x_466_);
v___x_468_ = v_reuseFailAlloc_523_;
goto v_reusejp_467_;
}
v_reusejp_467_:
{
lean_object* v___x_469_; lean_object* v___x_471_; 
v___x_469_ = lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime(v_fst_427_, v___x_460_);
if (v_isShared_450_ == 0)
{
lean_ctor_set_tag(v___x_449_, 5);
lean_ctor_set(v___x_449_, 1, v___x_469_);
lean_ctor_set(v___x_449_, 0, v___x_468_);
v___x_471_ = v___x_449_;
goto v_reusejp_470_;
}
else
{
lean_object* v_reuseFailAlloc_522_; 
v_reuseFailAlloc_522_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_522_, 0, v___x_468_);
lean_ctor_set(v_reuseFailAlloc_522_, 1, v___x_469_);
v___x_471_ = v_reuseFailAlloc_522_;
goto v_reusejp_470_;
}
v_reusejp_470_:
{
lean_object* v___x_472_; lean_object* v___x_474_; 
v___x_472_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_default___closed__14));
if (v_isShared_446_ == 0)
{
lean_ctor_set_tag(v___x_445_, 5);
lean_ctor_set(v___x_445_, 1, v___x_472_);
lean_ctor_set(v___x_445_, 0, v___x_471_);
v___x_474_ = v___x_445_;
goto v_reusejp_473_;
}
else
{
lean_object* v_reuseFailAlloc_521_; 
v_reuseFailAlloc_521_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_521_, 0, v___x_471_);
lean_ctor_set(v_reuseFailAlloc_521_, 1, v___x_472_);
v___x_474_ = v_reuseFailAlloc_521_;
goto v_reusejp_473_;
}
v_reusejp_473_:
{
lean_object* v___x_475_; lean_object* v___x_477_; 
v___x_475_ = lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime(v_fst_431_, v___x_460_);
if (v_isShared_442_ == 0)
{
lean_ctor_set_tag(v___x_441_, 5);
lean_ctor_set(v___x_441_, 1, v___x_475_);
lean_ctor_set(v___x_441_, 0, v___x_474_);
v___x_477_ = v___x_441_;
goto v_reusejp_476_;
}
else
{
lean_object* v_reuseFailAlloc_520_; 
v_reuseFailAlloc_520_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_520_, 0, v___x_474_);
lean_ctor_set(v_reuseFailAlloc_520_, 1, v___x_475_);
v___x_477_ = v_reuseFailAlloc_520_;
goto v_reusejp_476_;
}
v_reusejp_476_:
{
lean_object* v___x_478_; lean_object* v___x_480_; 
v___x_478_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_default___closed__16));
if (v_isShared_438_ == 0)
{
lean_ctor_set_tag(v___x_437_, 5);
lean_ctor_set(v___x_437_, 1, v___x_478_);
lean_ctor_set(v___x_437_, 0, v___x_477_);
v___x_480_ = v___x_437_;
goto v_reusejp_479_;
}
else
{
lean_object* v_reuseFailAlloc_519_; 
v_reuseFailAlloc_519_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_519_, 0, v___x_477_);
lean_ctor_set(v_reuseFailAlloc_519_, 1, v___x_478_);
v___x_480_ = v_reuseFailAlloc_519_;
goto v_reusejp_479_;
}
v_reusejp_479_:
{
lean_object* v___x_481_; lean_object* v___x_483_; 
v___x_481_ = lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime(v_fst_435_, v___x_460_);
if (v_isShared_434_ == 0)
{
lean_ctor_set_tag(v___x_433_, 5);
lean_ctor_set(v___x_433_, 1, v___x_481_);
lean_ctor_set(v___x_433_, 0, v___x_480_);
v___x_483_ = v___x_433_;
goto v_reusejp_482_;
}
else
{
lean_object* v_reuseFailAlloc_518_; 
v_reuseFailAlloc_518_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_518_, 0, v___x_480_);
lean_ctor_set(v_reuseFailAlloc_518_, 1, v___x_481_);
v___x_483_ = v_reuseFailAlloc_518_;
goto v_reusejp_482_;
}
v_reusejp_482_:
{
lean_object* v___x_484_; lean_object* v___x_486_; 
v___x_484_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_default___closed__18));
if (v_isShared_430_ == 0)
{
lean_ctor_set_tag(v___x_429_, 5);
lean_ctor_set(v___x_429_, 1, v___x_484_);
lean_ctor_set(v___x_429_, 0, v___x_483_);
v___x_486_ = v___x_429_;
goto v_reusejp_485_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v___x_483_);
lean_ctor_set(v_reuseFailAlloc_517_, 1, v___x_484_);
v___x_486_ = v_reuseFailAlloc_517_;
goto v_reusejp_485_;
}
v_reusejp_485_:
{
lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___y_504_; lean_object* v___x_509_; lean_object* v___x_510_; uint8_t v___x_511_; 
v___x_487_ = lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime(v_fst_443_, v___x_460_);
v___x_488_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_488_, 0, v___x_486_);
lean_ctor_set(v___x_488_, 1, v___x_487_);
v___x_489_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_default___closed__20));
v___x_490_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_490_, 0, v___x_488_);
lean_ctor_set(v___x_490_, 1, v___x_489_);
v___x_491_ = lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime(v_fst_447_, v___x_460_);
v___x_492_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_492_, 0, v___x_490_);
lean_ctor_set(v___x_492_, 1, v___x_491_);
v___x_493_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_default___closed__22));
v___x_494_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_494_, 0, v___x_492_);
lean_ctor_set(v___x_494_, 1, v___x_493_);
v___x_495_ = lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime(v_fst_439_, v___x_460_);
v___x_496_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_496_, 0, v___x_494_);
lean_ctor_set(v___x_496_, 1, v___x_495_);
v___x_497_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_default___closed__24));
v___x_498_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_498_, 0, v___x_496_);
lean_ctor_set(v___x_498_, 1, v___x_497_);
v___x_499_ = lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtTime(v_fst_451_, v___x_460_);
v___x_500_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_500_, 0, v___x_498_);
lean_ctor_set(v___x_500_, 1, v___x_499_);
v___x_501_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_default___closed__26));
v___x_502_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_502_, 0, v___x_500_);
lean_ctor_set(v___x_502_, 1, v___x_501_);
v___x_509_ = lean_mk_empty_array_with_capacity(v_size_455_);
lean_dec(v_size_455_);
v___x_510_ = lean_array_get_size(v_buckets_456_);
v___x_511_ = lean_nat_dec_lt(v_total_415_, v___x_510_);
if (v___x_511_ == 0)
{
lean_dec_ref(v_buckets_456_);
v___y_504_ = v___x_509_;
goto v___jp_503_;
}
else
{
uint8_t v___x_512_; 
v___x_512_ = lean_nat_dec_le(v___x_510_, v___x_510_);
if (v___x_512_ == 0)
{
if (v___x_511_ == 0)
{
lean_dec_ref(v_buckets_456_);
v___y_504_ = v___x_509_;
goto v___jp_503_;
}
else
{
size_t v___x_513_; lean_object* v___x_514_; 
v___x_513_ = lean_usize_of_nat(v___x_510_);
v___x_514_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_StatsReport_default_spec__2(v_buckets_456_, v___x_418_, v___x_513_, v___x_509_);
lean_dec_ref(v_buckets_456_);
v___y_504_ = v___x_514_;
goto v___jp_503_;
}
}
else
{
size_t v___x_515_; lean_object* v___x_516_; 
v___x_515_ = lean_usize_of_nat(v___x_510_);
v___x_516_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_StatsReport_default_spec__2(v_buckets_456_, v___x_418_, v___x_515_, v___x_509_);
lean_dec_ref(v_buckets_456_);
v___y_504_ = v___x_516_;
goto v___jp_503_;
}
}
v___jp_503_:
{
lean_object* v___x_505_; lean_object* v___x_506_; lean_object* v___x_507_; lean_object* v___x_508_; 
v___x_505_ = lp_aesop_Aesop_sortRuleStatsTotals(v___y_504_);
v___x_506_ = lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats(v___x_505_);
lean_dec_ref(v___x_505_);
v___x_507_ = l_Std_Format_indentD(v___x_506_);
v___x_508_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_508_, 0, v___x_502_);
lean_ctor_set(v___x_508_, 1, v___x_507_);
return v___x_508_;
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
}
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsReport_default___boxed(lean_object* v_statsArray_540_){
_start:
{
lean_object* v_res_541_; 
v_res_541_ = lp_aesop_Aesop_StatsReport_default(v_statsArray_540_);
lean_dec_ref(v_statsArray_540_);
return v_res_541_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated(lean_object* v_a_555_){
_start:
{
lean_object* v___y_557_; lean_object* v___y_558_; 
if (lean_obj_tag(v_a_555_) == 0)
{
lean_object* v___x_563_; 
v___x_563_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__3));
return v___x_563_;
}
else
{
lean_object* v_val_564_; lean_object* v___x_566_; uint8_t v_isShared_567_; uint8_t v_isSharedCheck_581_; 
v_val_564_ = lean_ctor_get(v_a_555_, 0);
v_isSharedCheck_581_ = !lean_is_exclusive(v_a_555_);
if (v_isSharedCheck_581_ == 0)
{
v___x_566_ = v_a_555_;
v_isShared_567_ = v_isSharedCheck_581_;
goto v_resetjp_565_;
}
else
{
lean_inc(v_val_564_);
lean_dec(v_a_555_);
v___x_566_ = lean_box(0);
v_isShared_567_ = v_isSharedCheck_581_;
goto v_resetjp_565_;
}
v_resetjp_565_:
{
uint8_t v_method_568_; uint8_t v_perfect_569_; lean_object* v___y_571_; 
v_method_568_ = lean_ctor_get_uint8(v_val_564_, 0);
v_perfect_569_ = lean_ctor_get_uint8(v_val_564_, 1);
lean_dec(v_val_564_);
if (v_method_568_ == 0)
{
lean_object* v___x_579_; 
v___x_579_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__8));
v___y_571_ = v___x_579_;
goto v___jp_570_;
}
else
{
lean_object* v___x_580_; 
v___x_580_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__9));
v___y_571_ = v___x_580_;
goto v___jp_570_;
}
v___jp_570_:
{
lean_object* v___x_573_; 
lean_inc_ref(v___y_571_);
if (v_isShared_567_ == 0)
{
lean_ctor_set_tag(v___x_566_, 3);
lean_ctor_set(v___x_566_, 0, v___y_571_);
v___x_573_ = v___x_566_;
goto v_reusejp_572_;
}
else
{
lean_object* v_reuseFailAlloc_578_; 
v_reuseFailAlloc_578_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_578_, 0, v___y_571_);
v___x_573_ = v_reuseFailAlloc_578_;
goto v_reusejp_572_;
}
v_reusejp_572_:
{
lean_object* v___x_574_; lean_object* v___x_575_; 
v___x_574_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__5));
v___x_575_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_575_, 0, v___x_573_);
lean_ctor_set(v___x_575_, 1, v___x_574_);
if (v_perfect_569_ == 0)
{
lean_object* v___x_576_; 
v___x_576_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__6));
v___y_557_ = v___x_575_;
v___y_558_ = v___x_576_;
goto v___jp_556_;
}
else
{
lean_object* v___x_577_; 
v___x_577_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__7));
v___y_557_ = v___x_575_;
v___y_558_ = v___x_577_;
goto v___jp_556_;
}
}
}
}
}
v___jp_556_:
{
lean_object* v___x_559_; lean_object* v___x_560_; lean_object* v___x_561_; lean_object* v___x_562_; 
lean_inc_ref(v___y_558_);
v___x_559_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_559_, 0, v___y_558_);
v___x_560_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_560_, 0, v___y_557_);
lean_ctor_set(v___x_560_, 1, v___x_559_);
v___x_561_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__1));
v___x_562_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_562_, 0, v___x_560_);
lean_ctor_set(v___x_562_, 1, v___x_561_);
return v___x_562_;
}
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___redArg___lam__0(uint8_t v___x_582_, lean_object* v_x_583_, lean_object* v_y_584_){
_start:
{
uint8_t v___x_585_; 
v___x_585_ = lp_aesop_Aesop_instOrdNanos_ord(v_x_583_, v_y_584_);
if (v___x_585_ == 0)
{
return v___x_582_;
}
else
{
uint8_t v___x_586_; 
v___x_586_ = 0;
return v___x_586_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___redArg___lam__0___boxed(lean_object* v___x_587_, lean_object* v_x_588_, lean_object* v_y_589_){
_start:
{
uint8_t v___x_780__boxed_590_; uint8_t v_res_591_; lean_object* v_r_592_; 
v___x_780__boxed_590_ = lean_unbox(v___x_587_);
v_res_591_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___redArg___lam__0(v___x_780__boxed_590_, v_x_588_, v_y_589_);
lean_dec(v_y_589_);
lean_dec(v_x_588_);
v_r_592_ = lean_box(v_res_591_);
return v_r_592_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0_spec__1___redArg(lean_object* v_hi_593_, lean_object* v_pivot_594_, lean_object* v_as_595_, lean_object* v_i_596_, lean_object* v_k_597_){
_start:
{
uint8_t v___x_598_; 
v___x_598_ = lean_nat_dec_lt(v_k_597_, v_hi_593_);
if (v___x_598_ == 0)
{
lean_object* v___x_599_; lean_object* v___x_600_; 
lean_dec(v_k_597_);
v___x_599_ = lean_array_fswap(v_as_595_, v_i_596_, v_hi_593_);
v___x_600_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_600_, 0, v_i_596_);
lean_ctor_set(v___x_600_, 1, v___x_599_);
return v___x_600_;
}
else
{
lean_object* v___x_601_; uint8_t v___x_602_; 
v___x_601_ = lean_array_fget_borrowed(v_as_595_, v_k_597_);
v___x_602_ = lp_aesop_Aesop_instOrdNanos_ord(v___x_601_, v_pivot_594_);
if (v___x_602_ == 0)
{
lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; 
v___x_603_ = lean_array_fswap(v_as_595_, v_i_596_, v_k_597_);
v___x_604_ = lean_unsigned_to_nat(1u);
v___x_605_ = lean_nat_add(v_i_596_, v___x_604_);
lean_dec(v_i_596_);
v___x_606_ = lean_nat_add(v_k_597_, v___x_604_);
lean_dec(v_k_597_);
v_as_595_ = v___x_603_;
v_i_596_ = v___x_605_;
v_k_597_ = v___x_606_;
goto _start;
}
else
{
lean_object* v___x_608_; lean_object* v___x_609_; 
v___x_608_ = lean_unsigned_to_nat(1u);
v___x_609_ = lean_nat_add(v_k_597_, v___x_608_);
lean_dec(v_k_597_);
v_k_597_ = v___x_609_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_hi_611_, lean_object* v_pivot_612_, lean_object* v_as_613_, lean_object* v_i_614_, lean_object* v_k_615_){
_start:
{
lean_object* v_res_616_; 
v_res_616_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0_spec__1___redArg(v_hi_611_, v_pivot_612_, v_as_613_, v_i_614_, v_k_615_);
lean_dec(v_pivot_612_);
lean_dec(v_hi_611_);
return v_res_616_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___redArg(lean_object* v_n_617_, lean_object* v_as_618_, lean_object* v_lo_619_, lean_object* v_hi_620_){
_start:
{
lean_object* v___y_622_; uint8_t v___x_632_; 
v___x_632_ = lean_nat_dec_lt(v_lo_619_, v_hi_620_);
if (v___x_632_ == 0)
{
lean_dec(v_lo_619_);
return v_as_618_;
}
else
{
lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v_mid_635_; lean_object* v___y_637_; lean_object* v___y_643_; lean_object* v___x_648_; lean_object* v___x_649_; uint8_t v___x_650_; 
v___x_633_ = lean_nat_add(v_lo_619_, v_hi_620_);
v___x_634_ = lean_unsigned_to_nat(1u);
v_mid_635_ = lean_nat_shiftr(v___x_633_, v___x_634_);
lean_dec(v___x_633_);
v___x_648_ = lean_array_fget_borrowed(v_as_618_, v_mid_635_);
v___x_649_ = lean_array_fget_borrowed(v_as_618_, v_lo_619_);
v___x_650_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___redArg___lam__0(v___x_632_, v___x_648_, v___x_649_);
if (v___x_650_ == 0)
{
v___y_643_ = v_as_618_;
goto v___jp_642_;
}
else
{
lean_object* v___x_651_; 
v___x_651_ = lean_array_fswap(v_as_618_, v_lo_619_, v_mid_635_);
v___y_643_ = v___x_651_;
goto v___jp_642_;
}
v___jp_636_:
{
lean_object* v___x_638_; lean_object* v___x_639_; uint8_t v___x_640_; 
v___x_638_ = lean_array_fget_borrowed(v___y_637_, v_mid_635_);
v___x_639_ = lean_array_fget_borrowed(v___y_637_, v_hi_620_);
v___x_640_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___redArg___lam__0(v___x_632_, v___x_638_, v___x_639_);
if (v___x_640_ == 0)
{
lean_dec(v_mid_635_);
v___y_622_ = v___y_637_;
goto v___jp_621_;
}
else
{
lean_object* v___x_641_; 
v___x_641_ = lean_array_fswap(v___y_637_, v_mid_635_, v_hi_620_);
lean_dec(v_mid_635_);
v___y_622_ = v___x_641_;
goto v___jp_621_;
}
}
v___jp_642_:
{
lean_object* v___x_644_; lean_object* v___x_645_; uint8_t v___x_646_; 
v___x_644_ = lean_array_fget_borrowed(v___y_643_, v_hi_620_);
v___x_645_ = lean_array_fget_borrowed(v___y_643_, v_lo_619_);
v___x_646_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___redArg___lam__0(v___x_632_, v___x_644_, v___x_645_);
if (v___x_646_ == 0)
{
v___y_637_ = v___y_643_;
goto v___jp_636_;
}
else
{
lean_object* v___x_647_; 
v___x_647_ = lean_array_fswap(v___y_643_, v_lo_619_, v_hi_620_);
v___y_637_ = v___x_647_;
goto v___jp_636_;
}
}
}
v___jp_621_:
{
lean_object* v_pivot_623_; lean_object* v___x_624_; lean_object* v_fst_625_; lean_object* v_snd_626_; uint8_t v___x_627_; 
v_pivot_623_ = lean_array_fget(v___y_622_, v_hi_620_);
lean_inc_n(v_lo_619_, 2);
v___x_624_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0_spec__1___redArg(v_hi_620_, v_pivot_623_, v___y_622_, v_lo_619_, v_lo_619_);
lean_dec(v_pivot_623_);
v_fst_625_ = lean_ctor_get(v___x_624_, 0);
lean_inc(v_fst_625_);
v_snd_626_ = lean_ctor_get(v___x_624_, 1);
lean_inc(v_snd_626_);
lean_dec_ref(v___x_624_);
v___x_627_ = lean_nat_dec_le(v_hi_620_, v_fst_625_);
if (v___x_627_ == 0)
{
lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; 
v___x_628_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___redArg(v_n_617_, v_snd_626_, v_lo_619_, v_fst_625_);
v___x_629_ = lean_unsigned_to_nat(1u);
v___x_630_ = lean_nat_add(v_fst_625_, v___x_629_);
lean_dec(v_fst_625_);
v_as_618_ = v___x_628_;
v_lo_619_ = v___x_630_;
goto _start;
}
else
{
lean_dec(v_fst_625_);
lean_dec(v_lo_619_);
return v_snd_626_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___redArg___boxed(lean_object* v_n_652_, lean_object* v_as_653_, lean_object* v_lo_654_, lean_object* v_hi_655_){
_start:
{
lean_object* v_res_656_; 
v_res_656_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___redArg(v_n_652_, v_as_653_, v_lo_654_, v_hi_655_);
lean_dec(v_hi_655_);
lean_dec(v_n_652_);
return v_res_656_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0(lean_object* v_xs_657_){
_start:
{
lean_object* v___x_658_; lean_object* v___x_659_; uint8_t v___x_660_; 
v___x_658_ = lean_array_get_size(v_xs_657_);
v___x_659_ = lean_unsigned_to_nat(0u);
v___x_660_ = lean_nat_dec_eq(v___x_658_, v___x_659_);
if (v___x_660_ == 0)
{
lean_object* v___x_661_; lean_object* v___x_662_; lean_object* v___y_664_; uint8_t v___x_668_; 
v___x_661_ = lean_unsigned_to_nat(1u);
v___x_662_ = lean_nat_sub(v___x_658_, v___x_661_);
v___x_668_ = lean_nat_dec_le(v___x_659_, v___x_662_);
if (v___x_668_ == 0)
{
lean_inc(v___x_662_);
v___y_664_ = v___x_662_;
goto v___jp_663_;
}
else
{
v___y_664_ = v___x_659_;
goto v___jp_663_;
}
v___jp_663_:
{
uint8_t v___x_665_; 
v___x_665_ = lean_nat_dec_le(v___y_664_, v___x_662_);
if (v___x_665_ == 0)
{
lean_object* v___x_666_; 
lean_dec(v___x_662_);
lean_inc(v___y_664_);
v___x_666_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___redArg(v___x_658_, v_xs_657_, v___y_664_, v___y_664_);
lean_dec(v___y_664_);
return v___x_666_;
}
else
{
lean_object* v___x_667_; 
v___x_667_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___redArg(v___x_658_, v_xs_657_, v___y_664_, v___x_662_);
lean_dec(v___x_662_);
return v___x_667_;
}
}
}
else
{
return v_xs_657_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__1(lean_object* v_as_669_, size_t v_i_670_, size_t v_stop_671_, lean_object* v_b_672_){
_start:
{
uint8_t v___x_673_; 
v___x_673_ = lean_usize_dec_eq(v_i_670_, v_stop_671_);
if (v___x_673_ == 0)
{
lean_object* v___x_674_; lean_object* v___x_675_; size_t v___x_676_; size_t v___x_677_; 
v___x_674_ = lean_array_uget_borrowed(v_as_669_, v_i_670_);
v___x_675_ = lean_nat_add(v_b_672_, v___x_674_);
lean_dec(v_b_672_);
v___x_676_ = ((size_t)1ULL);
v___x_677_ = lean_usize_add(v_i_670_, v___x_676_);
v_i_670_ = v___x_677_;
v_b_672_ = v___x_675_;
goto _start;
}
else
{
return v_b_672_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__1___boxed(lean_object* v_as_679_, lean_object* v_i_680_, lean_object* v_stop_681_, lean_object* v_b_682_){
_start:
{
size_t v_i_boxed_683_; size_t v_stop_boxed_684_; lean_object* v_res_685_; 
v_i_boxed_683_ = lean_unbox_usize(v_i_680_);
lean_dec(v_i_680_);
v_stop_boxed_684_ = lean_unbox_usize(v_stop_681_);
lean_dec(v_stop_681_);
v_res_685_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__1(v_as_679_, v_i_boxed_683_, v_stop_boxed_684_, v_b_682_);
lean_dec_ref(v_as_679_);
return v_res_685_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes(lean_object* v_ns_707_){
_start:
{
lean_object* v_ns_708_; lean_object* v___x_709_; lean_object* v___y_711_; lean_object* v___y_712_; uint8_t v___y_713_; lean_object* v___y_714_; lean_object* v___y_715_; lean_object* v___x_766_; lean_object* v___y_768_; uint8_t v___y_769_; lean_object* v___y_770_; lean_object* v___y_771_; lean_object* v___y_777_; uint8_t v___y_778_; lean_object* v___y_779_; lean_object* v___y_783_; uint8_t v___x_787_; 
v_ns_708_ = lp_aesop_Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0(v_ns_707_);
v___x_709_ = lean_unsigned_to_nat(0u);
v___x_766_ = lean_array_get_size(v_ns_708_);
v___x_787_ = lean_nat_dec_lt(v___x_709_, v___x_766_);
if (v___x_787_ == 0)
{
v___y_783_ = v___x_709_;
goto v___jp_782_;
}
else
{
uint8_t v___x_788_; 
v___x_788_ = lean_nat_dec_le(v___x_766_, v___x_766_);
if (v___x_788_ == 0)
{
if (v___x_787_ == 0)
{
v___y_783_ = v___x_709_;
goto v___jp_782_;
}
else
{
size_t v___x_789_; size_t v___x_790_; lean_object* v___x_791_; 
v___x_789_ = ((size_t)0ULL);
v___x_790_ = lean_usize_of_nat(v___x_766_);
v___x_791_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__1(v_ns_708_, v___x_789_, v___x_790_, v___x_709_);
v___y_783_ = v___x_791_;
goto v___jp_782_;
}
}
else
{
size_t v___x_792_; size_t v___x_793_; lean_object* v___x_794_; 
v___x_792_ = ((size_t)0ULL);
v___x_793_ = lean_usize_of_nat(v___x_766_);
v___x_794_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__1(v_ns_708_, v___x_792_, v___x_793_, v___x_709_);
v___y_783_ = v___x_794_;
goto v___jp_782_;
}
}
v___jp_710_:
{
lean_object* v_median_716_; lean_object* v___x_717_; lean_object* v___x_718_; double v___x_719_; lean_object* v_pct80_720_; lean_object* v___x_721_; double v___x_722_; lean_object* v_pct95_723_; lean_object* v___x_724_; double v___x_725_; lean_object* v_pct99_726_; lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; lean_object* v___x_761_; lean_object* v___x_762_; lean_object* v___x_763_; lean_object* v___x_764_; lean_object* v___x_765_; 
v_median_716_ = lp_aesop_Aesop_sortedMedianD___redArg(v___x_709_, v_ns_708_);
v___x_717_ = lean_unsigned_to_nat(80u);
v___x_718_ = lean_unsigned_to_nat(2u);
v___x_719_ = l_Float_ofScientific(v___x_717_, v___y_713_, v___x_718_);
v_pct80_720_ = lp_aesop_Aesop_sortedPercentileD___redArg(v___x_719_, v___x_709_, v_ns_708_);
v___x_721_ = lean_unsigned_to_nat(95u);
v___x_722_ = l_Float_ofScientific(v___x_721_, v___y_713_, v___x_718_);
v_pct95_723_ = lp_aesop_Aesop_sortedPercentileD___redArg(v___x_722_, v___x_709_, v_ns_708_);
v___x_724_ = lean_unsigned_to_nat(99u);
v___x_725_ = l_Float_ofScientific(v___x_724_, v___y_713_, v___x_718_);
v_pct99_726_ = lp_aesop_Aesop_sortedPercentileD___redArg(v___x_725_, v___x_709_, v_ns_708_);
lean_dec_ref(v_ns_708_);
v___x_727_ = lp_aesop_Aesop_Nanos_printAsMillis(v___y_712_);
v___x_728_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_728_, 0, v___x_727_);
v___x_729_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__1));
v___x_730_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_730_, 0, v___x_728_);
lean_ctor_set(v___x_730_, 1, v___x_729_);
v___x_731_ = lp_aesop_Aesop_Nanos_printAsMillis(v___y_714_);
v___x_732_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_732_, 0, v___x_731_);
v___x_733_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_733_, 0, v___x_730_);
lean_ctor_set(v___x_733_, 1, v___x_732_);
v___x_734_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__3));
v___x_735_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_735_, 0, v___x_733_);
lean_ctor_set(v___x_735_, 1, v___x_734_);
v___x_736_ = lp_aesop_Aesop_Nanos_printAsMillis(v___y_711_);
v___x_737_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_737_, 0, v___x_736_);
v___x_738_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_738_, 0, v___x_735_);
lean_ctor_set(v___x_738_, 1, v___x_737_);
v___x_739_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__5));
v___x_740_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_740_, 0, v___x_738_);
lean_ctor_set(v___x_740_, 1, v___x_739_);
v___x_741_ = lp_aesop_Aesop_Nanos_printAsMillis(v_median_716_);
v___x_742_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_742_, 0, v___x_741_);
v___x_743_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_743_, 0, v___x_740_);
lean_ctor_set(v___x_743_, 1, v___x_742_);
v___x_744_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__7));
v___x_745_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_745_, 0, v___x_743_);
lean_ctor_set(v___x_745_, 1, v___x_744_);
v___x_746_ = lp_aesop_Aesop_Nanos_printAsMillis(v_pct80_720_);
v___x_747_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_747_, 0, v___x_746_);
v___x_748_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_748_, 0, v___x_745_);
lean_ctor_set(v___x_748_, 1, v___x_747_);
v___x_749_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__9));
v___x_750_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_750_, 0, v___x_748_);
lean_ctor_set(v___x_750_, 1, v___x_749_);
v___x_751_ = lp_aesop_Aesop_Nanos_printAsMillis(v_pct95_723_);
v___x_752_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_752_, 0, v___x_751_);
v___x_753_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_753_, 0, v___x_750_);
lean_ctor_set(v___x_753_, 1, v___x_752_);
v___x_754_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__11));
v___x_755_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_755_, 0, v___x_753_);
lean_ctor_set(v___x_755_, 1, v___x_754_);
v___x_756_ = lp_aesop_Aesop_Nanos_printAsMillis(v_pct99_726_);
v___x_757_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_757_, 0, v___x_756_);
v___x_758_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_758_, 0, v___x_755_);
lean_ctor_set(v___x_758_, 1, v___x_757_);
v___x_759_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes___closed__13));
v___x_760_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_760_, 0, v___x_758_);
lean_ctor_set(v___x_760_, 1, v___x_759_);
v___x_761_ = lp_aesop_Aesop_Nanos_printAsMillis(v___y_715_);
v___x_762_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_762_, 0, v___x_761_);
v___x_763_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_763_, 0, v___x_760_);
lean_ctor_set(v___x_763_, 1, v___x_762_);
v___x_764_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated___closed__1));
v___x_765_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_765_, 0, v___x_763_);
lean_ctor_set(v___x_765_, 1, v___x_764_);
return v___x_765_;
}
v___jp_767_:
{
lean_object* v___x_772_; lean_object* v___x_773_; uint8_t v___x_774_; 
v___x_772_ = lean_unsigned_to_nat(1u);
v___x_773_ = lean_nat_sub(v___x_766_, v___x_772_);
v___x_774_ = lean_nat_dec_lt(v___x_773_, v___x_766_);
if (v___x_774_ == 0)
{
lean_dec(v___x_773_);
v___y_711_ = v___y_768_;
v___y_712_ = v___y_770_;
v___y_713_ = v___y_769_;
v___y_714_ = v___y_771_;
v___y_715_ = v___x_709_;
goto v___jp_710_;
}
else
{
lean_object* v___x_775_; 
v___x_775_ = lean_array_fget(v_ns_708_, v___x_773_);
lean_dec(v___x_773_);
v___y_711_ = v___y_768_;
v___y_712_ = v___y_770_;
v___y_713_ = v___y_769_;
v___y_714_ = v___y_771_;
v___y_715_ = v___x_775_;
goto v___jp_710_;
}
}
v___jp_776_:
{
uint8_t v___x_780_; 
v___x_780_ = lean_nat_dec_lt(v___x_709_, v___x_766_);
if (v___x_780_ == 0)
{
v___y_768_ = v___y_779_;
v___y_769_ = v___y_778_;
v___y_770_ = v___y_777_;
v___y_771_ = v___x_709_;
goto v___jp_767_;
}
else
{
lean_object* v___x_781_; 
v___x_781_ = lean_array_fget(v_ns_708_, v___x_709_);
v___y_768_ = v___y_779_;
v___y_769_ = v___y_778_;
v___y_770_ = v___y_777_;
v___y_771_ = v___x_781_;
goto v___jp_767_;
}
}
v___jp_782_:
{
uint8_t v___x_784_; uint8_t v___x_785_; 
v___x_784_ = lean_nat_dec_eq(v___x_766_, v___x_709_);
v___x_785_ = 1;
if (v___x_784_ == 0)
{
lean_object* v___x_786_; 
v___x_786_ = lean_nat_div(v___y_783_, v___x_766_);
v___y_777_ = v___y_783_;
v___y_778_ = v___x_785_;
v___y_779_ = v___x_786_;
goto v___jp_776_;
}
else
{
v___y_777_ = v___y_783_;
v___y_778_ = v___x_785_;
v___y_779_ = v___x_709_;
goto v___jp_776_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0(lean_object* v_n_795_, lean_object* v_as_796_, lean_object* v_lo_797_, lean_object* v_hi_798_, lean_object* v_w_799_, lean_object* v_hlo_800_, lean_object* v_hhi_801_){
_start:
{
lean_object* v___x_802_; 
v___x_802_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___redArg(v_n_795_, v_as_796_, v_lo_797_, v_hi_798_);
return v___x_802_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0___boxed(lean_object* v_n_803_, lean_object* v_as_804_, lean_object* v_lo_805_, lean_object* v_hi_806_, lean_object* v_w_807_, lean_object* v_hlo_808_, lean_object* v_hhi_809_){
_start:
{
lean_object* v_res_810_; 
v_res_810_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0(v_n_803_, v_as_804_, v_lo_805_, v_hi_806_, v_w_807_, v_hlo_808_, v_hhi_809_);
lean_dec(v_hi_806_);
lean_dec(v_n_803_);
return v_res_810_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0_spec__1(lean_object* v_n_811_, lean_object* v_lo_812_, lean_object* v_hi_813_, lean_object* v_hhi_814_, lean_object* v_pivot_815_, lean_object* v_as_816_, lean_object* v_i_817_, lean_object* v_k_818_, lean_object* v_ilo_819_, lean_object* v_ik_820_, lean_object* v_w_821_){
_start:
{
lean_object* v___x_822_; 
v___x_822_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0_spec__1___redArg(v_hi_813_, v_pivot_815_, v_as_816_, v_i_817_, v_k_818_);
return v___x_822_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0_spec__1___boxed(lean_object* v_n_823_, lean_object* v_lo_824_, lean_object* v_hi_825_, lean_object* v_hhi_826_, lean_object* v_pivot_827_, lean_object* v_as_828_, lean_object* v_i_829_, lean_object* v_k_830_, lean_object* v_ilo_831_, lean_object* v_ik_832_, lean_object* v_w_833_){
_start:
{
lean_object* v_res_834_; 
v_res_834_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Array_qsortOrd___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes_spec__0_spec__0_spec__1(v_n_823_, v_lo_824_, v_hi_825_, v_hhi_826_, v_pivot_827_, v_as_828_, v_i_829_, v_k_830_, v_ilo_831_, v_ik_832_, v_w_833_);
lean_dec(v_pivot_827_);
lean_dec(v_hi_825_);
lean_dec(v_lo_824_);
lean_dec(v_n_823_);
return v_res_834_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_foldl___at___00Std_Format_joinSep___at___00Aesop_StatsReport_scriptsCore_spec__3_spec__3(lean_object* v_x_835_, lean_object* v_x_836_, lean_object* v_x_837_){
_start:
{
if (lean_obj_tag(v_x_837_) == 0)
{
lean_dec(v_x_835_);
return v_x_836_;
}
else
{
lean_object* v_head_838_; lean_object* v_tail_839_; lean_object* v___x_841_; uint8_t v_isShared_842_; uint8_t v_isSharedCheck_848_; 
v_head_838_ = lean_ctor_get(v_x_837_, 0);
v_tail_839_ = lean_ctor_get(v_x_837_, 1);
v_isSharedCheck_848_ = !lean_is_exclusive(v_x_837_);
if (v_isSharedCheck_848_ == 0)
{
v___x_841_ = v_x_837_;
v_isShared_842_ = v_isSharedCheck_848_;
goto v_resetjp_840_;
}
else
{
lean_inc(v_tail_839_);
lean_inc(v_head_838_);
lean_dec(v_x_837_);
v___x_841_ = lean_box(0);
v_isShared_842_ = v_isSharedCheck_848_;
goto v_resetjp_840_;
}
v_resetjp_840_:
{
lean_object* v___x_844_; 
lean_inc(v_x_835_);
if (v_isShared_842_ == 0)
{
lean_ctor_set_tag(v___x_841_, 5);
lean_ctor_set(v___x_841_, 1, v_x_835_);
lean_ctor_set(v___x_841_, 0, v_x_836_);
v___x_844_ = v___x_841_;
goto v_reusejp_843_;
}
else
{
lean_object* v_reuseFailAlloc_847_; 
v_reuseFailAlloc_847_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_847_, 0, v_x_836_);
lean_ctor_set(v_reuseFailAlloc_847_, 1, v_x_835_);
v___x_844_ = v_reuseFailAlloc_847_;
goto v_reusejp_843_;
}
v_reusejp_843_:
{
lean_object* v___x_845_; 
v___x_845_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_845_, 0, v___x_844_);
lean_ctor_set(v___x_845_, 1, v_head_838_);
v_x_836_ = v___x_845_;
v_x_837_ = v_tail_839_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Std_Format_joinSep___at___00Aesop_StatsReport_scriptsCore_spec__3(lean_object* v_x_849_, lean_object* v_x_850_){
_start:
{
if (lean_obj_tag(v_x_849_) == 0)
{
lean_object* v___x_851_; 
lean_dec(v_x_850_);
v___x_851_ = lean_box(0);
return v___x_851_;
}
else
{
lean_object* v_tail_852_; 
v_tail_852_ = lean_ctor_get(v_x_849_, 1);
if (lean_obj_tag(v_tail_852_) == 0)
{
lean_object* v_head_853_; 
lean_dec(v_x_850_);
v_head_853_ = lean_ctor_get(v_x_849_, 0);
lean_inc(v_head_853_);
lean_dec_ref_known(v_x_849_, 2);
return v_head_853_;
}
else
{
lean_object* v_head_854_; lean_object* v___x_855_; 
lean_inc(v_tail_852_);
v_head_854_ = lean_ctor_get(v_x_849_, 0);
lean_inc(v_head_854_);
lean_dec_ref_known(v_x_849_, 2);
v___x_855_ = lp_aesop_List_foldl___at___00Std_Format_joinSep___at___00Aesop_StatsReport_scriptsCore_spec__3_spec__3(v_x_850_, v_head_854_, v_tail_852_);
return v___x_855_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4_spec__5___redArg(lean_object* v_hi_856_, lean_object* v_pivot_857_, lean_object* v_as_858_, lean_object* v_i_859_, lean_object* v_k_860_){
_start:
{
uint8_t v___x_861_; 
v___x_861_ = lean_nat_dec_lt(v_k_860_, v_hi_856_);
if (v___x_861_ == 0)
{
lean_object* v___x_862_; lean_object* v___x_863_; 
lean_dec(v_k_860_);
v___x_862_ = lean_array_fswap(v_as_858_, v_i_859_, v_hi_856_);
v___x_863_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_863_, 0, v_i_859_);
lean_ctor_set(v___x_863_, 1, v___x_862_);
return v___x_863_;
}
else
{
lean_object* v_stats_864_; lean_object* v_script_865_; lean_object* v___x_866_; lean_object* v_stats_867_; lean_object* v_script_868_; uint8_t v___x_869_; 
v_stats_864_ = lean_ctor_get(v_pivot_857_, 3);
v_script_865_ = lean_ctor_get(v_stats_864_, 5);
v___x_866_ = lean_array_fget_borrowed(v_as_858_, v_k_860_);
v_stats_867_ = lean_ctor_get(v___x_866_, 3);
v_script_868_ = lean_ctor_get(v_stats_867_, 5);
v___x_869_ = lean_nat_dec_lt(v_script_865_, v_script_868_);
if (v___x_869_ == 0)
{
lean_object* v___x_870_; lean_object* v___x_871_; 
v___x_870_ = lean_unsigned_to_nat(1u);
v___x_871_ = lean_nat_add(v_k_860_, v___x_870_);
lean_dec(v_k_860_);
v_k_860_ = v___x_871_;
goto _start;
}
else
{
lean_object* v___x_873_; lean_object* v___x_874_; lean_object* v___x_875_; lean_object* v___x_876_; 
v___x_873_ = lean_array_fswap(v_as_858_, v_i_859_, v_k_860_);
v___x_874_ = lean_unsigned_to_nat(1u);
v___x_875_ = lean_nat_add(v_i_859_, v___x_874_);
lean_dec(v_i_859_);
v___x_876_ = lean_nat_add(v_k_860_, v___x_874_);
lean_dec(v_k_860_);
v_as_858_ = v___x_873_;
v_i_859_ = v___x_875_;
v_k_860_ = v___x_876_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4_spec__5___redArg___boxed(lean_object* v_hi_878_, lean_object* v_pivot_879_, lean_object* v_as_880_, lean_object* v_i_881_, lean_object* v_k_882_){
_start:
{
lean_object* v_res_883_; 
v_res_883_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4_spec__5___redArg(v_hi_878_, v_pivot_879_, v_as_880_, v_i_881_, v_k_882_);
lean_dec_ref(v_pivot_879_);
lean_dec(v_hi_878_);
return v_res_883_;
}
}
LEAN_EXPORT uint8_t lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___redArg___lam__0(lean_object* v_s_u2081_884_, lean_object* v_s_u2082_885_){
_start:
{
lean_object* v_stats_886_; lean_object* v_stats_887_; lean_object* v_script_888_; lean_object* v_script_889_; uint8_t v___x_890_; 
v_stats_886_ = lean_ctor_get(v_s_u2082_885_, 3);
v_stats_887_ = lean_ctor_get(v_s_u2081_884_, 3);
v_script_888_ = lean_ctor_get(v_stats_886_, 5);
v_script_889_ = lean_ctor_get(v_stats_887_, 5);
v___x_890_ = lean_nat_dec_lt(v_script_888_, v_script_889_);
return v___x_890_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___redArg___lam__0___boxed(lean_object* v_s_u2081_891_, lean_object* v_s_u2082_892_){
_start:
{
uint8_t v_res_893_; lean_object* v_r_894_; 
v_res_893_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___redArg___lam__0(v_s_u2081_891_, v_s_u2082_892_);
lean_dec_ref(v_s_u2082_892_);
lean_dec_ref(v_s_u2081_891_);
v_r_894_ = lean_box(v_res_893_);
return v_r_894_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___redArg(lean_object* v_n_895_, lean_object* v_as_896_, lean_object* v_lo_897_, lean_object* v_hi_898_){
_start:
{
lean_object* v___y_900_; uint8_t v___x_910_; 
v___x_910_ = lean_nat_dec_lt(v_lo_897_, v_hi_898_);
if (v___x_910_ == 0)
{
lean_dec(v_lo_897_);
return v_as_896_;
}
else
{
lean_object* v___x_911_; lean_object* v___x_912_; lean_object* v_mid_913_; lean_object* v___y_915_; lean_object* v___y_921_; lean_object* v___x_926_; lean_object* v___x_927_; uint8_t v___x_928_; 
v___x_911_ = lean_nat_add(v_lo_897_, v_hi_898_);
v___x_912_ = lean_unsigned_to_nat(1u);
v_mid_913_ = lean_nat_shiftr(v___x_911_, v___x_912_);
lean_dec(v___x_911_);
v___x_926_ = lean_array_fget_borrowed(v_as_896_, v_mid_913_);
v___x_927_ = lean_array_fget_borrowed(v_as_896_, v_lo_897_);
v___x_928_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___redArg___lam__0(v___x_926_, v___x_927_);
if (v___x_928_ == 0)
{
v___y_921_ = v_as_896_;
goto v___jp_920_;
}
else
{
lean_object* v___x_929_; 
v___x_929_ = lean_array_fswap(v_as_896_, v_lo_897_, v_mid_913_);
v___y_921_ = v___x_929_;
goto v___jp_920_;
}
v___jp_914_:
{
lean_object* v___x_916_; lean_object* v___x_917_; uint8_t v___x_918_; 
v___x_916_ = lean_array_fget_borrowed(v___y_915_, v_mid_913_);
v___x_917_ = lean_array_fget_borrowed(v___y_915_, v_hi_898_);
v___x_918_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___redArg___lam__0(v___x_916_, v___x_917_);
if (v___x_918_ == 0)
{
lean_dec(v_mid_913_);
v___y_900_ = v___y_915_;
goto v___jp_899_;
}
else
{
lean_object* v___x_919_; 
v___x_919_ = lean_array_fswap(v___y_915_, v_mid_913_, v_hi_898_);
lean_dec(v_mid_913_);
v___y_900_ = v___x_919_;
goto v___jp_899_;
}
}
v___jp_920_:
{
lean_object* v___x_922_; lean_object* v___x_923_; uint8_t v___x_924_; 
v___x_922_ = lean_array_fget_borrowed(v___y_921_, v_hi_898_);
v___x_923_ = lean_array_fget_borrowed(v___y_921_, v_lo_897_);
v___x_924_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___redArg___lam__0(v___x_922_, v___x_923_);
if (v___x_924_ == 0)
{
v___y_915_ = v___y_921_;
goto v___jp_914_;
}
else
{
lean_object* v___x_925_; 
v___x_925_ = lean_array_fswap(v___y_921_, v_lo_897_, v_hi_898_);
v___y_915_ = v___x_925_;
goto v___jp_914_;
}
}
}
v___jp_899_:
{
lean_object* v_pivot_901_; lean_object* v___x_902_; lean_object* v_fst_903_; lean_object* v_snd_904_; uint8_t v___x_905_; 
v_pivot_901_ = lean_array_fget(v___y_900_, v_hi_898_);
lean_inc_n(v_lo_897_, 2);
v___x_902_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4_spec__5___redArg(v_hi_898_, v_pivot_901_, v___y_900_, v_lo_897_, v_lo_897_);
lean_dec(v_pivot_901_);
v_fst_903_ = lean_ctor_get(v___x_902_, 0);
lean_inc(v_fst_903_);
v_snd_904_ = lean_ctor_get(v___x_902_, 1);
lean_inc(v_snd_904_);
lean_dec_ref(v___x_902_);
v___x_905_ = lean_nat_dec_le(v_hi_898_, v_fst_903_);
if (v___x_905_ == 0)
{
lean_object* v___x_906_; lean_object* v___x_907_; lean_object* v___x_908_; 
v___x_906_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___redArg(v_n_895_, v_snd_904_, v_lo_897_, v_fst_903_);
v___x_907_ = lean_unsigned_to_nat(1u);
v___x_908_ = lean_nat_add(v_fst_903_, v___x_907_);
lean_dec(v_fst_903_);
v_as_896_ = v___x_906_;
v_lo_897_ = v___x_908_;
goto _start;
}
else
{
lean_dec(v_fst_903_);
lean_dec(v_lo_897_);
return v_snd_904_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___redArg___boxed(lean_object* v_n_930_, lean_object* v_as_931_, lean_object* v_lo_932_, lean_object* v_hi_933_){
_start:
{
lean_object* v_res_934_; 
v_res_934_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___redArg(v_n_930_, v_as_931_, v_lo_932_, v_hi_933_);
lean_dec(v_hi_933_);
lean_dec(v_n_930_);
return v_res_934_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_StatsReport_scriptsCore_spec__0(lean_object* v_as_935_, size_t v_sz_936_, size_t v_i_937_, lean_object* v_b_938_){
_start:
{
lean_object* v_a_940_; uint8_t v___x_944_; 
v___x_944_ = lean_usize_dec_lt(v_i_937_, v_sz_936_);
if (v___x_944_ == 0)
{
return v_b_938_;
}
else
{
lean_object* v_snd_945_; lean_object* v_snd_946_; lean_object* v_snd_947_; lean_object* v_snd_948_; lean_object* v_fst_949_; lean_object* v___x_951_; uint8_t v_isShared_952_; uint8_t v_isSharedCheck_1068_; 
v_snd_945_ = lean_ctor_get(v_b_938_, 1);
lean_inc(v_snd_945_);
v_snd_946_ = lean_ctor_get(v_snd_945_, 1);
lean_inc(v_snd_946_);
v_snd_947_ = lean_ctor_get(v_snd_946_, 1);
lean_inc(v_snd_947_);
v_snd_948_ = lean_ctor_get(v_snd_947_, 1);
lean_inc(v_snd_948_);
v_fst_949_ = lean_ctor_get(v_b_938_, 0);
v_isSharedCheck_1068_ = !lean_is_exclusive(v_b_938_);
if (v_isSharedCheck_1068_ == 0)
{
lean_object* v_unused_1069_; 
v_unused_1069_ = lean_ctor_get(v_b_938_, 1);
lean_dec(v_unused_1069_);
v___x_951_ = v_b_938_;
v_isShared_952_ = v_isSharedCheck_1068_;
goto v_resetjp_950_;
}
else
{
lean_inc(v_fst_949_);
lean_dec(v_b_938_);
v___x_951_ = lean_box(0);
v_isShared_952_ = v_isSharedCheck_1068_;
goto v_resetjp_950_;
}
v_resetjp_950_:
{
lean_object* v_fst_953_; lean_object* v___x_955_; uint8_t v_isShared_956_; uint8_t v_isSharedCheck_1066_; 
v_fst_953_ = lean_ctor_get(v_snd_945_, 0);
v_isSharedCheck_1066_ = !lean_is_exclusive(v_snd_945_);
if (v_isSharedCheck_1066_ == 0)
{
lean_object* v_unused_1067_; 
v_unused_1067_ = lean_ctor_get(v_snd_945_, 1);
lean_dec(v_unused_1067_);
v___x_955_ = v_snd_945_;
v_isShared_956_ = v_isSharedCheck_1066_;
goto v_resetjp_954_;
}
else
{
lean_inc(v_fst_953_);
lean_dec(v_snd_945_);
v___x_955_ = lean_box(0);
v_isShared_956_ = v_isSharedCheck_1066_;
goto v_resetjp_954_;
}
v_resetjp_954_:
{
lean_object* v_fst_957_; lean_object* v___x_959_; uint8_t v_isShared_960_; uint8_t v_isSharedCheck_1064_; 
v_fst_957_ = lean_ctor_get(v_snd_946_, 0);
v_isSharedCheck_1064_ = !lean_is_exclusive(v_snd_946_);
if (v_isSharedCheck_1064_ == 0)
{
lean_object* v_unused_1065_; 
v_unused_1065_ = lean_ctor_get(v_snd_946_, 1);
lean_dec(v_unused_1065_);
v___x_959_ = v_snd_946_;
v_isShared_960_ = v_isSharedCheck_1064_;
goto v_resetjp_958_;
}
else
{
lean_inc(v_fst_957_);
lean_dec(v_snd_946_);
v___x_959_ = lean_box(0);
v_isShared_960_ = v_isSharedCheck_1064_;
goto v_resetjp_958_;
}
v_resetjp_958_:
{
lean_object* v_fst_961_; lean_object* v___x_963_; uint8_t v_isShared_964_; uint8_t v_isSharedCheck_1062_; 
v_fst_961_ = lean_ctor_get(v_snd_947_, 0);
v_isSharedCheck_1062_ = !lean_is_exclusive(v_snd_947_);
if (v_isSharedCheck_1062_ == 0)
{
lean_object* v_unused_1063_; 
v_unused_1063_ = lean_ctor_get(v_snd_947_, 1);
lean_dec(v_unused_1063_);
v___x_963_ = v_snd_947_;
v_isShared_964_ = v_isSharedCheck_1062_;
goto v_resetjp_962_;
}
else
{
lean_inc(v_fst_961_);
lean_dec(v_snd_947_);
v___x_963_ = lean_box(0);
v_isShared_964_ = v_isSharedCheck_1062_;
goto v_resetjp_962_;
}
v_resetjp_962_:
{
lean_object* v_fst_965_; lean_object* v_snd_966_; lean_object* v___x_968_; uint8_t v_isShared_969_; uint8_t v_isSharedCheck_1061_; 
v_fst_965_ = lean_ctor_get(v_snd_948_, 0);
v_snd_966_ = lean_ctor_get(v_snd_948_, 1);
v_isSharedCheck_1061_ = !lean_is_exclusive(v_snd_948_);
if (v_isSharedCheck_1061_ == 0)
{
v___x_968_ = v_snd_948_;
v_isShared_969_ = v_isSharedCheck_1061_;
goto v_resetjp_967_;
}
else
{
lean_inc(v_snd_966_);
lean_inc(v_fst_965_);
lean_dec(v_snd_948_);
v___x_968_ = lean_box(0);
v_isShared_969_ = v_isSharedCheck_1061_;
goto v_resetjp_967_;
}
v_resetjp_967_:
{
lean_object* v_a_970_; lean_object* v_stats_971_; lean_object* v_total_972_; lean_object* v_script_973_; lean_object* v_scriptGenerated_974_; lean_object* v___x_975_; 
v_a_970_ = lean_array_uget_borrowed(v_as_935_, v_i_937_);
v_stats_971_ = lean_ctor_get(v_a_970_, 3);
v_total_972_ = lean_ctor_get(v_stats_971_, 0);
v_script_973_ = lean_ctor_get(v_stats_971_, 5);
v_scriptGenerated_974_ = lean_ctor_get(v_stats_971_, 7);
lean_inc(v_total_972_);
v___x_975_ = lean_array_push(v_fst_965_, v_total_972_);
if (lean_obj_tag(v_scriptGenerated_974_) == 0)
{
lean_object* v___x_977_; 
if (v_isShared_969_ == 0)
{
lean_ctor_set(v___x_968_, 0, v___x_975_);
v___x_977_ = v___x_968_;
goto v_reusejp_976_;
}
else
{
lean_object* v_reuseFailAlloc_990_; 
v_reuseFailAlloc_990_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_990_, 0, v___x_975_);
lean_ctor_set(v_reuseFailAlloc_990_, 1, v_snd_966_);
v___x_977_ = v_reuseFailAlloc_990_;
goto v_reusejp_976_;
}
v_reusejp_976_:
{
lean_object* v___x_979_; 
if (v_isShared_964_ == 0)
{
lean_ctor_set(v___x_963_, 1, v___x_977_);
v___x_979_ = v___x_963_;
goto v_reusejp_978_;
}
else
{
lean_object* v_reuseFailAlloc_989_; 
v_reuseFailAlloc_989_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_989_, 0, v_fst_961_);
lean_ctor_set(v_reuseFailAlloc_989_, 1, v___x_977_);
v___x_979_ = v_reuseFailAlloc_989_;
goto v_reusejp_978_;
}
v_reusejp_978_:
{
lean_object* v___x_981_; 
if (v_isShared_960_ == 0)
{
lean_ctor_set(v___x_959_, 1, v___x_979_);
v___x_981_ = v___x_959_;
goto v_reusejp_980_;
}
else
{
lean_object* v_reuseFailAlloc_988_; 
v_reuseFailAlloc_988_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_988_, 0, v_fst_957_);
lean_ctor_set(v_reuseFailAlloc_988_, 1, v___x_979_);
v___x_981_ = v_reuseFailAlloc_988_;
goto v_reusejp_980_;
}
v_reusejp_980_:
{
lean_object* v___x_983_; 
if (v_isShared_956_ == 0)
{
lean_ctor_set(v___x_955_, 1, v___x_981_);
v___x_983_ = v___x_955_;
goto v_reusejp_982_;
}
else
{
lean_object* v_reuseFailAlloc_987_; 
v_reuseFailAlloc_987_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_987_, 0, v_fst_953_);
lean_ctor_set(v_reuseFailAlloc_987_, 1, v___x_981_);
v___x_983_ = v_reuseFailAlloc_987_;
goto v_reusejp_982_;
}
v_reusejp_982_:
{
lean_object* v___x_985_; 
if (v_isShared_952_ == 0)
{
lean_ctor_set(v___x_951_, 1, v___x_983_);
v___x_985_ = v___x_951_;
goto v_reusejp_984_;
}
else
{
lean_object* v_reuseFailAlloc_986_; 
v_reuseFailAlloc_986_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_986_, 0, v_fst_949_);
lean_ctor_set(v_reuseFailAlloc_986_, 1, v___x_983_);
v___x_985_ = v_reuseFailAlloc_986_;
goto v_reusejp_984_;
}
v_reusejp_984_:
{
v_a_940_ = v___x_985_;
goto v___jp_939_;
}
}
}
}
}
}
else
{
lean_object* v_val_991_; uint8_t v_method_992_; uint8_t v_perfect_993_; lean_object* v___x_994_; 
v_val_991_ = lean_ctor_get(v_scriptGenerated_974_, 0);
v_method_992_ = lean_ctor_get_uint8(v_val_991_, 0);
v_perfect_993_ = lean_ctor_get_uint8(v_val_991_, 1);
lean_inc(v_script_973_);
v___x_994_ = lean_array_push(v_snd_966_, v_script_973_);
if (v_method_992_ == 0)
{
lean_object* v___x_995_; lean_object* v___x_996_; 
v___x_995_ = lean_unsigned_to_nat(1u);
v___x_996_ = lean_nat_add(v_fst_949_, v___x_995_);
lean_dec(v_fst_949_);
if (v_perfect_993_ == 0)
{
lean_object* v___x_998_; 
if (v_isShared_969_ == 0)
{
lean_ctor_set(v___x_968_, 1, v___x_994_);
lean_ctor_set(v___x_968_, 0, v___x_975_);
v___x_998_ = v___x_968_;
goto v_reusejp_997_;
}
else
{
lean_object* v_reuseFailAlloc_1011_; 
v_reuseFailAlloc_1011_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1011_, 0, v___x_975_);
lean_ctor_set(v_reuseFailAlloc_1011_, 1, v___x_994_);
v___x_998_ = v_reuseFailAlloc_1011_;
goto v_reusejp_997_;
}
v_reusejp_997_:
{
lean_object* v___x_1000_; 
if (v_isShared_964_ == 0)
{
lean_ctor_set(v___x_963_, 1, v___x_998_);
v___x_1000_ = v___x_963_;
goto v_reusejp_999_;
}
else
{
lean_object* v_reuseFailAlloc_1010_; 
v_reuseFailAlloc_1010_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1010_, 0, v_fst_961_);
lean_ctor_set(v_reuseFailAlloc_1010_, 1, v___x_998_);
v___x_1000_ = v_reuseFailAlloc_1010_;
goto v_reusejp_999_;
}
v_reusejp_999_:
{
lean_object* v___x_1002_; 
if (v_isShared_960_ == 0)
{
lean_ctor_set(v___x_959_, 1, v___x_1000_);
v___x_1002_ = v___x_959_;
goto v_reusejp_1001_;
}
else
{
lean_object* v_reuseFailAlloc_1009_; 
v_reuseFailAlloc_1009_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1009_, 0, v_fst_957_);
lean_ctor_set(v_reuseFailAlloc_1009_, 1, v___x_1000_);
v___x_1002_ = v_reuseFailAlloc_1009_;
goto v_reusejp_1001_;
}
v_reusejp_1001_:
{
lean_object* v___x_1004_; 
if (v_isShared_956_ == 0)
{
lean_ctor_set(v___x_955_, 1, v___x_1002_);
v___x_1004_ = v___x_955_;
goto v_reusejp_1003_;
}
else
{
lean_object* v_reuseFailAlloc_1008_; 
v_reuseFailAlloc_1008_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1008_, 0, v_fst_953_);
lean_ctor_set(v_reuseFailAlloc_1008_, 1, v___x_1002_);
v___x_1004_ = v_reuseFailAlloc_1008_;
goto v_reusejp_1003_;
}
v_reusejp_1003_:
{
lean_object* v___x_1006_; 
if (v_isShared_952_ == 0)
{
lean_ctor_set(v___x_951_, 1, v___x_1004_);
lean_ctor_set(v___x_951_, 0, v___x_996_);
v___x_1006_ = v___x_951_;
goto v_reusejp_1005_;
}
else
{
lean_object* v_reuseFailAlloc_1007_; 
v_reuseFailAlloc_1007_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1007_, 0, v___x_996_);
lean_ctor_set(v_reuseFailAlloc_1007_, 1, v___x_1004_);
v___x_1006_ = v_reuseFailAlloc_1007_;
goto v_reusejp_1005_;
}
v_reusejp_1005_:
{
v_a_940_ = v___x_1006_;
goto v___jp_939_;
}
}
}
}
}
}
else
{
lean_object* v___x_1012_; lean_object* v___x_1014_; 
v___x_1012_ = lean_nat_add(v_fst_953_, v___x_995_);
lean_dec(v_fst_953_);
if (v_isShared_969_ == 0)
{
lean_ctor_set(v___x_968_, 1, v___x_994_);
lean_ctor_set(v___x_968_, 0, v___x_975_);
v___x_1014_ = v___x_968_;
goto v_reusejp_1013_;
}
else
{
lean_object* v_reuseFailAlloc_1027_; 
v_reuseFailAlloc_1027_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1027_, 0, v___x_975_);
lean_ctor_set(v_reuseFailAlloc_1027_, 1, v___x_994_);
v___x_1014_ = v_reuseFailAlloc_1027_;
goto v_reusejp_1013_;
}
v_reusejp_1013_:
{
lean_object* v___x_1016_; 
if (v_isShared_964_ == 0)
{
lean_ctor_set(v___x_963_, 1, v___x_1014_);
v___x_1016_ = v___x_963_;
goto v_reusejp_1015_;
}
else
{
lean_object* v_reuseFailAlloc_1026_; 
v_reuseFailAlloc_1026_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1026_, 0, v_fst_961_);
lean_ctor_set(v_reuseFailAlloc_1026_, 1, v___x_1014_);
v___x_1016_ = v_reuseFailAlloc_1026_;
goto v_reusejp_1015_;
}
v_reusejp_1015_:
{
lean_object* v___x_1018_; 
if (v_isShared_960_ == 0)
{
lean_ctor_set(v___x_959_, 1, v___x_1016_);
v___x_1018_ = v___x_959_;
goto v_reusejp_1017_;
}
else
{
lean_object* v_reuseFailAlloc_1025_; 
v_reuseFailAlloc_1025_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1025_, 0, v_fst_957_);
lean_ctor_set(v_reuseFailAlloc_1025_, 1, v___x_1016_);
v___x_1018_ = v_reuseFailAlloc_1025_;
goto v_reusejp_1017_;
}
v_reusejp_1017_:
{
lean_object* v___x_1020_; 
if (v_isShared_956_ == 0)
{
lean_ctor_set(v___x_955_, 1, v___x_1018_);
lean_ctor_set(v___x_955_, 0, v___x_1012_);
v___x_1020_ = v___x_955_;
goto v_reusejp_1019_;
}
else
{
lean_object* v_reuseFailAlloc_1024_; 
v_reuseFailAlloc_1024_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1024_, 0, v___x_1012_);
lean_ctor_set(v_reuseFailAlloc_1024_, 1, v___x_1018_);
v___x_1020_ = v_reuseFailAlloc_1024_;
goto v_reusejp_1019_;
}
v_reusejp_1019_:
{
lean_object* v___x_1022_; 
if (v_isShared_952_ == 0)
{
lean_ctor_set(v___x_951_, 1, v___x_1020_);
lean_ctor_set(v___x_951_, 0, v___x_996_);
v___x_1022_ = v___x_951_;
goto v_reusejp_1021_;
}
else
{
lean_object* v_reuseFailAlloc_1023_; 
v_reuseFailAlloc_1023_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1023_, 0, v___x_996_);
lean_ctor_set(v_reuseFailAlloc_1023_, 1, v___x_1020_);
v___x_1022_ = v_reuseFailAlloc_1023_;
goto v_reusejp_1021_;
}
v_reusejp_1021_:
{
v_a_940_ = v___x_1022_;
goto v___jp_939_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_1028_; lean_object* v___x_1029_; 
v___x_1028_ = lean_unsigned_to_nat(1u);
v___x_1029_ = lean_nat_add(v_fst_957_, v___x_1028_);
lean_dec(v_fst_957_);
if (v_perfect_993_ == 0)
{
lean_object* v___x_1031_; 
if (v_isShared_969_ == 0)
{
lean_ctor_set(v___x_968_, 1, v___x_994_);
lean_ctor_set(v___x_968_, 0, v___x_975_);
v___x_1031_ = v___x_968_;
goto v_reusejp_1030_;
}
else
{
lean_object* v_reuseFailAlloc_1044_; 
v_reuseFailAlloc_1044_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1044_, 0, v___x_975_);
lean_ctor_set(v_reuseFailAlloc_1044_, 1, v___x_994_);
v___x_1031_ = v_reuseFailAlloc_1044_;
goto v_reusejp_1030_;
}
v_reusejp_1030_:
{
lean_object* v___x_1033_; 
if (v_isShared_964_ == 0)
{
lean_ctor_set(v___x_963_, 1, v___x_1031_);
v___x_1033_ = v___x_963_;
goto v_reusejp_1032_;
}
else
{
lean_object* v_reuseFailAlloc_1043_; 
v_reuseFailAlloc_1043_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1043_, 0, v_fst_961_);
lean_ctor_set(v_reuseFailAlloc_1043_, 1, v___x_1031_);
v___x_1033_ = v_reuseFailAlloc_1043_;
goto v_reusejp_1032_;
}
v_reusejp_1032_:
{
lean_object* v___x_1035_; 
if (v_isShared_960_ == 0)
{
lean_ctor_set(v___x_959_, 1, v___x_1033_);
lean_ctor_set(v___x_959_, 0, v___x_1029_);
v___x_1035_ = v___x_959_;
goto v_reusejp_1034_;
}
else
{
lean_object* v_reuseFailAlloc_1042_; 
v_reuseFailAlloc_1042_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1042_, 0, v___x_1029_);
lean_ctor_set(v_reuseFailAlloc_1042_, 1, v___x_1033_);
v___x_1035_ = v_reuseFailAlloc_1042_;
goto v_reusejp_1034_;
}
v_reusejp_1034_:
{
lean_object* v___x_1037_; 
if (v_isShared_956_ == 0)
{
lean_ctor_set(v___x_955_, 1, v___x_1035_);
v___x_1037_ = v___x_955_;
goto v_reusejp_1036_;
}
else
{
lean_object* v_reuseFailAlloc_1041_; 
v_reuseFailAlloc_1041_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1041_, 0, v_fst_953_);
lean_ctor_set(v_reuseFailAlloc_1041_, 1, v___x_1035_);
v___x_1037_ = v_reuseFailAlloc_1041_;
goto v_reusejp_1036_;
}
v_reusejp_1036_:
{
lean_object* v___x_1039_; 
if (v_isShared_952_ == 0)
{
lean_ctor_set(v___x_951_, 1, v___x_1037_);
v___x_1039_ = v___x_951_;
goto v_reusejp_1038_;
}
else
{
lean_object* v_reuseFailAlloc_1040_; 
v_reuseFailAlloc_1040_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1040_, 0, v_fst_949_);
lean_ctor_set(v_reuseFailAlloc_1040_, 1, v___x_1037_);
v___x_1039_ = v_reuseFailAlloc_1040_;
goto v_reusejp_1038_;
}
v_reusejp_1038_:
{
v_a_940_ = v___x_1039_;
goto v___jp_939_;
}
}
}
}
}
}
else
{
lean_object* v___x_1045_; lean_object* v___x_1047_; 
v___x_1045_ = lean_nat_add(v_fst_961_, v___x_1028_);
lean_dec(v_fst_961_);
if (v_isShared_969_ == 0)
{
lean_ctor_set(v___x_968_, 1, v___x_994_);
lean_ctor_set(v___x_968_, 0, v___x_975_);
v___x_1047_ = v___x_968_;
goto v_reusejp_1046_;
}
else
{
lean_object* v_reuseFailAlloc_1060_; 
v_reuseFailAlloc_1060_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1060_, 0, v___x_975_);
lean_ctor_set(v_reuseFailAlloc_1060_, 1, v___x_994_);
v___x_1047_ = v_reuseFailAlloc_1060_;
goto v_reusejp_1046_;
}
v_reusejp_1046_:
{
lean_object* v___x_1049_; 
if (v_isShared_964_ == 0)
{
lean_ctor_set(v___x_963_, 1, v___x_1047_);
lean_ctor_set(v___x_963_, 0, v___x_1045_);
v___x_1049_ = v___x_963_;
goto v_reusejp_1048_;
}
else
{
lean_object* v_reuseFailAlloc_1059_; 
v_reuseFailAlloc_1059_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1059_, 0, v___x_1045_);
lean_ctor_set(v_reuseFailAlloc_1059_, 1, v___x_1047_);
v___x_1049_ = v_reuseFailAlloc_1059_;
goto v_reusejp_1048_;
}
v_reusejp_1048_:
{
lean_object* v___x_1051_; 
if (v_isShared_960_ == 0)
{
lean_ctor_set(v___x_959_, 1, v___x_1049_);
lean_ctor_set(v___x_959_, 0, v___x_1029_);
v___x_1051_ = v___x_959_;
goto v_reusejp_1050_;
}
else
{
lean_object* v_reuseFailAlloc_1058_; 
v_reuseFailAlloc_1058_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1058_, 0, v___x_1029_);
lean_ctor_set(v_reuseFailAlloc_1058_, 1, v___x_1049_);
v___x_1051_ = v_reuseFailAlloc_1058_;
goto v_reusejp_1050_;
}
v_reusejp_1050_:
{
lean_object* v___x_1053_; 
if (v_isShared_956_ == 0)
{
lean_ctor_set(v___x_955_, 1, v___x_1051_);
v___x_1053_ = v___x_955_;
goto v_reusejp_1052_;
}
else
{
lean_object* v_reuseFailAlloc_1057_; 
v_reuseFailAlloc_1057_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1057_, 0, v_fst_953_);
lean_ctor_set(v_reuseFailAlloc_1057_, 1, v___x_1051_);
v___x_1053_ = v_reuseFailAlloc_1057_;
goto v_reusejp_1052_;
}
v_reusejp_1052_:
{
lean_object* v___x_1055_; 
if (v_isShared_952_ == 0)
{
lean_ctor_set(v___x_951_, 1, v___x_1053_);
v___x_1055_ = v___x_951_;
goto v_reusejp_1054_;
}
else
{
lean_object* v_reuseFailAlloc_1056_; 
v_reuseFailAlloc_1056_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1056_, 0, v_fst_949_);
lean_ctor_set(v_reuseFailAlloc_1056_, 1, v___x_1053_);
v___x_1055_ = v_reuseFailAlloc_1056_;
goto v_reusejp_1054_;
}
v_reusejp_1054_:
{
v_a_940_ = v___x_1055_;
goto v___jp_939_;
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
}
}
}
}
v___jp_939_:
{
size_t v___x_941_; size_t v___x_942_; 
v___x_941_ = ((size_t)1ULL);
v___x_942_ = lean_usize_add(v_i_937_, v___x_941_);
v_i_937_ = v___x_942_;
v_b_938_ = v_a_940_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_StatsReport_scriptsCore_spec__0___boxed(lean_object* v_as_1070_, lean_object* v_sz_1071_, lean_object* v_i_1072_, lean_object* v_b_1073_){
_start:
{
size_t v_sz_boxed_1074_; size_t v_i_boxed_1075_; lean_object* v_res_1076_; 
v_sz_boxed_1074_ = lean_unbox_usize(v_sz_1071_);
lean_dec(v_sz_1071_);
v_i_boxed_1075_ = lean_unbox_usize(v_i_1072_);
lean_dec(v_i_1072_);
v_res_1076_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_StatsReport_scriptsCore_spec__0(v_as_1070_, v_sz_boxed_1074_, v_i_boxed_1075_, v_b_1073_);
lean_dec_ref(v_as_1070_);
return v_res_1076_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_StatsReport_scriptsCore_spec__1___redArg(lean_object* v_a_1077_, lean_object* v_b_1078_){
_start:
{
lean_object* v_array_1079_; lean_object* v_start_1080_; lean_object* v_stop_1081_; lean_object* v___x_1083_; uint8_t v_isShared_1084_; uint8_t v_isSharedCheck_1094_; 
v_array_1079_ = lean_ctor_get(v_a_1077_, 0);
v_start_1080_ = lean_ctor_get(v_a_1077_, 1);
v_stop_1081_ = lean_ctor_get(v_a_1077_, 2);
v_isSharedCheck_1094_ = !lean_is_exclusive(v_a_1077_);
if (v_isSharedCheck_1094_ == 0)
{
v___x_1083_ = v_a_1077_;
v_isShared_1084_ = v_isSharedCheck_1094_;
goto v_resetjp_1082_;
}
else
{
lean_inc(v_stop_1081_);
lean_inc(v_start_1080_);
lean_inc(v_array_1079_);
lean_dec(v_a_1077_);
v___x_1083_ = lean_box(0);
v_isShared_1084_ = v_isSharedCheck_1094_;
goto v_resetjp_1082_;
}
v_resetjp_1082_:
{
uint8_t v___x_1085_; 
v___x_1085_ = lean_nat_dec_lt(v_start_1080_, v_stop_1081_);
if (v___x_1085_ == 0)
{
lean_del_object(v___x_1083_);
lean_dec(v_stop_1081_);
lean_dec(v_start_1080_);
lean_dec_ref(v_array_1079_);
return v_b_1078_;
}
else
{
lean_object* v___x_1086_; lean_object* v___x_1087_; lean_object* v___x_1089_; 
v___x_1086_ = lean_unsigned_to_nat(1u);
v___x_1087_ = lean_nat_add(v_start_1080_, v___x_1086_);
lean_inc_ref(v_array_1079_);
if (v_isShared_1084_ == 0)
{
lean_ctor_set(v___x_1083_, 1, v___x_1087_);
v___x_1089_ = v___x_1083_;
goto v_reusejp_1088_;
}
else
{
lean_object* v_reuseFailAlloc_1093_; 
v_reuseFailAlloc_1093_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1093_, 0, v_array_1079_);
lean_ctor_set(v_reuseFailAlloc_1093_, 1, v___x_1087_);
lean_ctor_set(v_reuseFailAlloc_1093_, 2, v_stop_1081_);
v___x_1089_ = v_reuseFailAlloc_1093_;
goto v_reusejp_1088_;
}
v_reusejp_1088_:
{
lean_object* v___x_1090_; lean_object* v___x_1091_; 
v___x_1090_ = lean_array_fget(v_array_1079_, v_start_1080_);
lean_dec(v_start_1080_);
lean_dec_ref(v_array_1079_);
v___x_1091_ = lean_array_push(v_b_1078_, v___x_1090_);
v_a_1077_ = v___x_1089_;
v_b_1078_ = v___x_1091_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2(size_t v_sz_1110_, size_t v_i_1111_, lean_object* v_bs_1112_){
_start:
{
uint8_t v___x_1113_; 
v___x_1113_ = lean_usize_dec_lt(v_i_1111_, v_sz_1110_);
if (v___x_1113_ == 0)
{
return v_bs_1112_;
}
else
{
lean_object* v_v_1114_; lean_object* v_position_x3f_1115_; lean_object* v___x_1116_; lean_object* v_bs_x27_1117_; lean_object* v___y_1119_; 
v_v_1114_ = lean_array_uget(v_bs_1112_, v_i_1111_);
v_position_x3f_1115_ = lean_ctor_get(v_v_1114_, 2);
lean_inc(v_position_x3f_1115_);
v___x_1116_ = lean_unsigned_to_nat(0u);
v_bs_x27_1117_ = lean_array_uset(v_bs_1112_, v_i_1111_, v___x_1116_);
if (lean_obj_tag(v_position_x3f_1115_) == 0)
{
lean_object* v___x_1147_; 
v___x_1147_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__9));
v___y_1119_ = v___x_1147_;
goto v___jp_1118_;
}
else
{
lean_object* v_val_1148_; lean_object* v___x_1150_; uint8_t v_isShared_1151_; uint8_t v_isSharedCheck_1169_; 
v_val_1148_ = lean_ctor_get(v_position_x3f_1115_, 0);
v_isSharedCheck_1169_ = !lean_is_exclusive(v_position_x3f_1115_);
if (v_isSharedCheck_1169_ == 0)
{
v___x_1150_ = v_position_x3f_1115_;
v_isShared_1151_ = v_isSharedCheck_1169_;
goto v_resetjp_1149_;
}
else
{
lean_inc(v_val_1148_);
lean_dec(v_position_x3f_1115_);
v___x_1150_ = lean_box(0);
v_isShared_1151_ = v_isSharedCheck_1169_;
goto v_resetjp_1149_;
}
v_resetjp_1149_:
{
lean_object* v_line_1152_; lean_object* v_column_1153_; lean_object* v___x_1155_; uint8_t v_isShared_1156_; uint8_t v_isSharedCheck_1168_; 
v_line_1152_ = lean_ctor_get(v_val_1148_, 0);
v_column_1153_ = lean_ctor_get(v_val_1148_, 1);
v_isSharedCheck_1168_ = !lean_is_exclusive(v_val_1148_);
if (v_isSharedCheck_1168_ == 0)
{
v___x_1155_ = v_val_1148_;
v_isShared_1156_ = v_isSharedCheck_1168_;
goto v_resetjp_1154_;
}
else
{
lean_inc(v_column_1153_);
lean_inc(v_line_1152_);
lean_dec(v_val_1148_);
v___x_1155_ = lean_box(0);
v_isShared_1156_ = v_isSharedCheck_1168_;
goto v_resetjp_1154_;
}
v_resetjp_1154_:
{
lean_object* v___x_1157_; lean_object* v___x_1159_; 
v___x_1157_ = l_Nat_reprFast(v_line_1152_);
if (v_isShared_1151_ == 0)
{
lean_ctor_set_tag(v___x_1150_, 3);
lean_ctor_set(v___x_1150_, 0, v___x_1157_);
v___x_1159_ = v___x_1150_;
goto v_reusejp_1158_;
}
else
{
lean_object* v_reuseFailAlloc_1167_; 
v_reuseFailAlloc_1167_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1167_, 0, v___x_1157_);
v___x_1159_ = v_reuseFailAlloc_1167_;
goto v_reusejp_1158_;
}
v_reusejp_1158_:
{
lean_object* v___x_1160_; lean_object* v___x_1162_; 
v___x_1160_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__1));
if (v_isShared_1156_ == 0)
{
lean_ctor_set_tag(v___x_1155_, 5);
lean_ctor_set(v___x_1155_, 1, v___x_1160_);
lean_ctor_set(v___x_1155_, 0, v___x_1159_);
v___x_1162_ = v___x_1155_;
goto v_reusejp_1161_;
}
else
{
lean_object* v_reuseFailAlloc_1166_; 
v_reuseFailAlloc_1166_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1166_, 0, v___x_1159_);
lean_ctor_set(v_reuseFailAlloc_1166_, 1, v___x_1160_);
v___x_1162_ = v_reuseFailAlloc_1166_;
goto v_reusejp_1161_;
}
v_reusejp_1161_:
{
lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; 
v___x_1163_ = l_Nat_reprFast(v_column_1153_);
v___x_1164_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1164_, 0, v___x_1163_);
v___x_1165_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1165_, 0, v___x_1162_);
lean_ctor_set(v___x_1165_, 1, v___x_1164_);
v___y_1119_ = v___x_1165_;
goto v___jp_1118_;
}
}
}
}
}
v___jp_1118_:
{
lean_object* v_stats_1120_; lean_object* v_fileName_1121_; lean_object* v_total_1122_; lean_object* v_script_1123_; lean_object* v_scriptGenerated_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; lean_object* v___x_1128_; lean_object* v___x_1129_; lean_object* v___x_1130_; lean_object* v___x_1131_; lean_object* v___x_1132_; lean_object* v___x_1133_; lean_object* v___x_1134_; lean_object* v___x_1135_; lean_object* v___x_1136_; lean_object* v___x_1137_; lean_object* v___x_1138_; lean_object* v___x_1139_; lean_object* v___x_1140_; lean_object* v___x_1141_; lean_object* v___x_1142_; size_t v___x_1143_; size_t v___x_1144_; lean_object* v___x_1145_; 
v_stats_1120_ = lean_ctor_get(v_v_1114_, 3);
lean_inc_ref(v_stats_1120_);
v_fileName_1121_ = lean_ctor_get(v_v_1114_, 1);
lean_inc_ref(v_fileName_1121_);
lean_dec(v_v_1114_);
v_total_1122_ = lean_ctor_get(v_stats_1120_, 0);
lean_inc(v_total_1122_);
v_script_1123_ = lean_ctor_get(v_stats_1120_, 5);
lean_inc(v_script_1123_);
v_scriptGenerated_1124_ = lean_ctor_get(v_stats_1120_, 7);
lean_inc(v_scriptGenerated_1124_);
lean_dec_ref(v_stats_1120_);
v___x_1125_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1125_, 0, v_fileName_1121_);
v___x_1126_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__1));
v___x_1127_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1127_, 0, v___x_1125_);
lean_ctor_set(v___x_1127_, 1, v___x_1126_);
v___x_1128_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1128_, 0, v___x_1127_);
lean_ctor_set(v___x_1128_, 1, v___y_1119_);
v___x_1129_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__3));
v___x_1130_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1130_, 0, v___x_1128_);
lean_ctor_set(v___x_1130_, 1, v___x_1129_);
v___x_1131_ = lp_aesop_Aesop_Nanos_printAsMillis(v_script_1123_);
v___x_1132_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1132_, 0, v___x_1131_);
v___x_1133_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1133_, 0, v___x_1130_);
lean_ctor_set(v___x_1133_, 1, v___x_1132_);
v___x_1134_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__5));
v___x_1135_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1135_, 0, v___x_1133_);
lean_ctor_set(v___x_1135_, 1, v___x_1134_);
v___x_1136_ = lp_aesop_Aesop_Nanos_printAsMillis(v_total_1122_);
v___x_1137_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1137_, 0, v___x_1136_);
v___x_1138_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1138_, 0, v___x_1135_);
lean_ctor_set(v___x_1138_, 1, v___x_1137_);
v___x_1139_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___closed__7));
v___x_1140_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1140_, 0, v___x_1138_);
lean_ctor_set(v___x_1140_, 1, v___x_1139_);
v___x_1141_ = lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtScriptGenerated(v_scriptGenerated_1124_);
v___x_1142_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1142_, 0, v___x_1140_);
lean_ctor_set(v___x_1142_, 1, v___x_1141_);
v___x_1143_ = ((size_t)1ULL);
v___x_1144_ = lean_usize_add(v_i_1111_, v___x_1143_);
v___x_1145_ = lean_array_uset(v_bs_x27_1117_, v_i_1111_, v___x_1142_);
v_i_1111_ = v___x_1144_;
v_bs_1112_ = v___x_1145_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2___boxed(lean_object* v_sz_1170_, lean_object* v_i_1171_, lean_object* v_bs_1172_){
_start:
{
size_t v_sz_boxed_1173_; size_t v_i_boxed_1174_; lean_object* v_res_1175_; 
v_sz_boxed_1173_ = lean_unbox_usize(v_sz_1170_);
lean_dec(v_sz_1170_);
v_i_boxed_1174_ = lean_unbox_usize(v_i_1171_);
lean_dec(v_i_1171_);
v_res_1175_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2(v_sz_boxed_1173_, v_i_boxed_1174_, v_bs_1172_);
return v_res_1175_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_StatsReport_scriptsCore_spec__5(lean_object* v_as_1176_, size_t v_i_1177_, size_t v_stop_1178_, lean_object* v_b_1179_){
_start:
{
lean_object* v___y_1181_; uint8_t v___x_1185_; 
v___x_1185_ = lean_usize_dec_eq(v_i_1177_, v_stop_1178_);
if (v___x_1185_ == 0)
{
lean_object* v___x_1186_; lean_object* v_stats_1187_; lean_object* v_scriptGenerated_1188_; 
v___x_1186_ = lean_array_uget_borrowed(v_as_1176_, v_i_1177_);
v_stats_1187_ = lean_ctor_get(v___x_1186_, 3);
v_scriptGenerated_1188_ = lean_ctor_get(v_stats_1187_, 7);
if (lean_obj_tag(v_scriptGenerated_1188_) == 0)
{
v___y_1181_ = v_b_1179_;
goto v___jp_1180_;
}
else
{
lean_object* v_val_1189_; uint8_t v_hasMVar_1190_; 
v_val_1189_ = lean_ctor_get(v_scriptGenerated_1188_, 0);
v_hasMVar_1190_ = lean_ctor_get_uint8(v_val_1189_, 2);
if (v_hasMVar_1190_ == 0)
{
v___y_1181_ = v_b_1179_;
goto v___jp_1180_;
}
else
{
lean_object* v___x_1191_; 
lean_inc(v___x_1186_);
v___x_1191_ = lean_array_push(v_b_1179_, v___x_1186_);
v___y_1181_ = v___x_1191_;
goto v___jp_1180_;
}
}
}
else
{
return v_b_1179_;
}
v___jp_1180_:
{
size_t v___x_1182_; size_t v___x_1183_; 
v___x_1182_ = ((size_t)1ULL);
v___x_1183_ = lean_usize_add(v_i_1177_, v___x_1182_);
v_i_1177_ = v___x_1183_;
v_b_1179_ = v___y_1181_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_StatsReport_scriptsCore_spec__5___boxed(lean_object* v_as_1192_, lean_object* v_i_1193_, lean_object* v_stop_1194_, lean_object* v_b_1195_){
_start:
{
size_t v_i_boxed_1196_; size_t v_stop_boxed_1197_; lean_object* v_res_1198_; 
v_i_boxed_1196_ = lean_unbox_usize(v_i_1193_);
lean_dec(v_i_1193_);
v_stop_boxed_1197_ = lean_unbox_usize(v_stop_1194_);
lean_dec(v_stop_1194_);
v_res_1198_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_StatsReport_scriptsCore_spec__5(v_as_1192_, v_i_boxed_1196_, v_stop_boxed_1197_, v_b_1195_);
lean_dec_ref(v_as_1192_);
return v_res_1198_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsReport_scriptsCore(lean_object* v_nSlowest_1231_, uint8_t v_nontrivialOnly_1232_, lean_object* v_statsArray_1233_){
_start:
{
lean_object* v___y_1235_; lean_object* v___y_1236_; lean_object* v___y_1237_; lean_object* v___y_1238_; lean_object* v___y_1239_; lean_object* v___y_1240_; lean_object* v___y_1241_; lean_object* v___y_1242_; lean_object* v___y_1243_; lean_object* v___y_1244_; size_t v___y_1293_; lean_object* v___y_1294_; lean_object* v___y_1295_; lean_object* v___y_1296_; lean_object* v___y_1297_; lean_object* v___y_1298_; lean_object* v___y_1299_; lean_object* v___y_1300_; lean_object* v___y_1301_; lean_object* v___y_1302_; lean_object* v___y_1314_; size_t v___y_1315_; lean_object* v___y_1316_; lean_object* v___y_1317_; lean_object* v___y_1318_; lean_object* v___y_1319_; lean_object* v___y_1320_; lean_object* v___y_1321_; lean_object* v___y_1322_; lean_object* v___y_1323_; lean_object* v___y_1330_; size_t v___y_1331_; lean_object* v___y_1332_; lean_object* v___y_1333_; lean_object* v___y_1334_; lean_object* v___y_1335_; lean_object* v___y_1336_; lean_object* v___y_1337_; lean_object* v___y_1338_; lean_object* v___y_1339_; lean_object* v___y_1340_; lean_object* v___y_1341_; lean_object* v___y_1344_; size_t v___y_1345_; lean_object* v___y_1346_; lean_object* v___y_1347_; lean_object* v___y_1348_; lean_object* v___y_1349_; lean_object* v___y_1350_; lean_object* v___y_1351_; lean_object* v___y_1352_; lean_object* v___y_1353_; lean_object* v___y_1354_; lean_object* v___y_1355_; lean_object* v___y_1358_; 
if (v_nontrivialOnly_1232_ == 0)
{
v___y_1358_ = v_statsArray_1233_;
goto v___jp_1357_;
}
else
{
lean_object* v___x_1384_; lean_object* v___x_1385_; lean_object* v___x_1386_; uint8_t v___x_1387_; 
v___x_1384_ = lean_unsigned_to_nat(0u);
v___x_1385_ = lean_array_get_size(v_statsArray_1233_);
v___x_1386_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_scriptsCore___closed__20));
v___x_1387_ = lean_nat_dec_lt(v___x_1384_, v___x_1385_);
if (v___x_1387_ == 0)
{
lean_dec_ref(v_statsArray_1233_);
v___y_1358_ = v___x_1386_;
goto v___jp_1357_;
}
else
{
uint8_t v___x_1388_; 
v___x_1388_ = lean_nat_dec_le(v___x_1385_, v___x_1385_);
if (v___x_1388_ == 0)
{
if (v___x_1387_ == 0)
{
lean_dec_ref(v_statsArray_1233_);
v___y_1358_ = v___x_1386_;
goto v___jp_1357_;
}
else
{
size_t v___x_1389_; size_t v___x_1390_; lean_object* v___x_1391_; 
v___x_1389_ = ((size_t)0ULL);
v___x_1390_ = lean_usize_of_nat(v___x_1385_);
v___x_1391_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_StatsReport_scriptsCore_spec__5(v_statsArray_1233_, v___x_1389_, v___x_1390_, v___x_1386_);
lean_dec_ref(v_statsArray_1233_);
v___y_1358_ = v___x_1391_;
goto v___jp_1357_;
}
}
else
{
size_t v___x_1392_; size_t v___x_1393_; lean_object* v___x_1394_; 
v___x_1392_ = ((size_t)0ULL);
v___x_1393_ = lean_usize_of_nat(v___x_1385_);
v___x_1394_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_StatsReport_scriptsCore_spec__5(v_statsArray_1233_, v___x_1392_, v___x_1393_, v___x_1386_);
lean_dec_ref(v_statsArray_1233_);
v___y_1358_ = v___x_1394_;
goto v___jp_1357_;
}
}
}
v___jp_1234_:
{
lean_object* v___x_1245_; lean_object* v___x_1246_; lean_object* v___x_1247_; lean_object* v___x_1248_; lean_object* v___x_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; lean_object* v___x_1252_; lean_object* v___x_1253_; lean_object* v___x_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; lean_object* v___x_1260_; lean_object* v___x_1261_; lean_object* v___x_1262_; lean_object* v___x_1263_; lean_object* v___x_1264_; lean_object* v___x_1265_; lean_object* v___x_1266_; lean_object* v___x_1267_; lean_object* v___x_1268_; lean_object* v___x_1269_; lean_object* v___x_1270_; lean_object* v___x_1271_; lean_object* v___x_1272_; lean_object* v___x_1273_; lean_object* v___x_1274_; lean_object* v___x_1275_; lean_object* v___x_1276_; lean_object* v___x_1277_; lean_object* v___x_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1283_; lean_object* v___x_1284_; lean_object* v___x_1285_; lean_object* v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1288_; lean_object* v___x_1289_; lean_object* v___x_1290_; lean_object* v___x_1291_; 
lean_inc(v___y_1244_);
v___x_1245_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1245_, 0, v___y_1239_);
lean_ctor_set(v___x_1245_, 1, v___y_1244_);
v___x_1246_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_scriptsCore___closed__1));
v___x_1247_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1247_, 0, v___x_1245_);
lean_ctor_set(v___x_1247_, 1, v___x_1246_);
v___x_1248_ = lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes(v___y_1241_);
v___x_1249_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1249_, 0, v___x_1247_);
lean_ctor_set(v___x_1249_, 1, v___x_1248_);
v___x_1250_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_scriptsCore___closed__3));
v___x_1251_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1251_, 0, v___x_1249_);
lean_ctor_set(v___x_1251_, 1, v___x_1250_);
lean_inc(v___y_1238_);
v___x_1252_ = lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_scriptsCore_fmtTimes(v___y_1238_);
v___x_1253_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1253_, 0, v___x_1251_);
lean_ctor_set(v___x_1253_, 1, v___x_1252_);
v___x_1254_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_scriptsCore___closed__5));
v___x_1255_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1255_, 0, v___x_1253_);
lean_ctor_set(v___x_1255_, 1, v___x_1254_);
v___x_1256_ = lean_array_get_size(v___y_1238_);
lean_dec(v___y_1238_);
v___x_1257_ = l_Nat_reprFast(v___x_1256_);
v___x_1258_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1258_, 0, v___x_1257_);
v___x_1259_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1259_, 0, v___x_1255_);
lean_ctor_set(v___x_1259_, 1, v___x_1258_);
v___x_1260_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_scriptsCore___closed__7));
v___x_1261_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1261_, 0, v___x_1259_);
lean_ctor_set(v___x_1261_, 1, v___x_1260_);
v___x_1262_ = l_Nat_reprFast(v___y_1242_);
v___x_1263_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1263_, 0, v___x_1262_);
v___x_1264_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1264_, 0, v___x_1261_);
lean_ctor_set(v___x_1264_, 1, v___x_1263_);
v___x_1265_ = ((lean_object*)(lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats_spec__0___closed__7));
v___x_1266_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1266_, 0, v___x_1264_);
lean_ctor_set(v___x_1266_, 1, v___x_1265_);
v___x_1267_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_scriptsCore___closed__9));
v___x_1268_ = l_Nat_reprFast(v___y_1235_);
v___x_1269_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1269_, 0, v___x_1268_);
v___x_1270_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1270_, 0, v___x_1267_);
lean_ctor_set(v___x_1270_, 1, v___x_1269_);
v___x_1271_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_scriptsCore___closed__11));
v___x_1272_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1272_, 0, v___x_1270_);
lean_ctor_set(v___x_1272_, 1, v___x_1271_);
v___x_1273_ = l_Nat_reprFast(v___y_1236_);
v___x_1274_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1274_, 0, v___x_1273_);
v___x_1275_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1275_, 0, v___x_1272_);
lean_ctor_set(v___x_1275_, 1, v___x_1274_);
v___x_1276_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1276_, 0, v___x_1275_);
lean_ctor_set(v___x_1276_, 1, v___x_1265_);
v___x_1277_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1277_, 0, v___x_1266_);
lean_ctor_set(v___x_1277_, 1, v___x_1276_);
v___x_1278_ = l_Nat_reprFast(v___y_1243_);
v___x_1279_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1279_, 0, v___x_1278_);
v___x_1280_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1280_, 0, v___x_1267_);
lean_ctor_set(v___x_1280_, 1, v___x_1279_);
v___x_1281_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_scriptsCore___closed__13));
v___x_1282_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1282_, 0, v___x_1280_);
lean_ctor_set(v___x_1282_, 1, v___x_1281_);
v___x_1283_ = l_Nat_reprFast(v___y_1237_);
v___x_1284_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1284_, 0, v___x_1283_);
v___x_1285_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1285_, 0, v___x_1282_);
lean_ctor_set(v___x_1285_, 1, v___x_1284_);
v___x_1286_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_scriptsCore___closed__15));
v___x_1287_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1287_, 0, v___x_1285_);
lean_ctor_set(v___x_1287_, 1, v___x_1286_);
v___x_1288_ = lean_array_to_list(v___y_1240_);
v___x_1289_ = lp_aesop_Std_Format_joinSep___at___00Aesop_StatsReport_scriptsCore_spec__3(v___x_1288_, v___x_1265_);
v___x_1290_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1290_, 0, v___x_1287_);
lean_ctor_set(v___x_1290_, 1, v___x_1289_);
v___x_1291_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1291_, 0, v___x_1277_);
lean_ctor_set(v___x_1291_, 1, v___x_1290_);
return v___x_1291_;
}
v___jp_1292_:
{
size_t v_sz_1303_; lean_object* v___x_1304_; lean_object* v___x_1305_; lean_object* v___x_1306_; lean_object* v___x_1307_; lean_object* v___x_1308_; lean_object* v___x_1309_; lean_object* v___x_1310_; 
v_sz_1303_ = lean_array_size(v___y_1295_);
v___x_1304_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Aesop_StatsReport_scriptsCore_spec__2(v_sz_1303_, v___y_1293_, v___y_1295_);
v___x_1305_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_default___closed__10));
v___x_1306_ = l_Nat_reprFast(v___y_1298_);
v___x_1307_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_1307_, 0, v___x_1306_);
v___x_1308_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1308_, 0, v___x_1305_);
lean_ctor_set(v___x_1308_, 1, v___x_1307_);
v___x_1309_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_scriptsCore___closed__17));
v___x_1310_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_1310_, 0, v___x_1308_);
lean_ctor_set(v___x_1310_, 1, v___x_1309_);
if (v_nontrivialOnly_1232_ == 0)
{
lean_object* v___x_1311_; 
v___x_1311_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Report_0__Aesop_StatsReport_default_fmtRuleStats___closed__1));
v___y_1235_ = v___y_1294_;
v___y_1236_ = v___y_1296_;
v___y_1237_ = v___y_1302_;
v___y_1238_ = v___y_1297_;
v___y_1239_ = v___x_1310_;
v___y_1240_ = v___x_1304_;
v___y_1241_ = v___y_1299_;
v___y_1242_ = v___y_1300_;
v___y_1243_ = v___y_1301_;
v___y_1244_ = v___x_1311_;
goto v___jp_1234_;
}
else
{
lean_object* v___x_1312_; 
v___x_1312_ = ((lean_object*)(lp_aesop_Aesop_StatsReport_scriptsCore___closed__19));
v___y_1235_ = v___y_1294_;
v___y_1236_ = v___y_1296_;
v___y_1237_ = v___y_1302_;
v___y_1238_ = v___y_1297_;
v___y_1239_ = v___x_1310_;
v___y_1240_ = v___x_1304_;
v___y_1241_ = v___y_1299_;
v___y_1242_ = v___y_1300_;
v___y_1243_ = v___y_1301_;
v___y_1244_ = v___x_1312_;
goto v___jp_1234_;
}
}
v___jp_1313_:
{
lean_object* v___x_1324_; lean_object* v___x_1325_; lean_object* v___x_1326_; lean_object* v___x_1327_; uint8_t v___x_1328_; 
lean_inc(v_nSlowest_1231_);
lean_inc(v___y_1314_);
v___x_1324_ = l_Array_toSubarray___redArg(v___y_1323_, v___y_1314_, v_nSlowest_1231_);
v___x_1325_ = lean_mk_empty_array_with_capacity(v___y_1314_);
lean_dec(v___y_1314_);
v___x_1326_ = lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_StatsReport_scriptsCore_spec__1___redArg(v___x_1324_, v___x_1325_);
v___x_1327_ = lean_array_get_size(v___x_1326_);
v___x_1328_ = lean_nat_dec_le(v___x_1327_, v_nSlowest_1231_);
if (v___x_1328_ == 0)
{
v___y_1293_ = v___y_1315_;
v___y_1294_ = v___y_1316_;
v___y_1295_ = v___x_1326_;
v___y_1296_ = v___y_1317_;
v___y_1297_ = v___y_1319_;
v___y_1298_ = v___y_1318_;
v___y_1299_ = v___y_1320_;
v___y_1300_ = v___y_1321_;
v___y_1301_ = v___y_1322_;
v___y_1302_ = v_nSlowest_1231_;
goto v___jp_1292_;
}
else
{
lean_dec(v_nSlowest_1231_);
v___y_1293_ = v___y_1315_;
v___y_1294_ = v___y_1316_;
v___y_1295_ = v___x_1326_;
v___y_1296_ = v___y_1317_;
v___y_1297_ = v___y_1319_;
v___y_1298_ = v___y_1318_;
v___y_1299_ = v___y_1320_;
v___y_1300_ = v___y_1321_;
v___y_1301_ = v___y_1322_;
v___y_1302_ = v___x_1327_;
goto v___jp_1292_;
}
}
v___jp_1329_:
{
lean_object* v___x_1342_; 
v___x_1342_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___redArg(v___y_1337_, v___y_1334_, v___y_1336_, v___y_1341_);
lean_dec(v___y_1341_);
v___y_1314_ = v___y_1330_;
v___y_1315_ = v___y_1331_;
v___y_1316_ = v___y_1332_;
v___y_1317_ = v___y_1333_;
v___y_1318_ = v___y_1337_;
v___y_1319_ = v___y_1335_;
v___y_1320_ = v___y_1338_;
v___y_1321_ = v___y_1339_;
v___y_1322_ = v___y_1340_;
v___y_1323_ = v___x_1342_;
goto v___jp_1313_;
}
v___jp_1343_:
{
uint8_t v___x_1356_; 
v___x_1356_ = lean_nat_dec_le(v___y_1355_, v___y_1354_);
if (v___x_1356_ == 0)
{
lean_dec(v___y_1354_);
lean_inc(v___y_1355_);
v___y_1330_ = v___y_1344_;
v___y_1331_ = v___y_1345_;
v___y_1332_ = v___y_1346_;
v___y_1333_ = v___y_1347_;
v___y_1334_ = v___y_1348_;
v___y_1335_ = v___y_1350_;
v___y_1336_ = v___y_1355_;
v___y_1337_ = v___y_1349_;
v___y_1338_ = v___y_1351_;
v___y_1339_ = v___y_1352_;
v___y_1340_ = v___y_1353_;
v___y_1341_ = v___y_1355_;
goto v___jp_1329_;
}
else
{
v___y_1330_ = v___y_1344_;
v___y_1331_ = v___y_1345_;
v___y_1332_ = v___y_1346_;
v___y_1333_ = v___y_1347_;
v___y_1334_ = v___y_1348_;
v___y_1335_ = v___y_1350_;
v___y_1336_ = v___y_1355_;
v___y_1337_ = v___y_1349_;
v___y_1338_ = v___y_1351_;
v___y_1339_ = v___y_1352_;
v___y_1340_ = v___y_1353_;
v___y_1341_ = v___y_1354_;
goto v___jp_1329_;
}
}
v___jp_1357_:
{
lean_object* v_staticallyStructured_1359_; lean_object* v___x_1360_; lean_object* v_totalTimes_1361_; lean_object* v___x_1362_; lean_object* v___x_1363_; lean_object* v___x_1364_; lean_object* v___x_1365_; lean_object* v___x_1366_; size_t v_sz_1367_; size_t v___x_1368_; lean_object* v___x_1369_; lean_object* v_snd_1370_; lean_object* v_snd_1371_; lean_object* v_snd_1372_; lean_object* v_snd_1373_; lean_object* v_fst_1374_; lean_object* v_fst_1375_; lean_object* v_fst_1376_; lean_object* v_fst_1377_; lean_object* v_fst_1378_; lean_object* v_snd_1379_; uint8_t v___x_1380_; 
v_staticallyStructured_1359_ = lean_unsigned_to_nat(0u);
v___x_1360_ = lean_array_get_size(v___y_1358_);
v_totalTimes_1361_ = lean_mk_empty_array_with_capacity(v___x_1360_);
lean_inc_ref(v_totalTimes_1361_);
v___x_1362_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1362_, 0, v_totalTimes_1361_);
lean_ctor_set(v___x_1362_, 1, v_totalTimes_1361_);
v___x_1363_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1363_, 0, v_staticallyStructured_1359_);
lean_ctor_set(v___x_1363_, 1, v___x_1362_);
v___x_1364_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1364_, 0, v_staticallyStructured_1359_);
lean_ctor_set(v___x_1364_, 1, v___x_1363_);
v___x_1365_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1365_, 0, v_staticallyStructured_1359_);
lean_ctor_set(v___x_1365_, 1, v___x_1364_);
v___x_1366_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1366_, 0, v_staticallyStructured_1359_);
lean_ctor_set(v___x_1366_, 1, v___x_1365_);
v_sz_1367_ = lean_array_size(v___y_1358_);
v___x_1368_ = ((size_t)0ULL);
v___x_1369_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_StatsReport_scriptsCore_spec__0(v___y_1358_, v_sz_1367_, v___x_1368_, v___x_1366_);
v_snd_1370_ = lean_ctor_get(v___x_1369_, 1);
lean_inc(v_snd_1370_);
v_snd_1371_ = lean_ctor_get(v_snd_1370_, 1);
lean_inc(v_snd_1371_);
v_snd_1372_ = lean_ctor_get(v_snd_1371_, 1);
lean_inc(v_snd_1372_);
v_snd_1373_ = lean_ctor_get(v_snd_1372_, 1);
lean_inc(v_snd_1373_);
v_fst_1374_ = lean_ctor_get(v___x_1369_, 0);
lean_inc(v_fst_1374_);
lean_dec_ref(v___x_1369_);
v_fst_1375_ = lean_ctor_get(v_snd_1370_, 0);
lean_inc(v_fst_1375_);
lean_dec(v_snd_1370_);
v_fst_1376_ = lean_ctor_get(v_snd_1371_, 0);
lean_inc(v_fst_1376_);
lean_dec(v_snd_1371_);
v_fst_1377_ = lean_ctor_get(v_snd_1372_, 0);
lean_inc(v_fst_1377_);
lean_dec(v_snd_1372_);
v_fst_1378_ = lean_ctor_get(v_snd_1373_, 0);
lean_inc(v_fst_1378_);
v_snd_1379_ = lean_ctor_get(v_snd_1373_, 1);
lean_inc(v_snd_1379_);
lean_dec(v_snd_1373_);
v___x_1380_ = lean_nat_dec_eq(v___x_1360_, v_staticallyStructured_1359_);
if (v___x_1380_ == 0)
{
lean_object* v___x_1381_; lean_object* v___x_1382_; uint8_t v___x_1383_; 
v___x_1381_ = lean_unsigned_to_nat(1u);
v___x_1382_ = lean_nat_sub(v___x_1360_, v___x_1381_);
v___x_1383_ = lean_nat_dec_le(v_staticallyStructured_1359_, v___x_1382_);
if (v___x_1383_ == 0)
{
lean_inc(v___x_1382_);
v___y_1344_ = v_staticallyStructured_1359_;
v___y_1345_ = v___x_1368_;
v___y_1346_ = v_fst_1375_;
v___y_1347_ = v_fst_1376_;
v___y_1348_ = v___y_1358_;
v___y_1349_ = v___x_1360_;
v___y_1350_ = v_snd_1379_;
v___y_1351_ = v_fst_1378_;
v___y_1352_ = v_fst_1374_;
v___y_1353_ = v_fst_1377_;
v___y_1354_ = v___x_1382_;
v___y_1355_ = v___x_1382_;
goto v___jp_1343_;
}
else
{
v___y_1344_ = v_staticallyStructured_1359_;
v___y_1345_ = v___x_1368_;
v___y_1346_ = v_fst_1375_;
v___y_1347_ = v_fst_1376_;
v___y_1348_ = v___y_1358_;
v___y_1349_ = v___x_1360_;
v___y_1350_ = v_snd_1379_;
v___y_1351_ = v_fst_1378_;
v___y_1352_ = v_fst_1374_;
v___y_1353_ = v_fst_1377_;
v___y_1354_ = v___x_1382_;
v___y_1355_ = v_staticallyStructured_1359_;
goto v___jp_1343_;
}
}
else
{
v___y_1314_ = v_staticallyStructured_1359_;
v___y_1315_ = v___x_1368_;
v___y_1316_ = v_fst_1375_;
v___y_1317_ = v_fst_1376_;
v___y_1318_ = v___x_1360_;
v___y_1319_ = v_snd_1379_;
v___y_1320_ = v_fst_1378_;
v___y_1321_ = v_fst_1374_;
v___y_1322_ = v_fst_1377_;
v___y_1323_ = v___y_1358_;
goto v___jp_1313_;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsReport_scriptsCore___boxed(lean_object* v_nSlowest_1395_, lean_object* v_nontrivialOnly_1396_, lean_object* v_statsArray_1397_){
_start:
{
uint8_t v_nontrivialOnly_boxed_1398_; lean_object* v_res_1399_; 
v_nontrivialOnly_boxed_1398_ = lean_unbox(v_nontrivialOnly_1396_);
v_res_1399_ = lp_aesop_Aesop_StatsReport_scriptsCore(v_nSlowest_1395_, v_nontrivialOnly_boxed_1398_, v_statsArray_1397_);
return v_res_1399_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_StatsReport_scriptsCore_spec__1(lean_object* v_inst_1400_, lean_object* v_R_1401_, lean_object* v_a_1402_, lean_object* v_b_1403_){
_start:
{
lean_object* v___x_1404_; 
v___x_1404_ = lp_aesop___private_Init_WFExtrinsicFix_0__WellFounded_opaqueFix_u2082___at___00Aesop_StatsReport_scriptsCore_spec__1___redArg(v_a_1402_, v_b_1403_);
return v___x_1404_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4(lean_object* v_n_1405_, lean_object* v_as_1406_, lean_object* v_lo_1407_, lean_object* v_hi_1408_, lean_object* v_w_1409_, lean_object* v_hlo_1410_, lean_object* v_hhi_1411_){
_start:
{
lean_object* v___x_1412_; 
v___x_1412_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___redArg(v_n_1405_, v_as_1406_, v_lo_1407_, v_hi_1408_);
return v___x_1412_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4___boxed(lean_object* v_n_1413_, lean_object* v_as_1414_, lean_object* v_lo_1415_, lean_object* v_hi_1416_, lean_object* v_w_1417_, lean_object* v_hlo_1418_, lean_object* v_hhi_1419_){
_start:
{
lean_object* v_res_1420_; 
v_res_1420_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4(v_n_1413_, v_as_1414_, v_lo_1415_, v_hi_1416_, v_w_1417_, v_hlo_1418_, v_hhi_1419_);
lean_dec(v_hi_1416_);
lean_dec(v_n_1413_);
return v_res_1420_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4_spec__5(lean_object* v_n_1421_, lean_object* v_lo_1422_, lean_object* v_hi_1423_, lean_object* v_hhi_1424_, lean_object* v_pivot_1425_, lean_object* v_as_1426_, lean_object* v_i_1427_, lean_object* v_k_1428_, lean_object* v_ilo_1429_, lean_object* v_ik_1430_, lean_object* v_w_1431_){
_start:
{
lean_object* v___x_1432_; 
v___x_1432_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4_spec__5___redArg(v_hi_1423_, v_pivot_1425_, v_as_1426_, v_i_1427_, v_k_1428_);
return v___x_1432_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4_spec__5___boxed(lean_object* v_n_1433_, lean_object* v_lo_1434_, lean_object* v_hi_1435_, lean_object* v_hhi_1436_, lean_object* v_pivot_1437_, lean_object* v_as_1438_, lean_object* v_i_1439_, lean_object* v_k_1440_, lean_object* v_ilo_1441_, lean_object* v_ik_1442_, lean_object* v_w_1443_){
_start:
{
lean_object* v_res_1444_; 
v_res_1444_ = lp_aesop___private_Init_Data_Array_QSort_Basic_0__Array_qpartition_loop___at___00__private_Init_Data_Array_QSort_Basic_0__Array_qsort_sort___at___00Aesop_StatsReport_scriptsCore_spec__4_spec__5(v_n_1433_, v_lo_1434_, v_hi_1435_, v_hhi_1436_, v_pivot_1437_, v_as_1438_, v_i_1439_, v_k_1440_, v_ilo_1441_, v_ik_1442_, v_w_1443_);
lean_dec_ref(v_pivot_1437_);
lean_dec(v_hi_1435_);
lean_dec(v_lo_1434_);
lean_dec(v_n_1433_);
return v_res_1444_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsReport_scripts(lean_object* v_a_1445_){
_start:
{
lean_object* v___x_1446_; uint8_t v___x_1447_; lean_object* v___x_1448_; 
v___x_1446_ = lean_unsigned_to_nat(30u);
v___x_1447_ = 0;
v___x_1448_ = lp_aesop_Aesop_StatsReport_scriptsCore(v___x_1446_, v___x_1447_, v_a_1445_);
return v___x_1448_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsReport_scriptsNontrivial(lean_object* v_a_1449_){
_start:
{
lean_object* v___x_1450_; uint8_t v___x_1451_; lean_object* v___x_1452_; 
v___x_1450_ = lean_unsigned_to_nat(30u);
v___x_1451_ = 1;
v___x_1452_ = lp_aesop_Aesop_StatsReport_scriptsCore(v___x_1450_, v___x_1451_, v_a_1449_);
return v___x_1452_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Percent(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Stats_Extension(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Stats_Report(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Percent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Stats_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Stats_Report(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Percent(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Stats_Extension(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Stats_Report(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Percent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Stats_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Stats_Report(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Stats_Report(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Stats_Report(builtin);
}
#ifdef __cplusplus
}
#endif
