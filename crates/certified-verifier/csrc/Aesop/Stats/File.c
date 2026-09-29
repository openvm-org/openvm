// Lean compiler output
// Module: Aesop.Stats.File
// Imports: public import Init public meta import Init public import Aesop.Stats.Basic
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
lean_object* l_Std_Format_pretty(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_PrettyPrinter_ppCategory___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lp_aesop_Aesop_instToJsonRuleStats_toJson(lean_object*);
size_t lean_usize_add(size_t, size_t);
extern lean_object* lp_aesop_Aesop_aesop_stats_file;
lean_object* l_Lean_Option_get___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* lean_io_prim_handle_lock(lean_object*, uint8_t);
lean_object* l_Lean_JsonNumber_fromNat(lean_object*);
lean_object* lp_aesop_Aesop_instToJsonScriptGenerated_toJson(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* lp_aesop_Aesop_instToJsonGoalStats_toJson(lean_object*);
lean_object* l_Lean_instToJsonPosition_toJson(lean_object*);
lean_object* l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(lean_object*, uint8_t);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_foldl___at___00Array_appendList_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Json_mkObj(lean_object*);
lean_object* l_Lean_Json_compress(lean_object*);
lean_object* l_IO_FS_Handle_putStrLn(lean_object*, lean_object*);
lean_object* lean_io_prim_handle_unlock(lean_object*);
lean_object* l_IO_FS_withFile___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_KVMap_instValueString;
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__3(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__4(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__2_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1(lean_object*);
static const lean_string_object lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "total"};
static const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__0_value;
static const lean_string_object lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "configParsing"};
static const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__1 = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__1_value;
static const lean_string_object lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "ruleSetConstruction"};
static const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__2 = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__2_value;
static const lean_string_object lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "search"};
static const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__3 = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__3_value;
static const lean_string_object lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "ruleSelection"};
static const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__4 = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__4_value;
static const lean_string_object lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "script"};
static const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__5 = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__5_value;
static const lean_string_object lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "forwardState"};
static const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__6 = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__6_value;
static const lean_string_object lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "scriptGenerated"};
static const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__7 = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__7_value;
static const lean_string_object lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "ruleStats"};
static const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__8 = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__8_value;
static const lean_string_object lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "goalStats"};
static const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__9 = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__9_value;
static const lean_string_object lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "syntax"};
static const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__10 = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__10_value;
static const lean_string_object lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "file"};
static const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__11 = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__11_value;
static const lean_string_object lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "position"};
static const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__12 = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__12_value;
static const lean_string_object lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "declaration"};
static const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__13 = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__13_value;
static const lean_string_object lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "goalSolved"};
static const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__14 = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__14_value;
static const lean_array_object lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__15 = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__15_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson(lean_object*);
static const lean_closure_object lp_aesop_Aesop_instToJsonStatsFileRecord___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop_Aesop_instToJsonStatsFileRecord_toJson, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord___closed__0 = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord___closed__0_value;
LEAN_EXPORT const lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord = (const lean_object*)&lp_aesop_Aesop_instToJsonStatsFileRecord___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "tactic"};
static const lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__0 = (const lean_object*)&lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__0_value;
static const lean_ctor_object lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(99, 76, 33, 121, 85, 143, 17, 224)}};
static const lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__1 = (const lean_object*)&lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__1_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__2(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__3(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__0(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__1(uint8_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__2___closed__0 = (const lean_object*)&lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__0(lean_object* v_x_1_){
_start:
{
if (lean_obj_tag(v_x_1_) == 0)
{
lean_object* v___x_2_; 
v___x_2_ = lean_box(0);
return v___x_2_;
}
else
{
lean_object* v_val_3_; lean_object* v___x_4_; 
v_val_3_ = lean_ctor_get(v_x_1_, 0);
v___x_4_ = lp_aesop_Aesop_instToJsonScriptGenerated_toJson(v_val_3_);
return v___x_4_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__0___boxed(lean_object* v_x_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_aesop_Lean_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__0(v_x_5_);
lean_dec(v_x_5_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__3(lean_object* v_x_7_){
_start:
{
if (lean_obj_tag(v_x_7_) == 0)
{
lean_object* v___x_8_; 
v___x_8_ = lean_box(0);
return v___x_8_;
}
else
{
lean_object* v_val_9_; lean_object* v___x_10_; 
v_val_9_ = lean_ctor_get(v_x_7_, 0);
lean_inc(v_val_9_);
lean_dec_ref_known(v_x_7_, 1);
v___x_10_ = l_Lean_instToJsonPosition_toJson(v_val_9_);
return v___x_10_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__4(lean_object* v_x_11_){
_start:
{
if (lean_obj_tag(v_x_11_) == 0)
{
lean_object* v___x_12_; 
v___x_12_ = lean_box(0);
return v___x_12_;
}
else
{
lean_object* v_val_13_; lean_object* v___x_15_; uint8_t v_isShared_16_; uint8_t v_isSharedCheck_22_; 
v_val_13_ = lean_ctor_get(v_x_11_, 0);
v_isSharedCheck_22_ = !lean_is_exclusive(v_x_11_);
if (v_isSharedCheck_22_ == 0)
{
v___x_15_ = v_x_11_;
v_isShared_16_ = v_isSharedCheck_22_;
goto v_resetjp_14_;
}
else
{
lean_inc(v_val_13_);
lean_dec(v_x_11_);
v___x_15_ = lean_box(0);
v_isShared_16_ = v_isSharedCheck_22_;
goto v_resetjp_14_;
}
v_resetjp_14_:
{
uint8_t v___x_17_; lean_object* v___x_18_; lean_object* v___x_20_; 
v___x_17_ = 1;
v___x_18_ = l_Lean_Name_toStringWithToken___at___00Lean_Name_toString_spec__0(v_val_13_, v___x_17_);
if (v_isShared_16_ == 0)
{
lean_ctor_set_tag(v___x_15_, 3);
lean_ctor_set(v___x_15_, 0, v___x_18_);
v___x_20_ = v___x_15_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_21_; 
v_reuseFailAlloc_21_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v_reuseFailAlloc_21_, 0, v___x_18_);
v___x_20_ = v_reuseFailAlloc_21_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
return v___x_20_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__5(lean_object* v_a_23_, lean_object* v_a_24_){
_start:
{
if (lean_obj_tag(v_a_23_) == 0)
{
lean_object* v___x_25_; 
v___x_25_ = lean_array_to_list(v_a_24_);
return v___x_25_;
}
else
{
lean_object* v_head_26_; lean_object* v_tail_27_; lean_object* v___x_28_; 
v_head_26_ = lean_ctor_get(v_a_23_, 0);
lean_inc(v_head_26_);
v_tail_27_ = lean_ctor_get(v_a_23_, 1);
lean_inc(v_tail_27_);
lean_dec_ref_known(v_a_23_, 2);
v___x_28_ = l_List_foldl___at___00Array_appendList_spec__0___redArg(v_a_24_, v_head_26_);
v_a_23_ = v_tail_27_;
v_a_24_ = v___x_28_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__2_spec__3(size_t v_sz_30_, size_t v_i_31_, lean_object* v_bs_32_){
_start:
{
uint8_t v___x_33_; 
v___x_33_ = lean_usize_dec_lt(v_i_31_, v_sz_30_);
if (v___x_33_ == 0)
{
return v_bs_32_;
}
else
{
lean_object* v_v_34_; lean_object* v___x_35_; lean_object* v_bs_x27_36_; lean_object* v___x_37_; size_t v___x_38_; size_t v___x_39_; lean_object* v___x_40_; 
v_v_34_ = lean_array_uget(v_bs_32_, v_i_31_);
v___x_35_ = lean_unsigned_to_nat(0u);
v_bs_x27_36_ = lean_array_uset(v_bs_32_, v_i_31_, v___x_35_);
v___x_37_ = lp_aesop_Aesop_instToJsonGoalStats_toJson(v_v_34_);
v___x_38_ = ((size_t)1ULL);
v___x_39_ = lean_usize_add(v_i_31_, v___x_38_);
v___x_40_ = lean_array_uset(v_bs_x27_36_, v_i_31_, v___x_37_);
v_i_31_ = v___x_39_;
v_bs_32_ = v___x_40_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__2_spec__3___boxed(lean_object* v_sz_42_, lean_object* v_i_43_, lean_object* v_bs_44_){
_start:
{
size_t v_sz_boxed_45_; size_t v_i_boxed_46_; lean_object* v_res_47_; 
v_sz_boxed_45_ = lean_unbox_usize(v_sz_42_);
lean_dec(v_sz_42_);
v_i_boxed_46_ = lean_unbox_usize(v_i_43_);
lean_dec(v_i_43_);
v_res_47_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__2_spec__3(v_sz_boxed_45_, v_i_boxed_46_, v_bs_44_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__2(lean_object* v_a_48_){
_start:
{
size_t v_sz_49_; size_t v___x_50_; lean_object* v___x_51_; lean_object* v___x_52_; 
v_sz_49_ = lean_array_size(v_a_48_);
v___x_50_ = ((size_t)0ULL);
v___x_51_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__2_spec__3(v_sz_49_, v___x_50_, v_a_48_);
v___x_52_ = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(v___x_52_, 0, v___x_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1_spec__1(size_t v_sz_53_, size_t v_i_54_, lean_object* v_bs_55_){
_start:
{
uint8_t v___x_56_; 
v___x_56_ = lean_usize_dec_lt(v_i_54_, v_sz_53_);
if (v___x_56_ == 0)
{
return v_bs_55_;
}
else
{
lean_object* v_v_57_; lean_object* v___x_58_; lean_object* v_bs_x27_59_; lean_object* v___x_60_; size_t v___x_61_; size_t v___x_62_; lean_object* v___x_63_; 
v_v_57_ = lean_array_uget(v_bs_55_, v_i_54_);
v___x_58_ = lean_unsigned_to_nat(0u);
v_bs_x27_59_ = lean_array_uset(v_bs_55_, v_i_54_, v___x_58_);
v___x_60_ = lp_aesop_Aesop_instToJsonRuleStats_toJson(v_v_57_);
v___x_61_ = ((size_t)1ULL);
v___x_62_ = lean_usize_add(v_i_54_, v___x_61_);
v___x_63_ = lean_array_uset(v_bs_x27_59_, v_i_54_, v___x_60_);
v_i_54_ = v___x_62_;
v_bs_55_ = v___x_63_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1_spec__1___boxed(lean_object* v_sz_65_, lean_object* v_i_66_, lean_object* v_bs_67_){
_start:
{
size_t v_sz_boxed_68_; size_t v_i_boxed_69_; lean_object* v_res_70_; 
v_sz_boxed_68_ = lean_unbox_usize(v_sz_65_);
lean_dec(v_sz_65_);
v_i_boxed_69_ = lean_unbox_usize(v_i_66_);
lean_dec(v_i_66_);
v_res_70_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1_spec__1(v_sz_boxed_68_, v_i_boxed_69_, v_bs_67_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1(lean_object* v_a_71_){
_start:
{
size_t v_sz_72_; size_t v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
v_sz_72_ = lean_array_size(v_a_71_);
v___x_73_ = ((size_t)0ULL);
v___x_74_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1_spec__1(v_sz_72_, v___x_73_, v_a_71_);
v___x_75_ = lean_alloc_ctor(4, 1, 0);
lean_ctor_set(v___x_75_, 0, v___x_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instToJsonStatsFileRecord_toJson(lean_object* v_x_93_){
_start:
{
lean_object* v_toStats_94_; lean_object* v_syntax_95_; lean_object* v_file_96_; lean_object* v_position_97_; lean_object* v_declaration_98_; uint8_t v_goalSolved_99_; lean_object* v_total_100_; lean_object* v_configParsing_101_; lean_object* v_ruleSetConstruction_102_; lean_object* v_search_103_; lean_object* v_ruleSelection_104_; lean_object* v_script_105_; lean_object* v_forwardState_106_; lean_object* v_scriptGenerated_107_; lean_object* v_ruleStats_108_; lean_object* v_goalStats_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; lean_object* v___x_164_; lean_object* v___x_165_; lean_object* v___x_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; lean_object* v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; 
v_toStats_94_ = lean_ctor_get(v_x_93_, 0);
lean_inc_ref(v_toStats_94_);
v_syntax_95_ = lean_ctor_get(v_x_93_, 1);
lean_inc_ref(v_syntax_95_);
v_file_96_ = lean_ctor_get(v_x_93_, 2);
lean_inc_ref(v_file_96_);
v_position_97_ = lean_ctor_get(v_x_93_, 3);
lean_inc(v_position_97_);
v_declaration_98_ = lean_ctor_get(v_x_93_, 4);
lean_inc(v_declaration_98_);
v_goalSolved_99_ = lean_ctor_get_uint8(v_x_93_, sizeof(void*)*5);
lean_dec_ref(v_x_93_);
v_total_100_ = lean_ctor_get(v_toStats_94_, 0);
lean_inc(v_total_100_);
v_configParsing_101_ = lean_ctor_get(v_toStats_94_, 1);
lean_inc(v_configParsing_101_);
v_ruleSetConstruction_102_ = lean_ctor_get(v_toStats_94_, 2);
lean_inc(v_ruleSetConstruction_102_);
v_search_103_ = lean_ctor_get(v_toStats_94_, 3);
lean_inc(v_search_103_);
v_ruleSelection_104_ = lean_ctor_get(v_toStats_94_, 4);
lean_inc(v_ruleSelection_104_);
v_script_105_ = lean_ctor_get(v_toStats_94_, 5);
lean_inc(v_script_105_);
v_forwardState_106_ = lean_ctor_get(v_toStats_94_, 6);
lean_inc(v_forwardState_106_);
v_scriptGenerated_107_ = lean_ctor_get(v_toStats_94_, 7);
lean_inc(v_scriptGenerated_107_);
v_ruleStats_108_ = lean_ctor_get(v_toStats_94_, 8);
lean_inc_ref(v_ruleStats_108_);
v_goalStats_109_ = lean_ctor_get(v_toStats_94_, 9);
lean_inc_ref(v_goalStats_109_);
lean_dec_ref(v_toStats_94_);
v___x_110_ = ((lean_object*)(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__0));
v___x_111_ = l_Lean_JsonNumber_fromNat(v_total_100_);
v___x_112_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_112_, 0, v___x_111_);
v___x_113_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_113_, 0, v___x_110_);
lean_ctor_set(v___x_113_, 1, v___x_112_);
v___x_114_ = lean_box(0);
v___x_115_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_115_, 0, v___x_113_);
lean_ctor_set(v___x_115_, 1, v___x_114_);
v___x_116_ = ((lean_object*)(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__1));
v___x_117_ = l_Lean_JsonNumber_fromNat(v_configParsing_101_);
v___x_118_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_118_, 0, v___x_117_);
v___x_119_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_119_, 0, v___x_116_);
lean_ctor_set(v___x_119_, 1, v___x_118_);
v___x_120_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_120_, 0, v___x_119_);
lean_ctor_set(v___x_120_, 1, v___x_114_);
v___x_121_ = ((lean_object*)(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__2));
v___x_122_ = l_Lean_JsonNumber_fromNat(v_ruleSetConstruction_102_);
v___x_123_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_123_, 0, v___x_122_);
v___x_124_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_124_, 0, v___x_121_);
lean_ctor_set(v___x_124_, 1, v___x_123_);
v___x_125_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_125_, 0, v___x_124_);
lean_ctor_set(v___x_125_, 1, v___x_114_);
v___x_126_ = ((lean_object*)(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__3));
v___x_127_ = l_Lean_JsonNumber_fromNat(v_search_103_);
v___x_128_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_128_, 0, v___x_127_);
v___x_129_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_129_, 0, v___x_126_);
lean_ctor_set(v___x_129_, 1, v___x_128_);
v___x_130_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_130_, 0, v___x_129_);
lean_ctor_set(v___x_130_, 1, v___x_114_);
v___x_131_ = ((lean_object*)(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__4));
v___x_132_ = l_Lean_JsonNumber_fromNat(v_ruleSelection_104_);
v___x_133_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_133_, 0, v___x_132_);
v___x_134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_134_, 0, v___x_131_);
lean_ctor_set(v___x_134_, 1, v___x_133_);
v___x_135_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_135_, 0, v___x_134_);
lean_ctor_set(v___x_135_, 1, v___x_114_);
v___x_136_ = ((lean_object*)(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__5));
v___x_137_ = l_Lean_JsonNumber_fromNat(v_script_105_);
v___x_138_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_138_, 0, v___x_137_);
v___x_139_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_139_, 0, v___x_136_);
lean_ctor_set(v___x_139_, 1, v___x_138_);
v___x_140_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_140_, 0, v___x_139_);
lean_ctor_set(v___x_140_, 1, v___x_114_);
v___x_141_ = ((lean_object*)(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__6));
v___x_142_ = l_Lean_JsonNumber_fromNat(v_forwardState_106_);
v___x_143_ = lean_alloc_ctor(2, 1, 0);
lean_ctor_set(v___x_143_, 0, v___x_142_);
v___x_144_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_144_, 0, v___x_141_);
lean_ctor_set(v___x_144_, 1, v___x_143_);
v___x_145_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_145_, 0, v___x_144_);
lean_ctor_set(v___x_145_, 1, v___x_114_);
v___x_146_ = ((lean_object*)(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__7));
v___x_147_ = lp_aesop_Lean_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__0(v_scriptGenerated_107_);
lean_dec(v_scriptGenerated_107_);
v___x_148_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_148_, 0, v___x_146_);
lean_ctor_set(v___x_148_, 1, v___x_147_);
v___x_149_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_149_, 0, v___x_148_);
lean_ctor_set(v___x_149_, 1, v___x_114_);
v___x_150_ = ((lean_object*)(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__8));
v___x_151_ = lp_aesop_Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__1(v_ruleStats_108_);
v___x_152_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_152_, 0, v___x_150_);
lean_ctor_set(v___x_152_, 1, v___x_151_);
v___x_153_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_153_, 0, v___x_152_);
lean_ctor_set(v___x_153_, 1, v___x_114_);
v___x_154_ = ((lean_object*)(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__9));
v___x_155_ = lp_aesop_Lean_Array_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__2(v_goalStats_109_);
v___x_156_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_156_, 0, v___x_154_);
lean_ctor_set(v___x_156_, 1, v___x_155_);
v___x_157_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_157_, 0, v___x_156_);
lean_ctor_set(v___x_157_, 1, v___x_114_);
v___x_158_ = ((lean_object*)(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__10));
v___x_159_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_159_, 0, v_syntax_95_);
v___x_160_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_160_, 0, v___x_158_);
lean_ctor_set(v___x_160_, 1, v___x_159_);
v___x_161_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_161_, 0, v___x_160_);
lean_ctor_set(v___x_161_, 1, v___x_114_);
v___x_162_ = ((lean_object*)(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__11));
v___x_163_ = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(v___x_163_, 0, v_file_96_);
v___x_164_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_164_, 0, v___x_162_);
lean_ctor_set(v___x_164_, 1, v___x_163_);
v___x_165_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_165_, 0, v___x_164_);
lean_ctor_set(v___x_165_, 1, v___x_114_);
v___x_166_ = ((lean_object*)(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__12));
v___x_167_ = lp_aesop_Lean_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__3(v_position_97_);
v___x_168_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_168_, 0, v___x_166_);
lean_ctor_set(v___x_168_, 1, v___x_167_);
v___x_169_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_169_, 0, v___x_168_);
lean_ctor_set(v___x_169_, 1, v___x_114_);
v___x_170_ = ((lean_object*)(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__13));
v___x_171_ = lp_aesop_Lean_Option_toJson___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__4(v_declaration_98_);
v___x_172_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_172_, 0, v___x_170_);
lean_ctor_set(v___x_172_, 1, v___x_171_);
v___x_173_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_173_, 0, v___x_172_);
lean_ctor_set(v___x_173_, 1, v___x_114_);
v___x_174_ = ((lean_object*)(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__14));
v___x_175_ = lean_alloc_ctor(1, 0, 1);
lean_ctor_set_uint8(v___x_175_, 0, v_goalSolved_99_);
v___x_176_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_176_, 0, v___x_174_);
lean_ctor_set(v___x_176_, 1, v___x_175_);
v___x_177_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_177_, 0, v___x_176_);
lean_ctor_set(v___x_177_, 1, v___x_114_);
v___x_178_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_178_, 0, v___x_177_);
lean_ctor_set(v___x_178_, 1, v___x_114_);
v___x_179_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_179_, 0, v___x_173_);
lean_ctor_set(v___x_179_, 1, v___x_178_);
v___x_180_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_180_, 0, v___x_169_);
lean_ctor_set(v___x_180_, 1, v___x_179_);
v___x_181_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_181_, 0, v___x_165_);
lean_ctor_set(v___x_181_, 1, v___x_180_);
v___x_182_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_182_, 0, v___x_161_);
lean_ctor_set(v___x_182_, 1, v___x_181_);
v___x_183_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_183_, 0, v___x_157_);
lean_ctor_set(v___x_183_, 1, v___x_182_);
v___x_184_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_184_, 0, v___x_153_);
lean_ctor_set(v___x_184_, 1, v___x_183_);
v___x_185_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_185_, 0, v___x_149_);
lean_ctor_set(v___x_185_, 1, v___x_184_);
v___x_186_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_186_, 0, v___x_145_);
lean_ctor_set(v___x_186_, 1, v___x_185_);
v___x_187_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_187_, 0, v___x_140_);
lean_ctor_set(v___x_187_, 1, v___x_186_);
v___x_188_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_188_, 0, v___x_135_);
lean_ctor_set(v___x_188_, 1, v___x_187_);
v___x_189_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_189_, 0, v___x_130_);
lean_ctor_set(v___x_189_, 1, v___x_188_);
v___x_190_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_190_, 0, v___x_125_);
lean_ctor_set(v___x_190_, 1, v___x_189_);
v___x_191_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_191_, 0, v___x_120_);
lean_ctor_set(v___x_191_, 1, v___x_190_);
v___x_192_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_192_, 0, v___x_115_);
lean_ctor_set(v___x_192_, 1, v___x_191_);
v___x_193_ = ((lean_object*)(lp_aesop_Aesop_instToJsonStatsFileRecord_toJson___closed__15));
v___x_194_ = lp_aesop___private_Init_Data_List_Impl_0__List_flatMapTR_go___at___00Aesop_instToJsonStatsFileRecord_toJson_spec__5(v___x_192_, v___x_193_);
v___x_195_ = l_Lean_Json_mkObj(v___x_194_);
lean_dec(v___x_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__0(lean_object* v_stats_198_, lean_object* v_file_199_, lean_object* v___y_200_, lean_object* v_declaration_201_, uint8_t v_goalSolved_202_, lean_object* v_toPure_203_, lean_object* v_____do__lift_204_){
_start:
{
lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v_syntax_207_; lean_object* v___x_208_; lean_object* v___x_209_; 
v___x_205_ = lean_cstr_to_nat("100000000000");
v___x_206_ = lean_unsigned_to_nat(0u);
v_syntax_207_ = l_Std_Format_pretty(v_____do__lift_204_, v___x_205_, v___x_206_, v___x_206_);
v___x_208_ = lean_alloc_ctor(0, 5, 1);
lean_ctor_set(v___x_208_, 0, v_stats_198_);
lean_ctor_set(v___x_208_, 1, v_syntax_207_);
lean_ctor_set(v___x_208_, 2, v_file_199_);
lean_ctor_set(v___x_208_, 3, v___y_200_);
lean_ctor_set(v___x_208_, 4, v_declaration_201_);
lean_ctor_set_uint8(v___x_208_, sizeof(void*)*5, v_goalSolved_202_);
v___x_209_ = lean_apply_2(v_toPure_203_, lean_box(0), v___x_208_);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__0___boxed(lean_object* v_stats_210_, lean_object* v_file_211_, lean_object* v___y_212_, lean_object* v_declaration_213_, lean_object* v_goalSolved_214_, lean_object* v_toPure_215_, lean_object* v_____do__lift_216_){
_start:
{
uint8_t v_goalSolved_boxed_217_; lean_object* v_res_218_; 
v_goalSolved_boxed_217_ = lean_unbox(v_goalSolved_214_);
v_res_218_ = lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__0(v_stats_210_, v_file_211_, v___y_212_, v_declaration_213_, v_goalSolved_boxed_217_, v_toPure_215_, v_____do__lift_216_);
return v_res_218_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1(lean_object* v_stats_222_, lean_object* v_file_223_, lean_object* v___y_224_, uint8_t v_goalSolved_225_, lean_object* v_toPure_226_, lean_object* v_aesopStx_227_, lean_object* v_inst_228_, lean_object* v_toBind_229_, lean_object* v_declaration_230_){
_start:
{
lean_object* v___x_231_; lean_object* v___f_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; 
v___x_231_ = lean_box(v_goalSolved_225_);
v___f_232_ = lean_alloc_closure((void*)(lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__0___boxed), 7, 6);
lean_closure_set(v___f_232_, 0, v_stats_222_);
lean_closure_set(v___f_232_, 1, v_file_223_);
lean_closure_set(v___f_232_, 2, v___y_224_);
lean_closure_set(v___f_232_, 3, v_declaration_230_);
lean_closure_set(v___f_232_, 4, v___x_231_);
lean_closure_set(v___f_232_, 5, v_toPure_226_);
v___x_233_ = ((lean_object*)(lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___closed__1));
v___x_234_ = lean_alloc_closure((void*)(l_Lean_PrettyPrinter_ppCategory___boxed), 5, 2);
lean_closure_set(v___x_234_, 0, v___x_233_);
lean_closure_set(v___x_234_, 1, v_aesopStx_227_);
v___x_235_ = lean_apply_2(v_inst_228_, lean_box(0), v___x_234_);
v___x_236_ = lean_apply_4(v_toBind_229_, lean_box(0), lean_box(0), v___x_235_, v___f_232_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___boxed(lean_object* v_stats_237_, lean_object* v_file_238_, lean_object* v___y_239_, lean_object* v_goalSolved_240_, lean_object* v_toPure_241_, lean_object* v_aesopStx_242_, lean_object* v_inst_243_, lean_object* v_toBind_244_, lean_object* v_declaration_245_){
_start:
{
uint8_t v_goalSolved_boxed_246_; lean_object* v_res_247_; 
v_goalSolved_boxed_246_ = lean_unbox(v_goalSolved_240_);
v_res_247_ = lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1(v_stats_237_, v_file_238_, v___y_239_, v_goalSolved_boxed_246_, v_toPure_241_, v_aesopStx_242_, v_inst_243_, v_toBind_244_, v_declaration_245_);
return v_res_247_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__2(lean_object* v_stats_248_, lean_object* v_file_249_, uint8_t v_goalSolved_250_, lean_object* v_toPure_251_, lean_object* v_aesopStx_252_, lean_object* v_inst_253_, lean_object* v_toBind_254_, lean_object* v_inst_255_, lean_object* v_fileMap_256_){
_start:
{
lean_object* v___y_258_; uint8_t v___x_262_; lean_object* v___x_263_; 
v___x_262_ = 0;
v___x_263_ = l_Lean_Syntax_getPos_x3f(v_aesopStx_252_, v___x_262_);
if (lean_obj_tag(v___x_263_) == 0)
{
lean_object* v___x_264_; 
lean_dec_ref(v_fileMap_256_);
v___x_264_ = lean_box(0);
v___y_258_ = v___x_264_;
goto v___jp_257_;
}
else
{
lean_object* v_val_265_; lean_object* v___x_267_; uint8_t v_isShared_268_; uint8_t v_isSharedCheck_273_; 
v_val_265_ = lean_ctor_get(v___x_263_, 0);
v_isSharedCheck_273_ = !lean_is_exclusive(v___x_263_);
if (v_isSharedCheck_273_ == 0)
{
v___x_267_ = v___x_263_;
v_isShared_268_ = v_isSharedCheck_273_;
goto v_resetjp_266_;
}
else
{
lean_inc(v_val_265_);
lean_dec(v___x_263_);
v___x_267_ = lean_box(0);
v_isShared_268_ = v_isSharedCheck_273_;
goto v_resetjp_266_;
}
v_resetjp_266_:
{
lean_object* v___x_269_; lean_object* v___x_271_; 
v___x_269_ = l_Lean_FileMap_toPosition(v_fileMap_256_, v_val_265_);
lean_dec(v_val_265_);
if (v_isShared_268_ == 0)
{
lean_ctor_set(v___x_267_, 0, v___x_269_);
v___x_271_ = v___x_267_;
goto v_reusejp_270_;
}
else
{
lean_object* v_reuseFailAlloc_272_; 
v_reuseFailAlloc_272_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_272_, 0, v___x_269_);
v___x_271_ = v_reuseFailAlloc_272_;
goto v_reusejp_270_;
}
v_reusejp_270_:
{
v___y_258_ = v___x_271_;
goto v___jp_257_;
}
}
}
v___jp_257_:
{
lean_object* v___x_259_; lean_object* v___f_260_; lean_object* v___x_261_; 
v___x_259_ = lean_box(v_goalSolved_250_);
lean_inc(v_toBind_254_);
v___f_260_ = lean_alloc_closure((void*)(lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__1___boxed), 9, 8);
lean_closure_set(v___f_260_, 0, v_stats_248_);
lean_closure_set(v___f_260_, 1, v_file_249_);
lean_closure_set(v___f_260_, 2, v___y_258_);
lean_closure_set(v___f_260_, 3, v___x_259_);
lean_closure_set(v___f_260_, 4, v_toPure_251_);
lean_closure_set(v___f_260_, 5, v_aesopStx_252_);
lean_closure_set(v___f_260_, 6, v_inst_253_);
lean_closure_set(v___f_260_, 7, v_toBind_254_);
v___x_261_ = lean_apply_4(v_toBind_254_, lean_box(0), lean_box(0), v_inst_255_, v___f_260_);
return v___x_261_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__2___boxed(lean_object* v_stats_274_, lean_object* v_file_275_, lean_object* v_goalSolved_276_, lean_object* v_toPure_277_, lean_object* v_aesopStx_278_, lean_object* v_inst_279_, lean_object* v_toBind_280_, lean_object* v_inst_281_, lean_object* v_fileMap_282_){
_start:
{
uint8_t v_goalSolved_boxed_283_; lean_object* v_res_284_; 
v_goalSolved_boxed_283_ = lean_unbox(v_goalSolved_276_);
v_res_284_ = lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__2(v_stats_274_, v_file_275_, v_goalSolved_boxed_283_, v_toPure_277_, v_aesopStx_278_, v_inst_279_, v_toBind_280_, v_inst_281_, v_fileMap_282_);
return v_res_284_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__3(lean_object* v_stats_285_, uint8_t v_goalSolved_286_, lean_object* v_toPure_287_, lean_object* v_aesopStx_288_, lean_object* v_inst_289_, lean_object* v_toBind_290_, lean_object* v_inst_291_, lean_object* v_toMonadFileMap_292_, lean_object* v_file_293_){
_start:
{
lean_object* v___x_294_; lean_object* v___f_295_; lean_object* v___x_296_; 
v___x_294_ = lean_box(v_goalSolved_286_);
lean_inc(v_toBind_290_);
v___f_295_ = lean_alloc_closure((void*)(lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__2___boxed), 9, 8);
lean_closure_set(v___f_295_, 0, v_stats_285_);
lean_closure_set(v___f_295_, 1, v_file_293_);
lean_closure_set(v___f_295_, 2, v___x_294_);
lean_closure_set(v___f_295_, 3, v_toPure_287_);
lean_closure_set(v___f_295_, 4, v_aesopStx_288_);
lean_closure_set(v___f_295_, 5, v_inst_289_);
lean_closure_set(v___f_295_, 6, v_toBind_290_);
lean_closure_set(v___f_295_, 7, v_inst_291_);
v___x_296_ = lean_apply_4(v_toBind_290_, lean_box(0), lean_box(0), v_toMonadFileMap_292_, v___f_295_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__3___boxed(lean_object* v_stats_297_, lean_object* v_goalSolved_298_, lean_object* v_toPure_299_, lean_object* v_aesopStx_300_, lean_object* v_inst_301_, lean_object* v_toBind_302_, lean_object* v_inst_303_, lean_object* v_toMonadFileMap_304_, lean_object* v_file_305_){
_start:
{
uint8_t v_goalSolved_boxed_306_; lean_object* v_res_307_; 
v_goalSolved_boxed_306_ = lean_unbox(v_goalSolved_298_);
v_res_307_ = lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__3(v_stats_297_, v_goalSolved_boxed_306_, v_toPure_299_, v_aesopStx_300_, v_inst_301_, v_toBind_302_, v_inst_303_, v_toMonadFileMap_304_, v_file_305_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg(lean_object* v_inst_308_, lean_object* v_inst_309_, lean_object* v_inst_310_, lean_object* v_inst_311_, lean_object* v_aesopStx_312_, uint8_t v_goalSolved_313_, lean_object* v_stats_314_){
_start:
{
lean_object* v_toApplicative_315_; lean_object* v_toBind_316_; lean_object* v_toMonadFileMap_317_; lean_object* v_getFileName_318_; lean_object* v_toPure_319_; lean_object* v___x_320_; lean_object* v___f_321_; lean_object* v___x_322_; 
v_toApplicative_315_ = lean_ctor_get(v_inst_308_, 0);
lean_inc_ref(v_toApplicative_315_);
v_toBind_316_ = lean_ctor_get(v_inst_308_, 1);
lean_inc_n(v_toBind_316_, 2);
lean_dec_ref(v_inst_308_);
v_toMonadFileMap_317_ = lean_ctor_get(v_inst_309_, 0);
lean_inc(v_toMonadFileMap_317_);
v_getFileName_318_ = lean_ctor_get(v_inst_309_, 2);
lean_inc(v_getFileName_318_);
lean_dec_ref(v_inst_309_);
v_toPure_319_ = lean_ctor_get(v_toApplicative_315_, 1);
lean_inc(v_toPure_319_);
lean_dec_ref(v_toApplicative_315_);
v___x_320_ = lean_box(v_goalSolved_313_);
v___f_321_ = lean_alloc_closure((void*)(lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___lam__3___boxed), 9, 8);
lean_closure_set(v___f_321_, 0, v_stats_314_);
lean_closure_set(v___f_321_, 1, v___x_320_);
lean_closure_set(v___f_321_, 2, v_toPure_319_);
lean_closure_set(v___f_321_, 3, v_aesopStx_312_);
lean_closure_set(v___f_321_, 4, v_inst_311_);
lean_closure_set(v___f_321_, 5, v_toBind_316_);
lean_closure_set(v___f_321_, 6, v_inst_310_);
lean_closure_set(v___f_321_, 7, v_toMonadFileMap_317_);
v___x_322_ = lean_apply_4(v_toBind_316_, lean_box(0), lean_box(0), v_getFileName_318_, v___f_321_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___redArg___boxed(lean_object* v_inst_323_, lean_object* v_inst_324_, lean_object* v_inst_325_, lean_object* v_inst_326_, lean_object* v_aesopStx_327_, lean_object* v_goalSolved_328_, lean_object* v_stats_329_){
_start:
{
uint8_t v_goalSolved_boxed_330_; lean_object* v_res_331_; 
v_goalSolved_boxed_330_ = lean_unbox(v_goalSolved_328_);
v_res_331_ = lp_aesop_Aesop_StatsFileRecord_ofStats___redArg(v_inst_323_, v_inst_324_, v_inst_325_, v_inst_326_, v_aesopStx_327_, v_goalSolved_boxed_330_, v_stats_329_);
return v_res_331_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats(lean_object* v_m_332_, lean_object* v_inst_333_, lean_object* v_inst_334_, lean_object* v_inst_335_, lean_object* v_inst_336_, lean_object* v_aesopStx_337_, uint8_t v_goalSolved_338_, lean_object* v_stats_339_){
_start:
{
lean_object* v___x_340_; 
v___x_340_ = lp_aesop_Aesop_StatsFileRecord_ofStats___redArg(v_inst_333_, v_inst_334_, v_inst_335_, v_inst_336_, v_aesopStx_337_, v_goalSolved_338_, v_stats_339_);
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsFileRecord_ofStats___boxed(lean_object* v_m_341_, lean_object* v_inst_342_, lean_object* v_inst_343_, lean_object* v_inst_344_, lean_object* v_inst_345_, lean_object* v_aesopStx_346_, lean_object* v_goalSolved_347_, lean_object* v_stats_348_){
_start:
{
uint8_t v_goalSolved_boxed_349_; lean_object* v_res_350_; 
v_goalSolved_boxed_349_ = lean_unbox(v_goalSolved_347_);
v_res_350_ = lp_aesop_Aesop_StatsFileRecord_ofStats(v_m_341_, v_inst_342_, v_inst_343_, v_inst_344_, v_inst_345_, v_aesopStx_346_, v_goalSolved_boxed_349_, v_stats_348_);
return v_res_350_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__0(uint8_t v___x_351_, lean_object* v_record_352_, lean_object* v_hdl_353_){
_start:
{
lean_object* v___x_355_; 
v___x_355_ = lean_io_prim_handle_lock(v_hdl_353_, v___x_351_);
if (lean_obj_tag(v___x_355_) == 0)
{
lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v_r_358_; 
lean_dec_ref_known(v___x_355_, 1);
v___x_356_ = lp_aesop_Aesop_instToJsonStatsFileRecord_toJson(v_record_352_);
v___x_357_ = l_Lean_Json_compress(v___x_356_);
v_r_358_ = l_IO_FS_Handle_putStrLn(v_hdl_353_, v___x_357_);
if (lean_obj_tag(v_r_358_) == 0)
{
lean_object* v_a_359_; lean_object* v___x_360_; 
v_a_359_ = lean_ctor_get(v_r_358_, 0);
lean_inc(v_a_359_);
lean_dec_ref_known(v_r_358_, 1);
v___x_360_ = lean_io_prim_handle_unlock(v_hdl_353_);
if (lean_obj_tag(v___x_360_) == 0)
{
lean_object* v___x_362_; uint8_t v_isShared_363_; uint8_t v_isSharedCheck_367_; 
v_isSharedCheck_367_ = !lean_is_exclusive(v___x_360_);
if (v_isSharedCheck_367_ == 0)
{
lean_object* v_unused_368_; 
v_unused_368_ = lean_ctor_get(v___x_360_, 0);
lean_dec(v_unused_368_);
v___x_362_ = v___x_360_;
v_isShared_363_ = v_isSharedCheck_367_;
goto v_resetjp_361_;
}
else
{
lean_dec(v___x_360_);
v___x_362_ = lean_box(0);
v_isShared_363_ = v_isSharedCheck_367_;
goto v_resetjp_361_;
}
v_resetjp_361_:
{
lean_object* v___x_365_; 
if (v_isShared_363_ == 0)
{
lean_ctor_set(v___x_362_, 0, v_a_359_);
v___x_365_ = v___x_362_;
goto v_reusejp_364_;
}
else
{
lean_object* v_reuseFailAlloc_366_; 
v_reuseFailAlloc_366_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_366_, 0, v_a_359_);
v___x_365_ = v_reuseFailAlloc_366_;
goto v_reusejp_364_;
}
v_reusejp_364_:
{
return v___x_365_;
}
}
}
else
{
lean_dec(v_a_359_);
return v___x_360_;
}
}
else
{
lean_object* v_a_369_; lean_object* v___x_370_; 
v_a_369_ = lean_ctor_get(v_r_358_, 0);
lean_inc(v_a_369_);
lean_dec_ref_known(v_r_358_, 1);
v___x_370_ = lean_io_prim_handle_unlock(v_hdl_353_);
if (lean_obj_tag(v___x_370_) == 0)
{
lean_object* v___x_372_; uint8_t v_isShared_373_; uint8_t v_isSharedCheck_377_; 
v_isSharedCheck_377_ = !lean_is_exclusive(v___x_370_);
if (v_isSharedCheck_377_ == 0)
{
lean_object* v_unused_378_; 
v_unused_378_ = lean_ctor_get(v___x_370_, 0);
lean_dec(v_unused_378_);
v___x_372_ = v___x_370_;
v_isShared_373_ = v_isSharedCheck_377_;
goto v_resetjp_371_;
}
else
{
lean_dec(v___x_370_);
v___x_372_ = lean_box(0);
v_isShared_373_ = v_isSharedCheck_377_;
goto v_resetjp_371_;
}
v_resetjp_371_:
{
lean_object* v___x_375_; 
if (v_isShared_373_ == 0)
{
lean_ctor_set_tag(v___x_372_, 1);
lean_ctor_set(v___x_372_, 0, v_a_369_);
v___x_375_ = v___x_372_;
goto v_reusejp_374_;
}
else
{
lean_object* v_reuseFailAlloc_376_; 
v_reuseFailAlloc_376_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_376_, 0, v_a_369_);
v___x_375_ = v_reuseFailAlloc_376_;
goto v_reusejp_374_;
}
v_reusejp_374_:
{
return v___x_375_;
}
}
}
else
{
lean_dec(v_a_369_);
return v___x_370_;
}
}
}
else
{
lean_dec_ref(v_record_352_);
return v___x_355_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__0___boxed(lean_object* v___x_379_, lean_object* v_record_380_, lean_object* v_hdl_381_, lean_object* v___y_382_){
_start:
{
uint8_t v___x_373__boxed_383_; lean_object* v_res_384_; 
v___x_373__boxed_383_ = lean_unbox(v___x_379_);
v_res_384_ = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__0(v___x_373__boxed_383_, v_record_380_, v_hdl_381_);
lean_dec(v_hdl_381_);
return v_res_384_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__1(uint8_t v___x_385_, lean_object* v_file_386_, lean_object* v_inst_387_, lean_object* v_record_388_){
_start:
{
lean_object* v___x_389_; lean_object* v___f_390_; uint8_t v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_394_; 
v___x_389_ = lean_box(v___x_385_);
v___f_390_ = lean_alloc_closure((void*)(lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_390_, 0, v___x_389_);
lean_closure_set(v___f_390_, 1, v_record_388_);
v___x_391_ = 4;
v___x_392_ = lean_box(v___x_391_);
v___x_393_ = lean_alloc_closure((void*)(l_IO_FS_withFile___boxed), 5, 4);
lean_closure_set(v___x_393_, 0, lean_box(0));
lean_closure_set(v___x_393_, 1, v_file_386_);
lean_closure_set(v___x_393_, 2, v___x_392_);
lean_closure_set(v___x_393_, 3, v___f_390_);
v___x_394_ = lean_apply_2(v_inst_387_, lean_box(0), v___x_393_);
return v___x_394_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__1___boxed(lean_object* v___x_395_, lean_object* v_file_396_, lean_object* v_inst_397_, lean_object* v_record_398_){
_start:
{
uint8_t v___x_431__boxed_399_; lean_object* v_res_400_; 
v___x_431__boxed_399_ = lean_unbox(v___x_395_);
v_res_400_ = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__1(v___x_431__boxed_399_, v_file_396_, v_inst_397_, v_record_398_);
return v_res_400_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__2(lean_object* v___x_402_, lean_object* v_inst_403_, lean_object* v_inst_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_inst_407_, lean_object* v_aesopStx_408_, uint8_t v_allGoalsSolved_409_, lean_object* v_stats_410_, lean_object* v_toBind_411_, lean_object* v_toApplicative_412_, lean_object* v_____do__lift_413_){
_start:
{
lean_object* v___x_414_; lean_object* v_file_415_; lean_object* v___x_416_; uint8_t v___x_417_; 
v___x_414_ = lp_aesop_Aesop_aesop_stats_file;
v_file_415_ = l_Lean_Option_get___redArg(v___x_402_, v_____do__lift_413_, v___x_414_);
v___x_416_ = ((lean_object*)(lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__2___closed__0));
v___x_417_ = lean_string_dec_eq(v_file_415_, v___x_416_);
if (v___x_417_ == 0)
{
uint8_t v___x_418_; lean_object* v___x_419_; lean_object* v___f_420_; lean_object* v___x_421_; lean_object* v___x_422_; 
lean_dec_ref(v_toApplicative_412_);
v___x_418_ = 1;
v___x_419_ = lean_box(v___x_418_);
v___f_420_ = lean_alloc_closure((void*)(lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_420_, 0, v___x_419_);
lean_closure_set(v___f_420_, 1, v_file_415_);
lean_closure_set(v___f_420_, 2, v_inst_403_);
v___x_421_ = lp_aesop_Aesop_StatsFileRecord_ofStats___redArg(v_inst_404_, v_inst_405_, v_inst_406_, v_inst_407_, v_aesopStx_408_, v_allGoalsSolved_409_, v_stats_410_);
v___x_422_ = lean_apply_4(v_toBind_411_, lean_box(0), lean_box(0), v___x_421_, v___f_420_);
return v___x_422_;
}
else
{
lean_object* v_toPure_423_; lean_object* v___x_424_; lean_object* v___x_425_; 
lean_dec(v_file_415_);
lean_dec(v_toBind_411_);
lean_dec_ref(v_stats_410_);
lean_dec(v_aesopStx_408_);
lean_dec(v_inst_407_);
lean_dec(v_inst_406_);
lean_dec_ref(v_inst_405_);
lean_dec_ref(v_inst_404_);
lean_dec(v_inst_403_);
v_toPure_423_ = lean_ctor_get(v_toApplicative_412_, 1);
lean_inc(v_toPure_423_);
lean_dec_ref(v_toApplicative_412_);
v___x_424_ = lean_box(0);
v___x_425_ = lean_apply_2(v_toPure_423_, lean_box(0), v___x_424_);
return v___x_425_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__2___boxed(lean_object* v___x_426_, lean_object* v_inst_427_, lean_object* v_inst_428_, lean_object* v_inst_429_, lean_object* v_inst_430_, lean_object* v_inst_431_, lean_object* v_aesopStx_432_, lean_object* v_allGoalsSolved_433_, lean_object* v_stats_434_, lean_object* v_toBind_435_, lean_object* v_toApplicative_436_, lean_object* v_____do__lift_437_){
_start:
{
uint8_t v_allGoalsSolved_boxed_438_; lean_object* v_res_439_; 
v_allGoalsSolved_boxed_438_ = lean_unbox(v_allGoalsSolved_433_);
v_res_439_ = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__2(v___x_426_, v_inst_427_, v_inst_428_, v_inst_429_, v_inst_430_, v_inst_431_, v_aesopStx_432_, v_allGoalsSolved_boxed_438_, v_stats_434_, v_toBind_435_, v_toApplicative_436_, v_____do__lift_437_);
lean_dec_ref(v_____do__lift_437_);
return v_res_439_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg(lean_object* v_inst_440_, lean_object* v_inst_441_, lean_object* v_inst_442_, lean_object* v_inst_443_, lean_object* v_inst_444_, lean_object* v_inst_445_, lean_object* v_aesopStx_446_, lean_object* v_stats_447_, uint8_t v_allGoalsSolved_448_){
_start:
{
lean_object* v___x_449_; lean_object* v_toApplicative_450_; lean_object* v_toBind_451_; lean_object* v___x_452_; lean_object* v___f_453_; lean_object* v___x_454_; 
v___x_449_ = l_Lean_KVMap_instValueString;
v_toApplicative_450_ = lean_ctor_get(v_inst_440_, 0);
lean_inc_ref(v_toApplicative_450_);
v_toBind_451_ = lean_ctor_get(v_inst_440_, 1);
lean_inc_n(v_toBind_451_, 2);
v___x_452_ = lean_box(v_allGoalsSolved_448_);
v___f_453_ = lean_alloc_closure((void*)(lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___lam__2___boxed), 12, 11);
lean_closure_set(v___f_453_, 0, v___x_449_);
lean_closure_set(v___f_453_, 1, v_inst_444_);
lean_closure_set(v___f_453_, 2, v_inst_440_);
lean_closure_set(v___f_453_, 3, v_inst_441_);
lean_closure_set(v___f_453_, 4, v_inst_443_);
lean_closure_set(v___f_453_, 5, v_inst_445_);
lean_closure_set(v___f_453_, 6, v_aesopStx_446_);
lean_closure_set(v___f_453_, 7, v___x_452_);
lean_closure_set(v___f_453_, 8, v_stats_447_);
lean_closure_set(v___f_453_, 9, v_toBind_451_);
lean_closure_set(v___f_453_, 10, v_toApplicative_450_);
v___x_454_ = lean_apply_4(v_toBind_451_, lean_box(0), lean_box(0), v_inst_442_, v___f_453_);
return v___x_454_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg___boxed(lean_object* v_inst_455_, lean_object* v_inst_456_, lean_object* v_inst_457_, lean_object* v_inst_458_, lean_object* v_inst_459_, lean_object* v_inst_460_, lean_object* v_aesopStx_461_, lean_object* v_stats_462_, lean_object* v_allGoalsSolved_463_){
_start:
{
uint8_t v_allGoalsSolved_boxed_464_; lean_object* v_res_465_; 
v_allGoalsSolved_boxed_464_ = lean_unbox(v_allGoalsSolved_463_);
v_res_465_ = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg(v_inst_455_, v_inst_456_, v_inst_457_, v_inst_458_, v_inst_459_, v_inst_460_, v_aesopStx_461_, v_stats_462_, v_allGoalsSolved_boxed_464_);
return v_res_465_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled(lean_object* v_m_466_, lean_object* v_inst_467_, lean_object* v_inst_468_, lean_object* v_inst_469_, lean_object* v_inst_470_, lean_object* v_inst_471_, lean_object* v_inst_472_, lean_object* v_aesopStx_473_, lean_object* v_stats_474_, uint8_t v_allGoalsSolved_475_){
_start:
{
lean_object* v___x_476_; 
v___x_476_ = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___redArg(v_inst_467_, v_inst_468_, v_inst_469_, v_inst_470_, v_inst_471_, v_inst_472_, v_aesopStx_473_, v_stats_474_, v_allGoalsSolved_475_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_appendStatsToStatsFileIfEnabled___boxed(lean_object* v_m_477_, lean_object* v_inst_478_, lean_object* v_inst_479_, lean_object* v_inst_480_, lean_object* v_inst_481_, lean_object* v_inst_482_, lean_object* v_inst_483_, lean_object* v_aesopStx_484_, lean_object* v_stats_485_, lean_object* v_allGoalsSolved_486_){
_start:
{
uint8_t v_allGoalsSolved_boxed_487_; lean_object* v_res_488_; 
v_allGoalsSolved_boxed_487_ = lean_unbox(v_allGoalsSolved_486_);
v_res_488_ = lp_aesop_Aesop_appendStatsToStatsFileIfEnabled(v_m_477_, v_inst_478_, v_inst_479_, v_inst_480_, v_inst_481_, v_inst_482_, v_inst_483_, v_aesopStx_484_, v_stats_485_, v_allGoalsSolved_boxed_487_);
return v_res_488_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Stats_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Stats_File(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Stats_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Stats_File(uint8_t builtin) {
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
lean_object* initialize_aesop_Aesop_Stats_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Stats_File(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Stats_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Stats_File(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Stats_File(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Stats_File(builtin);
}
#ifdef __cplusplus
}
#endif
