// Lean compiler output
// Module: Aesop.Stats.Extension
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
lean_object* lean_array_mk(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_registerSimplePersistentEnvExtension___redArg(lean_object*);
lean_object* l_Lean_SimplePersistentEnvExtension_getEntries___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instInhabitedPersistentEnvExtensionState___redArg(lean_object*);
lean_object* l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Syntax_getPos_x3f(lean_object*, uint8_t);
lean_object* l_Lean_FileMap_toPosition(lean_object*, lean_object*);
lean_object* l_Lean_PersistentEnvExtension_addEntry___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_aesop_Aesop_aesop_collectStats;
lean_object* l_Lean_Option_get___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_KVMap_instValueBool;
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_aesop_Aesop_StatsExtension_importedEntries___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop_Aesop_StatsExtension_importedEntries___closed__0 = (const lean_object*)&lp_aesop_Aesop_StatsExtension_importedEntries___closed__0_value;
static lean_once_cell_t lp_aesop_Aesop_StatsExtension_importedEntries___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_StatsExtension_importedEntries___closed__1;
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtension_importedEntries(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtension_importedEntries___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___lam__0_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___lam__0_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2____boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___lam__1_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___lam__1_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___lam__2_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_(lean_object*);
static const lean_closure_object lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__0_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___lam__0_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2____boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__0_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__0_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value;
static const lean_closure_object lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__1_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___lam__1_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2____boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__1_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__1_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value;
static const lean_closure_object lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__2_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___lam__2_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__2_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__2_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__3_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Aesop"};
static const lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__3_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__3_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value;
static const lean_string_object lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__4_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "statsExtension"};
static const lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__4_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__4_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__5_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__3_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(97, 226, 101, 135, 78, 117, 164, 248)}};
static const lean_ctor_object lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__5_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__5_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value_aux_0),((lean_object*)&lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__4_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value),LEAN_SCALAR_PTR_LITERAL(1, 230, 124, 168, 122, 62, 100, 180)}};
static const lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__5_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__5_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value;
static const lean_ctor_object lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__6_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)&lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__5_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__0_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__1_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value),((lean_object*)&lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__2_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__6_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_ = (const lean_object*)&lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__6_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2__value;
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_();
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_statsExtension;
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsIfEnabled___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsIfEnabled___redArg___lam__1(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsIfEnabled___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_recordStatsIfEnabled___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsIfEnabled___redArg___lam__2___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_aesop_Aesop_recordStatsIfEnabled___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_aesop_Aesop_recordStatsIfEnabled___redArg___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsIfEnabled___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsIfEnabled(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_mkStatsArray_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_mkStatsArray_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_forIn_x27_loop___at___00Aesop_mkStatsArray_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_forIn_x27_loop___at___00Aesop_mkStatsArray_spec__0___redArg___boxed(lean_object*, lean_object*);
static const lean_array_object lp_aesop_Aesop_mkStatsArray___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_aesop_Aesop_mkStatsArray___closed__0 = (const lean_object*)&lp_aesop_Aesop_mkStatsArray___closed__0_value;
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkStatsArray(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkStatsArray___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_forIn_x27_loop___at___00Aesop_mkStatsArray_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_List_forIn_x27_loop___at___00Aesop_mkStatsArray_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getStatsArray___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getStatsArray___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_getStatsArray(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___redArg___lam__0(lean_object* v_stx_1_, lean_object* v_fileName_2_, lean_object* v_stats_3_, lean_object* v_toPure_4_, lean_object* v_fileMap_5_){
_start:
{
lean_object* v___y_7_; uint8_t v___x_10_; lean_object* v___x_11_; 
v___x_10_ = 0;
v___x_11_ = l_Lean_Syntax_getPos_x3f(v_stx_1_, v___x_10_);
if (lean_obj_tag(v___x_11_) == 0)
{
lean_object* v___x_12_; 
lean_dec_ref(v_fileMap_5_);
v___x_12_ = lean_box(0);
v___y_7_ = v___x_12_;
goto v___jp_6_;
}
else
{
lean_object* v_val_13_; lean_object* v___x_15_; uint8_t v_isShared_16_; uint8_t v_isSharedCheck_21_; 
v_val_13_ = lean_ctor_get(v___x_11_, 0);
v_isSharedCheck_21_ = !lean_is_exclusive(v___x_11_);
if (v_isSharedCheck_21_ == 0)
{
v___x_15_ = v___x_11_;
v_isShared_16_ = v_isSharedCheck_21_;
goto v_resetjp_14_;
}
else
{
lean_inc(v_val_13_);
lean_dec(v___x_11_);
v___x_15_ = lean_box(0);
v_isShared_16_ = v_isSharedCheck_21_;
goto v_resetjp_14_;
}
v_resetjp_14_:
{
lean_object* v___x_17_; lean_object* v___x_19_; 
v___x_17_ = l_Lean_FileMap_toPosition(v_fileMap_5_, v_val_13_);
lean_dec(v_val_13_);
if (v_isShared_16_ == 0)
{
lean_ctor_set(v___x_15_, 0, v___x_17_);
v___x_19_ = v___x_15_;
goto v_reusejp_18_;
}
else
{
lean_object* v_reuseFailAlloc_20_; 
v_reuseFailAlloc_20_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_20_, 0, v___x_17_);
v___x_19_ = v_reuseFailAlloc_20_;
goto v_reusejp_18_;
}
v_reusejp_18_:
{
v___y_7_ = v___x_19_;
goto v___jp_6_;
}
}
}
v___jp_6_:
{
lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_8_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_8_, 0, v_stx_1_);
lean_ctor_set(v___x_8_, 1, v_fileName_2_);
lean_ctor_set(v___x_8_, 2, v___y_7_);
lean_ctor_set(v___x_8_, 3, v_stats_3_);
v___x_9_ = lean_apply_2(v_toPure_4_, lean_box(0), v___x_8_);
return v___x_9_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___redArg___lam__1(lean_object* v_stx_22_, lean_object* v_stats_23_, lean_object* v_toPure_24_, lean_object* v_toBind_25_, lean_object* v_toMonadFileMap_26_, lean_object* v_fileName_27_){
_start:
{
lean_object* v___f_28_; lean_object* v___x_29_; 
v___f_28_ = lean_alloc_closure((void*)(lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___redArg___lam__0), 5, 4);
lean_closure_set(v___f_28_, 0, v_stx_22_);
lean_closure_set(v___f_28_, 1, v_fileName_27_);
lean_closure_set(v___f_28_, 2, v_stats_23_);
lean_closure_set(v___f_28_, 3, v_toPure_24_);
v___x_29_ = lean_apply_4(v_toBind_25_, lean_box(0), lean_box(0), v_toMonadFileMap_26_, v___f_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___redArg(lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_stx_32_, lean_object* v_stats_33_){
_start:
{
lean_object* v_toApplicative_34_; lean_object* v_toBind_35_; lean_object* v_toMonadFileMap_36_; lean_object* v_getFileName_37_; lean_object* v_toPure_38_; lean_object* v___f_39_; lean_object* v___x_40_; 
v_toApplicative_34_ = lean_ctor_get(v_inst_30_, 0);
lean_inc_ref(v_toApplicative_34_);
v_toBind_35_ = lean_ctor_get(v_inst_30_, 1);
lean_inc_n(v_toBind_35_, 2);
lean_dec_ref(v_inst_30_);
v_toMonadFileMap_36_ = lean_ctor_get(v_inst_31_, 0);
lean_inc(v_toMonadFileMap_36_);
v_getFileName_37_ = lean_ctor_get(v_inst_31_, 2);
lean_inc(v_getFileName_37_);
lean_dec_ref(v_inst_31_);
v_toPure_38_ = lean_ctor_get(v_toApplicative_34_, 1);
lean_inc(v_toPure_38_);
lean_dec_ref(v_toApplicative_34_);
v___f_39_ = lean_alloc_closure((void*)(lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___redArg___lam__1), 6, 5);
lean_closure_set(v___f_39_, 0, v_stx_32_);
lean_closure_set(v___f_39_, 1, v_stats_33_);
lean_closure_set(v___f_39_, 2, v_toPure_38_);
lean_closure_set(v___f_39_, 3, v_toBind_35_);
lean_closure_set(v___f_39_, 4, v_toMonadFileMap_36_);
v___x_40_ = lean_apply_4(v_toBind_35_, lean_box(0), lean_box(0), v_getFileName_37_, v___f_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile(lean_object* v_m_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_stx_44_, lean_object* v_stats_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___redArg(v_inst_42_, v_inst_43_, v_stx_44_, v_stats_45_);
return v___x_46_;
}
}
static lean_object* _init_lp_aesop_Aesop_StatsExtension_importedEntries___closed__1(void){
_start:
{
lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_50_ = ((lean_object*)(lp_aesop_Aesop_StatsExtension_importedEntries___closed__0));
v___x_51_ = l_Lean_instInhabitedPersistentEnvExtensionState___redArg(v___x_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtension_importedEntries(lean_object* v_env_52_, lean_object* v_ext_53_){
_start:
{
lean_object* v_toEnvExtension_54_; lean_object* v_asyncMode_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; lean_object* v_importedEntries_59_; 
v_toEnvExtension_54_ = lean_ctor_get(v_ext_53_, 0);
v_asyncMode_55_ = lean_ctor_get(v_toEnvExtension_54_, 2);
v___x_56_ = lean_obj_once(&lp_aesop_Aesop_StatsExtension_importedEntries___closed__1, &lp_aesop_Aesop_StatsExtension_importedEntries___closed__1_once, _init_lp_aesop_Aesop_StatsExtension_importedEntries___closed__1);
v___x_57_ = lean_box(0);
v___x_58_ = l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(v___x_56_, v_toEnvExtension_54_, v_env_52_, v_asyncMode_55_, v___x_57_);
v_importedEntries_59_ = lean_ctor_get(v___x_58_, 0);
lean_inc_ref(v_importedEntries_59_);
lean_dec(v___x_58_);
return v_importedEntries_59_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_StatsExtension_importedEntries___boxed(lean_object* v_env_60_, lean_object* v_ext_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_aesop_Aesop_StatsExtension_importedEntries(v_env_60_, v_ext_61_);
lean_dec_ref(v_ext_61_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___lam__0_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_(lean_object* v_x_63_, lean_object* v_x_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lean_box(0);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___lam__0_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2____boxed(lean_object* v_x_66_, lean_object* v_x_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___lam__0_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_(v_x_66_, v_x_67_);
lean_dec_ref(v_x_67_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___lam__1_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_(lean_object* v_x_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lean_box(0);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___lam__1_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2____boxed(lean_object* v_x_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___lam__1_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_(v_x_71_);
lean_dec_ref(v_x_71_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___lam__2_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_(lean_object* v_es_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lean_array_mk(v_es_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_(){
_start:
{
lean_object* v___x_91_; lean_object* v___x_92_; 
v___x_91_ = ((lean_object*)(lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn___closed__6_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_));
v___x_92_ = l_Lean_registerSimplePersistentEnvExtension___redArg(v___x_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2____boxed(lean_object* v_a_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_();
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsIfEnabled___redArg___lam__0(lean_object* v_s_95_, lean_object* v_env_96_){
_start:
{
lean_object* v___x_97_; lean_object* v_toEnvExtension_98_; lean_object* v_asyncMode_99_; lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_97_ = lp_aesop_Aesop_statsExtension;
v_toEnvExtension_98_ = lean_ctor_get(v___x_97_, 0);
v_asyncMode_99_ = lean_ctor_get(v_toEnvExtension_98_, 2);
v___x_100_ = lean_box(0);
v___x_101_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_97_, v_env_96_, v_s_95_, v_asyncMode_99_, v___x_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsIfEnabled___redArg___lam__1(lean_object* v_toPure_102_, lean_object* v_inst_103_, lean_object* v___f_104_, uint8_t v_____do__lift_105_){
_start:
{
if (v_____do__lift_105_ == 0)
{
lean_object* v___x_106_; lean_object* v___x_107_; 
lean_dec_ref(v___f_104_);
lean_dec_ref(v_inst_103_);
v___x_106_ = lean_box(0);
v___x_107_ = lean_apply_2(v_toPure_102_, lean_box(0), v___x_106_);
return v___x_107_;
}
else
{
lean_object* v_modifyEnv_108_; lean_object* v___x_109_; 
lean_dec(v_toPure_102_);
v_modifyEnv_108_ = lean_ctor_get(v_inst_103_, 1);
lean_inc(v_modifyEnv_108_);
lean_dec_ref(v_inst_103_);
v___x_109_ = lean_apply_1(v_modifyEnv_108_, v___f_104_);
return v___x_109_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsIfEnabled___redArg___lam__1___boxed(lean_object* v_toPure_110_, lean_object* v_inst_111_, lean_object* v___f_112_, lean_object* v_____do__lift_113_){
_start:
{
uint8_t v_____do__lift_100__boxed_114_; lean_object* v_res_115_; 
v_____do__lift_100__boxed_114_ = lean_unbox(v_____do__lift_113_);
v_res_115_ = lp_aesop_Aesop_recordStatsIfEnabled___redArg___lam__1(v_toPure_110_, v_inst_111_, v___f_112_, v_____do__lift_100__boxed_114_);
return v_res_115_;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_recordStatsIfEnabled___redArg___lam__2(lean_object* v___x_116_, lean_object* v_opts_117_){
_start:
{
lean_object* v___x_118_; lean_object* v___x_119_; uint8_t v___x_120_; 
v___x_118_ = lp_aesop_Aesop_aesop_collectStats;
v___x_119_ = l_Lean_Option_get___redArg(v___x_116_, v_opts_117_, v___x_118_);
v___x_120_ = lean_unbox(v___x_119_);
lean_dec(v___x_119_);
return v___x_120_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsIfEnabled___redArg___lam__2___boxed(lean_object* v___x_121_, lean_object* v_opts_122_){
_start:
{
uint8_t v_res_123_; lean_object* v_r_124_; 
v_res_123_ = lp_aesop_Aesop_recordStatsIfEnabled___redArg___lam__2(v___x_121_, v_opts_122_);
lean_dec_ref(v_opts_122_);
v_r_124_ = lean_box(v_res_123_);
return v_r_124_;
}
}
static lean_object* _init_lp_aesop_Aesop_recordStatsIfEnabled___redArg___closed__0(void){
_start:
{
lean_object* v___x_125_; lean_object* v___f_126_; 
v___x_125_ = l_Lean_KVMap_instValueBool;
v___f_126_ = lean_alloc_closure((void*)(lp_aesop_Aesop_recordStatsIfEnabled___redArg___lam__2___boxed), 2, 1);
lean_closure_set(v___f_126_, 0, v___x_125_);
return v___f_126_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsIfEnabled___redArg(lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_inst_129_, lean_object* v_s_130_){
_start:
{
lean_object* v_toApplicative_131_; lean_object* v_toFunctor_132_; lean_object* v_toBind_133_; lean_object* v_toPure_134_; lean_object* v_map_135_; lean_object* v___f_136_; lean_object* v___f_137_; lean_object* v___f_138_; lean_object* v___x_139_; lean_object* v___x_140_; 
v_toApplicative_131_ = lean_ctor_get(v_inst_127_, 0);
lean_inc_ref(v_toApplicative_131_);
v_toFunctor_132_ = lean_ctor_get(v_toApplicative_131_, 0);
lean_inc_ref(v_toFunctor_132_);
v_toBind_133_ = lean_ctor_get(v_inst_127_, 1);
lean_inc(v_toBind_133_);
lean_dec_ref(v_inst_127_);
v_toPure_134_ = lean_ctor_get(v_toApplicative_131_, 1);
lean_inc(v_toPure_134_);
lean_dec_ref(v_toApplicative_131_);
v_map_135_ = lean_ctor_get(v_toFunctor_132_, 0);
lean_inc(v_map_135_);
lean_dec_ref(v_toFunctor_132_);
v___f_136_ = lean_alloc_closure((void*)(lp_aesop_Aesop_recordStatsIfEnabled___redArg___lam__0), 2, 1);
lean_closure_set(v___f_136_, 0, v_s_130_);
v___f_137_ = lean_alloc_closure((void*)(lp_aesop_Aesop_recordStatsIfEnabled___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_137_, 0, v_toPure_134_);
lean_closure_set(v___f_137_, 1, v_inst_128_);
lean_closure_set(v___f_137_, 2, v___f_136_);
v___f_138_ = lean_obj_once(&lp_aesop_Aesop_recordStatsIfEnabled___redArg___closed__0, &lp_aesop_Aesop_recordStatsIfEnabled___redArg___closed__0_once, _init_lp_aesop_Aesop_recordStatsIfEnabled___redArg___closed__0);
v___x_139_ = lean_apply_4(v_map_135_, lean_box(0), lean_box(0), v___f_138_, v_inst_129_);
v___x_140_ = lean_apply_4(v_toBind_133_, lean_box(0), lean_box(0), v___x_139_, v___f_137_);
return v___x_140_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsIfEnabled(lean_object* v_m_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_s_145_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = lp_aesop_Aesop_recordStatsIfEnabled___redArg(v_inst_142_, v_inst_143_, v_inst_144_, v_s_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___redArg___lam__0(lean_object* v_entry_147_, lean_object* v_env_148_){
_start:
{
lean_object* v___x_149_; lean_object* v_toEnvExtension_150_; lean_object* v_asyncMode_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
v___x_149_ = lp_aesop_Aesop_statsExtension;
v_toEnvExtension_150_ = lean_ctor_get(v___x_149_, 0);
v_asyncMode_151_ = lean_ctor_get(v_toEnvExtension_150_, 2);
v___x_152_ = lean_box(0);
v___x_153_ = l_Lean_PersistentEnvExtension_addEntry___redArg(v___x_149_, v_env_148_, v_entry_147_, v_asyncMode_151_, v___x_152_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___redArg___lam__1(lean_object* v_inst_154_, lean_object* v_entry_155_){
_start:
{
lean_object* v_modifyEnv_156_; lean_object* v___f_157_; lean_object* v___x_158_; 
v_modifyEnv_156_ = lean_ctor_get(v_inst_154_, 1);
lean_inc(v_modifyEnv_156_);
lean_dec_ref(v_inst_154_);
v___f_157_ = lean_alloc_closure((void*)(lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___redArg___lam__0), 2, 1);
lean_closure_set(v___f_157_, 0, v_entry_155_);
v___x_158_ = lean_apply_1(v_modifyEnv_156_, v___f_157_);
return v___x_158_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___redArg___lam__2(lean_object* v_toPure_159_, lean_object* v_inst_160_, lean_object* v_inst_161_, lean_object* v_aesopStx_162_, lean_object* v_stats_163_, lean_object* v_toBind_164_, lean_object* v___f_165_, uint8_t v_____do__lift_166_){
_start:
{
if (v_____do__lift_166_ == 0)
{
lean_object* v___x_167_; lean_object* v___x_168_; 
lean_dec(v___f_165_);
lean_dec(v_toBind_164_);
lean_dec_ref(v_stats_163_);
lean_dec(v_aesopStx_162_);
lean_dec_ref(v_inst_161_);
lean_dec_ref(v_inst_160_);
v___x_167_ = lean_box(0);
v___x_168_ = lean_apply_2(v_toPure_159_, lean_box(0), v___x_167_);
return v___x_168_;
}
else
{
lean_object* v___x_169_; lean_object* v___x_170_; 
lean_dec(v_toPure_159_);
v___x_169_ = lp_aesop_Aesop_StatsExtensionEntry_forCurrentFile___redArg(v_inst_160_, v_inst_161_, v_aesopStx_162_, v_stats_163_);
v___x_170_ = lean_apply_4(v_toBind_164_, lean_box(0), lean_box(0), v___x_169_, v___f_165_);
return v___x_170_;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___redArg___lam__2___boxed(lean_object* v_toPure_171_, lean_object* v_inst_172_, lean_object* v_inst_173_, lean_object* v_aesopStx_174_, lean_object* v_stats_175_, lean_object* v_toBind_176_, lean_object* v___f_177_, lean_object* v_____do__lift_178_){
_start:
{
uint8_t v_____do__lift_123__boxed_179_; lean_object* v_res_180_; 
v_____do__lift_123__boxed_179_ = lean_unbox(v_____do__lift_178_);
v_res_180_ = lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___redArg___lam__2(v_toPure_171_, v_inst_172_, v_inst_173_, v_aesopStx_174_, v_stats_175_, v_toBind_176_, v___f_177_, v_____do__lift_123__boxed_179_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___redArg(lean_object* v_inst_181_, lean_object* v_inst_182_, lean_object* v_inst_183_, lean_object* v_inst_184_, lean_object* v_aesopStx_185_, lean_object* v_stats_186_){
_start:
{
lean_object* v_toApplicative_187_; lean_object* v_toBind_188_; lean_object* v_toFunctor_189_; lean_object* v_toPure_190_; lean_object* v_map_191_; lean_object* v___f_192_; lean_object* v___f_193_; lean_object* v___f_194_; lean_object* v___x_195_; lean_object* v___x_196_; 
v_toApplicative_187_ = lean_ctor_get(v_inst_181_, 0);
v_toBind_188_ = lean_ctor_get(v_inst_181_, 1);
lean_inc_n(v_toBind_188_, 2);
v_toFunctor_189_ = lean_ctor_get(v_toApplicative_187_, 0);
v_toPure_190_ = lean_ctor_get(v_toApplicative_187_, 1);
lean_inc(v_toPure_190_);
v_map_191_ = lean_ctor_get(v_toFunctor_189_, 0);
lean_inc(v_map_191_);
v___f_192_ = lean_alloc_closure((void*)(lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___redArg___lam__1), 2, 1);
lean_closure_set(v___f_192_, 0, v_inst_182_);
v___f_193_ = lean_alloc_closure((void*)(lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___redArg___lam__2___boxed), 8, 7);
lean_closure_set(v___f_193_, 0, v_toPure_190_);
lean_closure_set(v___f_193_, 1, v_inst_181_);
lean_closure_set(v___f_193_, 2, v_inst_184_);
lean_closure_set(v___f_193_, 3, v_aesopStx_185_);
lean_closure_set(v___f_193_, 4, v_stats_186_);
lean_closure_set(v___f_193_, 5, v_toBind_188_);
lean_closure_set(v___f_193_, 6, v___f_192_);
v___f_194_ = lean_obj_once(&lp_aesop_Aesop_recordStatsIfEnabled___redArg___closed__0, &lp_aesop_Aesop_recordStatsIfEnabled___redArg___closed__0_once, _init_lp_aesop_Aesop_recordStatsIfEnabled___redArg___closed__0);
v___x_195_ = lean_apply_4(v_map_191_, lean_box(0), lean_box(0), v___f_194_, v_inst_183_);
v___x_196_ = lean_apply_4(v_toBind_188_, lean_box(0), lean_box(0), v___x_195_, v___f_193_);
return v___x_196_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled(lean_object* v_m_197_, lean_object* v_inst_198_, lean_object* v_inst_199_, lean_object* v_inst_200_, lean_object* v_inst_201_, lean_object* v_aesopStx_202_, lean_object* v_stats_203_){
_start:
{
lean_object* v___x_204_; 
v___x_204_ = lp_aesop_Aesop_recordStatsForCurrentFileIfEnabled___redArg(v_inst_198_, v_inst_199_, v_inst_200_, v_inst_201_, v_aesopStx_202_, v_stats_203_);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_mkStatsArray_spec__1(lean_object* v_as_205_, size_t v_sz_206_, size_t v_i_207_, lean_object* v_b_208_){
_start:
{
uint8_t v___x_209_; 
v___x_209_ = lean_usize_dec_lt(v_i_207_, v_sz_206_);
if (v___x_209_ == 0)
{
return v_b_208_;
}
else
{
lean_object* v_a_210_; lean_object* v___x_211_; size_t v___x_212_; size_t v___x_213_; 
v_a_210_ = lean_array_uget_borrowed(v_as_205_, v_i_207_);
v___x_211_ = l_Array_append___redArg(v_b_208_, v_a_210_);
v___x_212_ = ((size_t)1ULL);
v___x_213_ = lean_usize_add(v_i_207_, v___x_212_);
v_i_207_ = v___x_213_;
v_b_208_ = v___x_211_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_mkStatsArray_spec__1___boxed(lean_object* v_as_215_, lean_object* v_sz_216_, lean_object* v_i_217_, lean_object* v_b_218_){
_start:
{
size_t v_sz_boxed_219_; size_t v_i_boxed_220_; lean_object* v_res_221_; 
v_sz_boxed_219_ = lean_unbox_usize(v_sz_216_);
lean_dec(v_sz_216_);
v_i_boxed_220_ = lean_unbox_usize(v_i_217_);
lean_dec(v_i_217_);
v_res_221_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_mkStatsArray_spec__1(v_as_215_, v_sz_boxed_219_, v_i_boxed_220_, v_b_218_);
lean_dec_ref(v_as_215_);
return v_res_221_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_forIn_x27_loop___at___00Aesop_mkStatsArray_spec__0___redArg(lean_object* v_as_x27_222_, lean_object* v_b_223_){
_start:
{
if (lean_obj_tag(v_as_x27_222_) == 0)
{
return v_b_223_;
}
else
{
lean_object* v_head_224_; lean_object* v_tail_225_; lean_object* v___x_226_; 
v_head_224_ = lean_ctor_get(v_as_x27_222_, 0);
v_tail_225_ = lean_ctor_get(v_as_x27_222_, 1);
lean_inc(v_head_224_);
v___x_226_ = lean_array_push(v_b_223_, v_head_224_);
v_as_x27_222_ = v_tail_225_;
v_b_223_ = v___x_226_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_List_forIn_x27_loop___at___00Aesop_mkStatsArray_spec__0___redArg___boxed(lean_object* v_as_x27_228_, lean_object* v_b_229_){
_start:
{
lean_object* v_res_230_; 
v_res_230_ = lp_aesop_List_forIn_x27_loop___at___00Aesop_mkStatsArray_spec__0___redArg(v_as_x27_228_, v_b_229_);
lean_dec(v_as_x27_228_);
return v_res_230_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkStatsArray(lean_object* v_localEntries_233_, lean_object* v_importedEntries_234_){
_start:
{
lean_object* v_result_235_; lean_object* v___x_236_; size_t v_sz_237_; size_t v___x_238_; lean_object* v___x_239_; 
v_result_235_ = ((lean_object*)(lp_aesop_Aesop_mkStatsArray___closed__0));
v___x_236_ = lp_aesop_List_forIn_x27_loop___at___00Aesop_mkStatsArray_spec__0___redArg(v_localEntries_233_, v_result_235_);
v_sz_237_ = lean_array_size(v_importedEntries_234_);
v___x_238_ = ((size_t)0ULL);
v___x_239_ = lp_aesop___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Aesop_mkStatsArray_spec__1(v_importedEntries_234_, v_sz_237_, v___x_238_, v___x_236_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_mkStatsArray___boxed(lean_object* v_localEntries_240_, lean_object* v_importedEntries_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_aesop_Aesop_mkStatsArray(v_localEntries_240_, v_importedEntries_241_);
lean_dec_ref(v_importedEntries_241_);
lean_dec(v_localEntries_240_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_forIn_x27_loop___at___00Aesop_mkStatsArray_spec__0(lean_object* v_as_243_, lean_object* v_as_x27_244_, lean_object* v_b_245_, lean_object* v_a_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = lp_aesop_List_forIn_x27_loop___at___00Aesop_mkStatsArray_spec__0___redArg(v_as_x27_244_, v_b_245_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_aesop_List_forIn_x27_loop___at___00Aesop_mkStatsArray_spec__0___boxed(lean_object* v_as_248_, lean_object* v_as_x27_249_, lean_object* v_b_250_, lean_object* v_a_251_){
_start:
{
lean_object* v_res_252_; 
v_res_252_ = lp_aesop_List_forIn_x27_loop___at___00Aesop_mkStatsArray_spec__0(v_as_248_, v_as_x27_249_, v_b_250_, v_a_251_);
lean_dec(v_as_x27_249_);
lean_dec(v_as_248_);
return v_res_252_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getStatsArray___redArg___lam__0(lean_object* v___x_253_, lean_object* v_toPure_254_, lean_object* v_env_255_){
_start:
{
lean_object* v___x_256_; lean_object* v_toEnvExtension_257_; lean_object* v_asyncMode_258_; lean_object* v_current_259_; lean_object* v_imported_260_; lean_object* v___x_261_; lean_object* v___x_262_; 
v___x_256_ = lp_aesop_Aesop_statsExtension;
v_toEnvExtension_257_ = lean_ctor_get(v___x_256_, 0);
v_asyncMode_258_ = lean_ctor_get(v_toEnvExtension_257_, 2);
lean_inc_ref(v_env_255_);
v_current_259_ = l_Lean_SimplePersistentEnvExtension_getEntries___redArg(v___x_253_, v___x_256_, v_env_255_, v_asyncMode_258_);
v_imported_260_ = lp_aesop_Aesop_StatsExtension_importedEntries(v_env_255_, v___x_256_);
v___x_261_ = lp_aesop_Aesop_mkStatsArray(v_current_259_, v_imported_260_);
lean_dec_ref(v_imported_260_);
lean_dec(v_current_259_);
v___x_262_ = lean_apply_2(v_toPure_254_, lean_box(0), v___x_261_);
return v___x_262_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getStatsArray___redArg(lean_object* v_inst_263_, lean_object* v_inst_264_){
_start:
{
lean_object* v_toApplicative_265_; lean_object* v_toBind_266_; lean_object* v_getEnv_267_; lean_object* v_toPure_268_; lean_object* v___x_269_; lean_object* v___f_270_; lean_object* v___x_271_; 
v_toApplicative_265_ = lean_ctor_get(v_inst_263_, 0);
lean_inc_ref(v_toApplicative_265_);
v_toBind_266_ = lean_ctor_get(v_inst_263_, 1);
lean_inc(v_toBind_266_);
lean_dec_ref(v_inst_263_);
v_getEnv_267_ = lean_ctor_get(v_inst_264_, 0);
lean_inc(v_getEnv_267_);
lean_dec_ref(v_inst_264_);
v_toPure_268_ = lean_ctor_get(v_toApplicative_265_, 1);
lean_inc(v_toPure_268_);
lean_dec_ref(v_toApplicative_265_);
v___x_269_ = lean_box(0);
v___f_270_ = lean_alloc_closure((void*)(lp_aesop_Aesop_getStatsArray___redArg___lam__0), 3, 2);
lean_closure_set(v___f_270_, 0, v___x_269_);
lean_closure_set(v___f_270_, 1, v_toPure_268_);
v___x_271_ = lean_apply_4(v_toBind_266_, lean_box(0), lean_box(0), v_getEnv_267_, v___f_270_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_getStatsArray(lean_object* v_m_272_, lean_object* v_inst_273_, lean_object* v_inst_274_){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lp_aesop_Aesop_getStatsArray___redArg(v_inst_273_, v_inst_274_);
return v___x_275_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Stats_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_aesop_Aesop_Stats_Extension(uint8_t builtin) {
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
res = lp_aesop___private_Aesop_Stats_Extension_0__Aesop_initFn_00___x40_Aesop_Stats_Extension_2703840432____hygCtx___hyg_2_();
if (lean_io_result_is_error(res)) return res;
lp_aesop_Aesop_statsExtension = lean_io_result_get_value(res);
lean_mark_persistent(lp_aesop_Aesop_statsExtension);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_aesop_Aesop_Stats_Extension(uint8_t builtin) {
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
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Stats_Extension(uint8_t builtin) {
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
res = runtime_initialize_aesop_Aesop_Stats_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_aesop_Aesop_Stats_Extension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_aesop_Aesop_Stats_Extension(builtin);
}
#ifdef __cplusplus
}
#endif
