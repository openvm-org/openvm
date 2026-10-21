// Lean compiler output
// Module: Mathlib.Data.Multiset.Filter
// Imports: public import Init public meta import Init public import Mathlib.Data.Multiset.MapFold public import Mathlib.Data.Set.Function public import Mathlib.Order.Hom.Basic
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
lean_object* lp_mathlib_Multiset_map(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_List_filterTR_loop___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_filter___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Multiset_filterMap_spec__0___redArg(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Multiset_filterMap___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Multiset_filterMap___redArg___closed__0 = (const lean_object*)&lp_mathlib_Multiset_filterMap___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filterMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filterMap(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Multiset_filterMap_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_mapEmbedding___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_mapEmbedding___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_mapEmbedding(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Multiset_filter___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_b_2_){
_start:
{
lean_object* v___x_3_; uint8_t v___x_4_; 
v___x_3_ = lean_apply_1(v_inst_1_, v_b_2_);
v___x_4_ = lean_unbox(v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter___redArg___lam__0___boxed(lean_object* v_inst_5_, lean_object* v_b_6_){
_start:
{
uint8_t v_res_7_; lean_object* v_r_8_; 
v_res_7_ = lp_mathlib_Multiset_filter___redArg___lam__0(v_inst_5_, v_b_6_);
v_r_8_ = lean_box(v_res_7_);
return v_r_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter___redArg(lean_object* v_inst_9_, lean_object* v_s_10_){
_start:
{
lean_object* v___f_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
v___f_11_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_filter___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_11_, 0, v_inst_9_);
v___x_12_ = lean_box(0);
v___x_13_ = l_List_filterTR_loop___redArg(v___f_11_, v_s_10_, v___x_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filter(lean_object* v_00_u03b1_14_, lean_object* v_p_15_, lean_object* v_inst_16_, lean_object* v_s_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lp_mathlib_Multiset_filter___redArg(v_inst_16_, v_s_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Multiset_filterMap_spec__0___redArg(lean_object* v_f_19_, lean_object* v_a_20_, lean_object* v_a_21_){
_start:
{
if (lean_obj_tag(v_a_20_) == 0)
{
lean_object* v___x_22_; 
lean_dec_ref(v_f_19_);
v___x_22_ = lean_array_to_list(v_a_21_);
return v___x_22_;
}
else
{
lean_object* v_head_23_; lean_object* v_tail_24_; lean_object* v___x_25_; 
v_head_23_ = lean_ctor_get(v_a_20_, 0);
lean_inc(v_head_23_);
v_tail_24_ = lean_ctor_get(v_a_20_, 1);
lean_inc(v_tail_24_);
lean_dec_ref_known(v_a_20_, 2);
lean_inc_ref(v_f_19_);
v___x_25_ = lean_apply_1(v_f_19_, v_head_23_);
if (lean_obj_tag(v___x_25_) == 0)
{
v_a_20_ = v_tail_24_;
goto _start;
}
else
{
lean_object* v_val_27_; lean_object* v___x_28_; 
v_val_27_ = lean_ctor_get(v___x_25_, 0);
lean_inc(v_val_27_);
lean_dec_ref_known(v___x_25_, 1);
v___x_28_ = lean_array_push(v_a_21_, v_val_27_);
v_a_20_ = v_tail_24_;
v_a_21_ = v___x_28_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filterMap___redArg(lean_object* v_f_32_, lean_object* v_s_33_){
_start:
{
lean_object* v___x_34_; lean_object* v___x_35_; 
v___x_34_ = ((lean_object*)(lp_mathlib_Multiset_filterMap___redArg___closed__0));
v___x_35_ = lp_mathlib_List_filterMapTR_go___at___00Multiset_filterMap_spec__0___redArg(v_f_32_, v_s_33_, v___x_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_filterMap(lean_object* v_00_u03b1_36_, lean_object* v_00_u03b2_37_, lean_object* v_f_38_, lean_object* v_s_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_mathlib_Multiset_filterMap___redArg(v_f_38_, v_s_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Multiset_filterMap_spec__0(lean_object* v_00_u03b1_41_, lean_object* v_00_u03b2_42_, lean_object* v_f_43_, lean_object* v_a_44_, lean_object* v_a_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_mathlib_List_filterMapTR_go___at___00Multiset_filterMap_spec__0___redArg(v_f_43_, v_a_44_, v_a_45_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_mapEmbedding___redArg___lam__0(lean_object* v_f_47_, lean_object* v___y_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lean_apply_1(v_f_47_, v___y_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_mapEmbedding___redArg(lean_object* v_f_50_){
_start:
{
lean_object* v___f_51_; lean_object* v___x_52_; 
v___f_51_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_mapEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_51_, 0, v_f_50_);
v___x_52_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_map), 4, 3);
lean_closure_set(v___x_52_, 0, lean_box(0));
lean_closure_set(v___x_52_, 1, lean_box(0));
lean_closure_set(v___x_52_, 2, v___f_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_mapEmbedding(lean_object* v_00_u03b1_53_, lean_object* v_00_u03b2_54_, lean_object* v_f_55_){
_start:
{
lean_object* v___x_56_; 
v___x_56_ = lp_mathlib_Multiset_mapEmbedding___redArg(v_f_55_);
return v___x_56_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_MapFold(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Function(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Filter(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_MapFold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Function(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Multiset_Filter(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Multiset_MapFold(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Function(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_Filter(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_MapFold(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Function(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Filter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Multiset_Filter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Multiset_Filter(builtin);
}
#ifdef __cplusplus
}
#endif
