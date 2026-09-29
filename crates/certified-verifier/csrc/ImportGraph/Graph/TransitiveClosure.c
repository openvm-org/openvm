// Lean compiler output
// Module: ImportGraph.Graph.TransitiveClosure
// Imports: public import Init public meta import Init public import Lean.Data.NameMap.Basic
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
uint8_t l_Std_DTreeMap_Internal_Impl_contains___at___00Lean_NameMap_contains_spec__0___redArg(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_Lean_NameSet_ofList(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
size_t lean_usize_sub(size_t, size_t);
lean_object* l_Lean_NameSet_insert(lean_object*, lean_object*);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_NameSet_empty;
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__2___closed__0 = (const lean_object*)&lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__2___closed__0_value;
LEAN_EXPORT lean_object* lp_importGraph___private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__2(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldl___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_NameMap_transitiveClosure_spec__0_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_NameMap_transitiveClosure_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_NameMap_transitiveClosure(lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldl___at___00Lean_NameMap_transitiveClosure_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldl___at___00Lean_NameMap_transitiveClosure_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__0_spec__0(lean_object* v_init_1_, lean_object* v_x_2_){
_start:
{
if (lean_obj_tag(v_x_2_) == 0)
{
lean_object* v_k_3_; lean_object* v_l_4_; lean_object* v_r_5_; lean_object* v___x_6_; lean_object* v___x_7_; 
v_k_3_ = lean_ctor_get(v_x_2_, 1);
lean_inc(v_k_3_);
v_l_4_ = lean_ctor_get(v_x_2_, 3);
lean_inc(v_l_4_);
v_r_5_ = lean_ctor_get(v_x_2_, 4);
lean_inc(v_r_5_);
lean_dec_ref_known(v_x_2_, 5);
v___x_6_ = lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__0_spec__0(v_init_1_, v_l_4_);
v___x_7_ = l_Lean_NameSet_insert(v___x_6_, v_k_3_);
v_init_1_ = v___x_7_;
v_x_2_ = v_r_5_;
goto _start;
}
else
{
return v_init_1_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__1(lean_object* v___y_9_, lean_object* v_as_10_, size_t v_i_11_, size_t v_stop_12_, lean_object* v_b_13_){
_start:
{
uint8_t v___x_14_; 
v___x_14_ = lean_usize_dec_eq(v_i_11_, v_stop_12_);
if (v___x_14_ == 0)
{
size_t v___x_15_; size_t v___x_16_; lean_object* v___y_18_; lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_15_ = ((size_t)1ULL);
v___x_16_ = lean_usize_sub(v_i_11_, v___x_15_);
v___x_21_ = lean_array_uget_borrowed(v_as_10_, v___x_16_);
v___x_22_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v___y_9_, v___x_21_);
if (lean_obj_tag(v___x_22_) == 0)
{
lean_object* v___x_23_; 
v___x_23_ = l_Lean_NameSet_empty;
v___y_18_ = v___x_23_;
goto v___jp_17_;
}
else
{
lean_object* v_val_24_; 
v_val_24_ = lean_ctor_get(v___x_22_, 0);
lean_inc(v_val_24_);
lean_dec_ref_known(v___x_22_, 1);
v___y_18_ = v_val_24_;
goto v___jp_17_;
}
v___jp_17_:
{
lean_object* v___x_19_; 
v___x_19_ = lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__0_spec__0(v_b_13_, v___y_18_);
v_i_11_ = v___x_16_;
v_b_13_ = v___x_19_;
goto _start;
}
}
else
{
return v_b_13_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__1___boxed(lean_object* v___y_25_, lean_object* v_as_26_, lean_object* v_i_27_, lean_object* v_stop_28_, lean_object* v_b_29_){
_start:
{
size_t v_i_boxed_30_; size_t v_stop_boxed_31_; lean_object* v_res_32_; 
v_i_boxed_30_ = lean_unbox_usize(v_i_27_);
lean_dec(v_i_27_);
v_stop_boxed_31_ = lean_unbox_usize(v_stop_28_);
lean_dec(v_stop_28_);
v_res_32_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__1(v___y_25_, v_as_26_, v_i_boxed_30_, v_stop_boxed_31_, v_b_29_);
lean_dec_ref(v_as_26_);
lean_dec(v___y_25_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process(lean_object* v_m_35_, lean_object* v_r_36_, lean_object* v_n_37_, lean_object* v_i_38_){
_start:
{
uint8_t v___x_39_; 
v___x_39_ = l_Std_DTreeMap_Internal_Impl_contains___at___00Lean_NameMap_contains_spec__0___redArg(v_n_37_, v_r_36_);
if (v___x_39_ == 0)
{
lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___y_43_; uint8_t v___x_52_; 
v___x_40_ = lean_array_get_size(v_i_38_);
v___x_41_ = lean_unsigned_to_nat(0u);
v___x_52_ = lean_nat_dec_lt(v___x_41_, v___x_40_);
if (v___x_52_ == 0)
{
v___y_43_ = v_r_36_;
goto v___jp_42_;
}
else
{
size_t v___x_53_; size_t v___x_54_; lean_object* v___x_55_; 
v___x_53_ = lean_usize_of_nat(v___x_40_);
v___x_54_ = ((size_t)0ULL);
v___x_55_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__2(v_m_35_, v_i_38_, v___x_53_, v___x_54_, v_r_36_);
v___y_43_ = v___x_55_;
goto v___jp_42_;
}
v___jp_42_:
{
lean_object* v___x_44_; lean_object* v___x_45_; uint8_t v___x_46_; 
lean_inc_ref(v_i_38_);
v___x_44_ = lean_array_to_list(v_i_38_);
v___x_45_ = l_Lean_NameSet_ofList(v___x_44_);
lean_dec(v___x_44_);
v___x_46_ = lean_nat_dec_lt(v___x_41_, v___x_40_);
if (v___x_46_ == 0)
{
lean_object* v___x_47_; 
lean_dec_ref(v_i_38_);
v___x_47_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_n_37_, v___x_45_, v___y_43_);
return v___x_47_;
}
else
{
size_t v___x_48_; size_t v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_48_ = lean_usize_of_nat(v___x_40_);
v___x_49_ = ((size_t)0ULL);
v___x_50_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__1(v___y_43_, v_i_38_, v___x_48_, v___x_49_, v___x_45_);
lean_dec_ref(v_i_38_);
v___x_51_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_n_37_, v___x_50_, v___y_43_);
return v___x_51_;
}
}
}
else
{
lean_dec_ref(v_i_38_);
lean_dec(v_n_37_);
return v_r_36_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__2(lean_object* v_m_56_, lean_object* v_as_57_, size_t v_i_58_, size_t v_stop_59_, lean_object* v_b_60_){
_start:
{
uint8_t v___x_61_; 
v___x_61_ = lean_usize_dec_eq(v_i_58_, v_stop_59_);
if (v___x_61_ == 0)
{
size_t v___x_62_; size_t v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_62_ = ((size_t)1ULL);
v___x_63_ = lean_usize_sub(v_i_58_, v___x_62_);
v___x_64_ = lean_array_uget_borrowed(v_as_57_, v___x_63_);
v___x_65_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_m_56_, v___x_64_);
if (lean_obj_tag(v___x_65_) == 0)
{
lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_66_ = ((lean_object*)(lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__2___closed__0));
lean_inc(v___x_64_);
v___x_67_ = lp_importGraph___private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process(v_m_56_, v_b_60_, v___x_64_, v___x_66_);
v_i_58_ = v___x_63_;
v_b_60_ = v___x_67_;
goto _start;
}
else
{
lean_object* v_val_69_; lean_object* v___x_70_; 
v_val_69_ = lean_ctor_get(v___x_65_, 0);
lean_inc(v_val_69_);
lean_dec_ref_known(v___x_65_, 1);
lean_inc(v___x_64_);
v___x_70_ = lp_importGraph___private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process(v_m_56_, v_b_60_, v___x_64_, v_val_69_);
v_i_58_ = v___x_63_;
v_b_60_ = v___x_70_;
goto _start;
}
}
else
{
return v_b_60_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__2___boxed(lean_object* v_m_72_, lean_object* v_as_73_, lean_object* v_i_74_, lean_object* v_stop_75_, lean_object* v_b_76_){
_start:
{
size_t v_i_boxed_77_; size_t v_stop_boxed_78_; lean_object* v_res_79_; 
v_i_boxed_77_ = lean_unbox_usize(v_i_74_);
lean_dec(v_i_74_);
v_stop_boxed_78_ = lean_unbox_usize(v_stop_75_);
lean_dec(v_stop_75_);
v_res_79_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__2(v_m_72_, v_as_73_, v_i_boxed_77_, v_stop_boxed_78_, v_b_76_);
lean_dec_ref(v_as_73_);
lean_dec(v_m_72_);
return v_res_79_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process___boxed(lean_object* v_m_80_, lean_object* v_r_81_, lean_object* v_n_82_, lean_object* v_i_83_){
_start:
{
lean_object* v_res_84_; 
v_res_84_ = lp_importGraph___private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process(v_m_80_, v_r_81_, v_n_82_, v_i_83_);
lean_dec(v_m_80_);
return v_res_84_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldl___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__0(lean_object* v_init_85_, lean_object* v_t_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00__private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process_spec__0_spec__0(v_init_85_, v_t_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_NameMap_transitiveClosure_spec__0_spec__0(lean_object* v_m_88_, lean_object* v_init_89_, lean_object* v_x_90_){
_start:
{
if (lean_obj_tag(v_x_90_) == 0)
{
lean_object* v_k_91_; lean_object* v_v_92_; lean_object* v_l_93_; lean_object* v_r_94_; lean_object* v___x_95_; lean_object* v___x_96_; 
v_k_91_ = lean_ctor_get(v_x_90_, 1);
lean_inc(v_k_91_);
v_v_92_ = lean_ctor_get(v_x_90_, 2);
lean_inc(v_v_92_);
v_l_93_ = lean_ctor_get(v_x_90_, 3);
lean_inc(v_l_93_);
v_r_94_ = lean_ctor_get(v_x_90_, 4);
lean_inc(v_r_94_);
lean_dec_ref_known(v_x_90_, 5);
v___x_95_ = lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_NameMap_transitiveClosure_spec__0_spec__0(v_m_88_, v_init_89_, v_l_93_);
v___x_96_ = lp_importGraph___private_ImportGraph_Graph_TransitiveClosure_0__Lean_NameMap_transitiveClosure_process(v_m_88_, v___x_95_, v_k_91_, v_v_92_);
v_init_89_ = v___x_96_;
v_x_90_ = v_r_94_;
goto _start;
}
else
{
return v_init_89_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_NameMap_transitiveClosure_spec__0_spec__0___boxed(lean_object* v_m_98_, lean_object* v_init_99_, lean_object* v_x_100_){
_start:
{
lean_object* v_res_101_; 
v_res_101_ = lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_NameMap_transitiveClosure_spec__0_spec__0(v_m_98_, v_init_99_, v_x_100_);
lean_dec(v_m_98_);
return v_res_101_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_NameMap_transitiveClosure(lean_object* v_m_102_){
_start:
{
lean_object* v___x_103_; lean_object* v___x_104_; 
v___x_103_ = lean_box(1);
lean_inc(v_m_102_);
v___x_104_ = lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_NameMap_transitiveClosure_spec__0_spec__0(v_m_102_, v___x_103_, v_m_102_);
lean_dec(v_m_102_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldl___at___00Lean_NameMap_transitiveClosure_spec__0(lean_object* v_m_105_, lean_object* v_init_106_, lean_object* v_t_107_){
_start:
{
lean_object* v___x_108_; 
v___x_108_ = lp_importGraph_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_NameMap_transitiveClosure_spec__0_spec__0(v_m_105_, v_init_106_, v_t_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_foldl___at___00Lean_NameMap_transitiveClosure_spec__0___boxed(lean_object* v_m_109_, lean_object* v_init_110_, lean_object* v_t_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_importGraph_Std_DTreeMap_Internal_Impl_foldl___at___00Lean_NameMap_transitiveClosure_spec__0(v_m_109_, v_init_110_, v_t_111_);
lean_dec(v_m_109_);
return v_res_112_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Data_NameMap_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_importGraph_ImportGraph_Graph_TransitiveClosure(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Data_NameMap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_importGraph_ImportGraph_Graph_TransitiveClosure(uint8_t builtin) {
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
lean_object* initialize_Lean_Data_NameMap_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_importGraph_ImportGraph_Graph_TransitiveClosure(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Data_NameMap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Graph_TransitiveClosure(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_importGraph_ImportGraph_Graph_TransitiveClosure(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_importGraph_ImportGraph_Graph_TransitiveClosure(builtin);
}
#ifdef __cplusplus
}
#endif
