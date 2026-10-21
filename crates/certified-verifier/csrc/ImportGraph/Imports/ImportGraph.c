// Lean compiler output
// Module: ImportGraph.Imports.ImportGraph
// Imports: public import Init public meta import Init public import Lean.Environment public import Lean.Data.NameMap.Basic
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
size_t lean_usize_sub(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t l_Std_DTreeMap_Internal_Impl_contains___at___00Lean_NameMap_contains_spec__0___redArg(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
lean_object* l_Lean_Environment_getModuleIdx_x3f(lean_object*, lean_object*);
extern lean_object* l_Lean_instInhabitedModuleData_default;
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_maxView___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_minView___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importsOf_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Init"};
static const lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importsOf_spec__0___closed__0 = (const lean_object*)&lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importsOf_spec__0___closed__0_value;
static const lean_ctor_object lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importsOf_spec__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importsOf_spec__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(152, 102, 12, 179, 200, 220, 30, 26)}};
static const lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importsOf_spec__0___closed__1 = (const lean_object*)&lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importsOf_spec__0___closed__1_value;
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importsOf_spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importsOf_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Environment_importsOf_spec__1(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Environment_importsOf_spec__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_importGraph_Lean_Environment_importsOf___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_importGraph_Lean_Environment_importsOf___closed__0 = (const lean_object*)&lp_importGraph_Lean_Environment_importsOf___closed__0_value;
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_importsOf(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_importsOf___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_ImportGraph_Imports_ImportGraph_0__Lean_Environment_importGraph_process(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Imports_ImportGraph_0__Lean_Environment_importGraph_process_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Imports_ImportGraph_0__Lean_Environment_importGraph_process_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_ImportGraph_Imports_ImportGraph_0__Lean_Environment_importGraph_process___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importGraph_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importGraph_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Environment_importGraph_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Environment_importGraph_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_importGraph(lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_importGraph___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Environment_importGraph_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Environment_importGraph_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importsOf_spec__0(lean_object* v_as_4_, size_t v_i_5_, size_t v_stop_6_, lean_object* v_b_7_){
_start:
{
lean_object* v___y_9_; uint8_t v___x_13_; 
v___x_13_ = lean_usize_dec_eq(v_i_5_, v_stop_6_);
if (v___x_13_ == 0)
{
lean_object* v___x_14_; lean_object* v___x_15_; uint8_t v___x_16_; 
v___x_14_ = lean_array_uget_borrowed(v_as_4_, v_i_5_);
v___x_15_ = ((lean_object*)(lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importsOf_spec__0___closed__1));
v___x_16_ = lean_name_eq(v___x_14_, v___x_15_);
if (v___x_16_ == 0)
{
lean_object* v___x_17_; 
lean_inc(v___x_14_);
v___x_17_ = lean_array_push(v_b_7_, v___x_14_);
v___y_9_ = v___x_17_;
goto v___jp_8_;
}
else
{
v___y_9_ = v_b_7_;
goto v___jp_8_;
}
}
else
{
return v_b_7_;
}
v___jp_8_:
{
size_t v___x_10_; size_t v___x_11_; 
v___x_10_ = ((size_t)1ULL);
v___x_11_ = lean_usize_add(v_i_5_, v___x_10_);
v_i_5_ = v___x_11_;
v_b_7_ = v___y_9_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importsOf_spec__0___boxed(lean_object* v_as_18_, lean_object* v_i_19_, lean_object* v_stop_20_, lean_object* v_b_21_){
_start:
{
size_t v_i_boxed_22_; size_t v_stop_boxed_23_; lean_object* v_res_24_; 
v_i_boxed_22_ = lean_unbox_usize(v_i_19_);
lean_dec(v_i_19_);
v_stop_boxed_23_ = lean_unbox_usize(v_stop_20_);
lean_dec(v_stop_20_);
v_res_24_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importsOf_spec__0(v_as_18_, v_i_boxed_22_, v_stop_boxed_23_, v_b_21_);
lean_dec_ref(v_as_18_);
return v_res_24_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Environment_importsOf_spec__1(size_t v_sz_25_, size_t v_i_26_, lean_object* v_bs_27_){
_start:
{
uint8_t v___x_28_; 
v___x_28_ = lean_usize_dec_lt(v_i_26_, v_sz_25_);
if (v___x_28_ == 0)
{
return v_bs_27_;
}
else
{
lean_object* v_v_29_; lean_object* v_module_30_; lean_object* v___x_31_; lean_object* v_bs_x27_32_; size_t v___x_33_; size_t v___x_34_; lean_object* v___x_35_; 
v_v_29_ = lean_array_uget_borrowed(v_bs_27_, v_i_26_);
v_module_30_ = lean_ctor_get(v_v_29_, 0);
lean_inc(v_module_30_);
v___x_31_ = lean_unsigned_to_nat(0u);
v_bs_x27_32_ = lean_array_uset(v_bs_27_, v_i_26_, v___x_31_);
v___x_33_ = ((size_t)1ULL);
v___x_34_ = lean_usize_add(v_i_26_, v___x_33_);
v___x_35_ = lean_array_uset(v_bs_x27_32_, v_i_26_, v_module_30_);
v_i_26_ = v___x_34_;
v_bs_27_ = v___x_35_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Environment_importsOf_spec__1___boxed(lean_object* v_sz_37_, lean_object* v_i_38_, lean_object* v_bs_39_){
_start:
{
size_t v_sz_boxed_40_; size_t v_i_boxed_41_; lean_object* v_res_42_; 
v_sz_boxed_40_ = lean_unbox_usize(v_sz_37_);
lean_dec(v_sz_37_);
v_i_boxed_41_ = lean_unbox_usize(v_i_38_);
lean_dec(v_i_38_);
v_res_42_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Environment_importsOf_spec__1(v_sz_boxed_40_, v_i_boxed_41_, v_bs_39_);
return v_res_42_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_importsOf(lean_object* v_env_45_, lean_object* v_n_46_){
_start:
{
lean_object* v___y_48_; lean_object* v___x_60_; lean_object* v_mainModule_61_; lean_object* v_imports_62_; lean_object* v_moduleData_63_; uint8_t v___x_64_; 
v___x_60_ = l_Lean_Environment_header(v_env_45_);
v_mainModule_61_ = lean_ctor_get(v___x_60_, 0);
lean_inc(v_mainModule_61_);
v_imports_62_ = lean_ctor_get(v___x_60_, 1);
lean_inc_ref(v_imports_62_);
v_moduleData_63_ = lean_ctor_get(v___x_60_, 6);
lean_inc_ref(v_moduleData_63_);
lean_dec_ref(v___x_60_);
v___x_64_ = lean_name_eq(v_n_46_, v_mainModule_61_);
lean_dec(v_mainModule_61_);
if (v___x_64_ == 0)
{
lean_object* v___x_65_; 
lean_dec_ref(v_imports_62_);
v___x_65_ = l_Lean_Environment_getModuleIdx_x3f(v_env_45_, v_n_46_);
if (lean_obj_tag(v___x_65_) == 0)
{
lean_object* v___x_66_; 
lean_dec_ref(v_moduleData_63_);
v___x_66_ = ((lean_object*)(lp_importGraph_Lean_Environment_importsOf___closed__0));
v___y_48_ = v___x_66_;
goto v___jp_47_;
}
else
{
lean_object* v_val_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v_imports_70_; size_t v_sz_71_; size_t v___x_72_; lean_object* v___x_73_; 
v_val_67_ = lean_ctor_get(v___x_65_, 0);
lean_inc(v_val_67_);
lean_dec_ref_known(v___x_65_, 1);
v___x_68_ = l_Lean_instInhabitedModuleData_default;
v___x_69_ = lean_array_get(v___x_68_, v_moduleData_63_, v_val_67_);
lean_dec(v_val_67_);
lean_dec_ref(v_moduleData_63_);
v_imports_70_ = lean_ctor_get(v___x_69_, 0);
lean_inc_ref(v_imports_70_);
lean_dec(v___x_69_);
v_sz_71_ = lean_array_size(v_imports_70_);
v___x_72_ = ((size_t)0ULL);
v___x_73_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Environment_importsOf_spec__1(v_sz_71_, v___x_72_, v_imports_70_);
v___y_48_ = v___x_73_;
goto v___jp_47_;
}
}
else
{
size_t v_sz_74_; size_t v___x_75_; lean_object* v___x_76_; 
lean_dec_ref(v_moduleData_63_);
v_sz_74_ = lean_array_size(v_imports_62_);
v___x_75_ = ((size_t)0ULL);
v___x_76_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Lean_Environment_importsOf_spec__1(v_sz_74_, v___x_75_, v_imports_62_);
v___y_48_ = v___x_76_;
goto v___jp_47_;
}
v___jp_47_:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; uint8_t v___x_52_; 
v___x_49_ = lean_unsigned_to_nat(0u);
v___x_50_ = lean_array_get_size(v___y_48_);
v___x_51_ = ((lean_object*)(lp_importGraph_Lean_Environment_importsOf___closed__0));
v___x_52_ = lean_nat_dec_lt(v___x_49_, v___x_50_);
if (v___x_52_ == 0)
{
lean_dec_ref(v___y_48_);
return v___x_51_;
}
else
{
uint8_t v___x_53_; 
v___x_53_ = lean_nat_dec_le(v___x_50_, v___x_50_);
if (v___x_53_ == 0)
{
if (v___x_52_ == 0)
{
lean_dec_ref(v___y_48_);
return v___x_51_;
}
else
{
size_t v___x_54_; size_t v___x_55_; lean_object* v___x_56_; 
v___x_54_ = ((size_t)0ULL);
v___x_55_ = lean_usize_of_nat(v___x_50_);
v___x_56_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importsOf_spec__0(v___y_48_, v___x_54_, v___x_55_, v___x_51_);
lean_dec_ref(v___y_48_);
return v___x_56_;
}
}
else
{
size_t v___x_57_; size_t v___x_58_; lean_object* v___x_59_; 
v___x_57_ = ((size_t)0ULL);
v___x_58_ = lean_usize_of_nat(v___x_50_);
v___x_59_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importsOf_spec__0(v___y_48_, v___x_57_, v___x_58_, v___x_51_);
lean_dec_ref(v___y_48_);
return v___x_59_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_importsOf___boxed(lean_object* v_env_77_, lean_object* v_n_78_){
_start:
{
lean_object* v_res_79_; 
v_res_79_ = lp_importGraph_Lean_Environment_importsOf(v_env_77_, v_n_78_);
lean_dec(v_n_78_);
lean_dec_ref(v_env_77_);
return v_res_79_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_ImportGraph_Imports_ImportGraph_0__Lean_Environment_importGraph_process(lean_object* v_env_80_, lean_object* v_i_81_, lean_object* v_m_82_){
_start:
{
uint8_t v___x_83_; 
v___x_83_ = l_Std_DTreeMap_Internal_Impl_contains___at___00Lean_NameMap_contains_spec__0___redArg(v_i_81_, v_m_82_);
if (v___x_83_ == 0)
{
lean_object* v_imports_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; uint8_t v___x_88_; 
v_imports_84_ = lp_importGraph_Lean_Environment_importsOf(v_env_80_, v_i_81_);
lean_inc_ref(v_imports_84_);
v___x_85_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_i_81_, v_imports_84_, v_m_82_);
v___x_86_ = lean_array_get_size(v_imports_84_);
v___x_87_ = lean_unsigned_to_nat(0u);
v___x_88_ = lean_nat_dec_lt(v___x_87_, v___x_86_);
if (v___x_88_ == 0)
{
lean_dec_ref(v_imports_84_);
return v___x_85_;
}
else
{
size_t v___x_89_; size_t v___x_90_; lean_object* v___x_91_; 
v___x_89_ = lean_usize_of_nat(v___x_86_);
v___x_90_ = ((size_t)0ULL);
v___x_91_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Imports_ImportGraph_0__Lean_Environment_importGraph_process_spec__0(v_env_80_, v_imports_84_, v___x_89_, v___x_90_, v___x_85_);
lean_dec_ref(v_imports_84_);
return v___x_91_;
}
}
else
{
lean_dec(v_i_81_);
return v_m_82_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Imports_ImportGraph_0__Lean_Environment_importGraph_process_spec__0(lean_object* v_env_92_, lean_object* v_as_93_, size_t v_i_94_, size_t v_stop_95_, lean_object* v_b_96_){
_start:
{
uint8_t v___x_97_; 
v___x_97_ = lean_usize_dec_eq(v_i_94_, v_stop_95_);
if (v___x_97_ == 0)
{
size_t v___x_98_; size_t v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_98_ = ((size_t)1ULL);
v___x_99_ = lean_usize_sub(v_i_94_, v___x_98_);
v___x_100_ = lean_array_uget_borrowed(v_as_93_, v___x_99_);
lean_inc(v___x_100_);
v___x_101_ = lp_importGraph___private_ImportGraph_Imports_ImportGraph_0__Lean_Environment_importGraph_process(v_env_92_, v___x_100_, v_b_96_);
v_i_94_ = v___x_99_;
v_b_96_ = v___x_101_;
goto _start;
}
else
{
return v_b_96_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Imports_ImportGraph_0__Lean_Environment_importGraph_process_spec__0___boxed(lean_object* v_env_103_, lean_object* v_as_104_, lean_object* v_i_105_, lean_object* v_stop_106_, lean_object* v_b_107_){
_start:
{
size_t v_i_boxed_108_; size_t v_stop_boxed_109_; lean_object* v_res_110_; 
v_i_boxed_108_ = lean_unbox_usize(v_i_105_);
lean_dec(v_i_105_);
v_stop_boxed_109_ = lean_unbox_usize(v_stop_106_);
lean_dec(v_stop_106_);
v_res_110_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__private_ImportGraph_Imports_ImportGraph_0__Lean_Environment_importGraph_process_spec__0(v_env_103_, v_as_104_, v_i_boxed_108_, v_stop_boxed_109_, v_b_107_);
lean_dec_ref(v_as_104_);
lean_dec_ref(v_env_103_);
return v_res_110_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_ImportGraph_Imports_ImportGraph_0__Lean_Environment_importGraph_process___boxed(lean_object* v_env_111_, lean_object* v_i_112_, lean_object* v_m_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_importGraph___private_ImportGraph_Imports_ImportGraph_0__Lean_Environment_importGraph_process(v_env_111_, v_i_112_, v_m_113_);
lean_dec_ref(v_env_111_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importGraph_spec__1(lean_object* v_env_115_, lean_object* v_as_116_, size_t v_i_117_, size_t v_stop_118_, lean_object* v_b_119_){
_start:
{
uint8_t v___x_120_; 
v___x_120_ = lean_usize_dec_eq(v_i_117_, v_stop_118_);
if (v___x_120_ == 0)
{
lean_object* v___x_121_; lean_object* v___x_122_; size_t v___x_123_; size_t v___x_124_; 
v___x_121_ = lean_array_uget_borrowed(v_as_116_, v_i_117_);
lean_inc(v___x_121_);
v___x_122_ = lp_importGraph___private_ImportGraph_Imports_ImportGraph_0__Lean_Environment_importGraph_process(v_env_115_, v___x_121_, v_b_119_);
v___x_123_ = ((size_t)1ULL);
v___x_124_ = lean_usize_add(v_i_117_, v___x_123_);
v_i_117_ = v___x_124_;
v_b_119_ = v___x_122_;
goto _start;
}
else
{
return v_b_119_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importGraph_spec__1___boxed(lean_object* v_env_126_, lean_object* v_as_127_, lean_object* v_i_128_, lean_object* v_stop_129_, lean_object* v_b_130_){
_start:
{
size_t v_i_boxed_131_; size_t v_stop_boxed_132_; lean_object* v_res_133_; 
v_i_boxed_131_ = lean_unbox_usize(v_i_128_);
lean_dec(v_i_128_);
v_stop_boxed_132_ = lean_unbox_usize(v_stop_129_);
lean_dec(v_stop_129_);
v_res_133_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importGraph_spec__1(v_env_126_, v_as_127_, v_i_boxed_131_, v_stop_boxed_132_, v_b_130_);
lean_dec_ref(v_as_127_);
lean_dec_ref(v_env_126_);
return v_res_133_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Environment_importGraph_spec__0___redArg(lean_object* v_k_134_, lean_object* v_t_135_){
_start:
{
if (lean_obj_tag(v_t_135_) == 0)
{
lean_object* v_k_136_; lean_object* v_v_137_; lean_object* v_l_138_; lean_object* v_r_139_; lean_object* v___x_141_; uint8_t v_isShared_142_; uint8_t v_isSharedCheck_793_; 
v_k_136_ = lean_ctor_get(v_t_135_, 1);
v_v_137_ = lean_ctor_get(v_t_135_, 2);
v_l_138_ = lean_ctor_get(v_t_135_, 3);
v_r_139_ = lean_ctor_get(v_t_135_, 4);
v_isSharedCheck_793_ = !lean_is_exclusive(v_t_135_);
if (v_isSharedCheck_793_ == 0)
{
lean_object* v_unused_794_; 
v_unused_794_ = lean_ctor_get(v_t_135_, 0);
lean_dec(v_unused_794_);
v___x_141_ = v_t_135_;
v_isShared_142_ = v_isSharedCheck_793_;
goto v_resetjp_140_;
}
else
{
lean_inc(v_r_139_);
lean_inc(v_l_138_);
lean_inc(v_v_137_);
lean_inc(v_k_136_);
lean_dec(v_t_135_);
v___x_141_ = lean_box(0);
v_isShared_142_ = v_isSharedCheck_793_;
goto v_resetjp_140_;
}
v_resetjp_140_:
{
uint8_t v___x_143_; 
v___x_143_ = l___private_Lean_Data_Name_0__Lean_Name_quickCmpImpl(v_k_134_, v_k_136_);
switch(v___x_143_)
{
case 0:
{
lean_object* v_impl_144_; lean_object* v___x_145_; 
v_impl_144_ = lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Environment_importGraph_spec__0___redArg(v_k_134_, v_l_138_);
v___x_145_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_impl_144_) == 0)
{
if (lean_obj_tag(v_r_139_) == 0)
{
lean_object* v_size_146_; lean_object* v_size_147_; lean_object* v_k_148_; lean_object* v_v_149_; lean_object* v_l_150_; lean_object* v_r_151_; lean_object* v___x_152_; lean_object* v___x_153_; uint8_t v___x_154_; 
v_size_146_ = lean_ctor_get(v_impl_144_, 0);
lean_inc(v_size_146_);
v_size_147_ = lean_ctor_get(v_r_139_, 0);
v_k_148_ = lean_ctor_get(v_r_139_, 1);
v_v_149_ = lean_ctor_get(v_r_139_, 2);
v_l_150_ = lean_ctor_get(v_r_139_, 3);
lean_inc(v_l_150_);
v_r_151_ = lean_ctor_get(v_r_139_, 4);
v___x_152_ = lean_unsigned_to_nat(3u);
v___x_153_ = lean_nat_mul(v___x_152_, v_size_146_);
v___x_154_ = lean_nat_dec_lt(v___x_153_, v_size_147_);
lean_dec(v___x_153_);
if (v___x_154_ == 0)
{
lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_158_; 
lean_dec(v_l_150_);
v___x_155_ = lean_nat_add(v___x_145_, v_size_146_);
lean_dec(v_size_146_);
v___x_156_ = lean_nat_add(v___x_155_, v_size_147_);
lean_dec(v___x_155_);
if (v_isShared_142_ == 0)
{
lean_ctor_set(v___x_141_, 3, v_impl_144_);
lean_ctor_set(v___x_141_, 0, v___x_156_);
v___x_158_ = v___x_141_;
goto v_reusejp_157_;
}
else
{
lean_object* v_reuseFailAlloc_159_; 
v_reuseFailAlloc_159_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_159_, 0, v___x_156_);
lean_ctor_set(v_reuseFailAlloc_159_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_159_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_159_, 3, v_impl_144_);
lean_ctor_set(v_reuseFailAlloc_159_, 4, v_r_139_);
v___x_158_ = v_reuseFailAlloc_159_;
goto v_reusejp_157_;
}
v_reusejp_157_:
{
return v___x_158_;
}
}
else
{
lean_object* v___x_161_; uint8_t v_isShared_162_; uint8_t v_isSharedCheck_223_; 
lean_inc(v_r_151_);
lean_inc(v_v_149_);
lean_inc(v_k_148_);
lean_inc(v_size_147_);
v_isSharedCheck_223_ = !lean_is_exclusive(v_r_139_);
if (v_isSharedCheck_223_ == 0)
{
lean_object* v_unused_224_; lean_object* v_unused_225_; lean_object* v_unused_226_; lean_object* v_unused_227_; lean_object* v_unused_228_; 
v_unused_224_ = lean_ctor_get(v_r_139_, 4);
lean_dec(v_unused_224_);
v_unused_225_ = lean_ctor_get(v_r_139_, 3);
lean_dec(v_unused_225_);
v_unused_226_ = lean_ctor_get(v_r_139_, 2);
lean_dec(v_unused_226_);
v_unused_227_ = lean_ctor_get(v_r_139_, 1);
lean_dec(v_unused_227_);
v_unused_228_ = lean_ctor_get(v_r_139_, 0);
lean_dec(v_unused_228_);
v___x_161_ = v_r_139_;
v_isShared_162_ = v_isSharedCheck_223_;
goto v_resetjp_160_;
}
else
{
lean_dec(v_r_139_);
v___x_161_ = lean_box(0);
v_isShared_162_ = v_isSharedCheck_223_;
goto v_resetjp_160_;
}
v_resetjp_160_:
{
lean_object* v_size_163_; lean_object* v_k_164_; lean_object* v_v_165_; lean_object* v_l_166_; lean_object* v_r_167_; lean_object* v_size_168_; lean_object* v___x_169_; lean_object* v___x_170_; uint8_t v___x_171_; 
v_size_163_ = lean_ctor_get(v_l_150_, 0);
v_k_164_ = lean_ctor_get(v_l_150_, 1);
v_v_165_ = lean_ctor_get(v_l_150_, 2);
v_l_166_ = lean_ctor_get(v_l_150_, 3);
v_r_167_ = lean_ctor_get(v_l_150_, 4);
v_size_168_ = lean_ctor_get(v_r_151_, 0);
v___x_169_ = lean_unsigned_to_nat(2u);
v___x_170_ = lean_nat_mul(v___x_169_, v_size_168_);
v___x_171_ = lean_nat_dec_lt(v_size_163_, v___x_170_);
lean_dec(v___x_170_);
if (v___x_171_ == 0)
{
lean_object* v___x_173_; uint8_t v_isShared_174_; uint8_t v_isSharedCheck_199_; 
lean_inc(v_r_167_);
lean_inc(v_l_166_);
lean_inc(v_v_165_);
lean_inc(v_k_164_);
v_isSharedCheck_199_ = !lean_is_exclusive(v_l_150_);
if (v_isSharedCheck_199_ == 0)
{
lean_object* v_unused_200_; lean_object* v_unused_201_; lean_object* v_unused_202_; lean_object* v_unused_203_; lean_object* v_unused_204_; 
v_unused_200_ = lean_ctor_get(v_l_150_, 4);
lean_dec(v_unused_200_);
v_unused_201_ = lean_ctor_get(v_l_150_, 3);
lean_dec(v_unused_201_);
v_unused_202_ = lean_ctor_get(v_l_150_, 2);
lean_dec(v_unused_202_);
v_unused_203_ = lean_ctor_get(v_l_150_, 1);
lean_dec(v_unused_203_);
v_unused_204_ = lean_ctor_get(v_l_150_, 0);
lean_dec(v_unused_204_);
v___x_173_ = v_l_150_;
v_isShared_174_ = v_isSharedCheck_199_;
goto v_resetjp_172_;
}
else
{
lean_dec(v_l_150_);
v___x_173_ = lean_box(0);
v_isShared_174_ = v_isSharedCheck_199_;
goto v_resetjp_172_;
}
v_resetjp_172_:
{
lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___y_178_; lean_object* v___y_179_; lean_object* v___y_180_; lean_object* v___y_189_; 
v___x_175_ = lean_nat_add(v___x_145_, v_size_146_);
lean_dec(v_size_146_);
v___x_176_ = lean_nat_add(v___x_175_, v_size_147_);
lean_dec(v_size_147_);
if (lean_obj_tag(v_l_166_) == 0)
{
lean_object* v_size_197_; 
v_size_197_ = lean_ctor_get(v_l_166_, 0);
lean_inc(v_size_197_);
v___y_189_ = v_size_197_;
goto v___jp_188_;
}
else
{
lean_object* v___x_198_; 
v___x_198_ = lean_unsigned_to_nat(0u);
v___y_189_ = v___x_198_;
goto v___jp_188_;
}
v___jp_177_:
{
lean_object* v___x_181_; lean_object* v___x_183_; 
v___x_181_ = lean_nat_add(v___y_178_, v___y_180_);
lean_dec(v___y_180_);
lean_dec(v___y_178_);
if (v_isShared_174_ == 0)
{
lean_ctor_set(v___x_173_, 4, v_r_151_);
lean_ctor_set(v___x_173_, 3, v_r_167_);
lean_ctor_set(v___x_173_, 2, v_v_149_);
lean_ctor_set(v___x_173_, 1, v_k_148_);
lean_ctor_set(v___x_173_, 0, v___x_181_);
v___x_183_ = v___x_173_;
goto v_reusejp_182_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v___x_181_);
lean_ctor_set(v_reuseFailAlloc_187_, 1, v_k_148_);
lean_ctor_set(v_reuseFailAlloc_187_, 2, v_v_149_);
lean_ctor_set(v_reuseFailAlloc_187_, 3, v_r_167_);
lean_ctor_set(v_reuseFailAlloc_187_, 4, v_r_151_);
v___x_183_ = v_reuseFailAlloc_187_;
goto v_reusejp_182_;
}
v_reusejp_182_:
{
lean_object* v___x_185_; 
if (v_isShared_162_ == 0)
{
lean_ctor_set(v___x_161_, 4, v___x_183_);
lean_ctor_set(v___x_161_, 3, v___y_179_);
lean_ctor_set(v___x_161_, 2, v_v_165_);
lean_ctor_set(v___x_161_, 1, v_k_164_);
lean_ctor_set(v___x_161_, 0, v___x_176_);
v___x_185_ = v___x_161_;
goto v_reusejp_184_;
}
else
{
lean_object* v_reuseFailAlloc_186_; 
v_reuseFailAlloc_186_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_186_, 0, v___x_176_);
lean_ctor_set(v_reuseFailAlloc_186_, 1, v_k_164_);
lean_ctor_set(v_reuseFailAlloc_186_, 2, v_v_165_);
lean_ctor_set(v_reuseFailAlloc_186_, 3, v___y_179_);
lean_ctor_set(v_reuseFailAlloc_186_, 4, v___x_183_);
v___x_185_ = v_reuseFailAlloc_186_;
goto v_reusejp_184_;
}
v_reusejp_184_:
{
return v___x_185_;
}
}
}
v___jp_188_:
{
lean_object* v___x_190_; lean_object* v___x_192_; 
v___x_190_ = lean_nat_add(v___x_175_, v___y_189_);
lean_dec(v___y_189_);
lean_dec(v___x_175_);
if (v_isShared_142_ == 0)
{
lean_ctor_set(v___x_141_, 4, v_l_166_);
lean_ctor_set(v___x_141_, 3, v_impl_144_);
lean_ctor_set(v___x_141_, 0, v___x_190_);
v___x_192_ = v___x_141_;
goto v_reusejp_191_;
}
else
{
lean_object* v_reuseFailAlloc_196_; 
v_reuseFailAlloc_196_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_196_, 0, v___x_190_);
lean_ctor_set(v_reuseFailAlloc_196_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_196_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_196_, 3, v_impl_144_);
lean_ctor_set(v_reuseFailAlloc_196_, 4, v_l_166_);
v___x_192_ = v_reuseFailAlloc_196_;
goto v_reusejp_191_;
}
v_reusejp_191_:
{
lean_object* v___x_193_; 
v___x_193_ = lean_nat_add(v___x_145_, v_size_168_);
if (lean_obj_tag(v_r_167_) == 0)
{
lean_object* v_size_194_; 
v_size_194_ = lean_ctor_get(v_r_167_, 0);
lean_inc(v_size_194_);
v___y_178_ = v___x_193_;
v___y_179_ = v___x_192_;
v___y_180_ = v_size_194_;
goto v___jp_177_;
}
else
{
lean_object* v___x_195_; 
v___x_195_ = lean_unsigned_to_nat(0u);
v___y_178_ = v___x_193_;
v___y_179_ = v___x_192_;
v___y_180_ = v___x_195_;
goto v___jp_177_;
}
}
}
}
}
else
{
lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_209_; 
lean_del_object(v___x_141_);
v___x_205_ = lean_nat_add(v___x_145_, v_size_146_);
lean_dec(v_size_146_);
v___x_206_ = lean_nat_add(v___x_205_, v_size_147_);
lean_dec(v_size_147_);
v___x_207_ = lean_nat_add(v___x_205_, v_size_163_);
lean_dec(v___x_205_);
lean_inc_ref(v_impl_144_);
if (v_isShared_162_ == 0)
{
lean_ctor_set(v___x_161_, 4, v_l_150_);
lean_ctor_set(v___x_161_, 3, v_impl_144_);
lean_ctor_set(v___x_161_, 2, v_v_137_);
lean_ctor_set(v___x_161_, 1, v_k_136_);
lean_ctor_set(v___x_161_, 0, v___x_207_);
v___x_209_ = v___x_161_;
goto v_reusejp_208_;
}
else
{
lean_object* v_reuseFailAlloc_222_; 
v_reuseFailAlloc_222_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_222_, 0, v___x_207_);
lean_ctor_set(v_reuseFailAlloc_222_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_222_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_222_, 3, v_impl_144_);
lean_ctor_set(v_reuseFailAlloc_222_, 4, v_l_150_);
v___x_209_ = v_reuseFailAlloc_222_;
goto v_reusejp_208_;
}
v_reusejp_208_:
{
lean_object* v___x_211_; uint8_t v_isShared_212_; uint8_t v_isSharedCheck_216_; 
v_isSharedCheck_216_ = !lean_is_exclusive(v_impl_144_);
if (v_isSharedCheck_216_ == 0)
{
lean_object* v_unused_217_; lean_object* v_unused_218_; lean_object* v_unused_219_; lean_object* v_unused_220_; lean_object* v_unused_221_; 
v_unused_217_ = lean_ctor_get(v_impl_144_, 4);
lean_dec(v_unused_217_);
v_unused_218_ = lean_ctor_get(v_impl_144_, 3);
lean_dec(v_unused_218_);
v_unused_219_ = lean_ctor_get(v_impl_144_, 2);
lean_dec(v_unused_219_);
v_unused_220_ = lean_ctor_get(v_impl_144_, 1);
lean_dec(v_unused_220_);
v_unused_221_ = lean_ctor_get(v_impl_144_, 0);
lean_dec(v_unused_221_);
v___x_211_ = v_impl_144_;
v_isShared_212_ = v_isSharedCheck_216_;
goto v_resetjp_210_;
}
else
{
lean_dec(v_impl_144_);
v___x_211_ = lean_box(0);
v_isShared_212_ = v_isSharedCheck_216_;
goto v_resetjp_210_;
}
v_resetjp_210_:
{
lean_object* v___x_214_; 
if (v_isShared_212_ == 0)
{
lean_ctor_set(v___x_211_, 4, v_r_151_);
lean_ctor_set(v___x_211_, 3, v___x_209_);
lean_ctor_set(v___x_211_, 2, v_v_149_);
lean_ctor_set(v___x_211_, 1, v_k_148_);
lean_ctor_set(v___x_211_, 0, v___x_206_);
v___x_214_ = v___x_211_;
goto v_reusejp_213_;
}
else
{
lean_object* v_reuseFailAlloc_215_; 
v_reuseFailAlloc_215_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_215_, 0, v___x_206_);
lean_ctor_set(v_reuseFailAlloc_215_, 1, v_k_148_);
lean_ctor_set(v_reuseFailAlloc_215_, 2, v_v_149_);
lean_ctor_set(v_reuseFailAlloc_215_, 3, v___x_209_);
lean_ctor_set(v_reuseFailAlloc_215_, 4, v_r_151_);
v___x_214_ = v_reuseFailAlloc_215_;
goto v_reusejp_213_;
}
v_reusejp_213_:
{
return v___x_214_;
}
}
}
}
}
}
}
else
{
lean_object* v_size_229_; lean_object* v___x_230_; lean_object* v___x_232_; 
v_size_229_ = lean_ctor_get(v_impl_144_, 0);
lean_inc(v_size_229_);
v___x_230_ = lean_nat_add(v___x_145_, v_size_229_);
lean_dec(v_size_229_);
if (v_isShared_142_ == 0)
{
lean_ctor_set(v___x_141_, 3, v_impl_144_);
lean_ctor_set(v___x_141_, 0, v___x_230_);
v___x_232_ = v___x_141_;
goto v_reusejp_231_;
}
else
{
lean_object* v_reuseFailAlloc_233_; 
v_reuseFailAlloc_233_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_233_, 0, v___x_230_);
lean_ctor_set(v_reuseFailAlloc_233_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_233_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_233_, 3, v_impl_144_);
lean_ctor_set(v_reuseFailAlloc_233_, 4, v_r_139_);
v___x_232_ = v_reuseFailAlloc_233_;
goto v_reusejp_231_;
}
v_reusejp_231_:
{
return v___x_232_;
}
}
}
else
{
if (lean_obj_tag(v_r_139_) == 0)
{
lean_object* v_l_234_; 
v_l_234_ = lean_ctor_get(v_r_139_, 3);
lean_inc(v_l_234_);
if (lean_obj_tag(v_l_234_) == 0)
{
lean_object* v_r_235_; 
v_r_235_ = lean_ctor_get(v_r_139_, 4);
lean_inc(v_r_235_);
if (lean_obj_tag(v_r_235_) == 0)
{
lean_object* v_size_236_; lean_object* v_k_237_; lean_object* v_v_238_; lean_object* v___x_240_; uint8_t v_isShared_241_; uint8_t v_isSharedCheck_251_; 
v_size_236_ = lean_ctor_get(v_r_139_, 0);
v_k_237_ = lean_ctor_get(v_r_139_, 1);
v_v_238_ = lean_ctor_get(v_r_139_, 2);
v_isSharedCheck_251_ = !lean_is_exclusive(v_r_139_);
if (v_isSharedCheck_251_ == 0)
{
lean_object* v_unused_252_; lean_object* v_unused_253_; 
v_unused_252_ = lean_ctor_get(v_r_139_, 4);
lean_dec(v_unused_252_);
v_unused_253_ = lean_ctor_get(v_r_139_, 3);
lean_dec(v_unused_253_);
v___x_240_ = v_r_139_;
v_isShared_241_ = v_isSharedCheck_251_;
goto v_resetjp_239_;
}
else
{
lean_inc(v_v_238_);
lean_inc(v_k_237_);
lean_inc(v_size_236_);
lean_dec(v_r_139_);
v___x_240_ = lean_box(0);
v_isShared_241_ = v_isSharedCheck_251_;
goto v_resetjp_239_;
}
v_resetjp_239_:
{
lean_object* v_size_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_246_; 
v_size_242_ = lean_ctor_get(v_l_234_, 0);
v___x_243_ = lean_nat_add(v___x_145_, v_size_236_);
lean_dec(v_size_236_);
v___x_244_ = lean_nat_add(v___x_145_, v_size_242_);
if (v_isShared_241_ == 0)
{
lean_ctor_set(v___x_240_, 4, v_l_234_);
lean_ctor_set(v___x_240_, 3, v_impl_144_);
lean_ctor_set(v___x_240_, 2, v_v_137_);
lean_ctor_set(v___x_240_, 1, v_k_136_);
lean_ctor_set(v___x_240_, 0, v___x_244_);
v___x_246_ = v___x_240_;
goto v_reusejp_245_;
}
else
{
lean_object* v_reuseFailAlloc_250_; 
v_reuseFailAlloc_250_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_250_, 0, v___x_244_);
lean_ctor_set(v_reuseFailAlloc_250_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_250_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_250_, 3, v_impl_144_);
lean_ctor_set(v_reuseFailAlloc_250_, 4, v_l_234_);
v___x_246_ = v_reuseFailAlloc_250_;
goto v_reusejp_245_;
}
v_reusejp_245_:
{
lean_object* v___x_248_; 
if (v_isShared_142_ == 0)
{
lean_ctor_set(v___x_141_, 4, v_r_235_);
lean_ctor_set(v___x_141_, 3, v___x_246_);
lean_ctor_set(v___x_141_, 2, v_v_238_);
lean_ctor_set(v___x_141_, 1, v_k_237_);
lean_ctor_set(v___x_141_, 0, v___x_243_);
v___x_248_ = v___x_141_;
goto v_reusejp_247_;
}
else
{
lean_object* v_reuseFailAlloc_249_; 
v_reuseFailAlloc_249_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_249_, 0, v___x_243_);
lean_ctor_set(v_reuseFailAlloc_249_, 1, v_k_237_);
lean_ctor_set(v_reuseFailAlloc_249_, 2, v_v_238_);
lean_ctor_set(v_reuseFailAlloc_249_, 3, v___x_246_);
lean_ctor_set(v_reuseFailAlloc_249_, 4, v_r_235_);
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
else
{
lean_object* v_k_254_; lean_object* v_v_255_; lean_object* v___x_257_; uint8_t v_isShared_258_; uint8_t v_isSharedCheck_278_; 
v_k_254_ = lean_ctor_get(v_r_139_, 1);
v_v_255_ = lean_ctor_get(v_r_139_, 2);
v_isSharedCheck_278_ = !lean_is_exclusive(v_r_139_);
if (v_isSharedCheck_278_ == 0)
{
lean_object* v_unused_279_; lean_object* v_unused_280_; lean_object* v_unused_281_; 
v_unused_279_ = lean_ctor_get(v_r_139_, 4);
lean_dec(v_unused_279_);
v_unused_280_ = lean_ctor_get(v_r_139_, 3);
lean_dec(v_unused_280_);
v_unused_281_ = lean_ctor_get(v_r_139_, 0);
lean_dec(v_unused_281_);
v___x_257_ = v_r_139_;
v_isShared_258_ = v_isSharedCheck_278_;
goto v_resetjp_256_;
}
else
{
lean_inc(v_v_255_);
lean_inc(v_k_254_);
lean_dec(v_r_139_);
v___x_257_ = lean_box(0);
v_isShared_258_ = v_isSharedCheck_278_;
goto v_resetjp_256_;
}
v_resetjp_256_:
{
lean_object* v_k_259_; lean_object* v_v_260_; lean_object* v___x_262_; uint8_t v_isShared_263_; uint8_t v_isSharedCheck_274_; 
v_k_259_ = lean_ctor_get(v_l_234_, 1);
v_v_260_ = lean_ctor_get(v_l_234_, 2);
v_isSharedCheck_274_ = !lean_is_exclusive(v_l_234_);
if (v_isSharedCheck_274_ == 0)
{
lean_object* v_unused_275_; lean_object* v_unused_276_; lean_object* v_unused_277_; 
v_unused_275_ = lean_ctor_get(v_l_234_, 4);
lean_dec(v_unused_275_);
v_unused_276_ = lean_ctor_get(v_l_234_, 3);
lean_dec(v_unused_276_);
v_unused_277_ = lean_ctor_get(v_l_234_, 0);
lean_dec(v_unused_277_);
v___x_262_ = v_l_234_;
v_isShared_263_ = v_isSharedCheck_274_;
goto v_resetjp_261_;
}
else
{
lean_inc(v_v_260_);
lean_inc(v_k_259_);
lean_dec(v_l_234_);
v___x_262_ = lean_box(0);
v_isShared_263_ = v_isSharedCheck_274_;
goto v_resetjp_261_;
}
v_resetjp_261_:
{
lean_object* v___x_264_; lean_object* v___x_266_; 
v___x_264_ = lean_unsigned_to_nat(3u);
if (v_isShared_263_ == 0)
{
lean_ctor_set(v___x_262_, 4, v_r_235_);
lean_ctor_set(v___x_262_, 3, v_r_235_);
lean_ctor_set(v___x_262_, 2, v_v_137_);
lean_ctor_set(v___x_262_, 1, v_k_136_);
lean_ctor_set(v___x_262_, 0, v___x_145_);
v___x_266_ = v___x_262_;
goto v_reusejp_265_;
}
else
{
lean_object* v_reuseFailAlloc_273_; 
v_reuseFailAlloc_273_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_273_, 0, v___x_145_);
lean_ctor_set(v_reuseFailAlloc_273_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_273_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_273_, 3, v_r_235_);
lean_ctor_set(v_reuseFailAlloc_273_, 4, v_r_235_);
v___x_266_ = v_reuseFailAlloc_273_;
goto v_reusejp_265_;
}
v_reusejp_265_:
{
lean_object* v___x_268_; 
if (v_isShared_258_ == 0)
{
lean_ctor_set(v___x_257_, 3, v_r_235_);
lean_ctor_set(v___x_257_, 0, v___x_145_);
v___x_268_ = v___x_257_;
goto v_reusejp_267_;
}
else
{
lean_object* v_reuseFailAlloc_272_; 
v_reuseFailAlloc_272_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_272_, 0, v___x_145_);
lean_ctor_set(v_reuseFailAlloc_272_, 1, v_k_254_);
lean_ctor_set(v_reuseFailAlloc_272_, 2, v_v_255_);
lean_ctor_set(v_reuseFailAlloc_272_, 3, v_r_235_);
lean_ctor_set(v_reuseFailAlloc_272_, 4, v_r_235_);
v___x_268_ = v_reuseFailAlloc_272_;
goto v_reusejp_267_;
}
v_reusejp_267_:
{
lean_object* v___x_270_; 
if (v_isShared_142_ == 0)
{
lean_ctor_set(v___x_141_, 4, v___x_268_);
lean_ctor_set(v___x_141_, 3, v___x_266_);
lean_ctor_set(v___x_141_, 2, v_v_260_);
lean_ctor_set(v___x_141_, 1, v_k_259_);
lean_ctor_set(v___x_141_, 0, v___x_264_);
v___x_270_ = v___x_141_;
goto v_reusejp_269_;
}
else
{
lean_object* v_reuseFailAlloc_271_; 
v_reuseFailAlloc_271_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_271_, 0, v___x_264_);
lean_ctor_set(v_reuseFailAlloc_271_, 1, v_k_259_);
lean_ctor_set(v_reuseFailAlloc_271_, 2, v_v_260_);
lean_ctor_set(v_reuseFailAlloc_271_, 3, v___x_266_);
lean_ctor_set(v_reuseFailAlloc_271_, 4, v___x_268_);
v___x_270_ = v_reuseFailAlloc_271_;
goto v_reusejp_269_;
}
v_reusejp_269_:
{
return v___x_270_;
}
}
}
}
}
}
}
else
{
lean_object* v_r_282_; 
v_r_282_ = lean_ctor_get(v_r_139_, 4);
lean_inc(v_r_282_);
if (lean_obj_tag(v_r_282_) == 0)
{
lean_object* v_k_283_; lean_object* v_v_284_; lean_object* v___x_286_; uint8_t v_isShared_287_; uint8_t v_isSharedCheck_295_; 
v_k_283_ = lean_ctor_get(v_r_139_, 1);
v_v_284_ = lean_ctor_get(v_r_139_, 2);
v_isSharedCheck_295_ = !lean_is_exclusive(v_r_139_);
if (v_isSharedCheck_295_ == 0)
{
lean_object* v_unused_296_; lean_object* v_unused_297_; lean_object* v_unused_298_; 
v_unused_296_ = lean_ctor_get(v_r_139_, 4);
lean_dec(v_unused_296_);
v_unused_297_ = lean_ctor_get(v_r_139_, 3);
lean_dec(v_unused_297_);
v_unused_298_ = lean_ctor_get(v_r_139_, 0);
lean_dec(v_unused_298_);
v___x_286_ = v_r_139_;
v_isShared_287_ = v_isSharedCheck_295_;
goto v_resetjp_285_;
}
else
{
lean_inc(v_v_284_);
lean_inc(v_k_283_);
lean_dec(v_r_139_);
v___x_286_ = lean_box(0);
v_isShared_287_ = v_isSharedCheck_295_;
goto v_resetjp_285_;
}
v_resetjp_285_:
{
lean_object* v___x_288_; lean_object* v___x_290_; 
v___x_288_ = lean_unsigned_to_nat(3u);
if (v_isShared_287_ == 0)
{
lean_ctor_set(v___x_286_, 4, v_l_234_);
lean_ctor_set(v___x_286_, 2, v_v_137_);
lean_ctor_set(v___x_286_, 1, v_k_136_);
lean_ctor_set(v___x_286_, 0, v___x_145_);
v___x_290_ = v___x_286_;
goto v_reusejp_289_;
}
else
{
lean_object* v_reuseFailAlloc_294_; 
v_reuseFailAlloc_294_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_294_, 0, v___x_145_);
lean_ctor_set(v_reuseFailAlloc_294_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_294_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_294_, 3, v_l_234_);
lean_ctor_set(v_reuseFailAlloc_294_, 4, v_l_234_);
v___x_290_ = v_reuseFailAlloc_294_;
goto v_reusejp_289_;
}
v_reusejp_289_:
{
lean_object* v___x_292_; 
if (v_isShared_142_ == 0)
{
lean_ctor_set(v___x_141_, 4, v_r_282_);
lean_ctor_set(v___x_141_, 3, v___x_290_);
lean_ctor_set(v___x_141_, 2, v_v_284_);
lean_ctor_set(v___x_141_, 1, v_k_283_);
lean_ctor_set(v___x_141_, 0, v___x_288_);
v___x_292_ = v___x_141_;
goto v_reusejp_291_;
}
else
{
lean_object* v_reuseFailAlloc_293_; 
v_reuseFailAlloc_293_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_293_, 0, v___x_288_);
lean_ctor_set(v_reuseFailAlloc_293_, 1, v_k_283_);
lean_ctor_set(v_reuseFailAlloc_293_, 2, v_v_284_);
lean_ctor_set(v_reuseFailAlloc_293_, 3, v___x_290_);
lean_ctor_set(v_reuseFailAlloc_293_, 4, v_r_282_);
v___x_292_ = v_reuseFailAlloc_293_;
goto v_reusejp_291_;
}
v_reusejp_291_:
{
return v___x_292_;
}
}
}
}
else
{
lean_object* v_size_299_; lean_object* v_k_300_; lean_object* v_v_301_; lean_object* v___x_303_; uint8_t v_isShared_304_; uint8_t v_isSharedCheck_312_; 
v_size_299_ = lean_ctor_get(v_r_139_, 0);
v_k_300_ = lean_ctor_get(v_r_139_, 1);
v_v_301_ = lean_ctor_get(v_r_139_, 2);
v_isSharedCheck_312_ = !lean_is_exclusive(v_r_139_);
if (v_isSharedCheck_312_ == 0)
{
lean_object* v_unused_313_; lean_object* v_unused_314_; 
v_unused_313_ = lean_ctor_get(v_r_139_, 4);
lean_dec(v_unused_313_);
v_unused_314_ = lean_ctor_get(v_r_139_, 3);
lean_dec(v_unused_314_);
v___x_303_ = v_r_139_;
v_isShared_304_ = v_isSharedCheck_312_;
goto v_resetjp_302_;
}
else
{
lean_inc(v_v_301_);
lean_inc(v_k_300_);
lean_inc(v_size_299_);
lean_dec(v_r_139_);
v___x_303_ = lean_box(0);
v_isShared_304_ = v_isSharedCheck_312_;
goto v_resetjp_302_;
}
v_resetjp_302_:
{
lean_object* v___x_306_; 
if (v_isShared_304_ == 0)
{
lean_ctor_set(v___x_303_, 3, v_r_282_);
v___x_306_ = v___x_303_;
goto v_reusejp_305_;
}
else
{
lean_object* v_reuseFailAlloc_311_; 
v_reuseFailAlloc_311_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_311_, 0, v_size_299_);
lean_ctor_set(v_reuseFailAlloc_311_, 1, v_k_300_);
lean_ctor_set(v_reuseFailAlloc_311_, 2, v_v_301_);
lean_ctor_set(v_reuseFailAlloc_311_, 3, v_r_282_);
lean_ctor_set(v_reuseFailAlloc_311_, 4, v_r_282_);
v___x_306_ = v_reuseFailAlloc_311_;
goto v_reusejp_305_;
}
v_reusejp_305_:
{
lean_object* v___x_307_; lean_object* v___x_309_; 
v___x_307_ = lean_unsigned_to_nat(2u);
if (v_isShared_142_ == 0)
{
lean_ctor_set(v___x_141_, 4, v___x_306_);
lean_ctor_set(v___x_141_, 3, v_r_282_);
lean_ctor_set(v___x_141_, 0, v___x_307_);
v___x_309_ = v___x_141_;
goto v_reusejp_308_;
}
else
{
lean_object* v_reuseFailAlloc_310_; 
v_reuseFailAlloc_310_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_310_, 0, v___x_307_);
lean_ctor_set(v_reuseFailAlloc_310_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_310_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_310_, 3, v_r_282_);
lean_ctor_set(v_reuseFailAlloc_310_, 4, v___x_306_);
v___x_309_ = v_reuseFailAlloc_310_;
goto v_reusejp_308_;
}
v_reusejp_308_:
{
return v___x_309_;
}
}
}
}
}
}
else
{
lean_object* v___x_316_; 
if (v_isShared_142_ == 0)
{
lean_ctor_set(v___x_141_, 3, v_r_139_);
lean_ctor_set(v___x_141_, 0, v___x_145_);
v___x_316_ = v___x_141_;
goto v_reusejp_315_;
}
else
{
lean_object* v_reuseFailAlloc_317_; 
v_reuseFailAlloc_317_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_317_, 0, v___x_145_);
lean_ctor_set(v_reuseFailAlloc_317_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_317_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_317_, 3, v_r_139_);
lean_ctor_set(v_reuseFailAlloc_317_, 4, v_r_139_);
v___x_316_ = v_reuseFailAlloc_317_;
goto v_reusejp_315_;
}
v_reusejp_315_:
{
return v___x_316_;
}
}
}
}
case 1:
{
lean_del_object(v___x_141_);
lean_dec(v_v_137_);
lean_dec(v_k_136_);
if (lean_obj_tag(v_l_138_) == 0)
{
if (lean_obj_tag(v_r_139_) == 0)
{
lean_object* v_size_318_; lean_object* v_k_319_; lean_object* v_v_320_; lean_object* v_l_321_; lean_object* v_r_322_; lean_object* v_size_323_; lean_object* v_k_324_; lean_object* v_v_325_; lean_object* v_l_326_; lean_object* v_r_327_; lean_object* v___x_328_; uint8_t v___x_329_; 
v_size_318_ = lean_ctor_get(v_l_138_, 0);
v_k_319_ = lean_ctor_get(v_l_138_, 1);
v_v_320_ = lean_ctor_get(v_l_138_, 2);
v_l_321_ = lean_ctor_get(v_l_138_, 3);
v_r_322_ = lean_ctor_get(v_l_138_, 4);
lean_inc(v_r_322_);
v_size_323_ = lean_ctor_get(v_r_139_, 0);
v_k_324_ = lean_ctor_get(v_r_139_, 1);
v_v_325_ = lean_ctor_get(v_r_139_, 2);
v_l_326_ = lean_ctor_get(v_r_139_, 3);
lean_inc(v_l_326_);
v_r_327_ = lean_ctor_get(v_r_139_, 4);
v___x_328_ = lean_unsigned_to_nat(1u);
v___x_329_ = lean_nat_dec_lt(v_size_318_, v_size_323_);
if (v___x_329_ == 0)
{
lean_object* v___x_331_; uint8_t v_isShared_332_; uint8_t v_isSharedCheck_465_; 
lean_inc(v_l_321_);
lean_inc(v_v_320_);
lean_inc(v_k_319_);
v_isSharedCheck_465_ = !lean_is_exclusive(v_l_138_);
if (v_isSharedCheck_465_ == 0)
{
lean_object* v_unused_466_; lean_object* v_unused_467_; lean_object* v_unused_468_; lean_object* v_unused_469_; lean_object* v_unused_470_; 
v_unused_466_ = lean_ctor_get(v_l_138_, 4);
lean_dec(v_unused_466_);
v_unused_467_ = lean_ctor_get(v_l_138_, 3);
lean_dec(v_unused_467_);
v_unused_468_ = lean_ctor_get(v_l_138_, 2);
lean_dec(v_unused_468_);
v_unused_469_ = lean_ctor_get(v_l_138_, 1);
lean_dec(v_unused_469_);
v_unused_470_ = lean_ctor_get(v_l_138_, 0);
lean_dec(v_unused_470_);
v___x_331_ = v_l_138_;
v_isShared_332_ = v_isSharedCheck_465_;
goto v_resetjp_330_;
}
else
{
lean_dec(v_l_138_);
v___x_331_ = lean_box(0);
v_isShared_332_ = v_isSharedCheck_465_;
goto v_resetjp_330_;
}
v_resetjp_330_:
{
lean_object* v___x_333_; lean_object* v_tree_334_; 
v___x_333_ = l_Std_DTreeMap_Internal_Impl_maxView___redArg(v_k_319_, v_v_320_, v_l_321_, v_r_322_);
v_tree_334_ = lean_ctor_get(v___x_333_, 2);
lean_inc(v_tree_334_);
if (lean_obj_tag(v_tree_334_) == 0)
{
lean_object* v_k_335_; lean_object* v_v_336_; lean_object* v_size_337_; lean_object* v___x_338_; lean_object* v___x_339_; uint8_t v___x_340_; 
v_k_335_ = lean_ctor_get(v___x_333_, 0);
lean_inc(v_k_335_);
v_v_336_ = lean_ctor_get(v___x_333_, 1);
lean_inc(v_v_336_);
lean_dec_ref(v___x_333_);
v_size_337_ = lean_ctor_get(v_tree_334_, 0);
v___x_338_ = lean_unsigned_to_nat(3u);
v___x_339_ = lean_nat_mul(v___x_338_, v_size_337_);
v___x_340_ = lean_nat_dec_lt(v___x_339_, v_size_323_);
lean_dec(v___x_339_);
if (v___x_340_ == 0)
{
lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_344_; 
lean_dec(v_l_326_);
v___x_341_ = lean_nat_add(v___x_328_, v_size_337_);
v___x_342_ = lean_nat_add(v___x_341_, v_size_323_);
lean_dec(v___x_341_);
if (v_isShared_332_ == 0)
{
lean_ctor_set(v___x_331_, 4, v_r_139_);
lean_ctor_set(v___x_331_, 3, v_tree_334_);
lean_ctor_set(v___x_331_, 2, v_v_336_);
lean_ctor_set(v___x_331_, 1, v_k_335_);
lean_ctor_set(v___x_331_, 0, v___x_342_);
v___x_344_ = v___x_331_;
goto v_reusejp_343_;
}
else
{
lean_object* v_reuseFailAlloc_345_; 
v_reuseFailAlloc_345_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_345_, 0, v___x_342_);
lean_ctor_set(v_reuseFailAlloc_345_, 1, v_k_335_);
lean_ctor_set(v_reuseFailAlloc_345_, 2, v_v_336_);
lean_ctor_set(v_reuseFailAlloc_345_, 3, v_tree_334_);
lean_ctor_set(v_reuseFailAlloc_345_, 4, v_r_139_);
v___x_344_ = v_reuseFailAlloc_345_;
goto v_reusejp_343_;
}
v_reusejp_343_:
{
return v___x_344_;
}
}
else
{
lean_object* v___x_347_; uint8_t v_isShared_348_; uint8_t v_isSharedCheck_400_; 
lean_inc(v_r_327_);
lean_inc(v_v_325_);
lean_inc(v_k_324_);
lean_inc(v_size_323_);
v_isSharedCheck_400_ = !lean_is_exclusive(v_r_139_);
if (v_isSharedCheck_400_ == 0)
{
lean_object* v_unused_401_; lean_object* v_unused_402_; lean_object* v_unused_403_; lean_object* v_unused_404_; lean_object* v_unused_405_; 
v_unused_401_ = lean_ctor_get(v_r_139_, 4);
lean_dec(v_unused_401_);
v_unused_402_ = lean_ctor_get(v_r_139_, 3);
lean_dec(v_unused_402_);
v_unused_403_ = lean_ctor_get(v_r_139_, 2);
lean_dec(v_unused_403_);
v_unused_404_ = lean_ctor_get(v_r_139_, 1);
lean_dec(v_unused_404_);
v_unused_405_ = lean_ctor_get(v_r_139_, 0);
lean_dec(v_unused_405_);
v___x_347_ = v_r_139_;
v_isShared_348_ = v_isSharedCheck_400_;
goto v_resetjp_346_;
}
else
{
lean_dec(v_r_139_);
v___x_347_ = lean_box(0);
v_isShared_348_ = v_isSharedCheck_400_;
goto v_resetjp_346_;
}
v_resetjp_346_:
{
lean_object* v_size_349_; lean_object* v_k_350_; lean_object* v_v_351_; lean_object* v_l_352_; lean_object* v_r_353_; lean_object* v_size_354_; lean_object* v___x_355_; lean_object* v___x_356_; uint8_t v___x_357_; 
v_size_349_ = lean_ctor_get(v_l_326_, 0);
v_k_350_ = lean_ctor_get(v_l_326_, 1);
v_v_351_ = lean_ctor_get(v_l_326_, 2);
v_l_352_ = lean_ctor_get(v_l_326_, 3);
v_r_353_ = lean_ctor_get(v_l_326_, 4);
v_size_354_ = lean_ctor_get(v_r_327_, 0);
v___x_355_ = lean_unsigned_to_nat(2u);
v___x_356_ = lean_nat_mul(v___x_355_, v_size_354_);
v___x_357_ = lean_nat_dec_lt(v_size_349_, v___x_356_);
lean_dec(v___x_356_);
if (v___x_357_ == 0)
{
lean_object* v___x_359_; uint8_t v_isShared_360_; uint8_t v_isSharedCheck_385_; 
lean_inc(v_r_353_);
lean_inc(v_l_352_);
lean_inc(v_v_351_);
lean_inc(v_k_350_);
v_isSharedCheck_385_ = !lean_is_exclusive(v_l_326_);
if (v_isSharedCheck_385_ == 0)
{
lean_object* v_unused_386_; lean_object* v_unused_387_; lean_object* v_unused_388_; lean_object* v_unused_389_; lean_object* v_unused_390_; 
v_unused_386_ = lean_ctor_get(v_l_326_, 4);
lean_dec(v_unused_386_);
v_unused_387_ = lean_ctor_get(v_l_326_, 3);
lean_dec(v_unused_387_);
v_unused_388_ = lean_ctor_get(v_l_326_, 2);
lean_dec(v_unused_388_);
v_unused_389_ = lean_ctor_get(v_l_326_, 1);
lean_dec(v_unused_389_);
v_unused_390_ = lean_ctor_get(v_l_326_, 0);
lean_dec(v_unused_390_);
v___x_359_ = v_l_326_;
v_isShared_360_ = v_isSharedCheck_385_;
goto v_resetjp_358_;
}
else
{
lean_dec(v_l_326_);
v___x_359_ = lean_box(0);
v_isShared_360_ = v_isSharedCheck_385_;
goto v_resetjp_358_;
}
v_resetjp_358_:
{
lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___y_364_; lean_object* v___y_365_; lean_object* v___y_366_; lean_object* v___y_375_; 
v___x_361_ = lean_nat_add(v___x_328_, v_size_337_);
v___x_362_ = lean_nat_add(v___x_361_, v_size_323_);
lean_dec(v_size_323_);
if (lean_obj_tag(v_l_352_) == 0)
{
lean_object* v_size_383_; 
v_size_383_ = lean_ctor_get(v_l_352_, 0);
lean_inc(v_size_383_);
v___y_375_ = v_size_383_;
goto v___jp_374_;
}
else
{
lean_object* v___x_384_; 
v___x_384_ = lean_unsigned_to_nat(0u);
v___y_375_ = v___x_384_;
goto v___jp_374_;
}
v___jp_363_:
{
lean_object* v___x_367_; lean_object* v___x_369_; 
v___x_367_ = lean_nat_add(v___y_365_, v___y_366_);
lean_dec(v___y_366_);
lean_dec(v___y_365_);
if (v_isShared_360_ == 0)
{
lean_ctor_set(v___x_359_, 4, v_r_327_);
lean_ctor_set(v___x_359_, 3, v_r_353_);
lean_ctor_set(v___x_359_, 2, v_v_325_);
lean_ctor_set(v___x_359_, 1, v_k_324_);
lean_ctor_set(v___x_359_, 0, v___x_367_);
v___x_369_ = v___x_359_;
goto v_reusejp_368_;
}
else
{
lean_object* v_reuseFailAlloc_373_; 
v_reuseFailAlloc_373_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_373_, 0, v___x_367_);
lean_ctor_set(v_reuseFailAlloc_373_, 1, v_k_324_);
lean_ctor_set(v_reuseFailAlloc_373_, 2, v_v_325_);
lean_ctor_set(v_reuseFailAlloc_373_, 3, v_r_353_);
lean_ctor_set(v_reuseFailAlloc_373_, 4, v_r_327_);
v___x_369_ = v_reuseFailAlloc_373_;
goto v_reusejp_368_;
}
v_reusejp_368_:
{
lean_object* v___x_371_; 
if (v_isShared_348_ == 0)
{
lean_ctor_set(v___x_347_, 4, v___x_369_);
lean_ctor_set(v___x_347_, 3, v___y_364_);
lean_ctor_set(v___x_347_, 2, v_v_351_);
lean_ctor_set(v___x_347_, 1, v_k_350_);
lean_ctor_set(v___x_347_, 0, v___x_362_);
v___x_371_ = v___x_347_;
goto v_reusejp_370_;
}
else
{
lean_object* v_reuseFailAlloc_372_; 
v_reuseFailAlloc_372_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_372_, 0, v___x_362_);
lean_ctor_set(v_reuseFailAlloc_372_, 1, v_k_350_);
lean_ctor_set(v_reuseFailAlloc_372_, 2, v_v_351_);
lean_ctor_set(v_reuseFailAlloc_372_, 3, v___y_364_);
lean_ctor_set(v_reuseFailAlloc_372_, 4, v___x_369_);
v___x_371_ = v_reuseFailAlloc_372_;
goto v_reusejp_370_;
}
v_reusejp_370_:
{
return v___x_371_;
}
}
}
v___jp_374_:
{
lean_object* v___x_376_; lean_object* v___x_378_; 
v___x_376_ = lean_nat_add(v___x_361_, v___y_375_);
lean_dec(v___y_375_);
lean_dec(v___x_361_);
if (v_isShared_332_ == 0)
{
lean_ctor_set(v___x_331_, 4, v_l_352_);
lean_ctor_set(v___x_331_, 3, v_tree_334_);
lean_ctor_set(v___x_331_, 2, v_v_336_);
lean_ctor_set(v___x_331_, 1, v_k_335_);
lean_ctor_set(v___x_331_, 0, v___x_376_);
v___x_378_ = v___x_331_;
goto v_reusejp_377_;
}
else
{
lean_object* v_reuseFailAlloc_382_; 
v_reuseFailAlloc_382_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_382_, 0, v___x_376_);
lean_ctor_set(v_reuseFailAlloc_382_, 1, v_k_335_);
lean_ctor_set(v_reuseFailAlloc_382_, 2, v_v_336_);
lean_ctor_set(v_reuseFailAlloc_382_, 3, v_tree_334_);
lean_ctor_set(v_reuseFailAlloc_382_, 4, v_l_352_);
v___x_378_ = v_reuseFailAlloc_382_;
goto v_reusejp_377_;
}
v_reusejp_377_:
{
lean_object* v___x_379_; 
v___x_379_ = lean_nat_add(v___x_328_, v_size_354_);
if (lean_obj_tag(v_r_353_) == 0)
{
lean_object* v_size_380_; 
v_size_380_ = lean_ctor_get(v_r_353_, 0);
lean_inc(v_size_380_);
v___y_364_ = v___x_378_;
v___y_365_ = v___x_379_;
v___y_366_ = v_size_380_;
goto v___jp_363_;
}
else
{
lean_object* v___x_381_; 
v___x_381_ = lean_unsigned_to_nat(0u);
v___y_364_ = v___x_378_;
v___y_365_ = v___x_379_;
v___y_366_ = v___x_381_;
goto v___jp_363_;
}
}
}
}
}
else
{
lean_object* v___x_391_; lean_object* v___x_392_; lean_object* v___x_393_; lean_object* v___x_395_; 
v___x_391_ = lean_nat_add(v___x_328_, v_size_337_);
v___x_392_ = lean_nat_add(v___x_391_, v_size_323_);
lean_dec(v_size_323_);
v___x_393_ = lean_nat_add(v___x_391_, v_size_349_);
lean_dec(v___x_391_);
if (v_isShared_348_ == 0)
{
lean_ctor_set(v___x_347_, 4, v_l_326_);
lean_ctor_set(v___x_347_, 3, v_tree_334_);
lean_ctor_set(v___x_347_, 2, v_v_336_);
lean_ctor_set(v___x_347_, 1, v_k_335_);
lean_ctor_set(v___x_347_, 0, v___x_393_);
v___x_395_ = v___x_347_;
goto v_reusejp_394_;
}
else
{
lean_object* v_reuseFailAlloc_399_; 
v_reuseFailAlloc_399_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_399_, 0, v___x_393_);
lean_ctor_set(v_reuseFailAlloc_399_, 1, v_k_335_);
lean_ctor_set(v_reuseFailAlloc_399_, 2, v_v_336_);
lean_ctor_set(v_reuseFailAlloc_399_, 3, v_tree_334_);
lean_ctor_set(v_reuseFailAlloc_399_, 4, v_l_326_);
v___x_395_ = v_reuseFailAlloc_399_;
goto v_reusejp_394_;
}
v_reusejp_394_:
{
lean_object* v___x_397_; 
if (v_isShared_332_ == 0)
{
lean_ctor_set(v___x_331_, 4, v_r_327_);
lean_ctor_set(v___x_331_, 3, v___x_395_);
lean_ctor_set(v___x_331_, 2, v_v_325_);
lean_ctor_set(v___x_331_, 1, v_k_324_);
lean_ctor_set(v___x_331_, 0, v___x_392_);
v___x_397_ = v___x_331_;
goto v_reusejp_396_;
}
else
{
lean_object* v_reuseFailAlloc_398_; 
v_reuseFailAlloc_398_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_398_, 0, v___x_392_);
lean_ctor_set(v_reuseFailAlloc_398_, 1, v_k_324_);
lean_ctor_set(v_reuseFailAlloc_398_, 2, v_v_325_);
lean_ctor_set(v_reuseFailAlloc_398_, 3, v___x_395_);
lean_ctor_set(v_reuseFailAlloc_398_, 4, v_r_327_);
v___x_397_ = v_reuseFailAlloc_398_;
goto v_reusejp_396_;
}
v_reusejp_396_:
{
return v___x_397_;
}
}
}
}
}
}
else
{
lean_object* v___x_407_; uint8_t v_isShared_408_; uint8_t v_isSharedCheck_459_; 
lean_inc(v_r_327_);
lean_inc(v_v_325_);
lean_inc(v_k_324_);
lean_inc(v_size_323_);
v_isSharedCheck_459_ = !lean_is_exclusive(v_r_139_);
if (v_isSharedCheck_459_ == 0)
{
lean_object* v_unused_460_; lean_object* v_unused_461_; lean_object* v_unused_462_; lean_object* v_unused_463_; lean_object* v_unused_464_; 
v_unused_460_ = lean_ctor_get(v_r_139_, 4);
lean_dec(v_unused_460_);
v_unused_461_ = lean_ctor_get(v_r_139_, 3);
lean_dec(v_unused_461_);
v_unused_462_ = lean_ctor_get(v_r_139_, 2);
lean_dec(v_unused_462_);
v_unused_463_ = lean_ctor_get(v_r_139_, 1);
lean_dec(v_unused_463_);
v_unused_464_ = lean_ctor_get(v_r_139_, 0);
lean_dec(v_unused_464_);
v___x_407_ = v_r_139_;
v_isShared_408_ = v_isSharedCheck_459_;
goto v_resetjp_406_;
}
else
{
lean_dec(v_r_139_);
v___x_407_ = lean_box(0);
v_isShared_408_ = v_isSharedCheck_459_;
goto v_resetjp_406_;
}
v_resetjp_406_:
{
if (lean_obj_tag(v_l_326_) == 0)
{
if (lean_obj_tag(v_r_327_) == 0)
{
lean_object* v_k_409_; lean_object* v_v_410_; lean_object* v_size_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_415_; 
v_k_409_ = lean_ctor_get(v___x_333_, 0);
lean_inc(v_k_409_);
v_v_410_ = lean_ctor_get(v___x_333_, 1);
lean_inc(v_v_410_);
lean_dec_ref(v___x_333_);
v_size_411_ = lean_ctor_get(v_l_326_, 0);
v___x_412_ = lean_nat_add(v___x_328_, v_size_323_);
lean_dec(v_size_323_);
v___x_413_ = lean_nat_add(v___x_328_, v_size_411_);
if (v_isShared_408_ == 0)
{
lean_ctor_set(v___x_407_, 4, v_l_326_);
lean_ctor_set(v___x_407_, 3, v_tree_334_);
lean_ctor_set(v___x_407_, 2, v_v_410_);
lean_ctor_set(v___x_407_, 1, v_k_409_);
lean_ctor_set(v___x_407_, 0, v___x_413_);
v___x_415_ = v___x_407_;
goto v_reusejp_414_;
}
else
{
lean_object* v_reuseFailAlloc_419_; 
v_reuseFailAlloc_419_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_419_, 0, v___x_413_);
lean_ctor_set(v_reuseFailAlloc_419_, 1, v_k_409_);
lean_ctor_set(v_reuseFailAlloc_419_, 2, v_v_410_);
lean_ctor_set(v_reuseFailAlloc_419_, 3, v_tree_334_);
lean_ctor_set(v_reuseFailAlloc_419_, 4, v_l_326_);
v___x_415_ = v_reuseFailAlloc_419_;
goto v_reusejp_414_;
}
v_reusejp_414_:
{
lean_object* v___x_417_; 
if (v_isShared_332_ == 0)
{
lean_ctor_set(v___x_331_, 4, v_r_327_);
lean_ctor_set(v___x_331_, 3, v___x_415_);
lean_ctor_set(v___x_331_, 2, v_v_325_);
lean_ctor_set(v___x_331_, 1, v_k_324_);
lean_ctor_set(v___x_331_, 0, v___x_412_);
v___x_417_ = v___x_331_;
goto v_reusejp_416_;
}
else
{
lean_object* v_reuseFailAlloc_418_; 
v_reuseFailAlloc_418_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_418_, 0, v___x_412_);
lean_ctor_set(v_reuseFailAlloc_418_, 1, v_k_324_);
lean_ctor_set(v_reuseFailAlloc_418_, 2, v_v_325_);
lean_ctor_set(v_reuseFailAlloc_418_, 3, v___x_415_);
lean_ctor_set(v_reuseFailAlloc_418_, 4, v_r_327_);
v___x_417_ = v_reuseFailAlloc_418_;
goto v_reusejp_416_;
}
v_reusejp_416_:
{
return v___x_417_;
}
}
}
else
{
lean_object* v_k_420_; lean_object* v_v_421_; lean_object* v_k_422_; lean_object* v_v_423_; lean_object* v___x_425_; uint8_t v_isShared_426_; uint8_t v_isSharedCheck_437_; 
lean_dec(v_size_323_);
v_k_420_ = lean_ctor_get(v___x_333_, 0);
lean_inc(v_k_420_);
v_v_421_ = lean_ctor_get(v___x_333_, 1);
lean_inc(v_v_421_);
lean_dec_ref(v___x_333_);
v_k_422_ = lean_ctor_get(v_l_326_, 1);
v_v_423_ = lean_ctor_get(v_l_326_, 2);
v_isSharedCheck_437_ = !lean_is_exclusive(v_l_326_);
if (v_isSharedCheck_437_ == 0)
{
lean_object* v_unused_438_; lean_object* v_unused_439_; lean_object* v_unused_440_; 
v_unused_438_ = lean_ctor_get(v_l_326_, 4);
lean_dec(v_unused_438_);
v_unused_439_ = lean_ctor_get(v_l_326_, 3);
lean_dec(v_unused_439_);
v_unused_440_ = lean_ctor_get(v_l_326_, 0);
lean_dec(v_unused_440_);
v___x_425_ = v_l_326_;
v_isShared_426_ = v_isSharedCheck_437_;
goto v_resetjp_424_;
}
else
{
lean_inc(v_v_423_);
lean_inc(v_k_422_);
lean_dec(v_l_326_);
v___x_425_ = lean_box(0);
v_isShared_426_ = v_isSharedCheck_437_;
goto v_resetjp_424_;
}
v_resetjp_424_:
{
lean_object* v___x_427_; lean_object* v___x_429_; 
v___x_427_ = lean_unsigned_to_nat(3u);
if (v_isShared_426_ == 0)
{
lean_ctor_set(v___x_425_, 4, v_r_327_);
lean_ctor_set(v___x_425_, 3, v_r_327_);
lean_ctor_set(v___x_425_, 2, v_v_421_);
lean_ctor_set(v___x_425_, 1, v_k_420_);
lean_ctor_set(v___x_425_, 0, v___x_328_);
v___x_429_ = v___x_425_;
goto v_reusejp_428_;
}
else
{
lean_object* v_reuseFailAlloc_436_; 
v_reuseFailAlloc_436_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_436_, 0, v___x_328_);
lean_ctor_set(v_reuseFailAlloc_436_, 1, v_k_420_);
lean_ctor_set(v_reuseFailAlloc_436_, 2, v_v_421_);
lean_ctor_set(v_reuseFailAlloc_436_, 3, v_r_327_);
lean_ctor_set(v_reuseFailAlloc_436_, 4, v_r_327_);
v___x_429_ = v_reuseFailAlloc_436_;
goto v_reusejp_428_;
}
v_reusejp_428_:
{
lean_object* v___x_431_; 
if (v_isShared_408_ == 0)
{
lean_ctor_set(v___x_407_, 3, v_r_327_);
lean_ctor_set(v___x_407_, 0, v___x_328_);
v___x_431_ = v___x_407_;
goto v_reusejp_430_;
}
else
{
lean_object* v_reuseFailAlloc_435_; 
v_reuseFailAlloc_435_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_435_, 0, v___x_328_);
lean_ctor_set(v_reuseFailAlloc_435_, 1, v_k_324_);
lean_ctor_set(v_reuseFailAlloc_435_, 2, v_v_325_);
lean_ctor_set(v_reuseFailAlloc_435_, 3, v_r_327_);
lean_ctor_set(v_reuseFailAlloc_435_, 4, v_r_327_);
v___x_431_ = v_reuseFailAlloc_435_;
goto v_reusejp_430_;
}
v_reusejp_430_:
{
lean_object* v___x_433_; 
if (v_isShared_332_ == 0)
{
lean_ctor_set(v___x_331_, 4, v___x_431_);
lean_ctor_set(v___x_331_, 3, v___x_429_);
lean_ctor_set(v___x_331_, 2, v_v_423_);
lean_ctor_set(v___x_331_, 1, v_k_422_);
lean_ctor_set(v___x_331_, 0, v___x_427_);
v___x_433_ = v___x_331_;
goto v_reusejp_432_;
}
else
{
lean_object* v_reuseFailAlloc_434_; 
v_reuseFailAlloc_434_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_434_, 0, v___x_427_);
lean_ctor_set(v_reuseFailAlloc_434_, 1, v_k_422_);
lean_ctor_set(v_reuseFailAlloc_434_, 2, v_v_423_);
lean_ctor_set(v_reuseFailAlloc_434_, 3, v___x_429_);
lean_ctor_set(v_reuseFailAlloc_434_, 4, v___x_431_);
v___x_433_ = v_reuseFailAlloc_434_;
goto v_reusejp_432_;
}
v_reusejp_432_:
{
return v___x_433_;
}
}
}
}
}
}
else
{
if (lean_obj_tag(v_r_327_) == 0)
{
lean_object* v_k_441_; lean_object* v_v_442_; lean_object* v___x_443_; lean_object* v___x_445_; 
lean_dec(v_size_323_);
v_k_441_ = lean_ctor_get(v___x_333_, 0);
lean_inc(v_k_441_);
v_v_442_ = lean_ctor_get(v___x_333_, 1);
lean_inc(v_v_442_);
lean_dec_ref(v___x_333_);
v___x_443_ = lean_unsigned_to_nat(3u);
if (v_isShared_408_ == 0)
{
lean_ctor_set(v___x_407_, 4, v_l_326_);
lean_ctor_set(v___x_407_, 2, v_v_442_);
lean_ctor_set(v___x_407_, 1, v_k_441_);
lean_ctor_set(v___x_407_, 0, v___x_328_);
v___x_445_ = v___x_407_;
goto v_reusejp_444_;
}
else
{
lean_object* v_reuseFailAlloc_449_; 
v_reuseFailAlloc_449_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_449_, 0, v___x_328_);
lean_ctor_set(v_reuseFailAlloc_449_, 1, v_k_441_);
lean_ctor_set(v_reuseFailAlloc_449_, 2, v_v_442_);
lean_ctor_set(v_reuseFailAlloc_449_, 3, v_l_326_);
lean_ctor_set(v_reuseFailAlloc_449_, 4, v_l_326_);
v___x_445_ = v_reuseFailAlloc_449_;
goto v_reusejp_444_;
}
v_reusejp_444_:
{
lean_object* v___x_447_; 
if (v_isShared_332_ == 0)
{
lean_ctor_set(v___x_331_, 4, v_r_327_);
lean_ctor_set(v___x_331_, 3, v___x_445_);
lean_ctor_set(v___x_331_, 2, v_v_325_);
lean_ctor_set(v___x_331_, 1, v_k_324_);
lean_ctor_set(v___x_331_, 0, v___x_443_);
v___x_447_ = v___x_331_;
goto v_reusejp_446_;
}
else
{
lean_object* v_reuseFailAlloc_448_; 
v_reuseFailAlloc_448_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_448_, 0, v___x_443_);
lean_ctor_set(v_reuseFailAlloc_448_, 1, v_k_324_);
lean_ctor_set(v_reuseFailAlloc_448_, 2, v_v_325_);
lean_ctor_set(v_reuseFailAlloc_448_, 3, v___x_445_);
lean_ctor_set(v_reuseFailAlloc_448_, 4, v_r_327_);
v___x_447_ = v_reuseFailAlloc_448_;
goto v_reusejp_446_;
}
v_reusejp_446_:
{
return v___x_447_;
}
}
}
else
{
lean_object* v_k_450_; lean_object* v_v_451_; lean_object* v___x_453_; 
v_k_450_ = lean_ctor_get(v___x_333_, 0);
lean_inc(v_k_450_);
v_v_451_ = lean_ctor_get(v___x_333_, 1);
lean_inc(v_v_451_);
lean_dec_ref(v___x_333_);
if (v_isShared_408_ == 0)
{
lean_ctor_set(v___x_407_, 3, v_r_327_);
v___x_453_ = v___x_407_;
goto v_reusejp_452_;
}
else
{
lean_object* v_reuseFailAlloc_458_; 
v_reuseFailAlloc_458_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_458_, 0, v_size_323_);
lean_ctor_set(v_reuseFailAlloc_458_, 1, v_k_324_);
lean_ctor_set(v_reuseFailAlloc_458_, 2, v_v_325_);
lean_ctor_set(v_reuseFailAlloc_458_, 3, v_r_327_);
lean_ctor_set(v_reuseFailAlloc_458_, 4, v_r_327_);
v___x_453_ = v_reuseFailAlloc_458_;
goto v_reusejp_452_;
}
v_reusejp_452_:
{
lean_object* v___x_454_; lean_object* v___x_456_; 
v___x_454_ = lean_unsigned_to_nat(2u);
if (v_isShared_332_ == 0)
{
lean_ctor_set(v___x_331_, 4, v___x_453_);
lean_ctor_set(v___x_331_, 3, v_r_327_);
lean_ctor_set(v___x_331_, 2, v_v_451_);
lean_ctor_set(v___x_331_, 1, v_k_450_);
lean_ctor_set(v___x_331_, 0, v___x_454_);
v___x_456_ = v___x_331_;
goto v_reusejp_455_;
}
else
{
lean_object* v_reuseFailAlloc_457_; 
v_reuseFailAlloc_457_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_457_, 0, v___x_454_);
lean_ctor_set(v_reuseFailAlloc_457_, 1, v_k_450_);
lean_ctor_set(v_reuseFailAlloc_457_, 2, v_v_451_);
lean_ctor_set(v_reuseFailAlloc_457_, 3, v_r_327_);
lean_ctor_set(v_reuseFailAlloc_457_, 4, v___x_453_);
v___x_456_ = v_reuseFailAlloc_457_;
goto v_reusejp_455_;
}
v_reusejp_455_:
{
return v___x_456_;
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
lean_object* v___x_472_; uint8_t v_isShared_473_; uint8_t v_isSharedCheck_623_; 
lean_inc(v_r_327_);
lean_inc(v_v_325_);
lean_inc(v_k_324_);
v_isSharedCheck_623_ = !lean_is_exclusive(v_r_139_);
if (v_isSharedCheck_623_ == 0)
{
lean_object* v_unused_624_; lean_object* v_unused_625_; lean_object* v_unused_626_; lean_object* v_unused_627_; lean_object* v_unused_628_; 
v_unused_624_ = lean_ctor_get(v_r_139_, 4);
lean_dec(v_unused_624_);
v_unused_625_ = lean_ctor_get(v_r_139_, 3);
lean_dec(v_unused_625_);
v_unused_626_ = lean_ctor_get(v_r_139_, 2);
lean_dec(v_unused_626_);
v_unused_627_ = lean_ctor_get(v_r_139_, 1);
lean_dec(v_unused_627_);
v_unused_628_ = lean_ctor_get(v_r_139_, 0);
lean_dec(v_unused_628_);
v___x_472_ = v_r_139_;
v_isShared_473_ = v_isSharedCheck_623_;
goto v_resetjp_471_;
}
else
{
lean_dec(v_r_139_);
v___x_472_ = lean_box(0);
v_isShared_473_ = v_isSharedCheck_623_;
goto v_resetjp_471_;
}
v_resetjp_471_:
{
lean_object* v___x_474_; lean_object* v_tree_475_; 
v___x_474_ = l_Std_DTreeMap_Internal_Impl_minView___redArg(v_k_324_, v_v_325_, v_l_326_, v_r_327_);
v_tree_475_ = lean_ctor_get(v___x_474_, 2);
lean_inc(v_tree_475_);
if (lean_obj_tag(v_tree_475_) == 0)
{
lean_object* v_k_476_; lean_object* v_v_477_; lean_object* v_size_478_; lean_object* v___x_479_; lean_object* v___x_480_; uint8_t v___x_481_; 
v_k_476_ = lean_ctor_get(v___x_474_, 0);
lean_inc(v_k_476_);
v_v_477_ = lean_ctor_get(v___x_474_, 1);
lean_inc(v_v_477_);
lean_dec_ref(v___x_474_);
v_size_478_ = lean_ctor_get(v_tree_475_, 0);
v___x_479_ = lean_unsigned_to_nat(3u);
v___x_480_ = lean_nat_mul(v___x_479_, v_size_478_);
v___x_481_ = lean_nat_dec_lt(v___x_480_, v_size_318_);
lean_dec(v___x_480_);
if (v___x_481_ == 0)
{
lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_485_; 
lean_dec(v_r_322_);
v___x_482_ = lean_nat_add(v___x_328_, v_size_318_);
v___x_483_ = lean_nat_add(v___x_482_, v_size_478_);
lean_dec(v___x_482_);
if (v_isShared_473_ == 0)
{
lean_ctor_set(v___x_472_, 4, v_tree_475_);
lean_ctor_set(v___x_472_, 3, v_l_138_);
lean_ctor_set(v___x_472_, 2, v_v_477_);
lean_ctor_set(v___x_472_, 1, v_k_476_);
lean_ctor_set(v___x_472_, 0, v___x_483_);
v___x_485_ = v___x_472_;
goto v_reusejp_484_;
}
else
{
lean_object* v_reuseFailAlloc_486_; 
v_reuseFailAlloc_486_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_486_, 0, v___x_483_);
lean_ctor_set(v_reuseFailAlloc_486_, 1, v_k_476_);
lean_ctor_set(v_reuseFailAlloc_486_, 2, v_v_477_);
lean_ctor_set(v_reuseFailAlloc_486_, 3, v_l_138_);
lean_ctor_set(v_reuseFailAlloc_486_, 4, v_tree_475_);
v___x_485_ = v_reuseFailAlloc_486_;
goto v_reusejp_484_;
}
v_reusejp_484_:
{
return v___x_485_;
}
}
else
{
lean_object* v___x_488_; uint8_t v_isShared_489_; uint8_t v_isSharedCheck_552_; 
lean_inc(v_l_321_);
lean_inc(v_v_320_);
lean_inc(v_k_319_);
lean_inc(v_size_318_);
v_isSharedCheck_552_ = !lean_is_exclusive(v_l_138_);
if (v_isSharedCheck_552_ == 0)
{
lean_object* v_unused_553_; lean_object* v_unused_554_; lean_object* v_unused_555_; lean_object* v_unused_556_; lean_object* v_unused_557_; 
v_unused_553_ = lean_ctor_get(v_l_138_, 4);
lean_dec(v_unused_553_);
v_unused_554_ = lean_ctor_get(v_l_138_, 3);
lean_dec(v_unused_554_);
v_unused_555_ = lean_ctor_get(v_l_138_, 2);
lean_dec(v_unused_555_);
v_unused_556_ = lean_ctor_get(v_l_138_, 1);
lean_dec(v_unused_556_);
v_unused_557_ = lean_ctor_get(v_l_138_, 0);
lean_dec(v_unused_557_);
v___x_488_ = v_l_138_;
v_isShared_489_ = v_isSharedCheck_552_;
goto v_resetjp_487_;
}
else
{
lean_dec(v_l_138_);
v___x_488_ = lean_box(0);
v_isShared_489_ = v_isSharedCheck_552_;
goto v_resetjp_487_;
}
v_resetjp_487_:
{
lean_object* v_size_490_; lean_object* v_size_491_; lean_object* v_k_492_; lean_object* v_v_493_; lean_object* v_l_494_; lean_object* v_r_495_; lean_object* v___x_496_; lean_object* v___x_497_; uint8_t v___x_498_; 
v_size_490_ = lean_ctor_get(v_l_321_, 0);
v_size_491_ = lean_ctor_get(v_r_322_, 0);
v_k_492_ = lean_ctor_get(v_r_322_, 1);
v_v_493_ = lean_ctor_get(v_r_322_, 2);
v_l_494_ = lean_ctor_get(v_r_322_, 3);
v_r_495_ = lean_ctor_get(v_r_322_, 4);
v___x_496_ = lean_unsigned_to_nat(2u);
v___x_497_ = lean_nat_mul(v___x_496_, v_size_490_);
v___x_498_ = lean_nat_dec_lt(v_size_491_, v___x_497_);
lean_dec(v___x_497_);
if (v___x_498_ == 0)
{
lean_object* v___x_500_; uint8_t v_isShared_501_; uint8_t v_isSharedCheck_536_; 
lean_inc(v_r_495_);
lean_inc(v_l_494_);
lean_inc(v_v_493_);
lean_inc(v_k_492_);
lean_del_object(v___x_488_);
v_isSharedCheck_536_ = !lean_is_exclusive(v_r_322_);
if (v_isSharedCheck_536_ == 0)
{
lean_object* v_unused_537_; lean_object* v_unused_538_; lean_object* v_unused_539_; lean_object* v_unused_540_; lean_object* v_unused_541_; 
v_unused_537_ = lean_ctor_get(v_r_322_, 4);
lean_dec(v_unused_537_);
v_unused_538_ = lean_ctor_get(v_r_322_, 3);
lean_dec(v_unused_538_);
v_unused_539_ = lean_ctor_get(v_r_322_, 2);
lean_dec(v_unused_539_);
v_unused_540_ = lean_ctor_get(v_r_322_, 1);
lean_dec(v_unused_540_);
v_unused_541_ = lean_ctor_get(v_r_322_, 0);
lean_dec(v_unused_541_);
v___x_500_ = v_r_322_;
v_isShared_501_ = v_isSharedCheck_536_;
goto v_resetjp_499_;
}
else
{
lean_dec(v_r_322_);
v___x_500_ = lean_box(0);
v_isShared_501_ = v_isSharedCheck_536_;
goto v_resetjp_499_;
}
v_resetjp_499_:
{
lean_object* v___x_502_; lean_object* v___x_503_; lean_object* v___y_505_; lean_object* v___y_506_; lean_object* v___y_507_; lean_object* v___x_524_; lean_object* v___y_526_; 
v___x_502_ = lean_nat_add(v___x_328_, v_size_318_);
lean_dec(v_size_318_);
v___x_503_ = lean_nat_add(v___x_502_, v_size_478_);
lean_dec(v___x_502_);
v___x_524_ = lean_nat_add(v___x_328_, v_size_490_);
if (lean_obj_tag(v_l_494_) == 0)
{
lean_object* v_size_534_; 
v_size_534_ = lean_ctor_get(v_l_494_, 0);
lean_inc(v_size_534_);
v___y_526_ = v_size_534_;
goto v___jp_525_;
}
else
{
lean_object* v___x_535_; 
v___x_535_ = lean_unsigned_to_nat(0u);
v___y_526_ = v___x_535_;
goto v___jp_525_;
}
v___jp_504_:
{
lean_object* v___x_508_; lean_object* v___x_510_; 
v___x_508_ = lean_nat_add(v___y_506_, v___y_507_);
lean_dec(v___y_507_);
lean_dec(v___y_506_);
lean_inc_ref(v_tree_475_);
if (v_isShared_501_ == 0)
{
lean_ctor_set(v___x_500_, 4, v_tree_475_);
lean_ctor_set(v___x_500_, 3, v_r_495_);
lean_ctor_set(v___x_500_, 2, v_v_477_);
lean_ctor_set(v___x_500_, 1, v_k_476_);
lean_ctor_set(v___x_500_, 0, v___x_508_);
v___x_510_ = v___x_500_;
goto v_reusejp_509_;
}
else
{
lean_object* v_reuseFailAlloc_523_; 
v_reuseFailAlloc_523_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_523_, 0, v___x_508_);
lean_ctor_set(v_reuseFailAlloc_523_, 1, v_k_476_);
lean_ctor_set(v_reuseFailAlloc_523_, 2, v_v_477_);
lean_ctor_set(v_reuseFailAlloc_523_, 3, v_r_495_);
lean_ctor_set(v_reuseFailAlloc_523_, 4, v_tree_475_);
v___x_510_ = v_reuseFailAlloc_523_;
goto v_reusejp_509_;
}
v_reusejp_509_:
{
lean_object* v___x_512_; uint8_t v_isShared_513_; uint8_t v_isSharedCheck_517_; 
v_isSharedCheck_517_ = !lean_is_exclusive(v_tree_475_);
if (v_isSharedCheck_517_ == 0)
{
lean_object* v_unused_518_; lean_object* v_unused_519_; lean_object* v_unused_520_; lean_object* v_unused_521_; lean_object* v_unused_522_; 
v_unused_518_ = lean_ctor_get(v_tree_475_, 4);
lean_dec(v_unused_518_);
v_unused_519_ = lean_ctor_get(v_tree_475_, 3);
lean_dec(v_unused_519_);
v_unused_520_ = lean_ctor_get(v_tree_475_, 2);
lean_dec(v_unused_520_);
v_unused_521_ = lean_ctor_get(v_tree_475_, 1);
lean_dec(v_unused_521_);
v_unused_522_ = lean_ctor_get(v_tree_475_, 0);
lean_dec(v_unused_522_);
v___x_512_ = v_tree_475_;
v_isShared_513_ = v_isSharedCheck_517_;
goto v_resetjp_511_;
}
else
{
lean_dec(v_tree_475_);
v___x_512_ = lean_box(0);
v_isShared_513_ = v_isSharedCheck_517_;
goto v_resetjp_511_;
}
v_resetjp_511_:
{
lean_object* v___x_515_; 
if (v_isShared_513_ == 0)
{
lean_ctor_set(v___x_512_, 4, v___x_510_);
lean_ctor_set(v___x_512_, 3, v___y_505_);
lean_ctor_set(v___x_512_, 2, v_v_493_);
lean_ctor_set(v___x_512_, 1, v_k_492_);
lean_ctor_set(v___x_512_, 0, v___x_503_);
v___x_515_ = v___x_512_;
goto v_reusejp_514_;
}
else
{
lean_object* v_reuseFailAlloc_516_; 
v_reuseFailAlloc_516_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_516_, 0, v___x_503_);
lean_ctor_set(v_reuseFailAlloc_516_, 1, v_k_492_);
lean_ctor_set(v_reuseFailAlloc_516_, 2, v_v_493_);
lean_ctor_set(v_reuseFailAlloc_516_, 3, v___y_505_);
lean_ctor_set(v_reuseFailAlloc_516_, 4, v___x_510_);
v___x_515_ = v_reuseFailAlloc_516_;
goto v_reusejp_514_;
}
v_reusejp_514_:
{
return v___x_515_;
}
}
}
}
v___jp_525_:
{
lean_object* v___x_527_; lean_object* v___x_529_; 
v___x_527_ = lean_nat_add(v___x_524_, v___y_526_);
lean_dec(v___y_526_);
lean_dec(v___x_524_);
if (v_isShared_473_ == 0)
{
lean_ctor_set(v___x_472_, 4, v_l_494_);
lean_ctor_set(v___x_472_, 3, v_l_321_);
lean_ctor_set(v___x_472_, 2, v_v_320_);
lean_ctor_set(v___x_472_, 1, v_k_319_);
lean_ctor_set(v___x_472_, 0, v___x_527_);
v___x_529_ = v___x_472_;
goto v_reusejp_528_;
}
else
{
lean_object* v_reuseFailAlloc_533_; 
v_reuseFailAlloc_533_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_533_, 0, v___x_527_);
lean_ctor_set(v_reuseFailAlloc_533_, 1, v_k_319_);
lean_ctor_set(v_reuseFailAlloc_533_, 2, v_v_320_);
lean_ctor_set(v_reuseFailAlloc_533_, 3, v_l_321_);
lean_ctor_set(v_reuseFailAlloc_533_, 4, v_l_494_);
v___x_529_ = v_reuseFailAlloc_533_;
goto v_reusejp_528_;
}
v_reusejp_528_:
{
lean_object* v___x_530_; 
v___x_530_ = lean_nat_add(v___x_328_, v_size_478_);
if (lean_obj_tag(v_r_495_) == 0)
{
lean_object* v_size_531_; 
v_size_531_ = lean_ctor_get(v_r_495_, 0);
lean_inc(v_size_531_);
v___y_505_ = v___x_529_;
v___y_506_ = v___x_530_;
v___y_507_ = v_size_531_;
goto v___jp_504_;
}
else
{
lean_object* v___x_532_; 
v___x_532_ = lean_unsigned_to_nat(0u);
v___y_505_ = v___x_529_;
v___y_506_ = v___x_530_;
v___y_507_ = v___x_532_;
goto v___jp_504_;
}
}
}
}
}
else
{
lean_object* v___x_542_; lean_object* v___x_543_; lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_547_; 
v___x_542_ = lean_nat_add(v___x_328_, v_size_318_);
lean_dec(v_size_318_);
v___x_543_ = lean_nat_add(v___x_542_, v_size_478_);
lean_dec(v___x_542_);
v___x_544_ = lean_nat_add(v___x_328_, v_size_478_);
v___x_545_ = lean_nat_add(v___x_544_, v_size_491_);
lean_dec(v___x_544_);
if (v_isShared_473_ == 0)
{
lean_ctor_set(v___x_472_, 4, v_tree_475_);
lean_ctor_set(v___x_472_, 3, v_r_322_);
lean_ctor_set(v___x_472_, 2, v_v_477_);
lean_ctor_set(v___x_472_, 1, v_k_476_);
lean_ctor_set(v___x_472_, 0, v___x_545_);
v___x_547_ = v___x_472_;
goto v_reusejp_546_;
}
else
{
lean_object* v_reuseFailAlloc_551_; 
v_reuseFailAlloc_551_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_551_, 0, v___x_545_);
lean_ctor_set(v_reuseFailAlloc_551_, 1, v_k_476_);
lean_ctor_set(v_reuseFailAlloc_551_, 2, v_v_477_);
lean_ctor_set(v_reuseFailAlloc_551_, 3, v_r_322_);
lean_ctor_set(v_reuseFailAlloc_551_, 4, v_tree_475_);
v___x_547_ = v_reuseFailAlloc_551_;
goto v_reusejp_546_;
}
v_reusejp_546_:
{
lean_object* v___x_549_; 
if (v_isShared_489_ == 0)
{
lean_ctor_set(v___x_488_, 4, v___x_547_);
lean_ctor_set(v___x_488_, 0, v___x_543_);
v___x_549_ = v___x_488_;
goto v_reusejp_548_;
}
else
{
lean_object* v_reuseFailAlloc_550_; 
v_reuseFailAlloc_550_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_550_, 0, v___x_543_);
lean_ctor_set(v_reuseFailAlloc_550_, 1, v_k_319_);
lean_ctor_set(v_reuseFailAlloc_550_, 2, v_v_320_);
lean_ctor_set(v_reuseFailAlloc_550_, 3, v_l_321_);
lean_ctor_set(v_reuseFailAlloc_550_, 4, v___x_547_);
v___x_549_ = v_reuseFailAlloc_550_;
goto v_reusejp_548_;
}
v_reusejp_548_:
{
return v___x_549_;
}
}
}
}
}
}
else
{
if (lean_obj_tag(v_l_321_) == 0)
{
lean_object* v___x_559_; uint8_t v_isShared_560_; uint8_t v_isSharedCheck_581_; 
lean_inc_ref(v_l_321_);
lean_inc(v_v_320_);
lean_inc(v_k_319_);
lean_inc(v_size_318_);
v_isSharedCheck_581_ = !lean_is_exclusive(v_l_138_);
if (v_isSharedCheck_581_ == 0)
{
lean_object* v_unused_582_; lean_object* v_unused_583_; lean_object* v_unused_584_; lean_object* v_unused_585_; lean_object* v_unused_586_; 
v_unused_582_ = lean_ctor_get(v_l_138_, 4);
lean_dec(v_unused_582_);
v_unused_583_ = lean_ctor_get(v_l_138_, 3);
lean_dec(v_unused_583_);
v_unused_584_ = lean_ctor_get(v_l_138_, 2);
lean_dec(v_unused_584_);
v_unused_585_ = lean_ctor_get(v_l_138_, 1);
lean_dec(v_unused_585_);
v_unused_586_ = lean_ctor_get(v_l_138_, 0);
lean_dec(v_unused_586_);
v___x_559_ = v_l_138_;
v_isShared_560_ = v_isSharedCheck_581_;
goto v_resetjp_558_;
}
else
{
lean_dec(v_l_138_);
v___x_559_ = lean_box(0);
v_isShared_560_ = v_isSharedCheck_581_;
goto v_resetjp_558_;
}
v_resetjp_558_:
{
if (lean_obj_tag(v_r_322_) == 0)
{
lean_object* v_k_561_; lean_object* v_v_562_; lean_object* v_size_563_; lean_object* v___x_564_; lean_object* v___x_565_; lean_object* v___x_567_; 
v_k_561_ = lean_ctor_get(v___x_474_, 0);
lean_inc(v_k_561_);
v_v_562_ = lean_ctor_get(v___x_474_, 1);
lean_inc(v_v_562_);
lean_dec_ref(v___x_474_);
v_size_563_ = lean_ctor_get(v_r_322_, 0);
v___x_564_ = lean_nat_add(v___x_328_, v_size_318_);
lean_dec(v_size_318_);
v___x_565_ = lean_nat_add(v___x_328_, v_size_563_);
if (v_isShared_473_ == 0)
{
lean_ctor_set(v___x_472_, 4, v_tree_475_);
lean_ctor_set(v___x_472_, 3, v_r_322_);
lean_ctor_set(v___x_472_, 2, v_v_562_);
lean_ctor_set(v___x_472_, 1, v_k_561_);
lean_ctor_set(v___x_472_, 0, v___x_565_);
v___x_567_ = v___x_472_;
goto v_reusejp_566_;
}
else
{
lean_object* v_reuseFailAlloc_571_; 
v_reuseFailAlloc_571_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_571_, 0, v___x_565_);
lean_ctor_set(v_reuseFailAlloc_571_, 1, v_k_561_);
lean_ctor_set(v_reuseFailAlloc_571_, 2, v_v_562_);
lean_ctor_set(v_reuseFailAlloc_571_, 3, v_r_322_);
lean_ctor_set(v_reuseFailAlloc_571_, 4, v_tree_475_);
v___x_567_ = v_reuseFailAlloc_571_;
goto v_reusejp_566_;
}
v_reusejp_566_:
{
lean_object* v___x_569_; 
if (v_isShared_560_ == 0)
{
lean_ctor_set(v___x_559_, 4, v___x_567_);
lean_ctor_set(v___x_559_, 0, v___x_564_);
v___x_569_ = v___x_559_;
goto v_reusejp_568_;
}
else
{
lean_object* v_reuseFailAlloc_570_; 
v_reuseFailAlloc_570_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_570_, 0, v___x_564_);
lean_ctor_set(v_reuseFailAlloc_570_, 1, v_k_319_);
lean_ctor_set(v_reuseFailAlloc_570_, 2, v_v_320_);
lean_ctor_set(v_reuseFailAlloc_570_, 3, v_l_321_);
lean_ctor_set(v_reuseFailAlloc_570_, 4, v___x_567_);
v___x_569_ = v_reuseFailAlloc_570_;
goto v_reusejp_568_;
}
v_reusejp_568_:
{
return v___x_569_;
}
}
}
else
{
lean_object* v_k_572_; lean_object* v_v_573_; lean_object* v___x_574_; lean_object* v___x_576_; 
lean_dec(v_size_318_);
v_k_572_ = lean_ctor_get(v___x_474_, 0);
lean_inc(v_k_572_);
v_v_573_ = lean_ctor_get(v___x_474_, 1);
lean_inc(v_v_573_);
lean_dec_ref(v___x_474_);
v___x_574_ = lean_unsigned_to_nat(3u);
if (v_isShared_473_ == 0)
{
lean_ctor_set(v___x_472_, 4, v_r_322_);
lean_ctor_set(v___x_472_, 3, v_r_322_);
lean_ctor_set(v___x_472_, 2, v_v_573_);
lean_ctor_set(v___x_472_, 1, v_k_572_);
lean_ctor_set(v___x_472_, 0, v___x_328_);
v___x_576_ = v___x_472_;
goto v_reusejp_575_;
}
else
{
lean_object* v_reuseFailAlloc_580_; 
v_reuseFailAlloc_580_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_580_, 0, v___x_328_);
lean_ctor_set(v_reuseFailAlloc_580_, 1, v_k_572_);
lean_ctor_set(v_reuseFailAlloc_580_, 2, v_v_573_);
lean_ctor_set(v_reuseFailAlloc_580_, 3, v_r_322_);
lean_ctor_set(v_reuseFailAlloc_580_, 4, v_r_322_);
v___x_576_ = v_reuseFailAlloc_580_;
goto v_reusejp_575_;
}
v_reusejp_575_:
{
lean_object* v___x_578_; 
if (v_isShared_560_ == 0)
{
lean_ctor_set(v___x_559_, 4, v___x_576_);
lean_ctor_set(v___x_559_, 0, v___x_574_);
v___x_578_ = v___x_559_;
goto v_reusejp_577_;
}
else
{
lean_object* v_reuseFailAlloc_579_; 
v_reuseFailAlloc_579_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_579_, 0, v___x_574_);
lean_ctor_set(v_reuseFailAlloc_579_, 1, v_k_319_);
lean_ctor_set(v_reuseFailAlloc_579_, 2, v_v_320_);
lean_ctor_set(v_reuseFailAlloc_579_, 3, v_l_321_);
lean_ctor_set(v_reuseFailAlloc_579_, 4, v___x_576_);
v___x_578_ = v_reuseFailAlloc_579_;
goto v_reusejp_577_;
}
v_reusejp_577_:
{
return v___x_578_;
}
}
}
}
}
else
{
if (lean_obj_tag(v_r_322_) == 0)
{
lean_object* v___x_588_; uint8_t v_isShared_589_; uint8_t v_isSharedCheck_611_; 
lean_inc(v_l_321_);
lean_inc(v_v_320_);
lean_inc(v_k_319_);
v_isSharedCheck_611_ = !lean_is_exclusive(v_l_138_);
if (v_isSharedCheck_611_ == 0)
{
lean_object* v_unused_612_; lean_object* v_unused_613_; lean_object* v_unused_614_; lean_object* v_unused_615_; lean_object* v_unused_616_; 
v_unused_612_ = lean_ctor_get(v_l_138_, 4);
lean_dec(v_unused_612_);
v_unused_613_ = lean_ctor_get(v_l_138_, 3);
lean_dec(v_unused_613_);
v_unused_614_ = lean_ctor_get(v_l_138_, 2);
lean_dec(v_unused_614_);
v_unused_615_ = lean_ctor_get(v_l_138_, 1);
lean_dec(v_unused_615_);
v_unused_616_ = lean_ctor_get(v_l_138_, 0);
lean_dec(v_unused_616_);
v___x_588_ = v_l_138_;
v_isShared_589_ = v_isSharedCheck_611_;
goto v_resetjp_587_;
}
else
{
lean_dec(v_l_138_);
v___x_588_ = lean_box(0);
v_isShared_589_ = v_isSharedCheck_611_;
goto v_resetjp_587_;
}
v_resetjp_587_:
{
lean_object* v_k_590_; lean_object* v_v_591_; lean_object* v_k_592_; lean_object* v_v_593_; lean_object* v___x_595_; uint8_t v_isShared_596_; uint8_t v_isSharedCheck_607_; 
v_k_590_ = lean_ctor_get(v___x_474_, 0);
lean_inc(v_k_590_);
v_v_591_ = lean_ctor_get(v___x_474_, 1);
lean_inc(v_v_591_);
lean_dec_ref(v___x_474_);
v_k_592_ = lean_ctor_get(v_r_322_, 1);
v_v_593_ = lean_ctor_get(v_r_322_, 2);
v_isSharedCheck_607_ = !lean_is_exclusive(v_r_322_);
if (v_isSharedCheck_607_ == 0)
{
lean_object* v_unused_608_; lean_object* v_unused_609_; lean_object* v_unused_610_; 
v_unused_608_ = lean_ctor_get(v_r_322_, 4);
lean_dec(v_unused_608_);
v_unused_609_ = lean_ctor_get(v_r_322_, 3);
lean_dec(v_unused_609_);
v_unused_610_ = lean_ctor_get(v_r_322_, 0);
lean_dec(v_unused_610_);
v___x_595_ = v_r_322_;
v_isShared_596_ = v_isSharedCheck_607_;
goto v_resetjp_594_;
}
else
{
lean_inc(v_v_593_);
lean_inc(v_k_592_);
lean_dec(v_r_322_);
v___x_595_ = lean_box(0);
v_isShared_596_ = v_isSharedCheck_607_;
goto v_resetjp_594_;
}
v_resetjp_594_:
{
lean_object* v___x_597_; lean_object* v___x_599_; 
v___x_597_ = lean_unsigned_to_nat(3u);
if (v_isShared_596_ == 0)
{
lean_ctor_set(v___x_595_, 4, v_l_321_);
lean_ctor_set(v___x_595_, 3, v_l_321_);
lean_ctor_set(v___x_595_, 2, v_v_320_);
lean_ctor_set(v___x_595_, 1, v_k_319_);
lean_ctor_set(v___x_595_, 0, v___x_328_);
v___x_599_ = v___x_595_;
goto v_reusejp_598_;
}
else
{
lean_object* v_reuseFailAlloc_606_; 
v_reuseFailAlloc_606_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_606_, 0, v___x_328_);
lean_ctor_set(v_reuseFailAlloc_606_, 1, v_k_319_);
lean_ctor_set(v_reuseFailAlloc_606_, 2, v_v_320_);
lean_ctor_set(v_reuseFailAlloc_606_, 3, v_l_321_);
lean_ctor_set(v_reuseFailAlloc_606_, 4, v_l_321_);
v___x_599_ = v_reuseFailAlloc_606_;
goto v_reusejp_598_;
}
v_reusejp_598_:
{
lean_object* v___x_601_; 
if (v_isShared_473_ == 0)
{
lean_ctor_set(v___x_472_, 4, v_l_321_);
lean_ctor_set(v___x_472_, 3, v_l_321_);
lean_ctor_set(v___x_472_, 2, v_v_591_);
lean_ctor_set(v___x_472_, 1, v_k_590_);
lean_ctor_set(v___x_472_, 0, v___x_328_);
v___x_601_ = v___x_472_;
goto v_reusejp_600_;
}
else
{
lean_object* v_reuseFailAlloc_605_; 
v_reuseFailAlloc_605_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_605_, 0, v___x_328_);
lean_ctor_set(v_reuseFailAlloc_605_, 1, v_k_590_);
lean_ctor_set(v_reuseFailAlloc_605_, 2, v_v_591_);
lean_ctor_set(v_reuseFailAlloc_605_, 3, v_l_321_);
lean_ctor_set(v_reuseFailAlloc_605_, 4, v_l_321_);
v___x_601_ = v_reuseFailAlloc_605_;
goto v_reusejp_600_;
}
v_reusejp_600_:
{
lean_object* v___x_603_; 
if (v_isShared_589_ == 0)
{
lean_ctor_set(v___x_588_, 4, v___x_601_);
lean_ctor_set(v___x_588_, 3, v___x_599_);
lean_ctor_set(v___x_588_, 2, v_v_593_);
lean_ctor_set(v___x_588_, 1, v_k_592_);
lean_ctor_set(v___x_588_, 0, v___x_597_);
v___x_603_ = v___x_588_;
goto v_reusejp_602_;
}
else
{
lean_object* v_reuseFailAlloc_604_; 
v_reuseFailAlloc_604_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_604_, 0, v___x_597_);
lean_ctor_set(v_reuseFailAlloc_604_, 1, v_k_592_);
lean_ctor_set(v_reuseFailAlloc_604_, 2, v_v_593_);
lean_ctor_set(v_reuseFailAlloc_604_, 3, v___x_599_);
lean_ctor_set(v_reuseFailAlloc_604_, 4, v___x_601_);
v___x_603_ = v_reuseFailAlloc_604_;
goto v_reusejp_602_;
}
v_reusejp_602_:
{
return v___x_603_;
}
}
}
}
}
}
else
{
lean_object* v_k_617_; lean_object* v_v_618_; lean_object* v___x_619_; lean_object* v___x_621_; 
v_k_617_ = lean_ctor_get(v___x_474_, 0);
lean_inc(v_k_617_);
v_v_618_ = lean_ctor_get(v___x_474_, 1);
lean_inc(v_v_618_);
lean_dec_ref(v___x_474_);
v___x_619_ = lean_unsigned_to_nat(2u);
if (v_isShared_473_ == 0)
{
lean_ctor_set(v___x_472_, 4, v_r_322_);
lean_ctor_set(v___x_472_, 3, v_l_138_);
lean_ctor_set(v___x_472_, 2, v_v_618_);
lean_ctor_set(v___x_472_, 1, v_k_617_);
lean_ctor_set(v___x_472_, 0, v___x_619_);
v___x_621_ = v___x_472_;
goto v_reusejp_620_;
}
else
{
lean_object* v_reuseFailAlloc_622_; 
v_reuseFailAlloc_622_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_622_, 0, v___x_619_);
lean_ctor_set(v_reuseFailAlloc_622_, 1, v_k_617_);
lean_ctor_set(v_reuseFailAlloc_622_, 2, v_v_618_);
lean_ctor_set(v_reuseFailAlloc_622_, 3, v_l_138_);
lean_ctor_set(v_reuseFailAlloc_622_, 4, v_r_322_);
v___x_621_ = v_reuseFailAlloc_622_;
goto v_reusejp_620_;
}
v_reusejp_620_:
{
return v___x_621_;
}
}
}
}
}
}
}
else
{
return v_l_138_;
}
}
else
{
return v_r_139_;
}
}
default: 
{
lean_object* v_impl_629_; lean_object* v___x_630_; 
v_impl_629_ = lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Environment_importGraph_spec__0___redArg(v_k_134_, v_r_139_);
v___x_630_ = lean_unsigned_to_nat(1u);
if (lean_obj_tag(v_impl_629_) == 0)
{
if (lean_obj_tag(v_l_138_) == 0)
{
lean_object* v_size_631_; lean_object* v_size_632_; lean_object* v_k_633_; lean_object* v_v_634_; lean_object* v_l_635_; lean_object* v_r_636_; lean_object* v___x_637_; lean_object* v___x_638_; uint8_t v___x_639_; 
v_size_631_ = lean_ctor_get(v_impl_629_, 0);
lean_inc(v_size_631_);
v_size_632_ = lean_ctor_get(v_l_138_, 0);
v_k_633_ = lean_ctor_get(v_l_138_, 1);
v_v_634_ = lean_ctor_get(v_l_138_, 2);
v_l_635_ = lean_ctor_get(v_l_138_, 3);
v_r_636_ = lean_ctor_get(v_l_138_, 4);
lean_inc(v_r_636_);
v___x_637_ = lean_unsigned_to_nat(3u);
v___x_638_ = lean_nat_mul(v___x_637_, v_size_631_);
v___x_639_ = lean_nat_dec_lt(v___x_638_, v_size_632_);
lean_dec(v___x_638_);
if (v___x_639_ == 0)
{
lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_643_; 
lean_dec(v_r_636_);
v___x_640_ = lean_nat_add(v___x_630_, v_size_632_);
v___x_641_ = lean_nat_add(v___x_640_, v_size_631_);
lean_dec(v_size_631_);
lean_dec(v___x_640_);
if (v_isShared_142_ == 0)
{
lean_ctor_set(v___x_141_, 4, v_impl_629_);
lean_ctor_set(v___x_141_, 0, v___x_641_);
v___x_643_ = v___x_141_;
goto v_reusejp_642_;
}
else
{
lean_object* v_reuseFailAlloc_644_; 
v_reuseFailAlloc_644_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_644_, 0, v___x_641_);
lean_ctor_set(v_reuseFailAlloc_644_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_644_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_644_, 3, v_l_138_);
lean_ctor_set(v_reuseFailAlloc_644_, 4, v_impl_629_);
v___x_643_ = v_reuseFailAlloc_644_;
goto v_reusejp_642_;
}
v_reusejp_642_:
{
return v___x_643_;
}
}
else
{
lean_object* v___x_646_; uint8_t v_isShared_647_; uint8_t v_isSharedCheck_710_; 
lean_inc(v_l_635_);
lean_inc(v_v_634_);
lean_inc(v_k_633_);
lean_inc(v_size_632_);
v_isSharedCheck_710_ = !lean_is_exclusive(v_l_138_);
if (v_isSharedCheck_710_ == 0)
{
lean_object* v_unused_711_; lean_object* v_unused_712_; lean_object* v_unused_713_; lean_object* v_unused_714_; lean_object* v_unused_715_; 
v_unused_711_ = lean_ctor_get(v_l_138_, 4);
lean_dec(v_unused_711_);
v_unused_712_ = lean_ctor_get(v_l_138_, 3);
lean_dec(v_unused_712_);
v_unused_713_ = lean_ctor_get(v_l_138_, 2);
lean_dec(v_unused_713_);
v_unused_714_ = lean_ctor_get(v_l_138_, 1);
lean_dec(v_unused_714_);
v_unused_715_ = lean_ctor_get(v_l_138_, 0);
lean_dec(v_unused_715_);
v___x_646_ = v_l_138_;
v_isShared_647_ = v_isSharedCheck_710_;
goto v_resetjp_645_;
}
else
{
lean_dec(v_l_138_);
v___x_646_ = lean_box(0);
v_isShared_647_ = v_isSharedCheck_710_;
goto v_resetjp_645_;
}
v_resetjp_645_:
{
lean_object* v_size_648_; lean_object* v_size_649_; lean_object* v_k_650_; lean_object* v_v_651_; lean_object* v_l_652_; lean_object* v_r_653_; lean_object* v___x_654_; lean_object* v___x_655_; uint8_t v___x_656_; 
v_size_648_ = lean_ctor_get(v_l_635_, 0);
v_size_649_ = lean_ctor_get(v_r_636_, 0);
v_k_650_ = lean_ctor_get(v_r_636_, 1);
v_v_651_ = lean_ctor_get(v_r_636_, 2);
v_l_652_ = lean_ctor_get(v_r_636_, 3);
v_r_653_ = lean_ctor_get(v_r_636_, 4);
v___x_654_ = lean_unsigned_to_nat(2u);
v___x_655_ = lean_nat_mul(v___x_654_, v_size_648_);
v___x_656_ = lean_nat_dec_lt(v_size_649_, v___x_655_);
lean_dec(v___x_655_);
if (v___x_656_ == 0)
{
lean_object* v___x_658_; uint8_t v_isShared_659_; uint8_t v_isSharedCheck_685_; 
lean_inc(v_r_653_);
lean_inc(v_l_652_);
lean_inc(v_v_651_);
lean_inc(v_k_650_);
v_isSharedCheck_685_ = !lean_is_exclusive(v_r_636_);
if (v_isSharedCheck_685_ == 0)
{
lean_object* v_unused_686_; lean_object* v_unused_687_; lean_object* v_unused_688_; lean_object* v_unused_689_; lean_object* v_unused_690_; 
v_unused_686_ = lean_ctor_get(v_r_636_, 4);
lean_dec(v_unused_686_);
v_unused_687_ = lean_ctor_get(v_r_636_, 3);
lean_dec(v_unused_687_);
v_unused_688_ = lean_ctor_get(v_r_636_, 2);
lean_dec(v_unused_688_);
v_unused_689_ = lean_ctor_get(v_r_636_, 1);
lean_dec(v_unused_689_);
v_unused_690_ = lean_ctor_get(v_r_636_, 0);
lean_dec(v_unused_690_);
v___x_658_ = v_r_636_;
v_isShared_659_ = v_isSharedCheck_685_;
goto v_resetjp_657_;
}
else
{
lean_dec(v_r_636_);
v___x_658_ = lean_box(0);
v_isShared_659_ = v_isSharedCheck_685_;
goto v_resetjp_657_;
}
v_resetjp_657_:
{
lean_object* v___x_660_; lean_object* v___x_661_; lean_object* v___y_663_; lean_object* v___y_664_; lean_object* v___y_665_; lean_object* v___x_673_; lean_object* v___y_675_; 
v___x_660_ = lean_nat_add(v___x_630_, v_size_632_);
lean_dec(v_size_632_);
v___x_661_ = lean_nat_add(v___x_660_, v_size_631_);
lean_dec(v___x_660_);
v___x_673_ = lean_nat_add(v___x_630_, v_size_648_);
if (lean_obj_tag(v_l_652_) == 0)
{
lean_object* v_size_683_; 
v_size_683_ = lean_ctor_get(v_l_652_, 0);
lean_inc(v_size_683_);
v___y_675_ = v_size_683_;
goto v___jp_674_;
}
else
{
lean_object* v___x_684_; 
v___x_684_ = lean_unsigned_to_nat(0u);
v___y_675_ = v___x_684_;
goto v___jp_674_;
}
v___jp_662_:
{
lean_object* v___x_666_; lean_object* v___x_668_; 
v___x_666_ = lean_nat_add(v___y_664_, v___y_665_);
lean_dec(v___y_665_);
lean_dec(v___y_664_);
if (v_isShared_659_ == 0)
{
lean_ctor_set(v___x_658_, 4, v_impl_629_);
lean_ctor_set(v___x_658_, 3, v_r_653_);
lean_ctor_set(v___x_658_, 2, v_v_137_);
lean_ctor_set(v___x_658_, 1, v_k_136_);
lean_ctor_set(v___x_658_, 0, v___x_666_);
v___x_668_ = v___x_658_;
goto v_reusejp_667_;
}
else
{
lean_object* v_reuseFailAlloc_672_; 
v_reuseFailAlloc_672_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_672_, 0, v___x_666_);
lean_ctor_set(v_reuseFailAlloc_672_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_672_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_672_, 3, v_r_653_);
lean_ctor_set(v_reuseFailAlloc_672_, 4, v_impl_629_);
v___x_668_ = v_reuseFailAlloc_672_;
goto v_reusejp_667_;
}
v_reusejp_667_:
{
lean_object* v___x_670_; 
if (v_isShared_647_ == 0)
{
lean_ctor_set(v___x_646_, 4, v___x_668_);
lean_ctor_set(v___x_646_, 3, v___y_663_);
lean_ctor_set(v___x_646_, 2, v_v_651_);
lean_ctor_set(v___x_646_, 1, v_k_650_);
lean_ctor_set(v___x_646_, 0, v___x_661_);
v___x_670_ = v___x_646_;
goto v_reusejp_669_;
}
else
{
lean_object* v_reuseFailAlloc_671_; 
v_reuseFailAlloc_671_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_671_, 0, v___x_661_);
lean_ctor_set(v_reuseFailAlloc_671_, 1, v_k_650_);
lean_ctor_set(v_reuseFailAlloc_671_, 2, v_v_651_);
lean_ctor_set(v_reuseFailAlloc_671_, 3, v___y_663_);
lean_ctor_set(v_reuseFailAlloc_671_, 4, v___x_668_);
v___x_670_ = v_reuseFailAlloc_671_;
goto v_reusejp_669_;
}
v_reusejp_669_:
{
return v___x_670_;
}
}
}
v___jp_674_:
{
lean_object* v___x_676_; lean_object* v___x_678_; 
v___x_676_ = lean_nat_add(v___x_673_, v___y_675_);
lean_dec(v___y_675_);
lean_dec(v___x_673_);
if (v_isShared_142_ == 0)
{
lean_ctor_set(v___x_141_, 4, v_l_652_);
lean_ctor_set(v___x_141_, 3, v_l_635_);
lean_ctor_set(v___x_141_, 2, v_v_634_);
lean_ctor_set(v___x_141_, 1, v_k_633_);
lean_ctor_set(v___x_141_, 0, v___x_676_);
v___x_678_ = v___x_141_;
goto v_reusejp_677_;
}
else
{
lean_object* v_reuseFailAlloc_682_; 
v_reuseFailAlloc_682_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_682_, 0, v___x_676_);
lean_ctor_set(v_reuseFailAlloc_682_, 1, v_k_633_);
lean_ctor_set(v_reuseFailAlloc_682_, 2, v_v_634_);
lean_ctor_set(v_reuseFailAlloc_682_, 3, v_l_635_);
lean_ctor_set(v_reuseFailAlloc_682_, 4, v_l_652_);
v___x_678_ = v_reuseFailAlloc_682_;
goto v_reusejp_677_;
}
v_reusejp_677_:
{
lean_object* v___x_679_; 
v___x_679_ = lean_nat_add(v___x_630_, v_size_631_);
lean_dec(v_size_631_);
if (lean_obj_tag(v_r_653_) == 0)
{
lean_object* v_size_680_; 
v_size_680_ = lean_ctor_get(v_r_653_, 0);
lean_inc(v_size_680_);
v___y_663_ = v___x_678_;
v___y_664_ = v___x_679_;
v___y_665_ = v_size_680_;
goto v___jp_662_;
}
else
{
lean_object* v___x_681_; 
v___x_681_ = lean_unsigned_to_nat(0u);
v___y_663_ = v___x_678_;
v___y_664_ = v___x_679_;
v___y_665_ = v___x_681_;
goto v___jp_662_;
}
}
}
}
}
else
{
lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; lean_object* v___x_694_; lean_object* v___x_696_; 
lean_del_object(v___x_141_);
v___x_691_ = lean_nat_add(v___x_630_, v_size_632_);
lean_dec(v_size_632_);
v___x_692_ = lean_nat_add(v___x_691_, v_size_631_);
lean_dec(v___x_691_);
v___x_693_ = lean_nat_add(v___x_630_, v_size_631_);
lean_dec(v_size_631_);
v___x_694_ = lean_nat_add(v___x_693_, v_size_649_);
lean_dec(v___x_693_);
lean_inc_ref(v_impl_629_);
if (v_isShared_647_ == 0)
{
lean_ctor_set(v___x_646_, 4, v_impl_629_);
lean_ctor_set(v___x_646_, 3, v_r_636_);
lean_ctor_set(v___x_646_, 2, v_v_137_);
lean_ctor_set(v___x_646_, 1, v_k_136_);
lean_ctor_set(v___x_646_, 0, v___x_694_);
v___x_696_ = v___x_646_;
goto v_reusejp_695_;
}
else
{
lean_object* v_reuseFailAlloc_709_; 
v_reuseFailAlloc_709_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_709_, 0, v___x_694_);
lean_ctor_set(v_reuseFailAlloc_709_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_709_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_709_, 3, v_r_636_);
lean_ctor_set(v_reuseFailAlloc_709_, 4, v_impl_629_);
v___x_696_ = v_reuseFailAlloc_709_;
goto v_reusejp_695_;
}
v_reusejp_695_:
{
lean_object* v___x_698_; uint8_t v_isShared_699_; uint8_t v_isSharedCheck_703_; 
v_isSharedCheck_703_ = !lean_is_exclusive(v_impl_629_);
if (v_isSharedCheck_703_ == 0)
{
lean_object* v_unused_704_; lean_object* v_unused_705_; lean_object* v_unused_706_; lean_object* v_unused_707_; lean_object* v_unused_708_; 
v_unused_704_ = lean_ctor_get(v_impl_629_, 4);
lean_dec(v_unused_704_);
v_unused_705_ = lean_ctor_get(v_impl_629_, 3);
lean_dec(v_unused_705_);
v_unused_706_ = lean_ctor_get(v_impl_629_, 2);
lean_dec(v_unused_706_);
v_unused_707_ = lean_ctor_get(v_impl_629_, 1);
lean_dec(v_unused_707_);
v_unused_708_ = lean_ctor_get(v_impl_629_, 0);
lean_dec(v_unused_708_);
v___x_698_ = v_impl_629_;
v_isShared_699_ = v_isSharedCheck_703_;
goto v_resetjp_697_;
}
else
{
lean_dec(v_impl_629_);
v___x_698_ = lean_box(0);
v_isShared_699_ = v_isSharedCheck_703_;
goto v_resetjp_697_;
}
v_resetjp_697_:
{
lean_object* v___x_701_; 
if (v_isShared_699_ == 0)
{
lean_ctor_set(v___x_698_, 4, v___x_696_);
lean_ctor_set(v___x_698_, 3, v_l_635_);
lean_ctor_set(v___x_698_, 2, v_v_634_);
lean_ctor_set(v___x_698_, 1, v_k_633_);
lean_ctor_set(v___x_698_, 0, v___x_692_);
v___x_701_ = v___x_698_;
goto v_reusejp_700_;
}
else
{
lean_object* v_reuseFailAlloc_702_; 
v_reuseFailAlloc_702_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_702_, 0, v___x_692_);
lean_ctor_set(v_reuseFailAlloc_702_, 1, v_k_633_);
lean_ctor_set(v_reuseFailAlloc_702_, 2, v_v_634_);
lean_ctor_set(v_reuseFailAlloc_702_, 3, v_l_635_);
lean_ctor_set(v_reuseFailAlloc_702_, 4, v___x_696_);
v___x_701_ = v_reuseFailAlloc_702_;
goto v_reusejp_700_;
}
v_reusejp_700_:
{
return v___x_701_;
}
}
}
}
}
}
}
else
{
lean_object* v_size_716_; lean_object* v___x_717_; lean_object* v___x_719_; 
v_size_716_ = lean_ctor_get(v_impl_629_, 0);
lean_inc(v_size_716_);
v___x_717_ = lean_nat_add(v___x_630_, v_size_716_);
lean_dec(v_size_716_);
if (v_isShared_142_ == 0)
{
lean_ctor_set(v___x_141_, 4, v_impl_629_);
lean_ctor_set(v___x_141_, 0, v___x_717_);
v___x_719_ = v___x_141_;
goto v_reusejp_718_;
}
else
{
lean_object* v_reuseFailAlloc_720_; 
v_reuseFailAlloc_720_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_720_, 0, v___x_717_);
lean_ctor_set(v_reuseFailAlloc_720_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_720_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_720_, 3, v_l_138_);
lean_ctor_set(v_reuseFailAlloc_720_, 4, v_impl_629_);
v___x_719_ = v_reuseFailAlloc_720_;
goto v_reusejp_718_;
}
v_reusejp_718_:
{
return v___x_719_;
}
}
}
else
{
if (lean_obj_tag(v_l_138_) == 0)
{
lean_object* v_l_721_; 
v_l_721_ = lean_ctor_get(v_l_138_, 3);
if (lean_obj_tag(v_l_721_) == 0)
{
lean_object* v_r_722_; 
lean_inc_ref(v_l_721_);
v_r_722_ = lean_ctor_get(v_l_138_, 4);
lean_inc(v_r_722_);
if (lean_obj_tag(v_r_722_) == 0)
{
lean_object* v_size_723_; lean_object* v_k_724_; lean_object* v_v_725_; lean_object* v___x_727_; uint8_t v_isShared_728_; uint8_t v_isSharedCheck_738_; 
v_size_723_ = lean_ctor_get(v_l_138_, 0);
v_k_724_ = lean_ctor_get(v_l_138_, 1);
v_v_725_ = lean_ctor_get(v_l_138_, 2);
v_isSharedCheck_738_ = !lean_is_exclusive(v_l_138_);
if (v_isSharedCheck_738_ == 0)
{
lean_object* v_unused_739_; lean_object* v_unused_740_; 
v_unused_739_ = lean_ctor_get(v_l_138_, 4);
lean_dec(v_unused_739_);
v_unused_740_ = lean_ctor_get(v_l_138_, 3);
lean_dec(v_unused_740_);
v___x_727_ = v_l_138_;
v_isShared_728_ = v_isSharedCheck_738_;
goto v_resetjp_726_;
}
else
{
lean_inc(v_v_725_);
lean_inc(v_k_724_);
lean_inc(v_size_723_);
lean_dec(v_l_138_);
v___x_727_ = lean_box(0);
v_isShared_728_ = v_isSharedCheck_738_;
goto v_resetjp_726_;
}
v_resetjp_726_:
{
lean_object* v_size_729_; lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_733_; 
v_size_729_ = lean_ctor_get(v_r_722_, 0);
v___x_730_ = lean_nat_add(v___x_630_, v_size_723_);
lean_dec(v_size_723_);
v___x_731_ = lean_nat_add(v___x_630_, v_size_729_);
if (v_isShared_728_ == 0)
{
lean_ctor_set(v___x_727_, 4, v_impl_629_);
lean_ctor_set(v___x_727_, 3, v_r_722_);
lean_ctor_set(v___x_727_, 2, v_v_137_);
lean_ctor_set(v___x_727_, 1, v_k_136_);
lean_ctor_set(v___x_727_, 0, v___x_731_);
v___x_733_ = v___x_727_;
goto v_reusejp_732_;
}
else
{
lean_object* v_reuseFailAlloc_737_; 
v_reuseFailAlloc_737_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_737_, 0, v___x_731_);
lean_ctor_set(v_reuseFailAlloc_737_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_737_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_737_, 3, v_r_722_);
lean_ctor_set(v_reuseFailAlloc_737_, 4, v_impl_629_);
v___x_733_ = v_reuseFailAlloc_737_;
goto v_reusejp_732_;
}
v_reusejp_732_:
{
lean_object* v___x_735_; 
if (v_isShared_142_ == 0)
{
lean_ctor_set(v___x_141_, 4, v___x_733_);
lean_ctor_set(v___x_141_, 3, v_l_721_);
lean_ctor_set(v___x_141_, 2, v_v_725_);
lean_ctor_set(v___x_141_, 1, v_k_724_);
lean_ctor_set(v___x_141_, 0, v___x_730_);
v___x_735_ = v___x_141_;
goto v_reusejp_734_;
}
else
{
lean_object* v_reuseFailAlloc_736_; 
v_reuseFailAlloc_736_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_736_, 0, v___x_730_);
lean_ctor_set(v_reuseFailAlloc_736_, 1, v_k_724_);
lean_ctor_set(v_reuseFailAlloc_736_, 2, v_v_725_);
lean_ctor_set(v_reuseFailAlloc_736_, 3, v_l_721_);
lean_ctor_set(v_reuseFailAlloc_736_, 4, v___x_733_);
v___x_735_ = v_reuseFailAlloc_736_;
goto v_reusejp_734_;
}
v_reusejp_734_:
{
return v___x_735_;
}
}
}
}
else
{
lean_object* v_k_741_; lean_object* v_v_742_; lean_object* v___x_744_; uint8_t v_isShared_745_; uint8_t v_isSharedCheck_753_; 
v_k_741_ = lean_ctor_get(v_l_138_, 1);
v_v_742_ = lean_ctor_get(v_l_138_, 2);
v_isSharedCheck_753_ = !lean_is_exclusive(v_l_138_);
if (v_isSharedCheck_753_ == 0)
{
lean_object* v_unused_754_; lean_object* v_unused_755_; lean_object* v_unused_756_; 
v_unused_754_ = lean_ctor_get(v_l_138_, 4);
lean_dec(v_unused_754_);
v_unused_755_ = lean_ctor_get(v_l_138_, 3);
lean_dec(v_unused_755_);
v_unused_756_ = lean_ctor_get(v_l_138_, 0);
lean_dec(v_unused_756_);
v___x_744_ = v_l_138_;
v_isShared_745_ = v_isSharedCheck_753_;
goto v_resetjp_743_;
}
else
{
lean_inc(v_v_742_);
lean_inc(v_k_741_);
lean_dec(v_l_138_);
v___x_744_ = lean_box(0);
v_isShared_745_ = v_isSharedCheck_753_;
goto v_resetjp_743_;
}
v_resetjp_743_:
{
lean_object* v___x_746_; lean_object* v___x_748_; 
v___x_746_ = lean_unsigned_to_nat(3u);
if (v_isShared_745_ == 0)
{
lean_ctor_set(v___x_744_, 3, v_r_722_);
lean_ctor_set(v___x_744_, 2, v_v_137_);
lean_ctor_set(v___x_744_, 1, v_k_136_);
lean_ctor_set(v___x_744_, 0, v___x_630_);
v___x_748_ = v___x_744_;
goto v_reusejp_747_;
}
else
{
lean_object* v_reuseFailAlloc_752_; 
v_reuseFailAlloc_752_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_752_, 0, v___x_630_);
lean_ctor_set(v_reuseFailAlloc_752_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_752_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_752_, 3, v_r_722_);
lean_ctor_set(v_reuseFailAlloc_752_, 4, v_r_722_);
v___x_748_ = v_reuseFailAlloc_752_;
goto v_reusejp_747_;
}
v_reusejp_747_:
{
lean_object* v___x_750_; 
if (v_isShared_142_ == 0)
{
lean_ctor_set(v___x_141_, 4, v___x_748_);
lean_ctor_set(v___x_141_, 3, v_l_721_);
lean_ctor_set(v___x_141_, 2, v_v_742_);
lean_ctor_set(v___x_141_, 1, v_k_741_);
lean_ctor_set(v___x_141_, 0, v___x_746_);
v___x_750_ = v___x_141_;
goto v_reusejp_749_;
}
else
{
lean_object* v_reuseFailAlloc_751_; 
v_reuseFailAlloc_751_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_751_, 0, v___x_746_);
lean_ctor_set(v_reuseFailAlloc_751_, 1, v_k_741_);
lean_ctor_set(v_reuseFailAlloc_751_, 2, v_v_742_);
lean_ctor_set(v_reuseFailAlloc_751_, 3, v_l_721_);
lean_ctor_set(v_reuseFailAlloc_751_, 4, v___x_748_);
v___x_750_ = v_reuseFailAlloc_751_;
goto v_reusejp_749_;
}
v_reusejp_749_:
{
return v___x_750_;
}
}
}
}
}
else
{
lean_object* v_r_757_; 
v_r_757_ = lean_ctor_get(v_l_138_, 4);
lean_inc(v_r_757_);
if (lean_obj_tag(v_r_757_) == 0)
{
lean_object* v_k_758_; lean_object* v_v_759_; lean_object* v___x_761_; uint8_t v_isShared_762_; uint8_t v_isSharedCheck_782_; 
lean_inc(v_l_721_);
v_k_758_ = lean_ctor_get(v_l_138_, 1);
v_v_759_ = lean_ctor_get(v_l_138_, 2);
v_isSharedCheck_782_ = !lean_is_exclusive(v_l_138_);
if (v_isSharedCheck_782_ == 0)
{
lean_object* v_unused_783_; lean_object* v_unused_784_; lean_object* v_unused_785_; 
v_unused_783_ = lean_ctor_get(v_l_138_, 4);
lean_dec(v_unused_783_);
v_unused_784_ = lean_ctor_get(v_l_138_, 3);
lean_dec(v_unused_784_);
v_unused_785_ = lean_ctor_get(v_l_138_, 0);
lean_dec(v_unused_785_);
v___x_761_ = v_l_138_;
v_isShared_762_ = v_isSharedCheck_782_;
goto v_resetjp_760_;
}
else
{
lean_inc(v_v_759_);
lean_inc(v_k_758_);
lean_dec(v_l_138_);
v___x_761_ = lean_box(0);
v_isShared_762_ = v_isSharedCheck_782_;
goto v_resetjp_760_;
}
v_resetjp_760_:
{
lean_object* v_k_763_; lean_object* v_v_764_; lean_object* v___x_766_; uint8_t v_isShared_767_; uint8_t v_isSharedCheck_778_; 
v_k_763_ = lean_ctor_get(v_r_757_, 1);
v_v_764_ = lean_ctor_get(v_r_757_, 2);
v_isSharedCheck_778_ = !lean_is_exclusive(v_r_757_);
if (v_isSharedCheck_778_ == 0)
{
lean_object* v_unused_779_; lean_object* v_unused_780_; lean_object* v_unused_781_; 
v_unused_779_ = lean_ctor_get(v_r_757_, 4);
lean_dec(v_unused_779_);
v_unused_780_ = lean_ctor_get(v_r_757_, 3);
lean_dec(v_unused_780_);
v_unused_781_ = lean_ctor_get(v_r_757_, 0);
lean_dec(v_unused_781_);
v___x_766_ = v_r_757_;
v_isShared_767_ = v_isSharedCheck_778_;
goto v_resetjp_765_;
}
else
{
lean_inc(v_v_764_);
lean_inc(v_k_763_);
lean_dec(v_r_757_);
v___x_766_ = lean_box(0);
v_isShared_767_ = v_isSharedCheck_778_;
goto v_resetjp_765_;
}
v_resetjp_765_:
{
lean_object* v___x_768_; lean_object* v___x_770_; 
v___x_768_ = lean_unsigned_to_nat(3u);
if (v_isShared_767_ == 0)
{
lean_ctor_set(v___x_766_, 4, v_l_721_);
lean_ctor_set(v___x_766_, 3, v_l_721_);
lean_ctor_set(v___x_766_, 2, v_v_759_);
lean_ctor_set(v___x_766_, 1, v_k_758_);
lean_ctor_set(v___x_766_, 0, v___x_630_);
v___x_770_ = v___x_766_;
goto v_reusejp_769_;
}
else
{
lean_object* v_reuseFailAlloc_777_; 
v_reuseFailAlloc_777_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_777_, 0, v___x_630_);
lean_ctor_set(v_reuseFailAlloc_777_, 1, v_k_758_);
lean_ctor_set(v_reuseFailAlloc_777_, 2, v_v_759_);
lean_ctor_set(v_reuseFailAlloc_777_, 3, v_l_721_);
lean_ctor_set(v_reuseFailAlloc_777_, 4, v_l_721_);
v___x_770_ = v_reuseFailAlloc_777_;
goto v_reusejp_769_;
}
v_reusejp_769_:
{
lean_object* v___x_772_; 
if (v_isShared_762_ == 0)
{
lean_ctor_set(v___x_761_, 4, v_l_721_);
lean_ctor_set(v___x_761_, 2, v_v_137_);
lean_ctor_set(v___x_761_, 1, v_k_136_);
lean_ctor_set(v___x_761_, 0, v___x_630_);
v___x_772_ = v___x_761_;
goto v_reusejp_771_;
}
else
{
lean_object* v_reuseFailAlloc_776_; 
v_reuseFailAlloc_776_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_776_, 0, v___x_630_);
lean_ctor_set(v_reuseFailAlloc_776_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_776_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_776_, 3, v_l_721_);
lean_ctor_set(v_reuseFailAlloc_776_, 4, v_l_721_);
v___x_772_ = v_reuseFailAlloc_776_;
goto v_reusejp_771_;
}
v_reusejp_771_:
{
lean_object* v___x_774_; 
if (v_isShared_142_ == 0)
{
lean_ctor_set(v___x_141_, 4, v___x_772_);
lean_ctor_set(v___x_141_, 3, v___x_770_);
lean_ctor_set(v___x_141_, 2, v_v_764_);
lean_ctor_set(v___x_141_, 1, v_k_763_);
lean_ctor_set(v___x_141_, 0, v___x_768_);
v___x_774_ = v___x_141_;
goto v_reusejp_773_;
}
else
{
lean_object* v_reuseFailAlloc_775_; 
v_reuseFailAlloc_775_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_775_, 0, v___x_768_);
lean_ctor_set(v_reuseFailAlloc_775_, 1, v_k_763_);
lean_ctor_set(v_reuseFailAlloc_775_, 2, v_v_764_);
lean_ctor_set(v_reuseFailAlloc_775_, 3, v___x_770_);
lean_ctor_set(v_reuseFailAlloc_775_, 4, v___x_772_);
v___x_774_ = v_reuseFailAlloc_775_;
goto v_reusejp_773_;
}
v_reusejp_773_:
{
return v___x_774_;
}
}
}
}
}
}
else
{
lean_object* v___x_786_; lean_object* v___x_788_; 
v___x_786_ = lean_unsigned_to_nat(2u);
if (v_isShared_142_ == 0)
{
lean_ctor_set(v___x_141_, 4, v_r_757_);
lean_ctor_set(v___x_141_, 0, v___x_786_);
v___x_788_ = v___x_141_;
goto v_reusejp_787_;
}
else
{
lean_object* v_reuseFailAlloc_789_; 
v_reuseFailAlloc_789_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_789_, 0, v___x_786_);
lean_ctor_set(v_reuseFailAlloc_789_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_789_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_789_, 3, v_l_138_);
lean_ctor_set(v_reuseFailAlloc_789_, 4, v_r_757_);
v___x_788_ = v_reuseFailAlloc_789_;
goto v_reusejp_787_;
}
v_reusejp_787_:
{
return v___x_788_;
}
}
}
}
else
{
lean_object* v___x_791_; 
if (v_isShared_142_ == 0)
{
lean_ctor_set(v___x_141_, 4, v_l_138_);
lean_ctor_set(v___x_141_, 0, v___x_630_);
v___x_791_ = v___x_141_;
goto v_reusejp_790_;
}
else
{
lean_object* v_reuseFailAlloc_792_; 
v_reuseFailAlloc_792_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_792_, 0, v___x_630_);
lean_ctor_set(v_reuseFailAlloc_792_, 1, v_k_136_);
lean_ctor_set(v_reuseFailAlloc_792_, 2, v_v_137_);
lean_ctor_set(v_reuseFailAlloc_792_, 3, v_l_138_);
lean_ctor_set(v_reuseFailAlloc_792_, 4, v_l_138_);
v___x_791_ = v_reuseFailAlloc_792_;
goto v_reusejp_790_;
}
v_reusejp_790_:
{
return v___x_791_;
}
}
}
}
}
}
}
else
{
return v_t_135_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Environment_importGraph_spec__0___redArg___boxed(lean_object* v_k_795_, lean_object* v_t_796_){
_start:
{
lean_object* v_res_797_; 
v_res_797_ = lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Environment_importGraph_spec__0___redArg(v_k_795_, v_t_796_);
lean_dec(v_k_795_);
return v_res_797_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_importGraph(lean_object* v_env_798_){
_start:
{
lean_object* v___y_800_; lean_object* v___x_803_; lean_object* v_mainModule_804_; lean_object* v_imports_805_; lean_object* v___x_806_; lean_object* v___x_807_; lean_object* v___x_808_; lean_object* v___x_809_; uint8_t v___x_810_; 
v___x_803_ = l_Lean_Environment_header(v_env_798_);
v_mainModule_804_ = lean_ctor_get(v___x_803_, 0);
lean_inc(v_mainModule_804_);
lean_dec_ref(v___x_803_);
v_imports_805_ = lp_importGraph_Lean_Environment_importsOf(v_env_798_, v_mainModule_804_);
v___x_806_ = lean_box(1);
lean_inc_ref(v_imports_805_);
v___x_807_ = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(v_mainModule_804_, v_imports_805_, v___x_806_);
v___x_808_ = lean_unsigned_to_nat(0u);
v___x_809_ = lean_array_get_size(v_imports_805_);
v___x_810_ = lean_nat_dec_lt(v___x_808_, v___x_809_);
if (v___x_810_ == 0)
{
lean_dec_ref(v_imports_805_);
v___y_800_ = v___x_807_;
goto v___jp_799_;
}
else
{
uint8_t v___x_811_; 
v___x_811_ = lean_nat_dec_le(v___x_809_, v___x_809_);
if (v___x_811_ == 0)
{
if (v___x_810_ == 0)
{
lean_dec_ref(v_imports_805_);
v___y_800_ = v___x_807_;
goto v___jp_799_;
}
else
{
size_t v___x_812_; size_t v___x_813_; lean_object* v___x_814_; 
v___x_812_ = ((size_t)0ULL);
v___x_813_ = lean_usize_of_nat(v___x_809_);
v___x_814_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importGraph_spec__1(v_env_798_, v_imports_805_, v___x_812_, v___x_813_, v___x_807_);
lean_dec_ref(v_imports_805_);
v___y_800_ = v___x_814_;
goto v___jp_799_;
}
}
else
{
size_t v___x_815_; size_t v___x_816_; lean_object* v___x_817_; 
v___x_815_ = ((size_t)0ULL);
v___x_816_ = lean_usize_of_nat(v___x_809_);
v___x_817_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_Environment_importGraph_spec__1(v_env_798_, v_imports_805_, v___x_815_, v___x_816_, v___x_807_);
lean_dec_ref(v_imports_805_);
v___y_800_ = v___x_817_;
goto v___jp_799_;
}
}
v___jp_799_:
{
lean_object* v___x_801_; lean_object* v___x_802_; 
v___x_801_ = lean_box(0);
v___x_802_ = lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Environment_importGraph_spec__0___redArg(v___x_801_, v___y_800_);
return v___x_802_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_importGraph___boxed(lean_object* v_env_818_){
_start:
{
lean_object* v_res_819_; 
v_res_819_ = lp_importGraph_Lean_Environment_importGraph(v_env_818_);
lean_dec_ref(v_env_818_);
return v_res_819_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Environment_importGraph_spec__0(lean_object* v_00_u03b2_820_, lean_object* v_k_821_, lean_object* v_t_822_, lean_object* v_h_823_){
_start:
{
lean_object* v___x_824_; 
v___x_824_ = lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Environment_importGraph_spec__0___redArg(v_k_821_, v_t_822_);
return v___x_824_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Environment_importGraph_spec__0___boxed(lean_object* v_00_u03b2_825_, lean_object* v_k_826_, lean_object* v_t_827_, lean_object* v_h_828_){
_start:
{
lean_object* v_res_829_; 
v_res_829_ = lp_importGraph_Std_DTreeMap_Internal_Impl_erase___at___00Lean_Environment_importGraph_spec__0(v_00_u03b2_825_, v_k_826_, v_t_827_, v_h_828_);
lean_dec(v_k_826_);
return v_res_829_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Environment(uint8_t builtin);
lean_object* runtime_initialize_Lean_Data_NameMap_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_importGraph_ImportGraph_Imports_ImportGraph(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Environment(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Data_NameMap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_importGraph_ImportGraph_Imports_ImportGraph(uint8_t builtin) {
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
lean_object* initialize_Lean_Environment(uint8_t builtin);
lean_object* initialize_Lean_Data_NameMap_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_importGraph_ImportGraph_Imports_ImportGraph(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Environment(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Data_NameMap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Imports_ImportGraph(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_importGraph_ImportGraph_Imports_ImportGraph(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_importGraph_ImportGraph_Imports_ImportGraph(builtin);
}
#ifdef __cplusplus
}
#endif
