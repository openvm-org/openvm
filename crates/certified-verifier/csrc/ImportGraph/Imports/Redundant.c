// Lean compiler output
// Module: ImportGraph.Imports.Redundant
// Imports: public import Init public meta import Init public import Lean.Environment public import Lean.Data.NameMap.Basic public import Lean.CoreM import ImportGraph.Imports.ImportGraph
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
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_NameSet_insert(lean_object*, lean_object*);
extern lean_object* l_Lean_NameSet_empty;
lean_object* lean_st_ref_get(lean_object*);
lean_object* lp_importGraph_Lean_Environment_importsOf(lean_object*, lean_object*);
lean_object* lp_importGraph_Lean_Environment_importGraph(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
uint8_t l_Lean_NameSet_contains(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Environment_header(lean_object*);
LEAN_EXPORT uint8_t lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__0_spec__0(lean_object*, lean_object*, size_t, size_t);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_importGraph_Array_contains___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Array_contains___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_importGraph___private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_importGraph___private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit___closed__0 = (const lean_object*)&lp_importGraph___private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit___closed__0_value;
LEAN_EXPORT lean_object* lp_importGraph___private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__2(lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph___private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_importGraph_Lean_Environment_findRedundantImports___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_importGraph_Lean_Environment_findRedundantImports___closed__0;
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_findRedundantImports(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_findRedundantImports___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_redundantImports___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_redundantImports___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_redundantImports(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_redundantImports___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__0_spec__0(lean_object* v_a_1_, lean_object* v_as_2_, size_t v_i_3_, size_t v_stop_4_){
_start:
{
uint8_t v___x_5_; 
v___x_5_ = lean_usize_dec_eq(v_i_3_, v_stop_4_);
if (v___x_5_ == 0)
{
lean_object* v___x_6_; uint8_t v___x_7_; 
v___x_6_ = lean_array_uget_borrowed(v_as_2_, v_i_3_);
v___x_7_ = lean_name_eq(v_a_1_, v___x_6_);
if (v___x_7_ == 0)
{
size_t v___x_8_; size_t v___x_9_; 
v___x_8_ = ((size_t)1ULL);
v___x_9_ = lean_usize_add(v_i_3_, v___x_8_);
v_i_3_ = v___x_9_;
goto _start;
}
else
{
return v___x_7_;
}
}
else
{
uint8_t v___x_11_; 
v___x_11_ = 0;
return v___x_11_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__0_spec__0___boxed(lean_object* v_a_12_, lean_object* v_as_13_, lean_object* v_i_14_, lean_object* v_stop_15_){
_start:
{
size_t v_i_boxed_16_; size_t v_stop_boxed_17_; uint8_t v_res_18_; lean_object* v_r_19_; 
v_i_boxed_16_ = lean_unbox_usize(v_i_14_);
lean_dec(v_i_14_);
v_stop_boxed_17_ = lean_unbox_usize(v_stop_15_);
lean_dec(v_stop_15_);
v_res_18_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__0_spec__0(v_a_12_, v_as_13_, v_i_boxed_16_, v_stop_boxed_17_);
lean_dec_ref(v_as_13_);
lean_dec(v_a_12_);
v_r_19_ = lean_box(v_res_18_);
return v_r_19_;
}
}
LEAN_EXPORT uint8_t lp_importGraph_Array_contains___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__0(lean_object* v_as_20_, lean_object* v_a_21_){
_start:
{
lean_object* v___x_22_; lean_object* v___x_23_; uint8_t v___x_24_; 
v___x_22_ = lean_unsigned_to_nat(0u);
v___x_23_ = lean_array_get_size(v_as_20_);
v___x_24_ = lean_nat_dec_lt(v___x_22_, v___x_23_);
if (v___x_24_ == 0)
{
return v___x_24_;
}
else
{
if (v___x_24_ == 0)
{
return v___x_24_;
}
else
{
size_t v___x_25_; size_t v___x_26_; uint8_t v___x_27_; 
v___x_25_ = ((size_t)0ULL);
v___x_26_ = lean_usize_of_nat(v___x_23_);
v___x_27_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any___at___00Array_contains___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__0_spec__0(v_a_21_, v_as_20_, v___x_25_, v___x_26_);
return v___x_27_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Array_contains___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__0___boxed(lean_object* v_as_28_, lean_object* v_a_29_){
_start:
{
uint8_t v_res_30_; lean_object* v_r_31_; 
v_res_30_ = lp_importGraph_Array_contains___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__0(v_as_28_, v_a_29_);
lean_dec(v_a_29_);
lean_dec_ref(v_as_28_);
v_r_31_ = lean_box(v_res_30_);
return v_r_31_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__1(lean_object* v_targets_32_, lean_object* v_as_33_, size_t v_i_34_, size_t v_stop_35_, lean_object* v_b_36_){
_start:
{
lean_object* v___y_38_; uint8_t v___x_42_; 
v___x_42_ = lean_usize_dec_eq(v_i_34_, v_stop_35_);
if (v___x_42_ == 0)
{
lean_object* v___x_43_; uint8_t v___x_44_; 
v___x_43_ = lean_array_uget_borrowed(v_as_33_, v_i_34_);
v___x_44_ = lp_importGraph_Array_contains___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__0(v_targets_32_, v___x_43_);
if (v___x_44_ == 0)
{
v___y_38_ = v_b_36_;
goto v___jp_37_;
}
else
{
lean_object* v___x_45_; 
lean_inc(v___x_43_);
v___x_45_ = l_Lean_NameSet_insert(v_b_36_, v___x_43_);
v___y_38_ = v___x_45_;
goto v___jp_37_;
}
}
else
{
return v_b_36_;
}
v___jp_37_:
{
size_t v___x_39_; size_t v___x_40_; 
v___x_39_ = ((size_t)1ULL);
v___x_40_ = lean_usize_add(v_i_34_, v___x_39_);
v_i_34_ = v___x_40_;
v_b_36_ = v___y_38_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__1___boxed(lean_object* v_targets_46_, lean_object* v_as_47_, lean_object* v_i_48_, lean_object* v_stop_49_, lean_object* v_b_50_){
_start:
{
size_t v_i_boxed_51_; size_t v_stop_boxed_52_; lean_object* v_res_53_; 
v_i_boxed_51_ = lean_unbox_usize(v_i_48_);
lean_dec(v_i_48_);
v_stop_boxed_52_ = lean_unbox_usize(v_stop_49_);
lean_dec(v_stop_49_);
v_res_53_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__1(v_targets_46_, v_as_47_, v_i_boxed_51_, v_stop_boxed_52_, v_b_50_);
lean_dec_ref(v_as_47_);
lean_dec_ref(v_targets_46_);
return v_res_53_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit(lean_object* v_00_u0393_56_, lean_object* v_targets_57_, lean_object* v_visited_58_, lean_object* v_seen_59_, lean_object* v_n_60_){
_start:
{
lean_object* v___y_62_; lean_object* v_fst_63_; lean_object* v_snd_64_; lean_object* v___y_81_; lean_object* v___y_82_; lean_object* v___y_86_; uint8_t v___x_98_; 
v___x_98_ = l_Lean_NameSet_contains(v_visited_58_, v_n_60_);
if (v___x_98_ == 0)
{
lean_object* v___x_99_; 
v___x_99_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_00_u0393_56_, v_n_60_);
if (lean_obj_tag(v___x_99_) == 0)
{
lean_object* v___x_100_; 
v___x_100_ = ((lean_object*)(lp_importGraph___private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit___closed__0));
v___y_86_ = v___x_100_;
goto v___jp_85_;
}
else
{
lean_object* v_val_101_; 
v_val_101_ = lean_ctor_get(v___x_99_, 0);
lean_inc(v_val_101_);
lean_dec_ref_known(v___x_99_, 1);
v___y_86_ = v_val_101_;
goto v___jp_85_;
}
}
else
{
lean_object* v___x_102_; 
lean_dec(v_n_60_);
v___x_102_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_102_, 0, v_visited_58_);
lean_ctor_set(v___x_102_, 1, v_seen_59_);
return v___x_102_;
}
v___jp_61_:
{
lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; uint8_t v___x_68_; 
v___x_65_ = l_Lean_NameSet_insert(v_fst_63_, v_n_60_);
v___x_66_ = lean_unsigned_to_nat(0u);
v___x_67_ = lean_array_get_size(v___y_62_);
v___x_68_ = lean_nat_dec_lt(v___x_66_, v___x_67_);
if (v___x_68_ == 0)
{
lean_object* v___x_69_; 
lean_dec_ref(v___y_62_);
v___x_69_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_69_, 0, v___x_65_);
lean_ctor_set(v___x_69_, 1, v_snd_64_);
return v___x_69_;
}
else
{
uint8_t v___x_70_; 
v___x_70_ = lean_nat_dec_le(v___x_67_, v___x_67_);
if (v___x_70_ == 0)
{
if (v___x_68_ == 0)
{
lean_object* v___x_71_; 
lean_dec_ref(v___y_62_);
v___x_71_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_71_, 0, v___x_65_);
lean_ctor_set(v___x_71_, 1, v_snd_64_);
return v___x_71_;
}
else
{
size_t v___x_72_; size_t v___x_73_; lean_object* v___x_74_; lean_object* v___x_75_; 
v___x_72_ = ((size_t)0ULL);
v___x_73_ = lean_usize_of_nat(v___x_67_);
v___x_74_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__1(v_targets_57_, v___y_62_, v___x_72_, v___x_73_, v_snd_64_);
lean_dec_ref(v___y_62_);
v___x_75_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_75_, 0, v___x_65_);
lean_ctor_set(v___x_75_, 1, v___x_74_);
return v___x_75_;
}
}
else
{
size_t v___x_76_; size_t v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_76_ = ((size_t)0ULL);
v___x_77_ = lean_usize_of_nat(v___x_67_);
v___x_78_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__1(v_targets_57_, v___y_62_, v___x_76_, v___x_77_, v_snd_64_);
lean_dec_ref(v___y_62_);
v___x_79_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_79_, 0, v___x_65_);
lean_ctor_set(v___x_79_, 1, v___x_78_);
return v___x_79_;
}
}
}
v___jp_80_:
{
lean_object* v_fst_83_; lean_object* v_snd_84_; 
v_fst_83_ = lean_ctor_get(v___y_82_, 0);
lean_inc(v_fst_83_);
v_snd_84_ = lean_ctor_get(v___y_82_, 1);
lean_inc(v_snd_84_);
lean_dec_ref(v___y_82_);
v___y_62_ = v___y_81_;
v_fst_63_ = v_fst_83_;
v_snd_64_ = v_snd_84_;
goto v___jp_61_;
}
v___jp_85_:
{
lean_object* v___x_87_; lean_object* v___x_88_; uint8_t v___x_89_; 
v___x_87_ = lean_unsigned_to_nat(0u);
v___x_88_ = lean_array_get_size(v___y_86_);
v___x_89_ = lean_nat_dec_lt(v___x_87_, v___x_88_);
if (v___x_89_ == 0)
{
v___y_62_ = v___y_86_;
v_fst_63_ = v_visited_58_;
v_snd_64_ = v_seen_59_;
goto v___jp_61_;
}
else
{
lean_object* v___x_90_; uint8_t v___x_91_; 
lean_inc(v_seen_59_);
lean_inc(v_visited_58_);
v___x_90_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_90_, 0, v_visited_58_);
lean_ctor_set(v___x_90_, 1, v_seen_59_);
v___x_91_ = lean_nat_dec_le(v___x_88_, v___x_88_);
if (v___x_91_ == 0)
{
if (v___x_89_ == 0)
{
lean_dec_ref_known(v___x_90_, 2);
v___y_62_ = v___y_86_;
v_fst_63_ = v_visited_58_;
v_snd_64_ = v_seen_59_;
goto v___jp_61_;
}
else
{
size_t v___x_92_; size_t v___x_93_; lean_object* v___x_94_; 
lean_dec(v_seen_59_);
lean_dec(v_visited_58_);
v___x_92_ = ((size_t)0ULL);
v___x_93_ = lean_usize_of_nat(v___x_88_);
v___x_94_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__2(v_00_u0393_56_, v_targets_57_, v___y_86_, v___x_92_, v___x_93_, v___x_90_);
v___y_81_ = v___y_86_;
v___y_82_ = v___x_94_;
goto v___jp_80_;
}
}
else
{
size_t v___x_95_; size_t v___x_96_; lean_object* v___x_97_; 
lean_dec(v_seen_59_);
lean_dec(v_visited_58_);
v___x_95_ = ((size_t)0ULL);
v___x_96_ = lean_usize_of_nat(v___x_88_);
v___x_97_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__2(v_00_u0393_56_, v_targets_57_, v___y_86_, v___x_95_, v___x_96_, v___x_90_);
v___y_81_ = v___y_86_;
v___y_82_ = v___x_97_;
goto v___jp_80_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__2(lean_object* v_00_u0393_103_, lean_object* v_targets_104_, lean_object* v_as_105_, size_t v_i_106_, size_t v_stop_107_, lean_object* v_b_108_){
_start:
{
uint8_t v___x_109_; 
v___x_109_ = lean_usize_dec_eq(v_i_106_, v_stop_107_);
if (v___x_109_ == 0)
{
lean_object* v_fst_110_; lean_object* v_snd_111_; lean_object* v___x_112_; lean_object* v___x_113_; size_t v___x_114_; size_t v___x_115_; 
v_fst_110_ = lean_ctor_get(v_b_108_, 0);
lean_inc(v_fst_110_);
v_snd_111_ = lean_ctor_get(v_b_108_, 1);
lean_inc(v_snd_111_);
lean_dec_ref(v_b_108_);
v___x_112_ = lean_array_uget_borrowed(v_as_105_, v_i_106_);
lean_inc(v___x_112_);
v___x_113_ = lp_importGraph___private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit(v_00_u0393_103_, v_targets_104_, v_fst_110_, v_snd_111_, v___x_112_);
v___x_114_ = ((size_t)1ULL);
v___x_115_ = lean_usize_add(v_i_106_, v___x_114_);
v_i_106_ = v___x_115_;
v_b_108_ = v___x_113_;
goto _start;
}
else
{
return v_b_108_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__2___boxed(lean_object* v_00_u0393_117_, lean_object* v_targets_118_, lean_object* v_as_119_, lean_object* v_i_120_, lean_object* v_stop_121_, lean_object* v_b_122_){
_start:
{
size_t v_i_boxed_123_; size_t v_stop_boxed_124_; lean_object* v_res_125_; 
v_i_boxed_123_ = lean_unbox_usize(v_i_120_);
lean_dec(v_i_120_);
v_stop_boxed_124_ = lean_unbox_usize(v_stop_121_);
lean_dec(v_stop_121_);
v_res_125_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__2(v_00_u0393_117_, v_targets_118_, v_as_119_, v_i_boxed_123_, v_stop_boxed_124_, v_b_122_);
lean_dec_ref(v_as_119_);
lean_dec_ref(v_targets_118_);
lean_dec(v_00_u0393_117_);
return v_res_125_;
}
}
LEAN_EXPORT lean_object* lp_importGraph___private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit___boxed(lean_object* v_00_u0393_126_, lean_object* v_targets_127_, lean_object* v_visited_128_, lean_object* v_seen_129_, lean_object* v_n_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_importGraph___private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit(v_00_u0393_126_, v_targets_127_, v_visited_128_, v_seen_129_, v_n_130_);
lean_dec_ref(v_targets_127_);
lean_dec(v_00_u0393_126_);
return v_res_131_;
}
}
static lean_object* _init_lp_importGraph_Lean_Environment_findRedundantImports___closed__0(void){
_start:
{
lean_object* v___x_132_; lean_object* v___x_133_; 
v___x_132_ = l_Lean_NameSet_empty;
v___x_133_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_133_, 0, v___x_132_);
lean_ctor_set(v___x_133_, 1, v___x_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_findRedundantImports(lean_object* v_env_134_, lean_object* v_imports_135_){
_start:
{
lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; uint8_t v___x_140_; 
v___x_136_ = l_Lean_NameSet_empty;
v___x_137_ = lean_obj_once(&lp_importGraph_Lean_Environment_findRedundantImports___closed__0, &lp_importGraph_Lean_Environment_findRedundantImports___closed__0_once, _init_lp_importGraph_Lean_Environment_findRedundantImports___closed__0);
v___x_138_ = lean_unsigned_to_nat(0u);
v___x_139_ = lean_array_get_size(v_imports_135_);
v___x_140_ = lean_nat_dec_lt(v___x_138_, v___x_139_);
if (v___x_140_ == 0)
{
return v___x_136_;
}
else
{
lean_object* v___x_141_; uint8_t v___x_142_; 
v___x_141_ = lp_importGraph_Lean_Environment_importGraph(v_env_134_);
v___x_142_ = lean_nat_dec_le(v___x_139_, v___x_139_);
if (v___x_142_ == 0)
{
if (v___x_140_ == 0)
{
lean_dec(v___x_141_);
return v___x_136_;
}
else
{
size_t v___x_143_; size_t v___x_144_; lean_object* v___x_145_; lean_object* v_snd_146_; 
v___x_143_ = ((size_t)0ULL);
v___x_144_ = lean_usize_of_nat(v___x_139_);
v___x_145_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__2(v___x_141_, v_imports_135_, v_imports_135_, v___x_143_, v___x_144_, v___x_137_);
lean_dec(v___x_141_);
v_snd_146_ = lean_ctor_get(v___x_145_, 1);
lean_inc(v_snd_146_);
lean_dec_ref(v___x_145_);
return v_snd_146_;
}
}
else
{
size_t v___x_147_; size_t v___x_148_; lean_object* v___x_149_; lean_object* v_snd_150_; 
v___x_147_ = ((size_t)0ULL);
v___x_148_ = lean_usize_of_nat(v___x_139_);
v___x_149_ = lp_importGraph___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_ImportGraph_Imports_Redundant_0__Lean_Environment_findRedundantImports_visit_spec__2(v___x_141_, v_imports_135_, v_imports_135_, v___x_147_, v___x_148_, v___x_137_);
lean_dec(v___x_141_);
v_snd_150_ = lean_ctor_get(v___x_149_, 1);
lean_inc(v_snd_150_);
lean_dec_ref(v___x_149_);
return v_snd_150_;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_Lean_Environment_findRedundantImports___boxed(lean_object* v_env_151_, lean_object* v_imports_152_){
_start:
{
lean_object* v_res_153_; 
v_res_153_ = lp_importGraph_Lean_Environment_findRedundantImports(v_env_151_, v_imports_152_);
lean_dec_ref(v_imports_152_);
lean_dec_ref(v_env_151_);
return v_res_153_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_redundantImports___redArg(lean_object* v_n_x3f_154_, lean_object* v_a_155_){
_start:
{
lean_object* v___x_157_; lean_object* v_env_158_; lean_object* v___y_160_; 
v___x_157_ = lean_st_ref_get(v_a_155_);
v_env_158_ = lean_ctor_get(v___x_157_, 0);
lean_inc_ref(v_env_158_);
lean_dec(v___x_157_);
if (lean_obj_tag(v_n_x3f_154_) == 0)
{
lean_object* v___x_164_; lean_object* v_mainModule_165_; 
v___x_164_ = l_Lean_Environment_header(v_env_158_);
v_mainModule_165_ = lean_ctor_get(v___x_164_, 0);
lean_inc(v_mainModule_165_);
lean_dec_ref(v___x_164_);
v___y_160_ = v_mainModule_165_;
goto v___jp_159_;
}
else
{
lean_object* v_val_166_; 
v_val_166_ = lean_ctor_get(v_n_x3f_154_, 0);
lean_inc(v_val_166_);
lean_dec_ref_known(v_n_x3f_154_, 1);
v___y_160_ = v_val_166_;
goto v___jp_159_;
}
v___jp_159_:
{
lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; 
v___x_161_ = lp_importGraph_Lean_Environment_importsOf(v_env_158_, v___y_160_);
lean_dec(v___y_160_);
v___x_162_ = lp_importGraph_Lean_Environment_findRedundantImports(v_env_158_, v___x_161_);
lean_dec_ref(v___x_161_);
lean_dec_ref(v_env_158_);
v___x_163_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_163_, 0, v___x_162_);
return v___x_163_;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_redundantImports___redArg___boxed(lean_object* v_n_x3f_167_, lean_object* v_a_168_, lean_object* v_a_169_){
_start:
{
lean_object* v_res_170_; 
v_res_170_ = lp_importGraph_redundantImports___redArg(v_n_x3f_167_, v_a_168_);
lean_dec(v_a_168_);
return v_res_170_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_redundantImports(lean_object* v_n_x3f_171_, lean_object* v_a_172_, lean_object* v_a_173_){
_start:
{
lean_object* v___x_175_; 
v___x_175_ = lp_importGraph_redundantImports___redArg(v_n_x3f_171_, v_a_173_);
return v___x_175_;
}
}
LEAN_EXPORT lean_object* lp_importGraph_redundantImports___boxed(lean_object* v_n_x3f_176_, lean_object* v_a_177_, lean_object* v_a_178_, lean_object* v_a_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_importGraph_redundantImports(v_n_x3f_176_, v_a_177_, v_a_178_);
lean_dec(v_a_178_);
lean_dec_ref(v_a_177_);
return v_res_180_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Environment(uint8_t builtin);
lean_object* runtime_initialize_Lean_Data_NameMap_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_CoreM(uint8_t builtin);
lean_object* runtime_initialize_importGraph_ImportGraph_Imports_ImportGraph(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_importGraph_ImportGraph_Imports_Redundant(uint8_t builtin) {
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
res = runtime_initialize_Lean_CoreM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Imports_ImportGraph(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_importGraph_ImportGraph_Imports_Redundant(uint8_t builtin) {
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
lean_object* initialize_Lean_CoreM(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_Imports_ImportGraph(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_importGraph_ImportGraph_Imports_Redundant(uint8_t builtin) {
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
res = initialize_Lean_CoreM(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_Imports_ImportGraph(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Imports_Redundant(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_importGraph_ImportGraph_Imports_Redundant(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_importGraph_ImportGraph_Imports_Redundant(builtin);
}
#ifdef __cplusplus
}
#endif
