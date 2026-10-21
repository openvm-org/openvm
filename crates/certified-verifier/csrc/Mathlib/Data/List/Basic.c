// Lean compiler output
// Module: Mathlib.Data.List.Basic
// Imports: public import Init public meta import Init public import Mathlib.Basic.Unique public import Mathlib.Data.List.Defs public import Mathlib.Data.List.Monad public import Mathlib.Tactic.Common public import Batteries.Data.List.Lemmas public import Mathlib.Data.Subtype public import Mathlib.Tactic.Attr.Core
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
uint8_t l_List_decidableBAll___redArg(lean_object*, lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_List_insert(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_uniqueOfIsEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instSingletonList___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_List_instSingletonList___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_instSingletonList___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_instSingletonList___closed__0 = (const lean_object*)&lp_mathlib_List_instSingletonList___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_instSingletonList(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instInsertOfDecidableEq__mathlib___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instInsertOfDecidableEq__mathlib(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_getLastI_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_getLastI_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_getLast_x3f_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_getLast_x3f_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_filterMap_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_filterMap_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_filter_match__1_splitter___redArg(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_filter_match__1_splitter___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_filter_match__1_splitter(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_filter_match__1_splitter___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidablePredForall___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidablePredForall___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidablePredForall(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidablePredForall___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_uniqueOfIsEmpty(lean_object* v_00_u03b1_1_, lean_object* v_inst_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_box(0);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instSingletonList___lam__0(lean_object* v_x_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = lean_box(0);
v___x_6_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_6_, 0, v_x_4_);
lean_ctor_set(v___x_6_, 1, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instSingletonList(lean_object* v_00_u03b1_8_){
_start:
{
lean_object* v___f_9_; 
v___f_9_ = ((lean_object*)(lp_mathlib_List_instSingletonList___closed__0));
return v___f_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instInsertOfDecidableEq__mathlib___redArg(lean_object* v_inst_10_){
_start:
{
lean_object* v___f_11_; lean_object* v___x_12_; 
v___f_11_ = lean_alloc_closure((void*)(l_instBEqOfDecidableEq___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_11_, 0, v_inst_10_);
v___x_12_ = lean_alloc_closure((void*)(l_List_insert), 4, 2);
lean_closure_set(v___x_12_, 0, lean_box(0));
lean_closure_set(v___x_12_, 1, v___f_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instInsertOfDecidableEq__mathlib(lean_object* v_00_u03b1_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_List_instInsertOfDecidableEq__mathlib___redArg(v_inst_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_getLastI_match__1_splitter___redArg(lean_object* v_x_16_, lean_object* v_h__1_17_, lean_object* v_h__2_18_, lean_object* v_h__3_19_, lean_object* v_h__4_20_){
_start:
{
if (lean_obj_tag(v_x_16_) == 0)
{
lean_object* v___x_21_; lean_object* v___x_22_; 
lean_dec(v_h__4_20_);
lean_dec(v_h__3_19_);
lean_dec(v_h__2_18_);
v___x_21_ = lean_box(0);
v___x_22_ = lean_apply_1(v_h__1_17_, v___x_21_);
return v___x_22_;
}
else
{
lean_object* v_tail_23_; 
lean_dec(v_h__1_17_);
v_tail_23_ = lean_ctor_get(v_x_16_, 1);
if (lean_obj_tag(v_tail_23_) == 0)
{
lean_object* v_head_24_; lean_object* v___x_25_; 
lean_dec(v_h__4_20_);
lean_dec(v_h__3_19_);
v_head_24_ = lean_ctor_get(v_x_16_, 0);
lean_inc(v_head_24_);
lean_dec_ref_known(v_x_16_, 2);
v___x_25_ = lean_apply_1(v_h__2_18_, v_head_24_);
return v___x_25_;
}
else
{
lean_object* v_tail_26_; 
lean_inc_ref(v_tail_23_);
lean_dec(v_h__2_18_);
v_tail_26_ = lean_ctor_get(v_tail_23_, 1);
if (lean_obj_tag(v_tail_26_) == 0)
{
lean_object* v_head_27_; lean_object* v_head_28_; lean_object* v___x_29_; 
lean_dec(v_h__4_20_);
v_head_27_ = lean_ctor_get(v_x_16_, 0);
lean_inc(v_head_27_);
lean_dec_ref_known(v_x_16_, 2);
v_head_28_ = lean_ctor_get(v_tail_23_, 0);
lean_inc(v_head_28_);
lean_dec_ref_known(v_tail_23_, 2);
v___x_29_ = lean_apply_2(v_h__3_19_, v_head_27_, v_head_28_);
return v___x_29_;
}
else
{
lean_object* v_head_30_; lean_object* v_head_31_; lean_object* v___x_32_; 
lean_inc(v_tail_26_);
lean_dec(v_h__3_19_);
v_head_30_ = lean_ctor_get(v_x_16_, 0);
lean_inc(v_head_30_);
lean_dec_ref_known(v_x_16_, 2);
v_head_31_ = lean_ctor_get(v_tail_23_, 0);
lean_inc(v_head_31_);
lean_dec_ref_known(v_tail_23_, 2);
v___x_32_ = lean_apply_4(v_h__4_20_, v_head_30_, v_head_31_, v_tail_26_, lean_box(0));
return v___x_32_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_getLastI_match__1_splitter(lean_object* v_00_u03b1_33_, lean_object* v_motive_34_, lean_object* v_x_35_, lean_object* v_h__1_36_, lean_object* v_h__2_37_, lean_object* v_h__3_38_, lean_object* v_h__4_39_){
_start:
{
if (lean_obj_tag(v_x_35_) == 0)
{
lean_object* v___x_40_; lean_object* v___x_41_; 
lean_dec(v_h__4_39_);
lean_dec(v_h__3_38_);
lean_dec(v_h__2_37_);
v___x_40_ = lean_box(0);
v___x_41_ = lean_apply_1(v_h__1_36_, v___x_40_);
return v___x_41_;
}
else
{
lean_object* v_tail_42_; 
lean_dec(v_h__1_36_);
v_tail_42_ = lean_ctor_get(v_x_35_, 1);
if (lean_obj_tag(v_tail_42_) == 0)
{
lean_object* v_head_43_; lean_object* v___x_44_; 
lean_dec(v_h__4_39_);
lean_dec(v_h__3_38_);
v_head_43_ = lean_ctor_get(v_x_35_, 0);
lean_inc(v_head_43_);
lean_dec_ref_known(v_x_35_, 2);
v___x_44_ = lean_apply_1(v_h__2_37_, v_head_43_);
return v___x_44_;
}
else
{
lean_object* v_tail_45_; 
lean_inc_ref(v_tail_42_);
lean_dec(v_h__2_37_);
v_tail_45_ = lean_ctor_get(v_tail_42_, 1);
if (lean_obj_tag(v_tail_45_) == 0)
{
lean_object* v_head_46_; lean_object* v_head_47_; lean_object* v___x_48_; 
lean_dec(v_h__4_39_);
v_head_46_ = lean_ctor_get(v_x_35_, 0);
lean_inc(v_head_46_);
lean_dec_ref_known(v_x_35_, 2);
v_head_47_ = lean_ctor_get(v_tail_42_, 0);
lean_inc(v_head_47_);
lean_dec_ref_known(v_tail_42_, 2);
v___x_48_ = lean_apply_2(v_h__3_38_, v_head_46_, v_head_47_);
return v___x_48_;
}
else
{
lean_object* v_head_49_; lean_object* v_head_50_; lean_object* v___x_51_; 
lean_inc(v_tail_45_);
lean_dec(v_h__3_38_);
v_head_49_ = lean_ctor_get(v_x_35_, 0);
lean_inc(v_head_49_);
lean_dec_ref_known(v_x_35_, 2);
v_head_50_ = lean_ctor_get(v_tail_42_, 0);
lean_inc(v_head_50_);
lean_dec_ref_known(v_tail_42_, 2);
v___x_51_ = lean_apply_4(v_h__4_39_, v_head_49_, v_head_50_, v_tail_45_, lean_box(0));
return v___x_51_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_getLast_x3f_match__1_splitter___redArg(lean_object* v_x_52_, lean_object* v_h__1_53_, lean_object* v_h__2_54_){
_start:
{
if (lean_obj_tag(v_x_52_) == 0)
{
lean_object* v___x_55_; lean_object* v___x_56_; 
lean_dec(v_h__2_54_);
v___x_55_ = lean_box(0);
v___x_56_ = lean_apply_1(v_h__1_53_, v___x_55_);
return v___x_56_;
}
else
{
lean_object* v_head_57_; lean_object* v_tail_58_; lean_object* v___x_59_; 
lean_dec(v_h__1_53_);
v_head_57_ = lean_ctor_get(v_x_52_, 0);
lean_inc(v_head_57_);
v_tail_58_ = lean_ctor_get(v_x_52_, 1);
lean_inc(v_tail_58_);
lean_dec_ref_known(v_x_52_, 2);
v___x_59_ = lean_apply_2(v_h__2_54_, v_head_57_, v_tail_58_);
return v___x_59_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_getLast_x3f_match__1_splitter(lean_object* v_00_u03b1_60_, lean_object* v_motive_61_, lean_object* v_x_62_, lean_object* v_h__1_63_, lean_object* v_h__2_64_){
_start:
{
if (lean_obj_tag(v_x_62_) == 0)
{
lean_object* v___x_65_; lean_object* v___x_66_; 
lean_dec(v_h__2_64_);
v___x_65_ = lean_box(0);
v___x_66_ = lean_apply_1(v_h__1_63_, v___x_65_);
return v___x_66_;
}
else
{
lean_object* v_head_67_; lean_object* v_tail_68_; lean_object* v___x_69_; 
lean_dec(v_h__1_63_);
v_head_67_ = lean_ctor_get(v_x_62_, 0);
lean_inc(v_head_67_);
v_tail_68_ = lean_ctor_get(v_x_62_, 1);
lean_inc(v_tail_68_);
lean_dec_ref_known(v_x_62_, 2);
v___x_69_ = lean_apply_2(v_h__2_64_, v_head_67_, v_tail_68_);
return v___x_69_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_filterMap_match__1_splitter___redArg(lean_object* v_x_70_, lean_object* v_h__1_71_, lean_object* v_h__2_72_){
_start:
{
if (lean_obj_tag(v_x_70_) == 0)
{
lean_object* v___x_73_; lean_object* v___x_74_; 
lean_dec(v_h__2_72_);
v___x_73_ = lean_box(0);
v___x_74_ = lean_apply_1(v_h__1_71_, v___x_73_);
return v___x_74_;
}
else
{
lean_object* v_val_75_; lean_object* v___x_76_; 
lean_dec(v_h__1_71_);
v_val_75_ = lean_ctor_get(v_x_70_, 0);
lean_inc(v_val_75_);
lean_dec_ref_known(v_x_70_, 1);
v___x_76_ = lean_apply_1(v_h__2_72_, v_val_75_);
return v___x_76_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_filterMap_match__1_splitter(lean_object* v_00_u03b2_77_, lean_object* v_motive_78_, lean_object* v_x_79_, lean_object* v_h__1_80_, lean_object* v_h__2_81_){
_start:
{
if (lean_obj_tag(v_x_79_) == 0)
{
lean_object* v___x_82_; lean_object* v___x_83_; 
lean_dec(v_h__2_81_);
v___x_82_ = lean_box(0);
v___x_83_ = lean_apply_1(v_h__1_80_, v___x_82_);
return v___x_83_;
}
else
{
lean_object* v_val_84_; lean_object* v___x_85_; 
lean_dec(v_h__1_80_);
v_val_84_ = lean_ctor_get(v_x_79_, 0);
lean_inc(v_val_84_);
lean_dec_ref_known(v_x_79_, 1);
v___x_85_ = lean_apply_1(v_h__2_81_, v_val_84_);
return v___x_85_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_filter_match__1_splitter___redArg(uint8_t v_x_86_, lean_object* v_h__1_87_, lean_object* v_h__2_88_){
_start:
{
if (v_x_86_ == 0)
{
lean_object* v___x_89_; lean_object* v___x_90_; 
lean_dec(v_h__1_87_);
v___x_89_ = lean_box(0);
v___x_90_ = lean_apply_1(v_h__2_88_, v___x_89_);
return v___x_90_;
}
else
{
lean_object* v___x_91_; lean_object* v___x_92_; 
lean_dec(v_h__2_88_);
v___x_91_ = lean_box(0);
v___x_92_ = lean_apply_1(v_h__1_87_, v___x_91_);
return v___x_92_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_filter_match__1_splitter___redArg___boxed(lean_object* v_x_93_, lean_object* v_h__1_94_, lean_object* v_h__2_95_){
_start:
{
uint8_t v_x_24__boxed_96_; lean_object* v_res_97_; 
v_x_24__boxed_96_ = lean_unbox(v_x_93_);
v_res_97_ = lp_mathlib___private_Mathlib_Data_List_Basic_0__List_filter_match__1_splitter___redArg(v_x_24__boxed_96_, v_h__1_94_, v_h__2_95_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_filter_match__1_splitter(lean_object* v_motive_98_, uint8_t v_x_99_, lean_object* v_h__1_100_, lean_object* v_h__2_101_){
_start:
{
if (v_x_99_ == 0)
{
lean_object* v___x_102_; lean_object* v___x_103_; 
lean_dec(v_h__1_100_);
v___x_102_ = lean_box(0);
v___x_103_ = lean_apply_1(v_h__2_101_, v___x_102_);
return v___x_103_;
}
else
{
lean_object* v___x_104_; lean_object* v___x_105_; 
lean_dec(v_h__2_101_);
v___x_104_ = lean_box(0);
v___x_105_ = lean_apply_1(v_h__1_100_, v___x_104_);
return v___x_105_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_List_Basic_0__List_filter_match__1_splitter___boxed(lean_object* v_motive_106_, lean_object* v_x_107_, lean_object* v_h__1_108_, lean_object* v_h__2_109_){
_start:
{
uint8_t v_x_35__boxed_110_; lean_object* v_res_111_; 
v_x_35__boxed_110_ = lean_unbox(v_x_107_);
v_res_111_ = lp_mathlib___private_Mathlib_Data_List_Basic_0__List_filter_match__1_splitter(v_motive_106_, v_x_35__boxed_110_, v_h__1_108_, v_h__2_109_);
return v_res_111_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidablePredForall___redArg(lean_object* v_inst_112_, lean_object* v_x_113_){
_start:
{
uint8_t v___x_114_; 
v___x_114_ = l_List_decidableBAll___redArg(v_inst_112_, v_x_113_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidablePredForall___redArg___boxed(lean_object* v_inst_115_, lean_object* v_x_116_){
_start:
{
uint8_t v_res_117_; lean_object* v_r_118_; 
v_res_117_ = lp_mathlib_List_instDecidablePredForall___redArg(v_inst_115_, v_x_116_);
v_r_118_ = lean_box(v_res_117_);
return v_r_118_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instDecidablePredForall(lean_object* v_00_u03b1_119_, lean_object* v_p_120_, lean_object* v_inst_121_, lean_object* v_x_122_){
_start:
{
uint8_t v___x_123_; 
v___x_123_ = l_List_decidableBAll___redArg(v_inst_121_, v_x_122_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instDecidablePredForall___boxed(lean_object* v_00_u03b1_124_, lean_object* v_p_125_, lean_object* v_inst_126_, lean_object* v_x_127_){
_start:
{
uint8_t v_res_128_; lean_object* v_r_129_; 
v_res_128_ = lp_mathlib_List_instDecidablePredForall(v_00_u03b1_124_, v_p_125_, v_inst_126_, v_x_127_);
v_r_129_ = lean_box(v_res_128_);
return v_r_129_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Unique(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Monad(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_List_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Subtype(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Unique(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Monad(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_List_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Subtype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Basic_Unique(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Monad(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Subtype(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Unique(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Monad(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Subtype(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
