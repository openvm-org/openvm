// Lean compiler output
// Module: Mathlib.Data.List.MinMax
// Imports: public import Init public meta import Init public import Mathlib.Data.List.Basic public import Mathlib.Order.BoundedOrder.Lattice public import Mathlib.Data.List.Induction public import Mathlib.Order.MinMax public import Mathlib.Order.WithBot
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
lean_object* l_List_foldl___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_argAux___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_argAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_argmax___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_argmax___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_argmax___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_argmax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_argmax___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_argmin___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_argmin___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_argmin___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_argmin(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_argmin___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_List_maximum___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_List_maximum___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_maximum___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_maximum___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_maximum(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_maximum___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_minimum___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_minimum(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_minimum___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_maximum__of__length__pos___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_maximum__of__length__pos(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_minimum__of__length__pos___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_minimum__of__length__pos(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_argAux___redArg(lean_object* v_inst_1_, lean_object* v_a_2_, lean_object* v_b_3_){
_start:
{
if (lean_obj_tag(v_a_2_) == 0)
{
lean_object* v___x_4_; 
lean_dec_ref(v_inst_1_);
v___x_4_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_4_, 0, v_b_3_);
return v___x_4_;
}
else
{
lean_object* v_c_5_; lean_object* v___x_6_; uint8_t v___x_7_; 
v_c_5_ = lean_ctor_get(v_a_2_, 0);
lean_inc(v_c_5_);
lean_inc(v_b_3_);
v___x_6_ = lean_apply_2(v_inst_1_, v_b_3_, v_c_5_);
v___x_7_ = lean_unbox(v___x_6_);
if (v___x_7_ == 0)
{
lean_dec(v_b_3_);
return v_a_2_;
}
else
{
lean_object* v___x_9_; uint8_t v_isShared_10_; uint8_t v_isSharedCheck_14_; 
v_isSharedCheck_14_ = !lean_is_exclusive(v_a_2_);
if (v_isSharedCheck_14_ == 0)
{
lean_object* v_unused_15_; 
v_unused_15_ = lean_ctor_get(v_a_2_, 0);
lean_dec(v_unused_15_);
v___x_9_ = v_a_2_;
v_isShared_10_ = v_isSharedCheck_14_;
goto v_resetjp_8_;
}
else
{
lean_dec(v_a_2_);
v___x_9_ = lean_box(0);
v_isShared_10_ = v_isSharedCheck_14_;
goto v_resetjp_8_;
}
v_resetjp_8_:
{
lean_object* v___x_12_; 
if (v_isShared_10_ == 0)
{
lean_ctor_set(v___x_9_, 0, v_b_3_);
v___x_12_ = v___x_9_;
goto v_reusejp_11_;
}
else
{
lean_object* v_reuseFailAlloc_13_; 
v_reuseFailAlloc_13_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_13_, 0, v_b_3_);
v___x_12_ = v_reuseFailAlloc_13_;
goto v_reusejp_11_;
}
v_reusejp_11_:
{
return v___x_12_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_argAux(lean_object* v_00_u03b1_16_, lean_object* v_r_17_, lean_object* v_inst_18_, lean_object* v_a_19_, lean_object* v_b_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lp_mathlib_List_argAux___redArg(v_inst_18_, v_a_19_, v_b_20_);
return v___x_21_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_argmax___redArg___lam__0(lean_object* v_f_22_, lean_object* v_inst_23_, lean_object* v_a_24_, lean_object* v_b_25_){
_start:
{
lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; uint8_t v___x_29_; 
lean_inc(v_f_22_);
v___x_26_ = lean_apply_1(v_f_22_, v_b_25_);
v___x_27_ = lean_apply_1(v_f_22_, v_a_24_);
v___x_28_ = lean_apply_2(v_inst_23_, v___x_26_, v___x_27_);
v___x_29_ = lean_unbox(v___x_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_argmax___redArg___lam__0___boxed(lean_object* v_f_30_, lean_object* v_inst_31_, lean_object* v_a_32_, lean_object* v_b_33_){
_start:
{
uint8_t v_res_34_; lean_object* v_r_35_; 
v_res_34_ = lp_mathlib_List_argmax___redArg___lam__0(v_f_30_, v_inst_31_, v_a_32_, v_b_33_);
v_r_35_ = lean_box(v_res_34_);
return v_r_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_argmax___redArg(lean_object* v_inst_36_, lean_object* v_f_37_, lean_object* v_l_38_){
_start:
{
lean_object* v___f_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; 
v___f_39_ = lean_alloc_closure((void*)(lp_mathlib_List_argmax___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_39_, 0, v_f_37_);
lean_closure_set(v___f_39_, 1, v_inst_36_);
v___x_40_ = lean_alloc_closure((void*)(lp_mathlib_List_argAux), 5, 3);
lean_closure_set(v___x_40_, 0, lean_box(0));
lean_closure_set(v___x_40_, 1, lean_box(0));
lean_closure_set(v___x_40_, 2, v___f_39_);
v___x_41_ = lean_box(0);
v___x_42_ = l_List_foldl___redArg(v___x_40_, v___x_41_, v_l_38_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_argmax(lean_object* v_00_u03b1_43_, lean_object* v_00_u03b2_44_, lean_object* v_inst_45_, lean_object* v_inst_46_, lean_object* v_f_47_, lean_object* v_l_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_mathlib_List_argmax___redArg(v_inst_46_, v_f_47_, v_l_48_);
return v___x_49_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_argmax___boxed(lean_object* v_00_u03b1_50_, lean_object* v_00_u03b2_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_f_54_, lean_object* v_l_55_){
_start:
{
lean_object* v_res_56_; 
v_res_56_ = lp_mathlib_List_argmax(v_00_u03b1_50_, v_00_u03b2_51_, v_inst_52_, v_inst_53_, v_f_54_, v_l_55_);
lean_dec_ref(v_inst_52_);
return v_res_56_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_argmin___redArg___lam__0(lean_object* v_f_57_, lean_object* v_inst_58_, lean_object* v_a_59_, lean_object* v_b_60_){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; uint8_t v___x_64_; 
lean_inc(v_f_57_);
v___x_61_ = lean_apply_1(v_f_57_, v_a_59_);
v___x_62_ = lean_apply_1(v_f_57_, v_b_60_);
v___x_63_ = lean_apply_2(v_inst_58_, v___x_61_, v___x_62_);
v___x_64_ = lean_unbox(v___x_63_);
return v___x_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_argmin___redArg___lam__0___boxed(lean_object* v_f_65_, lean_object* v_inst_66_, lean_object* v_a_67_, lean_object* v_b_68_){
_start:
{
uint8_t v_res_69_; lean_object* v_r_70_; 
v_res_69_ = lp_mathlib_List_argmin___redArg___lam__0(v_f_65_, v_inst_66_, v_a_67_, v_b_68_);
v_r_70_ = lean_box(v_res_69_);
return v_r_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_argmin___redArg(lean_object* v_inst_71_, lean_object* v_f_72_, lean_object* v_l_73_){
_start:
{
lean_object* v___f_74_; lean_object* v___x_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
v___f_74_ = lean_alloc_closure((void*)(lp_mathlib_List_argmin___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_74_, 0, v_f_72_);
lean_closure_set(v___f_74_, 1, v_inst_71_);
v___x_75_ = lean_alloc_closure((void*)(lp_mathlib_List_argAux), 5, 3);
lean_closure_set(v___x_75_, 0, lean_box(0));
lean_closure_set(v___x_75_, 1, lean_box(0));
lean_closure_set(v___x_75_, 2, v___f_74_);
v___x_76_ = lean_box(0);
v___x_77_ = l_List_foldl___redArg(v___x_75_, v___x_76_, v_l_73_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_argmin(lean_object* v_00_u03b1_78_, lean_object* v_00_u03b2_79_, lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_f_82_, lean_object* v_l_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = lp_mathlib_List_argmin___redArg(v_inst_81_, v_f_82_, v_l_83_);
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_argmin___boxed(lean_object* v_00_u03b1_85_, lean_object* v_00_u03b2_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_f_89_, lean_object* v_l_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_mathlib_List_argmin(v_00_u03b1_85_, v_00_u03b2_86_, v_inst_87_, v_inst_88_, v_f_89_, v_l_90_);
lean_dec_ref(v_inst_87_);
return v_res_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_maximum___redArg(lean_object* v_inst_93_, lean_object* v_l_94_){
_start:
{
lean_object* v___x_95_; lean_object* v___x_96_; 
v___x_95_ = ((lean_object*)(lp_mathlib_List_maximum___redArg___closed__0));
v___x_96_ = lp_mathlib_List_argmax___redArg(v_inst_93_, v___x_95_, v_l_94_);
return v___x_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_maximum(lean_object* v_00_u03b1_97_, lean_object* v_inst_98_, lean_object* v_inst_99_, lean_object* v_l_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lp_mathlib_List_maximum___redArg(v_inst_99_, v_l_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_maximum___boxed(lean_object* v_00_u03b1_102_, lean_object* v_inst_103_, lean_object* v_inst_104_, lean_object* v_l_105_){
_start:
{
lean_object* v_res_106_; 
v_res_106_ = lp_mathlib_List_maximum(v_00_u03b1_102_, v_inst_103_, v_inst_104_, v_l_105_);
lean_dec_ref(v_inst_103_);
return v_res_106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_minimum___redArg(lean_object* v_inst_107_, lean_object* v_l_108_){
_start:
{
lean_object* v___x_109_; lean_object* v___x_110_; 
v___x_109_ = ((lean_object*)(lp_mathlib_List_maximum___redArg___closed__0));
v___x_110_ = lp_mathlib_List_argmin___redArg(v_inst_107_, v___x_109_, v_l_108_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_minimum(lean_object* v_00_u03b1_111_, lean_object* v_inst_112_, lean_object* v_inst_113_, lean_object* v_l_114_){
_start:
{
lean_object* v___x_115_; 
v___x_115_ = lp_mathlib_List_minimum___redArg(v_inst_113_, v_l_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_minimum___boxed(lean_object* v_00_u03b1_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_l_119_){
_start:
{
lean_object* v_res_120_; 
v_res_120_ = lp_mathlib_List_minimum(v_00_u03b1_116_, v_inst_117_, v_inst_118_, v_l_119_);
lean_dec_ref(v_inst_117_);
return v_res_120_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_maximum__of__length__pos___redArg(lean_object* v_inst_121_, lean_object* v_l_122_){
_start:
{
lean_object* v_toDecidableLT_123_; lean_object* v___x_124_; lean_object* v_val_125_; 
v_toDecidableLT_123_ = lean_ctor_get(v_inst_121_, 6);
lean_inc_ref(v_toDecidableLT_123_);
lean_dec_ref(v_inst_121_);
v___x_124_ = lp_mathlib_List_maximum___redArg(v_toDecidableLT_123_, v_l_122_);
v_val_125_ = lean_ctor_get(v___x_124_, 0);
lean_inc(v_val_125_);
lean_dec(v___x_124_);
return v_val_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_maximum__of__length__pos(lean_object* v_00_u03b1_126_, lean_object* v_inst_127_, lean_object* v_l_128_, lean_object* v_h_129_){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lp_mathlib_List_maximum__of__length__pos___redArg(v_inst_127_, v_l_128_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_minimum__of__length__pos___redArg(lean_object* v_inst_131_, lean_object* v_l_132_){
_start:
{
lean_object* v_toDecidableLT_133_; lean_object* v___x_134_; lean_object* v_val_135_; 
v_toDecidableLT_133_ = lean_ctor_get(v_inst_131_, 6);
lean_inc_ref(v_toDecidableLT_133_);
lean_dec_ref(v_inst_131_);
v___x_134_ = lp_mathlib_List_minimum___redArg(v_toDecidableLT_133_, v_l_132_);
v_val_135_ = lean_ctor_get(v___x_134_, 0);
lean_inc(v_val_135_);
lean_dec(v___x_134_);
return v_val_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_minimum__of__length__pos(lean_object* v_00_u03b1_136_, lean_object* v_inst_137_, lean_object* v_l_138_, lean_object* v_h_139_){
_start:
{
lean_object* v___x_140_; 
v___x_140_ = lp_mathlib_List_minimum__of__length__pos___redArg(v_inst_137_, v_l_138_);
return v___x_140_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_BoundedOrder_Lattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Induction(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_MinMax(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_WithBot(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_MinMax(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_BoundedOrder_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Induction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_MinMax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_MinMax(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_List_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_BoundedOrder_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Induction(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_MinMax(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_WithBot(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_MinMax(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_BoundedOrder_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Induction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_MinMax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_MinMax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_MinMax(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_MinMax(builtin);
}
#ifdef __cplusplus
}
#endif
