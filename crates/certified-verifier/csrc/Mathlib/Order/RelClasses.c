// Lean compiler output
// Module: Mathlib.Order.RelClasses
// Imports: public import Init public meta import Init public import Mathlib.Basic.IsEmpty.Basic public import Mathlib.Order.OrderDual public import Mathlib.Tactic.CrossRefAttribute public import Mathlib.Tactic.MkIffOfInductiveProp
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
uint8_t lp_mathlib_decidableLTOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_decidableEqOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_decidableEqOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_decidableLTOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_WellFounded_fixC___redArg(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_partialOrderOfSO___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_partialOrderOfSO___closed__0 = (const lean_object*)&lp_mathlib_partialOrderOfSO___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_partialOrderOfSO(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_linearOrderOfSTO___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_linearOrderOfSTO___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_fix_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_fix_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_toWellFoundedRelation(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedLT_fix___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedLT_fix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedGT_fix___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedGT_fix(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedLT_toWellFoundedRelation(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedGT_toWellFoundedRelation(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsWellOrder_toHasWellFounded(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_partialOrderOfSO(lean_object* v_00_u03b1_4_, lean_object* v_r_5_, lean_object* v_inst_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = ((lean_object*)(lp_mathlib_partialOrderOfSO___closed__0));
return v___x_7_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_linearOrderOfSTO___redArg___lam__0(lean_object* v_inst_8_, lean_object* v_x_9_, lean_object* v_y_10_){
_start:
{
lean_object* v___x_11_; uint8_t v___x_12_; 
v___x_11_ = lean_apply_2(v_inst_8_, v_y_10_, v_x_9_);
v___x_12_ = lean_unbox(v___x_11_);
if (v___x_12_ == 0)
{
uint8_t v___x_13_; 
v___x_13_ = 1;
return v___x_13_;
}
else
{
uint8_t v___x_14_; 
v___x_14_ = 0;
return v___x_14_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg___lam__0___boxed(lean_object* v_inst_15_, lean_object* v_x_16_, lean_object* v_y_17_){
_start:
{
uint8_t v_res_18_; lean_object* v_r_19_; 
v_res_18_ = lp_mathlib_linearOrderOfSTO___redArg___lam__0(v_inst_15_, v_x_16_, v_y_17_);
v_r_19_ = lean_box(v_res_18_);
return v_r_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg___lam__1(lean_object* v_hD_20_, lean_object* v_x_21_, lean_object* v_y_22_){
_start:
{
lean_object* v___x_23_; uint8_t v___x_24_; 
lean_inc(v_y_22_);
lean_inc(v_x_21_);
v___x_23_ = lean_apply_2(v_hD_20_, v_x_21_, v_y_22_);
v___x_24_ = lean_unbox(v___x_23_);
if (v___x_24_ == 0)
{
lean_dec(v_y_22_);
return v_x_21_;
}
else
{
lean_dec(v_x_21_);
return v_y_22_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg___lam__2(lean_object* v_hD_25_, lean_object* v_x_26_, lean_object* v_y_27_){
_start:
{
lean_object* v___x_28_; uint8_t v___x_29_; 
lean_inc(v_y_27_);
lean_inc(v_x_26_);
v___x_28_ = lean_apply_2(v_hD_25_, v_x_26_, v_y_27_);
v___x_29_ = lean_unbox(v___x_28_);
if (v___x_29_ == 0)
{
lean_dec(v_x_26_);
return v_y_27_;
}
else
{
lean_dec(v_y_27_);
return v_x_26_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_linearOrderOfSTO___redArg___lam__3(lean_object* v_hD_30_, lean_object* v_a_31_, lean_object* v_b_32_){
_start:
{
uint8_t v___x_33_; 
lean_inc(v_b_32_);
lean_inc(v_a_31_);
lean_inc_ref(v_hD_30_);
v___x_33_ = lp_mathlib_decidableLTOfDecidableLE___redArg(v_hD_30_, v_a_31_, v_b_32_);
if (v___x_33_ == 0)
{
uint8_t v___x_34_; 
v___x_34_ = lp_mathlib_decidableEqOfDecidableLE___redArg(v_hD_30_, v_a_31_, v_b_32_);
if (v___x_34_ == 0)
{
uint8_t v___x_35_; 
v___x_35_ = 2;
return v___x_35_;
}
else
{
uint8_t v___x_36_; 
v___x_36_ = 1;
return v___x_36_;
}
}
else
{
uint8_t v___x_37_; 
lean_dec(v_b_32_);
lean_dec(v_a_31_);
lean_dec_ref(v_hD_30_);
v___x_37_ = 0;
return v___x_37_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg___lam__3___boxed(lean_object* v_hD_38_, lean_object* v_a_39_, lean_object* v_b_40_){
_start:
{
uint8_t v_res_41_; lean_object* v_r_42_; 
v_res_41_ = lp_mathlib_linearOrderOfSTO___redArg___lam__3(v_hD_38_, v_a_39_, v_b_40_);
v_r_42_ = lean_box(v_res_41_);
return v_r_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO___redArg(lean_object* v_inst_43_){
_start:
{
lean_object* v_hD_44_; lean_object* v___f_45_; lean_object* v___f_46_; lean_object* v___f_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v_hD_44_ = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfSTO___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v_hD_44_, 0, v_inst_43_);
lean_inc_ref_n(v_hD_44_, 5);
v___f_45_ = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfSTO___redArg___lam__1), 3, 1);
lean_closure_set(v___f_45_, 0, v_hD_44_);
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfSTO___redArg___lam__2), 3, 1);
lean_closure_set(v___f_46_, 0, v_hD_44_);
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfSTO___redArg___lam__3___boxed), 3, 1);
lean_closure_set(v___f_47_, 0, v_hD_44_);
v___x_48_ = ((lean_object*)(lp_mathlib_partialOrderOfSO___closed__0));
v___x_49_ = lean_alloc_closure((void*)(lp_mathlib_decidableEqOfDecidableLE___boxed), 5, 3);
lean_closure_set(v___x_49_, 0, lean_box(0));
lean_closure_set(v___x_49_, 1, v___x_48_);
lean_closure_set(v___x_49_, 2, v_hD_44_);
v___x_50_ = lean_alloc_closure((void*)(lp_mathlib_decidableLTOfDecidableLE___boxed), 5, 3);
lean_closure_set(v___x_50_, 0, lean_box(0));
lean_closure_set(v___x_50_, 1, v___x_48_);
lean_closure_set(v___x_50_, 2, v_hD_44_);
v___x_51_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_51_, 0, v___x_48_);
lean_ctor_set(v___x_51_, 1, v___f_46_);
lean_ctor_set(v___x_51_, 2, v___f_45_);
lean_ctor_set(v___x_51_, 3, v___f_47_);
lean_ctor_set(v___x_51_, 4, v_hD_44_);
lean_ctor_set(v___x_51_, 5, v___x_49_);
lean_ctor_set(v___x_51_, 6, v___x_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_linearOrderOfSTO(lean_object* v_00_u03b1_52_, lean_object* v_r_53_, lean_object* v_inst_54_, lean_object* v_inst_55_){
_start:
{
lean_object* v_hD_56_; lean_object* v___f_57_; lean_object* v___f_58_; lean_object* v___f_59_; lean_object* v___x_60_; lean_object* v___x_61_; lean_object* v___x_62_; lean_object* v___x_63_; 
v_hD_56_ = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfSTO___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v_hD_56_, 0, v_inst_55_);
lean_inc_ref_n(v_hD_56_, 5);
v___f_57_ = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfSTO___redArg___lam__1), 3, 1);
lean_closure_set(v___f_57_, 0, v_hD_56_);
v___f_58_ = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfSTO___redArg___lam__2), 3, 1);
lean_closure_set(v___f_58_, 0, v_hD_56_);
v___f_59_ = lean_alloc_closure((void*)(lp_mathlib_linearOrderOfSTO___redArg___lam__3___boxed), 3, 1);
lean_closure_set(v___f_59_, 0, v_hD_56_);
v___x_60_ = ((lean_object*)(lp_mathlib_partialOrderOfSO___closed__0));
v___x_61_ = lean_alloc_closure((void*)(lp_mathlib_decidableEqOfDecidableLE___boxed), 5, 3);
lean_closure_set(v___x_61_, 0, lean_box(0));
lean_closure_set(v___x_61_, 1, v___x_60_);
lean_closure_set(v___x_61_, 2, v_hD_56_);
v___x_62_ = lean_alloc_closure((void*)(lp_mathlib_decidableLTOfDecidableLE___boxed), 5, 3);
lean_closure_set(v___x_62_, 0, lean_box(0));
lean_closure_set(v___x_62_, 1, v___x_60_);
lean_closure_set(v___x_62_, 2, v_hD_56_);
v___x_63_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_63_, 0, v___x_60_);
lean_ctor_set(v___x_63_, 1, v___f_58_);
lean_ctor_set(v___x_63_, 2, v___f_57_);
lean_ctor_set(v___x_63_, 3, v___f_59_);
lean_ctor_set(v___x_63_, 4, v_hD_56_);
lean_ctor_set(v___x_63_, 5, v___x_61_);
lean_ctor_set(v___x_63_, 6, v___x_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_fix_x27___redArg(lean_object* v_F_64_, lean_object* v_x_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = l_WellFounded_fixC___redArg(v_F_64_, v_x_65_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_fix_x27(lean_object* v_00_u03b1_67_, lean_object* v_r_68_, lean_object* v_i_69_, lean_object* v_motive_70_, lean_object* v_F_71_, lean_object* v_x_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = l_WellFounded_fixC___redArg(v_F_71_, v_x_72_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFounded_toWellFoundedRelation(lean_object* v_00_u03b1_74_, lean_object* v_r_75_, lean_object* v_i_76_){
_start:
{
lean_object* v___x_77_; 
v___x_77_ = lean_box(0);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedLT_fix___redArg(lean_object* v_ind_78_, lean_object* v_x_79_){
_start:
{
lean_object* v___x_80_; 
v___x_80_ = l_WellFounded_fixC___redArg(v_ind_78_, v_x_79_);
return v___x_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedLT_fix(lean_object* v_00_u03b1_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_motive_84_, lean_object* v_ind_85_, lean_object* v_x_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = l_WellFounded_fixC___redArg(v_ind_85_, v_x_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedGT_fix___redArg(lean_object* v_ind_88_, lean_object* v_x_89_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = l_WellFounded_fixC___redArg(v_ind_88_, v_x_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedGT_fix(lean_object* v_00_u03b1_91_, lean_object* v_inst_92_, lean_object* v_inst_93_, lean_object* v_motive_94_, lean_object* v_ind_95_, lean_object* v_x_96_){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = l_WellFounded_fixC___redArg(v_ind_95_, v_x_96_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedLT_toWellFoundedRelation(lean_object* v_00_u03b1_98_, lean_object* v_inst_99_, lean_object* v_inst_100_){
_start:
{
lean_object* v___x_101_; 
v___x_101_ = lean_box(0);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WellFoundedGT_toWellFoundedRelation(lean_object* v_00_u03b1_102_, lean_object* v_inst_103_, lean_object* v_inst_104_){
_start:
{
lean_object* v___x_105_; 
v___x_105_ = lean_box(0);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsWellOrder_toHasWellFounded(lean_object* v_00_u03b1_106_, lean_object* v_inst_107_, lean_object* v_hwo_108_){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lean_box(0);
return v___x_109_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_IsEmpty_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_OrderDual(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_RelClasses(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_IsEmpty_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_OrderDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_RelClasses(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Basic_IsEmpty_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_OrderDual(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_RelClasses(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_IsEmpty_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_OrderDual(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_CrossRefAttribute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_MkIffOfInductiveProp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_RelClasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_RelClasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_RelClasses(builtin);
}
#ifdef __cplusplus
}
#endif
