// Lean compiler output
// Module: Mathlib.Data.List.Lex
// Imports: public import Init public meta import Init public import Mathlib.Data.List.Basic public import Mathlib.Data.Nat.Basic public import Mathlib.Order.RelClasses
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
LEAN_EXPORT uint8_t lp_mathlib_List_Lex_decidableRel___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Lex_decidableRel___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_Lex_decidableRel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Lex_decidableRel___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instLinearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instLinearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instLinearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instLinearOrder___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instLinearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instLinearOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_instLinearOrder___redArg___lam__4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instLinearOrder___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_List_instLinearOrder___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_List_instLinearOrder___redArg___closed__0 = (const lean_object*)&lp_mathlib_List_instLinearOrder___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_instLinearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instLinearOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_LE_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_LE_x27(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_Lex_decidableRel___redArg(lean_object* v_inst_1_, lean_object* v_inst_2_, lean_object* v_x_3_, lean_object* v_x_4_){
_start:
{
uint8_t v___x_5_; 
v___x_5_ = 0;
if (lean_obj_tag(v_x_4_) == 0)
{
lean_dec(v_x_3_);
lean_dec_ref(v_inst_2_);
lean_dec_ref(v_inst_1_);
return v___x_5_;
}
else
{
lean_object* v_head_6_; lean_object* v_tail_7_; uint8_t v___x_8_; 
v_head_6_ = lean_ctor_get(v_x_4_, 0);
lean_inc(v_head_6_);
v_tail_7_ = lean_ctor_get(v_x_4_, 1);
lean_inc(v_tail_7_);
lean_dec_ref_known(v_x_4_, 2);
v___x_8_ = 1;
if (lean_obj_tag(v_x_3_) == 0)
{
lean_dec(v_tail_7_);
lean_dec(v_head_6_);
lean_dec_ref(v_inst_2_);
lean_dec_ref(v_inst_1_);
return v___x_8_;
}
else
{
lean_object* v_head_9_; lean_object* v_tail_10_; lean_object* v___x_11_; lean_object* v___x_12_; uint8_t v___x_13_; 
v_head_9_ = lean_ctor_get(v_x_3_, 0);
lean_inc_n(v_head_9_, 2);
v_tail_10_ = lean_ctor_get(v_x_3_, 1);
lean_inc(v_tail_10_);
lean_dec_ref_known(v_x_3_, 2);
lean_inc_ref(v_inst_1_);
lean_inc(v_head_6_);
v___x_11_ = lean_apply_2(v_inst_1_, v_head_9_, v_head_6_);
lean_inc_ref(v_inst_2_);
v___x_12_ = lean_apply_2(v_inst_2_, v_head_9_, v_head_6_);
v___x_13_ = lean_unbox(v___x_12_);
if (v___x_13_ == 0)
{
uint8_t v___x_14_; 
v___x_14_ = lean_unbox(v___x_11_);
if (v___x_14_ == 0)
{
lean_dec(v_tail_10_);
lean_dec(v_tail_7_);
lean_dec_ref(v_inst_2_);
lean_dec_ref(v_inst_1_);
return v___x_5_;
}
else
{
uint8_t v___x_15_; 
v___x_15_ = lp_mathlib_List_Lex_decidableRel___redArg(v_inst_1_, v_inst_2_, v_tail_10_, v_tail_7_);
if (v___x_15_ == 0)
{
return v___x_5_;
}
else
{
return v___x_8_;
}
}
}
else
{
lean_dec(v_tail_10_);
lean_dec(v_tail_7_);
lean_dec_ref(v_inst_2_);
lean_dec_ref(v_inst_1_);
return v___x_8_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Lex_decidableRel___redArg___boxed(lean_object* v_inst_16_, lean_object* v_inst_17_, lean_object* v_x_18_, lean_object* v_x_19_){
_start:
{
uint8_t v_res_20_; lean_object* v_r_21_; 
v_res_20_ = lp_mathlib_List_Lex_decidableRel___redArg(v_inst_16_, v_inst_17_, v_x_18_, v_x_19_);
v_r_21_ = lean_box(v_res_20_);
return v_r_21_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_Lex_decidableRel(lean_object* v_00_u03b1_22_, lean_object* v_inst_23_, lean_object* v_r_24_, lean_object* v_inst_25_, lean_object* v_x_26_, lean_object* v_x_27_){
_start:
{
uint8_t v___x_28_; 
v___x_28_ = lp_mathlib_List_Lex_decidableRel___redArg(v_inst_23_, v_inst_25_, v_x_26_, v_x_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Lex_decidableRel___boxed(lean_object* v_00_u03b1_29_, lean_object* v_inst_30_, lean_object* v_r_31_, lean_object* v_inst_32_, lean_object* v_x_33_, lean_object* v_x_34_){
_start:
{
uint8_t v_res_35_; lean_object* v_r_36_; 
v_res_35_ = lp_mathlib_List_Lex_decidableRel(v_00_u03b1_29_, v_inst_30_, v_r_31_, v_inst_32_, v_x_33_, v_x_34_);
v_r_36_ = lean_box(v_res_35_);
return v_r_36_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instLinearOrder___redArg___lam__0(lean_object* v_inst_37_, lean_object* v_a_38_, lean_object* v_b_39_){
_start:
{
lean_object* v_toDecidableEq_40_; lean_object* v___x_41_; uint8_t v___x_42_; 
v_toDecidableEq_40_ = lean_ctor_get(v_inst_37_, 5);
lean_inc_ref(v_toDecidableEq_40_);
lean_dec_ref(v_inst_37_);
v___x_41_ = lean_apply_2(v_toDecidableEq_40_, v_a_38_, v_b_39_);
v___x_42_ = lean_unbox(v___x_41_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instLinearOrder___redArg___lam__0___boxed(lean_object* v_inst_43_, lean_object* v_a_44_, lean_object* v_b_45_){
_start:
{
uint8_t v_res_46_; lean_object* v_r_47_; 
v_res_46_ = lp_mathlib_List_instLinearOrder___redArg___lam__0(v_inst_43_, v_a_44_, v_b_45_);
v_r_47_ = lean_box(v_res_46_);
return v_r_47_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instLinearOrder___redArg___lam__1(lean_object* v_inst_48_, lean_object* v___f_49_, lean_object* v_x_50_, lean_object* v_y_51_){
_start:
{
lean_object* v_toDecidableLT_52_; uint8_t v___x_53_; 
v_toDecidableLT_52_ = lean_ctor_get(v_inst_48_, 6);
lean_inc_ref(v_toDecidableLT_52_);
lean_dec_ref(v_inst_48_);
v___x_53_ = lp_mathlib_List_Lex_decidableRel___redArg(v___f_49_, v_toDecidableLT_52_, v_y_51_, v_x_50_);
if (v___x_53_ == 0)
{
uint8_t v___x_54_; 
v___x_54_ = 1;
return v___x_54_;
}
else
{
uint8_t v___x_55_; 
v___x_55_ = 0;
return v___x_55_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instLinearOrder___redArg___lam__1___boxed(lean_object* v_inst_56_, lean_object* v___f_57_, lean_object* v_x_58_, lean_object* v_y_59_){
_start:
{
uint8_t v_res_60_; lean_object* v_r_61_; 
v_res_60_ = lp_mathlib_List_instLinearOrder___redArg___lam__1(v_inst_56_, v___f_57_, v_x_58_, v_y_59_);
v_r_61_ = lean_box(v_res_60_);
return v_r_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instLinearOrder___redArg___lam__2(lean_object* v_hD_62_, lean_object* v_x_63_, lean_object* v_y_64_){
_start:
{
lean_object* v___x_65_; uint8_t v___x_66_; 
lean_inc(v_y_64_);
lean_inc(v_x_63_);
v___x_65_ = lean_apply_2(v_hD_62_, v_x_63_, v_y_64_);
v___x_66_ = lean_unbox(v___x_65_);
if (v___x_66_ == 0)
{
lean_dec(v_x_63_);
return v_y_64_;
}
else
{
lean_dec(v_y_64_);
return v_x_63_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instLinearOrder___redArg___lam__3(lean_object* v_hD_67_, lean_object* v_x_68_, lean_object* v_y_69_){
_start:
{
lean_object* v___x_70_; uint8_t v___x_71_; 
lean_inc(v_y_69_);
lean_inc(v_x_68_);
v___x_70_ = lean_apply_2(v_hD_67_, v_x_68_, v_y_69_);
v___x_71_ = lean_unbox(v___x_70_);
if (v___x_71_ == 0)
{
lean_dec(v_y_69_);
return v_x_68_;
}
else
{
lean_dec(v_x_68_);
return v_y_69_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_instLinearOrder___redArg___lam__4(lean_object* v_hD_72_, lean_object* v_a_73_, lean_object* v_b_74_){
_start:
{
uint8_t v___x_75_; 
lean_inc(v_b_74_);
lean_inc(v_a_73_);
lean_inc_ref(v_hD_72_);
v___x_75_ = lp_mathlib_decidableLTOfDecidableLE___redArg(v_hD_72_, v_a_73_, v_b_74_);
if (v___x_75_ == 0)
{
uint8_t v___x_76_; 
v___x_76_ = lp_mathlib_decidableEqOfDecidableLE___redArg(v_hD_72_, v_a_73_, v_b_74_);
if (v___x_76_ == 0)
{
uint8_t v___x_77_; 
v___x_77_ = 2;
return v___x_77_;
}
else
{
uint8_t v___x_78_; 
v___x_78_ = 1;
return v___x_78_;
}
}
else
{
uint8_t v___x_79_; 
lean_dec(v_b_74_);
lean_dec(v_a_73_);
lean_dec_ref(v_hD_72_);
v___x_79_ = 0;
return v___x_79_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instLinearOrder___redArg___lam__4___boxed(lean_object* v_hD_80_, lean_object* v_a_81_, lean_object* v_b_82_){
_start:
{
uint8_t v_res_83_; lean_object* v_r_84_; 
v_res_83_ = lp_mathlib_List_instLinearOrder___redArg___lam__4(v_hD_80_, v_a_81_, v_b_82_);
v_r_84_ = lean_box(v_res_83_);
return v_r_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instLinearOrder___redArg(lean_object* v_inst_88_){
_start:
{
lean_object* v___f_89_; lean_object* v_hD_90_; lean_object* v___f_91_; lean_object* v___f_92_; lean_object* v___f_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
lean_inc_ref(v_inst_88_);
v___f_89_ = lean_alloc_closure((void*)(lp_mathlib_List_instLinearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_89_, 0, v_inst_88_);
v_hD_90_ = lean_alloc_closure((void*)(lp_mathlib_List_instLinearOrder___redArg___lam__1___boxed), 4, 2);
lean_closure_set(v_hD_90_, 0, v_inst_88_);
lean_closure_set(v_hD_90_, 1, v___f_89_);
lean_inc_ref_n(v_hD_90_, 5);
v___f_91_ = lean_alloc_closure((void*)(lp_mathlib_List_instLinearOrder___redArg___lam__2), 3, 1);
lean_closure_set(v___f_91_, 0, v_hD_90_);
v___f_92_ = lean_alloc_closure((void*)(lp_mathlib_List_instLinearOrder___redArg___lam__3), 3, 1);
lean_closure_set(v___f_92_, 0, v_hD_90_);
v___f_93_ = lean_alloc_closure((void*)(lp_mathlib_List_instLinearOrder___redArg___lam__4___boxed), 3, 1);
lean_closure_set(v___f_93_, 0, v_hD_90_);
v___x_94_ = ((lean_object*)(lp_mathlib_List_instLinearOrder___redArg___closed__0));
v___x_95_ = lean_alloc_closure((void*)(lp_mathlib_decidableEqOfDecidableLE___boxed), 5, 3);
lean_closure_set(v___x_95_, 0, lean_box(0));
lean_closure_set(v___x_95_, 1, v___x_94_);
lean_closure_set(v___x_95_, 2, v_hD_90_);
v___x_96_ = lean_alloc_closure((void*)(lp_mathlib_decidableLTOfDecidableLE___boxed), 5, 3);
lean_closure_set(v___x_96_, 0, lean_box(0));
lean_closure_set(v___x_96_, 1, v___x_94_);
lean_closure_set(v___x_96_, 2, v_hD_90_);
v___x_97_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_97_, 0, v___x_94_);
lean_ctor_set(v___x_97_, 1, v___f_91_);
lean_ctor_set(v___x_97_, 2, v___f_92_);
lean_ctor_set(v___x_97_, 3, v___f_93_);
lean_ctor_set(v___x_97_, 4, v_hD_90_);
lean_ctor_set(v___x_97_, 5, v___x_95_);
lean_ctor_set(v___x_97_, 6, v___x_96_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instLinearOrder(lean_object* v_00_u03b1_98_, lean_object* v_inst_99_){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = lp_mathlib_List_instLinearOrder___redArg(v_inst_99_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_LE_x27___redArg(lean_object* v_inst_101_){
_start:
{
lean_object* v___x_102_; lean_object* v_toPartialOrder_103_; lean_object* v_toLE_104_; 
v___x_102_ = lp_mathlib_List_instLinearOrder___redArg(v_inst_101_);
v_toPartialOrder_103_ = lean_ctor_get(v___x_102_, 0);
lean_inc_ref(v_toPartialOrder_103_);
lean_dec_ref(v___x_102_);
v_toLE_104_ = lean_ctor_get(v_toPartialOrder_103_, 0);
lean_inc(v_toLE_104_);
lean_dec_ref(v_toPartialOrder_103_);
return v_toLE_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_LE_x27(lean_object* v_00_u03b1_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lp_mathlib_List_LE_x27___redArg(v_inst_106_);
return v___x_107_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_RelClasses(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Lex(uint8_t builtin) {
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
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_RelClasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_Lex(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Nat_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_RelClasses(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_Lex(uint8_t builtin) {
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
res = initialize_mathlib_Mathlib_Data_Nat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_RelClasses(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_Lex(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_Lex(builtin);
}
#ifdef __cplusplus
}
#endif
