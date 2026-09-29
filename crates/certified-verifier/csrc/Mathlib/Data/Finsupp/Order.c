// Lean compiler output
// Module: Mathlib.Data.Finsupp.Order
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.BigOperators.Group.Finset public import Mathlib.Algebra.Order.Module.Defs public import Mathlib.Algebra.Order.Pi public import Mathlib.Algebra.Order.Sub.Basic public import Mathlib.Data.Finsupp.Basic public import Mathlib.Data.Finsupp.SMulWithZero public import Mathlib.Order.Preorder.Finsupp
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
uint8_t lp_mathlib_Multiset_decidableDforallMultiset___redArg(lean_object*, lean_object*);
uint8_t lp_mathlib_decidableLTOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_orderBot___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_orderBot___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_orderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_orderBot___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_orderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_orderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_decidableLE___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_decidableLE___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_decidableLE___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Finsupp_decidableLE___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Finsupp_decidableLE___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Finsupp_decidableLE___redArg___closed__0 = (const lean_object*)&lp_mathlib_Finsupp_decidableLE___redArg___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_decidableLE___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_decidableLE___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_decidableLE(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_decidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_decidableLT___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_decidableLT___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_decidableLT___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_decidableLT___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_decidableLT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_decidableLT___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_orderBot___redArg___lam__0(lean_object* v_toZero_1_, lean_object* v_x_2_){
_start:
{
lean_inc(v_toZero_1_);
return v_toZero_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_orderBot___redArg___lam__0___boxed(lean_object* v_toZero_3_, lean_object* v_x_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_mathlib_Finsupp_orderBot___redArg___lam__0(v_toZero_3_, v_x_4_);
lean_dec(v_x_4_);
lean_dec(v_toZero_3_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_orderBot___redArg(lean_object* v_inst_6_){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v_toZero_9_; lean_object* v___x_11_; uint8_t v_isShared_12_; uint8_t v_isSharedCheck_18_; 
v___x_7_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_6_);
v___x_8_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_7_);
v_toZero_9_ = lean_ctor_get(v___x_8_, 0);
v_isSharedCheck_18_ = !lean_is_exclusive(v___x_8_);
if (v_isSharedCheck_18_ == 0)
{
lean_object* v_unused_19_; 
v_unused_19_ = lean_ctor_get(v___x_8_, 1);
lean_dec(v_unused_19_);
v___x_11_ = v___x_8_;
v_isShared_12_ = v_isSharedCheck_18_;
goto v_resetjp_10_;
}
else
{
lean_inc(v_toZero_9_);
lean_dec(v___x_8_);
v___x_11_ = lean_box(0);
v_isShared_12_ = v_isSharedCheck_18_;
goto v_resetjp_10_;
}
v_resetjp_10_:
{
lean_object* v___f_13_; lean_object* v___x_14_; lean_object* v___x_16_; 
v___f_13_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_orderBot___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_13_, 0, v_toZero_9_);
v___x_14_ = lean_box(0);
if (v_isShared_12_ == 0)
{
lean_ctor_set(v___x_11_, 1, v___f_13_);
lean_ctor_set(v___x_11_, 0, v___x_14_);
v___x_16_ = v___x_11_;
goto v_reusejp_15_;
}
else
{
lean_object* v_reuseFailAlloc_17_; 
v_reuseFailAlloc_17_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_17_, 0, v___x_14_);
lean_ctor_set(v_reuseFailAlloc_17_, 1, v___f_13_);
v___x_16_ = v_reuseFailAlloc_17_;
goto v_reusejp_15_;
}
v_reusejp_15_:
{
return v___x_16_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_orderBot___redArg___boxed(lean_object* v_inst_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_Finsupp_orderBot___redArg(v_inst_20_);
lean_dec_ref(v_inst_20_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_orderBot(lean_object* v_00_u03b9_22_, lean_object* v_00_u03b1_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_){
_start:
{
lean_object* v___x_27_; 
v___x_27_ = lp_mathlib_Finsupp_orderBot___redArg(v_inst_24_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_orderBot___boxed(lean_object* v_00_u03b9_28_, lean_object* v_00_u03b1_29_, lean_object* v_inst_30_, lean_object* v_inst_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v_res_33_; 
v_res_33_ = lp_mathlib_Finsupp_orderBot(v_00_u03b9_28_, v_00_u03b1_29_, v_inst_30_, v_inst_31_, v_inst_32_);
lean_dec_ref(v_inst_31_);
lean_dec_ref(v_inst_30_);
return v_res_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_decidableLE___redArg___lam__0(lean_object* v_self_34_, lean_object* v___y_35_){
_start:
{
lean_object* v_toFun_36_; lean_object* v___x_37_; 
v_toFun_36_ = lean_ctor_get(v_self_34_, 1);
lean_inc(v_toFun_36_);
lean_dec_ref(v_self_34_);
v___x_37_ = lean_apply_1(v_toFun_36_, v___y_35_);
return v___x_37_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_decidableLE___redArg___lam__1(lean_object* v___f_38_, lean_object* v_f_39_, lean_object* v_g_40_, lean_object* v_inst_41_, lean_object* v_a_42_, lean_object* v_h_43_){
_start:
{
lean_object* v___x_44_; lean_object* v___x_45_; lean_object* v___x_46_; uint8_t v___x_47_; 
lean_inc(v___f_38_);
lean_inc(v_a_42_);
v___x_44_ = lean_apply_2(v___f_38_, v_f_39_, v_a_42_);
v___x_45_ = lean_apply_2(v___f_38_, v_g_40_, v_a_42_);
v___x_46_ = lean_apply_2(v_inst_41_, v___x_44_, v___x_45_);
v___x_47_ = lean_unbox(v___x_46_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_decidableLE___redArg___lam__1___boxed(lean_object* v___f_48_, lean_object* v_f_49_, lean_object* v_g_50_, lean_object* v_inst_51_, lean_object* v_a_52_, lean_object* v_h_53_){
_start:
{
uint8_t v_res_54_; lean_object* v_r_55_; 
v_res_54_ = lp_mathlib_Finsupp_decidableLE___redArg___lam__1(v___f_48_, v_f_49_, v_g_50_, v_inst_51_, v_a_52_, v_h_53_);
v_r_55_ = lean_box(v_res_54_);
return v_r_55_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_decidableLE___redArg(lean_object* v_inst_57_, lean_object* v_f_58_, lean_object* v_g_59_){
_start:
{
lean_object* v_support_60_; lean_object* v___f_61_; lean_object* v___f_62_; uint8_t v___x_63_; 
v_support_60_ = lean_ctor_get(v_f_58_, 0);
lean_inc(v_support_60_);
v___f_61_ = ((lean_object*)(lp_mathlib_Finsupp_decidableLE___redArg___closed__0));
v___f_62_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_decidableLE___redArg___lam__1___boxed), 6, 4);
lean_closure_set(v___f_62_, 0, v___f_61_);
lean_closure_set(v___f_62_, 1, v_f_58_);
lean_closure_set(v___f_62_, 2, v_g_59_);
lean_closure_set(v___f_62_, 3, v_inst_57_);
v___x_63_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v_support_60_, v___f_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_decidableLE___redArg___boxed(lean_object* v_inst_64_, lean_object* v_f_65_, lean_object* v_g_66_){
_start:
{
uint8_t v_res_67_; lean_object* v_r_68_; 
v_res_67_ = lp_mathlib_Finsupp_decidableLE___redArg(v_inst_64_, v_f_65_, v_g_66_);
v_r_68_ = lean_box(v_res_67_);
return v_r_68_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_decidableLE(lean_object* v_00_u03b9_69_, lean_object* v_00_u03b1_70_, lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_inst_73_, lean_object* v_inst_74_, lean_object* v_f_75_, lean_object* v_g_76_){
_start:
{
uint8_t v___x_77_; 
v___x_77_ = lp_mathlib_Finsupp_decidableLE___redArg(v_inst_74_, v_f_75_, v_g_76_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_decidableLE___boxed(lean_object* v_00_u03b9_78_, lean_object* v_00_u03b1_79_, lean_object* v_inst_80_, lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_inst_83_, lean_object* v_f_84_, lean_object* v_g_85_){
_start:
{
uint8_t v_res_86_; lean_object* v_r_87_; 
v_res_86_ = lp_mathlib_Finsupp_decidableLE(v_00_u03b9_78_, v_00_u03b1_79_, v_inst_80_, v_inst_81_, v_inst_82_, v_inst_83_, v_f_84_, v_g_85_);
lean_dec_ref(v_inst_81_);
lean_dec_ref(v_inst_80_);
v_r_87_ = lean_box(v_res_86_);
return v_r_87_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_decidableLT___redArg___lam__0(lean_object* v_inst_88_, lean_object* v_a_89_, lean_object* v_b_90_){
_start:
{
uint8_t v___x_91_; 
v___x_91_ = lp_mathlib_Finsupp_decidableLE___redArg(v_inst_88_, v_a_89_, v_b_90_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_decidableLT___redArg___lam__0___boxed(lean_object* v_inst_92_, lean_object* v_a_93_, lean_object* v_b_94_){
_start:
{
uint8_t v_res_95_; lean_object* v_r_96_; 
v_res_95_ = lp_mathlib_Finsupp_decidableLT___redArg___lam__0(v_inst_92_, v_a_93_, v_b_94_);
v_r_96_ = lean_box(v_res_95_);
return v_r_96_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_decidableLT___redArg(lean_object* v_inst_97_, lean_object* v_a_98_, lean_object* v_b_99_){
_start:
{
lean_object* v___f_100_; uint8_t v___x_101_; 
v___f_100_ = lean_alloc_closure((void*)(lp_mathlib_Finsupp_decidableLT___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_100_, 0, v_inst_97_);
v___x_101_ = lp_mathlib_decidableLTOfDecidableLE___redArg(v___f_100_, v_a_98_, v_b_99_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_decidableLT___redArg___boxed(lean_object* v_inst_102_, lean_object* v_a_103_, lean_object* v_b_104_){
_start:
{
uint8_t v_res_105_; lean_object* v_r_106_; 
v_res_105_ = lp_mathlib_Finsupp_decidableLT___redArg(v_inst_102_, v_a_103_, v_b_104_);
v_r_106_ = lean_box(v_res_105_);
return v_r_106_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Finsupp_decidableLT(lean_object* v_00_u03b9_107_, lean_object* v_00_u03b1_108_, lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_inst_111_, lean_object* v_inst_112_, lean_object* v_a_113_, lean_object* v_b_114_){
_start:
{
uint8_t v___x_115_; 
v___x_115_ = lp_mathlib_Finsupp_decidableLT___redArg(v_inst_112_, v_a_113_, v_b_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finsupp_decidableLT___boxed(lean_object* v_00_u03b9_116_, lean_object* v_00_u03b1_117_, lean_object* v_inst_118_, lean_object* v_inst_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_a_122_, lean_object* v_b_123_){
_start:
{
uint8_t v_res_124_; lean_object* v_r_125_; 
v_res_124_ = lp_mathlib_Finsupp_decidableLT(v_00_u03b9_116_, v_00_u03b1_117_, v_inst_118_, v_inst_119_, v_inst_120_, v_inst_121_, v_a_122_, v_b_123_);
lean_dec_ref(v_inst_119_);
lean_dec_ref(v_inst_118_);
v_r_125_ = lean_box(v_res_124_);
return v_r_125_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Module_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_SMulWithZero(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Preorder_Finsupp(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Finsupp_Order(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_SMulWithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Preorder_Finsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Finsupp_Order(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Module_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_SMulWithZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Preorder_Finsupp(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Order(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_BigOperators_Group_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Module_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Sub_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_SMulWithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Preorder_Finsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finsupp_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Finsupp_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Finsupp_Order(builtin);
}
#ifdef __cplusplus
}
#endif
