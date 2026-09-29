// Lean compiler output
// Module: Mathlib.Data.Multiset.Pi
// Imports: public import Init public meta import Init public import Mathlib.Data.Multiset.Bind
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
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_bind___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_rec___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Pi_empty(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Pi_empty___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Pi_cons___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Pi_cons___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Pi_cons(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Pi_cons___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_pi___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_pi___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Multiset_pi___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Multiset_Pi_empty___boxed, .m_arity = 4, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Multiset_pi___redArg___closed__0 = (const lean_object*)&lp_mathlib_Multiset_pi___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_Multiset_pi___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Multiset_pi___redArg___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Multiset_pi___redArg___closed__1 = (const lean_object*)&lp_mathlib_Multiset_pi___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Multiset_pi___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Pi_empty(lean_object* v_00_u03b1_1_, lean_object* v_00_u03b4_2_, lean_object* v_a_3_, lean_object* v_a_4_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Pi_empty___boxed(lean_object* v_00_u03b1_5_, lean_object* v_00_u03b4_6_, lean_object* v_a_7_, lean_object* v_a_8_){
_start:
{
lean_object* v_res_9_; 
v_res_9_ = lp_mathlib_Multiset_Pi_empty(v_00_u03b1_5_, v_00_u03b4_6_, v_a_7_, v_a_8_);
lean_dec(v_a_7_);
return v_res_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Pi_cons___redArg(lean_object* v_inst_10_, lean_object* v_a_11_, lean_object* v_b_12_, lean_object* v_f_13_, lean_object* v_a_x27_14_){
_start:
{
lean_object* v___x_15_; uint8_t v___x_16_; 
lean_inc(v_a_x27_14_);
v___x_15_ = lean_apply_2(v_inst_10_, v_a_x27_14_, v_a_11_);
v___x_16_ = lean_unbox(v___x_15_);
if (v___x_16_ == 0)
{
lean_object* v___x_17_; 
v___x_17_ = lean_apply_2(v_f_13_, v_a_x27_14_, lean_box(0));
return v___x_17_;
}
else
{
lean_dec(v_a_x27_14_);
lean_dec(v_f_13_);
lean_inc(v_b_12_);
return v_b_12_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Pi_cons___redArg___boxed(lean_object* v_inst_18_, lean_object* v_a_19_, lean_object* v_b_20_, lean_object* v_f_21_, lean_object* v_a_x27_22_){
_start:
{
lean_object* v_res_23_; 
v_res_23_ = lp_mathlib_Multiset_Pi_cons___redArg(v_inst_18_, v_a_19_, v_b_20_, v_f_21_, v_a_x27_22_);
lean_dec(v_b_20_);
return v_res_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Pi_cons(lean_object* v_00_u03b1_24_, lean_object* v_inst_25_, lean_object* v_00_u03b4_26_, lean_object* v_m_27_, lean_object* v_a_28_, lean_object* v_b_29_, lean_object* v_f_30_, lean_object* v_a_x27_31_, lean_object* v_ha_x27_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_Multiset_Pi_cons___redArg(v_inst_25_, v_a_28_, v_b_29_, v_f_30_, v_a_x27_31_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Pi_cons___boxed(lean_object* v_00_u03b1_34_, lean_object* v_inst_35_, lean_object* v_00_u03b4_36_, lean_object* v_m_37_, lean_object* v_a_38_, lean_object* v_b_39_, lean_object* v_f_40_, lean_object* v_a_x27_41_, lean_object* v_ha_x27_42_){
_start:
{
lean_object* v_res_43_; 
v_res_43_ = lp_mathlib_Multiset_Pi_cons(v_00_u03b1_34_, v_inst_35_, v_00_u03b4_36_, v_m_37_, v_a_38_, v_b_39_, v_f_40_, v_a_x27_41_, v_ha_x27_42_);
lean_dec(v_b_39_);
lean_dec(v_m_37_);
return v_res_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_pi___redArg___lam__0(lean_object* v_inst_44_, lean_object* v_m_45_, lean_object* v_a_46_, lean_object* v_p_47_, lean_object* v_b_48_){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_49_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_Pi_cons___boxed), 9, 6);
lean_closure_set(v___x_49_, 0, lean_box(0));
lean_closure_set(v___x_49_, 1, v_inst_44_);
lean_closure_set(v___x_49_, 2, lean_box(0));
lean_closure_set(v___x_49_, 3, v_m_45_);
lean_closure_set(v___x_49_, 4, v_a_46_);
lean_closure_set(v___x_49_, 5, v_b_48_);
v___x_50_ = lp_mathlib_Multiset_map___redArg(v___x_49_, v_p_47_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_pi___redArg___lam__1(lean_object* v_inst_51_, lean_object* v_t_52_, lean_object* v_a_53_, lean_object* v_m_54_, lean_object* v_p_55_){
_start:
{
lean_object* v___f_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
lean_inc(v_a_53_);
v___f_56_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_pi___redArg___lam__0), 5, 4);
lean_closure_set(v___f_56_, 0, v_inst_51_);
lean_closure_set(v___f_56_, 1, v_m_54_);
lean_closure_set(v___f_56_, 2, v_a_53_);
lean_closure_set(v___f_56_, 3, v_p_55_);
v___x_57_ = lean_apply_1(v_t_52_, v_a_53_);
v___x_58_ = lp_mathlib_Multiset_bind___redArg(v___x_57_, v___f_56_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_pi___redArg(lean_object* v_inst_63_, lean_object* v_m_64_, lean_object* v_t_65_){
_start:
{
lean_object* v___f_66_; lean_object* v___x_67_; lean_object* v___x_68_; 
v___f_66_ = lean_alloc_closure((void*)(lp_mathlib_Multiset_pi___redArg___lam__1), 5, 2);
lean_closure_set(v___f_66_, 0, v_inst_63_);
lean_closure_set(v___f_66_, 1, v_t_65_);
v___x_67_ = ((lean_object*)(lp_mathlib_Multiset_pi___redArg___closed__1));
v___x_68_ = lp_mathlib_Multiset_rec___redArg(v___x_67_, v___f_66_, v_m_64_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_pi(lean_object* v_00_u03b1_69_, lean_object* v_inst_70_, lean_object* v_00_u03b2_71_, lean_object* v_m_72_, lean_object* v_t_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lp_mathlib_Multiset_pi___redArg(v_inst_70_, v_m_72_, v_t_73_);
return v___x_74_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Bind(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Multiset_Pi(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Bind(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Multiset_Pi(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Bind(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Multiset_Pi(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Bind(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Multiset_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Multiset_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Multiset_Pi(builtin);
}
#ifdef __cplusplus
}
#endif
