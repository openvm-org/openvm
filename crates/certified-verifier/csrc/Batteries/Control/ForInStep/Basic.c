// Lean compiler output
// Module: Batteries.Control.ForInStep.Basic
// Imports: public import Init public meta import Init
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
LEAN_EXPORT lean_object* lp_batteries_ForInStep_bind___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_ForInStep_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_ForInStep_bindM___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_ForInStep_bindM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_ForInStep_bindM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_ForInStep_run___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_ForInStep_run___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_ForInStep_run(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_ForInStep_run___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_ForInStep_bindList___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_ForInStep_bindList___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_ForInStep_bindList(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_ForInStep_bind___redArg(lean_object* v_inst_1_, lean_object* v_a_2_, lean_object* v_f_3_){
_start:
{
lean_object* v_toApplicative_4_; 
v_toApplicative_4_ = lean_ctor_get(v_inst_1_, 0);
lean_inc_ref(v_toApplicative_4_);
lean_dec_ref(v_inst_1_);
if (lean_obj_tag(v_a_2_) == 0)
{
lean_object* v_toPure_5_; lean_object* v___x_6_; 
lean_dec(v_f_3_);
v_toPure_5_ = lean_ctor_get(v_toApplicative_4_, 1);
lean_inc(v_toPure_5_);
lean_dec_ref(v_toApplicative_4_);
v___x_6_ = lean_apply_2(v_toPure_5_, lean_box(0), v_a_2_);
return v___x_6_;
}
else
{
lean_object* v_a_7_; lean_object* v___x_8_; 
lean_dec_ref(v_toApplicative_4_);
v_a_7_ = lean_ctor_get(v_a_2_, 0);
lean_inc(v_a_7_);
lean_dec_ref_known(v_a_2_, 1);
v___x_8_ = lean_apply_1(v_f_3_, v_a_7_);
return v___x_8_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_ForInStep_bind(lean_object* v_m_9_, lean_object* v_00_u03b1_10_, lean_object* v_inst_11_, lean_object* v_a_12_, lean_object* v_f_13_){
_start:
{
lean_object* v_toApplicative_14_; 
v_toApplicative_14_ = lean_ctor_get(v_inst_11_, 0);
lean_inc_ref(v_toApplicative_14_);
lean_dec_ref(v_inst_11_);
if (lean_obj_tag(v_a_12_) == 0)
{
lean_object* v_toPure_15_; lean_object* v___x_16_; 
lean_dec(v_f_13_);
v_toPure_15_ = lean_ctor_get(v_toApplicative_14_, 1);
lean_inc(v_toPure_15_);
lean_dec_ref(v_toApplicative_14_);
v___x_16_ = lean_apply_2(v_toPure_15_, lean_box(0), v_a_12_);
return v___x_16_;
}
else
{
lean_object* v_a_17_; lean_object* v___x_18_; 
lean_dec_ref(v_toApplicative_14_);
v_a_17_ = lean_ctor_get(v_a_12_, 0);
lean_inc(v_a_17_);
lean_dec_ref_known(v_a_12_, 1);
v___x_18_ = lean_apply_1(v_f_13_, v_a_17_);
return v___x_18_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_ForInStep_bindM___redArg___lam__0(lean_object* v_toApplicative_19_, lean_object* v_f_20_, lean_object* v_x_21_){
_start:
{
if (lean_obj_tag(v_x_21_) == 0)
{
lean_object* v_toPure_22_; lean_object* v___x_23_; 
lean_dec(v_f_20_);
v_toPure_22_ = lean_ctor_get(v_toApplicative_19_, 1);
lean_inc(v_toPure_22_);
lean_dec_ref(v_toApplicative_19_);
v___x_23_ = lean_apply_2(v_toPure_22_, lean_box(0), v_x_21_);
return v___x_23_;
}
else
{
lean_object* v_a_24_; lean_object* v___x_25_; 
lean_dec_ref(v_toApplicative_19_);
v_a_24_ = lean_ctor_get(v_x_21_, 0);
lean_inc(v_a_24_);
lean_dec_ref_known(v_x_21_, 1);
v___x_25_ = lean_apply_1(v_f_20_, v_a_24_);
return v___x_25_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_ForInStep_bindM___redArg(lean_object* v_inst_26_, lean_object* v_a_27_, lean_object* v_f_28_){
_start:
{
lean_object* v_toApplicative_29_; lean_object* v_toBind_30_; lean_object* v___f_31_; lean_object* v___x_32_; 
v_toApplicative_29_ = lean_ctor_get(v_inst_26_, 0);
lean_inc_ref(v_toApplicative_29_);
v_toBind_30_ = lean_ctor_get(v_inst_26_, 1);
lean_inc(v_toBind_30_);
lean_dec_ref(v_inst_26_);
v___f_31_ = lean_alloc_closure((void*)(lp_batteries_ForInStep_bindM___redArg___lam__0), 3, 2);
lean_closure_set(v___f_31_, 0, v_toApplicative_29_);
lean_closure_set(v___f_31_, 1, v_f_28_);
v___x_32_ = lean_apply_4(v_toBind_30_, lean_box(0), lean_box(0), v_a_27_, v___f_31_);
return v___x_32_;
}
}
LEAN_EXPORT lean_object* lp_batteries_ForInStep_bindM(lean_object* v_m_33_, lean_object* v_00_u03b1_34_, lean_object* v_inst_35_, lean_object* v_a_36_, lean_object* v_f_37_){
_start:
{
lean_object* v_toApplicative_38_; lean_object* v_toBind_39_; lean_object* v___f_40_; lean_object* v___x_41_; 
v_toApplicative_38_ = lean_ctor_get(v_inst_35_, 0);
lean_inc_ref(v_toApplicative_38_);
v_toBind_39_ = lean_ctor_get(v_inst_35_, 1);
lean_inc(v_toBind_39_);
lean_dec_ref(v_inst_35_);
v___f_40_ = lean_alloc_closure((void*)(lp_batteries_ForInStep_bindM___redArg___lam__0), 3, 2);
lean_closure_set(v___f_40_, 0, v_toApplicative_38_);
lean_closure_set(v___f_40_, 1, v_f_37_);
v___x_41_ = lean_apply_4(v_toBind_39_, lean_box(0), lean_box(0), v_a_36_, v___f_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_batteries_ForInStep_run___redArg(lean_object* v_x_42_){
_start:
{
lean_object* v_a_43_; 
v_a_43_ = lean_ctor_get(v_x_42_, 0);
lean_inc(v_a_43_);
return v_a_43_;
}
}
LEAN_EXPORT lean_object* lp_batteries_ForInStep_run___redArg___boxed(lean_object* v_x_44_){
_start:
{
lean_object* v_res_45_; 
v_res_45_ = lp_batteries_ForInStep_run___redArg(v_x_44_);
lean_dec_ref(v_x_44_);
return v_res_45_;
}
}
LEAN_EXPORT lean_object* lp_batteries_ForInStep_run(lean_object* v_00_u03b1_46_, lean_object* v_x_47_){
_start:
{
lean_object* v_a_48_; 
v_a_48_ = lean_ctor_get(v_x_47_, 0);
lean_inc(v_a_48_);
return v_a_48_;
}
}
LEAN_EXPORT lean_object* lp_batteries_ForInStep_run___boxed(lean_object* v_00_u03b1_49_, lean_object* v_x_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_batteries_ForInStep_run(v_00_u03b1_49_, v_x_50_);
lean_dec_ref(v_x_50_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_batteries_ForInStep_bindList___redArg(lean_object* v_inst_52_, lean_object* v_f_53_, lean_object* v_x_54_, lean_object* v_x_55_){
_start:
{
if (lean_obj_tag(v_x_54_) == 0)
{
lean_object* v_toApplicative_56_; lean_object* v_toPure_57_; lean_object* v___x_58_; 
v_toApplicative_56_ = lean_ctor_get(v_inst_52_, 0);
lean_inc_ref(v_toApplicative_56_);
lean_dec(v_f_53_);
lean_dec_ref(v_inst_52_);
v_toPure_57_ = lean_ctor_get(v_toApplicative_56_, 1);
lean_inc(v_toPure_57_);
lean_dec_ref(v_toApplicative_56_);
v___x_58_ = lean_apply_2(v_toPure_57_, lean_box(0), v_x_55_);
return v___x_58_;
}
else
{
if (lean_obj_tag(v_x_55_) == 0)
{
lean_object* v_toApplicative_59_; lean_object* v_toPure_60_; lean_object* v___x_61_; 
v_toApplicative_59_ = lean_ctor_get(v_inst_52_, 0);
lean_inc_ref(v_toApplicative_59_);
lean_dec_ref_known(v_x_54_, 2);
lean_dec(v_f_53_);
lean_dec_ref(v_inst_52_);
v_toPure_60_ = lean_ctor_get(v_toApplicative_59_, 1);
lean_inc(v_toPure_60_);
lean_dec_ref(v_toApplicative_59_);
v___x_61_ = lean_apply_2(v_toPure_60_, lean_box(0), v_x_55_);
return v___x_61_;
}
else
{
lean_object* v_toBind_62_; lean_object* v_head_63_; lean_object* v_tail_64_; lean_object* v_a_65_; lean_object* v___f_66_; lean_object* v___x_67_; lean_object* v___x_68_; 
v_toBind_62_ = lean_ctor_get(v_inst_52_, 1);
lean_inc(v_toBind_62_);
v_head_63_ = lean_ctor_get(v_x_54_, 0);
lean_inc(v_head_63_);
v_tail_64_ = lean_ctor_get(v_x_54_, 1);
lean_inc(v_tail_64_);
lean_dec_ref_known(v_x_54_, 2);
v_a_65_ = lean_ctor_get(v_x_55_, 0);
lean_inc(v_a_65_);
lean_dec_ref_known(v_x_55_, 1);
lean_inc(v_f_53_);
v___f_66_ = lean_alloc_closure((void*)(lp_batteries_ForInStep_bindList___redArg___lam__0), 4, 3);
lean_closure_set(v___f_66_, 0, v_inst_52_);
lean_closure_set(v___f_66_, 1, v_f_53_);
lean_closure_set(v___f_66_, 2, v_tail_64_);
v___x_67_ = lean_apply_2(v_f_53_, v_head_63_, v_a_65_);
v___x_68_ = lean_apply_4(v_toBind_62_, lean_box(0), lean_box(0), v___x_67_, v___f_66_);
return v___x_68_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_ForInStep_bindList___redArg___lam__0(lean_object* v_inst_69_, lean_object* v_f_70_, lean_object* v_tail_71_, lean_object* v_x_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lp_batteries_ForInStep_bindList___redArg(v_inst_69_, v_f_70_, v_tail_71_, v_x_72_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_batteries_ForInStep_bindList(lean_object* v_m_74_, lean_object* v_00_u03b1_75_, lean_object* v_00_u03b2_76_, lean_object* v_inst_77_, lean_object* v_f_78_, lean_object* v_x_79_, lean_object* v_x_80_){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lp_batteries_ForInStep_bindList___redArg(v_inst_77_, v_f_78_, v_x_79_, v_x_80_);
return v___x_81_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Control_ForInStep_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Control_ForInStep_Basic(uint8_t builtin) {
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
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Control_ForInStep_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Control_ForInStep_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Control_ForInStep_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Control_ForInStep_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
