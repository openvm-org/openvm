// Lean compiler output
// Module: Mathlib.Data.Fintype.Pi
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Pi public import Mathlib.Data.Fintype.Basic public import Mathlib.Data.Set.Finite.Basic
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
uint8_t lp_mathlib_Fintype_decidableForallFintype___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_pi___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_fintype___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_ofEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_piFinset___redArg___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Fintype_piFinset___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fintype_piFinset___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fintype_piFinset___redArg___closed__0 = (const lean_object*)&lp_mathlib_Fintype_piFinset___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Fintype_piFinset___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_piFinset(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instFintype___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instFintype___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instFintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_RelHom_instFintype___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instFintype___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_RelHom_instFintype___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instFintype___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_RelHom_instFintype___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instFintype___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instFintype___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instFintype___redArg___lam__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instFintype___redArg___lam__4(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_RelHom_instFintype___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_RelHom_instFintype___redArg___lam__4, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_RelHom_instFintype___redArg___closed__0 = (const lean_object*)&lp_mathlib_RelHom_instFintype___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_RelHom_instFintype___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_RelHom_instFintype___redArg___closed__0_value),((lean_object*)&lp_mathlib_RelHom_instFintype___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_RelHom_instFintype___redArg___closed__1 = (const lean_object*)&lp_mathlib_RelHom_instFintype___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instFintype___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instFintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_piFinset___redArg___lam__0(lean_object* v_f_1_, lean_object* v_a_2_){
_start:
{
lean_object* v___x_3_; 
v___x_3_ = lean_apply_2(v_f_1_, v_a_2_, lean_box(0));
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_piFinset___redArg(lean_object* v_inst_5_, lean_object* v_inst_6_, lean_object* v_t_7_){
_start:
{
lean_object* v___f_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v___f_8_ = ((lean_object*)(lp_mathlib_Fintype_piFinset___redArg___closed__0));
v___x_9_ = lp_mathlib_Finset_pi___redArg(v_inst_5_, v_inst_6_, v_t_7_);
v___x_10_ = lp_mathlib_Finset_map___redArg(v___f_8_, v___x_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_piFinset(lean_object* v_00_u03b1_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_00_u03b4_14_, lean_object* v_t_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_mathlib_Fintype_piFinset___redArg(v_inst_12_, v_inst_13_, v_t_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instFintype___redArg___lam__0(lean_object* v_inst_17_, lean_object* v_x_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lean_apply_1(v_inst_17_, v_x_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instFintype___redArg(lean_object* v_inst_20_, lean_object* v_inst_21_, lean_object* v_inst_22_){
_start:
{
lean_object* v___f_23_; lean_object* v___x_24_; 
v___f_23_ = lean_alloc_closure((void*)(lp_mathlib_Pi_instFintype___redArg___lam__0), 2, 1);
lean_closure_set(v___f_23_, 0, v_inst_22_);
v___x_24_ = lp_mathlib_Fintype_piFinset___redArg(v_inst_20_, v_inst_21_, v___f_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instFintype(lean_object* v_00_u03b1_25_, lean_object* v_00_u03b2_26_, lean_object* v_inst_27_, lean_object* v_inst_28_, lean_object* v_inst_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_mathlib_Pi_instFintype___redArg(v_inst_27_, v_inst_28_, v_inst_29_);
return v___x_30_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_RelHom_instFintype___redArg___lam__0(lean_object* v_inst_31_, lean_object* v_a_32_, lean_object* v_a_33_, lean_object* v_inst_34_, lean_object* v_a_35_){
_start:
{
lean_object* v___x_36_; uint8_t v___x_37_; 
lean_inc(v_a_35_);
lean_inc(v_a_32_);
v___x_36_ = lean_apply_2(v_inst_31_, v_a_32_, v_a_35_);
v___x_37_ = lean_unbox(v___x_36_);
if (v___x_37_ == 0)
{
uint8_t v___x_38_; 
lean_dec(v_a_35_);
lean_dec_ref(v_inst_34_);
lean_dec(v_a_33_);
lean_dec(v_a_32_);
v___x_38_ = 1;
return v___x_38_;
}
else
{
lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; uint8_t v___x_42_; 
lean_inc(v_a_33_);
v___x_39_ = lean_apply_1(v_a_33_, v_a_32_);
v___x_40_ = lean_apply_1(v_a_33_, v_a_35_);
v___x_41_ = lean_apply_2(v_inst_34_, v___x_39_, v___x_40_);
v___x_42_ = lean_unbox(v___x_41_);
return v___x_42_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instFintype___redArg___lam__0___boxed(lean_object* v_inst_43_, lean_object* v_a_44_, lean_object* v_a_45_, lean_object* v_inst_46_, lean_object* v_a_47_){
_start:
{
uint8_t v_res_48_; lean_object* v_r_49_; 
v_res_48_ = lp_mathlib_RelHom_instFintype___redArg___lam__0(v_inst_43_, v_a_44_, v_a_45_, v_inst_46_, v_a_47_);
v_r_49_ = lean_box(v_res_48_);
return v_r_49_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_RelHom_instFintype___redArg___lam__1(lean_object* v_inst_50_, lean_object* v_a_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_a_54_){
_start:
{
lean_object* v___f_55_; uint8_t v___x_56_; 
v___f_55_ = lean_alloc_closure((void*)(lp_mathlib_RelHom_instFintype___redArg___lam__0___boxed), 5, 4);
lean_closure_set(v___f_55_, 0, v_inst_50_);
lean_closure_set(v___f_55_, 1, v_a_54_);
lean_closure_set(v___f_55_, 2, v_a_51_);
lean_closure_set(v___f_55_, 3, v_inst_52_);
v___x_56_ = lp_mathlib_Fintype_decidableForallFintype___redArg(v___f_55_, v_inst_53_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instFintype___redArg___lam__1___boxed(lean_object* v_inst_57_, lean_object* v_a_58_, lean_object* v_inst_59_, lean_object* v_inst_60_, lean_object* v_a_61_){
_start:
{
uint8_t v_res_62_; lean_object* v_r_63_; 
v_res_62_ = lp_mathlib_RelHom_instFintype___redArg___lam__1(v_inst_57_, v_a_58_, v_inst_59_, v_inst_60_, v_a_61_);
v_r_63_ = lean_box(v_res_62_);
return v_r_63_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_RelHom_instFintype___redArg___lam__2(lean_object* v_inst_64_, lean_object* v_inst_65_, lean_object* v_inst_66_, lean_object* v_a_67_){
_start:
{
lean_object* v___f_68_; uint8_t v___x_69_; 
lean_inc(v_inst_66_);
v___f_68_ = lean_alloc_closure((void*)(lp_mathlib_RelHom_instFintype___redArg___lam__1___boxed), 5, 4);
lean_closure_set(v___f_68_, 0, v_inst_64_);
lean_closure_set(v___f_68_, 1, v_a_67_);
lean_closure_set(v___f_68_, 2, v_inst_65_);
lean_closure_set(v___f_68_, 3, v_inst_66_);
v___x_69_ = lp_mathlib_Fintype_decidableForallFintype___redArg(v___f_68_, v_inst_66_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instFintype___redArg___lam__2___boxed(lean_object* v_inst_70_, lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_a_73_){
_start:
{
uint8_t v_res_74_; lean_object* v_r_75_; 
v_res_74_ = lp_mathlib_RelHom_instFintype___redArg___lam__2(v_inst_70_, v_inst_71_, v_inst_72_, v_a_73_);
v_r_75_ = lean_box(v_res_74_);
return v_r_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instFintype___redArg___lam__3(lean_object* v_inst_76_, lean_object* v_a_77_){
_start:
{
lean_inc(v_inst_76_);
return v_inst_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instFintype___redArg___lam__3___boxed(lean_object* v_inst_78_, lean_object* v_a_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_mathlib_RelHom_instFintype___redArg___lam__3(v_inst_78_, v_a_79_);
lean_dec(v_a_79_);
lean_dec(v_inst_78_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instFintype___redArg___lam__4(lean_object* v_f_81_, lean_object* v___y_82_){
_start:
{
lean_object* v___x_83_; 
v___x_83_ = lean_apply_1(v_f_81_, v___y_82_);
return v___x_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instFintype___redArg(lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_inst_89_, lean_object* v_inst_90_, lean_object* v_inst_91_){
_start:
{
lean_object* v___f_92_; lean_object* v___f_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
lean_inc(v_inst_87_);
v___f_92_ = lean_alloc_closure((void*)(lp_mathlib_RelHom_instFintype___redArg___lam__2___boxed), 4, 3);
lean_closure_set(v___f_92_, 0, v_inst_90_);
lean_closure_set(v___f_92_, 1, v_inst_91_);
lean_closure_set(v___f_92_, 2, v_inst_87_);
v___f_93_ = lean_alloc_closure((void*)(lp_mathlib_RelHom_instFintype___redArg___lam__3___boxed), 2, 1);
lean_closure_set(v___f_93_, 0, v_inst_88_);
v___x_94_ = lp_mathlib_Pi_instFintype___redArg(v_inst_89_, v_inst_87_, v___f_93_);
v___x_95_ = lp_mathlib_Subtype_fintype___redArg(v___f_92_, v___x_94_);
v___x_96_ = ((lean_object*)(lp_mathlib_RelHom_instFintype___redArg___closed__1));
v___x_97_ = lp_mathlib_Fintype_ofEquiv___redArg(v___x_95_, v___x_96_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RelHom_instFintype(lean_object* v_00_u03b1_98_, lean_object* v_00_u03b2_99_, lean_object* v_inst_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_r_103_, lean_object* v_s_104_, lean_object* v_inst_105_, lean_object* v_inst_106_){
_start:
{
lean_object* v___x_107_; 
v___x_107_ = lp_mathlib_RelHom_instFintype___redArg(v_inst_100_, v_inst_101_, v_inst_102_, v_inst_105_, v_inst_106_);
return v___x_107_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Pi(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Pi(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fintype_Pi(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Pi(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fintype_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fintype_Pi(builtin);
}
#ifdef __cplusplus
}
#endif
