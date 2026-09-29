// Lean compiler output
// Module: Mathlib.Data.List.Monad
// Imports: public import Init public meta import Init public import Mathlib.Init public import Batteries.Control.AlternativeMonad
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l___private_Init_Data_List_Impl_0__List_flatMapTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Function_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__3(lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_List_instMonad___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_List_instMonad___lam__4___closed__0 = (const lean_object*)&lp_mathlib_List_instMonad___lam__4___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__9(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__9___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__10(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_List_instMonad___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_instMonad___closed__0 = (const lean_object*)&lp_mathlib_List_instMonad___closed__0_value;
static const lean_closure_object lp_mathlib_List_instMonad___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_instMonad___lam__1, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_List_instMonad___closed__0_value)} };
static const lean_object* lp_mathlib_List_instMonad___closed__1 = (const lean_object*)&lp_mathlib_List_instMonad___closed__1_value;
static const lean_closure_object lp_mathlib_List_instMonad___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_instMonad___lam__2, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_instMonad___closed__2 = (const lean_object*)&lp_mathlib_List_instMonad___closed__2_value;
static const lean_closure_object lp_mathlib_List_instMonad___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_instMonad___lam__4, .m_arity = 5, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_List_instMonad___closed__0_value)} };
static const lean_object* lp_mathlib_List_instMonad___closed__3 = (const lean_object*)&lp_mathlib_List_instMonad___closed__3_value;
static const lean_closure_object lp_mathlib_List_instMonad___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_instMonad___lam__5, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_instMonad___closed__4 = (const lean_object*)&lp_mathlib_List_instMonad___closed__4_value;
static const lean_closure_object lp_mathlib_List_instMonad___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_instMonad___lam__8, .m_arity = 6, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_List_instMonad___closed__2_value),((lean_object*)&lp_mathlib_List_instMonad___closed__4_value)} };
static const lean_object* lp_mathlib_List_instMonad___closed__5 = (const lean_object*)&lp_mathlib_List_instMonad___closed__5_value;
static const lean_closure_object lp_mathlib_List_instMonad___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_instMonad___lam__10, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_instMonad___closed__6 = (const lean_object*)&lp_mathlib_List_instMonad___closed__6_value;
static const lean_ctor_object lp_mathlib_List_instMonad___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_List_instMonad___closed__0_value),((lean_object*)&lp_mathlib_List_instMonad___closed__1_value)}};
static const lean_object* lp_mathlib_List_instMonad___closed__7 = (const lean_object*)&lp_mathlib_List_instMonad___closed__7_value;
static const lean_ctor_object lp_mathlib_List_instMonad___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_List_instMonad___closed__7_value),((lean_object*)&lp_mathlib_List_instMonad___closed__2_value),((lean_object*)&lp_mathlib_List_instMonad___closed__3_value),((lean_object*)&lp_mathlib_List_instMonad___closed__5_value),((lean_object*)&lp_mathlib_List_instMonad___closed__6_value)}};
static const lean_object* lp_mathlib_List_instMonad___closed__8 = (const lean_object*)&lp_mathlib_List_instMonad___closed__8_value;
static const lean_ctor_object lp_mathlib_List_instMonad___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_List_instMonad___closed__8_value),((lean_object*)&lp_mathlib_List_instMonad___closed__4_value)}};
static const lean_object* lp_mathlib_List_instMonad___closed__9 = (const lean_object*)&lp_mathlib_List_instMonad___closed__9_value;
LEAN_EXPORT const lean_object* lp_mathlib_List_instMonad = (const lean_object*)&lp_mathlib_List_instMonad___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_List_instAlternativeMonad__mathlib___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_instAlternativeMonad__mathlib___lam__1(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_List_instAlternativeMonad__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_instAlternativeMonad__mathlib___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_instAlternativeMonad__mathlib___closed__0 = (const lean_object*)&lp_mathlib_List_instAlternativeMonad__mathlib___closed__0_value;
static const lean_closure_object lp_mathlib_List_instAlternativeMonad__mathlib___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_instAlternativeMonad__mathlib___lam__1, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_instAlternativeMonad__mathlib___closed__1 = (const lean_object*)&lp_mathlib_List_instAlternativeMonad__mathlib___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_List_instAlternativeMonad__mathlib;
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__0(lean_object* v_00_u03b1_1_, lean_object* v_00_u03b2_2_, lean_object* v_f_3_, lean_object* v_l_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = lean_box(0);
v___x_6_ = l_List_mapTR_loop___redArg(v_f_3_, v_l_4_, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__1(lean_object* v___f_7_, lean_object* v_00_u03b1_8_, lean_object* v_00_u03b2_9_, lean_object* v___y_10_, lean_object* v___y_11_){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_12_ = lean_alloc_closure((void*)(l_Function_const___boxed), 4, 3);
lean_closure_set(v___x_12_, 0, lean_box(0));
lean_closure_set(v___x_12_, 1, lean_box(0));
lean_closure_set(v___x_12_, 2, v___y_10_);
v___x_13_ = lean_apply_4(v___f_7_, lean_box(0), lean_box(0), v___x_12_, v___y_11_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__2(lean_object* v_00_u03b1_14_, lean_object* v_x_15_){
_start:
{
lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_16_ = lean_box(0);
v___x_17_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_17_, 0, v_x_15_);
lean_ctor_set(v___x_17_, 1, v___x_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__3(lean_object* v_x_18_, lean_object* v___f_19_, lean_object* v_y_20_){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_21_ = lean_box(0);
v___x_22_ = lean_apply_1(v_x_18_, v___x_21_);
v___x_23_ = lean_apply_4(v___f_19_, lean_box(0), lean_box(0), v_y_20_, v___x_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__4(lean_object* v___f_26_, lean_object* v_00_u03b1_27_, lean_object* v_00_u03b2_28_, lean_object* v_f_29_, lean_object* v_x_30_){
_start:
{
lean_object* v___f_31_; lean_object* v___x_32_; lean_object* v___x_33_; 
v___f_31_ = lean_alloc_closure((void*)(lp_mathlib_List_instMonad___lam__3), 3, 2);
lean_closure_set(v___f_31_, 0, v_x_30_);
lean_closure_set(v___f_31_, 1, v___f_26_);
v___x_32_ = ((lean_object*)(lp_mathlib_List_instMonad___lam__4___closed__0));
v___x_33_ = l___private_Init_Data_List_Impl_0__List_flatMapTR_go(lean_box(0), lean_box(0), v___f_31_, v_f_29_, v___x_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__5(lean_object* v_00_u03b1_34_, lean_object* v_00_u03b2_35_, lean_object* v_l_36_, lean_object* v_f_37_){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_38_ = ((lean_object*)(lp_mathlib_List_instMonad___lam__4___closed__0));
v___x_39_ = l___private_Init_Data_List_Impl_0__List_flatMapTR_go(lean_box(0), lean_box(0), v_f_37_, v_l_36_, v___x_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__6(lean_object* v___f_40_, lean_object* v_a_41_, lean_object* v_x_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lean_apply_2(v___f_40_, lean_box(0), v_a_41_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__6___boxed(lean_object* v___f_44_, lean_object* v_a_45_, lean_object* v_x_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_List_instMonad___lam__6(v___f_44_, v_a_45_, v_x_46_);
lean_dec(v_x_46_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__7(lean_object* v___f_48_, lean_object* v_y_49_, lean_object* v___f_50_, lean_object* v_a_51_){
_start:
{
lean_object* v___f_52_; lean_object* v___x_53_; lean_object* v___x_54_; lean_object* v___x_55_; 
v___f_52_ = lean_alloc_closure((void*)(lp_mathlib_List_instMonad___lam__6___boxed), 3, 2);
lean_closure_set(v___f_52_, 0, v___f_48_);
lean_closure_set(v___f_52_, 1, v_a_51_);
v___x_53_ = lean_box(0);
v___x_54_ = lean_apply_1(v_y_49_, v___x_53_);
v___x_55_ = lean_apply_4(v___f_50_, lean_box(0), lean_box(0), v___x_54_, v___f_52_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__8(lean_object* v___f_56_, lean_object* v___f_57_, lean_object* v_00_u03b1_58_, lean_object* v_00_u03b2_59_, lean_object* v_x_60_, lean_object* v_y_61_){
_start:
{
lean_object* v___f_62_; lean_object* v___x_63_; 
lean_inc_ref(v___f_57_);
v___f_62_ = lean_alloc_closure((void*)(lp_mathlib_List_instMonad___lam__7), 4, 3);
lean_closure_set(v___f_62_, 0, v___f_56_);
lean_closure_set(v___f_62_, 1, v_y_61_);
lean_closure_set(v___f_62_, 2, v___f_57_);
v___x_63_ = lean_apply_4(v___f_57_, lean_box(0), lean_box(0), v_x_60_, v___f_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__9(lean_object* v_y_64_, lean_object* v_x_65_){
_start:
{
lean_object* v___x_66_; lean_object* v___x_67_; 
v___x_66_ = lean_box(0);
v___x_67_ = lean_apply_1(v_y_64_, v___x_66_);
return v___x_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__9___boxed(lean_object* v_y_68_, lean_object* v_x_69_){
_start:
{
lean_object* v_res_70_; 
v_res_70_ = lp_mathlib_List_instMonad___lam__9(v_y_68_, v_x_69_);
lean_dec(v_x_69_);
return v_res_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instMonad___lam__10(lean_object* v_00_u03b1_71_, lean_object* v_00_u03b2_72_, lean_object* v_x_73_, lean_object* v_y_74_){
_start:
{
lean_object* v___f_75_; lean_object* v___x_76_; lean_object* v___x_77_; 
v___f_75_ = lean_alloc_closure((void*)(lp_mathlib_List_instMonad___lam__9___boxed), 2, 1);
lean_closure_set(v___f_75_, 0, v_y_74_);
v___x_76_ = ((lean_object*)(lp_mathlib_List_instMonad___lam__4___closed__0));
v___x_77_ = l___private_Init_Data_List_Impl_0__List_flatMapTR_go(lean_box(0), lean_box(0), v___f_75_, v_x_73_, v___x_76_);
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instAlternativeMonad__mathlib___lam__0(lean_object* v_00_u03b1_102_){
_start:
{
lean_object* v___x_103_; 
v___x_103_ = lean_box(0);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_instAlternativeMonad__mathlib___lam__1(lean_object* v_00_u03b1_104_, lean_object* v_l_105_, lean_object* v_l_x27_106_){
_start:
{
lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; 
v___x_107_ = lean_box(0);
v___x_108_ = lean_apply_1(v_l_x27_106_, v___x_107_);
v___x_109_ = l_List_appendTR___redArg(v_l_105_, v___x_108_);
return v___x_109_;
}
}
static lean_object* _init_lp_mathlib_List_instAlternativeMonad__mathlib(void){
_start:
{
lean_object* v___x_112_; lean_object* v_toApplicative_113_; lean_object* v___f_114_; lean_object* v___f_115_; lean_object* v___f_116_; lean_object* v___x_117_; lean_object* v___x_118_; 
v___x_112_ = ((lean_object*)(lp_mathlib_List_instMonad));
v_toApplicative_113_ = lean_ctor_get(v___x_112_, 0);
v___f_114_ = ((lean_object*)(lp_mathlib_List_instAlternativeMonad__mathlib___closed__0));
v___f_115_ = ((lean_object*)(lp_mathlib_List_instAlternativeMonad__mathlib___closed__1));
v___f_116_ = ((lean_object*)(lp_mathlib_List_instMonad___closed__4));
lean_inc_ref(v_toApplicative_113_);
v___x_117_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_117_, 0, v_toApplicative_113_);
lean_ctor_set(v___x_117_, 1, v___f_114_);
lean_ctor_set(v___x_117_, 2, v___f_115_);
v___x_118_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_118_, 0, v___x_117_);
lean_ctor_set(v___x_118_, 1, v___f_116_);
return v___x_118_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Control_AlternativeMonad(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_List_Monad(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Control_AlternativeMonad(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_List_instAlternativeMonad__mathlib = _init_lp_mathlib_List_instAlternativeMonad__mathlib();
lean_mark_persistent(lp_mathlib_List_instAlternativeMonad__mathlib);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_List_Monad(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Control_AlternativeMonad(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_List_Monad(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Control_AlternativeMonad(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_List_Monad(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_List_Monad(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_List_Monad(builtin);
}
#ifdef __cplusplus
}
#endif
