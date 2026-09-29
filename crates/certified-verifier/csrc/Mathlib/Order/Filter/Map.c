// Lean compiler output
// Module: Mathlib.Order.Filter.Map
// Imports: public import Init public meta import Init public import Mathlib.Control.Basic public import Mathlib.Data.Set.Lattice.Image public import Mathlib.Order.Filter.Basic
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
lean_object* lp_mathlib_Filter_bind___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Filter_instPure___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Filter_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Filter_instFunctor;
LEAN_EXPORT lean_object* lp_mathlib_Filter_monad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_monad___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_monad___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_monad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_monad___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_monad___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Filter_monad___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_monad___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_monad___closed__0 = (const lean_object*)&lp_mathlib_Filter_monad___closed__0_value;
static const lean_closure_object lp_mathlib_Filter_monad___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_monad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_monad___closed__1 = (const lean_object*)&lp_mathlib_Filter_monad___closed__1_value;
static const lean_closure_object lp_mathlib_Filter_monad___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_monad___lam__2___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_monad___closed__2 = (const lean_object*)&lp_mathlib_Filter_monad___closed__2_value;
static const lean_closure_object lp_mathlib_Filter_monad___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_instPure___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_monad___closed__3 = (const lean_object*)&lp_mathlib_Filter_monad___closed__3_value;
static const lean_closure_object lp_mathlib_Filter_monad___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_bind___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_monad___closed__4 = (const lean_object*)&lp_mathlib_Filter_monad___closed__4_value;
static const lean_closure_object lp_mathlib_Filter_monad___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_map___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_monad___closed__5 = (const lean_object*)&lp_mathlib_Filter_monad___closed__5_value;
static const lean_ctor_object lp_mathlib_Filter_monad___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Filter_monad___closed__5_value),((lean_object*)&lp_mathlib_Filter_monad___closed__0_value)}};
static const lean_object* lp_mathlib_Filter_monad___closed__6 = (const lean_object*)&lp_mathlib_Filter_monad___closed__6_value;
static const lean_ctor_object lp_mathlib_Filter_monad___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Filter_monad___closed__6_value),((lean_object*)&lp_mathlib_Filter_monad___closed__3_value),((lean_object*)&lp_mathlib_Filter_monad___closed__1_value),((lean_object*)&lp_mathlib_Filter_monad___closed__2_value),((lean_object*)&lp_mathlib_Filter_monad___closed__2_value)}};
static const lean_object* lp_mathlib_Filter_monad___closed__7 = (const lean_object*)&lp_mathlib_Filter_monad___closed__7_value;
static const lean_ctor_object lp_mathlib_Filter_monad___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Filter_monad___closed__7_value),((lean_object*)&lp_mathlib_Filter_monad___closed__4_value)}};
static const lean_object* lp_mathlib_Filter_monad___closed__8 = (const lean_object*)&lp_mathlib_Filter_monad___closed__8_value;
LEAN_EXPORT const lean_object* lp_mathlib_Filter_monad = (const lean_object*)&lp_mathlib_Filter_monad___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Filter_instAlternative___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instAlternative___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instAlternative___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instAlternative___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instAlternative___lam__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instAlternative___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_instAlternative___lam__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Filter_instAlternative___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_instAlternative___lam__0___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_instAlternative___closed__0 = (const lean_object*)&lp_mathlib_Filter_instAlternative___closed__0_value;
static const lean_closure_object lp_mathlib_Filter_instAlternative___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_instAlternative___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_instAlternative___closed__1 = (const lean_object*)&lp_mathlib_Filter_instAlternative___closed__1_value;
static const lean_closure_object lp_mathlib_Filter_instAlternative___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_instAlternative___lam__3, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_instAlternative___closed__2 = (const lean_object*)&lp_mathlib_Filter_instAlternative___closed__2_value;
static const lean_closure_object lp_mathlib_Filter_instAlternative___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Filter_instAlternative___lam__2___boxed, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Filter_instAlternative___closed__3 = (const lean_object*)&lp_mathlib_Filter_instAlternative___closed__3_value;
static lean_once_cell_t lp_mathlib_Filter_instAlternative___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_instAlternative___closed__4;
static lean_once_cell_t lp_mathlib_Filter_instAlternative___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Filter_instAlternative___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Filter_instAlternative;
LEAN_EXPORT lean_object* lp_mathlib_Filter_kernMap(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_kernMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Filter_monad___lam__0(lean_object* v_00_u03b1_1_, lean_object* v_00_u03b2_2_, lean_object* v___y_3_, lean_object* v___y_4_){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_box(0);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_monad___lam__0___boxed(lean_object* v_00_u03b1_6_, lean_object* v_00_u03b2_7_, lean_object* v___y_8_, lean_object* v___y_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib_Filter_monad___lam__0(v_00_u03b1_6_, v_00_u03b2_7_, v___y_8_, v___y_9_);
lean_dec(v___y_8_);
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_monad___lam__1(lean_object* v_00_u03b1_11_, lean_object* v_00_u03b2_12_, lean_object* v_f_13_, lean_object* v_x_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lean_box(0);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_monad___lam__1___boxed(lean_object* v_00_u03b1_16_, lean_object* v_00_u03b2_17_, lean_object* v_f_18_, lean_object* v_x_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_Filter_monad___lam__1(v_00_u03b1_16_, v_00_u03b2_17_, v_f_18_, v_x_19_);
lean_dec_ref(v_x_19_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_monad___lam__2(lean_object* v_00_u03b1_21_, lean_object* v_00_u03b2_22_, lean_object* v_x_23_, lean_object* v_y_24_){
_start:
{
lean_object* v___x_25_; 
v___x_25_ = lean_box(0);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_monad___lam__2___boxed(lean_object* v_00_u03b1_26_, lean_object* v_00_u03b2_27_, lean_object* v_x_28_, lean_object* v_y_29_){
_start:
{
lean_object* v_res_30_; 
v_res_30_ = lp_mathlib_Filter_monad___lam__2(v_00_u03b1_26_, v_00_u03b2_27_, v_x_28_, v_y_29_);
lean_dec_ref(v_y_29_);
return v_res_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instAlternative___lam__0(lean_object* v_00_u03b1_49_, lean_object* v_00_u03b2_50_, lean_object* v_x_51_, lean_object* v_y_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lean_box(0);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instAlternative___lam__0___boxed(lean_object* v_00_u03b1_54_, lean_object* v_00_u03b2_55_, lean_object* v_x_56_, lean_object* v_y_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib_Filter_instAlternative___lam__0(v_00_u03b1_54_, v_00_u03b2_55_, v_x_56_, v_y_57_);
lean_dec_ref(v_y_57_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instAlternative___lam__1(lean_object* v_00_u03b1_59_, lean_object* v_00_u03b2_60_, lean_object* v_a_61_, lean_object* v_b_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lean_box(0);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instAlternative___lam__1___boxed(lean_object* v_00_u03b1_64_, lean_object* v_00_u03b2_65_, lean_object* v_a_66_, lean_object* v_b_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_Filter_instAlternative___lam__1(v_00_u03b1_64_, v_00_u03b2_65_, v_a_66_, v_b_67_);
lean_dec_ref(v_b_67_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instAlternative___lam__3(lean_object* v_00_u03b1_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lean_box(0);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instAlternative___lam__2(lean_object* v_00_u03b1_71_, lean_object* v_x_72_, lean_object* v_y_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lean_box(0);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_instAlternative___lam__2___boxed(lean_object* v_00_u03b1_75_, lean_object* v_x_76_, lean_object* v_y_77_){
_start:
{
lean_object* v_res_78_; 
v_res_78_ = lp_mathlib_Filter_instAlternative___lam__2(v_00_u03b1_75_, v_x_76_, v_y_77_);
lean_dec_ref(v_y_77_);
return v_res_78_;
}
}
static lean_object* _init_lp_mathlib_Filter_instAlternative___closed__4(void){
_start:
{
lean_object* v___f_83_; lean_object* v___f_84_; lean_object* v___f_85_; lean_object* v___x_86_; lean_object* v___x_87_; 
v___f_83_ = ((lean_object*)(lp_mathlib_Filter_instAlternative___closed__1));
v___f_84_ = ((lean_object*)(lp_mathlib_Filter_instAlternative___closed__0));
v___f_85_ = ((lean_object*)(lp_mathlib_Filter_monad___closed__3));
v___x_86_ = lp_mathlib_Filter_instFunctor;
v___x_87_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_87_, 0, v___x_86_);
lean_ctor_set(v___x_87_, 1, v___f_85_);
lean_ctor_set(v___x_87_, 2, v___f_84_);
lean_ctor_set(v___x_87_, 3, v___f_83_);
lean_ctor_set(v___x_87_, 4, v___f_83_);
return v___x_87_;
}
}
static lean_object* _init_lp_mathlib_Filter_instAlternative___closed__5(void){
_start:
{
lean_object* v___f_88_; lean_object* v___f_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
v___f_88_ = ((lean_object*)(lp_mathlib_Filter_instAlternative___closed__3));
v___f_89_ = ((lean_object*)(lp_mathlib_Filter_instAlternative___closed__2));
v___x_90_ = lean_obj_once(&lp_mathlib_Filter_instAlternative___closed__4, &lp_mathlib_Filter_instAlternative___closed__4_once, _init_lp_mathlib_Filter_instAlternative___closed__4);
v___x_91_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_91_, 0, v___x_90_);
lean_ctor_set(v___x_91_, 1, v___f_89_);
lean_ctor_set(v___x_91_, 2, v___f_88_);
return v___x_91_;
}
}
static lean_object* _init_lp_mathlib_Filter_instAlternative(void){
_start:
{
lean_object* v___x_92_; 
v___x_92_ = lean_obj_once(&lp_mathlib_Filter_instAlternative___closed__5, &lp_mathlib_Filter_instAlternative___closed__5_once, _init_lp_mathlib_Filter_instAlternative___closed__5);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_kernMap(lean_object* v_00_u03b1_93_, lean_object* v_00_u03b2_94_, lean_object* v_m_95_, lean_object* v_f_96_){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = lean_box(0);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Filter_kernMap___boxed(lean_object* v_00_u03b1_98_, lean_object* v_00_u03b2_99_, lean_object* v_m_100_, lean_object* v_f_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_Filter_kernMap(v_00_u03b1_98_, v_00_u03b2_99_, v_m_100_, v_f_101_);
lean_dec(v_m_100_);
return v_res_102_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Image(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Filter_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Filter_Map(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Filter_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Filter_instAlternative = _init_lp_mathlib_Filter_instAlternative();
lean_mark_persistent(lp_mathlib_Filter_instAlternative);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Filter_Map(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Lattice_Image(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Filter_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Filter_Map(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Lattice_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Filter_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Filter_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Filter_Map(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Filter_Map(builtin);
}
#ifdef __cplusplus
}
#endif
