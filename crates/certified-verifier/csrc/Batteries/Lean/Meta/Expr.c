// Lean compiler output
// Module: Batteries.Lean.Meta.Expr
// Imports: public import Init public meta import Init public import Lean.Expr
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
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_string_compare(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_Literal_instOrd__batteries___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Literal_instOrd__batteries___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Lean_Literal_instOrd__batteries___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_Literal_instOrd__batteries___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_Literal_instOrd__batteries___closed__0 = (const lean_object*)&lp_batteries_Lean_Literal_instOrd__batteries___closed__0_value;
LEAN_EXPORT const lean_object* lp_batteries_Lean_Literal_instOrd__batteries = (const lean_object*)&lp_batteries_Lean_Literal_instOrd__batteries___closed__0_value;
LEAN_EXPORT uint8_t lp_batteries_Lean_Literal_instOrd__batteries___lam__0(lean_object* v_x_1_, lean_object* v_x_2_){
_start:
{
if (lean_obj_tag(v_x_1_) == 0)
{
if (lean_obj_tag(v_x_2_) == 0)
{
lean_object* v_val_3_; lean_object* v_val_4_; uint8_t v___x_5_; 
v_val_3_ = lean_ctor_get(v_x_1_, 0);
v_val_4_ = lean_ctor_get(v_x_2_, 0);
v___x_5_ = lean_nat_dec_lt(v_val_3_, v_val_4_);
if (v___x_5_ == 0)
{
uint8_t v___x_6_; 
v___x_6_ = lean_nat_dec_eq(v_val_3_, v_val_4_);
if (v___x_6_ == 0)
{
uint8_t v___x_7_; 
v___x_7_ = 2;
return v___x_7_;
}
else
{
uint8_t v___x_8_; 
v___x_8_ = 1;
return v___x_8_;
}
}
else
{
uint8_t v___x_9_; 
v___x_9_ = 0;
return v___x_9_;
}
}
else
{
uint8_t v___x_10_; 
v___x_10_ = 0;
return v___x_10_;
}
}
else
{
if (lean_obj_tag(v_x_2_) == 0)
{
uint8_t v___x_11_; 
v___x_11_ = 2;
return v___x_11_;
}
else
{
lean_object* v_val_12_; lean_object* v_val_13_; uint8_t v___x_14_; 
v_val_12_ = lean_ctor_get(v_x_1_, 0);
v_val_13_ = lean_ctor_get(v_x_2_, 0);
v___x_14_ = lean_string_compare(v_val_12_, v_val_13_);
return v___x_14_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Literal_instOrd__batteries___lam__0___boxed(lean_object* v_x_15_, lean_object* v_x_16_){
_start:
{
uint8_t v_res_17_; lean_object* v_r_18_; 
v_res_17_ = lp_batteries_Lean_Literal_instOrd__batteries___lam__0(v_x_15_, v_x_16_);
lean_dec_ref(v_x_16_);
lean_dec_ref(v_x_15_);
v_r_18_ = lean_box(v_res_17_);
return v_r_18_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Expr(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Lean_Meta_Expr(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Expr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Lean_Meta_Expr(uint8_t builtin) {
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
lean_object* initialize_Lean_Expr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_Meta_Expr(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Expr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Meta_Expr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Lean_Meta_Expr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Lean_Meta_Expr(builtin);
}
#ifdef __cplusplus
}
#endif
