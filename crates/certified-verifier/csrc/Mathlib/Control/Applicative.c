// Lean compiler output
// Module: Mathlib.Control.Applicative
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Monoid public import Mathlib.Control.Functor public import Mathlib.Control.Basic import Mathlib.Tactic.Attr.Register
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
lean_object* lp_mathlib_Functor_Const_functor(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instApplicativeConstOfOneOfMul___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeAddConstOfZeroOfAdd___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeAddConstOfZeroOfAdd(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_00_u03b1_2_, lean_object* v_x_3_){
_start:
{
lean_inc(v_inst_1_);
return v_inst_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__0___boxed(lean_object* v_inst_4_, lean_object* v_00_u03b1_5_, lean_object* v_x_6_){
_start:
{
lean_object* v_res_7_; 
v_res_7_ = lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__0(v_inst_4_, v_00_u03b1_5_, v_x_6_);
lean_dec(v_x_6_);
lean_dec(v_inst_4_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__1(lean_object* v_inst_8_, lean_object* v_00_u03b1_9_, lean_object* v_00_u03b2_10_, lean_object* v_f_11_, lean_object* v_x_12_){
_start:
{
lean_object* v___x_13_; lean_object* v_this_14_; lean_object* v___x_15_; 
v___x_13_ = lean_box(0);
v_this_14_ = lean_apply_1(v_x_12_, v___x_13_);
v___x_15_ = lean_apply_2(v_inst_8_, v_f_11_, v_this_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__2(lean_object* v___f_16_, lean_object* v_00_u03b1_17_, lean_object* v_00_u03b2_18_, lean_object* v_a_19_, lean_object* v_b_20_){
_start:
{
lean_object* v___x_21_; 
v___x_21_ = lean_apply_4(v___f_16_, lean_box(0), lean_box(0), v_a_19_, v_b_20_);
return v___x_21_;
}
}
static lean_object* _init_lp_mathlib_instApplicativeConstOfOneOfMul___redArg___closed__0(void){
_start:
{
lean_object* v___x_22_; 
v___x_22_ = lp_mathlib_Functor_Const_functor(lean_box(0));
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul___redArg(lean_object* v_inst_23_, lean_object* v_inst_24_){
_start:
{
lean_object* v___f_25_; lean_object* v___f_26_; lean_object* v___f_27_; lean_object* v___x_28_; lean_object* v___x_29_; 
v___f_25_ = lean_alloc_closure((void*)(lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_25_, 0, v_inst_23_);
v___f_26_ = lean_alloc_closure((void*)(lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__1), 5, 1);
lean_closure_set(v___f_26_, 0, v_inst_24_);
lean_inc_ref(v___f_26_);
v___f_27_ = lean_alloc_closure((void*)(lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__2), 5, 1);
lean_closure_set(v___f_27_, 0, v___f_26_);
v___x_28_ = lean_obj_once(&lp_mathlib_instApplicativeConstOfOneOfMul___redArg___closed__0, &lp_mathlib_instApplicativeConstOfOneOfMul___redArg___closed__0_once, _init_lp_mathlib_instApplicativeConstOfOneOfMul___redArg___closed__0);
lean_inc_ref(v___f_27_);
v___x_29_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_29_, 0, v___x_28_);
lean_ctor_set(v___x_29_, 1, v___f_25_);
lean_ctor_set(v___x_29_, 2, v___f_26_);
lean_ctor_set(v___x_29_, 3, v___f_27_);
lean_ctor_set(v___x_29_, 4, v___f_27_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeConstOfOneOfMul(lean_object* v_00_u03b1_30_, lean_object* v_inst_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_instApplicativeConstOfOneOfMul___redArg(v_inst_31_, v_inst_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeAddConstOfZeroOfAdd___redArg(lean_object* v_inst_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v___f_36_; lean_object* v___f_37_; lean_object* v___f_38_; lean_object* v___x_39_; lean_object* v___x_40_; 
v___f_36_ = lean_alloc_closure((void*)(lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_36_, 0, v_inst_34_);
v___f_37_ = lean_alloc_closure((void*)(lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__1), 5, 1);
lean_closure_set(v___f_37_, 0, v_inst_35_);
lean_inc_ref(v___f_37_);
v___f_38_ = lean_alloc_closure((void*)(lp_mathlib_instApplicativeConstOfOneOfMul___redArg___lam__2), 5, 1);
lean_closure_set(v___f_38_, 0, v___f_37_);
v___x_39_ = lean_obj_once(&lp_mathlib_instApplicativeConstOfOneOfMul___redArg___closed__0, &lp_mathlib_instApplicativeConstOfOneOfMul___redArg___closed__0_once, _init_lp_mathlib_instApplicativeConstOfOneOfMul___redArg___closed__0);
lean_inc_ref(v___f_38_);
v___x_40_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_40_, 0, v___x_39_);
lean_ctor_set(v___x_40_, 1, v___f_36_);
lean_ctor_set(v___x_40_, 2, v___f_37_);
lean_ctor_set(v___x_40_, 3, v___f_38_);
lean_ctor_set(v___x_40_, 4, v___f_38_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instApplicativeAddConstOfZeroOfAdd(lean_object* v_00_u03b1_41_, lean_object* v_inst_42_, lean_object* v_inst_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = lp_mathlib_instApplicativeAddConstOfZeroOfAdd___redArg(v_inst_42_, v_inst_43_);
return v___x_44_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Monoid(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Control_Functor(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Control_Applicative(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Functor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Control_Applicative(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Monoid(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Functor(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Register(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Control_Applicative(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Monoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Functor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Register(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Applicative(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Control_Applicative(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Control_Applicative(builtin);
}
#ifdef __cplusplus
}
#endif
