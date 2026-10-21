// Lean compiler output
// Module: Mathlib.Order.PropInstances
// Imports: public import Init public meta import Init public import Mathlib.Order.Disjoint
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
extern lean_object* lp_mathlib_Prop_partialOrder;
static lean_once_cell_t lp_mathlib_Prop_instDistribLattice___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Prop_instDistribLattice___closed__0;
static lean_once_cell_t lp_mathlib_Prop_instDistribLattice___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Prop_instDistribLattice___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Prop_instDistribLattice;
static const lean_ctor_object lp_mathlib_Prop_instBoundedOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Prop_instBoundedOrder___closed__0 = (const lean_object*)&lp_mathlib_Prop_instBoundedOrder___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Prop_instBoundedOrder = (const lean_object*)&lp_mathlib_Prop_instBoundedOrder___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_Prop_decidablePredBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prop_decidablePredBot___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Prop_decidablePredTop(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prop_decidablePredTop___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Prop_decidableRelBot(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prop_decidableRelBot___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Prop_decidableRelTop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prop_decidableRelTop___boxed(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Prop_instDistribLattice___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; 
v___x_1_ = lp_mathlib_Prop_partialOrder;
v___x_2_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2_, 0, v___x_1_);
lean_ctor_set(v___x_2_, 1, lean_box(0));
return v___x_2_;
}
}
static lean_object* _init_lp_mathlib_Prop_instDistribLattice___closed__1(void){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lean_obj_once(&lp_mathlib_Prop_instDistribLattice___closed__0, &lp_mathlib_Prop_instDistribLattice___closed__0_once, _init_lp_mathlib_Prop_instDistribLattice___closed__0);
v___x_4_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_4_, 0, v___x_3_);
lean_ctor_set(v___x_4_, 1, lean_box(0));
return v___x_4_;
}
}
static lean_object* _init_lp_mathlib_Prop_instDistribLattice(void){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_obj_once(&lp_mathlib_Prop_instDistribLattice___closed__1, &lp_mathlib_Prop_instDistribLattice___closed__1_once, _init_lp_mathlib_Prop_instDistribLattice___closed__1);
return v___x_5_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Prop_decidablePredBot(lean_object* v_00_u03b1_8_, lean_object* v_x_9_){
_start:
{
uint8_t v___x_10_; 
v___x_10_ = 0;
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prop_decidablePredBot___boxed(lean_object* v_00_u03b1_11_, lean_object* v_x_12_){
_start:
{
uint8_t v_res_13_; lean_object* v_r_14_; 
v_res_13_ = lp_mathlib_Prop_decidablePredBot(v_00_u03b1_11_, v_x_12_);
lean_dec(v_x_12_);
v_r_14_ = lean_box(v_res_13_);
return v_r_14_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Prop_decidablePredTop(lean_object* v_00_u03b1_15_, lean_object* v_x_16_){
_start:
{
uint8_t v___x_17_; 
v___x_17_ = 1;
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prop_decidablePredTop___boxed(lean_object* v_00_u03b1_18_, lean_object* v_x_19_){
_start:
{
uint8_t v_res_20_; lean_object* v_r_21_; 
v_res_20_ = lp_mathlib_Prop_decidablePredTop(v_00_u03b1_18_, v_x_19_);
lean_dec(v_x_19_);
v_r_21_ = lean_box(v_res_20_);
return v_r_21_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Prop_decidableRelBot(lean_object* v_00_u03b1_22_, lean_object* v_x_23_, lean_object* v_x_24_){
_start:
{
uint8_t v___x_25_; 
v___x_25_ = 0;
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prop_decidableRelBot___boxed(lean_object* v_00_u03b1_26_, lean_object* v_x_27_, lean_object* v_x_28_){
_start:
{
uint8_t v_res_29_; lean_object* v_r_30_; 
v_res_29_ = lp_mathlib_Prop_decidableRelBot(v_00_u03b1_26_, v_x_27_, v_x_28_);
lean_dec(v_x_28_);
lean_dec(v_x_27_);
v_r_30_ = lean_box(v_res_29_);
return v_r_30_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Prop_decidableRelTop(lean_object* v_00_u03b1_31_, lean_object* v_x_32_, lean_object* v_x_33_){
_start:
{
uint8_t v___x_34_; 
v___x_34_ = 1;
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prop_decidableRelTop___boxed(lean_object* v_00_u03b1_35_, lean_object* v_x_36_, lean_object* v_x_37_){
_start:
{
uint8_t v_res_38_; lean_object* v_r_39_; 
v_res_38_ = lp_mathlib_Prop_decidableRelTop(v_00_u03b1_35_, v_x_36_, v_x_37_);
lean_dec(v_x_37_);
lean_dec(v_x_36_);
v_r_39_ = lean_box(v_res_38_);
return v_r_39_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Disjoint(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_PropInstances(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Disjoint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Prop_instDistribLattice = _init_lp_mathlib_Prop_instDistribLattice();
lean_mark_persistent(lp_mathlib_Prop_instDistribLattice);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_PropInstances(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_Disjoint(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_PropInstances(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Disjoint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_PropInstances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_PropInstances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_PropInstances(builtin);
}
#ifdef __cplusplus
}
#endif
