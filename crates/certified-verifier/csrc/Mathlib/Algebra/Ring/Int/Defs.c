// Lean compiler output
// Module: Mathlib.Algebra.Ring.Int.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.CharZero.Defs public import Mathlib.Algebra.Ring.Defs public import Mathlib.Algebra.Group.Int.Defs public import Mathlib.Data.Int.Basic public import Mathlib.Data.Int.Cast.Basic
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
lean_object* l_Int_pow(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Int_instAddCommGroup;
extern lean_object* lp_mathlib_Int_instCommMonoid;
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* l_Int_ofNat___boxed(lean_object*);
extern lean_object* lp_mathlib_Int_instCommSemigroup;
lean_object* l_Int_neg___boxed(lean_object*);
lean_object* l_Int_sub___boxed(lean_object*, lean_object*);
lean_object* l_Int_mul___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing___lam__1___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Int_instCommRing___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_instCommRing___closed__0;
static lean_once_cell_t lp_mathlib_Int_instCommRing___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_instCommRing___closed__1;
static const lean_closure_object lp_mathlib_Int_instCommRing___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Int_instCommRing___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Int_instCommRing___closed__2 = (const lean_object*)&lp_mathlib_Int_instCommRing___closed__2_value;
static const lean_closure_object lp_mathlib_Int_instCommRing___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Int_instCommRing___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Int_instCommRing___closed__3 = (const lean_object*)&lp_mathlib_Int_instCommRing___closed__3_value;
static const lean_closure_object lp_mathlib_Int_instCommRing___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Int_neg___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Int_instCommRing___closed__4 = (const lean_object*)&lp_mathlib_Int_instCommRing___closed__4_value;
static const lean_closure_object lp_mathlib_Int_instCommRing___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Int_sub___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Int_instCommRing___closed__5 = (const lean_object*)&lp_mathlib_Int_instCommRing___closed__5_value;
static const lean_closure_object lp_mathlib_Int_instCommRing___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Int_mul___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Int_instCommRing___closed__6 = (const lean_object*)&lp_mathlib_Int_instCommRing___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing;
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommSemiring;
LEAN_EXPORT lean_object* lp_mathlib_Int_instSemiring;
LEAN_EXPORT lean_object* lp_mathlib_Int_instRing;
static lean_once_cell_t lp_mathlib_Int_instDistrib___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_instDistrib___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Int_instDistrib;
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing___lam__0(lean_object* v_x_1_){
_start:
{
lean_inc(v_x_1_);
return v_x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing___lam__0___boxed(lean_object* v_x_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_Int_instCommRing___lam__0(v_x_2_);
lean_dec(v_x_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing___lam__1(lean_object* v_n_4_, lean_object* v_x_5_){
_start:
{
lean_object* v___x_6_; 
v___x_6_ = l_Int_pow(v_x_5_, v_n_4_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instCommRing___lam__1___boxed(lean_object* v_n_7_, lean_object* v_x_8_){
_start:
{
lean_object* v_res_9_; 
v_res_9_ = lp_mathlib_Int_instCommRing___lam__1(v_n_7_, v_x_8_);
lean_dec(v_x_8_);
lean_dec(v_n_7_);
return v_res_9_;
}
}
static lean_object* _init_lp_mathlib_Int_instCommRing___closed__0(void){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_10_ = lp_mathlib_Int_instCommMonoid;
v___x_11_ = lp_mathlib_Monoid_toMulOneClass___redArg(v___x_10_);
return v___x_11_;
}
}
static lean_object* _init_lp_mathlib_Int_instCommRing___closed__1(void){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_12_ = lean_obj_once(&lp_mathlib_Int_instCommRing___closed__0, &lp_mathlib_Int_instCommRing___closed__0_once, _init_lp_mathlib_Int_instCommRing___closed__0);
v___x_13_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_12_);
return v___x_13_;
}
}
static lean_object* _init_lp_mathlib_Int_instCommRing(void){
_start:
{
lean_object* v___x_19_; lean_object* v_toAddMonoid_20_; lean_object* v___x_21_; lean_object* v_toOne_22_; lean_object* v___f_23_; lean_object* v___f_24_; lean_object* v___f_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___f_29_; lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; 
v___x_19_ = lp_mathlib_Int_instAddCommGroup;
v_toAddMonoid_20_ = lean_ctor_get(v___x_19_, 0);
v___x_21_ = lean_obj_once(&lp_mathlib_Int_instCommRing___closed__1, &lp_mathlib_Int_instCommRing___closed__1_once, _init_lp_mathlib_Int_instCommRing___closed__1);
v_toOne_22_ = lean_ctor_get(v___x_21_, 0);
v___f_23_ = ((lean_object*)(lp_mathlib_Int_instCommRing___closed__2));
v___f_24_ = ((lean_object*)(lp_mathlib_Int_instCommRing___closed__3));
v___f_25_ = lean_alloc_closure((void*)(l_Int_ofNat___boxed), 1, 0);
v___x_26_ = lp_mathlib_Int_instCommSemigroup;
v___x_27_ = ((lean_object*)(lp_mathlib_Int_instCommRing___closed__4));
v___x_28_ = ((lean_object*)(lp_mathlib_Int_instCommRing___closed__5));
v___f_29_ = ((lean_object*)(lp_mathlib_Int_instCommRing___closed__6));
lean_inc(v_toOne_22_);
v___x_30_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_30_, 0, v_toOne_22_);
lean_ctor_set(v___x_30_, 1, v___x_26_);
lean_ctor_set(v___x_30_, 2, v___f_24_);
lean_inc_ref(v_toAddMonoid_20_);
v___x_31_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_31_, 0, v_toAddMonoid_20_);
lean_ctor_set(v___x_31_, 1, v___x_30_);
lean_ctor_set(v___x_31_, 2, v___f_25_);
v___x_32_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_32_, 0, v___x_31_);
lean_ctor_set(v___x_32_, 1, v___x_27_);
lean_ctor_set(v___x_32_, 2, v___x_28_);
lean_ctor_set(v___x_32_, 3, v___f_29_);
lean_ctor_set(v___x_32_, 4, v___f_23_);
return v___x_32_;
}
}
static lean_object* _init_lp_mathlib_Int_instCommSemiring(void){
_start:
{
lean_object* v___x_33_; lean_object* v_toSemiring_34_; 
v___x_33_ = lp_mathlib_Int_instCommRing;
v_toSemiring_34_ = lean_ctor_get(v___x_33_, 0);
lean_inc_ref(v_toSemiring_34_);
return v_toSemiring_34_;
}
}
static lean_object* _init_lp_mathlib_Int_instSemiring(void){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lp_mathlib_Int_instCommSemiring;
return v___x_35_;
}
}
static lean_object* _init_lp_mathlib_Int_instRing(void){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Int_instCommRing;
return v___x_36_;
}
}
static lean_object* _init_lp_mathlib_Int_instDistrib___closed__0(void){
_start:
{
lean_object* v___x_37_; lean_object* v___x_38_; 
v___x_37_ = lp_mathlib_Int_instCommSemiring;
v___x_38_ = lp_mathlib_instDistribOfSemiring___redArg(v___x_37_);
return v___x_38_;
}
}
static lean_object* _init_lp_mathlib_Int_instDistrib(void){
_start:
{
lean_object* v___x_39_; 
v___x_39_ = lean_obj_once(&lp_mathlib_Int_instDistrib___closed__0, &lp_mathlib_Int_instDistrib___closed__0_once, _init_lp_mathlib_Int_instDistrib___closed__0);
return v___x_39_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_CharZero_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_CharZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Int_instCommRing = _init_lp_mathlib_Int_instCommRing();
lean_mark_persistent(lp_mathlib_Int_instCommRing);
lp_mathlib_Int_instCommSemiring = _init_lp_mathlib_Int_instCommSemiring();
lean_mark_persistent(lp_mathlib_Int_instCommSemiring);
lp_mathlib_Int_instSemiring = _init_lp_mathlib_Int_instSemiring();
lean_mark_persistent(lp_mathlib_Int_instSemiring);
lp_mathlib_Int_instRing = _init_lp_mathlib_Int_instRing();
lean_mark_persistent(lp_mathlib_Int_instRing);
lp_mathlib_Int_instDistrib = _init_lp_mathlib_Int_instDistrib();
lean_mark_persistent(lp_mathlib_Int_instDistrib);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_CharZero_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_CharZero_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
