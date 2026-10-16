// Lean compiler output
// Module: Mathlib.Data.Nat.Pairing
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Notation.Prod public import Mathlib.Data.Nat.Sqrt public import Mathlib.Data.Set.Lattice.Image
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
lean_object* l_Nat_sqrt(lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_pair(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_pair___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_unpair(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_unpair___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_pairEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_pairEquiv___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Nat_pairEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_pairEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_pairEquiv___closed__0 = (const lean_object*)&lp_mathlib_Nat_pairEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_Nat_pairEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_unpair___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_pairEquiv___closed__1 = (const lean_object*)&lp_mathlib_Nat_pairEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_Nat_pairEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Nat_pairEquiv___closed__0_value),((lean_object*)&lp_mathlib_Nat_pairEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_Nat_pairEquiv___closed__2 = (const lean_object*)&lp_mathlib_Nat_pairEquiv___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_pairEquiv = (const lean_object*)&lp_mathlib_Nat_pairEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_pair(lean_object* v_a_1_, lean_object* v_b_2_){
_start:
{
uint8_t v___x_3_; 
v___x_3_ = lean_nat_dec_lt(v_a_1_, v_b_2_);
if (v___x_3_ == 0)
{
lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_4_ = lean_nat_mul(v_a_1_, v_a_1_);
v___x_5_ = lean_nat_add(v___x_4_, v_a_1_);
lean_dec(v___x_4_);
v___x_6_ = lean_nat_add(v___x_5_, v_b_2_);
lean_dec(v___x_5_);
return v___x_6_;
}
else
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = lean_nat_mul(v_b_2_, v_b_2_);
v___x_8_ = lean_nat_add(v___x_7_, v_a_1_);
lean_dec(v___x_7_);
return v___x_8_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_pair___boxed(lean_object* v_a_9_, lean_object* v_b_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_mathlib_Nat_pair(v_a_9_, v_b_10_);
lean_dec(v_b_10_);
lean_dec(v_a_9_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_unpair(lean_object* v_n_12_){
_start:
{
lean_object* v_s_13_; lean_object* v___x_14_; lean_object* v___x_15_; uint8_t v___x_16_; 
v_s_13_ = l_Nat_sqrt(v_n_12_);
v___x_14_ = lean_nat_mul(v_s_13_, v_s_13_);
v___x_15_ = lean_nat_sub(v_n_12_, v___x_14_);
lean_dec(v___x_14_);
v___x_16_ = lean_nat_dec_lt(v___x_15_, v_s_13_);
if (v___x_16_ == 0)
{
lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_17_ = lean_nat_sub(v___x_15_, v_s_13_);
lean_dec(v___x_15_);
v___x_18_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_18_, 0, v_s_13_);
lean_ctor_set(v___x_18_, 1, v___x_17_);
return v___x_18_;
}
else
{
lean_object* v___x_19_; 
v___x_19_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_19_, 0, v___x_15_);
lean_ctor_set(v___x_19_, 1, v_s_13_);
return v___x_19_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_unpair___boxed(lean_object* v_n_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_Nat_unpair(v_n_20_);
lean_dec(v_n_20_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_pairEquiv___lam__0(lean_object* v___y_22_){
_start:
{
lean_object* v_fst_23_; lean_object* v_snd_24_; lean_object* v___x_25_; 
v_fst_23_ = lean_ctor_get(v___y_22_, 0);
v_snd_24_ = lean_ctor_get(v___y_22_, 1);
v___x_25_ = lp_mathlib_Nat_pair(v_fst_23_, v_snd_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_pairEquiv___lam__0___boxed(lean_object* v___y_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_Nat_pairEquiv___lam__0(v___y_26_);
lean_dec_ref(v___y_26_);
return v_res_27_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Sqrt(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Image(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Pairing(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Sqrt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_Pairing(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Sqrt(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Lattice_Image(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Pairing(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Sqrt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Lattice_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Pairing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_Pairing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_Pairing(builtin);
}
#ifdef __cplusplus
}
#endif
