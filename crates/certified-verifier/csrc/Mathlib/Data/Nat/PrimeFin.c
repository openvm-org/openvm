// Lean compiler output
// Module: Mathlib.Data.Nat.PrimeFin
// Imports: public import Init public meta import Init public import Mathlib.Basic.Countable.Defs public import Mathlib.Data.Nat.Factors public import Mathlib.Data.Nat.Prime.Infinite public import Mathlib.Data.Set.Finite.Lattice
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
lean_object* lean_array_mk(lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
size_t lean_usize_sub(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t l_List_decidableBAll___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_primeFactorsList(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4_spec__5___lam__0(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4_spec__5___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4_spec__5(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_toFinset___at___00Nat_primeFactors_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_primeFactors(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_primeFactors___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4_spec__5___lam__0(lean_object* v___x_1_, uint8_t v___x_2_, lean_object* v___y_3_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = lean_nat_dec_eq(v___x_1_, v___y_3_);
if (v___x_4_ == 0)
{
uint8_t v___x_5_; 
v___x_5_ = 1;
return v___x_5_;
}
else
{
return v___x_2_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4_spec__5___lam__0___boxed(lean_object* v___x_6_, lean_object* v___x_7_, lean_object* v___y_8_){
_start:
{
uint8_t v___x_269__boxed_9_; uint8_t v_res_10_; lean_object* v_r_11_; 
v___x_269__boxed_9_ = lean_unbox(v___x_7_);
v_res_10_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4_spec__5___lam__0(v___x_6_, v___x_269__boxed_9_, v___y_8_);
lean_dec(v___y_8_);
lean_dec(v___x_6_);
v_r_11_ = lean_box(v_res_10_);
return v_r_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4_spec__5(lean_object* v_as_12_, size_t v_i_13_, size_t v_stop_14_, lean_object* v_b_15_){
_start:
{
uint8_t v___x_16_; 
v___x_16_ = lean_usize_dec_eq(v_i_13_, v_stop_14_);
if (v___x_16_ == 0)
{
size_t v___x_17_; size_t v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___f_21_; uint8_t v___x_22_; 
v___x_17_ = ((size_t)1ULL);
v___x_18_ = lean_usize_sub(v_i_13_, v___x_17_);
v___x_19_ = lean_array_uget_borrowed(v_as_12_, v___x_18_);
v___x_20_ = lean_box(v___x_16_);
lean_inc(v___x_19_);
v___f_21_ = lean_alloc_closure((void*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4_spec__5___lam__0___boxed), 3, 2);
lean_closure_set(v___f_21_, 0, v___x_19_);
lean_closure_set(v___f_21_, 1, v___x_20_);
lean_inc(v_b_15_);
v___x_22_ = l_List_decidableBAll___redArg(v___f_21_, v_b_15_);
if (v___x_22_ == 0)
{
v_i_13_ = v___x_18_;
goto _start;
}
else
{
lean_object* v___x_24_; 
lean_inc(v___x_19_);
v___x_24_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_24_, 0, v___x_19_);
lean_ctor_set(v___x_24_, 1, v_b_15_);
v_i_13_ = v___x_18_;
v_b_15_ = v___x_24_;
goto _start;
}
}
else
{
return v_b_15_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4_spec__5___boxed(lean_object* v_as_26_, lean_object* v_i_27_, lean_object* v_stop_28_, lean_object* v_b_29_){
_start:
{
size_t v_i_boxed_30_; size_t v_stop_boxed_31_; lean_object* v_res_32_; 
v_i_boxed_30_ = lean_unbox_usize(v_i_27_);
lean_dec(v_i_27_);
v_stop_boxed_31_ = lean_unbox_usize(v_stop_28_);
lean_dec(v_stop_28_);
v_res_32_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4_spec__5(v_as_26_, v_i_boxed_30_, v_stop_boxed_31_, v_b_29_);
lean_dec_ref(v_as_26_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4(lean_object* v_init_33_, lean_object* v_l_34_){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; uint8_t v___x_38_; 
v___x_35_ = lean_array_mk(v_l_34_);
v___x_36_ = lean_array_get_size(v___x_35_);
v___x_37_ = lean_unsigned_to_nat(0u);
v___x_38_ = lean_nat_dec_lt(v___x_37_, v___x_36_);
if (v___x_38_ == 0)
{
lean_dec_ref(v___x_35_);
return v_init_33_;
}
else
{
size_t v___x_39_; size_t v___x_40_; lean_object* v___x_41_; 
v___x_39_ = lean_usize_of_nat(v___x_36_);
v___x_40_ = ((size_t)0ULL);
v___x_41_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4_spec__5(v___x_35_, v___x_39_, v___x_40_, v_init_33_);
lean_dec_ref(v___x_35_);
return v___x_41_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3___redArg(lean_object* v_l_42_){
_start:
{
lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_43_ = lean_box(0);
v___x_44_ = lp_mathlib_List_foldrTR___at___00List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3_spec__4(v___x_43_, v_l_42_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1(lean_object* v_s_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3___redArg(v_s_45_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0(lean_object* v_s_47_){
_start:
{
lean_object* v___x_48_; 
v___x_48_ = lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3___redArg(v_s_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_toFinset___at___00Nat_primeFactors_spec__0(lean_object* v_l_49_){
_start:
{
lean_object* v___x_50_; 
v___x_50_ = lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3___redArg(v_l_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_primeFactors(lean_object* v_n_51_){
_start:
{
lean_object* v___x_52_; lean_object* v___x_53_; 
v___x_52_ = lp_mathlib_Nat_primeFactorsList(v_n_51_);
v___x_53_ = lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3___redArg(v___x_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_primeFactors___boxed(lean_object* v_n_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_mathlib_Nat_primeFactors(v_n_54_);
lean_dec(v_n_54_);
return v_res_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2(lean_object* v_l_56_){
_start:
{
lean_object* v___x_57_; 
v___x_57_ = lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3___redArg(v_l_56_);
return v___x_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3(lean_object* v_R_58_, lean_object* v_l_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lp_mathlib_List_pwFilter___at___00List_dedup___at___00Multiset_dedup___at___00Multiset_toFinset___at___00List_toFinset___at___00Nat_primeFactors_spec__0_spec__0_spec__1_spec__2_spec__3___redArg(v_l_59_);
return v___x_60_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Countable_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Factors(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Prime_Infinite(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Lattice(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_PrimeFin(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Countable_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Factors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Prime_Infinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Nat_PrimeFin(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Basic_Countable_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Factors(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Prime_Infinite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Lattice(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_PrimeFin(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Countable_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Factors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Prime_Infinite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_PrimeFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Nat_PrimeFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Nat_PrimeFin(builtin);
}
#ifdef __cplusplus
}
#endif
