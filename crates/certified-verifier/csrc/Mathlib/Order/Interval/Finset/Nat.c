// Lean compiler output
// Module: Mathlib.Order.Interval.Finset.Nat
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Embedding public import Mathlib.Order.Interval.Multiset
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
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_List_range_x27TR_go(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__3___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Nat_instLocallyFiniteOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_instLocallyFiniteOrder___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___closed__0 = (const lean_object*)&lp_mathlib_Nat_instLocallyFiniteOrder___closed__0_value;
static const lean_closure_object lp_mathlib_Nat_instLocallyFiniteOrder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_instLocallyFiniteOrder___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___closed__1 = (const lean_object*)&lp_mathlib_Nat_instLocallyFiniteOrder___closed__1_value;
static const lean_closure_object lp_mathlib_Nat_instLocallyFiniteOrder___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_instLocallyFiniteOrder___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___closed__2 = (const lean_object*)&lp_mathlib_Nat_instLocallyFiniteOrder___closed__2_value;
static const lean_closure_object lp_mathlib_Nat_instLocallyFiniteOrder___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_instLocallyFiniteOrder___lam__3___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___closed__3 = (const lean_object*)&lp_mathlib_Nat_instLocallyFiniteOrder___closed__3_value;
static const lean_ctor_object lp_mathlib_Nat_instLocallyFiniteOrder___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Nat_instLocallyFiniteOrder___closed__0_value),((lean_object*)&lp_mathlib_Nat_instLocallyFiniteOrder___closed__1_value),((lean_object*)&lp_mathlib_Nat_instLocallyFiniteOrder___closed__2_value),((lean_object*)&lp_mathlib_Nat_instLocallyFiniteOrder___closed__3_value)}};
static const lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___closed__4 = (const lean_object*)&lp_mathlib_Nat_instLocallyFiniteOrder___closed__4_value;
LEAN_EXPORT const lean_object* lp_mathlib_Nat_instLocallyFiniteOrder = (const lean_object*)&lp_mathlib_Nat_instLocallyFiniteOrder___closed__4_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instUniqueSubtypeMemFinsetIicOfNat;
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__0(lean_object* v_a_1_, lean_object* v_b_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_3_ = lean_unsigned_to_nat(1u);
v___x_4_ = lean_nat_add(v_b_2_, v___x_3_);
v___x_5_ = lean_nat_sub(v___x_4_, v_a_1_);
lean_dec(v___x_4_);
v___x_6_ = lean_nat_add(v_a_1_, v___x_5_);
v___x_7_ = lean_box(0);
v___x_8_ = l_List_range_x27TR_go(v___x_3_, v___x_5_, v___x_6_, v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__0___boxed(lean_object* v_a_9_, lean_object* v_b_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_mathlib_Nat_instLocallyFiniteOrder___lam__0(v_a_9_, v_b_10_);
lean_dec(v_b_10_);
lean_dec(v_a_9_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__1(lean_object* v_a_12_, lean_object* v_b_13_){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_14_ = lean_nat_sub(v_b_13_, v_a_12_);
v___x_15_ = lean_unsigned_to_nat(1u);
v___x_16_ = lean_nat_add(v_a_12_, v___x_14_);
v___x_17_ = lean_box(0);
v___x_18_ = l_List_range_x27TR_go(v___x_15_, v___x_14_, v___x_16_, v___x_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__1___boxed(lean_object* v_a_19_, lean_object* v_b_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_Nat_instLocallyFiniteOrder___lam__1(v_a_19_, v_b_20_);
lean_dec(v_b_20_);
lean_dec(v_a_19_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__2(lean_object* v_a_22_, lean_object* v_b_23_){
_start:
{
lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_24_ = lean_unsigned_to_nat(1u);
v___x_25_ = lean_nat_add(v_a_22_, v___x_24_);
v___x_26_ = lean_nat_sub(v_b_23_, v_a_22_);
v___x_27_ = lean_nat_add(v___x_25_, v___x_26_);
lean_dec(v___x_25_);
v___x_28_ = lean_box(0);
v___x_29_ = l_List_range_x27TR_go(v___x_24_, v___x_26_, v___x_27_, v___x_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__2___boxed(lean_object* v_a_30_, lean_object* v_b_31_){
_start:
{
lean_object* v_res_32_; 
v_res_32_ = lp_mathlib_Nat_instLocallyFiniteOrder___lam__2(v_a_30_, v_b_31_);
lean_dec(v_b_31_);
lean_dec(v_a_30_);
return v_res_32_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__3(lean_object* v_a_33_, lean_object* v_b_34_){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_35_ = lean_unsigned_to_nat(1u);
v___x_36_ = lean_nat_add(v_a_33_, v___x_35_);
v___x_37_ = lean_nat_sub(v_b_34_, v_a_33_);
v___x_38_ = lean_nat_sub(v___x_37_, v___x_35_);
lean_dec(v___x_37_);
v___x_39_ = lean_nat_add(v___x_36_, v___x_38_);
lean_dec(v___x_36_);
v___x_40_ = lean_box(0);
v___x_41_ = l_List_range_x27TR_go(v___x_35_, v___x_38_, v___x_39_, v___x_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_instLocallyFiniteOrder___lam__3___boxed(lean_object* v_a_42_, lean_object* v_b_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_Nat_instLocallyFiniteOrder___lam__3(v_a_42_, v_b_43_);
lean_dec(v_b_43_);
lean_dec(v_a_42_);
return v_res_44_;
}
}
static lean_object* _init_lp_mathlib_Nat_instUniqueSubtypeMemFinsetIicOfNat(void){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lean_unsigned_to_nat(0u);
return v___x_55_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Embedding(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Multiset(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Embedding(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_instUniqueSubtypeMemFinsetIicOfNat = _init_lp_mathlib_Nat_instUniqueSubtypeMemFinsetIicOfNat();
lean_mark_persistent(lp_mathlib_Nat_instUniqueSubtypeMemFinsetIicOfNat);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Embedding(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Multiset(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Embedding(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Multiset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(builtin);
}
#ifdef __cplusplus
}
#endif
