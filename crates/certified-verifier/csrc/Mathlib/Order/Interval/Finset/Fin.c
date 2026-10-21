// Lean compiler output
// Module: Mathlib.Order.Interval.Finset.Fin
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Fin public import Mathlib.Order.Interval.Finset.Nat public import Mathlib.Order.Interval.Set.Fin
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
extern lean_object* lp_mathlib_Nat_instLocallyFiniteOrder;
lean_object* lp_mathlib_Finset_Ico___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_attachFin___redArg(lean_object*);
lean_object* lp_mathlib_Fin_instPartialOrder(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_Ioo___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_Ioc___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Finset_Icc___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_instCoheytingAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderBot(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_instHeytingAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___lam__3(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Fin_instLocallyFiniteOrder___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___closed__0;
static lean_once_cell_t lp_mathlib_Fin_instLocallyFiniteOrder___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___closed__1;
static lean_once_cell_t lp_mathlib_Fin_instLocallyFiniteOrder___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___closed__2;
static lean_once_cell_t lp_mathlib_Fin_instLocallyFiniteOrder___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___closed__3;
static lean_once_cell_t lp_mathlib_Fin_instLocallyFiniteOrder___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___boxed(lean_object*);
static lean_once_cell_t lp_mathlib_Fin_instLocallyFiniteOrderBot___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fin_instLocallyFiniteOrderBot___closed__0;
static lean_once_cell_t lp_mathlib_Fin_instLocallyFiniteOrderBot___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fin_instLocallyFiniteOrderBot___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderBot(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderBot___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderTop___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderTop___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fin_instLocallyFiniteOrderTop___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__0 = (const lean_object*)&lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__0_value;
static const lean_ctor_object lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__0_value),((lean_object*)&lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__0_value)}};
static const lean_object* lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__1 = (const lean_object*)&lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderTop(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderTop___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___lam__0(lean_object* v___x_1_, lean_object* v_a_2_, lean_object* v_b_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; 
v___x_4_ = lp_mathlib_Finset_Icc___redArg(v___x_1_, v_a_2_, v_b_3_);
v___x_5_ = lp_mathlib_Finset_attachFin___redArg(v___x_4_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___lam__1(lean_object* v___x_6_, lean_object* v_a_7_, lean_object* v_b_8_){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; 
v___x_9_ = lp_mathlib_Finset_Ico___redArg(v___x_6_, v_a_7_, v_b_8_);
v___x_10_ = lp_mathlib_Finset_attachFin___redArg(v___x_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___lam__2(lean_object* v___x_11_, lean_object* v_a_12_, lean_object* v_b_13_){
_start:
{
lean_object* v___x_14_; lean_object* v___x_15_; 
v___x_14_ = lp_mathlib_Finset_Ioc___redArg(v___x_11_, v_a_12_, v_b_13_);
v___x_15_ = lp_mathlib_Finset_attachFin___redArg(v___x_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___lam__3(lean_object* v___x_16_, lean_object* v_a_17_, lean_object* v_b_18_){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_19_ = lp_mathlib_Finset_Ioo___redArg(v___x_16_, v_a_17_, v_b_18_);
v___x_20_ = lp_mathlib_Finset_attachFin___redArg(v___x_19_);
return v___x_20_;
}
}
static lean_object* _init_lp_mathlib_Fin_instLocallyFiniteOrder___closed__0(void){
_start:
{
lean_object* v___x_21_; lean_object* v___f_22_; 
v___x_21_ = lp_mathlib_Nat_instLocallyFiniteOrder;
v___f_22_ = lean_alloc_closure((void*)(lp_mathlib_Fin_instLocallyFiniteOrder___lam__0), 3, 1);
lean_closure_set(v___f_22_, 0, v___x_21_);
return v___f_22_;
}
}
static lean_object* _init_lp_mathlib_Fin_instLocallyFiniteOrder___closed__1(void){
_start:
{
lean_object* v___x_23_; lean_object* v___f_24_; 
v___x_23_ = lp_mathlib_Nat_instLocallyFiniteOrder;
v___f_24_ = lean_alloc_closure((void*)(lp_mathlib_Fin_instLocallyFiniteOrder___lam__1), 3, 1);
lean_closure_set(v___f_24_, 0, v___x_23_);
return v___f_24_;
}
}
static lean_object* _init_lp_mathlib_Fin_instLocallyFiniteOrder___closed__2(void){
_start:
{
lean_object* v___x_25_; lean_object* v___f_26_; 
v___x_25_ = lp_mathlib_Nat_instLocallyFiniteOrder;
v___f_26_ = lean_alloc_closure((void*)(lp_mathlib_Fin_instLocallyFiniteOrder___lam__2), 3, 1);
lean_closure_set(v___f_26_, 0, v___x_25_);
return v___f_26_;
}
}
static lean_object* _init_lp_mathlib_Fin_instLocallyFiniteOrder___closed__3(void){
_start:
{
lean_object* v___x_27_; lean_object* v___f_28_; 
v___x_27_ = lp_mathlib_Nat_instLocallyFiniteOrder;
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_Fin_instLocallyFiniteOrder___lam__3), 3, 1);
lean_closure_set(v___f_28_, 0, v___x_27_);
return v___f_28_;
}
}
static lean_object* _init_lp_mathlib_Fin_instLocallyFiniteOrder___closed__4(void){
_start:
{
lean_object* v___f_29_; lean_object* v___f_30_; lean_object* v___f_31_; lean_object* v___f_32_; lean_object* v___x_33_; 
v___f_29_ = lean_obj_once(&lp_mathlib_Fin_instLocallyFiniteOrder___closed__3, &lp_mathlib_Fin_instLocallyFiniteOrder___closed__3_once, _init_lp_mathlib_Fin_instLocallyFiniteOrder___closed__3);
v___f_30_ = lean_obj_once(&lp_mathlib_Fin_instLocallyFiniteOrder___closed__2, &lp_mathlib_Fin_instLocallyFiniteOrder___closed__2_once, _init_lp_mathlib_Fin_instLocallyFiniteOrder___closed__2);
v___f_31_ = lean_obj_once(&lp_mathlib_Fin_instLocallyFiniteOrder___closed__1, &lp_mathlib_Fin_instLocallyFiniteOrder___closed__1_once, _init_lp_mathlib_Fin_instLocallyFiniteOrder___closed__1);
v___f_32_ = lean_obj_once(&lp_mathlib_Fin_instLocallyFiniteOrder___closed__0, &lp_mathlib_Fin_instLocallyFiniteOrder___closed__0_once, _init_lp_mathlib_Fin_instLocallyFiniteOrder___closed__0);
v___x_33_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_33_, 0, v___f_32_);
lean_ctor_set(v___x_33_, 1, v___f_31_);
lean_ctor_set(v___x_33_, 2, v___f_30_);
lean_ctor_set(v___x_33_, 3, v___f_29_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder(lean_object* v_n_34_){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lean_obj_once(&lp_mathlib_Fin_instLocallyFiniteOrder___closed__4, &lp_mathlib_Fin_instLocallyFiniteOrder___closed__4_once, _init_lp_mathlib_Fin_instLocallyFiniteOrder___closed__4);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrder___boxed(lean_object* v_n_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_mathlib_Fin_instLocallyFiniteOrder(v_n_36_);
lean_dec(v_n_36_);
return v_res_37_;
}
}
static lean_object* _init_lp_mathlib_Fin_instLocallyFiniteOrderBot___closed__0(void){
_start:
{
lean_object* v_zero_38_; lean_object* v___x_39_; 
v_zero_38_ = lean_unsigned_to_nat(0u);
v___x_39_ = lp_mathlib_Fin_instPartialOrder(v_zero_38_);
return v___x_39_;
}
}
static lean_object* _init_lp_mathlib_Fin_instLocallyFiniteOrderBot___closed__1(void){
_start:
{
lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_40_ = lean_obj_once(&lp_mathlib_Fin_instLocallyFiniteOrderBot___closed__0, &lp_mathlib_Fin_instLocallyFiniteOrderBot___closed__0_once, _init_lp_mathlib_Fin_instLocallyFiniteOrderBot___closed__0);
v___x_41_ = lp_mathlib_IsEmpty_toLocallyFiniteOrderBot(lean_box(0), v___x_40_, lean_box(0));
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderBot(lean_object* v_x_42_){
_start:
{
lean_object* v_zero_43_; uint8_t v_isZero_44_; 
v_zero_43_ = lean_unsigned_to_nat(0u);
v_isZero_44_ = lean_nat_dec_eq(v_x_42_, v_zero_43_);
if (v_isZero_44_ == 1)
{
lean_object* v___x_45_; 
v___x_45_ = lean_obj_once(&lp_mathlib_Fin_instLocallyFiniteOrderBot___closed__1, &lp_mathlib_Fin_instLocallyFiniteOrderBot___closed__1_once, _init_lp_mathlib_Fin_instLocallyFiniteOrderBot___closed__1);
return v___x_45_;
}
else
{
lean_object* v_one_46_; lean_object* v_n_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v_toOrderBot_51_; lean_object* v___x_52_; 
v_one_46_ = lean_unsigned_to_nat(1u);
v_n_47_ = lean_nat_sub(v_x_42_, v_one_46_);
v___x_48_ = lean_nat_add(v_n_47_, v_one_46_);
lean_dec(v_n_47_);
v___x_49_ = lp_mathlib_Fin_instLocallyFiniteOrder(v___x_48_);
v___x_50_ = lp_mathlib_Fin_instHeytingAlgebra___redArg(v___x_48_);
v_toOrderBot_51_ = lean_ctor_get(v___x_50_, 1);
lean_inc(v_toOrderBot_51_);
lean_dec_ref(v___x_50_);
v___x_52_ = lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg(v___x_49_, v_toOrderBot_51_);
return v___x_52_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderBot___boxed(lean_object* v_x_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_Fin_instLocallyFiniteOrderBot(v_x_53_);
lean_dec(v_x_53_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderTop___lam__0(lean_object* v_a_55_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderTop___lam__0___boxed(lean_object* v_a_56_){
_start:
{
lean_object* v_res_57_; 
v_res_57_ = lp_mathlib_Fin_instLocallyFiniteOrderTop___lam__0(v_a_56_);
lean_dec(v_a_56_);
return v_res_57_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderTop(lean_object* v_x_61_){
_start:
{
lean_object* v_zero_62_; uint8_t v_isZero_63_; 
v_zero_62_ = lean_unsigned_to_nat(0u);
v_isZero_63_ = lean_nat_dec_eq(v_x_61_, v_zero_62_);
if (v_isZero_63_ == 1)
{
lean_object* v___x_64_; 
v___x_64_ = ((lean_object*)(lp_mathlib_Fin_instLocallyFiniteOrderTop___closed__1));
return v___x_64_;
}
else
{
lean_object* v_one_65_; lean_object* v_n_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; lean_object* v_toOrderTop_70_; lean_object* v___x_71_; 
v_one_65_ = lean_unsigned_to_nat(1u);
v_n_66_ = lean_nat_sub(v_x_61_, v_one_65_);
v___x_67_ = lean_nat_add(v_n_66_, v_one_65_);
lean_dec(v_n_66_);
v___x_68_ = lp_mathlib_Fin_instLocallyFiniteOrder(v___x_67_);
v___x_69_ = lp_mathlib_Fin_instCoheytingAlgebra___redArg(v___x_67_);
v_toOrderTop_70_ = lean_ctor_get(v___x_69_, 1);
lean_inc(v_toOrderTop_70_);
lean_dec_ref(v___x_69_);
v___x_71_ = lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg(v___x_68_, v_toOrderTop_70_);
return v___x_71_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLocallyFiniteOrderTop___boxed(lean_object* v_x_72_){
_start:
{
lean_object* v_res_73_; 
v_res_73_ = lp_mathlib_Fin_instLocallyFiniteOrderTop(v_x_72_);
lean_dec(v_x_72_);
return v_res_73_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Fin(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Fin(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Fin(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Set_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Interval_Finset_Fin(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Fin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Fin(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Fin(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Finset_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Interval_Finset_Fin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Interval_Finset_Fin(builtin);
}
#ifdef __cplusplus
}
#endif
