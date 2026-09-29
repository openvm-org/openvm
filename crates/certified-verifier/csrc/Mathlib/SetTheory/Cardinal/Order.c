// Lean compiler output
// Module: Mathlib.SetTheory.Cardinal.Order
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Ring.Canonical public import Mathlib.Data.Fintype.Option public import Mathlib.Order.InitialSeg public import Mathlib.Order.Nat public import Mathlib.Order.SuccPred.CompleteLinearOrder public import Mathlib.SetTheory.Cardinal.Defs public import Mathlib.SetTheory.Cardinal.SchroederBernstein
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
extern lean_object* lp_mathlib_Cardinal_instAdd;
lean_object* l_nsmulRec___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Cardinal_instNatCast___lam__0(lean_object*);
lean_object* lp_mathlib_Quotient_map_u2082___at___00Cardinal_map_u2082_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Cardinal_instMul;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instLE;
static const lean_ctor_object lp_mathlib_Cardinal_partialOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Cardinal_partialOrder___closed__0 = (const lean_object*)&lp_mathlib_Cardinal_partialOrder___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Cardinal_partialOrder = (const lean_object*)&lp_mathlib_Cardinal_partialOrder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ltEmbedding___at___00Cardinal_liftInitialSeg_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ltEmbedding___at___00Cardinal_liftInitialSeg_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_liftInitialSeg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_liftInitialSeg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Cardinal_liftInitialSeg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Cardinal_liftInitialSeg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Cardinal_liftInitialSeg___closed__0 = (const lean_object*)&lp_mathlib_Cardinal_liftInitialSeg___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Cardinal_liftInitialSeg = (const lean_object*)&lp_mathlib_Cardinal_liftInitialSeg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Cardinal_commSemiring___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Cardinal_commSemiring___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Cardinal_commSemiring___closed__0 = (const lean_object*)&lp_mathlib_Cardinal_commSemiring___closed__0_value;
static const lean_closure_object lp_mathlib_Cardinal_commSemiring___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Cardinal_commSemiring___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Cardinal_commSemiring___closed__1 = (const lean_object*)&lp_mathlib_Cardinal_commSemiring___closed__1_value;
static lean_once_cell_t lp_mathlib_Cardinal_commSemiring___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Cardinal_commSemiring___closed__2;
static lean_once_cell_t lp_mathlib_Cardinal_commSemiring___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Cardinal_commSemiring___closed__3;
static lean_once_cell_t lp_mathlib_Cardinal_commSemiring___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Cardinal_commSemiring___closed__4;
static lean_once_cell_t lp_mathlib_Cardinal_commSemiring___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Cardinal_commSemiring___closed__5;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_orderBot;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instCommMonoidWithZero;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instCommMonoid;
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_instWellFoundedRelation;
static lean_object* _init_lp_mathlib_Cardinal_instLE(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_box(0);
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ltEmbedding___at___00Cardinal_liftInitialSeg_spec__0(lean_object* v_f_6_){
_start:
{
lean_inc(v_f_6_);
return v_f_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_ltEmbedding___at___00Cardinal_liftInitialSeg_spec__0___boxed(lean_object* v_f_7_){
_start:
{
lean_object* v_res_8_; 
v_res_8_ = lp_mathlib_OrderEmbedding_ltEmbedding___at___00Cardinal_liftInitialSeg_spec__0(v_f_7_);
lean_dec(v_f_7_);
return v_res_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_liftInitialSeg___lam__0(lean_object* v___y_9_){
_start:
{
lean_object* v___x_10_; 
v___x_10_ = lean_box(0);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_liftInitialSeg___lam__0___boxed(lean_object* v___y_11_){
_start:
{
lean_object* v_res_12_; 
v_res_12_ = lp_mathlib_Cardinal_liftInitialSeg___lam__0(v___y_11_);
lean_dec(v___y_11_);
return v_res_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring___lam__0(lean_object* v_n_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lean_box(0);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring___lam__0___boxed(lean_object* v_n_17_){
_start:
{
lean_object* v_res_18_; 
v_res_18_ = lp_mathlib_Cardinal_commSemiring___lam__0(v_n_17_);
lean_dec(v_n_17_);
return v_res_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring___lam__1(lean_object* v_n_19_, lean_object* v_c_20_){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lp_mathlib_Cardinal_instNatCast___lam__0(v_n_19_);
v___x_22_ = lp_mathlib_Quotient_map_u2082___at___00Cardinal_map_u2082_spec__0(lean_box(0), lean_box(0), v_c_20_, v___x_21_);
lean_dec(v___x_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Cardinal_commSemiring___lam__1___boxed(lean_object* v_n_23_, lean_object* v_c_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_Cardinal_commSemiring___lam__1(v_n_23_, v_c_24_);
lean_dec(v_c_24_);
lean_dec(v_n_23_);
return v_res_25_;
}
}
static lean_object* _init_lp_mathlib_Cardinal_commSemiring___closed__2(void){
_start:
{
lean_object* v___x_28_; lean_object* v___x_29_; 
v___x_28_ = lp_mathlib_Cardinal_instAdd;
v___x_29_ = lean_alloc_closure((void*)(l_nsmulRec___boxed), 5, 3);
lean_closure_set(v___x_29_, 0, lean_box(0));
lean_closure_set(v___x_29_, 1, lean_box(0));
lean_closure_set(v___x_29_, 2, v___x_28_);
return v___x_29_;
}
}
static lean_object* _init_lp_mathlib_Cardinal_commSemiring___closed__3(void){
_start:
{
lean_object* v___x_30_; lean_object* v___x_31_; lean_object* v___x_32_; 
v___x_30_ = lean_obj_once(&lp_mathlib_Cardinal_commSemiring___closed__2, &lp_mathlib_Cardinal_commSemiring___closed__2_once, _init_lp_mathlib_Cardinal_commSemiring___closed__2);
v___x_31_ = lp_mathlib_Cardinal_instAdd;
v___x_32_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_32_, 0, lean_box(0));
lean_ctor_set(v___x_32_, 1, v___x_31_);
lean_ctor_set(v___x_32_, 2, v___x_30_);
return v___x_32_;
}
}
static lean_object* _init_lp_mathlib_Cardinal_commSemiring___closed__4(void){
_start:
{
lean_object* v___f_33_; lean_object* v___x_34_; lean_object* v___x_35_; 
v___f_33_ = ((lean_object*)(lp_mathlib_Cardinal_commSemiring___closed__1));
v___x_34_ = lp_mathlib_Cardinal_instMul;
v___x_35_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_35_, 0, lean_box(0));
lean_ctor_set(v___x_35_, 1, v___x_34_);
lean_ctor_set(v___x_35_, 2, v___f_33_);
return v___x_35_;
}
}
static lean_object* _init_lp_mathlib_Cardinal_commSemiring___closed__5(void){
_start:
{
lean_object* v___f_36_; lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; 
v___f_36_ = ((lean_object*)(lp_mathlib_Cardinal_commSemiring___closed__0));
v___x_37_ = lean_obj_once(&lp_mathlib_Cardinal_commSemiring___closed__4, &lp_mathlib_Cardinal_commSemiring___closed__4_once, _init_lp_mathlib_Cardinal_commSemiring___closed__4);
v___x_38_ = lean_obj_once(&lp_mathlib_Cardinal_commSemiring___closed__3, &lp_mathlib_Cardinal_commSemiring___closed__3_once, _init_lp_mathlib_Cardinal_commSemiring___closed__3);
v___x_39_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_39_, 0, v___x_38_);
lean_ctor_set(v___x_39_, 1, v___x_37_);
lean_ctor_set(v___x_39_, 2, v___f_36_);
return v___x_39_;
}
}
static lean_object* _init_lp_mathlib_Cardinal_commSemiring(void){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lean_obj_once(&lp_mathlib_Cardinal_commSemiring___closed__5, &lp_mathlib_Cardinal_commSemiring___closed__5_once, _init_lp_mathlib_Cardinal_commSemiring___closed__5);
return v___x_40_;
}
}
static lean_object* _init_lp_mathlib_Cardinal_orderBot(void){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lean_box(0);
return v___x_41_;
}
}
static lean_object* _init_lp_mathlib_Cardinal_instCommMonoidWithZero(void){
_start:
{
lean_object* v___x_42_; lean_object* v_toMonoid_43_; lean_object* v___x_44_; 
v___x_42_ = lp_mathlib_Cardinal_commSemiring;
v_toMonoid_43_ = lean_ctor_get(v___x_42_, 1);
lean_inc_ref(v_toMonoid_43_);
v___x_44_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_44_, 0, v_toMonoid_43_);
lean_ctor_set(v___x_44_, 1, lean_box(0));
return v___x_44_;
}
}
static lean_object* _init_lp_mathlib_Cardinal_instCommMonoid(void){
_start:
{
lean_object* v___x_45_; lean_object* v_toMonoid_46_; 
v___x_45_ = lp_mathlib_Cardinal_commSemiring;
v_toMonoid_46_ = lean_ctor_get(v___x_45_, 1);
lean_inc_ref(v_toMonoid_46_);
return v_toMonoid_46_;
}
}
static lean_object* _init_lp_mathlib_Cardinal_instWellFoundedRelation(void){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lean_box(0);
return v___x_47_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Canonical(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Option(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_InitialSeg(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_SuccPred_CompleteLinearOrder(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_SchroederBernstein(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Order(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_InitialSeg(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_SuccPred_CompleteLinearOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_SchroederBernstein(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Cardinal_instLE = _init_lp_mathlib_Cardinal_instLE();
lean_mark_persistent(lp_mathlib_Cardinal_instLE);
lp_mathlib_Cardinal_commSemiring = _init_lp_mathlib_Cardinal_commSemiring();
lean_mark_persistent(lp_mathlib_Cardinal_commSemiring);
lp_mathlib_Cardinal_orderBot = _init_lp_mathlib_Cardinal_orderBot();
lean_mark_persistent(lp_mathlib_Cardinal_orderBot);
lp_mathlib_Cardinal_instCommMonoidWithZero = _init_lp_mathlib_Cardinal_instCommMonoidWithZero();
lean_mark_persistent(lp_mathlib_Cardinal_instCommMonoidWithZero);
lp_mathlib_Cardinal_instCommMonoid = _init_lp_mathlib_Cardinal_instCommMonoid();
lean_mark_persistent(lp_mathlib_Cardinal_instCommMonoid);
lp_mathlib_Cardinal_instWellFoundedRelation = _init_lp_mathlib_Cardinal_instWellFoundedRelation();
lean_mark_persistent(lp_mathlib_Cardinal_instWellFoundedRelation);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_SetTheory_Cardinal_Order(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Canonical(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Option(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_InitialSeg(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_SuccPred_CompleteLinearOrder(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_SchroederBernstein(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Order(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Canonical(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_InitialSeg(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_SuccPred_CompleteLinearOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_SchroederBernstein(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_SetTheory_Cardinal_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_SetTheory_Cardinal_Order(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_SetTheory_Cardinal_Order(builtin);
}
#ifdef __cplusplus
}
#endif
