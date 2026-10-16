// Lean compiler output
// Module: Mathlib.Tactic.Bound.Init
// Imports: public import Init public meta import Init public import Mathlib.Init public import Aesop.Frontend.Command
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Array_mkArray1___redArg(lean_object*);
lean_object* lean_array_get_size(lean_object*);
size_t lean_usize_of_nat(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lp_aesop_Aesop_Frontend_declareRuleSetUnchecked(lean_object*, uint8_t);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Bound_Init_0__initFn_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__spec__0(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Bound_Init_0__initFn_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__0_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Bound"};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__0_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__0_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__1_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__0_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__value),LEAN_SCALAR_PTR_LITERAL(248, 144, 81, 165, 73, 52, 205, 25)}};
static const lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__1_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_ = (const lean_object*)&lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__1_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__value;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__4_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__4_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__once = LEAN_ONCE_CELL_INITIALIZER;
static uint8_t lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_;
static lean_once_cell_t lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__6_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__once = LEAN_ONCE_CELL_INITIALIZER;
static size_t lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__6_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Bound_Init_0__initFn_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__spec__0(lean_object* v_as_1_, size_t v_i_2_, size_t v_stop_3_, lean_object* v_b_4_){
_start:
{
uint8_t v___x_6_; 
v___x_6_ = lean_usize_dec_eq(v_i_2_, v_stop_3_);
if (v___x_6_ == 0)
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = lean_array_uget_borrowed(v_as_1_, v_i_2_);
lean_inc(v___x_7_);
v___x_8_ = lp_aesop_Aesop_Frontend_declareRuleSetUnchecked(v___x_7_, v___x_6_);
if (lean_obj_tag(v___x_8_) == 0)
{
lean_object* v_a_9_; size_t v___x_10_; size_t v___x_11_; 
v_a_9_ = lean_ctor_get(v___x_8_, 0);
lean_inc(v_a_9_);
lean_dec_ref_known(v___x_8_, 1);
v___x_10_ = ((size_t)1ULL);
v___x_11_ = lean_usize_add(v_i_2_, v___x_10_);
v_i_2_ = v___x_11_;
v_b_4_ = v_a_9_;
goto _start;
}
else
{
return v___x_8_;
}
}
else
{
lean_object* v___x_13_; 
v___x_13_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_13_, 0, v_b_4_);
return v___x_13_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Bound_Init_0__initFn_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__spec__0___boxed(lean_object* v_as_14_, lean_object* v_i_15_, lean_object* v_stop_16_, lean_object* v_b_17_, lean_object* v___y_18_){
_start:
{
size_t v_i_boxed_19_; size_t v_stop_boxed_20_; lean_object* v_res_21_; 
v_i_boxed_19_ = lean_unbox_usize(v_i_15_);
lean_dec(v_i_15_);
v_stop_boxed_20_ = lean_unbox_usize(v_stop_16_);
lean_dec(v_stop_16_);
v_res_21_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Bound_Init_0__initFn_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__spec__0(v_as_14_, v_i_boxed_19_, v_stop_boxed_20_, v_b_17_);
lean_dec_ref(v_as_14_);
return v_res_21_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_(void){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; 
v___x_25_ = ((lean_object*)(lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__1_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_));
v___x_26_ = l_Array_mkArray1___redArg(v___x_25_);
return v___x_26_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_27_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_, &lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_);
v___x_28_ = lean_array_get_size(v___x_27_);
return v___x_28_;
}
}
static uint8_t _init_lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__4_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; uint8_t v___x_31_; 
v___x_29_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_, &lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_);
v___x_30_ = lean_unsigned_to_nat(0u);
v___x_31_ = lean_nat_dec_lt(v___x_30_, v___x_29_);
return v___x_31_;
}
}
static uint8_t _init_lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_(void){
_start:
{
lean_object* v___x_32_; uint8_t v___x_33_; 
v___x_32_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_, &lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_);
v___x_33_ = lean_nat_dec_le(v___x_32_, v___x_32_);
return v___x_33_;
}
}
static size_t _init_lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__6_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_(void){
_start:
{
lean_object* v___x_34_; size_t v___x_35_; 
v___x_34_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_, &lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__3_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_);
v___x_35_ = lean_usize_of_nat(v___x_34_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_(){
_start:
{
lean_object* v___x_37_; lean_object* v___x_38_; uint8_t v___x_39_; 
v___x_37_ = lean_obj_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_, &lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__2_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_);
v___x_38_ = lean_box(0);
v___x_39_ = lean_uint8_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__4_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_, &lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__4_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__4_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_);
if (v___x_39_ == 0)
{
lean_object* v___x_40_; 
v___x_40_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_40_, 0, v___x_38_);
return v___x_40_;
}
else
{
uint8_t v___x_41_; 
v___x_41_ = lean_uint8_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_, &lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__5_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_);
if (v___x_41_ == 0)
{
if (v___x_39_ == 0)
{
lean_object* v___x_42_; 
v___x_42_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_42_, 0, v___x_38_);
return v___x_42_;
}
else
{
size_t v___x_43_; size_t v___x_44_; lean_object* v___x_45_; 
v___x_43_ = ((size_t)0ULL);
v___x_44_ = lean_usize_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__6_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_, &lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__6_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__6_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_);
v___x_45_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Bound_Init_0__initFn_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__spec__0(v___x_37_, v___x_43_, v___x_44_, v___x_38_);
return v___x_45_;
}
}
else
{
size_t v___x_46_; size_t v___x_47_; lean_object* v___x_48_; 
v___x_46_ = ((size_t)0ULL);
v___x_47_ = lean_usize_once(&lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__6_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_, &lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__6_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__once, _init_lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn___closed__6_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_);
v___x_48_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__private_Mathlib_Tactic_Bound_Init_0__initFn_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3__spec__0(v___x_37_, v___x_46_, v___x_47_, v___x_38_);
return v___x_48_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3____boxed(lean_object* v_a_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_();
return v_res_50_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_aesop_Aesop_Frontend_Command(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Bound_Init(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_aesop_Aesop_Frontend_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Bound_Init(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Tactic_Bound_Init_0__initFn_00___x40_Mathlib_Tactic_Bound_Init_1206815526____hygCtx___hyg_3_();
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Frontend_Command(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Bound_Init(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Frontend_Command(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Bound_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Bound_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Bound_Init(builtin);
}
#ifdef __cplusplus
}
#endif
