// Lean compiler output
// Module: Mathlib.Algebra.Field.Rat
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Field.Defs public import Mathlib.Data.NNRat.Defs
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
lean_object* lp_mathlib_Rat_instNNRatCast___lam__0(lean_object*);
lean_object* l_Rat_zpow(lean_object*, lean_object*);
lean_object* l_Rat_mul(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Rat_commRing;
extern lean_object* lp_mathlib_Rat_commGroupWithZero;
lean_object* lp_batteries_instRatCastRat___lam__0(lean_object*);
lean_object* lp_mathlib_Rat_instNNRatCast___lam__0___boxed(lean_object*);
lean_object* lp_batteries_instRatCastRat___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_NNRat_instNNRatCast___lam__0___boxed(lean_object*);
lean_object* l_Rat_div(lean_object*, lean_object*);
extern lean_object* lp_mathlib_instCommSemiringNNRat;
lean_object* lp_mathlib_instDistribOfSemiring___redArg(lean_object*);
lean_object* l_Rat_inv(lean_object*);
lean_object* lp_mathlib_Field_toDivisionRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Rat_instField___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Rat_instField___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_instField___closed__0 = (const lean_object*)&lp_mathlib_Rat_instField___closed__0_value;
static const lean_closure_object lp_mathlib_Rat_instField___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Rat_instField___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_instField___closed__1 = (const lean_object*)&lp_mathlib_Rat_instField___closed__1_value;
static const lean_closure_object lp_mathlib_Rat_instField___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Rat_instNNRatCast___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_instField___closed__2 = (const lean_object*)&lp_mathlib_Rat_instField___closed__2_value;
static const lean_closure_object lp_mathlib_Rat_instField___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_instRatCastRat___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Rat_instField___closed__3 = (const lean_object*)&lp_mathlib_Rat_instField___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField;
static lean_once_cell_t lp_mathlib_Rat_instDivisionRing___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Rat_instDivisionRing___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instDivisionRing;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instInv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instInv___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_NNRat_instInv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NNRat_instInv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NNRat_instInv___closed__0 = (const lean_object*)&lp_mathlib_NNRat_instInv___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_NNRat_instInv = (const lean_object*)&lp_mathlib_NNRat_instInv___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instDiv___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instDiv___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NNRat_instDiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NNRat_instDiv___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NNRat_instDiv___closed__0 = (const lean_object*)&lp_mathlib_NNRat_instDiv___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_NNRat_instDiv = (const lean_object*)&lp_mathlib_NNRat_instDiv___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instZPow___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instZPow___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NNRat_instZPow___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NNRat_instZPow___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NNRat_instZPow___closed__0 = (const lean_object*)&lp_mathlib_NNRat_instZPow___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_NNRat_instZPow = (const lean_object*)&lp_mathlib_NNRat_instZPow___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instSemifield___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instSemifield___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instSemifield___lam__1(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_NNRat_instSemifield___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_NNRat_instSemifield___closed__0;
static const lean_closure_object lp_mathlib_NNRat_instSemifield___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NNRat_instSemifield___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NNRat_instSemifield___closed__1 = (const lean_object*)&lp_mathlib_NNRat_instSemifield___closed__1_value;
static const lean_closure_object lp_mathlib_NNRat_instSemifield___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NNRat_instNNRatCast___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NNRat_instSemifield___closed__2 = (const lean_object*)&lp_mathlib_NNRat_instSemifield___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instSemifield;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__0(lean_object* v_x_1_, lean_object* v___y_2_){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lp_mathlib_Rat_instNNRatCast___lam__0(v_x_1_);
v___x_4_ = l_Rat_mul(v___x_3_, v___y_2_);
lean_dec_ref(v___x_3_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__0___boxed(lean_object* v_x_5_, lean_object* v___y_6_){
_start:
{
lean_object* v_res_7_; 
v_res_7_ = lp_mathlib_Rat_instField___lam__0(v_x_5_, v___y_6_);
lean_dec_ref(v_x_5_);
return v_res_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__1(lean_object* v_x_8_, lean_object* v___y_9_){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_10_ = lp_batteries_instRatCastRat___lam__0(v_x_8_);
v___x_11_ = l_Rat_mul(v___x_10_, v___y_9_);
lean_dec_ref(v___x_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_instField___lam__1___boxed(lean_object* v_x_12_, lean_object* v___y_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_Rat_instField___lam__1(v_x_12_, v___y_13_);
lean_dec_ref(v_x_12_);
return v_res_14_;
}
}
static lean_object* _init_lp_mathlib_Rat_instField(void){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v_toInv_21_; lean_object* v_toDiv_22_; lean_object* v_toZPow_23_; lean_object* v___f_24_; lean_object* v___f_25_; lean_object* v___f_26_; lean_object* v___f_27_; lean_object* v___x_28_; 
v___x_19_ = lp_mathlib_Rat_commRing;
v___x_20_ = lp_mathlib_Rat_commGroupWithZero;
v_toInv_21_ = lean_ctor_get(v___x_20_, 1);
v_toDiv_22_ = lean_ctor_get(v___x_20_, 2);
v_toZPow_23_ = lean_ctor_get(v___x_20_, 3);
v___f_24_ = ((lean_object*)(lp_mathlib_Rat_instField___closed__0));
v___f_25_ = ((lean_object*)(lp_mathlib_Rat_instField___closed__1));
v___f_26_ = ((lean_object*)(lp_mathlib_Rat_instField___closed__2));
v___f_27_ = ((lean_object*)(lp_mathlib_Rat_instField___closed__3));
lean_inc(v_toZPow_23_);
lean_inc(v_toDiv_22_);
lean_inc(v_toInv_21_);
v___x_28_ = lean_alloc_ctor(0, 8, 0);
lean_ctor_set(v___x_28_, 0, v___x_19_);
lean_ctor_set(v___x_28_, 1, v_toInv_21_);
lean_ctor_set(v___x_28_, 2, v_toDiv_22_);
lean_ctor_set(v___x_28_, 3, v_toZPow_23_);
lean_ctor_set(v___x_28_, 4, v___f_26_);
lean_ctor_set(v___x_28_, 5, v___f_27_);
lean_ctor_set(v___x_28_, 6, v___f_24_);
lean_ctor_set(v___x_28_, 7, v___f_25_);
return v___x_28_;
}
}
static lean_object* _init_lp_mathlib_Rat_instDivisionRing___closed__0(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; 
v___x_29_ = lp_mathlib_Rat_instField;
v___x_30_ = lp_mathlib_Field_toDivisionRing___redArg(v___x_29_);
return v___x_30_;
}
}
static lean_object* _init_lp_mathlib_Rat_instDivisionRing(void){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lean_obj_once(&lp_mathlib_Rat_instDivisionRing___closed__0, &lp_mathlib_Rat_instDivisionRing___closed__0_once, _init_lp_mathlib_Rat_instDivisionRing___closed__0);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instInv___lam__0(lean_object* v_x_32_){
_start:
{
lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_33_ = lp_mathlib_Rat_instNNRatCast___lam__0(v_x_32_);
v___x_34_ = l_Rat_inv(v___x_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instInv___lam__0___boxed(lean_object* v_x_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_NNRat_instInv___lam__0(v_x_35_);
lean_dec_ref(v_x_35_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instDiv___lam__0(lean_object* v_x_39_, lean_object* v_y_40_){
_start:
{
lean_object* v___x_41_; lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_41_ = lp_mathlib_Rat_instNNRatCast___lam__0(v_x_39_);
v___x_42_ = lp_mathlib_Rat_instNNRatCast___lam__0(v_y_40_);
v___x_43_ = l_Rat_div(v___x_41_, v___x_42_);
lean_dec_ref(v___x_41_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instDiv___lam__0___boxed(lean_object* v_x_44_, lean_object* v_y_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_NNRat_instDiv___lam__0(v_x_44_, v_y_45_);
lean_dec_ref(v_y_45_);
lean_dec_ref(v_x_44_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instZPow___lam__0(lean_object* v_x_49_, lean_object* v_n_50_){
_start:
{
lean_object* v___x_51_; lean_object* v___x_52_; 
v___x_51_ = lp_mathlib_Rat_instNNRatCast___lam__0(v_x_49_);
v___x_52_ = l_Rat_zpow(v___x_51_, v_n_50_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instZPow___lam__0___boxed(lean_object* v_x_53_, lean_object* v_n_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_mathlib_NNRat_instZPow___lam__0(v_x_53_, v_n_54_);
lean_dec(v_n_54_);
lean_dec_ref(v_x_53_);
return v_res_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instSemifield___lam__0(lean_object* v_n_58_, lean_object* v_a_59_){
_start:
{
lean_object* v___x_60_; lean_object* v___x_61_; 
v___x_60_ = lp_mathlib_Rat_instNNRatCast___lam__0(v_a_59_);
v___x_61_ = l_Rat_zpow(v___x_60_, v_n_58_);
return v___x_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instSemifield___lam__0___boxed(lean_object* v_n_62_, lean_object* v_a_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_NNRat_instSemifield___lam__0(v_n_62_, v_a_63_);
lean_dec_ref(v_a_63_);
lean_dec(v_n_62_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instSemifield___lam__1(lean_object* v_toMul_65_, lean_object* v_q_66_, lean_object* v_a_67_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lean_apply_2(v_toMul_65_, v_q_66_, v_a_67_);
return v___x_68_;
}
}
static lean_object* _init_lp_mathlib_NNRat_instSemifield___closed__0(void){
_start:
{
lean_object* v___x_69_; lean_object* v___x_70_; 
v___x_69_ = lp_mathlib_instCommSemiringNNRat;
v___x_70_ = lp_mathlib_instDistribOfSemiring___redArg(v___x_69_);
return v___x_70_;
}
}
static lean_object* _init_lp_mathlib_NNRat_instSemifield(void){
_start:
{
lean_object* v___x_73_; lean_object* v___x_74_; lean_object* v_toMul_75_; lean_object* v___f_76_; lean_object* v___f_77_; lean_object* v___f_78_; lean_object* v___f_79_; lean_object* v___f_80_; lean_object* v___x_81_; 
v___x_73_ = lp_mathlib_instCommSemiringNNRat;
v___x_74_ = lean_obj_once(&lp_mathlib_NNRat_instSemifield___closed__0, &lp_mathlib_NNRat_instSemifield___closed__0_once, _init_lp_mathlib_NNRat_instSemifield___closed__0);
v_toMul_75_ = lean_ctor_get(v___x_74_, 0);
v___f_76_ = ((lean_object*)(lp_mathlib_NNRat_instSemifield___closed__1));
v___f_77_ = ((lean_object*)(lp_mathlib_NNRat_instInv___closed__0));
v___f_78_ = ((lean_object*)(lp_mathlib_NNRat_instDiv___closed__0));
v___f_79_ = ((lean_object*)(lp_mathlib_NNRat_instSemifield___closed__2));
lean_inc(v_toMul_75_);
v___f_80_ = lean_alloc_closure((void*)(lp_mathlib_NNRat_instSemifield___lam__1), 3, 1);
lean_closure_set(v___f_80_, 0, v_toMul_75_);
v___x_81_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_81_, 0, v___x_73_);
lean_ctor_set(v___x_81_, 1, v___f_77_);
lean_ctor_set(v___x_81_, 2, v___f_78_);
lean_ctor_set(v___x_81_, 3, v___f_76_);
lean_ctor_set(v___x_81_, 4, v___f_79_);
lean_ctor_set(v___x_81_, 5, v___f_80_);
return v___x_81_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_NNRat_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Field_Rat(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_NNRat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Rat_instField = _init_lp_mathlib_Rat_instField();
lean_mark_persistent(lp_mathlib_Rat_instField);
lp_mathlib_Rat_instDivisionRing = _init_lp_mathlib_Rat_instDivisionRing();
lean_mark_persistent(lp_mathlib_Rat_instDivisionRing);
lp_mathlib_NNRat_instSemifield = _init_lp_mathlib_NNRat_instSemifield();
lean_mark_persistent(lp_mathlib_NNRat_instSemifield);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Field_Rat(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_NNRat_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Field_Rat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_NNRat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Field_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Field_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Field_Rat(builtin);
}
#ifdef __cplusplus
}
#endif
