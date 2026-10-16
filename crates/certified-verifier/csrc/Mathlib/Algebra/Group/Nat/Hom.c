// Lean compiler output
// Module: Mathlib.Algebra.Group.Nat.Hom
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Nat.Defs public import Mathlib.Algebra.Group.TypeTags.Hom public import Mathlib.Tactic.Spread
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
extern lean_object* lp_mathlib_Nat_instAddCancelCommMonoid;
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Additive_ofMul(lean_object*);
lean_object* lp_mathlib_Additive_addMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeft(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_multiplesHom___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_multiplesHom___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_multiplesHom___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_multiplesHom___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_multiplesHom___redArg___closed__0 = (const lean_object*)&lp_mathlib_multiplesHom___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_multiplesHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_multiplesHom(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_powersHom___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_powersHom___redArg___closed__0;
static lean_once_cell_t lp_mathlib_powersHom___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_powersHom___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_powersHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_powersHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_multiplesAddHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_multiplesAddHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_powersMulHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_powersMulHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_multiplesHom___redArg___lam__0(lean_object* v_f_1_){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lean_unsigned_to_nat(1u);
v___x_3_ = lean_apply_1(v_f_1_, v___x_2_);
return v___x_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_multiplesHom___redArg___lam__1(lean_object* v_toNSMul_4_, lean_object* v_x_5_, lean_object* v___y_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lean_apply_2(v_toNSMul_4_, v___y_6_, v_x_5_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_multiplesHom___redArg(lean_object* v_inst_9_){
_start:
{
lean_object* v_toNSMul_10_; lean_object* v___f_11_; lean_object* v___f_12_; lean_object* v___x_13_; 
v_toNSMul_10_ = lean_ctor_get(v_inst_9_, 2);
lean_inc(v_toNSMul_10_);
lean_dec_ref(v_inst_9_);
v___f_11_ = ((lean_object*)(lp_mathlib_multiplesHom___redArg___closed__0));
v___f_12_ = lean_alloc_closure((void*)(lp_mathlib_multiplesHom___redArg___lam__1), 3, 1);
lean_closure_set(v___f_12_, 0, v_toNSMul_10_);
v___x_13_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_13_, 0, v___f_12_);
lean_ctor_set(v___x_13_, 1, v___f_11_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_multiplesHom(lean_object* v_M_14_, lean_object* v_inst_15_){
_start:
{
lean_object* v___x_16_; 
v___x_16_ = lp_mathlib_multiplesHom___redArg(v_inst_15_);
return v___x_16_;
}
}
static lean_object* _init_lp_mathlib_powersHom___redArg___closed__0(void){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_17_ = lp_mathlib_Nat_instAddCancelCommMonoid;
v___x_18_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_17_);
return v___x_18_;
}
}
static lean_object* _init_lp_mathlib_powersHom___redArg___closed__1(void){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_mathlib_Additive_ofMul(lean_box(0));
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_powersHom___redArg(lean_object* v_inst_20_){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_21_ = lean_obj_once(&lp_mathlib_powersHom___redArg___closed__0, &lp_mathlib_powersHom___redArg___closed__0_once, _init_lp_mathlib_powersHom___redArg___closed__0);
v___x_22_ = lean_obj_once(&lp_mathlib_powersHom___redArg___closed__1, &lp_mathlib_powersHom___redArg___closed__1_once, _init_lp_mathlib_powersHom___redArg___closed__1);
lean_inc_ref(v_inst_20_);
v___x_23_ = lp_mathlib_Additive_addMonoid___redArg(v_inst_20_);
v___x_24_ = lp_mathlib_multiplesHom___redArg(v___x_23_);
v___x_25_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_20_);
lean_dec_ref(v_inst_20_);
v___x_26_ = lp_mathlib_AddMonoidHom_toMultiplicativeLeft(lean_box(0), lean_box(0), v___x_21_, v___x_25_);
lean_dec_ref(v___x_25_);
v___x_27_ = lp_mathlib_Equiv_trans___redArg(v___x_24_, v___x_26_);
v___x_28_ = lp_mathlib_Equiv_trans___redArg(v___x_22_, v___x_27_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_powersHom(lean_object* v_M_29_, lean_object* v_inst_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lp_mathlib_powersHom___redArg(v_inst_30_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_multiplesAddHom___redArg(lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_multiplesHom___redArg(v_inst_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_multiplesAddHom(lean_object* v_M_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_multiplesHom___redArg(v_inst_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_powersMulHom___redArg(lean_object* v_inst_37_){
_start:
{
lean_object* v___x_38_; 
v___x_38_ = lp_mathlib_powersHom___redArg(v_inst_37_);
return v___x_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_powersMulHom(lean_object* v_M_39_, lean_object* v_inst_40_){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lp_mathlib_powersHom___redArg(v_inst_40_);
return v___x_41_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Hom(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Nat_Hom(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Hom(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Nat_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Nat_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Nat_Hom(builtin);
}
#ifdef __cplusplus
}
#endif
