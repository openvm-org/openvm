// Lean compiler output
// Module: Mathlib.Data.Int.Cast.Lemmas
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.TypeTags.Hom public import Mathlib.Algebra.Ring.Int.Defs public import Mathlib.Algebra.Ring.Parity
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
extern lean_object* lp_mathlib_Int_instAddMonoid;
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_Additive_subNegMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Additive_ofMul(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeft(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Int_instCommSemiring;
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* l_Int_cast(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_castRingHom___redArg(lean_object*);
lean_object* lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(lean_object*);
static lean_once_cell_t lp_mathlib_Int_ofNatHom___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_ofNatHom___closed__0;
static lean_once_cell_t lp_mathlib_Int_ofNatHom___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_ofNatHom___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Int_ofNatHom;
LEAN_EXPORT lean_object* lp_mathlib_Int_castAddHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_castAddHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_castRingHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_castRingHom(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_zmultiplesHom___redArg___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_zmultiplesHom___redArg___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_zmultiplesHom___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zmultiplesHom___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_zmultiplesHom___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_zmultiplesHom___redArg___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_zmultiplesHom___redArg___closed__0 = (const lean_object*)&lp_mathlib_zmultiplesHom___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_zmultiplesHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zmultiplesHom(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_zpowersHom___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_zpowersHom___redArg___closed__0;
static lean_once_cell_t lp_mathlib_zpowersHom___redArg___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_zpowersHom___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_zpowersHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zpowersHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zmultiplesAddHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zmultiplesAddHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zpowersMulHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_zpowersMulHom(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Int_ofNatHom___closed__0(void){
_start:
{
lean_object* v___x_1_; lean_object* v___x_2_; 
v___x_1_ = lp_mathlib_Int_instCommSemiring;
v___x_2_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v___x_1_);
return v___x_2_;
}
}
static lean_object* _init_lp_mathlib_Int_ofNatHom___closed__1(void){
_start:
{
lean_object* v___x_3_; lean_object* v___x_4_; 
v___x_3_ = lean_obj_once(&lp_mathlib_Int_ofNatHom___closed__0, &lp_mathlib_Int_ofNatHom___closed__0_once, _init_lp_mathlib_Int_ofNatHom___closed__0);
v___x_4_ = lp_mathlib_Nat_castRingHom___redArg(v___x_3_);
return v___x_4_;
}
}
static lean_object* _init_lp_mathlib_Int_ofNatHom(void){
_start:
{
lean_object* v___x_5_; 
v___x_5_ = lean_obj_once(&lp_mathlib_Int_ofNatHom___closed__1, &lp_mathlib_Int_ofNatHom___closed__1_once, _init_lp_mathlib_Int_ofNatHom___closed__1);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_castAddHom___redArg(lean_object* v_inst_6_){
_start:
{
lean_object* v_toIntCast_7_; lean_object* v___x_8_; 
v_toIntCast_7_ = lean_ctor_get(v_inst_6_, 0);
lean_inc(v_toIntCast_7_);
lean_dec_ref(v_inst_6_);
v___x_8_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_8_, 0, lean_box(0));
lean_closure_set(v___x_8_, 1, v_toIntCast_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_castAddHom(lean_object* v_00_u03b1_9_, lean_object* v_inst_10_){
_start:
{
lean_object* v___x_11_; 
v___x_11_ = lp_mathlib_Int_castAddHom___redArg(v_inst_10_);
return v___x_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_castRingHom___redArg(lean_object* v_inst_12_){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v_toIntCast_15_; lean_object* v___x_16_; 
v___x_13_ = lp_mathlib_NonAssocRing_toAddCommGroupWithOne___redArg(v_inst_12_);
v___x_14_ = lp_mathlib_AddCommGroupWithOne_toAddGroupWithOne___redArg(v___x_13_);
lean_dec_ref(v___x_13_);
v_toIntCast_15_ = lean_ctor_get(v___x_14_, 0);
lean_inc(v_toIntCast_15_);
lean_dec_ref(v___x_14_);
v___x_16_ = lean_alloc_closure((void*)(l_Int_cast), 3, 2);
lean_closure_set(v___x_16_, 0, lean_box(0));
lean_closure_set(v___x_16_, 1, v_toIntCast_15_);
return v___x_16_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_castRingHom(lean_object* v_00_u03b1_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v___x_19_; 
v___x_19_ = lp_mathlib_Int_castRingHom___redArg(v_inst_18_);
return v___x_19_;
}
}
static lean_object* _init_lp_mathlib_zmultiplesHom___redArg___lam__0___closed__0(void){
_start:
{
lean_object* v___x_20_; lean_object* v___x_21_; 
v___x_20_ = lean_unsigned_to_nat(1u);
v___x_21_ = lean_nat_to_int(v___x_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zmultiplesHom___redArg___lam__0(lean_object* v_f_22_){
_start:
{
lean_object* v___x_23_; lean_object* v___x_24_; 
v___x_23_ = lean_obj_once(&lp_mathlib_zmultiplesHom___redArg___lam__0___closed__0, &lp_mathlib_zmultiplesHom___redArg___lam__0___closed__0_once, _init_lp_mathlib_zmultiplesHom___redArg___lam__0___closed__0);
v___x_24_ = lean_apply_1(v_f_22_, v___x_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zmultiplesHom___redArg___lam__1(lean_object* v_toZSMul_25_, lean_object* v_x_26_, lean_object* v___y_27_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lean_apply_2(v_toZSMul_25_, v___y_27_, v_x_26_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zmultiplesHom___redArg(lean_object* v_inst_30_){
_start:
{
lean_object* v_toZSMul_31_; lean_object* v___f_32_; lean_object* v___f_33_; lean_object* v___x_34_; 
v_toZSMul_31_ = lean_ctor_get(v_inst_30_, 3);
lean_inc(v_toZSMul_31_);
lean_dec_ref(v_inst_30_);
v___f_32_ = ((lean_object*)(lp_mathlib_zmultiplesHom___redArg___closed__0));
v___f_33_ = lean_alloc_closure((void*)(lp_mathlib_zmultiplesHom___redArg___lam__1), 3, 1);
lean_closure_set(v___f_33_, 0, v_toZSMul_31_);
v___x_34_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_34_, 0, v___f_33_);
lean_ctor_set(v___x_34_, 1, v___f_32_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zmultiplesHom(lean_object* v_00_u03b2_35_, lean_object* v_inst_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib_zmultiplesHom___redArg(v_inst_36_);
return v___x_37_;
}
}
static lean_object* _init_lp_mathlib_zpowersHom___redArg___closed__0(void){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_38_ = lp_mathlib_Int_instAddMonoid;
v___x_39_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_38_);
return v___x_39_;
}
}
static lean_object* _init_lp_mathlib_zpowersHom___redArg___closed__1(void){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_mathlib_Additive_ofMul(lean_box(0));
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowersHom___redArg(lean_object* v_inst_41_){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v_toMonoid_44_; lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_42_ = lean_obj_once(&lp_mathlib_zpowersHom___redArg___closed__0, &lp_mathlib_zpowersHom___redArg___closed__0_once, _init_lp_mathlib_zpowersHom___redArg___closed__0);
lean_inc_ref(v_inst_41_);
v___x_43_ = lp_mathlib_Additive_subNegMonoid___redArg(v_inst_41_);
v_toMonoid_44_ = lean_ctor_get(v_inst_41_, 0);
lean_inc_ref(v_toMonoid_44_);
lean_dec_ref(v_inst_41_);
v___x_45_ = lean_obj_once(&lp_mathlib_zpowersHom___redArg___closed__1, &lp_mathlib_zpowersHom___redArg___closed__1_once, _init_lp_mathlib_zpowersHom___redArg___closed__1);
v___x_46_ = lp_mathlib_zmultiplesHom___redArg(v___x_43_);
v___x_47_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_toMonoid_44_);
lean_dec_ref(v_toMonoid_44_);
v___x_48_ = lp_mathlib_AddMonoidHom_toMultiplicativeLeft(lean_box(0), lean_box(0), v___x_42_, v___x_47_);
lean_dec_ref(v___x_47_);
v___x_49_ = lp_mathlib_Equiv_trans___redArg(v___x_46_, v___x_48_);
v___x_50_ = lp_mathlib_Equiv_trans___redArg(v___x_45_, v___x_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowersHom(lean_object* v_00_u03b1_51_, lean_object* v_inst_52_){
_start:
{
lean_object* v___x_53_; 
v___x_53_ = lp_mathlib_zpowersHom___redArg(v_inst_52_);
return v___x_53_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zmultiplesAddHom___redArg(lean_object* v_inst_54_){
_start:
{
lean_object* v___x_55_; 
v___x_55_ = lp_mathlib_zmultiplesHom___redArg(v_inst_54_);
return v___x_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zmultiplesAddHom(lean_object* v_00_u03b2_56_, lean_object* v_inst_57_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lp_mathlib_zmultiplesHom___redArg(v_inst_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowersMulHom___redArg(lean_object* v_inst_59_){
_start:
{
lean_object* v___x_60_; 
v___x_60_ = lp_mathlib_zpowersHom___redArg(v_inst_59_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_zpowersMulHom(lean_object* v_00_u03b1_61_, lean_object* v_inst_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lp_mathlib_zpowersHom___redArg(v_inst_62_);
return v___x_63_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Parity(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Parity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Int_ofNatHom = _init_lp_mathlib_Int_ofNatHom();
lean_mark_persistent(lp_mathlib_Int_ofNatHom);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Parity(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Parity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(builtin);
}
#ifdef __cplusplus
}
#endif
