// Lean compiler output
// Module: Mathlib.Algebra.GCDMonoid.Nat
// Imports: public import Init public meta import Init public import Mathlib.Algebra.GCDMonoid.Basic public import Mathlib.Algebra.Order.Group.Unbundled.Int public import Mathlib.Algebra.Ring.Int.Units public import Mathlib.Algebra.GroupWithZero.Nat
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
lean_object* l_Nat_lcm___boxed(lean_object*, lean_object*);
lean_object* l_Nat_gcd___boxed(lean_object*, lean_object*);
lean_object* l_Int_gcd(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
uint8_t lean_int_dec_le(lean_object*, lean_object*);
lean_object* lean_int_neg(lean_object*);
lean_object* lean_int_mul(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
extern lean_object* lp_mathlib_Nat_instCommMonoidWithZero;
lean_object* lp_mathlib_instStrongNormalizationMonoid___redArg(lean_object*);
lean_object* l_Int_lcm(lean_object*, lean_object*);
lean_object* l_Int_ofNat___boxed(lean_object*);
static const lean_closure_object lp_mathlib_instGCDMonoidNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_gcd___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instGCDMonoidNat___closed__0 = (const lean_object*)&lp_mathlib_instGCDMonoidNat___closed__0_value;
static const lean_closure_object lp_mathlib_instGCDMonoidNat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_lcm___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instGCDMonoidNat___closed__1 = (const lean_object*)&lp_mathlib_instGCDMonoidNat___closed__1_value;
static const lean_ctor_object lp_mathlib_instGCDMonoidNat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_instGCDMonoidNat___closed__0_value),((lean_object*)&lp_mathlib_instGCDMonoidNat___closed__1_value)}};
static const lean_object* lp_mathlib_instGCDMonoidNat___closed__2 = (const lean_object*)&lp_mathlib_instGCDMonoidNat___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_instGCDMonoidNat = (const lean_object*)&lp_mathlib_instGCDMonoidNat___closed__2_value;
static lean_once_cell_t lp_mathlib_instStrongNormalizedGCDMonoidNat___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instStrongNormalizedGCDMonoidNat___closed__0;
static lean_once_cell_t lp_mathlib_instStrongNormalizedGCDMonoidNat___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instStrongNormalizedGCDMonoidNat___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_instStrongNormalizedGCDMonoidNat;
static lean_once_cell_t lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__1;
static lean_once_cell_t lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__2;
static lean_once_cell_t lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__3;
static lean_once_cell_t lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Int_strongNormalizationMonoid___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_strongNormalizationMonoid___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Int_strongNormalizationMonoid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Int_strongNormalizationMonoid___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Int_strongNormalizationMonoid___closed__0 = (const lean_object*)&lp_mathlib_Int_strongNormalizationMonoid___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Int_strongNormalizationMonoid = (const lean_object*)&lp_mathlib_Int_strongNormalizationMonoid___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Int_normalizationMonoid = (const lean_object*)&lp_mathlib_Int_strongNormalizationMonoid___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Int_instGCDMonoid___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instGCDMonoid___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instGCDMonoid___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instGCDMonoid___lam__1___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Int_instGCDMonoid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Int_instGCDMonoid___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Int_instGCDMonoid___closed__0 = (const lean_object*)&lp_mathlib_Int_instGCDMonoid___closed__0_value;
static const lean_closure_object lp_mathlib_Int_instGCDMonoid___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Int_instGCDMonoid___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Int_instGCDMonoid___closed__1 = (const lean_object*)&lp_mathlib_Int_instGCDMonoid___closed__1_value;
static const lean_ctor_object lp_mathlib_Int_instGCDMonoid___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Int_instGCDMonoid___closed__1_value),((lean_object*)&lp_mathlib_Int_instGCDMonoid___closed__0_value)}};
static const lean_object* lp_mathlib_Int_instGCDMonoid___closed__2 = (const lean_object*)&lp_mathlib_Int_instGCDMonoid___closed__2_value;
LEAN_EXPORT const lean_object* lp_mathlib_Int_instGCDMonoid = (const lean_object*)&lp_mathlib_Int_instGCDMonoid___closed__2_value;
static const lean_ctor_object lp_mathlib_Int_instStrongNormalizedGCDMonoid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Int_strongNormalizationMonoid___closed__0_value),((lean_object*)&lp_mathlib_Int_instGCDMonoid___closed__2_value)}};
static const lean_object* lp_mathlib_Int_instStrongNormalizedGCDMonoid___closed__0 = (const lean_object*)&lp_mathlib_Int_instStrongNormalizedGCDMonoid___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Int_instStrongNormalizedGCDMonoid = (const lean_object*)&lp_mathlib_Int_instStrongNormalizedGCDMonoid___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00associatesIntEquivNat_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_out___at___00associatesIntEquivNat_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Associates_out___at___00associatesIntEquivNat_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_associatesIntEquivNat___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_associatesIntEquivNat___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_associatesIntEquivNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_associatesIntEquivNat___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_associatesIntEquivNat___closed__0 = (const lean_object*)&lp_mathlib_associatesIntEquivNat___closed__0_value;
static lean_once_cell_t lp_mathlib_associatesIntEquivNat___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_associatesIntEquivNat___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_associatesIntEquivNat;
static lean_object* _init_lp_mathlib_instStrongNormalizedGCDMonoidNat___closed__0(void){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = lp_mathlib_Nat_instCommMonoidWithZero;
v___x_8_ = lp_mathlib_instStrongNormalizationMonoid___redArg(v___x_7_);
return v___x_8_;
}
}
static lean_object* _init_lp_mathlib_instStrongNormalizedGCDMonoidNat___closed__1(void){
_start:
{
lean_object* v___x_9_; lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_9_ = ((lean_object*)(lp_mathlib_instGCDMonoidNat));
v___x_10_ = lean_obj_once(&lp_mathlib_instStrongNormalizedGCDMonoidNat___closed__0, &lp_mathlib_instStrongNormalizedGCDMonoidNat___closed__0_once, _init_lp_mathlib_instStrongNormalizedGCDMonoidNat___closed__0);
v___x_11_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_11_, 0, v___x_10_);
lean_ctor_set(v___x_11_, 1, v___x_9_);
return v___x_11_;
}
}
static lean_object* _init_lp_mathlib_instStrongNormalizedGCDMonoidNat(void){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_obj_once(&lp_mathlib_instStrongNormalizedGCDMonoidNat___closed__1, &lp_mathlib_instStrongNormalizedGCDMonoidNat___closed__1_once, _init_lp_mathlib_instStrongNormalizedGCDMonoidNat___closed__1);
return v___x_12_;
}
}
static lean_object* _init_lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__0(void){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_13_ = lean_unsigned_to_nat(0u);
v___x_14_ = lean_nat_to_int(v___x_13_);
return v___x_14_;
}
}
static lean_object* _init_lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__1(void){
_start:
{
lean_object* v___x_15_; lean_object* v___x_16_; 
v___x_15_ = lean_unsigned_to_nat(1u);
v___x_16_ = lean_nat_to_int(v___x_15_);
return v___x_16_;
}
}
static lean_object* _init_lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__2(void){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_17_ = lean_obj_once(&lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__1, &lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__1_once, _init_lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__1);
v___x_18_ = lean_int_neg(v___x_17_);
return v___x_18_;
}
}
static lean_object* _init_lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__3(void){
_start:
{
lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_19_ = lean_obj_once(&lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__2, &lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__2_once, _init_lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__2);
v___x_20_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_20_, 0, v___x_19_);
lean_ctor_set(v___x_20_, 1, v___x_19_);
return v___x_20_;
}
}
static lean_object* _init_lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__4(void){
_start:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lean_obj_once(&lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__1, &lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__1_once, _init_lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__1);
v___x_22_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_22_, 0, v___x_21_);
lean_ctor_set(v___x_22_, 1, v___x_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_strongNormalizationMonoid___lam__0(lean_object* v_a_23_){
_start:
{
lean_object* v___x_24_; uint8_t v___x_25_; 
v___x_24_ = lean_obj_once(&lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__0, &lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__0_once, _init_lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__0);
v___x_25_ = lean_int_dec_le(v___x_24_, v_a_23_);
if (v___x_25_ == 0)
{
lean_object* v___x_26_; 
v___x_26_ = lean_obj_once(&lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__3, &lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__3_once, _init_lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__3);
return v___x_26_;
}
else
{
lean_object* v___x_27_; 
v___x_27_ = lean_obj_once(&lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__4, &lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__4_once, _init_lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__4);
return v___x_27_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_strongNormalizationMonoid___lam__0___boxed(lean_object* v_a_28_){
_start:
{
lean_object* v_res_29_; 
v_res_29_ = lp_mathlib_Int_strongNormalizationMonoid___lam__0(v_a_28_);
lean_dec(v_a_28_);
return v_res_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instGCDMonoid___lam__0(lean_object* v_a_33_, lean_object* v_b_34_){
_start:
{
lean_object* v___x_35_; lean_object* v___x_36_; 
v___x_35_ = l_Int_lcm(v_a_33_, v_b_34_);
v___x_36_ = lean_nat_to_int(v___x_35_);
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instGCDMonoid___lam__0___boxed(lean_object* v_a_37_, lean_object* v_b_38_){
_start:
{
lean_object* v_res_39_; 
v_res_39_ = lp_mathlib_Int_instGCDMonoid___lam__0(v_a_37_, v_b_38_);
lean_dec(v_b_38_);
lean_dec(v_a_37_);
return v_res_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instGCDMonoid___lam__1(lean_object* v_a_40_, lean_object* v_b_41_){
_start:
{
lean_object* v___x_42_; lean_object* v___x_43_; 
v___x_42_ = l_Int_gcd(v_a_40_, v_b_41_);
v___x_43_ = lean_nat_to_int(v___x_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instGCDMonoid___lam__1___boxed(lean_object* v_a_44_, lean_object* v_b_45_){
_start:
{
lean_object* v_res_46_; 
v_res_46_ = lp_mathlib_Int_instGCDMonoid___lam__1(v_a_44_, v_b_45_);
lean_dec(v_b_45_);
lean_dec(v_a_44_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00associatesIntEquivNat_spec__1(lean_object* v_a_57_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lean_nat_to_int(v_a_57_);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0(lean_object* v_x_59_){
_start:
{
lean_object* v___x_60_; uint8_t v___x_61_; 
v___x_60_ = lean_obj_once(&lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__0, &lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__0_once, _init_lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__0);
v___x_61_ = lean_int_dec_le(v___x_60_, v_x_59_);
if (v___x_61_ == 0)
{
lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = lean_obj_once(&lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__2, &lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__2_once, _init_lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__2);
v___x_63_ = lean_int_mul(v_x_59_, v___x_62_);
return v___x_63_;
}
else
{
lean_object* v___x_64_; lean_object* v___x_65_; 
v___x_64_ = lean_obj_once(&lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__1, &lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__1_once, _init_lp_mathlib_Int_strongNormalizationMonoid___lam__0___closed__1);
v___x_65_ = lean_int_mul(v_x_59_, v___x_64_);
return v___x_65_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0___boxed(lean_object* v_x_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0(v_x_66_);
lean_dec(v_x_66_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_out___at___00associatesIntEquivNat_spec__0(lean_object* v_a_68_){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0(v_a_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Associates_out___at___00associatesIntEquivNat_spec__0___boxed(lean_object* v_a_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_mathlib_Associates_out___at___00associatesIntEquivNat_spec__0(v_a_70_);
lean_dec(v_a_70_);
return v_res_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_associatesIntEquivNat___lam__0(lean_object* v_x_72_){
_start:
{
lean_object* v___x_73_; lean_object* v___x_74_; 
v___x_73_ = lp_mathlib_normalize___at___00Associates_out___at___00associatesIntEquivNat_spec__0_spec__0(v_x_72_);
v___x_74_ = lean_nat_abs(v___x_73_);
lean_dec(v___x_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_associatesIntEquivNat___lam__0___boxed(lean_object* v_x_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_mathlib_associatesIntEquivNat___lam__0(v_x_75_);
lean_dec(v_x_75_);
return v_res_76_;
}
}
static lean_object* _init_lp_mathlib_associatesIntEquivNat___closed__1(void){
_start:
{
lean_object* v___f_78_; lean_object* v___f_79_; lean_object* v___x_80_; 
v___f_78_ = lean_alloc_closure((void*)(l_Int_ofNat___boxed), 1, 0);
v___f_79_ = ((lean_object*)(lp_mathlib_associatesIntEquivNat___closed__0));
v___x_80_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_80_, 0, v___f_79_);
lean_ctor_set(v___x_80_, 1, v___f_78_);
return v___x_80_;
}
}
static lean_object* _init_lp_mathlib_associatesIntEquivNat(void){
_start:
{
lean_object* v___x_81_; 
v___x_81_ = lean_obj_once(&lp_mathlib_associatesIntEquivNat___closed__1, &lp_mathlib_associatesIntEquivNat___closed__1_once, _init_lp_mathlib_associatesIntEquivNat___closed__1);
return v___x_81_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Nat(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instStrongNormalizedGCDMonoidNat = _init_lp_mathlib_instStrongNormalizedGCDMonoidNat();
lean_mark_persistent(lp_mathlib_instStrongNormalizedGCDMonoidNat);
lp_mathlib_associatesIntEquivNat = _init_lp_mathlib_associatesIntEquivNat();
lean_mark_persistent(lp_mathlib_associatesIntEquivNat);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Nat(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GCDMonoid_Nat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GCDMonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_GCDMonoid_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_GCDMonoid_Nat(builtin);
}
#ifdef __cplusplus
}
#endif
