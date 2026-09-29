// Lean compiler output
// Module: Mathlib.Data.ENat.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Sub.WithTop public import Mathlib.Data.ENat.Defs public import Mathlib.Order.Nat import Mathlib.Algebra.Order.Group.Nat
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
lean_object* l_Nat_decEq___boxed(lean_object*, lean_object*);
uint8_t l_Option_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_decLt___boxed(lean_object*, lean_object*);
uint8_t lp_mathlib_WithTop_decidableLT___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_decLe___boxed(lean_object*, lean_object*);
uint8_t lp_mathlib_WithTop_decidableLE___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lp_mathlib_Option_map_u2082___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_map___redArg(lean_object*, lean_object*);
lean_object* l_Nat_add___boxed(lean_object*, lean_object*);
lean_object* l_Nat_sub___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_sub___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_untopD___redArg(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instAddENat___aux__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_add___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instAddENat___aux__1___closed__0 = (const lean_object*)&lp_mathlib_instAddENat___aux__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instAddENat___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddENat___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instAddENat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instAddENat___lam__0, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_instAddENat___aux__1___closed__0_value)} };
static const lean_object* lp_mathlib_instAddENat___closed__0 = (const lean_object*)&lp_mathlib_instAddENat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instAddENat = (const lean_object*)&lp_mathlib_instAddENat___closed__0_value;
static const lean_closure_object lp_mathlib_instSubENat___aux__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_sub___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instSubENat___aux__1___closed__0 = (const lean_object*)&lp_mathlib_instSubENat___aux__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instSubENat___aux__1(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_WithTop_sub___at___00instSubENat_spec__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_WithTop_sub___at___00instSubENat_spec__0___closed__0 = (const lean_object*)&lp_mathlib_WithTop_sub___at___00instSubENat_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithTop_sub___at___00instSubENat_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_sub___at___00instSubENat_spec__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instSubENat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithTop_sub___at___00instSubENat_spec__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instSubENat___closed__0 = (const lean_object*)&lp_mathlib_instSubENat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instSubENat = (const lean_object*)&lp_mathlib_instSubENat___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instLEENat;
LEAN_EXPORT lean_object* lp_mathlib_instLTENat;
LEAN_EXPORT const lean_object* lp_mathlib_instBotENat___aux__1 = (const lean_object*)&lp_mathlib_WithTop_sub___at___00instSubENat_spec__0___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instBotENat = (const lean_object*)&lp_mathlib_WithTop_sub___at___00instSubENat_spec__0___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instZeroENat___aux__1 = (const lean_object*)&lp_mathlib_WithTop_sub___at___00instSubENat_spec__0___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instZeroENat = (const lean_object*)&lp_mathlib_WithTop_sub___at___00instSubENat_spec__0___closed__0_value;
static const lean_ctor_object lp_mathlib_instOneENat___aux__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_instOneENat___aux__1___closed__0 = (const lean_object*)&lp_mathlib_instOneENat___aux__1___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instOneENat___aux__1 = (const lean_object*)&lp_mathlib_instOneENat___aux__1___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instOneENat = (const lean_object*)&lp_mathlib_instOneENat___aux__1___closed__0_value;
static const lean_ctor_object lp_mathlib_instPreorderENat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_instPreorderENat___closed__0 = (const lean_object*)&lp_mathlib_instPreorderENat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instPreorderENat = (const lean_object*)&lp_mathlib_instPreorderENat___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__4___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__4___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instLinearOrderENat___aux__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instLinearOrderENat___aux__4___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instLinearOrderENat___aux__4___closed__0 = (const lean_object*)&lp_mathlib_instLinearOrderENat___aux__4___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__4(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instLinearOrderENat___aux__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_decLt___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instLinearOrderENat___aux__6___closed__0 = (const lean_object*)&lp_mathlib_instLinearOrderENat___aux__6___closed__0_value;
static const lean_closure_object lp_mathlib_instLinearOrderENat___aux__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_decEq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instLinearOrderENat___aux__6___closed__1 = (const lean_object*)&lp_mathlib_instLinearOrderENat___aux__6___closed__1_value;
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderENat___aux__6(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__6___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instLinearOrderENat___aux__9___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Nat_decLe___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instLinearOrderENat___aux__9___closed__0 = (const lean_object*)&lp_mathlib_instLinearOrderENat___aux__9___closed__0_value;
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderENat___aux__9(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__9___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderENat___aux__11(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__11___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderENat___aux__13(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__13___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderENat___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___lam__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instLinearOrderENat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instLinearOrderENat___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instLinearOrderENat___closed__0 = (const lean_object*)&lp_mathlib_instLinearOrderENat___closed__0_value;
static const lean_closure_object lp_mathlib_instLinearOrderENat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instAddENat___lam__0, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_instLinearOrderENat___aux__4___closed__0_value)} };
static const lean_object* lp_mathlib_instLinearOrderENat___closed__1 = (const lean_object*)&lp_mathlib_instLinearOrderENat___closed__1_value;
static const lean_closure_object lp_mathlib_instLinearOrderENat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instLinearOrderENat___lam__3___boxed, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_instLinearOrderENat___aux__6___closed__1_value)} };
static const lean_object* lp_mathlib_instLinearOrderENat___closed__2 = (const lean_object*)&lp_mathlib_instLinearOrderENat___closed__2_value;
static lean_once_cell_t lp_mathlib_instLinearOrderENat___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instLinearOrderENat___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat;
LEAN_EXPORT lean_object* lp_mathlib_instOrderTopENat;
LEAN_EXPORT const lean_object* lp_mathlib_instOrderBotENat = (const lean_object*)&lp_mathlib_WithTop_sub___at___00instSubENat_spec__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_ENat_lift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ENat_lift___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ENat_lift(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ENat_lift___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ENat_instWellFoundedRelation;
LEAN_EXPORT lean_object* lp_mathlib_ENat_toNat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ENat_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ENat_map(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_ENatMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_ENatMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_ENatMap(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_ENatMap___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_ENatMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_ENatMap(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_ENatMap___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_ENatMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_ENatMap(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddHom_ENatMap___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddENat___aux__1(lean_object* v_a_2_, lean_object* v_a_3_){
_start:
{
lean_object* v___f_4_; lean_object* v___x_5_; 
v___f_4_ = ((lean_object*)(lp_mathlib_instAddENat___aux__1___closed__0));
v___x_5_ = lp_mathlib_Option_map_u2082___redArg(v___f_4_, v_a_2_, v_a_3_);
return v___x_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instAddENat___lam__0(lean_object* v___f_6_, lean_object* v___y_7_, lean_object* v___y_8_){
_start:
{
lean_object* v___x_9_; 
v___x_9_ = lp_mathlib_Option_map_u2082___redArg(v___f_6_, v___y_7_, v___y_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSubENat___aux__1(lean_object* v_x_14_, lean_object* v_x_15_){
_start:
{
lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_16_ = ((lean_object*)(lp_mathlib_instSubENat___aux__1___closed__0));
v___x_17_ = lean_unsigned_to_nat(0u);
v___x_18_ = lp_mathlib_WithTop_sub___redArg(v___x_16_, v___x_17_, v_x_14_, v_x_15_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_sub___at___00instSubENat_spec__0(lean_object* v_x_21_, lean_object* v_x_22_){
_start:
{
if (lean_obj_tag(v_x_22_) == 0)
{
lean_object* v___x_23_; 
lean_dec(v_x_21_);
v___x_23_ = ((lean_object*)(lp_mathlib_WithTop_sub___at___00instSubENat_spec__0___closed__0));
return v___x_23_;
}
else
{
if (lean_obj_tag(v_x_21_) == 0)
{
lean_object* v___x_24_; 
v___x_24_ = lean_box(0);
return v___x_24_;
}
else
{
lean_object* v_val_25_; lean_object* v_val_26_; lean_object* v___x_28_; uint8_t v_isShared_29_; uint8_t v_isSharedCheck_34_; 
v_val_25_ = lean_ctor_get(v_x_22_, 0);
v_val_26_ = lean_ctor_get(v_x_21_, 0);
v_isSharedCheck_34_ = !lean_is_exclusive(v_x_21_);
if (v_isSharedCheck_34_ == 0)
{
v___x_28_ = v_x_21_;
v_isShared_29_ = v_isSharedCheck_34_;
goto v_resetjp_27_;
}
else
{
lean_inc(v_val_26_);
lean_dec(v_x_21_);
v___x_28_ = lean_box(0);
v_isShared_29_ = v_isSharedCheck_34_;
goto v_resetjp_27_;
}
v_resetjp_27_:
{
lean_object* v___x_30_; lean_object* v___x_32_; 
v___x_30_ = lean_nat_sub(v_val_26_, v_val_25_);
lean_dec(v_val_26_);
if (v_isShared_29_ == 0)
{
lean_ctor_set(v___x_28_, 0, v___x_30_);
v___x_32_ = v___x_28_;
goto v_reusejp_31_;
}
else
{
lean_object* v_reuseFailAlloc_33_; 
v_reuseFailAlloc_33_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_33_, 0, v___x_30_);
v___x_32_ = v_reuseFailAlloc_33_;
goto v_reusejp_31_;
}
v_reusejp_31_:
{
return v___x_32_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_sub___at___00instSubENat_spec__0___boxed(lean_object* v_x_35_, lean_object* v_x_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_mathlib_WithTop_sub___at___00instSubENat_spec__0(v_x_35_, v_x_36_);
lean_dec(v_x_36_);
return v_res_37_;
}
}
static lean_object* _init_lp_mathlib_instLEENat(void){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lean_box(0);
return v___x_40_;
}
}
static lean_object* _init_lp_mathlib_instLTENat(void){
_start:
{
lean_object* v___x_41_; 
v___x_41_ = lean_box(0);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__2(lean_object* v_a_54_, lean_object* v_b_55_){
_start:
{
if (lean_obj_tag(v_a_54_) == 0)
{
if (lean_obj_tag(v_b_55_) == 0)
{
lean_object* v___x_56_; 
v___x_56_ = lean_box(0);
return v___x_56_;
}
else
{
lean_inc_ref(v_b_55_);
return v_b_55_;
}
}
else
{
if (lean_obj_tag(v_b_55_) == 0)
{
lean_inc_ref(v_a_54_);
return v_a_54_;
}
else
{
lean_object* v_val_57_; lean_object* v_val_58_; uint8_t v___x_59_; 
v_val_57_ = lean_ctor_get(v_a_54_, 0);
v_val_58_ = lean_ctor_get(v_b_55_, 0);
v___x_59_ = lean_nat_dec_le(v_val_57_, v_val_58_);
if (v___x_59_ == 0)
{
lean_inc_ref(v_b_55_);
return v_b_55_;
}
else
{
lean_inc_ref(v_a_54_);
return v_a_54_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__2___boxed(lean_object* v_a_60_, lean_object* v_b_61_){
_start:
{
lean_object* v_res_62_; 
v_res_62_ = lp_mathlib_instLinearOrderENat___aux__2(v_a_60_, v_b_61_);
lean_dec(v_b_61_);
lean_dec(v_a_60_);
return v_res_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__4___lam__0(lean_object* v_x1_63_, lean_object* v_x2_64_){
_start:
{
uint8_t v___x_65_; 
v___x_65_ = lean_nat_dec_le(v_x1_63_, v_x2_64_);
if (v___x_65_ == 0)
{
lean_inc(v_x1_63_);
return v_x1_63_;
}
else
{
lean_inc(v_x2_64_);
return v_x2_64_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__4___lam__0___boxed(lean_object* v_x1_66_, lean_object* v_x2_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_instLinearOrderENat___aux__4___lam__0(v_x1_66_, v_x2_67_);
lean_dec(v_x2_67_);
lean_dec(v_x1_66_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__4(lean_object* v_a_70_, lean_object* v_b_71_){
_start:
{
lean_object* v___f_72_; lean_object* v___x_73_; 
v___f_72_ = ((lean_object*)(lp_mathlib_instLinearOrderENat___aux__4___closed__0));
v___x_73_ = lp_mathlib_Option_map_u2082___redArg(v___f_72_, v_a_70_, v_b_71_);
return v___x_73_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderENat___aux__6(lean_object* v_a_76_, lean_object* v_b_77_){
_start:
{
lean_object* v___x_78_; uint8_t v___x_79_; 
v___x_78_ = ((lean_object*)(lp_mathlib_instLinearOrderENat___aux__6___closed__0));
lean_inc(v_b_77_);
lean_inc(v_a_76_);
v___x_79_ = lp_mathlib_WithTop_decidableLT___redArg(v___x_78_, v_a_76_, v_b_77_);
if (v___x_79_ == 0)
{
lean_object* v___f_80_; uint8_t v___x_81_; 
v___f_80_ = ((lean_object*)(lp_mathlib_instLinearOrderENat___aux__6___closed__1));
v___x_81_ = l_Option_instDecidableEq___redArg(v___f_80_, v_a_76_, v_b_77_);
if (v___x_81_ == 0)
{
uint8_t v___x_82_; 
v___x_82_ = 2;
return v___x_82_;
}
else
{
uint8_t v___x_83_; 
v___x_83_ = 1;
return v___x_83_;
}
}
else
{
uint8_t v___x_84_; 
lean_dec(v_b_77_);
lean_dec(v_a_76_);
v___x_84_ = 0;
return v___x_84_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__6___boxed(lean_object* v_a_85_, lean_object* v_b_86_){
_start:
{
uint8_t v_res_87_; lean_object* v_r_88_; 
v_res_87_ = lp_mathlib_instLinearOrderENat___aux__6(v_a_85_, v_b_86_);
v_r_88_ = lean_box(v_res_87_);
return v_r_88_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderENat___aux__9(lean_object* v_a_90_, lean_object* v_b_91_){
_start:
{
lean_object* v___x_92_; uint8_t v___x_93_; 
v___x_92_ = ((lean_object*)(lp_mathlib_instLinearOrderENat___aux__9___closed__0));
v___x_93_ = lp_mathlib_WithTop_decidableLE___redArg(v___x_92_, v_a_90_, v_b_91_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__9___boxed(lean_object* v_a_94_, lean_object* v_b_95_){
_start:
{
uint8_t v_res_96_; lean_object* v_r_97_; 
v_res_96_ = lp_mathlib_instLinearOrderENat___aux__9(v_a_94_, v_b_95_);
v_r_97_ = lean_box(v_res_96_);
return v_r_97_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderENat___aux__11(lean_object* v_a_98_, lean_object* v_b_99_){
_start:
{
lean_object* v___f_100_; uint8_t v___x_101_; 
v___f_100_ = ((lean_object*)(lp_mathlib_instLinearOrderENat___aux__6___closed__1));
v___x_101_ = l_Option_instDecidableEq___redArg(v___f_100_, v_a_98_, v_b_99_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__11___boxed(lean_object* v_a_102_, lean_object* v_b_103_){
_start:
{
uint8_t v_res_104_; lean_object* v_r_105_; 
v_res_104_ = lp_mathlib_instLinearOrderENat___aux__11(v_a_102_, v_b_103_);
v_r_105_ = lean_box(v_res_104_);
return v_r_105_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderENat___aux__13(lean_object* v_a_106_, lean_object* v_b_107_){
_start:
{
lean_object* v___x_108_; uint8_t v___x_109_; 
v___x_108_ = ((lean_object*)(lp_mathlib_instLinearOrderENat___aux__6___closed__0));
v___x_109_ = lp_mathlib_WithTop_decidableLT___redArg(v___x_108_, v_a_106_, v_b_107_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___aux__13___boxed(lean_object* v_a_110_, lean_object* v_b_111_){
_start:
{
uint8_t v_res_112_; lean_object* v_r_113_; 
v_res_112_ = lp_mathlib_instLinearOrderENat___aux__13(v_a_110_, v_b_111_);
v_r_113_ = lean_box(v_res_112_);
return v_r_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___lam__0(lean_object* v___y_114_, lean_object* v___y_115_){
_start:
{
if (lean_obj_tag(v___y_114_) == 0)
{
if (lean_obj_tag(v___y_115_) == 0)
{
lean_object* v___x_116_; 
v___x_116_ = lean_box(0);
return v___x_116_;
}
else
{
lean_inc_ref(v___y_115_);
return v___y_115_;
}
}
else
{
if (lean_obj_tag(v___y_115_) == 0)
{
lean_inc_ref(v___y_114_);
return v___y_114_;
}
else
{
lean_object* v_val_117_; lean_object* v_val_118_; uint8_t v___x_119_; 
v_val_117_ = lean_ctor_get(v___y_114_, 0);
v_val_118_ = lean_ctor_get(v___y_115_, 0);
v___x_119_ = lean_nat_dec_le(v_val_117_, v_val_118_);
if (v___x_119_ == 0)
{
lean_inc_ref(v___y_115_);
return v___y_115_;
}
else
{
lean_inc_ref(v___y_114_);
return v___y_114_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___lam__0___boxed(lean_object* v___y_120_, lean_object* v___y_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_instLinearOrderENat___lam__0(v___y_120_, v___y_121_);
lean_dec(v___y_121_);
lean_dec(v___y_120_);
return v_res_122_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderENat___lam__3(lean_object* v___f_123_, lean_object* v___y_124_, lean_object* v___y_125_){
_start:
{
lean_object* v___x_126_; uint8_t v___x_127_; 
v___x_126_ = ((lean_object*)(lp_mathlib_instLinearOrderENat___aux__6___closed__0));
lean_inc(v___y_125_);
lean_inc(v___y_124_);
v___x_127_ = lp_mathlib_WithTop_decidableLT___redArg(v___x_126_, v___y_124_, v___y_125_);
if (v___x_127_ == 0)
{
uint8_t v___x_128_; 
v___x_128_ = l_Option_instDecidableEq___redArg(v___f_123_, v___y_124_, v___y_125_);
if (v___x_128_ == 0)
{
uint8_t v___x_129_; 
v___x_129_ = 2;
return v___x_129_;
}
else
{
uint8_t v___x_130_; 
v___x_130_ = 1;
return v___x_130_;
}
}
else
{
uint8_t v___x_131_; 
lean_dec(v___y_125_);
lean_dec(v___y_124_);
lean_dec_ref(v___f_123_);
v___x_131_ = 0;
return v___x_131_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderENat___lam__3___boxed(lean_object* v___f_132_, lean_object* v___y_133_, lean_object* v___y_134_){
_start:
{
uint8_t v_res_135_; lean_object* v_r_136_; 
v_res_135_ = lp_mathlib_instLinearOrderENat___lam__3(v___f_132_, v___y_133_, v___y_134_);
v_r_136_ = lean_box(v_res_135_);
return v_r_136_;
}
}
static lean_object* _init_lp_mathlib_instLinearOrderENat___closed__3(void){
_start:
{
lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___f_145_; lean_object* v___f_146_; lean_object* v___f_147_; lean_object* v___x_148_; lean_object* v___x_149_; 
v___x_142_ = lean_alloc_closure((void*)(lp_mathlib_instLinearOrderENat___aux__13___boxed), 2, 0);
v___x_143_ = lean_alloc_closure((void*)(lp_mathlib_instLinearOrderENat___aux__11___boxed), 2, 0);
v___x_144_ = lean_alloc_closure((void*)(lp_mathlib_instLinearOrderENat___aux__9___boxed), 2, 0);
v___f_145_ = ((lean_object*)(lp_mathlib_instLinearOrderENat___closed__2));
v___f_146_ = ((lean_object*)(lp_mathlib_instLinearOrderENat___closed__1));
v___f_147_ = ((lean_object*)(lp_mathlib_instLinearOrderENat___closed__0));
v___x_148_ = ((lean_object*)(lp_mathlib_instPreorderENat));
v___x_149_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_149_, 0, v___x_148_);
lean_ctor_set(v___x_149_, 1, v___f_147_);
lean_ctor_set(v___x_149_, 2, v___f_146_);
lean_ctor_set(v___x_149_, 3, v___f_145_);
lean_ctor_set(v___x_149_, 4, v___x_144_);
lean_ctor_set(v___x_149_, 5, v___x_143_);
lean_ctor_set(v___x_149_, 6, v___x_142_);
return v___x_149_;
}
}
static lean_object* _init_lp_mathlib_instLinearOrderENat(void){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lean_obj_once(&lp_mathlib_instLinearOrderENat___closed__3, &lp_mathlib_instLinearOrderENat___closed__3_once, _init_lp_mathlib_instLinearOrderENat___closed__3);
return v___x_150_;
}
}
static lean_object* _init_lp_mathlib_instOrderTopENat(void){
_start:
{
lean_object* v___x_151_; 
v___x_151_ = lean_box(0);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENat_lift___redArg(lean_object* v_x_153_){
_start:
{
lean_object* v_val_154_; 
v_val_154_ = lean_ctor_get(v_x_153_, 0);
lean_inc(v_val_154_);
return v_val_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENat_lift___redArg___boxed(lean_object* v_x_155_){
_start:
{
lean_object* v_res_156_; 
v_res_156_ = lp_mathlib_ENat_lift___redArg(v_x_155_);
lean_dec(v_x_155_);
return v_res_156_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENat_lift(lean_object* v_x_157_, lean_object* v_h_158_){
_start:
{
lean_object* v_val_159_; 
v_val_159_ = lean_ctor_get(v_x_157_, 0);
lean_inc(v_val_159_);
return v_val_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENat_lift___boxed(lean_object* v_x_160_, lean_object* v_h_161_){
_start:
{
lean_object* v_res_162_; 
v_res_162_ = lp_mathlib_ENat_lift(v_x_160_, v_h_161_);
lean_dec(v_x_160_);
return v_res_162_;
}
}
static lean_object* _init_lp_mathlib_ENat_instWellFoundedRelation(void){
_start:
{
lean_object* v___x_163_; 
v___x_163_ = lean_box(0);
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENat_toNat(lean_object* v_x_164_){
_start:
{
lean_object* v___x_165_; lean_object* v___x_166_; 
v___x_165_ = lean_unsigned_to_nat(0u);
v___x_166_ = lp_mathlib_WithTop_untopD___redArg(v___x_165_, v_x_164_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENat_map___redArg(lean_object* v_f_167_, lean_object* v_k_168_){
_start:
{
lean_object* v___x_169_; 
v___x_169_ = lp_mathlib_WithTop_map___redArg(v_f_167_, v_k_168_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENat_map(lean_object* v_00_u03b1_170_, lean_object* v_f_171_, lean_object* v_k_172_){
_start:
{
lean_object* v___x_173_; 
v___x_173_ = lp_mathlib_WithTop_map___redArg(v_f_171_, v_k_172_);
return v___x_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_ENatMap___redArg___lam__0(lean_object* v_f_174_, lean_object* v___y_175_){
_start:
{
lean_object* v___x_176_; 
v___x_176_ = lean_apply_1(v_f_174_, v___y_175_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_ENatMap___redArg(lean_object* v_f_177_){
_start:
{
lean_object* v___f_178_; lean_object* v___x_179_; 
v___f_178_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_ENatMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_178_, 0, v_f_177_);
v___x_179_ = lean_alloc_closure((void*)(lp_mathlib_ENat_map), 3, 2);
lean_closure_set(v___x_179_, 0, lean_box(0));
lean_closure_set(v___x_179_, 1, v___f_178_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_ENatMap(lean_object* v_N_180_, lean_object* v_inst_181_, lean_object* v_f_182_){
_start:
{
lean_object* v___x_183_; 
v___x_183_ = lp_mathlib_OneHom_ENatMap___redArg(v_f_182_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_ENatMap___boxed(lean_object* v_N_184_, lean_object* v_inst_185_, lean_object* v_f_186_){
_start:
{
lean_object* v_res_187_; 
v_res_187_ = lp_mathlib_OneHom_ENatMap(v_N_184_, v_inst_185_, v_f_186_);
lean_dec(v_inst_185_);
return v_res_187_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_ENatMap___redArg(lean_object* v_f_188_){
_start:
{
lean_object* v___f_189_; lean_object* v___x_190_; 
v___f_189_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_ENatMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_189_, 0, v_f_188_);
v___x_190_ = lean_alloc_closure((void*)(lp_mathlib_ENat_map), 3, 2);
lean_closure_set(v___x_190_, 0, lean_box(0));
lean_closure_set(v___x_190_, 1, v___f_189_);
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_ENatMap(lean_object* v_N_191_, lean_object* v_inst_192_, lean_object* v_f_193_){
_start:
{
lean_object* v___x_194_; 
v___x_194_ = lp_mathlib_ZeroHom_ENatMap___redArg(v_f_193_);
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_ENatMap___boxed(lean_object* v_N_195_, lean_object* v_inst_196_, lean_object* v_f_197_){
_start:
{
lean_object* v_res_198_; 
v_res_198_ = lp_mathlib_ZeroHom_ENatMap(v_N_195_, v_inst_196_, v_f_197_);
lean_dec(v_inst_196_);
return v_res_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_ENatMap___redArg(lean_object* v_f_199_){
_start:
{
lean_object* v___f_200_; lean_object* v___x_201_; 
v___f_200_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_ENatMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_200_, 0, v_f_199_);
v___x_201_ = lean_alloc_closure((void*)(lp_mathlib_ENat_map), 3, 2);
lean_closure_set(v___x_201_, 0, lean_box(0));
lean_closure_set(v___x_201_, 1, v___f_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_ENatMap(lean_object* v_N_202_, lean_object* v_inst_203_, lean_object* v_f_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lp_mathlib_AddHom_ENatMap___redArg(v_f_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddHom_ENatMap___boxed(lean_object* v_N_206_, lean_object* v_inst_207_, lean_object* v_f_208_){
_start:
{
lean_object* v_res_209_; 
v_res_209_ = lp_mathlib_AddHom_ENatMap(v_N_206_, v_inst_207_, v_f_208_);
lean_dec(v_inst_207_);
return v_res_209_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_WithTop(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_ENat_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Nat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_ENat_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Sub_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ENat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instLEENat = _init_lp_mathlib_instLEENat();
lean_mark_persistent(lp_mathlib_instLEENat);
lp_mathlib_instLTENat = _init_lp_mathlib_instLTENat();
lean_mark_persistent(lp_mathlib_instLTENat);
lp_mathlib_instLinearOrderENat = _init_lp_mathlib_instLinearOrderENat();
lean_mark_persistent(lp_mathlib_instLinearOrderENat);
lp_mathlib_instOrderTopENat = _init_lp_mathlib_instOrderTopENat();
lean_mark_persistent(lp_mathlib_instOrderTopENat);
lp_mathlib_ENat_instWellFoundedRelation = _init_lp_mathlib_ENat_instWellFoundedRelation();
lean_mark_persistent(lp_mathlib_ENat_instWellFoundedRelation);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_ENat_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Sub_WithTop(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ENat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Nat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_ENat_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Sub_WithTop(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ENat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Nat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_ENat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_ENat_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_ENat_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
