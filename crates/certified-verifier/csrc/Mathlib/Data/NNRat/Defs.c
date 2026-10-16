// Lean compiler output
// Module: Mathlib.Data.NNRat.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.Group.Unbundled.Int public import Mathlib.Algebra.Order.Nonneg.Basic public import Mathlib.Algebra.Order.Ring.Unbundled.Rat public import Mathlib.Algebra.Ring.Rat public import Mathlib.Data.Set.Operations public import Mathlib.Order.Bounds.Defs public import Mathlib.Order.GaloisConnection.Defs
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
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Rat_ofInt(lean_object*);
uint8_t l_Rat_instDecidableLe(lean_object*, lean_object*);
uint8_t l_Rat_blt(lean_object*, lean_object*);
uint8_t l_instDecidableEqRat_decEq(lean_object*, lean_object*);
lean_object* lp_mathlib_NNRat_num(lean_object*);
lean_object* l_Rat_sub(lean_object*, lean_object*);
lean_object* l_Rat_instNatCast___lam__0(lean_object*);
lean_object* l_Rat_add(lean_object*, lean_object*);
lean_object* l_Rat_neg(lean_object*);
lean_object* l_Rat_divInt(lean_object*, lean_object*);
lean_object* l_Rat_pow(lean_object*, lean_object*);
lean_object* l_Rat_mul(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Rat_commSemiring;
lean_object* lp_mathlib_instMulZeroClassOfSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
lean_object* l_Rat_mul___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
extern lean_object* lp_mathlib_Rat_instSemilatticeSup;
lean_object* lp_mathlib_Nonneg_toNonneg___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_specialised__high__priority__simp__lemma;
static lean_once_cell_t lp_mathlib_instCommSemiringNNRat___aux__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instCommSemiringNNRat___aux__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___aux__1;
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___aux__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___aux__8(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instCommSemiringNNRat___aux__13___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instCommSemiringNNRat___aux__13___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___aux__13;
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___aux__15(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___aux__15___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___aux__20(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___aux__20___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_instCommSemiringNNRat___aux__28___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instCommSemiringNNRat___aux__28___closed__0;
static lean_once_cell_t lp_mathlib_instCommSemiringNNRat___aux__28___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instCommSemiringNNRat___aux__28___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___aux__28(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00instCommSemiringNNRat_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00instCommSemiringNNRat_spec__1(lean_object*);
static const lean_closure_object lp_mathlib_instCommSemiringNNRat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Rat_add, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instCommSemiringNNRat___closed__0 = (const lean_object*)&lp_mathlib_instCommSemiringNNRat___closed__0_value;
static const lean_closure_object lp_mathlib_instCommSemiringNNRat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instCommSemiringNNRat___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instCommSemiringNNRat___closed__1 = (const lean_object*)&lp_mathlib_instCommSemiringNNRat___closed__1_value;
static const lean_closure_object lp_mathlib_instCommSemiringNNRat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Rat_mul___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instCommSemiringNNRat___closed__2 = (const lean_object*)&lp_mathlib_instCommSemiringNNRat___closed__2_value;
static const lean_closure_object lp_mathlib_instCommSemiringNNRat___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instCommSemiringNNRat___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instCommSemiringNNRat___closed__3 = (const lean_object*)&lp_mathlib_instCommSemiringNNRat___closed__3_value;
static const lean_closure_object lp_mathlib_instCommSemiringNNRat___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Nat_cast___at___00instCommSemiringNNRat_spec__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instCommSemiringNNRat___closed__4 = (const lean_object*)&lp_mathlib_instCommSemiringNNRat___closed__4_value;
static lean_once_cell_t lp_mathlib_instCommSemiringNNRat___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instCommSemiringNNRat___closed__5;
static lean_once_cell_t lp_mathlib_instCommSemiringNNRat___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instCommSemiringNNRat___closed__6;
static lean_once_cell_t lp_mathlib_instCommSemiringNNRat___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instCommSemiringNNRat___closed__7;
static lean_once_cell_t lp_mathlib_instCommSemiringNNRat___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instCommSemiringNNRat___closed__8;
static lean_once_cell_t lp_mathlib_instCommSemiringNNRat___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instCommSemiringNNRat___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat;
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_cast___at___00instCommSemiringNNRat_spec__0_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_cast___at___00Nat_cast___at___00instCommSemiringNNRat_spec__1_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instAddCancelCommMonoidNNRat;
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___aux__9(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___aux__11(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderNNRat___aux__13(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___aux__13___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderNNRat___aux__16(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___aux__16___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderNNRat___aux__18(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___aux__18___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderNNRat___aux__20(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___aux__20___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___lam__1(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderNNRat___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___lam__2___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instLinearOrderNNRat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instLinearOrderNNRat___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instLinearOrderNNRat___closed__0 = (const lean_object*)&lp_mathlib_instLinearOrderNNRat___closed__0_value;
static const lean_closure_object lp_mathlib_instLinearOrderNNRat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instLinearOrderNNRat___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instLinearOrderNNRat___closed__1 = (const lean_object*)&lp_mathlib_instLinearOrderNNRat___closed__1_value;
static const lean_closure_object lp_mathlib_instLinearOrderNNRat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instLinearOrderNNRat___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instLinearOrderNNRat___closed__2 = (const lean_object*)&lp_mathlib_instLinearOrderNNRat___closed__2_value;
static const lean_ctor_object lp_mathlib_instLinearOrderNNRat___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_instLinearOrderNNRat___closed__3 = (const lean_object*)&lp_mathlib_instLinearOrderNNRat___closed__3_value;
static lean_once_cell_t lp_mathlib_instLinearOrderNNRat___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instLinearOrderNNRat___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat;
static lean_once_cell_t lp_mathlib_instSubNNRat___aux__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instSubNNRat___aux__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instSubNNRat___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_toNonneg___at___00instSubNNRat_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instSubNNRat___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_instSubNNRat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_instSubNNRat___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_instSubNNRat___closed__0 = (const lean_object*)&lp_mathlib_instSubNNRat___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_instSubNNRat = (const lean_object*)&lp_mathlib_instSubNNRat___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedNNRat___aux__1;
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedNNRat;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_instOrderBot;
LEAN_EXPORT lean_object* lp_mathlib_Rat_toNNRat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_cast___at___00NNRat_gi_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_cast___at___00NNRat_gi_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_NNRat_gi___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Rat_toNNRat, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NNRat_gi___closed__0 = (const lean_object*)&lp_mathlib_NNRat_gi___closed__0_value;
static const lean_closure_object lp_mathlib_NNRat_gi___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___redArg___lam__0, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_NNRat_gi___closed__0_value)} };
static const lean_object* lp_mathlib_NNRat_gi___closed__1 = (const lean_object*)&lp_mathlib_NNRat_gi___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_NNRat_gi = (const lean_object*)&lp_mathlib_NNRat_gi___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_NNRat_coeHom___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_coeHom___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_NNRat_coeHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_NNRat_coeHom___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_NNRat_coeHom___closed__0 = (const lean_object*)&lp_mathlib_NNRat_coeHom___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_NNRat_coeHom = (const lean_object*)&lp_mathlib_NNRat_coeHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_abs___at___00Rat_nnabs_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_nnabs(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_divNat(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_numDenCasesOn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NNRat_numDenCasesOn(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_LibraryNote_specialised__high__priority__simp__lemma(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_box(0);
return v___x_1_;
}
}
static lean_object* _init_lp_mathlib_instCommSemiringNNRat___aux__1___closed__0(void){
_start:
{
lean_object* v___x_2_; lean_object* v___x_3_; 
v___x_2_ = lean_unsigned_to_nat(0u);
v___x_3_ = l_Rat_instNatCast___lam__0(v___x_2_);
return v___x_3_;
}
}
static lean_object* _init_lp_mathlib_instCommSemiringNNRat___aux__1(void){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_obj_once(&lp_mathlib_instCommSemiringNNRat___aux__1___closed__0, &lp_mathlib_instCommSemiringNNRat___aux__1___closed__0_once, _init_lp_mathlib_instCommSemiringNNRat___aux__1___closed__0);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___aux__3(lean_object* v_x_5_, lean_object* v_y_6_){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = l_Rat_add(v_x_5_, v_y_6_);
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___aux__8(lean_object* v_n_8_, lean_object* v_x_9_){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; 
v___x_10_ = l_Rat_instNatCast___lam__0(v_n_8_);
v___x_11_ = l_Rat_mul(v___x_10_, v_x_9_);
lean_dec_ref(v___x_10_);
return v___x_11_;
}
}
static lean_object* _init_lp_mathlib_instCommSemiringNNRat___aux__13___closed__0(void){
_start:
{
lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_12_ = lean_unsigned_to_nat(1u);
v___x_13_ = l_Rat_instNatCast___lam__0(v___x_12_);
return v___x_13_;
}
}
static lean_object* _init_lp_mathlib_instCommSemiringNNRat___aux__13(void){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lean_obj_once(&lp_mathlib_instCommSemiringNNRat___aux__13___closed__0, &lp_mathlib_instCommSemiringNNRat___aux__13___closed__0_once, _init_lp_mathlib_instCommSemiringNNRat___aux__13___closed__0);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___aux__15(lean_object* v_x_15_, lean_object* v_y_16_){
_start:
{
lean_object* v___x_17_; 
v___x_17_ = l_Rat_mul(v_x_15_, v_y_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___aux__15___boxed(lean_object* v_x_18_, lean_object* v_y_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_instCommSemiringNNRat___aux__15(v_x_18_, v_y_19_);
lean_dec_ref(v_x_18_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___aux__20(lean_object* v_n_21_, lean_object* v_x_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = l_Rat_pow(v_x_22_, v_n_21_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___aux__20___boxed(lean_object* v_n_24_, lean_object* v_x_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_mathlib_instCommSemiringNNRat___aux__20(v_n_24_, v_x_25_);
lean_dec(v_n_24_);
return v_res_26_;
}
}
static lean_object* _init_lp_mathlib_instCommSemiringNNRat___aux__28___closed__0(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; 
v___x_27_ = lp_mathlib_Rat_commSemiring;
v___x_28_ = lp_mathlib_Semiring_toNonAssocSemiring___redArg(v___x_27_);
return v___x_28_;
}
}
static lean_object* _init_lp_mathlib_instCommSemiringNNRat___aux__28___closed__1(void){
_start:
{
lean_object* v___x_29_; lean_object* v___x_30_; 
v___x_29_ = lean_obj_once(&lp_mathlib_instCommSemiringNNRat___aux__28___closed__0, &lp_mathlib_instCommSemiringNNRat___aux__28___closed__0_once, _init_lp_mathlib_instCommSemiringNNRat___aux__28___closed__0);
v___x_30_ = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(v___x_29_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___aux__28(lean_object* v_n_31_){
_start:
{
lean_object* v___x_32_; lean_object* v_toNatCast_33_; lean_object* v___x_34_; 
v___x_32_ = lean_obj_once(&lp_mathlib_instCommSemiringNNRat___aux__28___closed__1, &lp_mathlib_instCommSemiringNNRat___aux__28___closed__1_once, _init_lp_mathlib_instCommSemiringNNRat___aux__28___closed__1);
v_toNatCast_33_ = lean_ctor_get(v___x_32_, 0);
lean_inc(v_toNatCast_33_);
v___x_34_ = lean_apply_1(v_toNatCast_33_, v_n_31_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00instCommSemiringNNRat_spec__0(lean_object* v_a_35_){
_start:
{
lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_36_ = lean_nat_to_int(v_a_35_);
v___x_37_ = l_Rat_ofInt(v___x_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___lam__0(lean_object* v___y_38_, lean_object* v___y_39_){
_start:
{
lean_object* v___x_40_; lean_object* v___x_41_; 
v___x_40_ = lp_mathlib_Nat_cast___at___00instCommSemiringNNRat_spec__0(v___y_38_);
v___x_41_ = l_Rat_mul(v___x_40_, v___y_39_);
lean_dec_ref(v___x_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___lam__1(lean_object* v___y_42_, lean_object* v___y_43_){
_start:
{
lean_object* v___x_44_; 
v___x_44_ = l_Rat_pow(v___y_43_, v___y_42_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCommSemiringNNRat___lam__1___boxed(lean_object* v___y_45_, lean_object* v___y_46_){
_start:
{
lean_object* v_res_47_; 
v_res_47_ = lp_mathlib_instCommSemiringNNRat___lam__1(v___y_45_, v___y_46_);
lean_dec(v___y_45_);
return v_res_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00instCommSemiringNNRat_spec__1(lean_object* v_a_48_){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_49_ = lean_nat_to_int(v_a_48_);
v___x_50_ = l_Rat_ofInt(v___x_49_);
return v___x_50_;
}
}
static lean_object* _init_lp_mathlib_instCommSemiringNNRat___closed__5(void){
_start:
{
lean_object* v___x_56_; lean_object* v___x_57_; 
v___x_56_ = lean_unsigned_to_nat(0u);
v___x_57_ = lp_mathlib_Nat_cast___at___00instCommSemiringNNRat_spec__0(v___x_56_);
return v___x_57_;
}
}
static lean_object* _init_lp_mathlib_instCommSemiringNNRat___closed__6(void){
_start:
{
lean_object* v___f_58_; lean_object* v___f_59_; lean_object* v___x_60_; lean_object* v___x_61_; 
v___f_58_ = ((lean_object*)(lp_mathlib_instCommSemiringNNRat___closed__1));
v___f_59_ = ((lean_object*)(lp_mathlib_instCommSemiringNNRat___closed__0));
v___x_60_ = lean_obj_once(&lp_mathlib_instCommSemiringNNRat___closed__5, &lp_mathlib_instCommSemiringNNRat___closed__5_once, _init_lp_mathlib_instCommSemiringNNRat___closed__5);
v___x_61_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_61_, 0, v___x_60_);
lean_ctor_set(v___x_61_, 1, v___f_59_);
lean_ctor_set(v___x_61_, 2, v___f_58_);
return v___x_61_;
}
}
static lean_object* _init_lp_mathlib_instCommSemiringNNRat___closed__7(void){
_start:
{
lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = lean_unsigned_to_nat(1u);
v___x_63_ = lp_mathlib_Nat_cast___at___00instCommSemiringNNRat_spec__0(v___x_62_);
return v___x_63_;
}
}
static lean_object* _init_lp_mathlib_instCommSemiringNNRat___closed__8(void){
_start:
{
lean_object* v___f_64_; lean_object* v___f_65_; lean_object* v___x_66_; lean_object* v___x_67_; 
v___f_64_ = ((lean_object*)(lp_mathlib_instCommSemiringNNRat___closed__3));
v___f_65_ = ((lean_object*)(lp_mathlib_instCommSemiringNNRat___closed__2));
v___x_66_ = lean_obj_once(&lp_mathlib_instCommSemiringNNRat___closed__7, &lp_mathlib_instCommSemiringNNRat___closed__7_once, _init_lp_mathlib_instCommSemiringNNRat___closed__7);
v___x_67_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_67_, 0, v___x_66_);
lean_ctor_set(v___x_67_, 1, v___f_65_);
lean_ctor_set(v___x_67_, 2, v___f_64_);
return v___x_67_;
}
}
static lean_object* _init_lp_mathlib_instCommSemiringNNRat___closed__9(void){
_start:
{
lean_object* v___f_68_; lean_object* v___x_69_; lean_object* v___x_70_; lean_object* v___x_71_; 
v___f_68_ = ((lean_object*)(lp_mathlib_instCommSemiringNNRat___closed__4));
v___x_69_ = lean_obj_once(&lp_mathlib_instCommSemiringNNRat___closed__8, &lp_mathlib_instCommSemiringNNRat___closed__8_once, _init_lp_mathlib_instCommSemiringNNRat___closed__8);
v___x_70_ = lean_obj_once(&lp_mathlib_instCommSemiringNNRat___closed__6, &lp_mathlib_instCommSemiringNNRat___closed__6_once, _init_lp_mathlib_instCommSemiringNNRat___closed__6);
v___x_71_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_71_, 0, v___x_70_);
lean_ctor_set(v___x_71_, 1, v___x_69_);
lean_ctor_set(v___x_71_, 2, v___f_68_);
return v___x_71_;
}
}
static lean_object* _init_lp_mathlib_instCommSemiringNNRat(void){
_start:
{
lean_object* v___x_72_; 
v___x_72_ = lean_obj_once(&lp_mathlib_instCommSemiringNNRat___closed__9, &lp_mathlib_instCommSemiringNNRat___closed__9_once, _init_lp_mathlib_instCommSemiringNNRat___closed__9);
return v___x_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_cast___at___00Nat_cast___at___00instCommSemiringNNRat_spec__0_spec__0(lean_object* v_a_73_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lean_nat_to_int(v_a_73_);
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_cast___at___00Nat_cast___at___00instCommSemiringNNRat_spec__1_spec__2(lean_object* v_a_75_){
_start:
{
lean_object* v___x_76_; 
v___x_76_ = l_Rat_ofInt(v_a_75_);
return v___x_76_;
}
}
static lean_object* _init_lp_mathlib_instAddCancelCommMonoidNNRat(void){
_start:
{
lean_object* v___x_77_; lean_object* v_toAddCommMonoid_78_; 
v___x_77_ = lp_mathlib_instCommSemiringNNRat;
v_toAddCommMonoid_78_ = lean_ctor_get(v___x_77_, 0);
lean_inc_ref(v_toAddCommMonoid_78_);
return v_toAddCommMonoid_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___aux__9(lean_object* v_a_79_, lean_object* v_a_80_){
_start:
{
uint8_t v___x_81_; 
lean_inc_ref(v_a_80_);
lean_inc_ref(v_a_79_);
v___x_81_ = l_Rat_instDecidableLe(v_a_79_, v_a_80_);
if (v___x_81_ == 0)
{
lean_dec_ref(v_a_79_);
return v_a_80_;
}
else
{
lean_dec_ref(v_a_80_);
return v_a_79_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___aux__11(lean_object* v_a_82_, lean_object* v_a_83_){
_start:
{
uint8_t v___x_84_; 
lean_inc_ref(v_a_83_);
lean_inc_ref(v_a_82_);
v___x_84_ = l_Rat_instDecidableLe(v_a_82_, v_a_83_);
if (v___x_84_ == 0)
{
lean_dec_ref(v_a_83_);
return v_a_82_;
}
else
{
lean_dec_ref(v_a_82_);
return v_a_83_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderNNRat___aux__13(lean_object* v_a_85_, lean_object* v_b_86_){
_start:
{
uint8_t v___x_87_; 
lean_inc_ref(v_b_86_);
lean_inc_ref(v_a_85_);
v___x_87_ = l_Rat_blt(v_a_85_, v_b_86_);
if (v___x_87_ == 0)
{
uint8_t v___x_88_; 
v___x_88_ = l_instDecidableEqRat_decEq(v_a_85_, v_b_86_);
lean_dec_ref(v_b_86_);
lean_dec_ref(v_a_85_);
if (v___x_88_ == 0)
{
uint8_t v___x_89_; 
v___x_89_ = 2;
return v___x_89_;
}
else
{
uint8_t v___x_90_; 
v___x_90_ = 1;
return v___x_90_;
}
}
else
{
uint8_t v___x_91_; 
lean_dec_ref(v_b_86_);
lean_dec_ref(v_a_85_);
v___x_91_ = 0;
return v___x_91_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___aux__13___boxed(lean_object* v_a_92_, lean_object* v_b_93_){
_start:
{
uint8_t v_res_94_; lean_object* v_r_95_; 
v_res_94_ = lp_mathlib_instLinearOrderNNRat___aux__13(v_a_92_, v_b_93_);
v_r_95_ = lean_box(v_res_94_);
return v_r_95_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderNNRat___aux__16(lean_object* v_a_96_, lean_object* v_b_97_){
_start:
{
uint8_t v___x_98_; 
v___x_98_ = l_Rat_instDecidableLe(v_a_96_, v_b_97_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___aux__16___boxed(lean_object* v_a_99_, lean_object* v_b_100_){
_start:
{
uint8_t v_res_101_; lean_object* v_r_102_; 
v_res_101_ = lp_mathlib_instLinearOrderNNRat___aux__16(v_a_99_, v_b_100_);
v_r_102_ = lean_box(v_res_101_);
return v_r_102_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderNNRat___aux__18(lean_object* v_a_103_, lean_object* v_b_104_){
_start:
{
uint8_t v___x_105_; 
v___x_105_ = l_instDecidableEqRat_decEq(v_a_103_, v_b_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___aux__18___boxed(lean_object* v_a_106_, lean_object* v_b_107_){
_start:
{
uint8_t v_res_108_; lean_object* v_r_109_; 
v_res_108_ = lp_mathlib_instLinearOrderNNRat___aux__18(v_a_106_, v_b_107_);
lean_dec_ref(v_b_107_);
lean_dec_ref(v_a_106_);
v_r_109_ = lean_box(v_res_108_);
return v_r_109_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderNNRat___aux__20(lean_object* v_a_110_, lean_object* v_b_111_){
_start:
{
uint8_t v___x_112_; 
v___x_112_ = l_Rat_blt(v_a_110_, v_b_111_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___aux__20___boxed(lean_object* v_a_113_, lean_object* v_b_114_){
_start:
{
uint8_t v_res_115_; lean_object* v_r_116_; 
v_res_115_ = lp_mathlib_instLinearOrderNNRat___aux__20(v_a_113_, v_b_114_);
v_r_116_ = lean_box(v_res_115_);
return v_r_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___lam__0(lean_object* v___y_117_, lean_object* v___y_118_){
_start:
{
uint8_t v___x_119_; 
lean_inc_ref(v___y_118_);
lean_inc_ref(v___y_117_);
v___x_119_ = l_Rat_instDecidableLe(v___y_117_, v___y_118_);
if (v___x_119_ == 0)
{
lean_dec_ref(v___y_117_);
return v___y_118_;
}
else
{
lean_dec_ref(v___y_118_);
return v___y_117_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___lam__1(lean_object* v___y_120_, lean_object* v___y_121_){
_start:
{
uint8_t v___x_122_; 
lean_inc_ref(v___y_121_);
lean_inc_ref(v___y_120_);
v___x_122_ = l_Rat_instDecidableLe(v___y_120_, v___y_121_);
if (v___x_122_ == 0)
{
lean_dec_ref(v___y_121_);
return v___y_120_;
}
else
{
lean_dec_ref(v___y_120_);
return v___y_121_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderNNRat___lam__2(lean_object* v___y_123_, lean_object* v___y_124_){
_start:
{
uint8_t v___x_125_; 
lean_inc_ref(v___y_124_);
lean_inc_ref(v___y_123_);
v___x_125_ = l_Rat_blt(v___y_123_, v___y_124_);
if (v___x_125_ == 0)
{
uint8_t v___x_126_; 
v___x_126_ = l_instDecidableEqRat_decEq(v___y_123_, v___y_124_);
lean_dec_ref(v___y_124_);
lean_dec_ref(v___y_123_);
if (v___x_126_ == 0)
{
uint8_t v___x_127_; 
v___x_127_ = 2;
return v___x_127_;
}
else
{
uint8_t v___x_128_; 
v___x_128_ = 1;
return v___x_128_;
}
}
else
{
uint8_t v___x_129_; 
lean_dec_ref(v___y_124_);
lean_dec_ref(v___y_123_);
v___x_129_ = 0;
return v___x_129_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderNNRat___lam__2___boxed(lean_object* v___y_130_, lean_object* v___y_131_){
_start:
{
uint8_t v_res_132_; lean_object* v_r_133_; 
v_res_132_ = lp_mathlib_instLinearOrderNNRat___lam__2(v___y_130_, v___y_131_);
v_r_133_ = lean_box(v_res_132_);
return v_r_133_;
}
}
static lean_object* _init_lp_mathlib_instLinearOrderNNRat___closed__4(void){
_start:
{
lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___f_143_; lean_object* v___f_144_; lean_object* v___f_145_; lean_object* v___x_146_; lean_object* v___x_147_; 
v___x_140_ = lean_alloc_closure((void*)(lp_mathlib_instLinearOrderNNRat___aux__20___boxed), 2, 0);
v___x_141_ = lean_alloc_closure((void*)(lp_mathlib_instLinearOrderNNRat___aux__18___boxed), 2, 0);
v___x_142_ = lean_alloc_closure((void*)(lp_mathlib_instLinearOrderNNRat___aux__16___boxed), 2, 0);
v___f_143_ = ((lean_object*)(lp_mathlib_instLinearOrderNNRat___closed__2));
v___f_144_ = ((lean_object*)(lp_mathlib_instLinearOrderNNRat___closed__1));
v___f_145_ = ((lean_object*)(lp_mathlib_instLinearOrderNNRat___closed__0));
v___x_146_ = ((lean_object*)(lp_mathlib_instLinearOrderNNRat___closed__3));
v___x_147_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_147_, 0, v___x_146_);
lean_ctor_set(v___x_147_, 1, v___f_145_);
lean_ctor_set(v___x_147_, 2, v___f_144_);
lean_ctor_set(v___x_147_, 3, v___f_143_);
lean_ctor_set(v___x_147_, 4, v___x_142_);
lean_ctor_set(v___x_147_, 5, v___x_141_);
lean_ctor_set(v___x_147_, 6, v___x_140_);
return v___x_147_;
}
}
static lean_object* _init_lp_mathlib_instLinearOrderNNRat(void){
_start:
{
lean_object* v___x_148_; 
v___x_148_ = lean_obj_once(&lp_mathlib_instLinearOrderNNRat___closed__4, &lp_mathlib_instLinearOrderNNRat___closed__4_once, _init_lp_mathlib_instLinearOrderNNRat___closed__4);
return v___x_148_;
}
}
static lean_object* _init_lp_mathlib_instSubNNRat___aux__1___closed__0(void){
_start:
{
lean_object* v___x_149_; lean_object* v___x_150_; 
v___x_149_ = lp_mathlib_Rat_commSemiring;
v___x_150_ = lp_mathlib_instMulZeroClassOfSemiring___redArg(v___x_149_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSubNNRat___aux__1(lean_object* v_x_151_, lean_object* v_y_152_){
_start:
{
lean_object* v___x_153_; lean_object* v_toZero_154_; lean_object* v___x_155_; lean_object* v___x_156_; lean_object* v___x_157_; 
v___x_153_ = lean_obj_once(&lp_mathlib_instSubNNRat___aux__1___closed__0, &lp_mathlib_instSubNNRat___aux__1___closed__0_once, _init_lp_mathlib_instSubNNRat___aux__1___closed__0);
v_toZero_154_ = lean_ctor_get(v___x_153_, 1);
v___x_155_ = lp_mathlib_Rat_instSemilatticeSup;
v___x_156_ = l_Rat_sub(v_x_151_, v_y_152_);
lean_inc(v_toZero_154_);
v___x_157_ = lp_mathlib_Nonneg_toNonneg___redArg(v_toZero_154_, v___x_155_, v___x_156_);
return v___x_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nonneg_toNonneg___at___00instSubNNRat_spec__0(lean_object* v_a_158_){
_start:
{
lean_object* v___x_159_; uint8_t v___x_160_; 
v___x_159_ = lean_obj_once(&lp_mathlib_instCommSemiringNNRat___closed__5, &lp_mathlib_instCommSemiringNNRat___closed__5_once, _init_lp_mathlib_instCommSemiringNNRat___closed__5);
lean_inc_ref(v_a_158_);
v___x_160_ = l_Rat_instDecidableLe(v_a_158_, v___x_159_);
if (v___x_160_ == 0)
{
return v_a_158_;
}
else
{
lean_dec_ref(v_a_158_);
return v___x_159_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instSubNNRat___lam__0(lean_object* v___y_161_, lean_object* v___y_162_){
_start:
{
lean_object* v___x_163_; lean_object* v___x_164_; 
v___x_163_ = l_Rat_sub(v___y_161_, v___y_162_);
v___x_164_ = lp_mathlib_Nonneg_toNonneg___at___00instSubNNRat_spec__0(v___x_163_);
return v___x_164_;
}
}
static lean_object* _init_lp_mathlib_instInhabitedNNRat___aux__1(void){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lean_obj_once(&lp_mathlib_instCommSemiringNNRat___aux__1___closed__0, &lp_mathlib_instCommSemiringNNRat___aux__1___closed__0_once, _init_lp_mathlib_instCommSemiringNNRat___aux__1___closed__0);
return v___x_167_;
}
}
static lean_object* _init_lp_mathlib_instInhabitedNNRat(void){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lean_obj_once(&lp_mathlib_instCommSemiringNNRat___closed__5, &lp_mathlib_instCommSemiringNNRat___closed__5_once, _init_lp_mathlib_instCommSemiringNNRat___closed__5);
return v___x_168_;
}
}
static lean_object* _init_lp_mathlib_NNRat_instOrderBot(void){
_start:
{
lean_object* v___x_169_; 
v___x_169_ = lean_obj_once(&lp_mathlib_instCommSemiringNNRat___closed__5, &lp_mathlib_instCommSemiringNNRat___closed__5_once, _init_lp_mathlib_instCommSemiringNNRat___closed__5);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_toNNRat(lean_object* v_q_170_){
_start:
{
lean_object* v___x_171_; uint8_t v___x_172_; 
v___x_171_ = lean_obj_once(&lp_mathlib_instCommSemiringNNRat___closed__5, &lp_mathlib_instCommSemiringNNRat___closed__5_once, _init_lp_mathlib_instCommSemiringNNRat___closed__5);
lean_inc_ref(v_q_170_);
v___x_172_ = l_Rat_instDecidableLe(v_q_170_, v___x_171_);
if (v___x_172_ == 0)
{
return v_q_170_;
}
else
{
lean_dec_ref(v_q_170_);
return v___x_171_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_cast___at___00NNRat_gi_spec__0(lean_object* v_a_173_){
_start:
{
lean_inc_ref(v_a_173_);
return v_a_173_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_cast___at___00NNRat_gi_spec__0___boxed(lean_object* v_a_174_){
_start:
{
lean_object* v_res_175_; 
v_res_175_ = lp_mathlib_NNRat_cast___at___00NNRat_gi_spec__0(v_a_174_);
lean_dec_ref(v_a_174_);
return v_res_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___redArg___lam__0(lean_object* v_l_176_, lean_object* v_x_177_, lean_object* v_x_178_){
_start:
{
lean_object* v___x_179_; 
v___x_179_ = lean_apply_1(v_l_176_, v_x_177_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___redArg(lean_object* v_l_180_){
_start:
{
lean_object* v___f_181_; 
v___f_181_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___redArg___lam__0), 3, 1);
lean_closure_set(v___f_181_, 0, v_l_180_);
return v___f_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1(lean_object* v_l_182_, lean_object* v_u_183_, lean_object* v_hu_184_, lean_object* v_hl_185_, lean_object* v_h__u__l_186_, lean_object* v_h__l__u_187_){
_start:
{
lean_object* v___f_188_; 
v___f_188_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___redArg___lam__0), 3, 1);
lean_closure_set(v___f_188_, 0, v_l_182_);
return v___f_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1___boxed(lean_object* v_l_189_, lean_object* v_u_190_, lean_object* v_hu_191_, lean_object* v_hl_192_, lean_object* v_h__u__l_193_, lean_object* v_h__l__u_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_mathlib_GaloisInsertion_monotoneIntro___at___00NNRat_gi_spec__1(v_l_189_, v_u_190_, v_hu_191_, v_hl_192_, v_h__u__l_193_, v_h__l__u_194_);
lean_dec_ref(v_u_190_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_coeHom___lam__0(lean_object* v___y_200_){
_start:
{
lean_inc_ref(v___y_200_);
return v___y_200_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_coeHom___lam__0___boxed(lean_object* v___y_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib_NNRat_coeHom___lam__0(v___y_201_);
lean_dec_ref(v___y_201_);
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_abs___at___00Rat_nnabs_spec__0(lean_object* v_a_205_){
_start:
{
lean_object* v___x_206_; uint8_t v___x_207_; 
lean_inc_ref_n(v_a_205_, 2);
v___x_206_ = l_Rat_neg(v_a_205_);
lean_inc_ref(v___x_206_);
v___x_207_ = l_Rat_instDecidableLe(v_a_205_, v___x_206_);
if (v___x_207_ == 0)
{
lean_dec_ref(v___x_206_);
return v_a_205_;
}
else
{
lean_dec_ref(v_a_205_);
return v___x_206_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_nnabs(lean_object* v_x_208_){
_start:
{
lean_object* v___x_209_; 
v___x_209_ = lp_mathlib_abs___at___00Rat_nnabs_spec__0(v_x_208_);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_divNat(lean_object* v_n_210_, lean_object* v_d_211_){
_start:
{
lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; 
v___x_212_ = lean_nat_to_int(v_n_210_);
v___x_213_ = lean_nat_to_int(v_d_211_);
v___x_214_ = l_Rat_divInt(v___x_212_, v___x_213_);
lean_dec(v___x_213_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_numDenCasesOn___redArg(lean_object* v_q_215_, lean_object* v_H_216_){
_start:
{
lean_object* v_den_217_; lean_object* v___x_218_; lean_object* v___x_219_; 
v_den_217_ = lean_ctor_get(v_q_215_, 1);
lean_inc(v_den_217_);
v___x_218_ = lp_mathlib_NNRat_num(v_q_215_);
lean_dec_ref(v_q_215_);
v___x_219_ = lean_apply_4(v_H_216_, v___x_218_, v_den_217_, lean_box(0), lean_box(0));
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NNRat_numDenCasesOn(lean_object* v_C_220_, lean_object* v_q_221_, lean_object* v_H_222_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lp_mathlib_NNRat_numDenCasesOn___redArg(v_q_221_, v_H_222_);
return v___x_223_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Ring_Rat(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Operations(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Bounds_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_NNRat_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Ring_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Bounds_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instCommSemiringNNRat___aux__1 = _init_lp_mathlib_instCommSemiringNNRat___aux__1();
lean_mark_persistent(lp_mathlib_instCommSemiringNNRat___aux__1);
lp_mathlib_instCommSemiringNNRat___aux__13 = _init_lp_mathlib_instCommSemiringNNRat___aux__13();
lean_mark_persistent(lp_mathlib_instCommSemiringNNRat___aux__13);
lp_mathlib_instCommSemiringNNRat = _init_lp_mathlib_instCommSemiringNNRat();
lean_mark_persistent(lp_mathlib_instCommSemiringNNRat);
lp_mathlib_instAddCancelCommMonoidNNRat = _init_lp_mathlib_instAddCancelCommMonoidNNRat();
lean_mark_persistent(lp_mathlib_instAddCancelCommMonoidNNRat);
lp_mathlib_instLinearOrderNNRat = _init_lp_mathlib_instLinearOrderNNRat();
lean_mark_persistent(lp_mathlib_instLinearOrderNNRat);
lp_mathlib_instInhabitedNNRat___aux__1 = _init_lp_mathlib_instInhabitedNNRat___aux__1();
lean_mark_persistent(lp_mathlib_instInhabitedNNRat___aux__1);
lp_mathlib_instInhabitedNNRat = _init_lp_mathlib_instInhabitedNNRat();
lean_mark_persistent(lp_mathlib_instInhabitedNNRat);
lp_mathlib_NNRat_instOrderBot = _init_lp_mathlib_NNRat_instOrderBot();
lean_mark_persistent(lp_mathlib_NNRat_instOrderBot);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_NNRat_Defs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_specialised__high__priority__simp__lemma = _init_lp_mathlib_LibraryNote_specialised__high__priority__simp__lemma();
lean_mark_persistent(lp_mathlib_LibraryNote_specialised__high__priority__simp__lemma);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Rat(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Bounds_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_NNRat_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Group_Unbundled_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Nonneg_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Unbundled_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Rat(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Bounds_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_NNRat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_NNRat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_NNRat_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
