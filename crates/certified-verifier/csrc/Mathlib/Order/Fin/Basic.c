// Lean compiler output
// Module: Mathlib.Order.Fin.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Order.IsBotOne public import Mathlib.Data.Fin.Embedding public import Mathlib.Data.Fin.Rev public import Mathlib.Order.Heyting.Basic public import Mathlib.Order.Hom.Basic
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
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_mod(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_instOrdNat___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_valEmbedding___lam__0___boxed(lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_succAbove___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_instDecidableEqFin___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Fin_decLe___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Fin_decLt___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
lean_object* lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_Fin_equivSubtype(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* l_Fin_natAdd___boxed(lean_object*, lean_object*, lean_object*);
lean_object* l_Fin_succ___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_revPerm(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Fin_predAbove___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instMax__mathlib___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instMax__mathlib___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Fin_instMax__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fin_instMax__mathlib___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fin_instMax__mathlib___closed__0 = (const lean_object*)&lp_mathlib_Fin_instMax__mathlib___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Fin_instMax__mathlib(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instMax__mathlib___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instMin__mathlib___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instMin__mathlib___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Fin_instMin__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fin_instMin__mathlib___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fin_instMin__mathlib___closed__0 = (const lean_object*)&lp_mathlib_Fin_instMin__mathlib___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Fin_instMin__mathlib(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instMin__mathlib___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Fin_instLinearOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instOrdNat___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fin_instLinearOrder___closed__0 = (const lean_object*)&lp_mathlib_Fin_instLinearOrder___closed__0_value;
static const lean_ctor_object lp_mathlib_Fin_instLinearOrder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Fin_instLinearOrder___closed__1 = (const lean_object*)&lp_mathlib_Fin_instLinearOrder___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLinearOrder(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBoundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBoundedOrder___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBoundedOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBoundedOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instPartialOrder(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLattice(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instHeytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instHeytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCoheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCoheytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_predAboveOrderHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_orderIsoSubtype(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_orderIsoSubtype___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castOrderIso___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castOrderIso___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Fin_castOrderIso___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fin_castOrderIso___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fin_castOrderIso___closed__0 = (const lean_object*)&lp_mathlib_Fin_castOrderIso___closed__0_value;
static const lean_ctor_object lp_mathlib_Fin_castOrderIso___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Fin_castOrderIso___closed__0_value),((lean_object*)&lp_mathlib_Fin_castOrderIso___closed__0_value)}};
static const lean_object* lp_mathlib_Fin_castOrderIso___closed__1 = (const lean_object*)&lp_mathlib_Fin_castOrderIso___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Fin_castOrderIso(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castOrderIso___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Fin_revOrderIso___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Fin_revOrderIso___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Fin_revOrderIso(lean_object*);
static const lean_closure_object lp_mathlib_Fin_valOrderEmb___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Fin_valEmbedding___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Fin_valOrderEmb___closed__0 = (const lean_object*)&lp_mathlib_Fin_valOrderEmb___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Fin_valOrderEmb(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_valOrderEmb___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_OrderEmbedding_instInhabitedOrderEmbeddingNat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_OrderEmbedding_instInhabitedOrderEmbeddingNat___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_succOrderEmb(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEOrderEmb(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEOrderEmb___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castAddOrderEmb(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castAddOrderEmb___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castSuccOrderEmb(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_castSuccOrderEmb___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatOrderEmb___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatOrderEmb___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatOrderEmb___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatOrderEmb(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatOrderEmb___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAddOrderEmb(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAboveOrderEmb(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fin_instMax__mathlib___lam__0(lean_object* v_x_1_, lean_object* v_y_2_){
_start:
{
uint8_t v___x_3_; 
v___x_3_ = lean_nat_dec_le(v_x_1_, v_y_2_);
if (v___x_3_ == 0)
{
lean_inc(v_x_1_);
return v_x_1_;
}
else
{
lean_inc(v_y_2_);
return v_y_2_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instMax__mathlib___lam__0___boxed(lean_object* v_x_4_, lean_object* v_y_5_){
_start:
{
lean_object* v_res_6_; 
v_res_6_ = lp_mathlib_Fin_instMax__mathlib___lam__0(v_x_4_, v_y_5_);
lean_dec(v_y_5_);
lean_dec(v_x_4_);
return v_res_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instMax__mathlib(lean_object* v_n_8_){
_start:
{
lean_object* v___f_9_; 
v___f_9_ = ((lean_object*)(lp_mathlib_Fin_instMax__mathlib___closed__0));
return v___f_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instMax__mathlib___boxed(lean_object* v_n_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_mathlib_Fin_instMax__mathlib(v_n_10_);
lean_dec(v_n_10_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instMin__mathlib___lam__0(lean_object* v_x_12_, lean_object* v_y_13_){
_start:
{
uint8_t v___x_14_; 
v___x_14_ = lean_nat_dec_le(v_x_12_, v_y_13_);
if (v___x_14_ == 0)
{
lean_inc(v_y_13_);
return v_y_13_;
}
else
{
lean_inc(v_x_12_);
return v_x_12_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instMin__mathlib___lam__0___boxed(lean_object* v_x_15_, lean_object* v_y_16_){
_start:
{
lean_object* v_res_17_; 
v_res_17_ = lp_mathlib_Fin_instMin__mathlib___lam__0(v_x_15_, v_y_16_);
lean_dec(v_y_16_);
lean_dec(v_x_15_);
return v_res_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instMin__mathlib(lean_object* v_n_19_){
_start:
{
lean_object* v___f_20_; 
v___f_20_ = ((lean_object*)(lp_mathlib_Fin_instMin__mathlib___closed__0));
return v___f_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instMin__mathlib___boxed(lean_object* v_n_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_Fin_instMin__mathlib(v_n_21_);
lean_dec(v_n_21_);
return v_res_22_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLinearOrder(lean_object* v_n_27_){
_start:
{
lean_object* v___f_28_; lean_object* v___f_29_; lean_object* v___f_30_; lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; 
v___f_28_ = ((lean_object*)(lp_mathlib_Fin_instMax__mathlib___closed__0));
v___f_29_ = ((lean_object*)(lp_mathlib_Fin_instMin__mathlib___closed__0));
v___f_30_ = ((lean_object*)(lp_mathlib_Fin_instLinearOrder___closed__0));
lean_inc_n(v_n_27_, 2);
v___x_31_ = lean_alloc_closure((void*)(l_instDecidableEqFin___boxed), 3, 1);
lean_closure_set(v___x_31_, 0, v_n_27_);
v___x_32_ = lean_alloc_closure((void*)(l_Fin_decLe___boxed), 3, 1);
lean_closure_set(v___x_32_, 0, v_n_27_);
v___x_33_ = lean_alloc_closure((void*)(l_Fin_decLt___boxed), 3, 1);
lean_closure_set(v___x_33_, 0, v_n_27_);
v___x_34_ = ((lean_object*)(lp_mathlib_Fin_instLinearOrder___closed__1));
v___x_35_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_35_, 0, v___x_34_);
lean_ctor_set(v___x_35_, 1, v___f_29_);
lean_ctor_set(v___x_35_, 2, v___f_28_);
lean_ctor_set(v___x_35_, 3, v___f_30_);
lean_ctor_set(v___x_35_, 4, v___x_32_);
lean_ctor_set(v___x_35_, 5, v___x_31_);
lean_ctor_set(v___x_35_, 6, v___x_33_);
return v___x_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBoundedOrder___redArg(lean_object* v_n_36_){
_start:
{
lean_object* v___x_37_; lean_object* v___x_38_; lean_object* v___x_39_; lean_object* v___x_40_; lean_object* v___x_41_; lean_object* v___x_42_; 
v___x_37_ = lean_unsigned_to_nat(0u);
v___x_38_ = lean_nat_mod(v___x_37_, v_n_36_);
v___x_39_ = lean_unsigned_to_nat(1u);
v___x_40_ = lean_nat_add(v___x_38_, v___x_39_);
v___x_41_ = lean_nat_sub(v_n_36_, v___x_40_);
lean_dec(v___x_40_);
v___x_42_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_42_, 0, v___x_41_);
lean_ctor_set(v___x_42_, 1, v___x_38_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBoundedOrder___redArg___boxed(lean_object* v_n_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_Fin_instBoundedOrder___redArg(v_n_43_);
lean_dec(v_n_43_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBoundedOrder(lean_object* v_n_45_, lean_object* v_inst_46_){
_start:
{
lean_object* v___x_47_; 
v___x_47_ = lp_mathlib_Fin_instBoundedOrder___redArg(v_n_45_);
return v___x_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBoundedOrder___boxed(lean_object* v_n_48_, lean_object* v_inst_49_){
_start:
{
lean_object* v_res_50_; 
v_res_50_ = lp_mathlib_Fin_instBoundedOrder(v_n_48_, v_inst_49_);
lean_dec(v_n_48_);
return v_res_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__0(lean_object* v_n_51_, lean_object* v_a_52_, lean_object* v_b_53_){
_start:
{
uint8_t v___x_54_; 
v___x_54_ = lean_nat_dec_le(v_a_52_, v_b_53_);
if (v___x_54_ == 0)
{
lean_inc(v_a_52_);
return v_a_52_;
}
else
{
lean_object* v___x_55_; lean_object* v___x_56_; 
v___x_55_ = lean_unsigned_to_nat(0u);
v___x_56_ = lean_nat_mod(v___x_55_, v_n_51_);
return v___x_56_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__0___boxed(lean_object* v_n_57_, lean_object* v_a_58_, lean_object* v_b_59_){
_start:
{
lean_object* v_res_60_; 
v_res_60_ = lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__0(v_n_57_, v_a_58_, v_b_59_);
lean_dec(v_b_59_);
lean_dec(v_a_58_);
lean_dec(v_n_57_);
return v_res_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__1(lean_object* v_n_61_, lean_object* v_a_62_){
_start:
{
lean_object* v___x_63_; lean_object* v___x_64_; lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; uint8_t v___x_68_; 
v___x_63_ = lean_unsigned_to_nat(0u);
v___x_64_ = lean_nat_mod(v___x_63_, v_n_61_);
v___x_65_ = lean_unsigned_to_nat(1u);
v___x_66_ = lean_nat_add(v___x_64_, v___x_65_);
v___x_67_ = lean_nat_sub(v_n_61_, v___x_66_);
lean_dec(v___x_66_);
v___x_68_ = lean_nat_dec_eq(v_a_62_, v___x_67_);
if (v___x_68_ == 0)
{
lean_dec(v___x_64_);
return v___x_67_;
}
else
{
lean_dec(v___x_67_);
return v___x_64_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__1___boxed(lean_object* v_n_69_, lean_object* v_a_70_){
_start:
{
lean_object* v_res_71_; 
v_res_71_ = lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__1(v_n_69_, v_a_70_);
lean_dec(v_a_70_);
lean_dec(v_n_69_);
return v_res_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__2(lean_object* v_n_72_, lean_object* v_a_73_){
_start:
{
lean_object* v___x_74_; lean_object* v___x_75_; uint8_t v___x_76_; 
v___x_74_ = lean_unsigned_to_nat(0u);
v___x_75_ = lean_nat_mod(v___x_74_, v_n_72_);
v___x_76_ = lean_nat_dec_eq(v_a_73_, v___x_75_);
if (v___x_76_ == 0)
{
return v___x_75_;
}
else
{
lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_77_ = lean_unsigned_to_nat(1u);
v___x_78_ = lean_nat_add(v___x_75_, v___x_77_);
lean_dec(v___x_75_);
v___x_79_ = lean_nat_sub(v_n_72_, v___x_78_);
lean_dec(v___x_78_);
return v___x_79_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__2___boxed(lean_object* v_n_80_, lean_object* v_a_81_){
_start:
{
lean_object* v_res_82_; 
v_res_82_ = lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__2(v_n_80_, v_a_81_);
lean_dec(v_a_81_);
lean_dec(v_n_80_);
return v_res_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__3(lean_object* v_n_83_, lean_object* v_a_84_, lean_object* v_b_85_){
_start:
{
uint8_t v___x_86_; 
v___x_86_ = lean_nat_dec_le(v_a_84_, v_b_85_);
if (v___x_86_ == 0)
{
lean_inc(v_b_85_);
return v_b_85_;
}
else
{
lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
v___x_87_ = lean_unsigned_to_nat(0u);
v___x_88_ = lean_nat_mod(v___x_87_, v_n_83_);
v___x_89_ = lean_unsigned_to_nat(1u);
v___x_90_ = lean_nat_add(v___x_88_, v___x_89_);
lean_dec(v___x_88_);
v___x_91_ = lean_nat_sub(v_n_83_, v___x_90_);
lean_dec(v___x_90_);
return v___x_91_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__3___boxed(lean_object* v_n_92_, lean_object* v_a_93_, lean_object* v_b_94_){
_start:
{
lean_object* v_res_95_; 
v_res_95_ = lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__3(v_n_92_, v_a_93_, v_b_94_);
lean_dec(v_b_94_);
lean_dec(v_a_93_);
lean_dec(v_n_92_);
return v_res_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra___redArg(lean_object* v_n_96_){
_start:
{
lean_object* v___f_97_; lean_object* v___f_98_; lean_object* v___f_99_; lean_object* v___f_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; 
lean_inc_n(v_n_96_, 5);
v___f_97_ = lean_alloc_closure((void*)(lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_97_, 0, v_n_96_);
v___f_98_ = lean_alloc_closure((void*)(lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__1___boxed), 2, 1);
lean_closure_set(v___f_98_, 0, v_n_96_);
v___f_99_ = lean_alloc_closure((void*)(lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__2___boxed), 2, 1);
lean_closure_set(v___f_99_, 0, v_n_96_);
v___f_100_ = lean_alloc_closure((void*)(lp_mathlib_Fin_instBiheytingAlgebra___redArg___lam__3___boxed), 3, 1);
lean_closure_set(v___f_100_, 0, v_n_96_);
v___x_101_ = lp_mathlib_Fin_instLinearOrder(v_n_96_);
v___x_102_ = lp_mathlib_LinearOrder_toLattice___redArg(v___x_101_);
lean_dec_ref(v___x_101_);
v___x_103_ = lean_unsigned_to_nat(0u);
v___x_104_ = lean_nat_mod(v___x_103_, v_n_96_);
v___x_105_ = lean_unsigned_to_nat(1u);
v___x_106_ = lean_nat_add(v___x_104_, v___x_105_);
v___x_107_ = lean_nat_sub(v_n_96_, v___x_106_);
lean_dec(v___x_106_);
lean_dec(v_n_96_);
v___x_108_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_108_, 0, v___x_102_);
lean_ctor_set(v___x_108_, 1, v___x_107_);
lean_ctor_set(v___x_108_, 2, v___f_100_);
v___x_109_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_109_, 0, v___x_108_);
lean_ctor_set(v___x_109_, 1, v___x_104_);
lean_ctor_set(v___x_109_, 2, v___f_99_);
v___x_110_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_110_, 0, v___x_109_);
lean_ctor_set(v___x_110_, 1, v___f_97_);
lean_ctor_set(v___x_110_, 2, v___f_98_);
return v___x_110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instBiheytingAlgebra(lean_object* v_n_111_, lean_object* v_inst_112_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = lp_mathlib_Fin_instBiheytingAlgebra___redArg(v_n_111_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instPartialOrder(lean_object* v_n_114_){
_start:
{
lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v_toPartialOrder_118_; 
v___x_115_ = lp_mathlib_Fin_instLinearOrder(v_n_114_);
v___x_116_ = lp_mathlib_LinearOrder_toLattice___redArg(v___x_115_);
lean_dec_ref(v___x_115_);
v___x_117_ = lp_mathlib_Lattice_toSemilatticeInf___redArg(v___x_116_);
v_toPartialOrder_118_ = lean_ctor_get(v___x_117_, 0);
lean_inc_ref(v_toPartialOrder_118_);
lean_dec_ref(v___x_117_);
return v_toPartialOrder_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instLattice(lean_object* v_n_119_){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_120_ = lp_mathlib_Fin_instLinearOrder(v_n_119_);
v___x_121_ = lp_mathlib_LinearOrder_toLattice___redArg(v___x_120_);
lean_dec_ref(v___x_120_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instHeytingAlgebra___redArg(lean_object* v_n_122_){
_start:
{
lean_object* v___x_123_; lean_object* v_toHeytingAlgebra_124_; 
v___x_123_ = lp_mathlib_Fin_instBiheytingAlgebra___redArg(v_n_122_);
v_toHeytingAlgebra_124_ = lean_ctor_get(v___x_123_, 0);
lean_inc_ref(v_toHeytingAlgebra_124_);
lean_dec_ref(v___x_123_);
return v_toHeytingAlgebra_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instHeytingAlgebra(lean_object* v_n_125_, lean_object* v_inst_126_){
_start:
{
lean_object* v___x_127_; 
v___x_127_ = lp_mathlib_Fin_instHeytingAlgebra___redArg(v_n_125_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCoheytingAlgebra___redArg(lean_object* v_n_128_){
_start:
{
lean_object* v___x_129_; lean_object* v___x_130_; 
v___x_129_ = lp_mathlib_Fin_instBiheytingAlgebra___redArg(v_n_128_);
v___x_130_ = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(v___x_129_);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_instCoheytingAlgebra(lean_object* v_n_131_, lean_object* v_inst_132_){
_start:
{
lean_object* v___x_133_; 
v___x_133_ = lp_mathlib_Fin_instCoheytingAlgebra___redArg(v_n_131_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_predAboveOrderHom(lean_object* v_n_134_, lean_object* v_p_135_){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lean_alloc_closure((void*)(lp_mathlib_Fin_predAbove___boxed), 3, 2);
lean_closure_set(v___x_136_, 0, v_n_134_);
lean_closure_set(v___x_136_, 1, v_p_135_);
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_orderIsoSubtype(lean_object* v_n_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = lp_mathlib_Fin_equivSubtype(v_n_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_orderIsoSubtype___boxed(lean_object* v_n_139_){
_start:
{
lean_object* v_res_140_; 
v_res_140_ = lp_mathlib_Fin_orderIsoSubtype(v_n_139_);
lean_dec(v_n_139_);
return v_res_140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castOrderIso___lam__0(lean_object* v___y_141_){
_start:
{
lean_inc(v___y_141_);
return v___y_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castOrderIso___lam__0___boxed(lean_object* v___y_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_mathlib_Fin_castOrderIso___lam__0(v___y_142_);
lean_dec(v___y_142_);
return v_res_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castOrderIso(lean_object* v_m_147_, lean_object* v_n_148_, lean_object* v_eq_149_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = ((lean_object*)(lp_mathlib_Fin_castOrderIso___closed__1));
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castOrderIso___boxed(lean_object* v_m_151_, lean_object* v_n_152_, lean_object* v_eq_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib_Fin_castOrderIso(v_m_151_, v_n_152_, v_eq_153_);
lean_dec(v_n_152_);
lean_dec(v_m_151_);
return v_res_154_;
}
}
static lean_object* _init_lp_mathlib_Fin_revOrderIso___closed__0(void){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_revOrderIso(lean_object* v_n_156_){
_start:
{
lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; 
v___x_157_ = lean_obj_once(&lp_mathlib_Fin_revOrderIso___closed__0, &lp_mathlib_Fin_revOrderIso___closed__0_once, _init_lp_mathlib_Fin_revOrderIso___closed__0);
v___x_158_ = lp_mathlib_Fin_revPerm(v_n_156_);
v___x_159_ = lp_mathlib_Equiv_trans___redArg(v___x_157_, v___x_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_valOrderEmb(lean_object* v_n_161_){
_start:
{
lean_object* v___f_162_; 
v___f_162_ = ((lean_object*)(lp_mathlib_Fin_valOrderEmb___closed__0));
return v___f_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_valOrderEmb___boxed(lean_object* v_n_163_){
_start:
{
lean_object* v_res_164_; 
v_res_164_ = lp_mathlib_Fin_valOrderEmb(v_n_163_);
lean_dec(v_n_163_);
return v_res_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_OrderEmbedding_instInhabitedOrderEmbeddingNat(lean_object* v_n_165_){
_start:
{
lean_object* v___f_166_; 
v___f_166_ = ((lean_object*)(lp_mathlib_Fin_valOrderEmb___closed__0));
return v___f_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_OrderEmbedding_instInhabitedOrderEmbeddingNat___boxed(lean_object* v_n_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_mathlib_Fin_OrderEmbedding_instInhabitedOrderEmbeddingNat(v_n_167_);
lean_dec(v_n_167_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succOrderEmb(lean_object* v_n_169_){
_start:
{
lean_object* v___x_170_; 
v___x_170_ = lean_alloc_closure((void*)(l_Fin_succ___boxed), 2, 1);
lean_closure_set(v___x_170_, 0, v_n_169_);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEOrderEmb(lean_object* v_m_171_, lean_object* v_n_172_, lean_object* v_h_173_){
_start:
{
lean_object* v___f_174_; 
v___f_174_ = ((lean_object*)(lp_mathlib_Fin_castOrderIso___closed__0));
return v___f_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castLEOrderEmb___boxed(lean_object* v_m_175_, lean_object* v_n_176_, lean_object* v_h_177_){
_start:
{
lean_object* v_res_178_; 
v_res_178_ = lp_mathlib_Fin_castLEOrderEmb(v_m_175_, v_n_176_, v_h_177_);
lean_dec(v_n_176_);
lean_dec(v_m_175_);
return v_res_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castAddOrderEmb(lean_object* v_n_179_, lean_object* v_m_180_){
_start:
{
lean_object* v___f_181_; 
v___f_181_ = ((lean_object*)(lp_mathlib_Fin_castOrderIso___closed__0));
return v___f_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castAddOrderEmb___boxed(lean_object* v_n_182_, lean_object* v_m_183_){
_start:
{
lean_object* v_res_184_; 
v_res_184_ = lp_mathlib_Fin_castAddOrderEmb(v_n_182_, v_m_183_);
lean_dec(v_m_183_);
lean_dec(v_n_182_);
return v_res_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castSuccOrderEmb(lean_object* v_n_185_){
_start:
{
lean_object* v___f_186_; 
v___f_186_ = ((lean_object*)(lp_mathlib_Fin_castOrderIso___closed__0));
return v___f_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_castSuccOrderEmb___boxed(lean_object* v_n_187_){
_start:
{
lean_object* v_res_188_; 
v_res_188_ = lp_mathlib_Fin_castSuccOrderEmb(v_n_187_);
lean_dec(v_n_187_);
return v_res_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatOrderEmb___redArg___lam__0(lean_object* v_m_189_, lean_object* v_x_190_){
_start:
{
lean_object* v___x_191_; 
v___x_191_ = lean_nat_add(v_x_190_, v_m_189_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatOrderEmb___redArg___lam__0___boxed(lean_object* v_m_192_, lean_object* v_x_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_mathlib_Fin_addNatOrderEmb___redArg___lam__0(v_m_192_, v_x_193_);
lean_dec(v_x_193_);
lean_dec(v_m_192_);
return v_res_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatOrderEmb___redArg(lean_object* v_m_195_){
_start:
{
lean_object* v___f_196_; 
v___f_196_ = lean_alloc_closure((void*)(lp_mathlib_Fin_addNatOrderEmb___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_196_, 0, v_m_195_);
return v___f_196_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatOrderEmb(lean_object* v_n_197_, lean_object* v_m_198_){
_start:
{
lean_object* v___f_199_; 
v___f_199_ = lean_alloc_closure((void*)(lp_mathlib_Fin_addNatOrderEmb___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_199_, 0, v_m_198_);
return v___f_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_addNatOrderEmb___boxed(lean_object* v_n_200_, lean_object* v_m_201_){
_start:
{
lean_object* v_res_202_; 
v_res_202_ = lp_mathlib_Fin_addNatOrderEmb(v_n_200_, v_m_201_);
lean_dec(v_n_200_);
return v_res_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_natAddOrderEmb(lean_object* v_m_203_, lean_object* v_n_204_){
_start:
{
lean_object* v___x_205_; 
v___x_205_ = lean_alloc_closure((void*)(l_Fin_natAdd___boxed), 3, 2);
lean_closure_set(v___x_205_, 0, v_m_203_);
lean_closure_set(v___x_205_, 1, v_n_204_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fin_succAboveOrderEmb(lean_object* v_n_206_, lean_object* v_p_207_){
_start:
{
lean_object* v___x_208_; 
v___x_208_ = lean_alloc_closure((void*)(lp_mathlib_Fin_succAbove___boxed), 3, 2);
lean_closure_set(v___x_208_, 0, v_n_206_);
lean_closure_set(v___x_208_, 1, v_p_207_);
return v___x_208_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Order_IsBotOne(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_Embedding(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fin_Rev(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Heyting_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Fin_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Order_IsBotOne(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_Embedding(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fin_Rev(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Heyting_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Fin_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Order_IsBotOne(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fin_Embedding(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fin_Rev(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Heyting_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Fin_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_IsBotOne(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_Embedding(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fin_Rev(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Heyting_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Fin_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Fin_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
