// Lean compiler output
// Module: Mathlib.Order.Hom.CompleteLattice
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Lattice.Image public import Mathlib.Order.Hom.BoundedLattice
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
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_OrderDual_instCompleteLattice___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(lean_object*);
lean_object* lp_mathlib_InfHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_tosSupHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_tosSupHom___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_tosSupHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_tosSupHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSSupHomOfSSupHomClass___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSSupHomOfSSupHomClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSSupHomOfSSupHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSSupHomOfSSupHomClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSInfHomOfSInfHomClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSInfHomOfSInfHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSInfHomOfSInfHomClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCFrameHomOfFrameHomClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCFrameHomOfFrameHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCFrameHomOfFrameHomClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCCompleteLatticeHomOfCompleteLatticeHomClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCCompleteLatticeHomOfCompleteLatticeHomClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCCompleteLatticeHomOfCompleteLatticeHomClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instFunLike___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_sSupHom_instFunLike___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_sSupHom_instFunLike___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_sSupHom_instFunLike___closed__0 = (const lean_object*)&lp_mathlib_sSupHom_instFunLike___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instFunLike(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instFunLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instFunLike(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instFunLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_sSupHom_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_sSupHom_id___closed__0 = (const lean_object*)&lp_mathlib_sSupHom_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_comp___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_comp___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_sSupHom_instPartialOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_sSupHom_instPartialOrder___closed__0 = (const lean_object*)&lp_mathlib_sSupHom_instPartialOrder___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instBot___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instBot___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instBot(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instTop___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instTop___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instTop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instOrderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instOrderBot(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instOrderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instOrderTop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instOrderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_toLatticeHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_toLatticeHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_toLatticeHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_instPartialOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_OrderIso_toCompleteLatticeHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_OrderIso_toCompleteLatticeHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_OrderIso_toCompleteLatticeHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_OrderIso_toCompleteLatticeHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_toBoundedLatticeHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_toBoundedLatticeHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_toBoundedLatticeHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_instInhabited(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_sSupHom_dual___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_sSupHom_dual___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_dual___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_sSupHom_dual___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_sSupHom_dual___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_sSupHom_dual___closed__0 = (const lean_object*)&lp_mathlib_sSupHom_dual___closed__0_value;
static const lean_ctor_object lp_mathlib_sSupHom_dual___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_sSupHom_dual___closed__0_value),((lean_object*)&lp_mathlib_sSupHom_dual___closed__0_value)}};
static const lean_object* lp_mathlib_sSupHom_dual___closed__1 = (const lean_object*)&lp_mathlib_sSupHom_dual___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_dual(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_dual___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_dual(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_dual___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_dual___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_dual___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_dual___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_dual(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_setPreimage(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_setPreimage___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_setImage(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_setImage___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toOrderIsoSet(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toOrderIsoSet___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_supsSupHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_supsSupHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_supsSupHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infsInfHom___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infsInfHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infsInfHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_tosSupHom___redArg(lean_object* v_self_1_){
_start:
{
lean_inc(v_self_1_);
return v_self_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_tosSupHom___redArg___boxed(lean_object* v_self_2_){
_start:
{
lean_object* v_res_3_; 
v_res_3_ = lp_mathlib_CompleteLatticeHom_tosSupHom___redArg(v_self_2_);
lean_dec(v_self_2_);
return v_res_3_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_tosSupHom(lean_object* v_00_u03b1_4_, lean_object* v_00_u03b2_5_, lean_object* v_inst_6_, lean_object* v_inst_7_, lean_object* v_self_8_){
_start:
{
lean_inc(v_self_8_);
return v_self_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_tosSupHom___boxed(lean_object* v_00_u03b1_9_, lean_object* v_00_u03b2_10_, lean_object* v_inst_11_, lean_object* v_inst_12_, lean_object* v_self_13_){
_start:
{
lean_object* v_res_14_; 
v_res_14_ = lp_mathlib_CompleteLatticeHom_tosSupHom(v_00_u03b1_9_, v_00_u03b2_10_, v_inst_11_, v_inst_12_, v_self_13_);
lean_dec(v_self_13_);
lean_dec_ref(v_inst_12_);
lean_dec_ref(v_inst_11_);
return v_res_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSSupHomOfSSupHomClass___redArg___lam__0(lean_object* v_inst_15_, lean_object* v_f_16_, lean_object* v___y_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lean_apply_2(v_inst_15_, v_f_16_, v___y_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSSupHomOfSSupHomClass___redArg(lean_object* v_inst_19_){
_start:
{
lean_object* v___f_20_; 
v___f_20_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSSupHomOfSSupHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_20_, 0, v_inst_19_);
return v___f_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSSupHomOfSSupHomClass(lean_object* v_F_21_, lean_object* v_00_u03b1_22_, lean_object* v_00_u03b2_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_inst_26_, lean_object* v_inst_27_){
_start:
{
lean_object* v___f_28_; 
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSSupHomOfSSupHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_28_, 0, v_inst_24_);
return v___f_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSSupHomOfSSupHomClass___boxed(lean_object* v_F_29_, lean_object* v_00_u03b1_30_, lean_object* v_00_u03b2_31_, lean_object* v_inst_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_instCoeTCSSupHomOfSSupHomClass(v_F_29_, v_00_u03b1_30_, v_00_u03b2_31_, v_inst_32_, v_inst_33_, v_inst_34_, v_inst_35_);
lean_dec(v_inst_34_);
lean_dec(v_inst_33_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSInfHomOfSInfHomClass___redArg(lean_object* v_inst_37_){
_start:
{
lean_object* v___f_38_; 
v___f_38_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSSupHomOfSSupHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_38_, 0, v_inst_37_);
return v___f_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSInfHomOfSInfHomClass(lean_object* v_F_39_, lean_object* v_00_u03b1_40_, lean_object* v_00_u03b2_41_, lean_object* v_inst_42_, lean_object* v_inst_43_, lean_object* v_inst_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v___f_46_; 
v___f_46_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSSupHomOfSSupHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_46_, 0, v_inst_42_);
return v___f_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCSInfHomOfSInfHomClass___boxed(lean_object* v_F_47_, lean_object* v_00_u03b1_48_, lean_object* v_00_u03b2_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_inst_53_){
_start:
{
lean_object* v_res_54_; 
v_res_54_ = lp_mathlib_instCoeTCSInfHomOfSInfHomClass(v_F_47_, v_00_u03b1_48_, v_00_u03b2_49_, v_inst_50_, v_inst_51_, v_inst_52_, v_inst_53_);
lean_dec(v_inst_52_);
lean_dec(v_inst_51_);
return v_res_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCFrameHomOfFrameHomClass___redArg(lean_object* v_inst_55_){
_start:
{
lean_object* v___f_56_; 
v___f_56_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSSupHomOfSSupHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_56_, 0, v_inst_55_);
return v___f_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCFrameHomOfFrameHomClass(lean_object* v_F_57_, lean_object* v_00_u03b1_58_, lean_object* v_00_u03b2_59_, lean_object* v_inst_60_, lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_inst_63_){
_start:
{
lean_object* v___f_64_; 
v___f_64_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSSupHomOfSSupHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_64_, 0, v_inst_60_);
return v___f_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCFrameHomOfFrameHomClass___boxed(lean_object* v_F_65_, lean_object* v_00_u03b1_66_, lean_object* v_00_u03b2_67_, lean_object* v_inst_68_, lean_object* v_inst_69_, lean_object* v_inst_70_, lean_object* v_inst_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_mathlib_instCoeTCFrameHomOfFrameHomClass(v_F_65_, v_00_u03b1_66_, v_00_u03b2_67_, v_inst_68_, v_inst_69_, v_inst_70_, v_inst_71_);
lean_dec_ref(v_inst_70_);
lean_dec_ref(v_inst_69_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCCompleteLatticeHomOfCompleteLatticeHomClass___redArg(lean_object* v_inst_73_){
_start:
{
lean_object* v___f_74_; 
v___f_74_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSSupHomOfSSupHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_74_, 0, v_inst_73_);
return v___f_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCCompleteLatticeHomOfCompleteLatticeHomClass(lean_object* v_F_75_, lean_object* v_00_u03b1_76_, lean_object* v_00_u03b2_77_, lean_object* v_inst_78_, lean_object* v_inst_79_, lean_object* v_inst_80_, lean_object* v_inst_81_){
_start:
{
lean_object* v___f_82_; 
v___f_82_ = lean_alloc_closure((void*)(lp_mathlib_instCoeTCSSupHomOfSSupHomClass___redArg___lam__0), 3, 1);
lean_closure_set(v___f_82_, 0, v_inst_78_);
return v___f_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instCoeTCCompleteLatticeHomOfCompleteLatticeHomClass___boxed(lean_object* v_F_83_, lean_object* v_00_u03b1_84_, lean_object* v_00_u03b2_85_, lean_object* v_inst_86_, lean_object* v_inst_87_, lean_object* v_inst_88_, lean_object* v_inst_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_mathlib_instCoeTCCompleteLatticeHomOfCompleteLatticeHomClass(v_F_83_, v_00_u03b1_84_, v_00_u03b2_85_, v_inst_86_, v_inst_87_, v_inst_88_, v_inst_89_);
lean_dec_ref(v_inst_88_);
lean_dec_ref(v_inst_87_);
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instFunLike___lam__0(lean_object* v_self_91_, lean_object* v___y_92_){
_start:
{
lean_object* v___x_93_; 
v___x_93_ = lean_apply_1(v_self_91_, v___y_92_);
return v___x_93_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instFunLike(lean_object* v_00_u03b1_95_, lean_object* v_00_u03b2_96_, lean_object* v_inst_97_, lean_object* v_inst_98_){
_start:
{
lean_object* v___f_99_; 
v___f_99_ = ((lean_object*)(lp_mathlib_sSupHom_instFunLike___closed__0));
return v___f_99_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instFunLike___boxed(lean_object* v_00_u03b1_100_, lean_object* v_00_u03b2_101_, lean_object* v_inst_102_, lean_object* v_inst_103_){
_start:
{
lean_object* v_res_104_; 
v_res_104_ = lp_mathlib_sSupHom_instFunLike(v_00_u03b1_100_, v_00_u03b2_101_, v_inst_102_, v_inst_103_);
lean_dec(v_inst_103_);
lean_dec(v_inst_102_);
return v_res_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instFunLike(lean_object* v_00_u03b1_105_, lean_object* v_00_u03b2_106_, lean_object* v_inst_107_, lean_object* v_inst_108_){
_start:
{
lean_object* v___f_109_; 
v___f_109_ = ((lean_object*)(lp_mathlib_sSupHom_instFunLike___closed__0));
return v___f_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instFunLike___boxed(lean_object* v_00_u03b1_110_, lean_object* v_00_u03b2_111_, lean_object* v_inst_112_, lean_object* v_inst_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_mathlib_sInfHom_instFunLike(v_00_u03b1_110_, v_00_u03b2_111_, v_inst_112_, v_inst_113_);
lean_dec(v_inst_113_);
lean_dec(v_inst_112_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_copy___redArg(lean_object* v_f_x27_115_){
_start:
{
lean_inc(v_f_x27_115_);
return v_f_x27_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_copy___redArg___boxed(lean_object* v_f_x27_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_sSupHom_copy___redArg(v_f_x27_116_);
lean_dec(v_f_x27_116_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_copy(lean_object* v_00_u03b1_118_, lean_object* v_00_u03b2_119_, lean_object* v_inst_120_, lean_object* v_inst_121_, lean_object* v_f_122_, lean_object* v_f_x27_123_, lean_object* v_h_124_){
_start:
{
lean_inc(v_f_x27_123_);
return v_f_x27_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_copy___boxed(lean_object* v_00_u03b1_125_, lean_object* v_00_u03b2_126_, lean_object* v_inst_127_, lean_object* v_inst_128_, lean_object* v_f_129_, lean_object* v_f_x27_130_, lean_object* v_h_131_){
_start:
{
lean_object* v_res_132_; 
v_res_132_ = lp_mathlib_sSupHom_copy(v_00_u03b1_125_, v_00_u03b2_126_, v_inst_127_, v_inst_128_, v_f_129_, v_f_x27_130_, v_h_131_);
lean_dec(v_f_x27_130_);
lean_dec(v_f_129_);
lean_dec(v_inst_128_);
lean_dec(v_inst_127_);
return v_res_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_copy___redArg(lean_object* v_f_x27_133_){
_start:
{
lean_inc(v_f_x27_133_);
return v_f_x27_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_copy___redArg___boxed(lean_object* v_f_x27_134_){
_start:
{
lean_object* v_res_135_; 
v_res_135_ = lp_mathlib_sInfHom_copy___redArg(v_f_x27_134_);
lean_dec(v_f_x27_134_);
return v_res_135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_copy(lean_object* v_00_u03b1_136_, lean_object* v_00_u03b2_137_, lean_object* v_inst_138_, lean_object* v_inst_139_, lean_object* v_f_140_, lean_object* v_f_x27_141_, lean_object* v_h_142_){
_start:
{
lean_inc(v_f_x27_141_);
return v_f_x27_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_copy___boxed(lean_object* v_00_u03b1_143_, lean_object* v_00_u03b2_144_, lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_f_147_, lean_object* v_f_x27_148_, lean_object* v_h_149_){
_start:
{
lean_object* v_res_150_; 
v_res_150_ = lp_mathlib_sInfHom_copy(v_00_u03b1_143_, v_00_u03b2_144_, v_inst_145_, v_inst_146_, v_f_147_, v_f_x27_148_, v_h_149_);
lean_dec(v_f_x27_148_);
lean_dec(v_f_147_);
lean_dec(v_inst_146_);
lean_dec(v_inst_145_);
return v_res_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_id(lean_object* v_00_u03b1_152_, lean_object* v_inst_153_){
_start:
{
lean_object* v___x_154_; 
v___x_154_ = ((lean_object*)(lp_mathlib_sSupHom_id___closed__0));
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_id___boxed(lean_object* v_00_u03b1_155_, lean_object* v_inst_156_){
_start:
{
lean_object* v_res_157_; 
v_res_157_ = lp_mathlib_sSupHom_id(v_00_u03b1_155_, v_inst_156_);
lean_dec(v_inst_156_);
return v_res_157_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_id(lean_object* v_00_u03b1_158_, lean_object* v_inst_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = ((lean_object*)(lp_mathlib_sSupHom_id___closed__0));
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_id___boxed(lean_object* v_00_u03b1_161_, lean_object* v_inst_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_mathlib_sInfHom_id(v_00_u03b1_161_, v_inst_162_);
lean_dec(v_inst_162_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instInhabited(lean_object* v_00_u03b1_164_, lean_object* v_inst_165_){
_start:
{
lean_object* v___x_166_; 
v___x_166_ = ((lean_object*)(lp_mathlib_sSupHom_id___closed__0));
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instInhabited___boxed(lean_object* v_00_u03b1_167_, lean_object* v_inst_168_){
_start:
{
lean_object* v_res_169_; 
v_res_169_ = lp_mathlib_sSupHom_instInhabited(v_00_u03b1_167_, v_inst_168_);
lean_dec(v_inst_168_);
return v_res_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instInhabited(lean_object* v_00_u03b1_170_, lean_object* v_inst_171_){
_start:
{
lean_object* v___x_172_; 
v___x_172_ = ((lean_object*)(lp_mathlib_sSupHom_id___closed__0));
return v___x_172_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instInhabited___boxed(lean_object* v_00_u03b1_173_, lean_object* v_inst_174_){
_start:
{
lean_object* v_res_175_; 
v_res_175_ = lp_mathlib_sInfHom_instInhabited(v_00_u03b1_173_, v_inst_174_);
lean_dec(v_inst_174_);
return v_res_175_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_comp___redArg___lam__0(lean_object* v_f_176_, lean_object* v___y_177_){
_start:
{
lean_object* v___x_178_; 
v___x_178_ = lean_apply_1(v_f_176_, v___y_177_);
return v___x_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_comp___redArg___lam__1(lean_object* v_g_179_, lean_object* v___y_180_){
_start:
{
lean_object* v___x_181_; 
v___x_181_ = lean_apply_1(v_g_179_, v___y_180_);
return v___x_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_comp___redArg(lean_object* v_f_182_, lean_object* v_g_183_){
_start:
{
lean_object* v___f_184_; lean_object* v___f_185_; lean_object* v___x_186_; 
v___f_184_ = lean_alloc_closure((void*)(lp_mathlib_sSupHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_184_, 0, v_f_182_);
v___f_185_ = lean_alloc_closure((void*)(lp_mathlib_sSupHom_comp___redArg___lam__1), 2, 1);
lean_closure_set(v___f_185_, 0, v_g_183_);
v___x_186_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_186_, 0, lean_box(0));
lean_closure_set(v___x_186_, 1, lean_box(0));
lean_closure_set(v___x_186_, 2, lean_box(0));
lean_closure_set(v___x_186_, 3, v___f_184_);
lean_closure_set(v___x_186_, 4, v___f_185_);
return v___x_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_comp(lean_object* v_00_u03b1_187_, lean_object* v_00_u03b2_188_, lean_object* v_00_u03b3_189_, lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_inst_192_, lean_object* v_f_193_, lean_object* v_g_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lp_mathlib_sSupHom_comp___redArg(v_f_193_, v_g_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_comp___boxed(lean_object* v_00_u03b1_196_, lean_object* v_00_u03b2_197_, lean_object* v_00_u03b3_198_, lean_object* v_inst_199_, lean_object* v_inst_200_, lean_object* v_inst_201_, lean_object* v_f_202_, lean_object* v_g_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib_sSupHom_comp(v_00_u03b1_196_, v_00_u03b2_197_, v_00_u03b3_198_, v_inst_199_, v_inst_200_, v_inst_201_, v_f_202_, v_g_203_);
lean_dec(v_inst_201_);
lean_dec(v_inst_200_);
lean_dec(v_inst_199_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_comp___redArg(lean_object* v_f_205_, lean_object* v_g_206_){
_start:
{
lean_object* v___f_207_; lean_object* v___f_208_; lean_object* v___x_209_; 
v___f_207_ = lean_alloc_closure((void*)(lp_mathlib_sSupHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_207_, 0, v_f_205_);
v___f_208_ = lean_alloc_closure((void*)(lp_mathlib_sSupHom_comp___redArg___lam__1), 2, 1);
lean_closure_set(v___f_208_, 0, v_g_206_);
v___x_209_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_209_, 0, lean_box(0));
lean_closure_set(v___x_209_, 1, lean_box(0));
lean_closure_set(v___x_209_, 2, lean_box(0));
lean_closure_set(v___x_209_, 3, v___f_207_);
lean_closure_set(v___x_209_, 4, v___f_208_);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_comp(lean_object* v_00_u03b1_210_, lean_object* v_00_u03b2_211_, lean_object* v_00_u03b3_212_, lean_object* v_inst_213_, lean_object* v_inst_214_, lean_object* v_inst_215_, lean_object* v_f_216_, lean_object* v_g_217_){
_start:
{
lean_object* v___x_218_; 
v___x_218_ = lp_mathlib_sInfHom_comp___redArg(v_f_216_, v_g_217_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_comp___boxed(lean_object* v_00_u03b1_219_, lean_object* v_00_u03b2_220_, lean_object* v_00_u03b3_221_, lean_object* v_inst_222_, lean_object* v_inst_223_, lean_object* v_inst_224_, lean_object* v_f_225_, lean_object* v_g_226_){
_start:
{
lean_object* v_res_227_; 
v_res_227_ = lp_mathlib_sInfHom_comp(v_00_u03b1_219_, v_00_u03b2_220_, v_00_u03b3_221_, v_inst_222_, v_inst_223_, v_inst_224_, v_f_225_, v_g_226_);
lean_dec(v_inst_224_);
lean_dec(v_inst_223_);
lean_dec(v_inst_222_);
return v_res_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instPartialOrder(lean_object* v_00_u03b1_231_, lean_object* v_00_u03b2_232_, lean_object* v_inst_233_, lean_object* v_x_234_){
_start:
{
lean_object* v___x_235_; 
v___x_235_ = ((lean_object*)(lp_mathlib_sSupHom_instPartialOrder___closed__0));
return v___x_235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instPartialOrder___boxed(lean_object* v_00_u03b1_236_, lean_object* v_00_u03b2_237_, lean_object* v_inst_238_, lean_object* v_x_239_){
_start:
{
lean_object* v_res_240_; 
v_res_240_ = lp_mathlib_sSupHom_instPartialOrder(v_00_u03b1_236_, v_00_u03b2_237_, v_inst_238_, v_x_239_);
lean_dec_ref(v_x_239_);
lean_dec(v_inst_238_);
return v_res_240_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instPartialOrder(lean_object* v_00_u03b1_241_, lean_object* v_00_u03b2_242_, lean_object* v_inst_243_, lean_object* v_x_244_){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = ((lean_object*)(lp_mathlib_sSupHom_instPartialOrder___closed__0));
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instPartialOrder___boxed(lean_object* v_00_u03b1_246_, lean_object* v_00_u03b2_247_, lean_object* v_inst_248_, lean_object* v_x_249_){
_start:
{
lean_object* v_res_250_; 
v_res_250_ = lp_mathlib_sInfHom_instPartialOrder(v_00_u03b1_246_, v_00_u03b2_247_, v_inst_248_, v_x_249_);
lean_dec_ref(v_x_249_);
lean_dec(v_inst_248_);
return v_res_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instBot___redArg___lam__0(lean_object* v_toOrderBot_251_, lean_object* v_x_252_){
_start:
{
lean_inc(v_toOrderBot_251_);
return v_toOrderBot_251_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instBot___redArg___lam__0___boxed(lean_object* v_toOrderBot_253_, lean_object* v_x_254_){
_start:
{
lean_object* v_res_255_; 
v_res_255_ = lp_mathlib_sSupHom_instBot___redArg___lam__0(v_toOrderBot_253_, v_x_254_);
lean_dec(v_x_254_);
lean_dec(v_toOrderBot_253_);
return v_res_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instBot___redArg(lean_object* v_x_256_){
_start:
{
lean_object* v_toBoundedOrder_257_; lean_object* v_toOrderBot_258_; lean_object* v___f_259_; 
v_toBoundedOrder_257_ = lean_ctor_get(v_x_256_, 3);
lean_inc_ref(v_toBoundedOrder_257_);
lean_dec_ref(v_x_256_);
v_toOrderBot_258_ = lean_ctor_get(v_toBoundedOrder_257_, 1);
lean_inc(v_toOrderBot_258_);
lean_dec_ref(v_toBoundedOrder_257_);
v___f_259_ = lean_alloc_closure((void*)(lp_mathlib_sSupHom_instBot___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_259_, 0, v_toOrderBot_258_);
return v___f_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instBot(lean_object* v_00_u03b1_260_, lean_object* v_00_u03b2_261_, lean_object* v_inst_262_, lean_object* v_x_263_){
_start:
{
lean_object* v___x_264_; 
v___x_264_ = lp_mathlib_sSupHom_instBot___redArg(v_x_263_);
return v___x_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instBot___boxed(lean_object* v_00_u03b1_265_, lean_object* v_00_u03b2_266_, lean_object* v_inst_267_, lean_object* v_x_268_){
_start:
{
lean_object* v_res_269_; 
v_res_269_ = lp_mathlib_sSupHom_instBot(v_00_u03b1_265_, v_00_u03b2_266_, v_inst_267_, v_x_268_);
lean_dec(v_inst_267_);
return v_res_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instTop___redArg___lam__0(lean_object* v_toOrderTop_270_, lean_object* v_x_271_){
_start:
{
lean_inc(v_toOrderTop_270_);
return v_toOrderTop_270_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instTop___redArg___lam__0___boxed(lean_object* v_toOrderTop_272_, lean_object* v_x_273_){
_start:
{
lean_object* v_res_274_; 
v_res_274_ = lp_mathlib_sInfHom_instTop___redArg___lam__0(v_toOrderTop_272_, v_x_273_);
lean_dec(v_x_273_);
lean_dec(v_toOrderTop_272_);
return v_res_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instTop___redArg(lean_object* v_x_275_){
_start:
{
lean_object* v_toBoundedOrder_276_; lean_object* v_toOrderTop_277_; lean_object* v___f_278_; 
v_toBoundedOrder_276_ = lean_ctor_get(v_x_275_, 3);
lean_inc_ref(v_toBoundedOrder_276_);
lean_dec_ref(v_x_275_);
v_toOrderTop_277_ = lean_ctor_get(v_toBoundedOrder_276_, 0);
lean_inc(v_toOrderTop_277_);
lean_dec_ref(v_toBoundedOrder_276_);
v___f_278_ = lean_alloc_closure((void*)(lp_mathlib_sInfHom_instTop___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_278_, 0, v_toOrderTop_277_);
return v___f_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instTop(lean_object* v_00_u03b1_279_, lean_object* v_00_u03b2_280_, lean_object* v_inst_281_, lean_object* v_x_282_){
_start:
{
lean_object* v___x_283_; 
v___x_283_ = lp_mathlib_sInfHom_instTop___redArg(v_x_282_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instTop___boxed(lean_object* v_00_u03b1_284_, lean_object* v_00_u03b2_285_, lean_object* v_inst_286_, lean_object* v_x_287_){
_start:
{
lean_object* v_res_288_; 
v_res_288_ = lp_mathlib_sInfHom_instTop(v_00_u03b1_284_, v_00_u03b2_285_, v_inst_286_, v_x_287_);
lean_dec(v_inst_286_);
return v_res_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instOrderBot___redArg(lean_object* v_x_289_){
_start:
{
lean_object* v___x_290_; 
v___x_290_ = lp_mathlib_sSupHom_instBot___redArg(v_x_289_);
return v___x_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instOrderBot(lean_object* v_00_u03b1_291_, lean_object* v_00_u03b2_292_, lean_object* v_inst_293_, lean_object* v_x_294_){
_start:
{
lean_object* v___x_295_; 
v___x_295_ = lp_mathlib_sSupHom_instBot___redArg(v_x_294_);
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_instOrderBot___boxed(lean_object* v_00_u03b1_296_, lean_object* v_00_u03b2_297_, lean_object* v_inst_298_, lean_object* v_x_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib_sSupHom_instOrderBot(v_00_u03b1_296_, v_00_u03b2_297_, v_inst_298_, v_x_299_);
lean_dec(v_inst_298_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instOrderTop___redArg(lean_object* v_x_301_){
_start:
{
lean_object* v___x_302_; 
v___x_302_ = lp_mathlib_sInfHom_instTop___redArg(v_x_301_);
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instOrderTop(lean_object* v_00_u03b1_303_, lean_object* v_00_u03b2_304_, lean_object* v_inst_305_, lean_object* v_x_306_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lp_mathlib_sInfHom_instTop___redArg(v_x_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_instOrderTop___boxed(lean_object* v_00_u03b1_308_, lean_object* v_00_u03b2_309_, lean_object* v_inst_310_, lean_object* v_x_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_mathlib_sInfHom_instOrderTop(v_00_u03b1_308_, v_00_u03b2_309_, v_inst_310_, v_x_311_);
lean_dec(v_inst_310_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_toLatticeHom___redArg(lean_object* v_f_313_){
_start:
{
lean_object* v___f_314_; 
v___f_314_ = lean_alloc_closure((void*)(lp_mathlib_sSupHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_314_, 0, v_f_313_);
return v___f_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_toLatticeHom(lean_object* v_00_u03b1_315_, lean_object* v_00_u03b2_316_, lean_object* v_inst_317_, lean_object* v_inst_318_, lean_object* v_f_319_){
_start:
{
lean_object* v___f_320_; 
v___f_320_ = lean_alloc_closure((void*)(lp_mathlib_sSupHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_320_, 0, v_f_319_);
return v___f_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_toLatticeHom___boxed(lean_object* v_00_u03b1_321_, lean_object* v_00_u03b2_322_, lean_object* v_inst_323_, lean_object* v_inst_324_, lean_object* v_f_325_){
_start:
{
lean_object* v_res_326_; 
v_res_326_ = lp_mathlib_FrameHom_toLatticeHom(v_00_u03b1_321_, v_00_u03b2_322_, v_inst_323_, v_inst_324_, v_f_325_);
lean_dec_ref(v_inst_324_);
lean_dec_ref(v_inst_323_);
return v_res_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_copy___redArg(lean_object* v_f_x27_327_){
_start:
{
lean_inc(v_f_x27_327_);
return v_f_x27_327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_copy___redArg___boxed(lean_object* v_f_x27_328_){
_start:
{
lean_object* v_res_329_; 
v_res_329_ = lp_mathlib_FrameHom_copy___redArg(v_f_x27_328_);
lean_dec(v_f_x27_328_);
return v_res_329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_copy(lean_object* v_00_u03b1_330_, lean_object* v_00_u03b2_331_, lean_object* v_inst_332_, lean_object* v_inst_333_, lean_object* v_f_334_, lean_object* v_f_x27_335_, lean_object* v_h_336_){
_start:
{
lean_inc(v_f_x27_335_);
return v_f_x27_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_copy___boxed(lean_object* v_00_u03b1_337_, lean_object* v_00_u03b2_338_, lean_object* v_inst_339_, lean_object* v_inst_340_, lean_object* v_f_341_, lean_object* v_f_x27_342_, lean_object* v_h_343_){
_start:
{
lean_object* v_res_344_; 
v_res_344_ = lp_mathlib_FrameHom_copy(v_00_u03b1_337_, v_00_u03b2_338_, v_inst_339_, v_inst_340_, v_f_341_, v_f_x27_342_, v_h_343_);
lean_dec(v_f_x27_342_);
lean_dec(v_f_341_);
lean_dec_ref(v_inst_340_);
lean_dec_ref(v_inst_339_);
return v_res_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_id(lean_object* v_00_u03b1_345_, lean_object* v_inst_346_){
_start:
{
lean_object* v___x_347_; 
v___x_347_ = ((lean_object*)(lp_mathlib_sSupHom_id___closed__0));
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_id___boxed(lean_object* v_00_u03b1_348_, lean_object* v_inst_349_){
_start:
{
lean_object* v_res_350_; 
v_res_350_ = lp_mathlib_FrameHom_id(v_00_u03b1_348_, v_inst_349_);
lean_dec_ref(v_inst_349_);
return v_res_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_instInhabited(lean_object* v_00_u03b1_351_, lean_object* v_inst_352_){
_start:
{
lean_object* v___x_353_; 
v___x_353_ = ((lean_object*)(lp_mathlib_sSupHom_id___closed__0));
return v___x_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_instInhabited___boxed(lean_object* v_00_u03b1_354_, lean_object* v_inst_355_){
_start:
{
lean_object* v_res_356_; 
v_res_356_ = lp_mathlib_FrameHom_instInhabited(v_00_u03b1_354_, v_inst_355_);
lean_dec_ref(v_inst_355_);
return v_res_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_comp___redArg(lean_object* v_f_357_, lean_object* v_g_358_){
_start:
{
lean_object* v___x_359_; 
v___x_359_ = lp_mathlib_InfHom_comp___redArg(v_f_357_, v_g_358_);
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_comp(lean_object* v_00_u03b1_360_, lean_object* v_00_u03b2_361_, lean_object* v_00_u03b3_362_, lean_object* v_inst_363_, lean_object* v_inst_364_, lean_object* v_inst_365_, lean_object* v_f_366_, lean_object* v_g_367_){
_start:
{
lean_object* v___x_368_; 
v___x_368_ = lp_mathlib_InfHom_comp___redArg(v_f_366_, v_g_367_);
return v___x_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_comp___boxed(lean_object* v_00_u03b1_369_, lean_object* v_00_u03b2_370_, lean_object* v_00_u03b3_371_, lean_object* v_inst_372_, lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v_f_375_, lean_object* v_g_376_){
_start:
{
lean_object* v_res_377_; 
v_res_377_ = lp_mathlib_FrameHom_comp(v_00_u03b1_369_, v_00_u03b2_370_, v_00_u03b3_371_, v_inst_372_, v_inst_373_, v_inst_374_, v_f_375_, v_g_376_);
lean_dec_ref(v_inst_374_);
lean_dec_ref(v_inst_373_);
lean_dec_ref(v_inst_372_);
return v_res_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_instPartialOrder(lean_object* v_00_u03b1_378_, lean_object* v_00_u03b2_379_, lean_object* v_inst_380_, lean_object* v_inst_381_){
_start:
{
lean_object* v___x_382_; 
v___x_382_ = ((lean_object*)(lp_mathlib_sSupHom_instPartialOrder___closed__0));
return v___x_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FrameHom_instPartialOrder___boxed(lean_object* v_00_u03b1_383_, lean_object* v_00_u03b2_384_, lean_object* v_inst_385_, lean_object* v_inst_386_){
_start:
{
lean_object* v_res_387_; 
v_res_387_ = lp_mathlib_FrameHom_instPartialOrder(v_00_u03b1_383_, v_00_u03b2_384_, v_inst_385_, v_inst_386_);
lean_dec_ref(v_inst_386_);
lean_dec_ref(v_inst_385_);
return v_res_387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_OrderIso_toCompleteLatticeHom___redArg___lam__0(lean_object* v_f_388_, lean_object* v___y_389_){
_start:
{
lean_object* v___x_390_; 
v___x_390_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_f_388_, v___y_389_);
return v___x_390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_OrderIso_toCompleteLatticeHom___redArg(lean_object* v_f_391_){
_start:
{
lean_object* v___f_392_; 
v___f_392_ = lean_alloc_closure((void*)(lp_mathlib_CompleteLatticeHom_OrderIso_toCompleteLatticeHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_392_, 0, v_f_391_);
return v___f_392_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_OrderIso_toCompleteLatticeHom(lean_object* v_00_u03b1_393_, lean_object* v_00_u03b2_394_, lean_object* v_inst_395_, lean_object* v_inst_396_, lean_object* v_f_397_){
_start:
{
lean_object* v___f_398_; 
v___f_398_ = lean_alloc_closure((void*)(lp_mathlib_CompleteLatticeHom_OrderIso_toCompleteLatticeHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_398_, 0, v_f_397_);
return v___f_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_OrderIso_toCompleteLatticeHom___boxed(lean_object* v_00_u03b1_399_, lean_object* v_00_u03b2_400_, lean_object* v_inst_401_, lean_object* v_inst_402_, lean_object* v_f_403_){
_start:
{
lean_object* v_res_404_; 
v_res_404_ = lp_mathlib_CompleteLatticeHom_OrderIso_toCompleteLatticeHom(v_00_u03b1_399_, v_00_u03b2_400_, v_inst_401_, v_inst_402_, v_f_403_);
lean_dec_ref(v_inst_402_);
lean_dec_ref(v_inst_401_);
return v_res_404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_toBoundedLatticeHom___redArg(lean_object* v_f_405_){
_start:
{
lean_object* v___f_406_; 
v___f_406_ = lean_alloc_closure((void*)(lp_mathlib_sSupHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_406_, 0, v_f_405_);
return v___f_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_toBoundedLatticeHom(lean_object* v_00_u03b1_407_, lean_object* v_00_u03b2_408_, lean_object* v_inst_409_, lean_object* v_inst_410_, lean_object* v_f_411_){
_start:
{
lean_object* v___f_412_; 
v___f_412_ = lean_alloc_closure((void*)(lp_mathlib_sSupHom_comp___redArg___lam__0), 2, 1);
lean_closure_set(v___f_412_, 0, v_f_411_);
return v___f_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_toBoundedLatticeHom___boxed(lean_object* v_00_u03b1_413_, lean_object* v_00_u03b2_414_, lean_object* v_inst_415_, lean_object* v_inst_416_, lean_object* v_f_417_){
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_mathlib_CompleteLatticeHom_toBoundedLatticeHom(v_00_u03b1_413_, v_00_u03b2_414_, v_inst_415_, v_inst_416_, v_f_417_);
lean_dec_ref(v_inst_416_);
lean_dec_ref(v_inst_415_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_copy___redArg(lean_object* v_f_x27_419_){
_start:
{
lean_inc(v_f_x27_419_);
return v_f_x27_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_copy___redArg___boxed(lean_object* v_f_x27_420_){
_start:
{
lean_object* v_res_421_; 
v_res_421_ = lp_mathlib_CompleteLatticeHom_copy___redArg(v_f_x27_420_);
lean_dec(v_f_x27_420_);
return v_res_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_copy(lean_object* v_00_u03b1_422_, lean_object* v_00_u03b2_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_f_426_, lean_object* v_f_x27_427_, lean_object* v_h_428_){
_start:
{
lean_inc(v_f_x27_427_);
return v_f_x27_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_copy___boxed(lean_object* v_00_u03b1_429_, lean_object* v_00_u03b2_430_, lean_object* v_inst_431_, lean_object* v_inst_432_, lean_object* v_f_433_, lean_object* v_f_x27_434_, lean_object* v_h_435_){
_start:
{
lean_object* v_res_436_; 
v_res_436_ = lp_mathlib_CompleteLatticeHom_copy(v_00_u03b1_429_, v_00_u03b2_430_, v_inst_431_, v_inst_432_, v_f_433_, v_f_x27_434_, v_h_435_);
lean_dec(v_f_x27_434_);
lean_dec(v_f_433_);
lean_dec_ref(v_inst_432_);
lean_dec_ref(v_inst_431_);
return v_res_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_id(lean_object* v_00_u03b1_437_, lean_object* v_inst_438_){
_start:
{
lean_object* v___x_439_; 
v___x_439_ = ((lean_object*)(lp_mathlib_sSupHom_id___closed__0));
return v___x_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_id___boxed(lean_object* v_00_u03b1_440_, lean_object* v_inst_441_){
_start:
{
lean_object* v_res_442_; 
v_res_442_ = lp_mathlib_CompleteLatticeHom_id(v_00_u03b1_440_, v_inst_441_);
lean_dec_ref(v_inst_441_);
return v_res_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_instInhabited(lean_object* v_00_u03b1_443_, lean_object* v_inst_444_){
_start:
{
lean_object* v___x_445_; 
v___x_445_ = ((lean_object*)(lp_mathlib_sSupHom_id___closed__0));
return v___x_445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_instInhabited___boxed(lean_object* v_00_u03b1_446_, lean_object* v_inst_447_){
_start:
{
lean_object* v_res_448_; 
v_res_448_ = lp_mathlib_CompleteLatticeHom_instInhabited(v_00_u03b1_446_, v_inst_447_);
lean_dec_ref(v_inst_447_);
return v_res_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_comp___redArg(lean_object* v_f_449_, lean_object* v_g_450_){
_start:
{
lean_object* v___x_451_; 
v___x_451_ = lp_mathlib_sInfHom_comp___redArg(v_f_449_, v_g_450_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_comp(lean_object* v_00_u03b1_452_, lean_object* v_00_u03b2_453_, lean_object* v_00_u03b3_454_, lean_object* v_inst_455_, lean_object* v_inst_456_, lean_object* v_inst_457_, lean_object* v_f_458_, lean_object* v_g_459_){
_start:
{
lean_object* v___x_460_; 
v___x_460_ = lp_mathlib_sInfHom_comp___redArg(v_f_458_, v_g_459_);
return v___x_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_comp___boxed(lean_object* v_00_u03b1_461_, lean_object* v_00_u03b2_462_, lean_object* v_00_u03b3_463_, lean_object* v_inst_464_, lean_object* v_inst_465_, lean_object* v_inst_466_, lean_object* v_f_467_, lean_object* v_g_468_){
_start:
{
lean_object* v_res_469_; 
v_res_469_ = lp_mathlib_CompleteLatticeHom_comp(v_00_u03b1_461_, v_00_u03b2_462_, v_00_u03b3_463_, v_inst_464_, v_inst_465_, v_inst_466_, v_f_467_, v_g_468_);
lean_dec_ref(v_inst_466_);
lean_dec_ref(v_inst_465_);
lean_dec_ref(v_inst_464_);
return v_res_469_;
}
}
static lean_object* _init_lp_mathlib_sSupHom_dual___lam__0___closed__0(void){
_start:
{
lean_object* v___x_470_; 
v___x_470_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_dual___lam__0(lean_object* v_f_471_, lean_object* v___y_472_){
_start:
{
lean_object* v___x_473_; lean_object* v_toFun_474_; lean_object* v_toFun_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; 
v___x_473_ = lean_obj_once(&lp_mathlib_sSupHom_dual___lam__0___closed__0, &lp_mathlib_sSupHom_dual___lam__0___closed__0_once, _init_lp_mathlib_sSupHom_dual___lam__0___closed__0);
v_toFun_474_ = lean_ctor_get(v___x_473_, 0);
v_toFun_475_ = lean_ctor_get(v___x_473_, 0);
lean_inc(v_toFun_474_);
v___x_476_ = lean_apply_1(v_toFun_474_, v___y_472_);
v___x_477_ = lean_apply_1(v_f_471_, v___x_476_);
lean_inc(v_toFun_475_);
v___x_478_ = lean_apply_1(v_toFun_475_, v___x_477_);
return v___x_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_dual(lean_object* v_00_u03b1_482_, lean_object* v_00_u03b2_483_, lean_object* v_inst_484_, lean_object* v_inst_485_){
_start:
{
lean_object* v___x_486_; 
v___x_486_ = ((lean_object*)(lp_mathlib_sSupHom_dual___closed__1));
return v___x_486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_dual___boxed(lean_object* v_00_u03b1_487_, lean_object* v_00_u03b2_488_, lean_object* v_inst_489_, lean_object* v_inst_490_){
_start:
{
lean_object* v_res_491_; 
v_res_491_ = lp_mathlib_sSupHom_dual(v_00_u03b1_487_, v_00_u03b2_488_, v_inst_489_, v_inst_490_);
lean_dec(v_inst_490_);
lean_dec(v_inst_489_);
return v_res_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_dual(lean_object* v_00_u03b1_492_, lean_object* v_00_u03b2_493_, lean_object* v_inst_494_, lean_object* v_inst_495_){
_start:
{
lean_object* v___x_496_; 
v___x_496_ = ((lean_object*)(lp_mathlib_sSupHom_dual___closed__1));
return v___x_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sInfHom_dual___boxed(lean_object* v_00_u03b1_497_, lean_object* v_00_u03b2_498_, lean_object* v_inst_499_, lean_object* v_inst_500_){
_start:
{
lean_object* v_res_501_; 
v_res_501_ = lp_mathlib_sInfHom_dual(v_00_u03b1_497_, v_00_u03b2_498_, v_inst_499_, v_inst_500_);
lean_dec(v_inst_500_);
lean_dec(v_inst_499_);
return v_res_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_dual___redArg___lam__0(lean_object* v_toSupSet_502_, lean_object* v_toSupSet_503_, lean_object* v_f_504_, lean_object* v___y_505_){
_start:
{
lean_object* v___x_506_; lean_object* v_toFun_507_; lean_object* v___x_508_; 
v___x_506_ = lp_mathlib_sSupHom_dual(lean_box(0), lean_box(0), v_toSupSet_502_, v_toSupSet_503_);
v_toFun_507_ = lean_ctor_get(v___x_506_, 0);
lean_inc(v_toFun_507_);
lean_dec_ref(v___x_506_);
v___x_508_ = lean_apply_2(v_toFun_507_, v_f_504_, v___y_505_);
return v___x_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_dual___redArg___lam__0___boxed(lean_object* v_toSupSet_509_, lean_object* v_toSupSet_510_, lean_object* v_f_511_, lean_object* v___y_512_){
_start:
{
lean_object* v_res_513_; 
v_res_513_ = lp_mathlib_CompleteLatticeHom_dual___redArg___lam__0(v_toSupSet_509_, v_toSupSet_510_, v_f_511_, v___y_512_);
lean_dec(v_toSupSet_510_);
lean_dec(v_toSupSet_509_);
return v_res_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_dual___redArg(lean_object* v_inst_514_, lean_object* v_inst_515_){
_start:
{
lean_object* v___x_516_; lean_object* v___x_517_; lean_object* v___x_518_; lean_object* v_toSupSet_519_; lean_object* v___x_520_; lean_object* v_toSupSet_521_; lean_object* v___x_522_; lean_object* v_toSupSet_523_; lean_object* v___x_524_; lean_object* v_toSupSet_525_; lean_object* v___x_527_; uint8_t v_isShared_528_; uint8_t v_isSharedCheck_534_; 
lean_inc_ref(v_inst_514_);
v___x_516_ = lp_mathlib_OrderDual_instCompleteLattice___redArg(v_inst_514_);
lean_inc_ref(v_inst_515_);
v___x_517_ = lp_mathlib_OrderDual_instCompleteLattice___redArg(v_inst_515_);
v___x_518_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_514_);
v_toSupSet_519_ = lean_ctor_get(v___x_518_, 1);
lean_inc(v_toSupSet_519_);
lean_dec_ref(v___x_518_);
v___x_520_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v_inst_515_);
v_toSupSet_521_ = lean_ctor_get(v___x_520_, 1);
lean_inc(v_toSupSet_521_);
lean_dec_ref(v___x_520_);
v___x_522_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v___x_516_);
v_toSupSet_523_ = lean_ctor_get(v___x_522_, 1);
lean_inc(v_toSupSet_523_);
lean_dec_ref(v___x_522_);
v___x_524_ = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(v___x_517_);
v_toSupSet_525_ = lean_ctor_get(v___x_524_, 1);
v_isSharedCheck_534_ = !lean_is_exclusive(v___x_524_);
if (v_isSharedCheck_534_ == 0)
{
lean_object* v_unused_535_; 
v_unused_535_ = lean_ctor_get(v___x_524_, 0);
lean_dec(v_unused_535_);
v___x_527_ = v___x_524_;
v_isShared_528_ = v_isSharedCheck_534_;
goto v_resetjp_526_;
}
else
{
lean_inc(v_toSupSet_525_);
lean_dec(v___x_524_);
v___x_527_ = lean_box(0);
v_isShared_528_ = v_isSharedCheck_534_;
goto v_resetjp_526_;
}
v_resetjp_526_:
{
lean_object* v___f_529_; lean_object* v___f_530_; lean_object* v___x_532_; 
v___f_529_ = lean_alloc_closure((void*)(lp_mathlib_CompleteLatticeHom_dual___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_529_, 0, v_toSupSet_519_);
lean_closure_set(v___f_529_, 1, v_toSupSet_521_);
v___f_530_ = lean_alloc_closure((void*)(lp_mathlib_CompleteLatticeHom_dual___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_530_, 0, v_toSupSet_523_);
lean_closure_set(v___f_530_, 1, v_toSupSet_525_);
if (v_isShared_528_ == 0)
{
lean_ctor_set(v___x_527_, 1, v___f_530_);
lean_ctor_set(v___x_527_, 0, v___f_529_);
v___x_532_ = v___x_527_;
goto v_reusejp_531_;
}
else
{
lean_object* v_reuseFailAlloc_533_; 
v_reuseFailAlloc_533_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_533_, 0, v___f_529_);
lean_ctor_set(v_reuseFailAlloc_533_, 1, v___f_530_);
v___x_532_ = v_reuseFailAlloc_533_;
goto v_reusejp_531_;
}
v_reusejp_531_:
{
return v___x_532_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_dual(lean_object* v_00_u03b1_536_, lean_object* v_00_u03b2_537_, lean_object* v_inst_538_, lean_object* v_inst_539_){
_start:
{
lean_object* v___x_540_; 
v___x_540_ = lp_mathlib_CompleteLatticeHom_dual___redArg(v_inst_538_, v_inst_539_);
return v___x_540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_setPreimage(lean_object* v_00_u03b1_541_, lean_object* v_00_u03b2_542_, lean_object* v_f_543_){
_start:
{
lean_object* v___x_544_; 
v___x_544_ = lean_box(0);
return v___x_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLatticeHom_setPreimage___boxed(lean_object* v_00_u03b1_545_, lean_object* v_00_u03b2_546_, lean_object* v_f_547_){
_start:
{
lean_object* v_res_548_; 
v_res_548_ = lp_mathlib_CompleteLatticeHom_setPreimage(v_00_u03b1_545_, v_00_u03b2_546_, v_f_547_);
lean_dec(v_f_547_);
return v_res_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_setImage(lean_object* v_00_u03b1_549_, lean_object* v_00_u03b2_550_, lean_object* v_f_551_){
_start:
{
lean_object* v___x_552_; 
v___x_552_ = lean_box(0);
return v___x_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sSupHom_setImage___boxed(lean_object* v_00_u03b1_553_, lean_object* v_00_u03b2_554_, lean_object* v_f_555_){
_start:
{
lean_object* v_res_556_; 
v_res_556_ = lp_mathlib_sSupHom_setImage(v_00_u03b1_553_, v_00_u03b2_554_, v_f_555_);
lean_dec(v_f_555_);
return v_res_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toOrderIsoSet(lean_object* v_00_u03b1_557_, lean_object* v_00_u03b2_558_, lean_object* v_e_559_){
_start:
{
lean_object* v___x_560_; 
v___x_560_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_560_, 0, lean_box(0));
lean_ctor_set(v___x_560_, 1, lean_box(0));
return v___x_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_toOrderIsoSet___boxed(lean_object* v_00_u03b1_561_, lean_object* v_00_u03b2_562_, lean_object* v_e_563_){
_start:
{
lean_object* v_res_564_; 
v_res_564_ = lp_mathlib_Equiv_toOrderIsoSet(v_00_u03b1_561_, v_00_u03b2_562_, v_e_563_);
lean_dec_ref(v_e_563_);
return v_res_564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_supsSupHom___redArg___lam__0(lean_object* v_toSemilatticeSup_565_, lean_object* v_x_566_){
_start:
{
lean_object* v_fst_567_; lean_object* v_snd_568_; lean_object* v_sup_569_; lean_object* v___x_570_; 
v_fst_567_ = lean_ctor_get(v_x_566_, 0);
lean_inc(v_fst_567_);
v_snd_568_ = lean_ctor_get(v_x_566_, 1);
lean_inc(v_snd_568_);
lean_dec_ref(v_x_566_);
v_sup_569_ = lean_ctor_get(v_toSemilatticeSup_565_, 1);
lean_inc(v_sup_569_);
lean_dec_ref(v_toSemilatticeSup_565_);
v___x_570_ = lean_apply_2(v_sup_569_, v_fst_567_, v_snd_568_);
return v___x_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_supsSupHom___redArg(lean_object* v_inst_571_){
_start:
{
lean_object* v_toLattice_572_; lean_object* v_toSemilatticeSup_573_; lean_object* v___f_574_; 
v_toLattice_572_ = lean_ctor_get(v_inst_571_, 0);
lean_inc_ref(v_toLattice_572_);
lean_dec_ref(v_inst_571_);
v_toSemilatticeSup_573_ = lean_ctor_get(v_toLattice_572_, 0);
lean_inc_ref(v_toSemilatticeSup_573_);
lean_dec_ref(v_toLattice_572_);
v___f_574_ = lean_alloc_closure((void*)(lp_mathlib_supsSupHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_574_, 0, v_toSemilatticeSup_573_);
return v___f_574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_supsSupHom(lean_object* v_00_u03b1_575_, lean_object* v_inst_576_){
_start:
{
lean_object* v___x_577_; 
v___x_577_ = lp_mathlib_supsSupHom___redArg(v_inst_576_);
return v___x_577_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_infsInfHom___redArg___lam__0(lean_object* v_toLattice_578_, lean_object* v_x_579_){
_start:
{
lean_object* v_fst_580_; lean_object* v_snd_581_; lean_object* v_inf_582_; lean_object* v___x_583_; 
v_fst_580_ = lean_ctor_get(v_x_579_, 0);
lean_inc(v_fst_580_);
v_snd_581_ = lean_ctor_get(v_x_579_, 1);
lean_inc(v_snd_581_);
lean_dec_ref(v_x_579_);
v_inf_582_ = lean_ctor_get(v_toLattice_578_, 1);
lean_inc(v_inf_582_);
lean_dec_ref(v_toLattice_578_);
v___x_583_ = lean_apply_2(v_inf_582_, v_fst_580_, v_snd_581_);
return v___x_583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_infsInfHom___redArg(lean_object* v_inst_584_){
_start:
{
lean_object* v_toLattice_585_; lean_object* v___f_586_; 
v_toLattice_585_ = lean_ctor_get(v_inst_584_, 0);
lean_inc_ref(v_toLattice_585_);
lean_dec_ref(v_inst_584_);
v___f_586_ = lean_alloc_closure((void*)(lp_mathlib_infsInfHom___redArg___lam__0), 2, 1);
lean_closure_set(v___f_586_, 0, v_toLattice_585_);
return v___f_586_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_infsInfHom(lean_object* v_00_u03b1_587_, lean_object* v_inst_588_){
_start:
{
lean_object* v___x_589_; 
v___x_589_ = lp_mathlib_infsInfHom___redArg(v_inst_588_);
return v___x_589_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Image(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_CompleteLattice(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Lattice_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Hom_CompleteLattice(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Set_Lattice_Image(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Hom_CompleteLattice(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Lattice_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_CompleteLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Hom_CompleteLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Hom_CompleteLattice(builtin);
}
#ifdef __cplusplus
}
#endif
