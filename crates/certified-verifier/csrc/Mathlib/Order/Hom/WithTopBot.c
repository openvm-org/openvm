// Lean compiler output
// Module: Mathlib.Order.Hom.WithTopBot
// Imports: public import Init public meta import Init public import Mathlib.Order.Hom.BoundedLattice public import Mathlib.Order.WithBot public import Mathlib.Tactic.ApplyFun
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
lean_object* lp_mathlib_WithTop_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_untopD___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_map(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_WithTop_map(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_unbotD___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_recBotCoe___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_recTopCoe___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithBot_some(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_withTopCongr___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_WithTop_some(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_withBotCongr___redArg(lean_object*);
static lean_once_cell_t lp_mathlib_WithTop_toDualBotEquiv___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithTop_toDualBotEquiv___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_WithTop_toDualBotEquiv___lam__0(lean_object*);
static lean_once_cell_t lp_mathlib_WithTop_toDualBotEquiv___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_WithTop_toDualBotEquiv___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_WithTop_toDualBotEquiv___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_toDualBotEquiv___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_toDualBotEquiv___lam__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_WithTop_toDualBotEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithTop_toDualBotEquiv___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithTop_toDualBotEquiv___closed__0 = (const lean_object*)&lp_mathlib_WithTop_toDualBotEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_WithTop_toDualBotEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithTop_toDualBotEquiv___lam__1, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_WithTop_toDualBotEquiv___closed__0_value)} };
static const lean_object* lp_mathlib_WithTop_toDualBotEquiv___closed__1 = (const lean_object*)&lp_mathlib_WithTop_toDualBotEquiv___closed__1_value;
static const lean_closure_object lp_mathlib_WithTop_toDualBotEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithTop_toDualBotEquiv___lam__3___boxed, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_WithTop_toDualBotEquiv___closed__0_value)} };
static const lean_object* lp_mathlib_WithTop_toDualBotEquiv___closed__2 = (const lean_object*)&lp_mathlib_WithTop_toDualBotEquiv___closed__2_value;
static const lean_ctor_object lp_mathlib_WithTop_toDualBotEquiv___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_WithTop_toDualBotEquiv___closed__1_value),((lean_object*)&lp_mathlib_WithTop_toDualBotEquiv___closed__2_value)}};
static const lean_object* lp_mathlib_WithTop_toDualBotEquiv___closed__3 = (const lean_object*)&lp_mathlib_WithTop_toDualBotEquiv___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_WithTop_toDualBotEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_toDualTopEquiv___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_toDualTopEquiv___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_toDualTopEquiv___lam__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_WithBot_toDualTopEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithBot_toDualTopEquiv___lam__2, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_WithTop_toDualBotEquiv___closed__0_value)} };
static const lean_object* lp_mathlib_WithBot_toDualTopEquiv___closed__0 = (const lean_object*)&lp_mathlib_WithBot_toDualTopEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_WithBot_toDualTopEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithBot_toDualTopEquiv___lam__0___boxed, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_WithTop_toDualBotEquiv___closed__0_value)} };
static const lean_object* lp_mathlib_WithBot_toDualTopEquiv___closed__1 = (const lean_object*)&lp_mathlib_WithBot_toDualTopEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_WithBot_toDualTopEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_WithBot_toDualTopEquiv___closed__1_value),((lean_object*)&lp_mathlib_WithBot_toDualTopEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_WithBot_toDualTopEquiv___closed__2 = (const lean_object*)&lp_mathlib_WithBot_toDualTopEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_WithBot_toDualTopEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_coeWithTop___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_Function_Embedding_coeWithTop___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_coeWithTop___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Function_Embedding_coeWithTop___closed__0 = (const lean_object*)&lp_mathlib_Function_Embedding_coeWithTop___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_coeWithTop(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_coeWithBot(lean_object*);
static const lean_closure_object lp_mathlib_WithTop_coeOrderHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithTop_some, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_WithTop_coeOrderHom___closed__0 = (const lean_object*)&lp_mathlib_WithTop_coeOrderHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithTop_coeOrderHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_coeOrderHom___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_WithBot_coeOrderHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithBot_some, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_WithBot_coeOrderHom___closed__0 = (const lean_object*)&lp_mathlib_WithBot_coeOrderHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithBot_coeOrderHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_coeOrderHom___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_subtypeOrderIso___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_subtypeOrderIso___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_subtypeOrderIso___redArg___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_subtypeOrderIso___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_subtypeOrderIso___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_WithTop_subtypeOrderIso___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithTop_subtypeOrderIso___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithTop_subtypeOrderIso___redArg___closed__0 = (const lean_object*)&lp_mathlib_WithTop_subtypeOrderIso___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithTop_subtypeOrderIso___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_subtypeOrderIso(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_subtypeOrderIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_subtypeOrderIso___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_subtypeOrderIso___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_subtypeOrderIso___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_subtypeOrderIso(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_subtypeOrderIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_withBotMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_withBotMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_withBotMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_withBotMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_withTopMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_withTopMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_withTopMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_withBotMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_withBotMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_withBotMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_withTopMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_withTopMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_withTopMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withTopCongr___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withTopCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withTopCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withTopCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withBotCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withBotCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withBotCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withTop_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withTop_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withTop_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withTop_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withBot_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withBot_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withBot_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withBot_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withBot_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withBot_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withTop_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withTop_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withTop_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBot___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTopWithBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTopWithBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTopWithBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBotWithTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBotWithTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBotWithTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTop_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTop_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTop_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBot_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBot_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBot_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTopWithBot_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTopWithBot_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTopWithBot_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBotWithTop_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBotWithTop_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBotWithTop_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_WithTop_toDualBotEquiv___lam__0___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_toDualBotEquiv___lam__0(lean_object* v_a_2_){
_start:
{
lean_object* v___x_3_; lean_object* v_toFun_4_; lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_3_ = lean_obj_once(&lp_mathlib_WithTop_toDualBotEquiv___lam__0___closed__0, &lp_mathlib_WithTop_toDualBotEquiv___lam__0___closed__0_once, _init_lp_mathlib_WithTop_toDualBotEquiv___lam__0___closed__0);
v_toFun_4_ = lean_ctor_get(v___x_3_, 0);
lean_inc(v_toFun_4_);
v___x_5_ = lean_apply_1(v_toFun_4_, v_a_2_);
v___x_6_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_6_, 0, v___x_5_);
return v___x_6_;
}
}
static lean_object* _init_lp_mathlib_WithTop_toDualBotEquiv___lam__1___closed__0(void){
_start:
{
lean_object* v___x_7_; 
v___x_7_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_toDualBotEquiv___lam__1(lean_object* v___f_8_, lean_object* v_a_9_){
_start:
{
lean_object* v___x_10_; lean_object* v_toFun_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; 
v___x_10_ = lean_obj_once(&lp_mathlib_WithTop_toDualBotEquiv___lam__1___closed__0, &lp_mathlib_WithTop_toDualBotEquiv___lam__1___closed__0_once, _init_lp_mathlib_WithTop_toDualBotEquiv___lam__1___closed__0);
v_toFun_11_ = lean_ctor_get(v___x_10_, 0);
v___x_12_ = lean_box(0);
v___x_13_ = lp_mathlib_WithTop_recTopCoe___redArg(v___x_12_, v___f_8_, v_a_9_);
lean_inc(v_toFun_11_);
v___x_14_ = lean_apply_1(v_toFun_11_, v___x_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_toDualBotEquiv___lam__3(lean_object* v___x_15_, lean_object* v___f_16_, lean_object* v_a_17_){
_start:
{
lean_object* v___x_18_; lean_object* v_toFun_19_; lean_object* v___x_20_; lean_object* v___x_21_; 
v___x_18_ = lean_obj_once(&lp_mathlib_WithTop_toDualBotEquiv___lam__1___closed__0, &lp_mathlib_WithTop_toDualBotEquiv___lam__1___closed__0_once, _init_lp_mathlib_WithTop_toDualBotEquiv___lam__1___closed__0);
v_toFun_19_ = lean_ctor_get(v___x_18_, 0);
lean_inc(v_toFun_19_);
v___x_20_ = lean_apply_1(v_toFun_19_, v_a_17_);
v___x_21_ = lp_mathlib_WithBot_recBotCoe___redArg(v___x_15_, v___f_16_, v___x_20_);
return v___x_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_toDualBotEquiv___lam__3___boxed(lean_object* v___x_22_, lean_object* v___f_23_, lean_object* v_a_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_mathlib_WithTop_toDualBotEquiv___lam__3(v___x_22_, v___f_23_, v_a_24_);
lean_dec(v___x_22_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_toDualBotEquiv(lean_object* v_00_u03b1_35_, lean_object* v_inst_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = ((lean_object*)(lp_mathlib_WithTop_toDualBotEquiv___closed__3));
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_toDualTopEquiv___lam__2(lean_object* v___f_38_, lean_object* v_a_39_){
_start:
{
lean_object* v___x_40_; lean_object* v_toFun_41_; lean_object* v___x_42_; lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_40_ = lean_obj_once(&lp_mathlib_WithTop_toDualBotEquiv___lam__1___closed__0, &lp_mathlib_WithTop_toDualBotEquiv___lam__1___closed__0_once, _init_lp_mathlib_WithTop_toDualBotEquiv___lam__1___closed__0);
v_toFun_41_ = lean_ctor_get(v___x_40_, 0);
v___x_42_ = lean_box(0);
lean_inc(v_toFun_41_);
v___x_43_ = lean_apply_1(v_toFun_41_, v_a_39_);
v___x_44_ = lp_mathlib_WithTop_recTopCoe___redArg(v___x_42_, v___f_38_, v___x_43_);
return v___x_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_toDualTopEquiv___lam__0(lean_object* v___x_45_, lean_object* v___f_46_, lean_object* v_a_47_){
_start:
{
lean_object* v___x_48_; lean_object* v_toFun_49_; lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_48_ = lean_obj_once(&lp_mathlib_WithTop_toDualBotEquiv___lam__1___closed__0, &lp_mathlib_WithTop_toDualBotEquiv___lam__1___closed__0_once, _init_lp_mathlib_WithTop_toDualBotEquiv___lam__1___closed__0);
v_toFun_49_ = lean_ctor_get(v___x_48_, 0);
v___x_50_ = lp_mathlib_WithBot_recBotCoe___redArg(v___x_45_, v___f_46_, v_a_47_);
lean_inc(v_toFun_49_);
v___x_51_ = lean_apply_1(v_toFun_49_, v___x_50_);
return v___x_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_toDualTopEquiv___lam__0___boxed(lean_object* v___x_52_, lean_object* v___f_53_, lean_object* v_a_54_){
_start:
{
lean_object* v_res_55_; 
v_res_55_ = lp_mathlib_WithBot_toDualTopEquiv___lam__0(v___x_52_, v___f_53_, v_a_54_);
lean_dec(v___x_52_);
return v_res_55_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_toDualTopEquiv(lean_object* v_00_u03b1_64_, lean_object* v_inst_65_){
_start:
{
lean_object* v___x_66_; 
v___x_66_ = ((lean_object*)(lp_mathlib_WithBot_toDualTopEquiv___closed__2));
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_coeWithTop___lam__0(lean_object* v___y_67_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_68_, 0, v___y_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_coeWithTop(lean_object* v_00_u03b1_70_){
_start:
{
lean_object* v___f_71_; 
v___f_71_ = ((lean_object*)(lp_mathlib_Function_Embedding_coeWithTop___closed__0));
return v___f_71_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Embedding_coeWithBot(lean_object* v_00_u03b1_72_){
_start:
{
lean_object* v___f_73_; 
v___f_73_ = ((lean_object*)(lp_mathlib_Function_Embedding_coeWithTop___closed__0));
return v___f_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_coeOrderHom(lean_object* v_00_u03b1_75_, lean_object* v_inst_76_){
_start:
{
lean_object* v___x_77_; 
v___x_77_ = ((lean_object*)(lp_mathlib_WithTop_coeOrderHom___closed__0));
return v___x_77_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_coeOrderHom___boxed(lean_object* v_00_u03b1_78_, lean_object* v_inst_79_){
_start:
{
lean_object* v_res_80_; 
v_res_80_ = lp_mathlib_WithTop_coeOrderHom(v_00_u03b1_78_, v_inst_79_);
lean_dec_ref(v_inst_79_);
return v_res_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_coeOrderHom(lean_object* v_00_u03b1_82_, lean_object* v_inst_83_){
_start:
{
lean_object* v___x_84_; 
v___x_84_ = ((lean_object*)(lp_mathlib_WithBot_coeOrderHom___closed__0));
return v___x_84_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_coeOrderHom___boxed(lean_object* v_00_u03b1_85_, lean_object* v_inst_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_mathlib_WithBot_coeOrderHom(v_00_u03b1_85_, v_inst_86_);
lean_dec_ref(v_inst_86_);
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_subtypeOrderIso___redArg___lam__0(lean_object* v_inst_88_, lean_object* v_a_89_){
_start:
{
lean_object* v___x_90_; uint8_t v___x_91_; 
lean_inc(v_a_89_);
v___x_90_ = lean_apply_1(v_inst_88_, v_a_89_);
v___x_91_ = lean_unbox(v___x_90_);
if (v___x_91_ == 0)
{
lean_object* v___x_92_; 
v___x_92_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_92_, 0, v_a_89_);
return v___x_92_;
}
else
{
lean_object* v___x_93_; 
lean_dec(v_a_89_);
v___x_93_ = lean_box(0);
return v___x_93_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_subtypeOrderIso___redArg___lam__1(lean_object* v_self_94_){
_start:
{
lean_inc(v_self_94_);
return v_self_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_subtypeOrderIso___redArg___lam__1___boxed(lean_object* v_self_95_){
_start:
{
lean_object* v_res_96_; 
v_res_96_ = lp_mathlib_WithTop_subtypeOrderIso___redArg___lam__1(v_self_95_);
lean_dec(v_self_95_);
return v_res_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_subtypeOrderIso___redArg___lam__2(lean_object* v___f_97_, lean_object* v_inst_98_, lean_object* v_a_99_){
_start:
{
lean_object* v___x_100_; lean_object* v___x_101_; 
v___x_100_ = lp_mathlib_WithTop_map___redArg(v___f_97_, v_a_99_);
v___x_101_ = lp_mathlib_WithTop_untopD___redArg(v_inst_98_, v___x_100_);
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_subtypeOrderIso___redArg___lam__2___boxed(lean_object* v___f_102_, lean_object* v_inst_103_, lean_object* v_a_104_){
_start:
{
lean_object* v_res_105_; 
v_res_105_ = lp_mathlib_WithTop_subtypeOrderIso___redArg___lam__2(v___f_102_, v_inst_103_, v_a_104_);
lean_dec(v_inst_103_);
return v_res_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_subtypeOrderIso___redArg(lean_object* v_inst_107_, lean_object* v_inst_108_){
_start:
{
lean_object* v___f_109_; lean_object* v___f_110_; lean_object* v___f_111_; lean_object* v___x_112_; 
v___f_109_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_subtypeOrderIso___redArg___lam__0), 2, 1);
lean_closure_set(v___f_109_, 0, v_inst_108_);
v___f_110_ = ((lean_object*)(lp_mathlib_WithTop_subtypeOrderIso___redArg___closed__0));
v___f_111_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_subtypeOrderIso___redArg___lam__2___boxed), 3, 2);
lean_closure_set(v___f_111_, 0, v___f_110_);
lean_closure_set(v___f_111_, 1, v_inst_107_);
v___x_112_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_112_, 0, v___f_111_);
lean_ctor_set(v___x_112_, 1, v___f_109_);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_subtypeOrderIso(lean_object* v_00_u03b1_113_, lean_object* v_inst_114_, lean_object* v_inst_115_, lean_object* v_inst_116_){
_start:
{
lean_object* v___x_117_; 
v___x_117_ = lp_mathlib_WithTop_subtypeOrderIso___redArg(v_inst_115_, v_inst_116_);
return v___x_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_subtypeOrderIso___boxed(lean_object* v_00_u03b1_118_, lean_object* v_inst_119_, lean_object* v_inst_120_, lean_object* v_inst_121_){
_start:
{
lean_object* v_res_122_; 
v_res_122_ = lp_mathlib_WithTop_subtypeOrderIso(v_00_u03b1_118_, v_inst_119_, v_inst_120_, v_inst_121_);
lean_dec_ref(v_inst_119_);
return v_res_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_subtypeOrderIso___redArg___lam__2(lean_object* v___f_123_, lean_object* v_inst_124_, lean_object* v_a_125_){
_start:
{
lean_object* v___x_126_; lean_object* v___x_127_; 
v___x_126_ = lp_mathlib_WithBot_map___redArg(v___f_123_, v_a_125_);
v___x_127_ = lp_mathlib_WithBot_unbotD___redArg(v_inst_124_, v___x_126_);
return v___x_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_subtypeOrderIso___redArg___lam__2___boxed(lean_object* v___f_128_, lean_object* v_inst_129_, lean_object* v_a_130_){
_start:
{
lean_object* v_res_131_; 
v_res_131_ = lp_mathlib_WithBot_subtypeOrderIso___redArg___lam__2(v___f_128_, v_inst_129_, v_a_130_);
lean_dec(v_inst_129_);
return v_res_131_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_subtypeOrderIso___redArg(lean_object* v_inst_132_, lean_object* v_inst_133_){
_start:
{
lean_object* v___f_134_; lean_object* v___f_135_; lean_object* v___f_136_; lean_object* v___x_137_; 
v___f_134_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_subtypeOrderIso___redArg___lam__0), 2, 1);
lean_closure_set(v___f_134_, 0, v_inst_133_);
v___f_135_ = ((lean_object*)(lp_mathlib_WithTop_subtypeOrderIso___redArg___closed__0));
v___f_136_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_subtypeOrderIso___redArg___lam__2___boxed), 3, 2);
lean_closure_set(v___f_136_, 0, v___f_135_);
lean_closure_set(v___f_136_, 1, v_inst_132_);
v___x_137_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_137_, 0, v___f_136_);
lean_ctor_set(v___x_137_, 1, v___f_134_);
return v___x_137_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_subtypeOrderIso(lean_object* v_00_u03b1_138_, lean_object* v_inst_139_, lean_object* v_inst_140_, lean_object* v_inst_141_){
_start:
{
lean_object* v___x_142_; 
v___x_142_ = lp_mathlib_WithBot_subtypeOrderIso___redArg(v_inst_140_, v_inst_141_);
return v___x_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_subtypeOrderIso___boxed(lean_object* v_00_u03b1_143_, lean_object* v_inst_144_, lean_object* v_inst_145_, lean_object* v_inst_146_){
_start:
{
lean_object* v_res_147_; 
v_res_147_ = lp_mathlib_WithBot_subtypeOrderIso(v_00_u03b1_143_, v_inst_144_, v_inst_145_, v_inst_146_);
lean_dec_ref(v_inst_144_);
return v_res_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_withBotMap___redArg___lam__0(lean_object* v_f_148_, lean_object* v___y_149_){
_start:
{
lean_object* v___x_150_; 
v___x_150_ = lean_apply_1(v_f_148_, v___y_149_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_withBotMap___redArg(lean_object* v_f_151_){
_start:
{
lean_object* v___f_152_; lean_object* v___x_153_; 
v___f_152_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_withBotMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_152_, 0, v_f_151_);
v___x_153_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_map), 4, 3);
lean_closure_set(v___x_153_, 0, lean_box(0));
lean_closure_set(v___x_153_, 1, lean_box(0));
lean_closure_set(v___x_153_, 2, v___f_152_);
return v___x_153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_withBotMap(lean_object* v_00_u03b1_154_, lean_object* v_00_u03b2_155_, lean_object* v_inst_156_, lean_object* v_inst_157_, lean_object* v_f_158_){
_start:
{
lean_object* v___x_159_; 
v___x_159_ = lp_mathlib_OrderHom_withBotMap___redArg(v_f_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_withBotMap___boxed(lean_object* v_00_u03b1_160_, lean_object* v_00_u03b2_161_, lean_object* v_inst_162_, lean_object* v_inst_163_, lean_object* v_f_164_){
_start:
{
lean_object* v_res_165_; 
v_res_165_ = lp_mathlib_OrderHom_withBotMap(v_00_u03b1_160_, v_00_u03b2_161_, v_inst_162_, v_inst_163_, v_f_164_);
lean_dec_ref(v_inst_163_);
lean_dec_ref(v_inst_162_);
return v_res_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_withTopMap___redArg(lean_object* v_f_166_){
_start:
{
lean_object* v___f_167_; lean_object* v___x_168_; 
v___f_167_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_withBotMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_167_, 0, v_f_166_);
v___x_168_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_map), 4, 3);
lean_closure_set(v___x_168_, 0, lean_box(0));
lean_closure_set(v___x_168_, 1, lean_box(0));
lean_closure_set(v___x_168_, 2, v___f_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_withTopMap(lean_object* v_00_u03b1_169_, lean_object* v_00_u03b2_170_, lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_f_173_){
_start:
{
lean_object* v___x_174_; 
v___x_174_ = lp_mathlib_OrderHom_withTopMap___redArg(v_f_173_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_withTopMap___boxed(lean_object* v_00_u03b1_175_, lean_object* v_00_u03b2_176_, lean_object* v_inst_177_, lean_object* v_inst_178_, lean_object* v_f_179_){
_start:
{
lean_object* v_res_180_; 
v_res_180_ = lp_mathlib_OrderHom_withTopMap(v_00_u03b1_175_, v_00_u03b2_176_, v_inst_177_, v_inst_178_, v_f_179_);
lean_dec_ref(v_inst_178_);
lean_dec_ref(v_inst_177_);
return v_res_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_withBotMap___redArg(lean_object* v_f_181_){
_start:
{
lean_object* v___f_182_; lean_object* v___x_183_; 
v___f_182_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_withBotMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_182_, 0, v_f_181_);
v___x_183_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_map), 4, 3);
lean_closure_set(v___x_183_, 0, lean_box(0));
lean_closure_set(v___x_183_, 1, lean_box(0));
lean_closure_set(v___x_183_, 2, v___f_182_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_withBotMap(lean_object* v_00_u03b1_184_, lean_object* v_00_u03b2_185_, lean_object* v_inst_186_, lean_object* v_inst_187_, lean_object* v_f_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_mathlib_OrderEmbedding_withBotMap___redArg(v_f_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_withBotMap___boxed(lean_object* v_00_u03b1_190_, lean_object* v_00_u03b2_191_, lean_object* v_inst_192_, lean_object* v_inst_193_, lean_object* v_f_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_mathlib_OrderEmbedding_withBotMap(v_00_u03b1_190_, v_00_u03b2_191_, v_inst_192_, v_inst_193_, v_f_194_);
lean_dec_ref(v_inst_193_);
lean_dec_ref(v_inst_192_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_withTopMap___redArg(lean_object* v_f_196_){
_start:
{
lean_object* v___f_197_; lean_object* v___x_198_; 
v___f_197_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_withBotMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_197_, 0, v_f_196_);
v___x_198_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_map), 4, 3);
lean_closure_set(v___x_198_, 0, lean_box(0));
lean_closure_set(v___x_198_, 1, lean_box(0));
lean_closure_set(v___x_198_, 2, v___f_197_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_withTopMap(lean_object* v_00_u03b1_199_, lean_object* v_00_u03b2_200_, lean_object* v_inst_201_, lean_object* v_inst_202_, lean_object* v_f_203_){
_start:
{
lean_object* v___x_204_; 
v___x_204_ = lp_mathlib_OrderEmbedding_withTopMap___redArg(v_f_203_);
return v___x_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderEmbedding_withTopMap___boxed(lean_object* v_00_u03b1_205_, lean_object* v_00_u03b2_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_f_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_mathlib_OrderEmbedding_withTopMap(v_00_u03b1_205_, v_00_u03b2_206_, v_inst_207_, v_inst_208_, v_f_209_);
lean_dec_ref(v_inst_208_);
lean_dec_ref(v_inst_207_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withTopCongr___redArg___lam__0(lean_object* v_e_211_, lean_object* v___y_212_){
_start:
{
lean_object* v___x_213_; 
v___x_213_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_e_211_, v___y_212_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withTopCongr___redArg(lean_object* v_e_214_){
_start:
{
lean_object* v___x_215_; lean_object* v_invFun_216_; lean_object* v___x_218_; uint8_t v_isShared_219_; uint8_t v_isSharedCheck_225_; 
lean_inc_ref(v_e_214_);
v___x_215_ = lp_mathlib_Equiv_withTopCongr___redArg(v_e_214_);
v_invFun_216_ = lean_ctor_get(v___x_215_, 1);
v_isSharedCheck_225_ = !lean_is_exclusive(v___x_215_);
if (v_isSharedCheck_225_ == 0)
{
lean_object* v_unused_226_; 
v_unused_226_ = lean_ctor_get(v___x_215_, 0);
lean_dec(v_unused_226_);
v___x_218_ = v___x_215_;
v_isShared_219_ = v_isSharedCheck_225_;
goto v_resetjp_217_;
}
else
{
lean_inc(v_invFun_216_);
lean_dec(v___x_215_);
v___x_218_ = lean_box(0);
v_isShared_219_ = v_isSharedCheck_225_;
goto v_resetjp_217_;
}
v_resetjp_217_:
{
lean_object* v___f_220_; lean_object* v___x_221_; lean_object* v___x_223_; 
v___f_220_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_withTopCongr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_220_, 0, v_e_214_);
v___x_221_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_map), 4, 3);
lean_closure_set(v___x_221_, 0, lean_box(0));
lean_closure_set(v___x_221_, 1, lean_box(0));
lean_closure_set(v___x_221_, 2, v___f_220_);
if (v_isShared_219_ == 0)
{
lean_ctor_set(v___x_218_, 0, v___x_221_);
v___x_223_ = v___x_218_;
goto v_reusejp_222_;
}
else
{
lean_object* v_reuseFailAlloc_224_; 
v_reuseFailAlloc_224_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_224_, 0, v___x_221_);
lean_ctor_set(v_reuseFailAlloc_224_, 1, v_invFun_216_);
v___x_223_ = v_reuseFailAlloc_224_;
goto v_reusejp_222_;
}
v_reusejp_222_:
{
return v___x_223_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withTopCongr(lean_object* v_00_u03b1_227_, lean_object* v_00_u03b2_228_, lean_object* v_inst_229_, lean_object* v_inst_230_, lean_object* v_e_231_){
_start:
{
lean_object* v___x_232_; 
v___x_232_ = lp_mathlib_OrderIso_withTopCongr___redArg(v_e_231_);
return v___x_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withTopCongr___boxed(lean_object* v_00_u03b1_233_, lean_object* v_00_u03b2_234_, lean_object* v_inst_235_, lean_object* v_inst_236_, lean_object* v_e_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_mathlib_OrderIso_withTopCongr(v_00_u03b1_233_, v_00_u03b2_234_, v_inst_235_, v_inst_236_, v_e_237_);
lean_dec_ref(v_inst_236_);
lean_dec_ref(v_inst_235_);
return v_res_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withBotCongr___redArg(lean_object* v_e_239_){
_start:
{
lean_object* v___x_240_; lean_object* v_invFun_241_; lean_object* v___x_243_; uint8_t v_isShared_244_; uint8_t v_isSharedCheck_250_; 
lean_inc_ref(v_e_239_);
v___x_240_ = lp_mathlib_Equiv_withBotCongr___redArg(v_e_239_);
v_invFun_241_ = lean_ctor_get(v___x_240_, 1);
v_isSharedCheck_250_ = !lean_is_exclusive(v___x_240_);
if (v_isSharedCheck_250_ == 0)
{
lean_object* v_unused_251_; 
v_unused_251_ = lean_ctor_get(v___x_240_, 0);
lean_dec(v_unused_251_);
v___x_243_ = v___x_240_;
v_isShared_244_ = v_isSharedCheck_250_;
goto v_resetjp_242_;
}
else
{
lean_inc(v_invFun_241_);
lean_dec(v___x_240_);
v___x_243_ = lean_box(0);
v_isShared_244_ = v_isSharedCheck_250_;
goto v_resetjp_242_;
}
v_resetjp_242_:
{
lean_object* v___f_245_; lean_object* v___x_246_; lean_object* v___x_248_; 
v___f_245_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_withTopCongr___redArg___lam__0), 2, 1);
lean_closure_set(v___f_245_, 0, v_e_239_);
v___x_246_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_map), 4, 3);
lean_closure_set(v___x_246_, 0, lean_box(0));
lean_closure_set(v___x_246_, 1, lean_box(0));
lean_closure_set(v___x_246_, 2, v___f_245_);
if (v_isShared_244_ == 0)
{
lean_ctor_set(v___x_243_, 0, v___x_246_);
v___x_248_ = v___x_243_;
goto v_reusejp_247_;
}
else
{
lean_object* v_reuseFailAlloc_249_; 
v_reuseFailAlloc_249_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_249_, 0, v___x_246_);
lean_ctor_set(v_reuseFailAlloc_249_, 1, v_invFun_241_);
v___x_248_ = v_reuseFailAlloc_249_;
goto v_reusejp_247_;
}
v_reusejp_247_:
{
return v___x_248_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withBotCongr(lean_object* v_00_u03b1_252_, lean_object* v_00_u03b2_253_, lean_object* v_inst_254_, lean_object* v_inst_255_, lean_object* v_e_256_){
_start:
{
lean_object* v___x_257_; 
v___x_257_ = lp_mathlib_OrderIso_withBotCongr___redArg(v_e_256_);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_withBotCongr___boxed(lean_object* v_00_u03b1_258_, lean_object* v_00_u03b2_259_, lean_object* v_inst_260_, lean_object* v_inst_261_, lean_object* v_e_262_){
_start:
{
lean_object* v_res_263_; 
v_res_263_ = lp_mathlib_OrderIso_withBotCongr(v_00_u03b1_258_, v_00_u03b2_259_, v_inst_260_, v_inst_261_, v_e_262_);
lean_dec_ref(v_inst_261_);
lean_dec_ref(v_inst_260_);
return v_res_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withTop___redArg(lean_object* v_f_264_){
_start:
{
lean_object* v___f_265_; lean_object* v___x_266_; 
v___f_265_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_withBotMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_265_, 0, v_f_264_);
v___x_266_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_map), 4, 3);
lean_closure_set(v___x_266_, 0, lean_box(0));
lean_closure_set(v___x_266_, 1, lean_box(0));
lean_closure_set(v___x_266_, 2, v___f_265_);
return v___x_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withTop(lean_object* v_00_u03b1_267_, lean_object* v_00_u03b2_268_, lean_object* v_inst_269_, lean_object* v_inst_270_, lean_object* v_f_271_){
_start:
{
lean_object* v___x_272_; 
v___x_272_ = lp_mathlib_SupHom_withTop___redArg(v_f_271_);
return v___x_272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withTop___boxed(lean_object* v_00_u03b1_273_, lean_object* v_00_u03b2_274_, lean_object* v_inst_275_, lean_object* v_inst_276_, lean_object* v_f_277_){
_start:
{
lean_object* v_res_278_; 
v_res_278_ = lp_mathlib_SupHom_withTop(v_00_u03b1_273_, v_00_u03b2_274_, v_inst_275_, v_inst_276_, v_f_277_);
lean_dec_ref(v_inst_276_);
lean_dec_ref(v_inst_275_);
return v_res_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withBot___redArg(lean_object* v_f_279_){
_start:
{
lean_object* v___f_280_; lean_object* v___x_281_; 
v___f_280_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_withBotMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_280_, 0, v_f_279_);
v___x_281_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_map), 4, 3);
lean_closure_set(v___x_281_, 0, lean_box(0));
lean_closure_set(v___x_281_, 1, lean_box(0));
lean_closure_set(v___x_281_, 2, v___f_280_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withBot(lean_object* v_00_u03b1_282_, lean_object* v_00_u03b2_283_, lean_object* v_inst_284_, lean_object* v_inst_285_, lean_object* v_f_286_){
_start:
{
lean_object* v___x_287_; 
v___x_287_ = lp_mathlib_InfHom_withBot___redArg(v_f_286_);
return v___x_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withBot___boxed(lean_object* v_00_u03b1_288_, lean_object* v_00_u03b2_289_, lean_object* v_inst_290_, lean_object* v_inst_291_, lean_object* v_f_292_){
_start:
{
lean_object* v_res_293_; 
v_res_293_ = lp_mathlib_InfHom_withBot(v_00_u03b1_288_, v_00_u03b2_289_, v_inst_290_, v_inst_291_, v_f_292_);
lean_dec_ref(v_inst_291_);
lean_dec_ref(v_inst_290_);
return v_res_293_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withBot___redArg(lean_object* v_f_294_){
_start:
{
lean_object* v___f_295_; lean_object* v___x_296_; 
v___f_295_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_withBotMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_295_, 0, v_f_294_);
v___x_296_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_map), 4, 3);
lean_closure_set(v___x_296_, 0, lean_box(0));
lean_closure_set(v___x_296_, 1, lean_box(0));
lean_closure_set(v___x_296_, 2, v___f_295_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withBot(lean_object* v_00_u03b1_297_, lean_object* v_00_u03b2_298_, lean_object* v_inst_299_, lean_object* v_inst_300_, lean_object* v_f_301_){
_start:
{
lean_object* v___x_302_; 
v___x_302_ = lp_mathlib_SupHom_withBot___redArg(v_f_301_);
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withBot___boxed(lean_object* v_00_u03b1_303_, lean_object* v_00_u03b2_304_, lean_object* v_inst_305_, lean_object* v_inst_306_, lean_object* v_f_307_){
_start:
{
lean_object* v_res_308_; 
v_res_308_ = lp_mathlib_SupHom_withBot(v_00_u03b1_303_, v_00_u03b2_304_, v_inst_305_, v_inst_306_, v_f_307_);
lean_dec_ref(v_inst_306_);
lean_dec_ref(v_inst_305_);
return v_res_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withTop___redArg(lean_object* v_f_309_){
_start:
{
lean_object* v___f_310_; lean_object* v___x_311_; 
v___f_310_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_withBotMap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_310_, 0, v_f_309_);
v___x_311_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_map), 4, 3);
lean_closure_set(v___x_311_, 0, lean_box(0));
lean_closure_set(v___x_311_, 1, lean_box(0));
lean_closure_set(v___x_311_, 2, v___f_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withTop(lean_object* v_00_u03b1_312_, lean_object* v_00_u03b2_313_, lean_object* v_inst_314_, lean_object* v_inst_315_, lean_object* v_f_316_){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = lp_mathlib_InfHom_withTop___redArg(v_f_316_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withTop___boxed(lean_object* v_00_u03b1_318_, lean_object* v_00_u03b2_319_, lean_object* v_inst_320_, lean_object* v_inst_321_, lean_object* v_f_322_){
_start:
{
lean_object* v_res_323_; 
v_res_323_ = lp_mathlib_InfHom_withTop(v_00_u03b1_318_, v_00_u03b2_319_, v_inst_320_, v_inst_321_, v_f_322_);
lean_dec_ref(v_inst_321_);
lean_dec_ref(v_inst_320_);
return v_res_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withTop_x27___redArg___lam__0(lean_object* v_inst_324_, lean_object* v_f_325_, lean_object* v_a_326_){
_start:
{
if (lean_obj_tag(v_a_326_) == 0)
{
lean_dec(v_f_325_);
lean_inc(v_inst_324_);
return v_inst_324_;
}
else
{
lean_object* v_val_327_; lean_object* v___x_328_; 
v_val_327_ = lean_ctor_get(v_a_326_, 0);
lean_inc(v_val_327_);
lean_dec_ref_known(v_a_326_, 1);
v___x_328_ = lean_apply_1(v_f_325_, v_val_327_);
return v___x_328_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed(lean_object* v_inst_329_, lean_object* v_f_330_, lean_object* v_a_331_){
_start:
{
lean_object* v_res_332_; 
v_res_332_ = lp_mathlib_SupHom_withTop_x27___redArg___lam__0(v_inst_329_, v_f_330_, v_a_331_);
lean_dec(v_inst_329_);
return v_res_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withTop_x27___redArg(lean_object* v_inst_333_, lean_object* v_f_334_){
_start:
{
lean_object* v___f_335_; 
v___f_335_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_335_, 0, v_inst_333_);
lean_closure_set(v___f_335_, 1, v_f_334_);
return v___f_335_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withTop_x27(lean_object* v_00_u03b1_336_, lean_object* v_00_u03b2_337_, lean_object* v_inst_338_, lean_object* v_inst_339_, lean_object* v_inst_340_, lean_object* v_f_341_){
_start:
{
lean_object* v___f_342_; 
v___f_342_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_342_, 0, v_inst_340_);
lean_closure_set(v___f_342_, 1, v_f_341_);
return v___f_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withTop_x27___boxed(lean_object* v_00_u03b1_343_, lean_object* v_00_u03b2_344_, lean_object* v_inst_345_, lean_object* v_inst_346_, lean_object* v_inst_347_, lean_object* v_f_348_){
_start:
{
lean_object* v_res_349_; 
v_res_349_ = lp_mathlib_SupHom_withTop_x27(v_00_u03b1_343_, v_00_u03b2_344_, v_inst_345_, v_inst_346_, v_inst_347_, v_f_348_);
lean_dec_ref(v_inst_346_);
lean_dec_ref(v_inst_345_);
return v_res_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withBot_x27___redArg(lean_object* v_inst_350_, lean_object* v_f_351_){
_start:
{
lean_object* v___f_352_; 
v___f_352_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_352_, 0, v_inst_350_);
lean_closure_set(v___f_352_, 1, v_f_351_);
return v___f_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withBot_x27(lean_object* v_00_u03b1_353_, lean_object* v_00_u03b2_354_, lean_object* v_inst_355_, lean_object* v_inst_356_, lean_object* v_inst_357_, lean_object* v_f_358_){
_start:
{
lean_object* v___f_359_; 
v___f_359_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_359_, 0, v_inst_357_);
lean_closure_set(v___f_359_, 1, v_f_358_);
return v___f_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withBot_x27___boxed(lean_object* v_00_u03b1_360_, lean_object* v_00_u03b2_361_, lean_object* v_inst_362_, lean_object* v_inst_363_, lean_object* v_inst_364_, lean_object* v_f_365_){
_start:
{
lean_object* v_res_366_; 
v_res_366_ = lp_mathlib_InfHom_withBot_x27(v_00_u03b1_360_, v_00_u03b2_361_, v_inst_362_, v_inst_363_, v_inst_364_, v_f_365_);
lean_dec_ref(v_inst_363_);
lean_dec_ref(v_inst_362_);
return v_res_366_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withBot_x27___redArg(lean_object* v_inst_367_, lean_object* v_f_368_){
_start:
{
lean_object* v___f_369_; 
v___f_369_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_369_, 0, v_inst_367_);
lean_closure_set(v___f_369_, 1, v_f_368_);
return v___f_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withBot_x27(lean_object* v_00_u03b1_370_, lean_object* v_00_u03b2_371_, lean_object* v_inst_372_, lean_object* v_inst_373_, lean_object* v_inst_374_, lean_object* v_f_375_){
_start:
{
lean_object* v___f_376_; 
v___f_376_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_376_, 0, v_inst_374_);
lean_closure_set(v___f_376_, 1, v_f_375_);
return v___f_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SupHom_withBot_x27___boxed(lean_object* v_00_u03b1_377_, lean_object* v_00_u03b2_378_, lean_object* v_inst_379_, lean_object* v_inst_380_, lean_object* v_inst_381_, lean_object* v_f_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_mathlib_SupHom_withBot_x27(v_00_u03b1_377_, v_00_u03b2_378_, v_inst_379_, v_inst_380_, v_inst_381_, v_f_382_);
lean_dec_ref(v_inst_380_);
lean_dec_ref(v_inst_379_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withTop_x27___redArg(lean_object* v_inst_384_, lean_object* v_f_385_){
_start:
{
lean_object* v___f_386_; 
v___f_386_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_386_, 0, v_inst_384_);
lean_closure_set(v___f_386_, 1, v_f_385_);
return v___f_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withTop_x27(lean_object* v_00_u03b1_387_, lean_object* v_00_u03b2_388_, lean_object* v_inst_389_, lean_object* v_inst_390_, lean_object* v_inst_391_, lean_object* v_f_392_){
_start:
{
lean_object* v___f_393_; 
v___f_393_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_393_, 0, v_inst_391_);
lean_closure_set(v___f_393_, 1, v_f_392_);
return v___f_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_InfHom_withTop_x27___boxed(lean_object* v_00_u03b1_394_, lean_object* v_00_u03b2_395_, lean_object* v_inst_396_, lean_object* v_inst_397_, lean_object* v_inst_398_, lean_object* v_f_399_){
_start:
{
lean_object* v_res_400_; 
v_res_400_ = lp_mathlib_InfHom_withTop_x27(v_00_u03b1_394_, v_00_u03b2_395_, v_inst_396_, v_inst_397_, v_inst_398_, v_f_399_);
lean_dec_ref(v_inst_397_);
lean_dec_ref(v_inst_396_);
return v_res_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTop___redArg(lean_object* v_f_401_){
_start:
{
lean_object* v___x_402_; 
v___x_402_ = lp_mathlib_SupHom_withTop___redArg(v_f_401_);
return v___x_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTop(lean_object* v_00_u03b1_403_, lean_object* v_00_u03b2_404_, lean_object* v_inst_405_, lean_object* v_inst_406_, lean_object* v_f_407_){
_start:
{
lean_object* v___x_408_; 
v___x_408_ = lp_mathlib_SupHom_withTop___redArg(v_f_407_);
return v___x_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTop___boxed(lean_object* v_00_u03b1_409_, lean_object* v_00_u03b2_410_, lean_object* v_inst_411_, lean_object* v_inst_412_, lean_object* v_f_413_){
_start:
{
lean_object* v_res_414_; 
v_res_414_ = lp_mathlib_LatticeHom_withTop(v_00_u03b1_409_, v_00_u03b2_410_, v_inst_411_, v_inst_412_, v_f_413_);
lean_dec_ref(v_inst_412_);
lean_dec_ref(v_inst_411_);
return v_res_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBot___redArg___lam__0(lean_object* v_f_415_, lean_object* v___y_416_){
_start:
{
lean_object* v___x_41__overap_417_; lean_object* v___x_418_; 
v___x_41__overap_417_ = lp_mathlib_SupHom_withBot___redArg(v_f_415_);
v___x_418_ = lean_apply_1(v___x_41__overap_417_, v___y_416_);
return v___x_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBot___redArg(lean_object* v_f_419_){
_start:
{
lean_object* v___f_420_; 
v___f_420_ = lean_alloc_closure((void*)(lp_mathlib_LatticeHom_withBot___redArg___lam__0), 2, 1);
lean_closure_set(v___f_420_, 0, v_f_419_);
return v___f_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBot(lean_object* v_00_u03b1_421_, lean_object* v_00_u03b2_422_, lean_object* v_inst_423_, lean_object* v_inst_424_, lean_object* v_f_425_){
_start:
{
lean_object* v___f_426_; 
v___f_426_ = lean_alloc_closure((void*)(lp_mathlib_LatticeHom_withBot___redArg___lam__0), 2, 1);
lean_closure_set(v___f_426_, 0, v_f_425_);
return v___f_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBot___boxed(lean_object* v_00_u03b1_427_, lean_object* v_00_u03b2_428_, lean_object* v_inst_429_, lean_object* v_inst_430_, lean_object* v_f_431_){
_start:
{
lean_object* v_res_432_; 
v_res_432_ = lp_mathlib_LatticeHom_withBot(v_00_u03b1_427_, v_00_u03b2_428_, v_inst_429_, v_inst_430_, v_f_431_);
lean_dec_ref(v_inst_430_);
lean_dec_ref(v_inst_429_);
return v_res_432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTopWithBot___redArg(lean_object* v_f_433_){
_start:
{
lean_object* v___f_434_; lean_object* v___x_435_; 
v___f_434_ = lean_alloc_closure((void*)(lp_mathlib_LatticeHom_withBot___redArg___lam__0), 2, 1);
lean_closure_set(v___f_434_, 0, v_f_433_);
v___x_435_ = lp_mathlib_SupHom_withTop___redArg(v___f_434_);
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTopWithBot(lean_object* v_00_u03b1_436_, lean_object* v_00_u03b2_437_, lean_object* v_inst_438_, lean_object* v_inst_439_, lean_object* v_f_440_){
_start:
{
lean_object* v___x_441_; 
v___x_441_ = lp_mathlib_LatticeHom_withTopWithBot___redArg(v_f_440_);
return v___x_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTopWithBot___boxed(lean_object* v_00_u03b1_442_, lean_object* v_00_u03b2_443_, lean_object* v_inst_444_, lean_object* v_inst_445_, lean_object* v_f_446_){
_start:
{
lean_object* v_res_447_; 
v_res_447_ = lp_mathlib_LatticeHom_withTopWithBot(v_00_u03b1_442_, v_00_u03b2_443_, v_inst_444_, v_inst_445_, v_f_446_);
lean_dec_ref(v_inst_445_);
lean_dec_ref(v_inst_444_);
return v_res_447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBotWithTop___redArg(lean_object* v_f_448_){
_start:
{
lean_object* v___x_449_; lean_object* v___f_450_; 
v___x_449_ = lp_mathlib_SupHom_withTop___redArg(v_f_448_);
v___f_450_ = lean_alloc_closure((void*)(lp_mathlib_LatticeHom_withBot___redArg___lam__0), 2, 1);
lean_closure_set(v___f_450_, 0, v___x_449_);
return v___f_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBotWithTop(lean_object* v_00_u03b1_451_, lean_object* v_00_u03b2_452_, lean_object* v_inst_453_, lean_object* v_inst_454_, lean_object* v_f_455_){
_start:
{
lean_object* v___x_456_; 
v___x_456_ = lp_mathlib_LatticeHom_withBotWithTop___redArg(v_f_455_);
return v___x_456_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBotWithTop___boxed(lean_object* v_00_u03b1_457_, lean_object* v_00_u03b2_458_, lean_object* v_inst_459_, lean_object* v_inst_460_, lean_object* v_f_461_){
_start:
{
lean_object* v_res_462_; 
v_res_462_ = lp_mathlib_LatticeHom_withBotWithTop(v_00_u03b1_457_, v_00_u03b2_458_, v_inst_459_, v_inst_460_, v_f_461_);
lean_dec_ref(v_inst_460_);
lean_dec_ref(v_inst_459_);
return v_res_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTop_x27___redArg(lean_object* v_inst_463_, lean_object* v_f_464_){
_start:
{
lean_object* v___f_465_; 
v___f_465_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_465_, 0, v_inst_463_);
lean_closure_set(v___f_465_, 1, v_f_464_);
return v___f_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTop_x27(lean_object* v_00_u03b1_466_, lean_object* v_00_u03b2_467_, lean_object* v_inst_468_, lean_object* v_inst_469_, lean_object* v_inst_470_, lean_object* v_f_471_){
_start:
{
lean_object* v___f_472_; 
v___f_472_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_472_, 0, v_inst_470_);
lean_closure_set(v___f_472_, 1, v_f_471_);
return v___f_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTop_x27___boxed(lean_object* v_00_u03b1_473_, lean_object* v_00_u03b2_474_, lean_object* v_inst_475_, lean_object* v_inst_476_, lean_object* v_inst_477_, lean_object* v_f_478_){
_start:
{
lean_object* v_res_479_; 
v_res_479_ = lp_mathlib_LatticeHom_withTop_x27(v_00_u03b1_473_, v_00_u03b2_474_, v_inst_475_, v_inst_476_, v_inst_477_, v_f_478_);
lean_dec_ref(v_inst_476_);
lean_dec_ref(v_inst_475_);
return v_res_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBot_x27___redArg(lean_object* v_inst_480_, lean_object* v_f_481_){
_start:
{
lean_object* v___f_482_; 
v___f_482_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_482_, 0, v_inst_480_);
lean_closure_set(v___f_482_, 1, v_f_481_);
return v___f_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBot_x27(lean_object* v_00_u03b1_483_, lean_object* v_00_u03b2_484_, lean_object* v_inst_485_, lean_object* v_inst_486_, lean_object* v_inst_487_, lean_object* v_f_488_){
_start:
{
lean_object* v___f_489_; 
v___f_489_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_489_, 0, v_inst_487_);
lean_closure_set(v___f_489_, 1, v_f_488_);
return v___f_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBot_x27___boxed(lean_object* v_00_u03b1_490_, lean_object* v_00_u03b2_491_, lean_object* v_inst_492_, lean_object* v_inst_493_, lean_object* v_inst_494_, lean_object* v_f_495_){
_start:
{
lean_object* v_res_496_; 
v_res_496_ = lp_mathlib_LatticeHom_withBot_x27(v_00_u03b1_490_, v_00_u03b2_491_, v_inst_492_, v_inst_493_, v_inst_494_, v_f_495_);
lean_dec_ref(v_inst_493_);
lean_dec_ref(v_inst_492_);
return v_res_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTopWithBot_x27___redArg(lean_object* v_inst_497_, lean_object* v_f_498_){
_start:
{
lean_object* v_toOrderTop_499_; lean_object* v_toOrderBot_500_; lean_object* v___f_501_; lean_object* v___f_502_; 
v_toOrderTop_499_ = lean_ctor_get(v_inst_497_, 0);
lean_inc(v_toOrderTop_499_);
v_toOrderBot_500_ = lean_ctor_get(v_inst_497_, 1);
lean_inc(v_toOrderBot_500_);
lean_dec_ref(v_inst_497_);
v___f_501_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_501_, 0, v_toOrderBot_500_);
lean_closure_set(v___f_501_, 1, v_f_498_);
v___f_502_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_502_, 0, v_toOrderTop_499_);
lean_closure_set(v___f_502_, 1, v___f_501_);
return v___f_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTopWithBot_x27(lean_object* v_00_u03b1_503_, lean_object* v_00_u03b2_504_, lean_object* v_inst_505_, lean_object* v_inst_506_, lean_object* v_inst_507_, lean_object* v_f_508_){
_start:
{
lean_object* v___x_509_; 
v___x_509_ = lp_mathlib_LatticeHom_withTopWithBot_x27___redArg(v_inst_507_, v_f_508_);
return v___x_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withTopWithBot_x27___boxed(lean_object* v_00_u03b1_510_, lean_object* v_00_u03b2_511_, lean_object* v_inst_512_, lean_object* v_inst_513_, lean_object* v_inst_514_, lean_object* v_f_515_){
_start:
{
lean_object* v_res_516_; 
v_res_516_ = lp_mathlib_LatticeHom_withTopWithBot_x27(v_00_u03b1_510_, v_00_u03b2_511_, v_inst_512_, v_inst_513_, v_inst_514_, v_f_515_);
lean_dec_ref(v_inst_513_);
lean_dec_ref(v_inst_512_);
return v_res_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBotWithTop_x27___redArg(lean_object* v_inst_517_, lean_object* v_f_518_){
_start:
{
lean_object* v_toOrderTop_519_; lean_object* v_toOrderBot_520_; lean_object* v___f_521_; lean_object* v___f_522_; 
v_toOrderTop_519_ = lean_ctor_get(v_inst_517_, 0);
lean_inc(v_toOrderTop_519_);
v_toOrderBot_520_ = lean_ctor_get(v_inst_517_, 1);
lean_inc(v_toOrderBot_520_);
lean_dec_ref(v_inst_517_);
v___f_521_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_521_, 0, v_toOrderTop_519_);
lean_closure_set(v___f_521_, 1, v_f_518_);
v___f_522_ = lean_alloc_closure((void*)(lp_mathlib_SupHom_withTop_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_522_, 0, v_toOrderBot_520_);
lean_closure_set(v___f_522_, 1, v___f_521_);
return v___f_522_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBotWithTop_x27(lean_object* v_00_u03b1_523_, lean_object* v_00_u03b2_524_, lean_object* v_inst_525_, lean_object* v_inst_526_, lean_object* v_inst_527_, lean_object* v_f_528_){
_start:
{
lean_object* v___x_529_; 
v___x_529_ = lp_mathlib_LatticeHom_withBotWithTop_x27___redArg(v_inst_527_, v_f_528_);
return v___x_529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LatticeHom_withBotWithTop_x27___boxed(lean_object* v_00_u03b1_530_, lean_object* v_00_u03b2_531_, lean_object* v_inst_532_, lean_object* v_inst_533_, lean_object* v_inst_534_, lean_object* v_f_535_){
_start:
{
lean_object* v_res_536_; 
v_res_536_ = lp_mathlib_LatticeHom_withBotWithTop_x27(v_00_u03b1_530_, v_00_u03b2_531_, v_inst_532_, v_inst_533_, v_inst_534_, v_f_535_);
lean_dec_ref(v_inst_533_);
lean_dec_ref(v_inst_532_);
return v_res_536_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_WithBot(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_ApplyFun(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_WithTopBot(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_ApplyFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Hom_WithTopBot(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_WithBot(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_ApplyFun(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Hom_WithTopBot(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_BoundedLattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_WithBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_ApplyFun(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_WithTopBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Hom_WithTopBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Hom_WithTopBot(builtin);
}
#ifdef __cplusplus
}
#endif
