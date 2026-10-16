// Lean compiler output
// Module: Mathlib.Order.Antisymmetrization
// Imports: public import Init public meta import Init public import Mathlib.Logic.Relation public import Mathlib.Order.Hom.Basic public import Mathlib.Tactic.Tauto
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
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkAppM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Lean_MVarId_assignIfDefEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Quotient_map_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Quotient_lift_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Function_uncurry(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Quotient_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_decidableEqOfDecidableLE___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_decidableEqOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AntisymmRel_decidableRel___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AntisymmRel_decidableRel___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AntisymmRel_decidableRel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AntisymmRel_decidableRel___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "GCongr"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "AntisymmRel"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "left"};
static const lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__4_value;
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(139, 222, 98, 232, 116, 132, 69, 249)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__5_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(146, 153, 133, 104, 132, 205, 224, 102)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__5_value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__5_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(57, 2, 185, 221, 240, 5, 144, 43)}};
static const lean_ctor_object lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__5_value_aux_3),((lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(100, 101, 74, 109, 87, 67, 34, 43)}};
static const lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___closed__0_value;
LEAN_EXPORT const lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft = (const lean_object*)&lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AntisymmRel_setoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toAntisymmetrization___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toAntisymmetrization___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toAntisymmetrization(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_toAntisymmetrization___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAntisymmetrization___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAntisymmetrization___aux__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAntisymmetrization___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAntisymmetrization___aux__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAntisymmetrization___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAntisymmetrization___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAntisymmetrization(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAntisymmetrization___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransLeAntisymmRel(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransLeAntisymmRel___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransAntisymmRelLe(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransAntisymmRelLe___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransLtAntisymmRelLe(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransLtAntisymmRelLe___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransAntisymmRelLeLt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransAntisymmRelLeLt___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_instPartialOrderAntisymmetrization___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_instPartialOrderAntisymmetrization___closed__0 = (const lean_object*)&lp_mathlib_instPartialOrderAntisymmetrization___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_instPartialOrderAntisymmetrization(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instPartialOrderAntisymmetrization___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_antisymmetrization___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_antisymmetrization___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_antisymmetrization(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_antisymmetrization___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderIso_dualAntisymmetrization___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_OrderIso_dualAntisymmetrization___closed__0 = (const lean_object*)&lp_mathlib_OrderIso_dualAntisymmetrization___closed__0_value;
static const lean_closure_object lp_mathlib_OrderIso_dualAntisymmetrization___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*6, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Quotient_map_x27, .m_arity = 7, .m_num_fixed = 6, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OrderIso_dualAntisymmetrization___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_OrderIso_dualAntisymmetrization___closed__1 = (const lean_object*)&lp_mathlib_OrderIso_dualAntisymmetrization___closed__1_value;
static const lean_ctor_object lp_mathlib_OrderIso_dualAntisymmetrization___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_OrderIso_dualAntisymmetrization___closed__1_value),((lean_object*)&lp_mathlib_OrderIso_dualAntisymmetrization___closed__1_value)}};
static const lean_object* lp_mathlib_OrderIso_dualAntisymmetrization___closed__2 = (const lean_object*)&lp_mathlib_OrderIso_dualAntisymmetrization___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_dualAntisymmetrization(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_dualAntisymmetrization___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransSymmGenLeAntisymmRel(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransSymmGenLeAntisymmRel___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransAntisymmRelLeSymmGen(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTransAntisymmRelLeSymmGen___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Antisymmetrization_prodEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Antisymmetrization_prodEquiv___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Antisymmetrization_prodEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Antisymmetrization_prodEquiv___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Antisymmetrization_prodEquiv___closed__0 = (const lean_object*)&lp_mathlib_Antisymmetrization_prodEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_Antisymmetrization_prodEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Antisymmetrization_prodEquiv___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Antisymmetrization_prodEquiv___closed__1 = (const lean_object*)&lp_mathlib_Antisymmetrization_prodEquiv___closed__1_value;
static const lean_closure_object lp_mathlib_Antisymmetrization_prodEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*5, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Quotient_lift, .m_arity = 6, .m_num_fixed = 5, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Antisymmetrization_prodEquiv___closed__0_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Antisymmetrization_prodEquiv___closed__2 = (const lean_object*)&lp_mathlib_Antisymmetrization_prodEquiv___closed__2_value;
static const lean_closure_object lp_mathlib_Antisymmetrization_prodEquiv___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*7, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Quotient_lift_u2082, .m_arity = 9, .m_num_fixed = 7, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Antisymmetrization_prodEquiv___closed__1_value),((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_Antisymmetrization_prodEquiv___closed__3 = (const lean_object*)&lp_mathlib_Antisymmetrization_prodEquiv___closed__3_value;
static const lean_closure_object lp_mathlib_Antisymmetrization_prodEquiv___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*4, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Function_uncurry, .m_arity = 5, .m_num_fixed = 4, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Antisymmetrization_prodEquiv___closed__3_value)} };
static const lean_object* lp_mathlib_Antisymmetrization_prodEquiv___closed__4 = (const lean_object*)&lp_mathlib_Antisymmetrization_prodEquiv___closed__4_value;
static const lean_ctor_object lp_mathlib_Antisymmetrization_prodEquiv___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Antisymmetrization_prodEquiv___closed__2_value),((lean_object*)&lp_mathlib_Antisymmetrization_prodEquiv___closed__4_value)}};
static const lean_object* lp_mathlib_Antisymmetrization_prodEquiv___closed__5 = (const lean_object*)&lp_mathlib_Antisymmetrization_prodEquiv___closed__5_value;
LEAN_EXPORT lean_object* lp_mathlib_Antisymmetrization_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Antisymmetrization_prodEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AntisymmRel_decidableRel___redArg(lean_object* v_inst_1_, lean_object* v_x_2_, lean_object* v_x_3_){
_start:
{
lean_object* v___x_4_; lean_object* v___x_5_; uint8_t v___x_6_; 
lean_inc_ref(v_inst_1_);
lean_inc(v_x_2_);
lean_inc(v_x_3_);
v___x_4_ = lean_apply_2(v_inst_1_, v_x_3_, v_x_2_);
v___x_5_ = lean_apply_2(v_inst_1_, v_x_2_, v_x_3_);
v___x_6_ = lean_unbox(v___x_5_);
if (v___x_6_ == 0)
{
uint8_t v___x_7_; 
v___x_7_ = lean_unbox(v___x_5_);
return v___x_7_;
}
else
{
uint8_t v___x_8_; 
v___x_8_ = lean_unbox(v___x_4_);
return v___x_8_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AntisymmRel_decidableRel___redArg___boxed(lean_object* v_inst_9_, lean_object* v_x_10_, lean_object* v_x_11_){
_start:
{
uint8_t v_res_12_; lean_object* v_r_13_; 
v_res_12_ = lp_mathlib_AntisymmRel_decidableRel___redArg(v_inst_9_, v_x_10_, v_x_11_);
v_r_13_ = lean_box(v_res_12_);
return v_r_13_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AntisymmRel_decidableRel(lean_object* v_00_u03b1_14_, lean_object* v_r_15_, lean_object* v_inst_16_, lean_object* v_x_17_, lean_object* v_x_18_){
_start:
{
uint8_t v___x_19_; 
v___x_19_ = lp_mathlib_AntisymmRel_decidableRel___redArg(v_inst_16_, v_x_17_, v_x_18_);
return v___x_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AntisymmRel_decidableRel___boxed(lean_object* v_00_u03b1_20_, lean_object* v_r_21_, lean_object* v_inst_22_, lean_object* v_x_23_, lean_object* v_x_24_){
_start:
{
uint8_t v_res_25_; lean_object* v_r_26_; 
v_res_25_ = lp_mathlib_AntisymmRel_decidableRel(v_00_u03b1_20_, v_r_21_, v_inst_22_, v_x_23_, v_x_24_);
v_r_26_ = lean_box(v_res_25_);
return v_r_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0(lean_object* v_h_38_, lean_object* v_goal_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_){
_start:
{
lean_object* v___x_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; lean_object* v___x_49_; 
v___x_45_ = ((lean_object*)(lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___closed__5));
v___x_46_ = lean_unsigned_to_nat(1u);
v___x_47_ = lean_mk_empty_array_with_capacity(v___x_46_);
v___x_48_ = lean_array_push(v___x_47_, v_h_38_);
v___x_49_ = l_Lean_Meta_mkAppM(v___x_45_, v___x_48_, v___y_40_, v___y_41_, v___y_42_, v___y_43_);
if (lean_obj_tag(v___x_49_) == 0)
{
lean_object* v_a_50_; lean_object* v___x_51_; 
v_a_50_ = lean_ctor_get(v___x_49_, 0);
lean_inc(v_a_50_);
lean_dec_ref_known(v___x_49_, 1);
v___x_51_ = lp_batteries_Lean_MVarId_assignIfDefEq(v_goal_39_, v_a_50_, v___y_40_, v___y_41_, v___y_42_, v___y_43_);
return v___x_51_;
}
else
{
lean_object* v_a_52_; lean_object* v___x_54_; uint8_t v_isShared_55_; uint8_t v_isSharedCheck_59_; 
lean_dec(v_goal_39_);
v_a_52_ = lean_ctor_get(v___x_49_, 0);
v_isSharedCheck_59_ = !lean_is_exclusive(v___x_49_);
if (v_isSharedCheck_59_ == 0)
{
v___x_54_ = v___x_49_;
v_isShared_55_ = v_isSharedCheck_59_;
goto v_resetjp_53_;
}
else
{
lean_inc(v_a_52_);
lean_dec(v___x_49_);
v___x_54_ = lean_box(0);
v_isShared_55_ = v_isSharedCheck_59_;
goto v_resetjp_53_;
}
v_resetjp_53_:
{
lean_object* v___x_57_; 
if (v_isShared_55_ == 0)
{
v___x_57_ = v___x_54_;
goto v_reusejp_56_;
}
else
{
lean_object* v_reuseFailAlloc_58_; 
v_reuseFailAlloc_58_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_58_, 0, v_a_52_);
v___x_57_ = v_reuseFailAlloc_58_;
goto v_reusejp_56_;
}
v_reusejp_56_:
{
return v___x_57_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0___boxed(lean_object* v_h_60_, lean_object* v_goal_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_mathlib_Mathlib_Tactic_GCongr_exactAntisymmRelLeft___lam__0(v_h_60_, v_goal_61_, v___y_62_, v___y_63_, v___y_64_, v___y_65_);
lean_dec(v___y_65_);
lean_dec_ref(v___y_64_);
lean_dec(v___y_63_);
lean_dec_ref(v___y_62_);
return v_res_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AntisymmRel_setoid(lean_object* v_00_u03b1_70_, lean_object* v_r_71_, lean_object* v_inst_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lean_box(0);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toAntisymmetrization___redArg(lean_object* v_a_74_){
_start:
{
lean_inc(v_a_74_);
return v_a_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toAntisymmetrization___redArg___boxed(lean_object* v_a_75_){
_start:
{
lean_object* v_res_76_; 
v_res_76_ = lp_mathlib_toAntisymmetrization___redArg(v_a_75_);
lean_dec(v_a_75_);
return v_res_76_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toAntisymmetrization(lean_object* v_00_u03b1_77_, lean_object* v_r_78_, lean_object* v_inst_79_, lean_object* v_a_80_){
_start:
{
lean_inc(v_a_80_);
return v_a_80_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_toAntisymmetrization___boxed(lean_object* v_00_u03b1_81_, lean_object* v_r_82_, lean_object* v_inst_83_, lean_object* v_a_84_){
_start:
{
lean_object* v_res_85_; 
v_res_85_ = lp_mathlib_toAntisymmetrization(v_00_u03b1_81_, v_r_82_, v_inst_83_, v_a_84_);
lean_dec(v_a_84_);
return v_res_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAntisymmetrization___aux__1___redArg(lean_object* v_inst_86_){
_start:
{
lean_inc(v_inst_86_);
return v_inst_86_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAntisymmetrization___aux__1___redArg___boxed(lean_object* v_inst_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_mathlib_instInhabitedAntisymmetrization___aux__1___redArg(v_inst_87_);
lean_dec(v_inst_87_);
return v_res_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAntisymmetrization___aux__1(lean_object* v_00_u03b1_89_, lean_object* v_r_90_, lean_object* v_inst_91_, lean_object* v_inst_92_){
_start:
{
lean_inc(v_inst_92_);
return v_inst_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAntisymmetrization___aux__1___boxed(lean_object* v_00_u03b1_93_, lean_object* v_r_94_, lean_object* v_inst_95_, lean_object* v_inst_96_){
_start:
{
lean_object* v_res_97_; 
v_res_97_ = lp_mathlib_instInhabitedAntisymmetrization___aux__1(v_00_u03b1_93_, v_r_94_, v_inst_95_, v_inst_96_);
lean_dec(v_inst_96_);
return v_res_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAntisymmetrization___redArg(lean_object* v_inst_98_){
_start:
{
lean_inc(v_inst_98_);
return v_inst_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAntisymmetrization___redArg___boxed(lean_object* v_inst_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib_instInhabitedAntisymmetrization___redArg(v_inst_99_);
lean_dec(v_inst_99_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAntisymmetrization(lean_object* v_00_u03b1_101_, lean_object* v_r_102_, lean_object* v_inst_103_, lean_object* v_inst_104_){
_start:
{
lean_inc(v_inst_104_);
return v_inst_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instInhabitedAntisymmetrization___boxed(lean_object* v_00_u03b1_105_, lean_object* v_r_106_, lean_object* v_inst_107_, lean_object* v_inst_108_){
_start:
{
lean_object* v_res_109_; 
v_res_109_ = lp_mathlib_instInhabitedAntisymmetrization(v_00_u03b1_105_, v_r_106_, v_inst_107_, v_inst_108_);
lean_dec(v_inst_108_);
return v_res_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransLeAntisymmRel(lean_object* v_00_u03b1_110_, lean_object* v_inst_111_){
_start:
{
lean_object* v___x_112_; 
v___x_112_ = lean_box(0);
return v___x_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransLeAntisymmRel___boxed(lean_object* v_00_u03b1_113_, lean_object* v_inst_114_){
_start:
{
lean_object* v_res_115_; 
v_res_115_ = lp_mathlib_instTransLeAntisymmRel(v_00_u03b1_113_, v_inst_114_);
lean_dec_ref(v_inst_114_);
return v_res_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransAntisymmRelLe(lean_object* v_00_u03b1_116_, lean_object* v_inst_117_){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lean_box(0);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransAntisymmRelLe___boxed(lean_object* v_00_u03b1_119_, lean_object* v_inst_120_){
_start:
{
lean_object* v_res_121_; 
v_res_121_ = lp_mathlib_instTransAntisymmRelLe(v_00_u03b1_119_, v_inst_120_);
lean_dec_ref(v_inst_120_);
return v_res_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransLtAntisymmRelLe(lean_object* v_00_u03b1_122_, lean_object* v_inst_123_){
_start:
{
lean_object* v___x_124_; 
v___x_124_ = lean_box(0);
return v___x_124_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransLtAntisymmRelLe___boxed(lean_object* v_00_u03b1_125_, lean_object* v_inst_126_){
_start:
{
lean_object* v_res_127_; 
v_res_127_ = lp_mathlib_instTransLtAntisymmRelLe(v_00_u03b1_125_, v_inst_126_);
lean_dec_ref(v_inst_126_);
return v_res_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransAntisymmRelLeLt(lean_object* v_00_u03b1_128_, lean_object* v_inst_129_){
_start:
{
lean_object* v___x_130_; 
v___x_130_ = lean_box(0);
return v___x_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransAntisymmRelLeLt___boxed(lean_object* v_00_u03b1_131_, lean_object* v_inst_132_){
_start:
{
lean_object* v_res_133_; 
v_res_133_ = lp_mathlib_instTransAntisymmRelLeLt(v_00_u03b1_131_, v_inst_132_);
lean_dec_ref(v_inst_132_);
return v_res_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instPartialOrderAntisymmetrization(lean_object* v_00_u03b1_137_, lean_object* v_inst_138_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = ((lean_object*)(lp_mathlib_instPartialOrderAntisymmetrization___closed__0));
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instPartialOrderAntisymmetrization___boxed(lean_object* v_00_u03b1_140_, lean_object* v_inst_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_mathlib_instPartialOrderAntisymmetrization(v_00_u03b1_140_, v_inst_141_);
lean_dec_ref(v_inst_141_);
return v_res_142_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__0(lean_object* v_inst_143_, lean_object* v_x_144_, lean_object* v_x_145_){
_start:
{
lean_object* v___x_146_; uint8_t v___x_147_; 
v___x_146_ = lean_apply_2(v_inst_143_, v_x_144_, v_x_145_);
v___x_147_ = lean_unbox(v___x_146_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__0___boxed(lean_object* v_inst_148_, lean_object* v_x_149_, lean_object* v_x_150_){
_start:
{
uint8_t v_res_151_; lean_object* v_r_152_; 
v_res_151_ = lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__0(v_inst_148_, v_x_149_, v_x_150_);
v_r_152_ = lean_box(v_res_151_);
return v_r_152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__3(lean_object* v_inst_153_, lean_object* v_a_154_, lean_object* v_b_155_){
_start:
{
lean_object* v_this_156_; uint8_t v___x_157_; 
lean_inc(v_b_155_);
lean_inc(v_a_154_);
v_this_156_ = lean_apply_2(v_inst_153_, v_a_154_, v_b_155_);
v___x_157_ = lean_unbox(v_this_156_);
if (v___x_157_ == 0)
{
lean_dec(v_b_155_);
return v_a_154_;
}
else
{
lean_dec(v_a_154_);
return v_b_155_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__1(lean_object* v_inst_158_, lean_object* v_a_159_, lean_object* v_b_160_){
_start:
{
lean_object* v_this_161_; uint8_t v___x_162_; 
lean_inc(v_b_160_);
lean_inc(v_a_159_);
v_this_161_ = lean_apply_2(v_inst_158_, v_a_159_, v_b_160_);
v___x_162_ = lean_unbox(v_this_161_);
if (v___x_162_ == 0)
{
lean_dec(v_a_159_);
return v_b_160_;
}
else
{
lean_dec(v_b_160_);
return v_a_159_;
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__2(lean_object* v_inst_163_, lean_object* v___f_164_, lean_object* v_a_165_, lean_object* v_b_166_){
_start:
{
lean_object* v_this_167_; uint8_t v___x_168_; 
lean_inc(v_b_166_);
lean_inc(v_a_165_);
v_this_167_ = lean_apply_2(v_inst_163_, v_a_165_, v_b_166_);
v___x_168_ = lean_unbox(v_this_167_);
if (v___x_168_ == 0)
{
uint8_t v___x_169_; 
v___x_169_ = lp_mathlib_decidableEqOfDecidableLE___redArg(v___f_164_, v_a_165_, v_b_166_);
if (v___x_169_ == 0)
{
uint8_t v___x_170_; 
v___x_170_ = 2;
return v___x_170_;
}
else
{
uint8_t v___x_171_; 
v___x_171_ = 1;
return v___x_171_;
}
}
else
{
uint8_t v___x_172_; 
lean_dec(v_b_166_);
lean_dec(v_a_165_);
lean_dec_ref(v___f_164_);
v___x_172_ = 0;
return v___x_172_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__2___boxed(lean_object* v_inst_173_, lean_object* v___f_174_, lean_object* v_a_175_, lean_object* v_b_176_){
_start:
{
uint8_t v_res_177_; lean_object* v_r_178_; 
v_res_177_ = lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__2(v_inst_173_, v___f_174_, v_a_175_, v_b_176_);
v_r_178_ = lean_box(v_res_177_);
return v_r_178_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg(lean_object* v_inst_179_, lean_object* v_inst_180_, lean_object* v_inst_181_){
_start:
{
lean_object* v___f_182_; lean_object* v___f_183_; lean_object* v___f_184_; lean_object* v___f_185_; lean_object* v___f_186_; lean_object* v___x_187_; lean_object* v___x_188_; lean_object* v___x_189_; 
lean_inc_ref(v_inst_181_);
v___f_182_ = lean_alloc_closure((void*)(lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_182_, 0, v_inst_181_);
lean_inc_ref_n(v_inst_180_, 2);
v___f_183_ = lean_alloc_closure((void*)(lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_183_, 0, v_inst_180_);
v___f_184_ = lean_alloc_closure((void*)(lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__3), 3, 1);
lean_closure_set(v___f_184_, 0, v_inst_180_);
v___f_185_ = lean_alloc_closure((void*)(lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__1), 3, 1);
lean_closure_set(v___f_185_, 0, v_inst_180_);
lean_inc_ref_n(v___f_183_, 2);
v___f_186_ = lean_alloc_closure((void*)(lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___lam__2___boxed), 4, 2);
lean_closure_set(v___f_186_, 0, v_inst_181_);
lean_closure_set(v___f_186_, 1, v___f_183_);
v___x_187_ = lp_mathlib_instPartialOrderAntisymmetrization(lean_box(0), v_inst_179_);
lean_inc_ref(v___x_187_);
v___x_188_ = lean_alloc_closure((void*)(lp_mathlib_decidableEqOfDecidableLE___boxed), 5, 3);
lean_closure_set(v___x_188_, 0, lean_box(0));
lean_closure_set(v___x_188_, 1, v___x_187_);
lean_closure_set(v___x_188_, 2, v___f_183_);
v___x_189_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v___x_189_, 0, v___x_187_);
lean_ctor_set(v___x_189_, 1, v___f_185_);
lean_ctor_set(v___x_189_, 2, v___f_184_);
lean_ctor_set(v___x_189_, 3, v___f_186_);
lean_ctor_set(v___x_189_, 4, v___f_183_);
lean_ctor_set(v___x_189_, 5, v___x_188_);
lean_ctor_set(v___x_189_, 6, v___f_182_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg___boxed(lean_object* v_inst_190_, lean_object* v_inst_191_, lean_object* v_inst_192_){
_start:
{
lean_object* v_res_193_; 
v_res_193_ = lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg(v_inst_190_, v_inst_191_, v_inst_192_);
lean_dec_ref(v_inst_190_);
return v_res_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal(lean_object* v_00_u03b1_194_, lean_object* v_inst_195_, lean_object* v_inst_196_, lean_object* v_inst_197_, lean_object* v_inst_198_){
_start:
{
lean_object* v___x_199_; 
v___x_199_ = lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___redArg(v_inst_195_, v_inst_196_, v_inst_197_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal___boxed(lean_object* v_00_u03b1_200_, lean_object* v_inst_201_, lean_object* v_inst_202_, lean_object* v_inst_203_, lean_object* v_inst_204_){
_start:
{
lean_object* v_res_205_; 
v_res_205_ = lp_mathlib_instLinearOrderAntisymmetrizationLeOfDecidableLEOfDecidableLTOfTotal(v_00_u03b1_200_, v_inst_201_, v_inst_202_, v_inst_203_, v_inst_204_);
lean_dec_ref(v_inst_201_);
return v_res_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_antisymmetrization___redArg___lam__0(lean_object* v_f_206_, lean_object* v___y_207_){
_start:
{
lean_object* v___x_208_; 
v___x_208_ = lean_apply_1(v_f_206_, v___y_207_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_antisymmetrization___redArg(lean_object* v_f_209_){
_start:
{
lean_object* v___f_210_; lean_object* v___x_211_; lean_object* v___x_212_; 
v___f_210_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_antisymmetrization___redArg___lam__0), 2, 1);
lean_closure_set(v___f_210_, 0, v_f_209_);
v___x_211_ = lean_box(0);
v___x_212_ = lean_alloc_closure((void*)(lp_mathlib_Quotient_map_x27), 7, 6);
lean_closure_set(v___x_212_, 0, lean_box(0));
lean_closure_set(v___x_212_, 1, lean_box(0));
lean_closure_set(v___x_212_, 2, v___x_211_);
lean_closure_set(v___x_212_, 3, v___x_211_);
lean_closure_set(v___x_212_, 4, v___f_210_);
lean_closure_set(v___x_212_, 5, lean_box(0));
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_antisymmetrization(lean_object* v_00_u03b1_213_, lean_object* v_00_u03b2_214_, lean_object* v_inst_215_, lean_object* v_inst_216_, lean_object* v_f_217_){
_start:
{
lean_object* v___x_218_; 
v___x_218_ = lp_mathlib_OrderHom_antisymmetrization___redArg(v_f_217_);
return v___x_218_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderHom_antisymmetrization___boxed(lean_object* v_00_u03b1_219_, lean_object* v_00_u03b2_220_, lean_object* v_inst_221_, lean_object* v_inst_222_, lean_object* v_f_223_){
_start:
{
lean_object* v_res_224_; 
v_res_224_ = lp_mathlib_OrderHom_antisymmetrization(v_00_u03b1_219_, v_00_u03b2_220_, v_inst_221_, v_inst_222_, v_f_223_);
lean_dec_ref(v_inst_222_);
lean_dec_ref(v_inst_221_);
return v_res_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_dualAntisymmetrization(lean_object* v_00_u03b1_231_, lean_object* v_inst_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = ((lean_object*)(lp_mathlib_OrderIso_dualAntisymmetrization___closed__2));
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_dualAntisymmetrization___boxed(lean_object* v_00_u03b1_234_, lean_object* v_inst_235_){
_start:
{
lean_object* v_res_236_; 
v_res_236_ = lp_mathlib_OrderIso_dualAntisymmetrization(v_00_u03b1_234_, v_inst_235_);
lean_dec_ref(v_inst_235_);
return v_res_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransSymmGenLeAntisymmRel(lean_object* v_00_u03b1_237_, lean_object* v_inst_238_){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lean_box(0);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransSymmGenLeAntisymmRel___boxed(lean_object* v_00_u03b1_240_, lean_object* v_inst_241_){
_start:
{
lean_object* v_res_242_; 
v_res_242_ = lp_mathlib_instTransSymmGenLeAntisymmRel(v_00_u03b1_240_, v_inst_241_);
lean_dec_ref(v_inst_241_);
return v_res_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransAntisymmRelLeSymmGen(lean_object* v_00_u03b1_243_, lean_object* v_inst_244_){
_start:
{
lean_object* v___x_245_; 
v___x_245_ = lean_box(0);
return v___x_245_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTransAntisymmRelLeSymmGen___boxed(lean_object* v_00_u03b1_246_, lean_object* v_inst_247_){
_start:
{
lean_object* v_res_248_; 
v_res_248_ = lp_mathlib_instTransAntisymmRelLeSymmGen(v_00_u03b1_246_, v_inst_247_);
lean_dec_ref(v_inst_247_);
return v_res_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Antisymmetrization_prodEquiv___lam__0(lean_object* v_ab_249_){
_start:
{
lean_object* v_fst_250_; lean_object* v_snd_251_; lean_object* v___x_253_; uint8_t v_isShared_254_; uint8_t v_isSharedCheck_258_; 
v_fst_250_ = lean_ctor_get(v_ab_249_, 0);
v_snd_251_ = lean_ctor_get(v_ab_249_, 1);
v_isSharedCheck_258_ = !lean_is_exclusive(v_ab_249_);
if (v_isSharedCheck_258_ == 0)
{
v___x_253_ = v_ab_249_;
v_isShared_254_ = v_isSharedCheck_258_;
goto v_resetjp_252_;
}
else
{
lean_inc(v_snd_251_);
lean_inc(v_fst_250_);
lean_dec(v_ab_249_);
v___x_253_ = lean_box(0);
v_isShared_254_ = v_isSharedCheck_258_;
goto v_resetjp_252_;
}
v_resetjp_252_:
{
lean_object* v___x_256_; 
if (v_isShared_254_ == 0)
{
v___x_256_ = v___x_253_;
goto v_reusejp_255_;
}
else
{
lean_object* v_reuseFailAlloc_257_; 
v_reuseFailAlloc_257_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_257_, 0, v_fst_250_);
lean_ctor_set(v_reuseFailAlloc_257_, 1, v_snd_251_);
v___x_256_ = v_reuseFailAlloc_257_;
goto v_reusejp_255_;
}
v_reusejp_255_:
{
return v___x_256_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Antisymmetrization_prodEquiv___lam__1(lean_object* v_a_259_, lean_object* v_b_260_){
_start:
{
lean_object* v___x_261_; 
v___x_261_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_261_, 0, v_a_259_);
lean_ctor_set(v___x_261_, 1, v_b_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Antisymmetrization_prodEquiv(lean_object* v_00_u03b1_275_, lean_object* v_00_u03b2_276_, lean_object* v_inst_277_, lean_object* v_inst_278_){
_start:
{
lean_object* v___x_279_; 
v___x_279_ = ((lean_object*)(lp_mathlib_Antisymmetrization_prodEquiv___closed__5));
return v___x_279_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Antisymmetrization_prodEquiv___boxed(lean_object* v_00_u03b1_280_, lean_object* v_00_u03b2_281_, lean_object* v_inst_282_, lean_object* v_inst_283_){
_start:
{
lean_object* v_res_284_; 
v_res_284_ = lp_mathlib_Antisymmetrization_prodEquiv(v_00_u03b1_280_, v_00_u03b2_281_, v_inst_282_, v_inst_283_);
lean_dec_ref(v_inst_283_);
lean_dec_ref(v_inst_282_);
return v_res_284_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Relation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Tauto(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Antisymmetrization(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Relation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Tauto(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Antisymmetrization(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Logic_Relation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Tauto(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Antisymmetrization(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Relation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Tauto(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Antisymmetrization(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Antisymmetrization(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Antisymmetrization(builtin);
}
#ifdef __cplusplus
}
#endif
