// Lean compiler output
// Module: Fundamentals.Spec.Poseidon2.Generic
// Imports: public import Init public meta import Init public import Init public import Fundamentals.Spec.FieldOps
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
lean_object* l_List_reverse___redArg(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l_List_get_x3fInternal___redArg(lean_object*, lean_object*);
lean_object* l_Array_ofFn___redArg(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_WIDTH;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_RATE;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfFields___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfFields___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfFields___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfFields(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfNats___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfNats___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfNats___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfNats(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4OfFields___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4OfFields(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_sbox___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_sbox(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_addRoundConstants___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_addRoundConstants___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_addRoundConstants___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_addRoundConstants(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Fundamentals_Poseidon2_Generic_applySBoxToAll_spec__0___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Fundamentals_Poseidon2_Generic_applySBoxToAll_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applySBoxToAll___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applySBoxToAll(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Fundamentals_Poseidon2_Generic_applySBoxToAll_spec__0(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Fundamentals_Poseidon2_Generic_applySBoxToAll_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applyMat4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applyMat4(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalLinearLayer___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalLinearLayer(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_divByTwoPow___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_divByTwoPow(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_sumState_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_sumState___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_sumState(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_sumState_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalLinearLayer___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalLinearLayer(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalRound___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalRound(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRound___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRound___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRound___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRound(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(448208942) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(572403254) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1200041953) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(256487465) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__2_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__3_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1425273457) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__3_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(977184635) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__4_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__5_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(360728943) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__6_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(89648862) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__6_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__7 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__7_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(747903232) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__7_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__8 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__8_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1925334750) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__8_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__9 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__9_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1932426223) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__9_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__10 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__10_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(588815102) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__10_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__11 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__11_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1796380621) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__11_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__12 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__12_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1621102414) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__12_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__13 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__13_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1185780729) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__13_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__14 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__14_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1774958255) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__14_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__15 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__15_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1348741381) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__16 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__16_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1391160226) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__16_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__17 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__17_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1827584334) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__17_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__18 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__18_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1963448500) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__18_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__19 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__19_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(456061748) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__19_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__20 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__20_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(333311454) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__20_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__21 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__21_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(32019634) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__21_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__22 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__22_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1065944411) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__22_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__23 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__23_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(86189867) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__23_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__24 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__24_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1536873262) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__24_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__25 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__25_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(889997530) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__25_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__26 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__26_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(646827752) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__26_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__27 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__27_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(547326025) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__27_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__28 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__28_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(953948096) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__28_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__29 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__29_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(944884184) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__29_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__30 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__30_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1215789478) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__30_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__31 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__31_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(25886717) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__32 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__32_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(704371273) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__32_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__33 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__33_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(256335831) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__33_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__34 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__34_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(889367200) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__34_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__35 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__35_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(763177444) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__35_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__36 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__36_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(669703072) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__36_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__37 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__37_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(159256268) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__37_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__38 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__38_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(222820492) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__38_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__39 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__39_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(573163527) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__39_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__40 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__40_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(359890076) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__40_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__41 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__41_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1050669594) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__41_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__42 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__42_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1988915530) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__42_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__43 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__43_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(79691676) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__43_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__44 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__44_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1763866748) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__44_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__45 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__45_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(104111868) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__45_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__46 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__46_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(88424255) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__46_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__47 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__47_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(506915399) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__48 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__48_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(751272725) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__48_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__49 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__49_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(36972490) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__49_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__50 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__50_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(754867989) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__50_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__51 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__51_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1212119315) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__51_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__52 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__52_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(5351995) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__52_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__53 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__53_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1046925956) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__53_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__54 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__54_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(401996362) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__54_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__55 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__55_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(344647910) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__55_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__56 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__56_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1851729162) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__56_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__57 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__57_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1053320300) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__57_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__58 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__58_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(777848065) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__58_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__59 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__59_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1384520381) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__59_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__60 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__60_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(454499742) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__60_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__61 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__61_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1833211857) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__61_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__62 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__62_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(51754520) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__62_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__63 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__63_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__63_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__64 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__64_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__47_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__64_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__65 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__65_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__31_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__65_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__66 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__66_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__15_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__66_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__67 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__67_value;
LEAN_EXPORT const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat___closed__67_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(542259047) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1781349648) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(549271463) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(744004766) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__2_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__3_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1923111974) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__3_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1178602734) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__4_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__5_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1886297254) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__6_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(183414327) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__6_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__7 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__7_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(528965731) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__7_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__8 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__8_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(321330495) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__8_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__9 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__9_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1702593455) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__9_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__10 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__10_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1700391016) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__10_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__11 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__11_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1990744480) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__11_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__12 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__12_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1502529704) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__12_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__13 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__13_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1870549801) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__13_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__14 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__14_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1922082829) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__14_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__15 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__15_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(904680097) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__16 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__16_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1235297680) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__16_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__17 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__17_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1520185679) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__17_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__18 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__18_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(386838401) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__18_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__19 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__19_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(43203215) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__19_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__20 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__20_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(301835475) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__20_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__21 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__21_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1843351545) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__21_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__22 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__22_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1641600456) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__22_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__23 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__23_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1338992758) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__23_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__24 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__24_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(742828095) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__24_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__25 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__25_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1226350925) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__25_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__26 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__26_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1558555932) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__26_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__27 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__27_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(340311124) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__27_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__28 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__28_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(503426110) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__28_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__29 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__29_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(715456982) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__29_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__30 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__30_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1536158148) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__30_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__31 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__31_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1250800299) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__32 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__32_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1172395131) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__32_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__33 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__33_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(695185447) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__33_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__34 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__34_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(507181886) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__34_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__35 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__35_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1530706910) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__35_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__36 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__36_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1867507480) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__36_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__37 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__37_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1681743681) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__37_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__38 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__38_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1358282574) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__38_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__39 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__39_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1979521776) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__39_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__40 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__40_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(213827818) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__40_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__41 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__41_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(107190701) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__41_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__42 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__42_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(532844013) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__42_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__43 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__43_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(247083962) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__43_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__44 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__44_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(913384905) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__44_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__45 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__45_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1581784677) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__45_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__46 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__46_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1491801617) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__46_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__47 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__47_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1619482808) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__48 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__48_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1015795079) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__48_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__49 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__49_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1682806907) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__49_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__50 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__50_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1066647396) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__50_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__51 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__51_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1897591937) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__51_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__52 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__52_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1417398904) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__52_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__53 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__53_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1342765939) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__53_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__54 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__54_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(890862029) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__54_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__55 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__55_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(744214112) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__55_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__56 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__56_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(59414691) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__56_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__57 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__57_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(108246855) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__57_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__58 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__58_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1404253825) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__58_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__59 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__59_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(494676004) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__59_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__60 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__60_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(498481458) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__60_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__61 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__61_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(817684387) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__61_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__62 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__62_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1503161625) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__62_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__63 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__63_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__63_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__64 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__64_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__47_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__64_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__65 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__65_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__31_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__65_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__66 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__66_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__15_value),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__66_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__67 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__67_value;
LEAN_EXPORT const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat___closed__67_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(605745517) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__0 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__0_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(212616710) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__0_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__1 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__1_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(557776863) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__1_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__2 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__2_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(786108885) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__2_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__3 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__3_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(190525218) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__3_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__4 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__4_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1009879353) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__4_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__5 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__5_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1629555936) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__5_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__6 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__6_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1311448267) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__6_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__7 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__7_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(311365592) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__7_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__8 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__8_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(422793067) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__8_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__9 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__9_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(945325693) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__9_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__10 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__10_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1765533241) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__10_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__11 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__11_value;
static const lean_ctor_object lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(1518359488) << 1) | 1)),((lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__11_value)}};
static const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__12 = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__12_value;
LEAN_EXPORT const lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat = (const lean_object*)&lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat___closed__12_value;
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_initialExternalRounds_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_Generic_initialExternalRounds_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_initialExternalRounds___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_initialExternalRounds(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_Generic_initialExternalRounds_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_initialExternalRounds_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_Generic_internalRounds_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_internalRounds_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRounds___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRounds(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_Generic_internalRounds_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_internalRounds_spec__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_finalExternalRounds___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_finalExternalRounds(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_permute___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_permute(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressDigest___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressDigest(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_WIDTH(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lean_unsigned_to_nat(16u);
return v___x_1_;
}
}
static lean_object* _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_RATE(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(8u);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(lean_object* v_fo_3_, lean_object* v_state_4_, lean_object* v_idx_5_){
_start:
{
lean_object* v___x_6_; lean_object* v___x_7_; 
v___x_6_ = lean_array_to_list(v_state_4_);
v___x_7_ = l_List_get_x3fInternal___redArg(v___x_6_, v_idx_5_);
lean_dec(v___x_6_);
if (lean_obj_tag(v___x_7_) == 0)
{
lean_object* v_toRingOps_8_; lean_object* v_toSemiringOps_9_; lean_object* v_zero_10_; 
v_toRingOps_8_ = lean_ctor_get(v_fo_3_, 0);
v_toSemiringOps_9_ = lean_ctor_get(v_toRingOps_8_, 0);
v_zero_10_ = lean_ctor_get(v_toSemiringOps_9_, 0);
lean_inc(v_zero_10_);
return v_zero_10_;
}
else
{
lean_object* v_val_11_; 
v_val_11_ = lean_ctor_get(v___x_7_, 0);
lean_inc(v_val_11_);
lean_dec_ref_known(v___x_7_, 1);
return v_val_11_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg___boxed(lean_object* v_fo_12_, lean_object* v_state_13_, lean_object* v_idx_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_12_, v_state_13_, v_idx_14_);
lean_dec_ref(v_fo_12_);
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt(lean_object* v_K_16_, lean_object* v_fo_17_, lean_object* v_state_18_, lean_object* v_idx_19_){
_start:
{
lean_object* v___x_20_; 
v___x_20_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_17_, v_state_18_, v_idx_19_);
return v___x_20_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___boxed(lean_object* v_K_21_, lean_object* v_fo_22_, lean_object* v_state_23_, lean_object* v_idx_24_){
_start:
{
lean_object* v_res_25_; 
v_res_25_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt(v_K_21_, v_fo_22_, v_state_23_, v_idx_24_);
lean_dec_ref(v_fo_22_);
return v_res_25_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfFields___redArg___lam__0(lean_object* v_xs_26_, lean_object* v_fo_27_, lean_object* v_idx_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = l_List_get_x3fInternal___redArg(v_xs_26_, v_idx_28_);
if (lean_obj_tag(v___x_29_) == 0)
{
lean_object* v_toRingOps_30_; lean_object* v_toSemiringOps_31_; lean_object* v_zero_32_; 
v_toRingOps_30_ = lean_ctor_get(v_fo_27_, 0);
v_toSemiringOps_31_ = lean_ctor_get(v_toRingOps_30_, 0);
v_zero_32_ = lean_ctor_get(v_toSemiringOps_31_, 0);
lean_inc(v_zero_32_);
return v_zero_32_;
}
else
{
lean_object* v_val_33_; 
v_val_33_ = lean_ctor_get(v___x_29_, 0);
lean_inc(v_val_33_);
lean_dec_ref_known(v___x_29_, 1);
return v_val_33_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfFields___redArg___lam__0___boxed(lean_object* v_xs_34_, lean_object* v_fo_35_, lean_object* v_idx_36_){
_start:
{
lean_object* v_res_37_; 
v_res_37_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfFields___redArg___lam__0(v_xs_34_, v_fo_35_, v_idx_36_);
lean_dec_ref(v_fo_35_);
lean_dec(v_xs_34_);
return v_res_37_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfFields___redArg(lean_object* v_fo_38_, lean_object* v_xs_39_){
_start:
{
lean_object* v___f_40_; lean_object* v___x_41_; lean_object* v___x_42_; 
v___f_40_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfFields___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_40_, 0, v_xs_39_);
lean_closure_set(v___f_40_, 1, v_fo_38_);
v___x_41_ = lean_unsigned_to_nat(16u);
v___x_42_ = l_Array_ofFn___redArg(v___x_41_, v___f_40_);
return v___x_42_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfFields(lean_object* v_K_43_, lean_object* v_fo_44_, lean_object* v_xs_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfFields___redArg(v_fo_44_, v_xs_45_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfNats___redArg___lam__0(lean_object* v_fo_47_, lean_object* v_ns_48_, lean_object* v_idx_49_){
_start:
{
lean_object* v_toRingOps_50_; lean_object* v_toSemiringOps_51_; lean_object* v_natCast_52_; lean_object* v___x_53_; 
v_toRingOps_50_ = lean_ctor_get(v_fo_47_, 0);
lean_inc_ref(v_toRingOps_50_);
lean_dec_ref(v_fo_47_);
v_toSemiringOps_51_ = lean_ctor_get(v_toRingOps_50_, 0);
lean_inc_ref(v_toSemiringOps_51_);
lean_dec_ref(v_toRingOps_50_);
v_natCast_52_ = lean_ctor_get(v_toSemiringOps_51_, 2);
lean_inc(v_natCast_52_);
lean_dec_ref(v_toSemiringOps_51_);
v___x_53_ = l_List_get_x3fInternal___redArg(v_ns_48_, v_idx_49_);
if (lean_obj_tag(v___x_53_) == 0)
{
lean_object* v___x_54_; lean_object* v___x_55_; 
v___x_54_ = lean_unsigned_to_nat(0u);
v___x_55_ = lean_apply_1(v_natCast_52_, v___x_54_);
return v___x_55_;
}
else
{
lean_object* v_val_56_; lean_object* v___x_57_; 
v_val_56_ = lean_ctor_get(v___x_53_, 0);
lean_inc(v_val_56_);
lean_dec_ref_known(v___x_53_, 1);
v___x_57_ = lean_apply_1(v_natCast_52_, v_val_56_);
return v___x_57_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfNats___redArg___lam__0___boxed(lean_object* v_fo_58_, lean_object* v_ns_59_, lean_object* v_idx_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfNats___redArg___lam__0(v_fo_58_, v_ns_59_, v_idx_60_);
lean_dec(v_ns_59_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfNats___redArg(lean_object* v_fo_62_, lean_object* v_ns_63_){
_start:
{
lean_object* v___f_64_; lean_object* v___x_65_; lean_object* v___x_66_; 
v___f_64_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfNats___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_64_, 0, v_fo_62_);
lean_closure_set(v___f_64_, 1, v_ns_63_);
v___x_65_ = lean_unsigned_to_nat(16u);
v___x_66_ = l_Array_ofFn___redArg(v___x_65_, v___f_64_);
return v___x_66_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfNats(lean_object* v_K_67_, lean_object* v_fo_68_, lean_object* v_ns_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfNats___redArg(v_fo_68_, v_ns_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(lean_object* v_fo_71_, lean_object* v_x_72_, lean_object* v_idx_73_){
_start:
{
lean_object* v___x_74_; lean_object* v___x_75_; 
v___x_74_ = lean_array_to_list(v_x_72_);
v___x_75_ = l_List_get_x3fInternal___redArg(v___x_74_, v_idx_73_);
lean_dec(v___x_74_);
if (lean_obj_tag(v___x_75_) == 0)
{
lean_object* v_toRingOps_76_; lean_object* v_toSemiringOps_77_; lean_object* v_zero_78_; 
v_toRingOps_76_ = lean_ctor_get(v_fo_71_, 0);
v_toSemiringOps_77_ = lean_ctor_get(v_toRingOps_76_, 0);
v_zero_78_ = lean_ctor_get(v_toSemiringOps_77_, 0);
lean_inc(v_zero_78_);
return v_zero_78_;
}
else
{
lean_object* v_val_79_; 
v_val_79_ = lean_ctor_get(v___x_75_, 0);
lean_inc(v_val_79_);
lean_dec_ref_known(v___x_75_, 1);
return v_val_79_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg___boxed(lean_object* v_fo_80_, lean_object* v_x_81_, lean_object* v_idx_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_80_, v_x_81_, v_idx_82_);
lean_dec_ref(v_fo_80_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At(lean_object* v_K_84_, lean_object* v_fo_85_, lean_object* v_x_86_, lean_object* v_idx_87_){
_start:
{
lean_object* v___x_88_; 
v___x_88_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_85_, v_x_86_, v_idx_87_);
return v___x_88_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___boxed(lean_object* v_K_89_, lean_object* v_fo_90_, lean_object* v_x_91_, lean_object* v_idx_92_){
_start:
{
lean_object* v_res_93_; 
v_res_93_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At(v_K_89_, v_fo_90_, v_x_91_, v_idx_92_);
lean_dec_ref(v_fo_90_);
return v_res_93_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4OfFields___redArg(lean_object* v_fo_94_, lean_object* v_xs_95_){
_start:
{
lean_object* v___f_96_; lean_object* v___x_97_; lean_object* v___x_98_; 
v___f_96_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfFields___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_96_, 0, v_xs_95_);
lean_closure_set(v___f_96_, 1, v_fo_94_);
v___x_97_ = lean_unsigned_to_nat(4u);
v___x_98_ = l_Array_ofFn___redArg(v___x_97_, v___f_96_);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4OfFields(lean_object* v_K_99_, lean_object* v_fo_100_, lean_object* v_xs_101_){
_start:
{
lean_object* v___x_102_; 
v___x_102_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4OfFields___redArg(v_fo_100_, v_xs_101_);
return v___x_102_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_sbox___redArg(lean_object* v_fo_103_, lean_object* v_x_104_){
_start:
{
lean_object* v_toRingOps_105_; lean_object* v_toSemiringOps_106_; lean_object* v_pow_107_; lean_object* v___x_108_; lean_object* v___x_109_; 
v_toRingOps_105_ = lean_ctor_get(v_fo_103_, 0);
lean_inc_ref(v_toRingOps_105_);
lean_dec_ref(v_fo_103_);
v_toSemiringOps_106_ = lean_ctor_get(v_toRingOps_105_, 0);
lean_inc_ref(v_toSemiringOps_106_);
lean_dec_ref(v_toRingOps_105_);
v_pow_107_ = lean_ctor_get(v_toSemiringOps_106_, 5);
lean_inc(v_pow_107_);
lean_dec_ref(v_toSemiringOps_106_);
v___x_108_ = lean_unsigned_to_nat(7u);
v___x_109_ = lean_apply_2(v_pow_107_, v_x_104_, v___x_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_sbox(lean_object* v_K_110_, lean_object* v_fo_111_, lean_object* v_x_112_){
_start:
{
lean_object* v___x_113_; 
v___x_113_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_sbox___redArg(v_fo_111_, v_x_112_);
return v___x_113_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_addRoundConstants___redArg___lam__0(lean_object* v_fo_114_, lean_object* v_state_115_, lean_object* v_rc_116_, lean_object* v_idx_117_){
_start:
{
lean_object* v_toRingOps_118_; lean_object* v_toSemiringOps_119_; lean_object* v_add_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
v_toRingOps_118_ = lean_ctor_get(v_fo_114_, 0);
lean_inc_ref(v_toRingOps_118_);
lean_dec_ref(v_fo_114_);
v_toSemiringOps_119_ = lean_ctor_get(v_toRingOps_118_, 0);
lean_inc_ref(v_toSemiringOps_119_);
lean_dec_ref(v_toRingOps_118_);
v_add_120_ = lean_ctor_get(v_toSemiringOps_119_, 3);
lean_inc(v_add_120_);
lean_dec_ref(v_toSemiringOps_119_);
v___x_121_ = lean_array_fget_borrowed(v_state_115_, v_idx_117_);
v___x_122_ = lean_array_fget_borrowed(v_rc_116_, v_idx_117_);
lean_inc(v___x_122_);
lean_inc(v___x_121_);
v___x_123_ = lean_apply_2(v_add_120_, v___x_121_, v___x_122_);
return v___x_123_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_addRoundConstants___redArg___lam__0___boxed(lean_object* v_fo_124_, lean_object* v_state_125_, lean_object* v_rc_126_, lean_object* v_idx_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_addRoundConstants___redArg___lam__0(v_fo_124_, v_state_125_, v_rc_126_, v_idx_127_);
lean_dec(v_idx_127_);
lean_dec_ref(v_rc_126_);
lean_dec_ref(v_state_125_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_addRoundConstants___redArg(lean_object* v_fo_129_, lean_object* v_rc_130_, lean_object* v_state_131_){
_start:
{
lean_object* v___f_132_; lean_object* v___x_133_; lean_object* v___x_134_; 
v___f_132_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_addRoundConstants___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_132_, 0, v_fo_129_);
lean_closure_set(v___f_132_, 1, v_state_131_);
lean_closure_set(v___f_132_, 2, v_rc_130_);
v___x_133_ = lean_unsigned_to_nat(16u);
v___x_134_ = l_Array_ofFn___redArg(v___x_133_, v___f_132_);
return v___x_134_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_addRoundConstants(lean_object* v_K_135_, lean_object* v_fo_136_, lean_object* v_rc_137_, lean_object* v_state_138_){
_start:
{
lean_object* v___x_139_; 
v___x_139_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_addRoundConstants___redArg(v_fo_136_, v_rc_137_, v_state_138_);
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Fundamentals_Poseidon2_Generic_applySBoxToAll_spec__0___redArg(lean_object* v_fo_140_, size_t v_sz_141_, size_t v_i_142_, lean_object* v_bs_143_){
_start:
{
uint8_t v___x_144_; 
v___x_144_ = lean_usize_dec_lt(v_i_142_, v_sz_141_);
if (v___x_144_ == 0)
{
lean_dec_ref(v_fo_140_);
return v_bs_143_;
}
else
{
lean_object* v_v_145_; lean_object* v___x_146_; lean_object* v_bs_x27_147_; lean_object* v___x_148_; size_t v___x_149_; size_t v___x_150_; lean_object* v___x_151_; 
v_v_145_ = lean_array_uget(v_bs_143_, v_i_142_);
v___x_146_ = lean_unsigned_to_nat(0u);
v_bs_x27_147_ = lean_array_uset(v_bs_143_, v_i_142_, v___x_146_);
lean_inc_ref(v_fo_140_);
v___x_148_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_sbox___redArg(v_fo_140_, v_v_145_);
v___x_149_ = ((size_t)1ULL);
v___x_150_ = lean_usize_add(v_i_142_, v___x_149_);
v___x_151_ = lean_array_uset(v_bs_x27_147_, v_i_142_, v___x_148_);
v_i_142_ = v___x_150_;
v_bs_143_ = v___x_151_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Fundamentals_Poseidon2_Generic_applySBoxToAll_spec__0___redArg___boxed(lean_object* v_fo_153_, lean_object* v_sz_154_, lean_object* v_i_155_, lean_object* v_bs_156_){
_start:
{
size_t v_sz_boxed_157_; size_t v_i_boxed_158_; lean_object* v_res_159_; 
v_sz_boxed_157_ = lean_unbox_usize(v_sz_154_);
lean_dec(v_sz_154_);
v_i_boxed_158_ = lean_unbox_usize(v_i_155_);
lean_dec(v_i_155_);
v_res_159_ = lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Fundamentals_Poseidon2_Generic_applySBoxToAll_spec__0___redArg(v_fo_153_, v_sz_boxed_157_, v_i_boxed_158_, v_bs_156_);
return v_res_159_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applySBoxToAll___redArg(lean_object* v_fo_160_, lean_object* v_state_161_){
_start:
{
size_t v_sz_162_; size_t v___x_163_; lean_object* v___x_164_; 
v_sz_162_ = lean_array_size(v_state_161_);
v___x_163_ = ((size_t)0ULL);
v___x_164_ = lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Fundamentals_Poseidon2_Generic_applySBoxToAll_spec__0___redArg(v_fo_160_, v_sz_162_, v___x_163_, v_state_161_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applySBoxToAll(lean_object* v_K_165_, lean_object* v_fo_166_, lean_object* v_state_167_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applySBoxToAll___redArg(v_fo_166_, v_state_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Fundamentals_Poseidon2_Generic_applySBoxToAll_spec__0(lean_object* v_K_169_, lean_object* v_fo_170_, size_t v_sz_171_, size_t v_i_172_, lean_object* v_bs_173_){
_start:
{
lean_object* v___x_174_; 
v___x_174_ = lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Fundamentals_Poseidon2_Generic_applySBoxToAll_spec__0___redArg(v_fo_170_, v_sz_171_, v_i_172_, v_bs_173_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Fundamentals_Poseidon2_Generic_applySBoxToAll_spec__0___boxed(lean_object* v_K_175_, lean_object* v_fo_176_, lean_object* v_sz_177_, lean_object* v_i_178_, lean_object* v_bs_179_){
_start:
{
size_t v_sz_boxed_180_; size_t v_i_boxed_181_; lean_object* v_res_182_; 
v_sz_boxed_180_ = lean_unbox_usize(v_sz_177_);
lean_dec(v_sz_177_);
v_i_boxed_181_ = lean_unbox_usize(v_i_178_);
lean_dec(v_i_178_);
v_res_182_ = lp_swirl_x2dfv___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00Fundamentals_Poseidon2_Generic_applySBoxToAll_spec__0(v_K_175_, v_fo_176_, v_sz_boxed_180_, v_i_boxed_181_, v_bs_179_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applyMat4___redArg(lean_object* v_fo_183_, lean_object* v_x_184_){
_start:
{
lean_object* v_toRingOps_185_; lean_object* v_toSemiringOps_186_; lean_object* v___x_188_; uint8_t v_isShared_189_; uint8_t v_isSharedCheck_218_; 
v_toRingOps_185_ = lean_ctor_get(v_fo_183_, 0);
lean_inc_ref(v_toRingOps_185_);
v_toSemiringOps_186_ = lean_ctor_get(v_toRingOps_185_, 0);
v_isSharedCheck_218_ = !lean_is_exclusive(v_toRingOps_185_);
if (v_isSharedCheck_218_ == 0)
{
lean_object* v_unused_219_; 
v_unused_219_ = lean_ctor_get(v_toRingOps_185_, 1);
lean_dec(v_unused_219_);
v___x_188_ = v_toRingOps_185_;
v_isShared_189_ = v_isSharedCheck_218_;
goto v_resetjp_187_;
}
else
{
lean_inc(v_toSemiringOps_186_);
lean_dec(v_toRingOps_185_);
v___x_188_ = lean_box(0);
v_isShared_189_ = v_isSharedCheck_218_;
goto v_resetjp_187_;
}
v_resetjp_187_:
{
lean_object* v_add_190_; lean_object* v___x_191_; lean_object* v___x_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v_t01_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; lean_object* v_t23_200_; lean_object* v_t0123_201_; lean_object* v_t01123_202_; lean_object* v_t01233_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_210_; lean_object* v___x_212_; 
v_add_190_ = lean_ctor_get(v_toSemiringOps_186_, 3);
lean_inc_n(v_add_190_, 11);
lean_dec_ref(v_toSemiringOps_186_);
v___x_191_ = lean_unsigned_to_nat(0u);
lean_inc_ref_n(v_x_184_, 3);
v___x_192_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_183_, v_x_184_, v___x_191_);
v___x_193_ = lean_unsigned_to_nat(1u);
v___x_194_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_183_, v_x_184_, v___x_193_);
lean_inc(v___x_194_);
lean_inc_n(v___x_192_, 2);
v_t01_195_ = lean_apply_2(v_add_190_, v___x_192_, v___x_194_);
v___x_196_ = lean_unsigned_to_nat(2u);
v___x_197_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_183_, v_x_184_, v___x_196_);
v___x_198_ = lean_unsigned_to_nat(3u);
v___x_199_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_183_, v_x_184_, v___x_198_);
lean_inc(v___x_199_);
lean_inc_n(v___x_197_, 2);
v_t23_200_ = lean_apply_2(v_add_190_, v___x_197_, v___x_199_);
lean_inc(v_t23_200_);
lean_inc(v_t01_195_);
v_t0123_201_ = lean_apply_2(v_add_190_, v_t01_195_, v_t23_200_);
lean_inc(v_t0123_201_);
v_t01123_202_ = lean_apply_2(v_add_190_, v_t0123_201_, v___x_194_);
v_t01233_203_ = lean_apply_2(v_add_190_, v_t0123_201_, v___x_199_);
lean_inc(v_t01123_202_);
v___x_204_ = lean_apply_2(v_add_190_, v_t01123_202_, v_t01_195_);
v___x_205_ = lean_apply_2(v_add_190_, v_t01123_202_, v___x_197_);
v___x_206_ = lean_apply_2(v_add_190_, v___x_205_, v___x_197_);
lean_inc(v_t01233_203_);
v___x_207_ = lean_apply_2(v_add_190_, v_t01233_203_, v_t23_200_);
v___x_208_ = lean_apply_2(v_add_190_, v_t01233_203_, v___x_192_);
v___x_209_ = lean_apply_2(v_add_190_, v___x_208_, v___x_192_);
v___x_210_ = lean_box(0);
if (v_isShared_189_ == 0)
{
lean_ctor_set_tag(v___x_188_, 1);
lean_ctor_set(v___x_188_, 1, v___x_210_);
lean_ctor_set(v___x_188_, 0, v___x_209_);
v___x_212_ = v___x_188_;
goto v_reusejp_211_;
}
else
{
lean_object* v_reuseFailAlloc_217_; 
v_reuseFailAlloc_217_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_217_, 0, v___x_209_);
lean_ctor_set(v_reuseFailAlloc_217_, 1, v___x_210_);
v___x_212_ = v_reuseFailAlloc_217_;
goto v_reusejp_211_;
}
v_reusejp_211_:
{
lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; 
v___x_213_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_213_, 0, v___x_207_);
lean_ctor_set(v___x_213_, 1, v___x_212_);
v___x_214_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_214_, 0, v___x_206_);
lean_ctor_set(v___x_214_, 1, v___x_213_);
v___x_215_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_215_, 0, v___x_204_);
lean_ctor_set(v___x_215_, 1, v___x_214_);
v___x_216_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4OfFields___redArg(v_fo_183_, v___x_215_);
return v___x_216_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applyMat4(lean_object* v_K_220_, lean_object* v_fo_221_, lean_object* v_x_222_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applyMat4___redArg(v_fo_221_, v_x_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalLinearLayer___redArg(lean_object* v_fo_224_, lean_object* v_state_225_){
_start:
{
lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v_chunk0_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; lean_object* v___x_253_; lean_object* v_chunk1_254_; lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; lean_object* v___x_259_; lean_object* v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v_chunk2_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___x_272_; lean_object* v___x_273_; lean_object* v___x_274_; lean_object* v___x_275_; lean_object* v___x_276_; lean_object* v_toRingOps_277_; lean_object* v_toSemiringOps_278_; lean_object* v___x_280_; uint8_t v_isShared_281_; uint8_t v_isSharedCheck_352_; 
v___x_226_ = lean_unsigned_to_nat(0u);
lean_inc_ref_n(v_state_225_, 15);
v___x_227_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_224_, v_state_225_, v___x_226_);
v___x_228_ = lean_unsigned_to_nat(1u);
v___x_229_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_224_, v_state_225_, v___x_228_);
v___x_230_ = lean_unsigned_to_nat(2u);
v___x_231_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_224_, v_state_225_, v___x_230_);
v___x_232_ = lean_unsigned_to_nat(3u);
v___x_233_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_224_, v_state_225_, v___x_232_);
v___x_234_ = lean_box(0);
v___x_235_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_235_, 0, v___x_233_);
lean_ctor_set(v___x_235_, 1, v___x_234_);
v___x_236_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_236_, 0, v___x_231_);
lean_ctor_set(v___x_236_, 1, v___x_235_);
v___x_237_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_237_, 0, v___x_229_);
lean_ctor_set(v___x_237_, 1, v___x_236_);
v___x_238_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_238_, 0, v___x_227_);
lean_ctor_set(v___x_238_, 1, v___x_237_);
lean_inc_ref_n(v_fo_224_, 6);
v___x_239_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4OfFields___redArg(v_fo_224_, v___x_238_);
v_chunk0_240_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applyMat4___redArg(v_fo_224_, v___x_239_);
v___x_241_ = lean_unsigned_to_nat(4u);
v___x_242_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_224_, v_state_225_, v___x_241_);
v___x_243_ = lean_unsigned_to_nat(5u);
v___x_244_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_224_, v_state_225_, v___x_243_);
v___x_245_ = lean_unsigned_to_nat(6u);
v___x_246_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_224_, v_state_225_, v___x_245_);
v___x_247_ = lean_unsigned_to_nat(7u);
v___x_248_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_224_, v_state_225_, v___x_247_);
v___x_249_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_249_, 0, v___x_248_);
lean_ctor_set(v___x_249_, 1, v___x_234_);
v___x_250_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_250_, 0, v___x_246_);
lean_ctor_set(v___x_250_, 1, v___x_249_);
v___x_251_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_251_, 0, v___x_244_);
lean_ctor_set(v___x_251_, 1, v___x_250_);
v___x_252_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_252_, 0, v___x_242_);
lean_ctor_set(v___x_252_, 1, v___x_251_);
v___x_253_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4OfFields___redArg(v_fo_224_, v___x_252_);
v_chunk1_254_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applyMat4___redArg(v_fo_224_, v___x_253_);
v___x_255_ = lean_unsigned_to_nat(8u);
v___x_256_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_224_, v_state_225_, v___x_255_);
v___x_257_ = lean_unsigned_to_nat(9u);
v___x_258_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_224_, v_state_225_, v___x_257_);
v___x_259_ = lean_unsigned_to_nat(10u);
v___x_260_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_224_, v_state_225_, v___x_259_);
v___x_261_ = lean_unsigned_to_nat(11u);
v___x_262_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_224_, v_state_225_, v___x_261_);
v___x_263_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_263_, 0, v___x_262_);
lean_ctor_set(v___x_263_, 1, v___x_234_);
v___x_264_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_264_, 0, v___x_260_);
lean_ctor_set(v___x_264_, 1, v___x_263_);
v___x_265_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_265_, 0, v___x_258_);
lean_ctor_set(v___x_265_, 1, v___x_264_);
v___x_266_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_266_, 0, v___x_256_);
lean_ctor_set(v___x_266_, 1, v___x_265_);
v___x_267_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4OfFields___redArg(v_fo_224_, v___x_266_);
v_chunk2_268_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applyMat4___redArg(v_fo_224_, v___x_267_);
v___x_269_ = lean_unsigned_to_nat(12u);
v___x_270_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_224_, v_state_225_, v___x_269_);
v___x_271_ = lean_unsigned_to_nat(13u);
v___x_272_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_224_, v_state_225_, v___x_271_);
v___x_273_ = lean_unsigned_to_nat(14u);
v___x_274_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_224_, v_state_225_, v___x_273_);
v___x_275_ = lean_unsigned_to_nat(15u);
v___x_276_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_224_, v_state_225_, v___x_275_);
v_toRingOps_277_ = lean_ctor_get(v_fo_224_, 0);
lean_inc_ref(v_toRingOps_277_);
v_toSemiringOps_278_ = lean_ctor_get(v_toRingOps_277_, 0);
v_isSharedCheck_352_ = !lean_is_exclusive(v_toRingOps_277_);
if (v_isSharedCheck_352_ == 0)
{
lean_object* v_unused_353_; 
v_unused_353_ = lean_ctor_get(v_toRingOps_277_, 1);
lean_dec(v_unused_353_);
v___x_280_ = v_toRingOps_277_;
v_isShared_281_ = v_isSharedCheck_352_;
goto v_resetjp_279_;
}
else
{
lean_inc(v_toSemiringOps_278_);
lean_dec(v_toRingOps_277_);
v___x_280_ = lean_box(0);
v_isShared_281_ = v_isSharedCheck_352_;
goto v_resetjp_279_;
}
v_resetjp_279_:
{
lean_object* v_add_282_; lean_object* v___x_284_; 
v_add_282_ = lean_ctor_get(v_toSemiringOps_278_, 3);
lean_inc(v_add_282_);
lean_dec_ref(v_toSemiringOps_278_);
if (v_isShared_281_ == 0)
{
lean_ctor_set_tag(v___x_280_, 1);
lean_ctor_set(v___x_280_, 1, v___x_234_);
lean_ctor_set(v___x_280_, 0, v___x_276_);
v___x_284_ = v___x_280_;
goto v_reusejp_283_;
}
else
{
lean_object* v_reuseFailAlloc_351_; 
v_reuseFailAlloc_351_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_351_, 0, v___x_276_);
lean_ctor_set(v_reuseFailAlloc_351_, 1, v___x_234_);
v___x_284_ = v_reuseFailAlloc_351_;
goto v_reusejp_283_;
}
v_reusejp_283_:
{
lean_object* v___x_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v_chunk3_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___x_293_; lean_object* v___x_294_; lean_object* v___x_295_; lean_object* v_lane0_296_; lean_object* v___x_297_; lean_object* v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; lean_object* v___x_301_; lean_object* v___x_302_; lean_object* v_lane1_303_; lean_object* v___x_304_; lean_object* v___x_305_; lean_object* v___x_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v_lane2_310_; lean_object* v___x_311_; lean_object* v___x_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; lean_object* v_lane3_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; lean_object* v___x_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; lean_object* v___x_342_; lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; 
v___x_285_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_285_, 0, v___x_274_);
lean_ctor_set(v___x_285_, 1, v___x_284_);
v___x_286_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_286_, 0, v___x_272_);
lean_ctor_set(v___x_286_, 1, v___x_285_);
v___x_287_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_287_, 0, v___x_270_);
lean_ctor_set(v___x_287_, 1, v___x_286_);
lean_inc_ref_n(v_fo_224_, 2);
v___x_288_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4OfFields___redArg(v_fo_224_, v___x_287_);
v_chunk3_289_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applyMat4___redArg(v_fo_224_, v___x_288_);
lean_inc_ref_n(v_chunk0_240_, 3);
v___x_290_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_224_, v_chunk0_240_, v___x_226_);
lean_inc_ref_n(v_chunk1_254_, 3);
v___x_291_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_224_, v_chunk1_254_, v___x_226_);
lean_inc_n(v_add_282_, 27);
lean_inc(v___x_291_);
lean_inc(v___x_290_);
v___x_292_ = lean_apply_2(v_add_282_, v___x_290_, v___x_291_);
lean_inc_ref_n(v_chunk2_268_, 3);
v___x_293_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_224_, v_chunk2_268_, v___x_226_);
lean_inc_ref_n(v_chunk3_289_, 3);
v___x_294_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_224_, v_chunk3_289_, v___x_226_);
lean_inc(v___x_294_);
lean_inc(v___x_293_);
v___x_295_ = lean_apply_2(v_add_282_, v___x_293_, v___x_294_);
v_lane0_296_ = lean_apply_2(v_add_282_, v___x_292_, v___x_295_);
v___x_297_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_224_, v_chunk0_240_, v___x_228_);
v___x_298_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_224_, v_chunk1_254_, v___x_228_);
lean_inc(v___x_298_);
lean_inc(v___x_297_);
v___x_299_ = lean_apply_2(v_add_282_, v___x_297_, v___x_298_);
v___x_300_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_224_, v_chunk2_268_, v___x_228_);
v___x_301_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_224_, v_chunk3_289_, v___x_228_);
lean_inc(v___x_301_);
lean_inc(v___x_300_);
v___x_302_ = lean_apply_2(v_add_282_, v___x_300_, v___x_301_);
v_lane1_303_ = lean_apply_2(v_add_282_, v___x_299_, v___x_302_);
v___x_304_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_224_, v_chunk0_240_, v___x_230_);
v___x_305_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_224_, v_chunk1_254_, v___x_230_);
lean_inc(v___x_305_);
lean_inc(v___x_304_);
v___x_306_ = lean_apply_2(v_add_282_, v___x_304_, v___x_305_);
v___x_307_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_224_, v_chunk2_268_, v___x_230_);
v___x_308_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_224_, v_chunk3_289_, v___x_230_);
lean_inc(v___x_308_);
lean_inc(v___x_307_);
v___x_309_ = lean_apply_2(v_add_282_, v___x_307_, v___x_308_);
v_lane2_310_ = lean_apply_2(v_add_282_, v___x_306_, v___x_309_);
v___x_311_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_224_, v_chunk0_240_, v___x_232_);
v___x_312_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_224_, v_chunk1_254_, v___x_232_);
lean_inc(v___x_312_);
lean_inc(v___x_311_);
v___x_313_ = lean_apply_2(v_add_282_, v___x_311_, v___x_312_);
v___x_314_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_224_, v_chunk2_268_, v___x_232_);
v___x_315_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_vec4At___redArg(v_fo_224_, v_chunk3_289_, v___x_232_);
lean_inc(v___x_315_);
lean_inc(v___x_314_);
v___x_316_ = lean_apply_2(v_add_282_, v___x_314_, v___x_315_);
v_lane3_317_ = lean_apply_2(v_add_282_, v___x_313_, v___x_316_);
lean_inc_n(v_lane0_296_, 3);
v___x_318_ = lean_apply_2(v_add_282_, v___x_290_, v_lane0_296_);
lean_inc_n(v_lane1_303_, 3);
v___x_319_ = lean_apply_2(v_add_282_, v___x_297_, v_lane1_303_);
lean_inc_n(v_lane2_310_, 3);
v___x_320_ = lean_apply_2(v_add_282_, v___x_304_, v_lane2_310_);
lean_inc_n(v_lane3_317_, 3);
v___x_321_ = lean_apply_2(v_add_282_, v___x_311_, v_lane3_317_);
v___x_322_ = lean_apply_2(v_add_282_, v___x_291_, v_lane0_296_);
v___x_323_ = lean_apply_2(v_add_282_, v___x_298_, v_lane1_303_);
v___x_324_ = lean_apply_2(v_add_282_, v___x_305_, v_lane2_310_);
v___x_325_ = lean_apply_2(v_add_282_, v___x_312_, v_lane3_317_);
v___x_326_ = lean_apply_2(v_add_282_, v___x_293_, v_lane0_296_);
v___x_327_ = lean_apply_2(v_add_282_, v___x_300_, v_lane1_303_);
v___x_328_ = lean_apply_2(v_add_282_, v___x_307_, v_lane2_310_);
v___x_329_ = lean_apply_2(v_add_282_, v___x_314_, v_lane3_317_);
v___x_330_ = lean_apply_2(v_add_282_, v___x_294_, v_lane0_296_);
v___x_331_ = lean_apply_2(v_add_282_, v___x_301_, v_lane1_303_);
v___x_332_ = lean_apply_2(v_add_282_, v___x_308_, v_lane2_310_);
v___x_333_ = lean_apply_2(v_add_282_, v___x_315_, v_lane3_317_);
v___x_334_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_334_, 0, v___x_333_);
lean_ctor_set(v___x_334_, 1, v___x_234_);
v___x_335_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_335_, 0, v___x_332_);
lean_ctor_set(v___x_335_, 1, v___x_334_);
v___x_336_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_336_, 0, v___x_331_);
lean_ctor_set(v___x_336_, 1, v___x_335_);
v___x_337_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_337_, 0, v___x_330_);
lean_ctor_set(v___x_337_, 1, v___x_336_);
v___x_338_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_338_, 0, v___x_329_);
lean_ctor_set(v___x_338_, 1, v___x_337_);
v___x_339_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_339_, 0, v___x_328_);
lean_ctor_set(v___x_339_, 1, v___x_338_);
v___x_340_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_340_, 0, v___x_327_);
lean_ctor_set(v___x_340_, 1, v___x_339_);
v___x_341_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_341_, 0, v___x_326_);
lean_ctor_set(v___x_341_, 1, v___x_340_);
v___x_342_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_342_, 0, v___x_325_);
lean_ctor_set(v___x_342_, 1, v___x_341_);
v___x_343_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_343_, 0, v___x_324_);
lean_ctor_set(v___x_343_, 1, v___x_342_);
v___x_344_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_344_, 0, v___x_323_);
lean_ctor_set(v___x_344_, 1, v___x_343_);
v___x_345_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_345_, 0, v___x_322_);
lean_ctor_set(v___x_345_, 1, v___x_344_);
v___x_346_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_346_, 0, v___x_321_);
lean_ctor_set(v___x_346_, 1, v___x_345_);
v___x_347_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_347_, 0, v___x_320_);
lean_ctor_set(v___x_347_, 1, v___x_346_);
v___x_348_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_348_, 0, v___x_319_);
lean_ctor_set(v___x_348_, 1, v___x_347_);
v___x_349_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_349_, 0, v___x_318_);
lean_ctor_set(v___x_349_, 1, v___x_348_);
v___x_350_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfFields___redArg(v_fo_224_, v___x_349_);
return v___x_350_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalLinearLayer(lean_object* v_K_354_, lean_object* v_fo_355_, lean_object* v_state_356_){
_start:
{
lean_object* v___x_357_; 
v___x_357_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalLinearLayer___redArg(v_fo_355_, v_state_356_);
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_divByTwoPow___redArg(lean_object* v_fo_358_, lean_object* v_k_359_, lean_object* v_x_360_){
_start:
{
lean_object* v_toRingOps_361_; lean_object* v_toSemiringOps_362_; lean_object* v_inv_363_; lean_object* v_natCast_364_; lean_object* v_mul_365_; lean_object* v_pow_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; 
v_toRingOps_361_ = lean_ctor_get(v_fo_358_, 0);
v_toSemiringOps_362_ = lean_ctor_get(v_toRingOps_361_, 0);
lean_inc_ref(v_toSemiringOps_362_);
v_inv_363_ = lean_ctor_get(v_fo_358_, 1);
lean_inc(v_inv_363_);
lean_dec_ref(v_fo_358_);
v_natCast_364_ = lean_ctor_get(v_toSemiringOps_362_, 2);
lean_inc(v_natCast_364_);
v_mul_365_ = lean_ctor_get(v_toSemiringOps_362_, 4);
lean_inc(v_mul_365_);
v_pow_366_ = lean_ctor_get(v_toSemiringOps_362_, 5);
lean_inc(v_pow_366_);
lean_dec_ref(v_toSemiringOps_362_);
v___x_367_ = lean_unsigned_to_nat(2u);
v___x_368_ = lean_apply_1(v_natCast_364_, v___x_367_);
v___x_369_ = lean_apply_2(v_pow_366_, v___x_368_, v_k_359_);
v___x_370_ = lean_apply_1(v_inv_363_, v___x_369_);
v___x_371_ = lean_apply_2(v_mul_365_, v_x_360_, v___x_370_);
return v___x_371_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_divByTwoPow(lean_object* v_K_372_, lean_object* v_fo_373_, lean_object* v_k_374_, lean_object* v_x_375_){
_start:
{
lean_object* v___x_376_; 
v___x_376_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_divByTwoPow___redArg(v_fo_373_, v_k_374_, v_x_375_);
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_sumState_spec__0___redArg(lean_object* v___x_377_, lean_object* v_x_378_, lean_object* v_x_379_){
_start:
{
if (lean_obj_tag(v_x_379_) == 0)
{
lean_dec(v___x_377_);
return v_x_378_;
}
else
{
lean_object* v_head_380_; lean_object* v_tail_381_; lean_object* v___x_382_; 
v_head_380_ = lean_ctor_get(v_x_379_, 0);
lean_inc(v_head_380_);
v_tail_381_ = lean_ctor_get(v_x_379_, 1);
lean_inc(v_tail_381_);
lean_dec_ref_known(v_x_379_, 2);
lean_inc(v___x_377_);
v___x_382_ = lean_apply_2(v___x_377_, v_x_378_, v_head_380_);
v_x_378_ = v___x_382_;
v_x_379_ = v_tail_381_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_sumState___redArg(lean_object* v_fo_384_, lean_object* v_state_385_){
_start:
{
lean_object* v_toRingOps_386_; lean_object* v_toSemiringOps_387_; lean_object* v_zero_388_; lean_object* v_add_389_; lean_object* v___x_390_; lean_object* v___x_391_; 
v_toRingOps_386_ = lean_ctor_get(v_fo_384_, 0);
lean_inc_ref(v_toRingOps_386_);
lean_dec_ref(v_fo_384_);
v_toSemiringOps_387_ = lean_ctor_get(v_toRingOps_386_, 0);
lean_inc_ref(v_toSemiringOps_387_);
lean_dec_ref(v_toRingOps_386_);
v_zero_388_ = lean_ctor_get(v_toSemiringOps_387_, 0);
lean_inc(v_zero_388_);
v_add_389_ = lean_ctor_get(v_toSemiringOps_387_, 3);
lean_inc(v_add_389_);
lean_dec_ref(v_toSemiringOps_387_);
v___x_390_ = lean_array_to_list(v_state_385_);
v___x_391_ = lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_sumState_spec__0___redArg(v_add_389_, v_zero_388_, v___x_390_);
return v___x_391_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_sumState(lean_object* v_K_392_, lean_object* v_fo_393_, lean_object* v_state_394_){
_start:
{
lean_object* v___x_395_; 
v___x_395_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_sumState___redArg(v_fo_393_, v_state_394_);
return v___x_395_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_sumState_spec__0(lean_object* v_K_396_, lean_object* v___x_397_, lean_object* v_x_398_, lean_object* v_x_399_){
_start:
{
lean_object* v___x_400_; 
v___x_400_ = lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_sumState_spec__0___redArg(v___x_397_, v_x_398_, v_x_399_);
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalLinearLayer___redArg(lean_object* v_fo_401_, lean_object* v_state_402_){
_start:
{
lean_object* v_toRingOps_403_; lean_object* v_toSemiringOps_404_; lean_object* v_sub_405_; lean_object* v___x_407_; uint8_t v_isShared_408_; uint8_t v_isSharedCheck_501_; 
v_toRingOps_403_ = lean_ctor_get(v_fo_401_, 0);
lean_inc_ref(v_toRingOps_403_);
v_toSemiringOps_404_ = lean_ctor_get(v_toRingOps_403_, 0);
v_sub_405_ = lean_ctor_get(v_toRingOps_403_, 1);
v_isSharedCheck_501_ = !lean_is_exclusive(v_toRingOps_403_);
if (v_isSharedCheck_501_ == 0)
{
v___x_407_ = v_toRingOps_403_;
v_isShared_408_ = v_isSharedCheck_501_;
goto v_resetjp_406_;
}
else
{
lean_inc(v_sub_405_);
lean_inc(v_toSemiringOps_404_);
lean_dec(v_toRingOps_403_);
v___x_407_ = lean_box(0);
v_isShared_408_ = v_isSharedCheck_501_;
goto v_resetjp_406_;
}
v_resetjp_406_:
{
lean_object* v_add_409_; lean_object* v_total_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; lean_object* v___x_419_; lean_object* v___x_420_; lean_object* v___x_421_; lean_object* v___x_422_; lean_object* v___x_423_; lean_object* v___x_424_; lean_object* v___x_425_; lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_483_; 
v_add_409_ = lean_ctor_get(v_toSemiringOps_404_, 3);
lean_inc_n(v_add_409_, 21);
lean_dec_ref(v_toSemiringOps_404_);
lean_inc_ref_n(v_state_402_, 16);
lean_inc_ref_n(v_fo_401_, 10);
v_total_410_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_sumState___redArg(v_fo_401_, v_state_402_);
v___x_411_ = lean_unsigned_to_nat(0u);
v___x_412_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_401_, v_state_402_, v___x_411_);
lean_inc(v___x_412_);
v___x_413_ = lean_apply_2(v_add_409_, v___x_412_, v___x_412_);
lean_inc_n(v_sub_405_, 6);
lean_inc_n(v_total_410_, 15);
v___x_414_ = lean_apply_2(v_sub_405_, v_total_410_, v___x_413_);
v___x_415_ = lean_unsigned_to_nat(1u);
v___x_416_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_401_, v_state_402_, v___x_415_);
v___x_417_ = lean_apply_2(v_add_409_, v_total_410_, v___x_416_);
v___x_418_ = lean_unsigned_to_nat(2u);
v___x_419_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_401_, v_state_402_, v___x_418_);
lean_inc(v___x_419_);
v___x_420_ = lean_apply_2(v_add_409_, v___x_419_, v___x_419_);
v___x_421_ = lean_apply_2(v_add_409_, v_total_410_, v___x_420_);
v___x_422_ = lean_unsigned_to_nat(3u);
v___x_423_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_401_, v_state_402_, v___x_422_);
v___x_424_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_divByTwoPow___redArg(v_fo_401_, v___x_415_, v___x_423_);
v___x_425_ = lean_apply_2(v_add_409_, v_total_410_, v___x_424_);
v___x_426_ = lean_unsigned_to_nat(4u);
v___x_427_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_401_, v_state_402_, v___x_426_);
lean_inc_n(v___x_427_, 2);
v___x_428_ = lean_apply_2(v_add_409_, v___x_427_, v___x_427_);
v___x_429_ = lean_apply_2(v_add_409_, v___x_428_, v___x_427_);
v___x_430_ = lean_apply_2(v_add_409_, v_total_410_, v___x_429_);
v___x_431_ = lean_unsigned_to_nat(5u);
v___x_432_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_401_, v_state_402_, v___x_431_);
lean_inc_n(v___x_432_, 3);
v___x_433_ = lean_apply_2(v_add_409_, v___x_432_, v___x_432_);
v___x_434_ = lean_apply_2(v_add_409_, v___x_433_, v___x_432_);
v___x_435_ = lean_apply_2(v_add_409_, v___x_434_, v___x_432_);
v___x_436_ = lean_apply_2(v_add_409_, v_total_410_, v___x_435_);
v___x_437_ = lean_unsigned_to_nat(6u);
v___x_438_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_401_, v_state_402_, v___x_437_);
v___x_439_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_divByTwoPow___redArg(v_fo_401_, v___x_415_, v___x_438_);
v___x_440_ = lean_apply_2(v_sub_405_, v_total_410_, v___x_439_);
v___x_441_ = lean_unsigned_to_nat(7u);
v___x_442_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_401_, v_state_402_, v___x_441_);
lean_inc_n(v___x_442_, 2);
v___x_443_ = lean_apply_2(v_add_409_, v___x_442_, v___x_442_);
v___x_444_ = lean_apply_2(v_add_409_, v___x_443_, v___x_442_);
v___x_445_ = lean_apply_2(v_sub_405_, v_total_410_, v___x_444_);
v___x_446_ = lean_unsigned_to_nat(8u);
v___x_447_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_401_, v_state_402_, v___x_446_);
lean_inc_n(v___x_447_, 3);
v___x_448_ = lean_apply_2(v_add_409_, v___x_447_, v___x_447_);
v___x_449_ = lean_apply_2(v_add_409_, v___x_448_, v___x_447_);
v___x_450_ = lean_apply_2(v_add_409_, v___x_449_, v___x_447_);
v___x_451_ = lean_apply_2(v_sub_405_, v_total_410_, v___x_450_);
v___x_452_ = lean_unsigned_to_nat(9u);
v___x_453_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_401_, v_state_402_, v___x_452_);
v___x_454_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_divByTwoPow___redArg(v_fo_401_, v___x_446_, v___x_453_);
v___x_455_ = lean_apply_2(v_add_409_, v_total_410_, v___x_454_);
v___x_456_ = lean_unsigned_to_nat(10u);
v___x_457_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_401_, v_state_402_, v___x_456_);
v___x_458_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_divByTwoPow___redArg(v_fo_401_, v___x_418_, v___x_457_);
v___x_459_ = lean_apply_2(v_add_409_, v_total_410_, v___x_458_);
v___x_460_ = lean_unsigned_to_nat(11u);
v___x_461_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_401_, v_state_402_, v___x_460_);
v___x_462_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_divByTwoPow___redArg(v_fo_401_, v___x_422_, v___x_461_);
v___x_463_ = lean_apply_2(v_add_409_, v_total_410_, v___x_462_);
v___x_464_ = lean_unsigned_to_nat(27u);
v___x_465_ = lean_unsigned_to_nat(12u);
v___x_466_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_401_, v_state_402_, v___x_465_);
v___x_467_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_divByTwoPow___redArg(v_fo_401_, v___x_464_, v___x_466_);
v___x_468_ = lean_apply_2(v_add_409_, v_total_410_, v___x_467_);
v___x_469_ = lean_unsigned_to_nat(13u);
v___x_470_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_401_, v_state_402_, v___x_469_);
v___x_471_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_divByTwoPow___redArg(v_fo_401_, v___x_446_, v___x_470_);
v___x_472_ = lean_apply_2(v_sub_405_, v_total_410_, v___x_471_);
v___x_473_ = lean_unsigned_to_nat(14u);
v___x_474_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_401_, v_state_402_, v___x_473_);
v___x_475_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_divByTwoPow___redArg(v_fo_401_, v___x_426_, v___x_474_);
v___x_476_ = lean_apply_2(v_sub_405_, v_total_410_, v___x_475_);
v___x_477_ = lean_unsigned_to_nat(15u);
v___x_478_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_401_, v_state_402_, v___x_477_);
v___x_479_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_divByTwoPow___redArg(v_fo_401_, v___x_464_, v___x_478_);
v___x_480_ = lean_apply_2(v_sub_405_, v_total_410_, v___x_479_);
v___x_481_ = lean_box(0);
if (v_isShared_408_ == 0)
{
lean_ctor_set_tag(v___x_407_, 1);
lean_ctor_set(v___x_407_, 1, v___x_481_);
lean_ctor_set(v___x_407_, 0, v___x_480_);
v___x_483_ = v___x_407_;
goto v_reusejp_482_;
}
else
{
lean_object* v_reuseFailAlloc_500_; 
v_reuseFailAlloc_500_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_500_, 0, v___x_480_);
lean_ctor_set(v_reuseFailAlloc_500_, 1, v___x_481_);
v___x_483_ = v_reuseFailAlloc_500_;
goto v_reusejp_482_;
}
v_reusejp_482_:
{
lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; 
v___x_484_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_484_, 0, v___x_476_);
lean_ctor_set(v___x_484_, 1, v___x_483_);
v___x_485_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_485_, 0, v___x_472_);
lean_ctor_set(v___x_485_, 1, v___x_484_);
v___x_486_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_486_, 0, v___x_468_);
lean_ctor_set(v___x_486_, 1, v___x_485_);
v___x_487_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_487_, 0, v___x_463_);
lean_ctor_set(v___x_487_, 1, v___x_486_);
v___x_488_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_488_, 0, v___x_459_);
lean_ctor_set(v___x_488_, 1, v___x_487_);
v___x_489_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_489_, 0, v___x_455_);
lean_ctor_set(v___x_489_, 1, v___x_488_);
v___x_490_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_490_, 0, v___x_451_);
lean_ctor_set(v___x_490_, 1, v___x_489_);
v___x_491_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_491_, 0, v___x_445_);
lean_ctor_set(v___x_491_, 1, v___x_490_);
v___x_492_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_492_, 0, v___x_440_);
lean_ctor_set(v___x_492_, 1, v___x_491_);
v___x_493_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_493_, 0, v___x_436_);
lean_ctor_set(v___x_493_, 1, v___x_492_);
v___x_494_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_494_, 0, v___x_430_);
lean_ctor_set(v___x_494_, 1, v___x_493_);
v___x_495_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_495_, 0, v___x_425_);
lean_ctor_set(v___x_495_, 1, v___x_494_);
v___x_496_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_496_, 0, v___x_421_);
lean_ctor_set(v___x_496_, 1, v___x_495_);
v___x_497_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_497_, 0, v___x_417_);
lean_ctor_set(v___x_497_, 1, v___x_496_);
v___x_498_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_498_, 0, v___x_414_);
lean_ctor_set(v___x_498_, 1, v___x_497_);
v___x_499_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfFields___redArg(v_fo_401_, v___x_498_);
return v___x_499_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalLinearLayer(lean_object* v_K_502_, lean_object* v_fo_503_, lean_object* v_state_504_){
_start:
{
lean_object* v___x_505_; 
v___x_505_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalLinearLayer___redArg(v_fo_503_, v_state_504_);
return v___x_505_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalRound___redArg(lean_object* v_fo_506_, lean_object* v_rc_507_, lean_object* v_state_508_){
_start:
{
lean_object* v___x_509_; lean_object* v___x_510_; lean_object* v___x_511_; 
lean_inc_ref_n(v_fo_506_, 2);
v___x_509_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_addRoundConstants___redArg(v_fo_506_, v_rc_507_, v_state_508_);
v___x_510_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_applySBoxToAll___redArg(v_fo_506_, v___x_509_);
v___x_511_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalLinearLayer___redArg(v_fo_506_, v___x_510_);
return v___x_511_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalRound(lean_object* v_K_512_, lean_object* v_fo_513_, lean_object* v_rc_514_, lean_object* v_state_515_){
_start:
{
lean_object* v___x_516_; 
v___x_516_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalRound___redArg(v_fo_513_, v_rc_514_, v_state_515_);
return v___x_516_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRound___redArg___lam__0(lean_object* v_state_517_, lean_object* v_fo_518_, lean_object* v_rc_519_, lean_object* v_idx_520_){
_start:
{
lean_object* v___x_521_; uint8_t v___x_522_; 
v___x_521_ = lean_unsigned_to_nat(0u);
v___x_522_ = lean_nat_dec_eq(v_idx_520_, v___x_521_);
if (v___x_522_ == 0)
{
lean_object* v___x_523_; 
lean_dec(v_rc_519_);
lean_dec_ref(v_fo_518_);
v___x_523_ = lean_array_fget(v_state_517_, v_idx_520_);
lean_dec_ref(v_state_517_);
return v___x_523_;
}
else
{
lean_object* v_toRingOps_524_; lean_object* v_toSemiringOps_525_; lean_object* v_add_526_; lean_object* v___x_527_; lean_object* v___x_528_; lean_object* v___x_529_; 
v_toRingOps_524_ = lean_ctor_get(v_fo_518_, 0);
v_toSemiringOps_525_ = lean_ctor_get(v_toRingOps_524_, 0);
v_add_526_ = lean_ctor_get(v_toSemiringOps_525_, 3);
v___x_527_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_518_, v_state_517_, v___x_521_);
lean_inc(v_add_526_);
v___x_528_ = lean_apply_2(v_add_526_, v___x_527_, v_rc_519_);
v___x_529_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_sbox___redArg(v_fo_518_, v___x_528_);
return v___x_529_;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRound___redArg___lam__0___boxed(lean_object* v_state_530_, lean_object* v_fo_531_, lean_object* v_rc_532_, lean_object* v_idx_533_){
_start:
{
lean_object* v_res_534_; 
v_res_534_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRound___redArg___lam__0(v_state_530_, v_fo_531_, v_rc_532_, v_idx_533_);
lean_dec(v_idx_533_);
return v_res_534_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRound___redArg(lean_object* v_fo_535_, lean_object* v_rc_536_, lean_object* v_state_537_){
_start:
{
lean_object* v___f_538_; lean_object* v___x_539_; lean_object* v_updated_540_; lean_object* v___x_541_; 
lean_inc_ref(v_fo_535_);
v___f_538_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRound___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_538_, 0, v_state_537_);
lean_closure_set(v___f_538_, 1, v_fo_535_);
lean_closure_set(v___f_538_, 2, v_rc_536_);
v___x_539_ = lean_unsigned_to_nat(16u);
v_updated_540_ = l_Array_ofFn___redArg(v___x_539_, v___f_538_);
v___x_541_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalLinearLayer___redArg(v_fo_535_, v_updated_540_);
return v___x_541_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRound(lean_object* v_K_542_, lean_object* v_fo_543_, lean_object* v_rc_544_, lean_object* v_state_545_){
_start:
{
lean_object* v___x_546_; 
v___x_546_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRound___redArg(v_fo_543_, v_rc_544_, v_state_545_);
return v___x_546_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_initialExternalRounds_spec__1___redArg(lean_object* v_fo_997_, lean_object* v_x_998_, lean_object* v_x_999_){
_start:
{
if (lean_obj_tag(v_x_999_) == 0)
{
lean_dec_ref(v_fo_997_);
return v_x_998_;
}
else
{
lean_object* v_head_1000_; lean_object* v_tail_1001_; lean_object* v___x_1002_; 
v_head_1000_ = lean_ctor_get(v_x_999_, 0);
lean_inc(v_head_1000_);
v_tail_1001_ = lean_ctor_get(v_x_999_, 1);
lean_inc(v_tail_1001_);
lean_dec_ref_known(v_x_999_, 2);
lean_inc_ref(v_fo_997_);
v___x_1002_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalRound___redArg(v_fo_997_, v_head_1000_, v_x_998_);
v_x_998_ = v___x_1002_;
v_x_999_ = v_tail_1001_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_Generic_initialExternalRounds_spec__0___redArg(lean_object* v_fo_1004_, lean_object* v_a_1005_, lean_object* v_a_1006_){
_start:
{
if (lean_obj_tag(v_a_1005_) == 0)
{
lean_object* v___x_1007_; 
lean_dec_ref(v_fo_1004_);
v___x_1007_ = l_List_reverse___redArg(v_a_1006_);
return v___x_1007_;
}
else
{
lean_object* v_head_1008_; lean_object* v_tail_1009_; lean_object* v___x_1011_; uint8_t v_isShared_1012_; uint8_t v_isSharedCheck_1018_; 
v_head_1008_ = lean_ctor_get(v_a_1005_, 0);
v_tail_1009_ = lean_ctor_get(v_a_1005_, 1);
v_isSharedCheck_1018_ = !lean_is_exclusive(v_a_1005_);
if (v_isSharedCheck_1018_ == 0)
{
v___x_1011_ = v_a_1005_;
v_isShared_1012_ = v_isSharedCheck_1018_;
goto v_resetjp_1010_;
}
else
{
lean_inc(v_tail_1009_);
lean_inc(v_head_1008_);
lean_dec(v_a_1005_);
v___x_1011_ = lean_box(0);
v_isShared_1012_ = v_isSharedCheck_1018_;
goto v_resetjp_1010_;
}
v_resetjp_1010_:
{
lean_object* v___x_1013_; lean_object* v___x_1015_; 
lean_inc_ref(v_fo_1004_);
v___x_1013_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateOfNats___redArg(v_fo_1004_, v_head_1008_);
if (v_isShared_1012_ == 0)
{
lean_ctor_set(v___x_1011_, 1, v_a_1006_);
lean_ctor_set(v___x_1011_, 0, v___x_1013_);
v___x_1015_ = v___x_1011_;
goto v_reusejp_1014_;
}
else
{
lean_object* v_reuseFailAlloc_1017_; 
v_reuseFailAlloc_1017_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1017_, 0, v___x_1013_);
lean_ctor_set(v_reuseFailAlloc_1017_, 1, v_a_1006_);
v___x_1015_ = v_reuseFailAlloc_1017_;
goto v_reusejp_1014_;
}
v_reusejp_1014_:
{
v_a_1005_ = v_tail_1009_;
v_a_1006_ = v___x_1015_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_initialExternalRounds___redArg(lean_object* v_fo_1019_, lean_object* v_state_1020_){
_start:
{
lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; 
lean_inc_ref_n(v_fo_1019_, 2);
v___x_1021_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalLinearLayer___redArg(v_fo_1019_, v_state_1020_);
v___x_1022_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalInitialRcNat));
v___x_1023_ = lean_box(0);
v___x_1024_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_Generic_initialExternalRounds_spec__0___redArg(v_fo_1019_, v___x_1022_, v___x_1023_);
v___x_1025_ = lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_initialExternalRounds_spec__1___redArg(v_fo_1019_, v___x_1021_, v___x_1024_);
return v___x_1025_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_initialExternalRounds(lean_object* v_K_1026_, lean_object* v_fo_1027_, lean_object* v_state_1028_){
_start:
{
lean_object* v___x_1029_; 
v___x_1029_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_initialExternalRounds___redArg(v_fo_1027_, v_state_1028_);
return v___x_1029_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_Generic_initialExternalRounds_spec__0(lean_object* v_K_1030_, lean_object* v_fo_1031_, lean_object* v_a_1032_, lean_object* v_a_1033_){
_start:
{
lean_object* v___x_1034_; 
v___x_1034_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_Generic_initialExternalRounds_spec__0___redArg(v_fo_1031_, v_a_1032_, v_a_1033_);
return v___x_1034_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_initialExternalRounds_spec__1(lean_object* v_K_1035_, lean_object* v_fo_1036_, lean_object* v_x_1037_, lean_object* v_x_1038_){
_start:
{
lean_object* v___x_1039_; 
v___x_1039_ = lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_initialExternalRounds_spec__1___redArg(v_fo_1036_, v_x_1037_, v_x_1038_);
return v___x_1039_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_Generic_internalRounds_spec__0___redArg(lean_object* v___x_1040_, lean_object* v_a_1041_, lean_object* v_a_1042_){
_start:
{
if (lean_obj_tag(v_a_1041_) == 0)
{
lean_object* v___x_1043_; 
lean_dec(v___x_1040_);
v___x_1043_ = l_List_reverse___redArg(v_a_1042_);
return v___x_1043_;
}
else
{
lean_object* v_head_1044_; lean_object* v_tail_1045_; lean_object* v___x_1047_; uint8_t v_isShared_1048_; uint8_t v_isSharedCheck_1054_; 
v_head_1044_ = lean_ctor_get(v_a_1041_, 0);
v_tail_1045_ = lean_ctor_get(v_a_1041_, 1);
v_isSharedCheck_1054_ = !lean_is_exclusive(v_a_1041_);
if (v_isSharedCheck_1054_ == 0)
{
v___x_1047_ = v_a_1041_;
v_isShared_1048_ = v_isSharedCheck_1054_;
goto v_resetjp_1046_;
}
else
{
lean_inc(v_tail_1045_);
lean_inc(v_head_1044_);
lean_dec(v_a_1041_);
v___x_1047_ = lean_box(0);
v_isShared_1048_ = v_isSharedCheck_1054_;
goto v_resetjp_1046_;
}
v_resetjp_1046_:
{
lean_object* v___x_1049_; lean_object* v___x_1051_; 
lean_inc(v___x_1040_);
v___x_1049_ = lean_apply_1(v___x_1040_, v_head_1044_);
if (v_isShared_1048_ == 0)
{
lean_ctor_set(v___x_1047_, 1, v_a_1042_);
lean_ctor_set(v___x_1047_, 0, v___x_1049_);
v___x_1051_ = v___x_1047_;
goto v_reusejp_1050_;
}
else
{
lean_object* v_reuseFailAlloc_1053_; 
v_reuseFailAlloc_1053_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1053_, 0, v___x_1049_);
lean_ctor_set(v_reuseFailAlloc_1053_, 1, v_a_1042_);
v___x_1051_ = v_reuseFailAlloc_1053_;
goto v_reusejp_1050_;
}
v_reusejp_1050_:
{
v_a_1041_ = v_tail_1045_;
v_a_1042_ = v___x_1051_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_internalRounds_spec__1___redArg(lean_object* v_fo_1055_, lean_object* v_x_1056_, lean_object* v_x_1057_){
_start:
{
if (lean_obj_tag(v_x_1057_) == 0)
{
lean_dec_ref(v_fo_1055_);
return v_x_1056_;
}
else
{
lean_object* v_head_1058_; lean_object* v_tail_1059_; lean_object* v___x_1060_; 
v_head_1058_ = lean_ctor_get(v_x_1057_, 0);
lean_inc(v_head_1058_);
v_tail_1059_ = lean_ctor_get(v_x_1057_, 1);
lean_inc(v_tail_1059_);
lean_dec_ref_known(v_x_1057_, 2);
lean_inc_ref(v_fo_1055_);
v___x_1060_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRound___redArg(v_fo_1055_, v_head_1058_, v_x_1056_);
v_x_1056_ = v___x_1060_;
v_x_1057_ = v_tail_1059_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRounds___redArg(lean_object* v_fo_1062_, lean_object* v_state_1063_){
_start:
{
lean_object* v_toRingOps_1064_; lean_object* v_toSemiringOps_1065_; lean_object* v_natCast_1066_; lean_object* v___x_1067_; lean_object* v___x_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; 
v_toRingOps_1064_ = lean_ctor_get(v_fo_1062_, 0);
v_toSemiringOps_1065_ = lean_ctor_get(v_toRingOps_1064_, 0);
v_natCast_1066_ = lean_ctor_get(v_toSemiringOps_1065_, 2);
v___x_1067_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRcNat));
v___x_1068_ = lean_box(0);
lean_inc(v_natCast_1066_);
v___x_1069_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_Generic_internalRounds_spec__0___redArg(v_natCast_1066_, v___x_1067_, v___x_1068_);
v___x_1070_ = lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_internalRounds_spec__1___redArg(v_fo_1062_, v_state_1063_, v___x_1069_);
return v___x_1070_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRounds(lean_object* v_K_1071_, lean_object* v_fo_1072_, lean_object* v_state_1073_){
_start:
{
lean_object* v___x_1074_; 
v___x_1074_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRounds___redArg(v_fo_1072_, v_state_1073_);
return v___x_1074_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_Generic_internalRounds_spec__0(lean_object* v_K_1075_, lean_object* v___x_1076_, lean_object* v_a_1077_, lean_object* v_a_1078_){
_start:
{
lean_object* v___x_1079_; 
v___x_1079_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_Generic_internalRounds_spec__0___redArg(v___x_1076_, v_a_1077_, v_a_1078_);
return v___x_1079_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_internalRounds_spec__1(lean_object* v_K_1080_, lean_object* v_fo_1081_, lean_object* v_x_1082_, lean_object* v_x_1083_){
_start:
{
lean_object* v___x_1084_; 
v___x_1084_ = lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_internalRounds_spec__1___redArg(v_fo_1081_, v_x_1082_, v_x_1083_);
return v___x_1084_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_finalExternalRounds___redArg(lean_object* v_fo_1085_, lean_object* v_state_1086_){
_start:
{
lean_object* v___x_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; lean_object* v___x_1090_; 
v___x_1087_ = ((lean_object*)(lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_externalFinalRcNat));
v___x_1088_ = lean_box(0);
lean_inc_ref(v_fo_1085_);
v___x_1089_ = lp_swirl_x2dfv_List_mapTR_loop___at___00Fundamentals_Poseidon2_Generic_initialExternalRounds_spec__0___redArg(v_fo_1085_, v___x_1087_, v___x_1088_);
v___x_1090_ = lp_swirl_x2dfv_List_foldl___at___00Fundamentals_Poseidon2_Generic_initialExternalRounds_spec__1___redArg(v_fo_1085_, v_state_1086_, v___x_1089_);
return v___x_1090_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_finalExternalRounds(lean_object* v_K_1091_, lean_object* v_fo_1092_, lean_object* v_state_1093_){
_start:
{
lean_object* v___x_1094_; 
v___x_1094_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_finalExternalRounds___redArg(v_fo_1092_, v_state_1093_);
return v___x_1094_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_permute___redArg(lean_object* v_fo_1095_, lean_object* v_state_1096_){
_start:
{
lean_object* v___x_1097_; lean_object* v___x_1098_; lean_object* v___x_1099_; 
lean_inc_ref_n(v_fo_1095_, 2);
v___x_1097_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_initialExternalRounds___redArg(v_fo_1095_, v_state_1096_);
v___x_1098_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_internalRounds___redArg(v_fo_1095_, v___x_1097_);
v___x_1099_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_finalExternalRounds___redArg(v_fo_1095_, v___x_1098_);
return v___x_1099_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_permute(lean_object* v_K_1100_, lean_object* v_fo_1101_, lean_object* v_state_1102_){
_start:
{
lean_object* v___x_1103_; 
v___x_1103_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_permute___redArg(v_fo_1101_, v_state_1102_);
return v___x_1103_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__0(lean_object* v_right_1104_, lean_object* v_fo_1105_, lean_object* v_left_1106_, lean_object* v_idx_1107_){
_start:
{
lean_object* v___x_1108_; uint8_t v___x_1109_; 
v___x_1108_ = lean_unsigned_to_nat(8u);
v___x_1109_ = lean_nat_dec_lt(v_idx_1107_, v___x_1108_);
if (v___x_1109_ == 0)
{
lean_object* v___x_1110_; lean_object* v___x_1111_; lean_object* v___x_1112_; 
lean_dec_ref(v_left_1106_);
v___x_1110_ = lean_array_to_list(v_right_1104_);
v___x_1111_ = lean_nat_sub(v_idx_1107_, v___x_1108_);
lean_dec(v_idx_1107_);
v___x_1112_ = l_List_get_x3fInternal___redArg(v___x_1110_, v___x_1111_);
lean_dec(v___x_1110_);
if (lean_obj_tag(v___x_1112_) == 0)
{
lean_object* v_toRingOps_1113_; lean_object* v_toSemiringOps_1114_; lean_object* v_zero_1115_; 
v_toRingOps_1113_ = lean_ctor_get(v_fo_1105_, 0);
v_toSemiringOps_1114_ = lean_ctor_get(v_toRingOps_1113_, 0);
v_zero_1115_ = lean_ctor_get(v_toSemiringOps_1114_, 0);
lean_inc(v_zero_1115_);
return v_zero_1115_;
}
else
{
lean_object* v_val_1116_; 
v_val_1116_ = lean_ctor_get(v___x_1112_, 0);
lean_inc(v_val_1116_);
lean_dec_ref_known(v___x_1112_, 1);
return v_val_1116_;
}
}
else
{
lean_object* v___x_1117_; lean_object* v___x_1118_; 
lean_dec_ref(v_right_1104_);
v___x_1117_ = lean_array_to_list(v_left_1106_);
v___x_1118_ = l_List_get_x3fInternal___redArg(v___x_1117_, v_idx_1107_);
lean_dec(v___x_1117_);
if (lean_obj_tag(v___x_1118_) == 0)
{
lean_object* v_toRingOps_1119_; lean_object* v_toSemiringOps_1120_; lean_object* v_zero_1121_; 
v_toRingOps_1119_ = lean_ctor_get(v_fo_1105_, 0);
v_toSemiringOps_1120_ = lean_ctor_get(v_toRingOps_1119_, 0);
v_zero_1121_ = lean_ctor_get(v_toSemiringOps_1120_, 0);
lean_inc(v_zero_1121_);
return v_zero_1121_;
}
else
{
lean_object* v_val_1122_; 
v_val_1122_ = lean_ctor_get(v___x_1118_, 0);
lean_inc(v_val_1122_);
lean_dec_ref_known(v___x_1118_, 1);
return v_val_1122_;
}
}
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__0___boxed(lean_object* v_right_1123_, lean_object* v_fo_1124_, lean_object* v_left_1125_, lean_object* v_idx_1126_){
_start:
{
lean_object* v_res_1127_; 
v_res_1127_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__0(v_right_1123_, v_fo_1124_, v_left_1125_, v_idx_1126_);
lean_dec_ref(v_fo_1124_);
return v_res_1127_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__1(lean_object* v_fo_1128_, lean_object* v_output_1129_, lean_object* v_idx_1130_){
_start:
{
lean_object* v___x_1131_; 
v___x_1131_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_1128_, v_output_1129_, v_idx_1130_);
return v___x_1131_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__1___boxed(lean_object* v_fo_1132_, lean_object* v_output_1133_, lean_object* v_idx_1134_){
_start:
{
lean_object* v_res_1135_; 
v_res_1135_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__1(v_fo_1132_, v_output_1133_, v_idx_1134_);
lean_dec_ref(v_fo_1132_);
return v_res_1135_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__2(lean_object* v___x_1136_, lean_object* v_fo_1137_, lean_object* v_output_1138_, lean_object* v_idx_1139_){
_start:
{
lean_object* v___x_1140_; lean_object* v___x_1141_; 
v___x_1140_ = lean_nat_add(v_idx_1139_, v___x_1136_);
v___x_1141_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_stateAt___redArg(v_fo_1137_, v_output_1138_, v___x_1140_);
return v___x_1141_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__2___boxed(lean_object* v___x_1142_, lean_object* v_fo_1143_, lean_object* v_output_1144_, lean_object* v_idx_1145_){
_start:
{
lean_object* v_res_1146_; 
v_res_1146_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__2(v___x_1142_, v_fo_1143_, v_output_1144_, v_idx_1145_);
lean_dec(v_idx_1145_);
lean_dec_ref(v_fo_1143_);
lean_dec(v___x_1142_);
return v_res_1146_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg(lean_object* v_fo_1147_, lean_object* v_left_1148_, lean_object* v_right_1149_){
_start:
{
lean_object* v___f_1150_; lean_object* v___x_1151_; lean_object* v_st_1152_; lean_object* v_output_1153_; lean_object* v___f_1154_; lean_object* v___x_1155_; lean_object* v___f_1156_; lean_object* v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; 
lean_inc_ref_n(v_fo_1147_, 3);
v___f_1150_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_1150_, 0, v_right_1149_);
lean_closure_set(v___f_1150_, 1, v_fo_1147_);
lean_closure_set(v___f_1150_, 2, v_left_1148_);
v___x_1151_ = lean_unsigned_to_nat(16u);
v_st_1152_ = l_Array_ofFn___redArg(v___x_1151_, v___f_1150_);
v_output_1153_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_permute___redArg(v_fo_1147_, v_st_1152_);
lean_inc_ref(v_output_1153_);
v___f_1154_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__1___boxed), 3, 2);
lean_closure_set(v___f_1154_, 0, v_fo_1147_);
lean_closure_set(v___f_1154_, 1, v_output_1153_);
v___x_1155_ = lean_unsigned_to_nat(8u);
v___f_1156_ = lean_alloc_closure((void*)(lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg___lam__2___boxed), 4, 3);
lean_closure_set(v___f_1156_, 0, v___x_1155_);
lean_closure_set(v___f_1156_, 1, v_fo_1147_);
lean_closure_set(v___f_1156_, 2, v_output_1153_);
v___x_1157_ = l_Array_ofFn___redArg(v___x_1155_, v___f_1154_);
v___x_1158_ = l_Array_ofFn___redArg(v___x_1155_, v___f_1156_);
v___x_1159_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1159_, 0, v___x_1157_);
lean_ctor_set(v___x_1159_, 1, v___x_1158_);
return v___x_1159_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity(lean_object* v_K_1160_, lean_object* v_fo_1161_, lean_object* v_left_1162_, lean_object* v_right_1163_){
_start:
{
lean_object* v___x_1164_; 
v___x_1164_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg(v_fo_1161_, v_left_1162_, v_right_1163_);
return v___x_1164_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressDigest___redArg(lean_object* v_fo_1165_, lean_object* v_left_1166_, lean_object* v_right_1167_){
_start:
{
lean_object* v___x_1168_; lean_object* v_fst_1169_; 
v___x_1168_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressWithCapacity___redArg(v_fo_1165_, v_left_1166_, v_right_1167_);
v_fst_1169_ = lean_ctor_get(v___x_1168_, 0);
lean_inc(v_fst_1169_);
lean_dec_ref(v___x_1168_);
return v_fst_1169_;
}
}
LEAN_EXPORT lean_object* lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressDigest(lean_object* v_K_1170_, lean_object* v_fo_1171_, lean_object* v_left_1172_, lean_object* v_right_1173_){
_start:
{
lean_object* v___x_1174_; 
v___x_1174_ = lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_compressDigest___redArg(v_fo_1171_, v_left_1172_, v_right_1173_);
return v___x_1174_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_FieldOps(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_swirl_x2dfv_Fundamentals_Spec_Poseidon2_Generic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize_runtime_module();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_swirl_x2dfv_Fundamentals_Spec_FieldOps(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_WIDTH = _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_WIDTH();
lean_mark_persistent(lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_WIDTH);
lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_RATE = _init_lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_RATE();
lean_mark_persistent(lp_swirl_x2dfv_Fundamentals_Poseidon2_Generic_RATE);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
