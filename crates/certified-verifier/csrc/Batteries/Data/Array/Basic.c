// Lean compiler output
// Module: Batteries.Data.Array.Basic
// Imports: public import Init public meta import Init import Batteries.Tactic.Alias import Batteries.Data.UInt
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
size_t lean_usize_sub(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t l_Array_contains___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Array_reverse___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Array_equalSet___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_equalSet___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Array_equalSet___redArg___lam__1(lean_object*, lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_equalSet___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Array_equalSet___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Array_equalSet___redArg___closed__0 = (const lean_object*)&lp_batteries_Array_equalSet___redArg___closed__0_value;
static const lean_closure_object lp_batteries_Array_equalSet___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Array_equalSet___redArg___closed__1 = (const lean_object*)&lp_batteries_Array_equalSet___redArg___closed__1_value;
static const lean_closure_object lp_batteries_Array_equalSet___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Array_equalSet___redArg___closed__2 = (const lean_object*)&lp_batteries_Array_equalSet___redArg___closed__2_value;
static const lean_closure_object lp_batteries_Array_equalSet___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Array_equalSet___redArg___closed__3 = (const lean_object*)&lp_batteries_Array_equalSet___redArg___closed__3_value;
static const lean_closure_object lp_batteries_Array_equalSet___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Array_equalSet___redArg___closed__4 = (const lean_object*)&lp_batteries_Array_equalSet___redArg___closed__4_value;
static const lean_closure_object lp_batteries_Array_equalSet___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Array_equalSet___redArg___closed__5 = (const lean_object*)&lp_batteries_Array_equalSet___redArg___closed__5_value;
static const lean_closure_object lp_batteries_Array_equalSet___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Array_equalSet___redArg___closed__6 = (const lean_object*)&lp_batteries_Array_equalSet___redArg___closed__6_value;
static const lean_ctor_object lp_batteries_Array_equalSet___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Array_equalSet___redArg___closed__0_value),((lean_object*)&lp_batteries_Array_equalSet___redArg___closed__1_value)}};
static const lean_object* lp_batteries_Array_equalSet___redArg___closed__7 = (const lean_object*)&lp_batteries_Array_equalSet___redArg___closed__7_value;
static const lean_ctor_object lp_batteries_Array_equalSet___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Array_equalSet___redArg___closed__7_value),((lean_object*)&lp_batteries_Array_equalSet___redArg___closed__2_value),((lean_object*)&lp_batteries_Array_equalSet___redArg___closed__3_value),((lean_object*)&lp_batteries_Array_equalSet___redArg___closed__4_value),((lean_object*)&lp_batteries_Array_equalSet___redArg___closed__5_value)}};
static const lean_object* lp_batteries_Array_equalSet___redArg___closed__8 = (const lean_object*)&lp_batteries_Array_equalSet___redArg___closed__8_value;
static const lean_ctor_object lp_batteries_Array_equalSet___redArg___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_batteries_Array_equalSet___redArg___closed__8_value),((lean_object*)&lp_batteries_Array_equalSet___redArg___closed__6_value)}};
static const lean_object* lp_batteries_Array_equalSet___redArg___closed__9 = (const lean_object*)&lp_batteries_Array_equalSet___redArg___closed__9_value;
LEAN_EXPORT uint8_t lp_batteries_Array_equalSet___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_equalSet___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Array_equalSet(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_equalSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinWith___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinWith___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinWith___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinWith___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_minWith___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_minWith___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_minWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_minWith___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinD___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinD___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinD(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinD___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_minD___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_minD___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_minD(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_minD___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMin_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMin_x3f___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMin_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMin_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinI___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinI___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinI(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinI___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_minI___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_minI___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_minI(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_minI___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxWith___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxWith___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxWith___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxWith___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_maxWith___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_maxWith___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_maxWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_maxWith___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxD___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxD___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxD(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxD___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_maxD___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_maxD___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_maxD(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_maxD___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMax_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMax_x3f___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMax_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMax_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxI___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxI___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxI(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxI___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_maxI___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_maxI___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_maxI(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_maxI___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_setN___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_setN___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_setN(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_setN___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg___lam__0(size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_scanlM_loop___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_scanlM_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_scanlM_loop___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_scanlM_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg___lam__0(size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMUnsafe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_scanrM_loop___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_scanrM_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_scanrM_loop___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_scanrM_loop___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_scanrM_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_scanrM_loop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_scanl___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_scanl___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_scanl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_scanr___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Array_scanr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Subarray_scanlM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Subarray_scanlM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Subarray_scanrM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Subarray_scanrM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Subarray_scanl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Subarray_scanl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Subarray_scanr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Subarray_scanr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Subarray_isEmpty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Subarray_isEmpty___redArg___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Subarray_isEmpty(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Subarray_isEmpty___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Subarray_contains___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Subarray_contains___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Subarray_contains___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Subarray_contains___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Subarray_contains(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Subarray_contains___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Subarray_popHead_x3f___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Subarray_popHead_x3f(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Array_equalSet___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_xs_2_, lean_object* v_v_3_){
_start:
{
uint8_t v___x_4_; 
v___x_4_ = l_Array_contains___redArg(v_inst_1_, v_xs_2_, v_v_3_);
if (v___x_4_ == 0)
{
uint8_t v___x_5_; 
v___x_5_ = 1;
return v___x_5_;
}
else
{
uint8_t v___x_6_; 
v___x_6_ = 0;
return v___x_6_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_equalSet___redArg___lam__0___boxed(lean_object* v_inst_7_, lean_object* v_xs_8_, lean_object* v_v_9_){
_start:
{
uint8_t v_res_10_; lean_object* v_r_11_; 
v_res_10_ = lp_batteries_Array_equalSet___redArg___lam__0(v_inst_7_, v_xs_8_, v_v_9_);
v_r_11_ = lean_box(v_res_10_);
return v_r_11_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Array_equalSet___redArg___lam__1(lean_object* v_inst_12_, lean_object* v_ys_13_, uint8_t v___x_14_, lean_object* v_v_15_){
_start:
{
uint8_t v___x_16_; 
v___x_16_ = l_Array_contains___redArg(v_inst_12_, v_ys_13_, v_v_15_);
if (v___x_16_ == 0)
{
return v___x_14_;
}
else
{
uint8_t v___x_17_; 
v___x_17_ = 0;
return v___x_17_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_equalSet___redArg___lam__1___boxed(lean_object* v_inst_18_, lean_object* v_ys_19_, lean_object* v___x_20_, lean_object* v_v_21_){
_start:
{
uint8_t v___x_169__boxed_22_; uint8_t v_res_23_; lean_object* v_r_24_; 
v___x_169__boxed_22_ = lean_unbox(v___x_20_);
v_res_23_ = lp_batteries_Array_equalSet___redArg___lam__1(v_inst_18_, v_ys_19_, v___x_169__boxed_22_, v_v_21_);
v_r_24_ = lean_box(v_res_23_);
return v_r_24_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Array_equalSet___redArg(lean_object* v_inst_44_, lean_object* v_xs_45_, lean_object* v_ys_46_){
_start:
{
lean_object* v___f_47_; lean_object* v___x_59_; lean_object* v___x_60_; lean_object* v___x_61_; uint8_t v___x_62_; 
lean_inc_ref(v_xs_45_);
lean_inc_ref(v_inst_44_);
v___f_47_ = lean_alloc_closure((void*)(lp_batteries_Array_equalSet___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_47_, 0, v_inst_44_);
lean_closure_set(v___f_47_, 1, v_xs_45_);
v___x_59_ = lean_unsigned_to_nat(0u);
v___x_60_ = lean_array_get_size(v_xs_45_);
v___x_61_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_62_ = lean_nat_dec_lt(v___x_59_, v___x_60_);
if (v___x_62_ == 0)
{
lean_dec_ref(v_xs_45_);
lean_dec_ref(v_inst_44_);
goto v___jp_48_;
}
else
{
if (v___x_62_ == 0)
{
lean_dec_ref(v_xs_45_);
lean_dec_ref(v_inst_44_);
goto v___jp_48_;
}
else
{
lean_object* v___x_63_; lean_object* v___f_64_; size_t v___x_65_; size_t v___x_66_; lean_object* v___x_67_; uint8_t v___x_68_; 
v___x_63_ = lean_box(v___x_62_);
lean_inc_ref(v_ys_46_);
v___f_64_ = lean_alloc_closure((void*)(lp_batteries_Array_equalSet___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_64_, 0, v_inst_44_);
lean_closure_set(v___f_64_, 1, v_ys_46_);
lean_closure_set(v___f_64_, 2, v___x_63_);
v___x_65_ = ((size_t)0ULL);
v___x_66_ = lean_usize_of_nat(v___x_60_);
v___x_67_ = l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_box(0), lean_box(0), v___x_61_, v___f_64_, v_xs_45_, v___x_65_, v___x_66_);
v___x_68_ = lean_unbox(v___x_67_);
lean_dec(v___x_67_);
if (v___x_68_ == 0)
{
goto v___jp_48_;
}
else
{
uint8_t v___x_69_; 
lean_dec_ref(v___f_47_);
lean_dec_ref(v_ys_46_);
v___x_69_ = 0;
return v___x_69_;
}
}
}
v___jp_48_:
{
lean_object* v___x_49_; lean_object* v___x_50_; lean_object* v___x_51_; uint8_t v___x_52_; 
v___x_49_ = lean_unsigned_to_nat(0u);
v___x_50_ = lean_array_get_size(v_ys_46_);
v___x_51_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_52_ = lean_nat_dec_lt(v___x_49_, v___x_50_);
if (v___x_52_ == 0)
{
uint8_t v___x_53_; 
lean_dec_ref(v___f_47_);
lean_dec_ref(v_ys_46_);
v___x_53_ = 1;
return v___x_53_;
}
else
{
if (v___x_52_ == 0)
{
lean_dec_ref(v___f_47_);
lean_dec_ref(v_ys_46_);
return v___x_52_;
}
else
{
size_t v___x_54_; size_t v___x_55_; lean_object* v___x_56_; uint8_t v___x_57_; 
v___x_54_ = ((size_t)0ULL);
v___x_55_ = lean_usize_of_nat(v___x_50_);
v___x_56_ = l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_box(0), lean_box(0), v___x_51_, v___f_47_, v_ys_46_, v___x_54_, v___x_55_);
v___x_57_ = lean_unbox(v___x_56_);
lean_dec(v___x_56_);
if (v___x_57_ == 0)
{
return v___x_52_;
}
else
{
uint8_t v___x_58_; 
v___x_58_ = 0;
return v___x_58_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_equalSet___redArg___boxed(lean_object* v_inst_70_, lean_object* v_xs_71_, lean_object* v_ys_72_){
_start:
{
uint8_t v_res_73_; lean_object* v_r_74_; 
v_res_73_ = lp_batteries_Array_equalSet___redArg(v_inst_70_, v_xs_71_, v_ys_72_);
v_r_74_ = lean_box(v_res_73_);
return v_r_74_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Array_equalSet(lean_object* v_00_u03b1_75_, lean_object* v_inst_76_, lean_object* v_xs_77_, lean_object* v_ys_78_){
_start:
{
uint8_t v___x_79_; 
v___x_79_ = lp_batteries_Array_equalSet___redArg(v_inst_76_, v_xs_77_, v_ys_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_equalSet___boxed(lean_object* v_00_u03b1_80_, lean_object* v_inst_81_, lean_object* v_xs_82_, lean_object* v_ys_83_){
_start:
{
uint8_t v_res_84_; lean_object* v_r_85_; 
v_res_84_ = lp_batteries_Array_equalSet(v_00_u03b1_80_, v_inst_81_, v_xs_82_, v_ys_83_);
v_r_85_ = lean_box(v_res_84_);
return v_r_85_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinWith___redArg___lam__0(lean_object* v_ord_86_, lean_object* v_x1_87_, lean_object* v_x2_88_){
_start:
{
lean_object* v___x_89_; uint8_t v___x_90_; 
lean_inc(v_x1_87_);
lean_inc(v_x2_88_);
v___x_89_ = lean_apply_2(v_ord_86_, v_x2_88_, v_x1_87_);
v___x_90_ = lean_unbox(v___x_89_);
if (v___x_90_ == 0)
{
lean_dec(v_x1_87_);
return v_x2_88_;
}
else
{
lean_dec(v_x2_88_);
return v_x1_87_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinWith___redArg(lean_object* v_ord_91_, lean_object* v_xs_92_, lean_object* v_d_93_, lean_object* v_start_94_, lean_object* v_stop_95_){
_start:
{
lean_object* v___x_96_; uint8_t v___x_97_; 
v___x_96_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_97_ = lean_nat_dec_lt(v_start_94_, v_stop_95_);
if (v___x_97_ == 0)
{
lean_dec_ref(v_xs_92_);
lean_dec_ref(v_ord_91_);
return v_d_93_;
}
else
{
lean_object* v___f_98_; lean_object* v___x_99_; uint8_t v___x_100_; 
v___f_98_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMinWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_98_, 0, v_ord_91_);
v___x_99_ = lean_array_get_size(v_xs_92_);
v___x_100_ = lean_nat_dec_le(v_stop_95_, v___x_99_);
if (v___x_100_ == 0)
{
uint8_t v___x_101_; 
v___x_101_ = lean_nat_dec_lt(v_start_94_, v___x_99_);
if (v___x_101_ == 0)
{
lean_dec_ref(v___f_98_);
lean_dec_ref(v_xs_92_);
return v_d_93_;
}
else
{
size_t v___x_102_; size_t v___x_103_; lean_object* v___x_104_; 
v___x_102_ = lean_usize_of_nat(v_start_94_);
v___x_103_ = lean_usize_of_nat(v___x_99_);
v___x_104_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_96_, v___f_98_, v_xs_92_, v___x_102_, v___x_103_, v_d_93_);
return v___x_104_;
}
}
else
{
size_t v___x_105_; size_t v___x_106_; lean_object* v___x_107_; 
v___x_105_ = lean_usize_of_nat(v_start_94_);
v___x_106_ = lean_usize_of_nat(v_stop_95_);
v___x_107_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_96_, v___f_98_, v_xs_92_, v___x_105_, v___x_106_, v_d_93_);
return v___x_107_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinWith___redArg___boxed(lean_object* v_ord_108_, lean_object* v_xs_109_, lean_object* v_d_110_, lean_object* v_start_111_, lean_object* v_stop_112_){
_start:
{
lean_object* v_res_113_; 
v_res_113_ = lp_batteries_Array_rangeMinWith___redArg(v_ord_108_, v_xs_109_, v_d_110_, v_start_111_, v_stop_112_);
lean_dec(v_stop_112_);
lean_dec(v_start_111_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinWith(lean_object* v_00_u03b1_114_, lean_object* v_ord_115_, lean_object* v_xs_116_, lean_object* v_d_117_, lean_object* v_start_118_, lean_object* v_stop_119_){
_start:
{
lean_object* v___x_120_; uint8_t v___x_121_; 
v___x_120_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_121_ = lean_nat_dec_lt(v_start_118_, v_stop_119_);
if (v___x_121_ == 0)
{
lean_dec_ref(v_xs_116_);
lean_dec_ref(v_ord_115_);
return v_d_117_;
}
else
{
lean_object* v___f_122_; lean_object* v___x_123_; uint8_t v___x_124_; 
v___f_122_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMinWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_122_, 0, v_ord_115_);
v___x_123_ = lean_array_get_size(v_xs_116_);
v___x_124_ = lean_nat_dec_le(v_stop_119_, v___x_123_);
if (v___x_124_ == 0)
{
uint8_t v___x_125_; 
v___x_125_ = lean_nat_dec_lt(v_start_118_, v___x_123_);
if (v___x_125_ == 0)
{
lean_dec_ref(v___f_122_);
lean_dec_ref(v_xs_116_);
return v_d_117_;
}
else
{
size_t v___x_126_; size_t v___x_127_; lean_object* v___x_128_; 
v___x_126_ = lean_usize_of_nat(v_start_118_);
v___x_127_ = lean_usize_of_nat(v___x_123_);
v___x_128_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_120_, v___f_122_, v_xs_116_, v___x_126_, v___x_127_, v_d_117_);
return v___x_128_;
}
}
else
{
size_t v___x_129_; size_t v___x_130_; lean_object* v___x_131_; 
v___x_129_ = lean_usize_of_nat(v_start_118_);
v___x_130_ = lean_usize_of_nat(v_stop_119_);
v___x_131_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_120_, v___f_122_, v_xs_116_, v___x_129_, v___x_130_, v_d_117_);
return v___x_131_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinWith___boxed(lean_object* v_00_u03b1_132_, lean_object* v_ord_133_, lean_object* v_xs_134_, lean_object* v_d_135_, lean_object* v_start_136_, lean_object* v_stop_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_batteries_Array_rangeMinWith(v_00_u03b1_132_, v_ord_133_, v_xs_134_, v_d_135_, v_start_136_, v_stop_137_);
lean_dec(v_stop_137_);
lean_dec(v_start_136_);
return v_res_138_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_minWith___redArg(lean_object* v_ord_139_, lean_object* v_xs_140_, lean_object* v_d_141_, lean_object* v_start_142_, lean_object* v_stop_143_){
_start:
{
lean_object* v___x_144_; uint8_t v___x_145_; 
v___x_144_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_145_ = lean_nat_dec_lt(v_start_142_, v_stop_143_);
if (v___x_145_ == 0)
{
lean_dec_ref(v_xs_140_);
lean_dec_ref(v_ord_139_);
return v_d_141_;
}
else
{
lean_object* v___f_146_; lean_object* v___x_147_; uint8_t v___x_148_; 
v___f_146_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMinWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_146_, 0, v_ord_139_);
v___x_147_ = lean_array_get_size(v_xs_140_);
v___x_148_ = lean_nat_dec_le(v_stop_143_, v___x_147_);
if (v___x_148_ == 0)
{
uint8_t v___x_149_; 
v___x_149_ = lean_nat_dec_lt(v_start_142_, v___x_147_);
if (v___x_149_ == 0)
{
lean_dec_ref(v___f_146_);
lean_dec_ref(v_xs_140_);
return v_d_141_;
}
else
{
size_t v___x_150_; size_t v___x_151_; lean_object* v___x_152_; 
v___x_150_ = lean_usize_of_nat(v_start_142_);
v___x_151_ = lean_usize_of_nat(v___x_147_);
v___x_152_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_144_, v___f_146_, v_xs_140_, v___x_150_, v___x_151_, v_d_141_);
return v___x_152_;
}
}
else
{
size_t v___x_153_; size_t v___x_154_; lean_object* v___x_155_; 
v___x_153_ = lean_usize_of_nat(v_start_142_);
v___x_154_ = lean_usize_of_nat(v_stop_143_);
v___x_155_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_144_, v___f_146_, v_xs_140_, v___x_153_, v___x_154_, v_d_141_);
return v___x_155_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_minWith___redArg___boxed(lean_object* v_ord_156_, lean_object* v_xs_157_, lean_object* v_d_158_, lean_object* v_start_159_, lean_object* v_stop_160_){
_start:
{
lean_object* v_res_161_; 
v_res_161_ = lp_batteries_Array_minWith___redArg(v_ord_156_, v_xs_157_, v_d_158_, v_start_159_, v_stop_160_);
lean_dec(v_stop_160_);
lean_dec(v_start_159_);
return v_res_161_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_minWith(lean_object* v_00_u03b1_162_, lean_object* v_ord_163_, lean_object* v_xs_164_, lean_object* v_d_165_, lean_object* v_start_166_, lean_object* v_stop_167_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lp_batteries_Array_minWith___redArg(v_ord_163_, v_xs_164_, v_d_165_, v_start_166_, v_stop_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_minWith___boxed(lean_object* v_00_u03b1_169_, lean_object* v_ord_170_, lean_object* v_xs_171_, lean_object* v_d_172_, lean_object* v_start_173_, lean_object* v_stop_174_){
_start:
{
lean_object* v_res_175_; 
v_res_175_ = lp_batteries_Array_minWith(v_00_u03b1_169_, v_ord_170_, v_xs_171_, v_d_172_, v_start_173_, v_stop_174_);
lean_dec(v_stop_174_);
lean_dec(v_start_173_);
return v_res_175_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinD___redArg(lean_object* v_ord_176_, lean_object* v_xs_177_, lean_object* v_d_178_, lean_object* v_start_179_, lean_object* v_stop_180_){
_start:
{
lean_object* v___x_181_; uint8_t v___x_182_; 
v___x_181_ = lean_array_get_size(v_xs_177_);
v___x_182_ = lean_nat_dec_lt(v_start_179_, v___x_181_);
if (v___x_182_ == 0)
{
lean_dec_ref(v_xs_177_);
lean_dec_ref(v_ord_176_);
lean_inc(v_d_178_);
return v_d_178_;
}
else
{
uint8_t v___x_183_; 
v___x_183_ = lean_nat_dec_lt(v_start_179_, v_stop_180_);
if (v___x_183_ == 0)
{
lean_dec_ref(v_xs_177_);
lean_dec_ref(v_ord_176_);
lean_inc(v_d_178_);
return v_d_178_;
}
else
{
lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; uint8_t v___x_188_; 
v___x_184_ = lean_array_fget(v_xs_177_, v_start_179_);
v___x_185_ = lean_unsigned_to_nat(1u);
v___x_186_ = lean_nat_add(v_start_179_, v___x_185_);
v___x_187_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_188_ = lean_nat_dec_lt(v___x_186_, v_stop_180_);
if (v___x_188_ == 0)
{
lean_dec(v___x_186_);
lean_dec_ref(v_xs_177_);
lean_dec_ref(v_ord_176_);
return v___x_184_;
}
else
{
lean_object* v___f_189_; uint8_t v___x_190_; 
v___f_189_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMinWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_189_, 0, v_ord_176_);
v___x_190_ = lean_nat_dec_le(v_stop_180_, v___x_181_);
if (v___x_190_ == 0)
{
uint8_t v___x_191_; 
v___x_191_ = lean_nat_dec_lt(v___x_186_, v___x_181_);
if (v___x_191_ == 0)
{
lean_dec_ref(v___f_189_);
lean_dec(v___x_186_);
lean_dec_ref(v_xs_177_);
return v___x_184_;
}
else
{
size_t v___x_192_; size_t v___x_193_; lean_object* v___x_194_; 
v___x_192_ = lean_usize_of_nat(v___x_186_);
lean_dec(v___x_186_);
v___x_193_ = lean_usize_of_nat(v___x_181_);
v___x_194_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_187_, v___f_189_, v_xs_177_, v___x_192_, v___x_193_, v___x_184_);
return v___x_194_;
}
}
else
{
size_t v___x_195_; size_t v___x_196_; lean_object* v___x_197_; 
v___x_195_ = lean_usize_of_nat(v___x_186_);
lean_dec(v___x_186_);
v___x_196_ = lean_usize_of_nat(v_stop_180_);
v___x_197_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_187_, v___f_189_, v_xs_177_, v___x_195_, v___x_196_, v___x_184_);
return v___x_197_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinD___redArg___boxed(lean_object* v_ord_198_, lean_object* v_xs_199_, lean_object* v_d_200_, lean_object* v_start_201_, lean_object* v_stop_202_){
_start:
{
lean_object* v_res_203_; 
v_res_203_ = lp_batteries_Array_rangeMinD___redArg(v_ord_198_, v_xs_199_, v_d_200_, v_start_201_, v_stop_202_);
lean_dec(v_stop_202_);
lean_dec(v_start_201_);
lean_dec(v_d_200_);
return v_res_203_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinD(lean_object* v_00_u03b1_204_, lean_object* v_ord_205_, lean_object* v_xs_206_, lean_object* v_d_207_, lean_object* v_start_208_, lean_object* v_stop_209_){
_start:
{
lean_object* v___x_210_; uint8_t v___x_211_; 
v___x_210_ = lean_array_get_size(v_xs_206_);
v___x_211_ = lean_nat_dec_lt(v_start_208_, v___x_210_);
if (v___x_211_ == 0)
{
lean_dec_ref(v_xs_206_);
lean_dec_ref(v_ord_205_);
lean_inc(v_d_207_);
return v_d_207_;
}
else
{
uint8_t v___x_212_; 
v___x_212_ = lean_nat_dec_lt(v_start_208_, v_stop_209_);
if (v___x_212_ == 0)
{
lean_dec_ref(v_xs_206_);
lean_dec_ref(v_ord_205_);
lean_inc(v_d_207_);
return v_d_207_;
}
else
{
lean_object* v___x_213_; lean_object* v___x_214_; lean_object* v___x_215_; lean_object* v___x_216_; uint8_t v___x_217_; 
v___x_213_ = lean_array_fget(v_xs_206_, v_start_208_);
v___x_214_ = lean_unsigned_to_nat(1u);
v___x_215_ = lean_nat_add(v_start_208_, v___x_214_);
v___x_216_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_217_ = lean_nat_dec_lt(v___x_215_, v_stop_209_);
if (v___x_217_ == 0)
{
lean_dec(v___x_215_);
lean_dec_ref(v_xs_206_);
lean_dec_ref(v_ord_205_);
return v___x_213_;
}
else
{
lean_object* v___f_218_; uint8_t v___x_219_; 
v___f_218_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMinWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_218_, 0, v_ord_205_);
v___x_219_ = lean_nat_dec_le(v_stop_209_, v___x_210_);
if (v___x_219_ == 0)
{
uint8_t v___x_220_; 
v___x_220_ = lean_nat_dec_lt(v___x_215_, v___x_210_);
if (v___x_220_ == 0)
{
lean_dec_ref(v___f_218_);
lean_dec(v___x_215_);
lean_dec_ref(v_xs_206_);
return v___x_213_;
}
else
{
size_t v___x_221_; size_t v___x_222_; lean_object* v___x_223_; 
v___x_221_ = lean_usize_of_nat(v___x_215_);
lean_dec(v___x_215_);
v___x_222_ = lean_usize_of_nat(v___x_210_);
v___x_223_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_216_, v___f_218_, v_xs_206_, v___x_221_, v___x_222_, v___x_213_);
return v___x_223_;
}
}
else
{
size_t v___x_224_; size_t v___x_225_; lean_object* v___x_226_; 
v___x_224_ = lean_usize_of_nat(v___x_215_);
lean_dec(v___x_215_);
v___x_225_ = lean_usize_of_nat(v_stop_209_);
v___x_226_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_216_, v___f_218_, v_xs_206_, v___x_224_, v___x_225_, v___x_213_);
return v___x_226_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinD___boxed(lean_object* v_00_u03b1_227_, lean_object* v_ord_228_, lean_object* v_xs_229_, lean_object* v_d_230_, lean_object* v_start_231_, lean_object* v_stop_232_){
_start:
{
lean_object* v_res_233_; 
v_res_233_ = lp_batteries_Array_rangeMinD(v_00_u03b1_227_, v_ord_228_, v_xs_229_, v_d_230_, v_start_231_, v_stop_232_);
lean_dec(v_stop_232_);
lean_dec(v_start_231_);
lean_dec(v_d_230_);
return v_res_233_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_minD___redArg(lean_object* v_ord_234_, lean_object* v_xs_235_, lean_object* v_d_236_, lean_object* v_start_237_, lean_object* v_stop_238_){
_start:
{
lean_object* v___x_239_; uint8_t v___x_240_; 
v___x_239_ = lean_array_get_size(v_xs_235_);
v___x_240_ = lean_nat_dec_lt(v_start_237_, v___x_239_);
if (v___x_240_ == 0)
{
lean_dec_ref(v_xs_235_);
lean_dec_ref(v_ord_234_);
lean_inc(v_d_236_);
return v_d_236_;
}
else
{
uint8_t v___x_241_; 
v___x_241_ = lean_nat_dec_lt(v_start_237_, v_stop_238_);
if (v___x_241_ == 0)
{
lean_dec_ref(v_xs_235_);
lean_dec_ref(v_ord_234_);
lean_inc(v_d_236_);
return v_d_236_;
}
else
{
lean_object* v___x_242_; lean_object* v___x_243_; lean_object* v___x_244_; lean_object* v___x_245_; uint8_t v___x_246_; 
v___x_242_ = lean_array_fget(v_xs_235_, v_start_237_);
v___x_243_ = lean_unsigned_to_nat(1u);
v___x_244_ = lean_nat_add(v_start_237_, v___x_243_);
v___x_245_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_246_ = lean_nat_dec_lt(v___x_244_, v_stop_238_);
if (v___x_246_ == 0)
{
lean_dec(v___x_244_);
lean_dec_ref(v_xs_235_);
lean_dec_ref(v_ord_234_);
return v___x_242_;
}
else
{
lean_object* v___f_247_; uint8_t v___x_248_; 
v___f_247_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMinWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_247_, 0, v_ord_234_);
v___x_248_ = lean_nat_dec_le(v_stop_238_, v___x_239_);
if (v___x_248_ == 0)
{
uint8_t v___x_249_; 
v___x_249_ = lean_nat_dec_lt(v___x_244_, v___x_239_);
if (v___x_249_ == 0)
{
lean_dec_ref(v___f_247_);
lean_dec(v___x_244_);
lean_dec_ref(v_xs_235_);
return v___x_242_;
}
else
{
size_t v___x_250_; size_t v___x_251_; lean_object* v___x_252_; 
v___x_250_ = lean_usize_of_nat(v___x_244_);
lean_dec(v___x_244_);
v___x_251_ = lean_usize_of_nat(v___x_239_);
v___x_252_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_245_, v___f_247_, v_xs_235_, v___x_250_, v___x_251_, v___x_242_);
return v___x_252_;
}
}
else
{
size_t v___x_253_; size_t v___x_254_; lean_object* v___x_255_; 
v___x_253_ = lean_usize_of_nat(v___x_244_);
lean_dec(v___x_244_);
v___x_254_ = lean_usize_of_nat(v_stop_238_);
v___x_255_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_245_, v___f_247_, v_xs_235_, v___x_253_, v___x_254_, v___x_242_);
return v___x_255_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_minD___redArg___boxed(lean_object* v_ord_256_, lean_object* v_xs_257_, lean_object* v_d_258_, lean_object* v_start_259_, lean_object* v_stop_260_){
_start:
{
lean_object* v_res_261_; 
v_res_261_ = lp_batteries_Array_minD___redArg(v_ord_256_, v_xs_257_, v_d_258_, v_start_259_, v_stop_260_);
lean_dec(v_stop_260_);
lean_dec(v_start_259_);
lean_dec(v_d_258_);
return v_res_261_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_minD(lean_object* v_00_u03b1_262_, lean_object* v_ord_263_, lean_object* v_xs_264_, lean_object* v_d_265_, lean_object* v_start_266_, lean_object* v_stop_267_){
_start:
{
lean_object* v___x_268_; 
v___x_268_ = lp_batteries_Array_minD___redArg(v_ord_263_, v_xs_264_, v_d_265_, v_start_266_, v_stop_267_);
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_minD___boxed(lean_object* v_00_u03b1_269_, lean_object* v_ord_270_, lean_object* v_xs_271_, lean_object* v_d_272_, lean_object* v_start_273_, lean_object* v_stop_274_){
_start:
{
lean_object* v_res_275_; 
v_res_275_ = lp_batteries_Array_minD(v_00_u03b1_269_, v_ord_270_, v_xs_271_, v_d_272_, v_start_273_, v_stop_274_);
lean_dec(v_stop_274_);
lean_dec(v_start_273_);
lean_dec(v_d_272_);
return v_res_275_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMin_x3f___redArg(lean_object* v_ord_276_, lean_object* v_xs_277_, lean_object* v_start_278_, lean_object* v_stop_279_){
_start:
{
lean_object* v___x_280_; uint8_t v___x_281_; 
v___x_280_ = lean_array_get_size(v_xs_277_);
v___x_281_ = lean_nat_dec_lt(v_start_278_, v___x_280_);
if (v___x_281_ == 0)
{
lean_object* v___x_282_; 
lean_dec_ref(v_xs_277_);
lean_dec_ref(v_ord_276_);
v___x_282_ = lean_box(0);
return v___x_282_;
}
else
{
uint8_t v___x_283_; 
v___x_283_ = lean_nat_dec_lt(v_start_278_, v_stop_279_);
if (v___x_283_ == 0)
{
lean_object* v___x_284_; 
lean_dec_ref(v_xs_277_);
lean_dec_ref(v_ord_276_);
v___x_284_ = lean_box(0);
return v___x_284_;
}
else
{
lean_object* v___x_285_; 
v___x_285_ = lean_array_fget(v_xs_277_, v_start_278_);
if (v___x_281_ == 0)
{
lean_object* v___x_286_; 
lean_dec_ref(v_xs_277_);
lean_dec_ref(v_ord_276_);
v___x_286_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_286_, 0, v___x_285_);
return v___x_286_;
}
else
{
if (v___x_283_ == 0)
{
lean_object* v___x_287_; 
lean_dec_ref(v_xs_277_);
lean_dec_ref(v_ord_276_);
v___x_287_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_287_, 0, v___x_285_);
return v___x_287_;
}
else
{
lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; uint8_t v___x_291_; 
v___x_288_ = lean_unsigned_to_nat(1u);
v___x_289_ = lean_nat_add(v_start_278_, v___x_288_);
v___x_290_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_291_ = lean_nat_dec_lt(v___x_289_, v_stop_279_);
if (v___x_291_ == 0)
{
lean_object* v___x_292_; 
lean_dec(v___x_289_);
lean_dec_ref(v_xs_277_);
lean_dec_ref(v_ord_276_);
v___x_292_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_292_, 0, v___x_285_);
return v___x_292_;
}
else
{
lean_object* v___f_293_; uint8_t v___x_294_; 
v___f_293_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMinWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_293_, 0, v_ord_276_);
v___x_294_ = lean_nat_dec_le(v_stop_279_, v___x_280_);
if (v___x_294_ == 0)
{
uint8_t v___x_295_; 
v___x_295_ = lean_nat_dec_lt(v___x_289_, v___x_280_);
if (v___x_295_ == 0)
{
lean_object* v___x_296_; 
lean_dec_ref(v___f_293_);
lean_dec(v___x_289_);
lean_dec_ref(v_xs_277_);
v___x_296_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_296_, 0, v___x_285_);
return v___x_296_;
}
else
{
size_t v___x_297_; size_t v___x_298_; lean_object* v___x_299_; lean_object* v___x_300_; 
v___x_297_ = lean_usize_of_nat(v___x_289_);
lean_dec(v___x_289_);
v___x_298_ = lean_usize_of_nat(v___x_280_);
v___x_299_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_290_, v___f_293_, v_xs_277_, v___x_297_, v___x_298_, v___x_285_);
v___x_300_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_300_, 0, v___x_299_);
return v___x_300_;
}
}
else
{
size_t v___x_301_; size_t v___x_302_; lean_object* v___x_303_; lean_object* v___x_304_; 
v___x_301_ = lean_usize_of_nat(v___x_289_);
lean_dec(v___x_289_);
v___x_302_ = lean_usize_of_nat(v_stop_279_);
v___x_303_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_290_, v___f_293_, v_xs_277_, v___x_301_, v___x_302_, v___x_285_);
v___x_304_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_304_, 0, v___x_303_);
return v___x_304_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMin_x3f___redArg___boxed(lean_object* v_ord_305_, lean_object* v_xs_306_, lean_object* v_start_307_, lean_object* v_stop_308_){
_start:
{
lean_object* v_res_309_; 
v_res_309_ = lp_batteries_Array_rangeMin_x3f___redArg(v_ord_305_, v_xs_306_, v_start_307_, v_stop_308_);
lean_dec(v_stop_308_);
lean_dec(v_start_307_);
return v_res_309_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMin_x3f(lean_object* v_00_u03b1_310_, lean_object* v_ord_311_, lean_object* v_xs_312_, lean_object* v_start_313_, lean_object* v_stop_314_){
_start:
{
lean_object* v___x_315_; uint8_t v___x_316_; 
v___x_315_ = lean_array_get_size(v_xs_312_);
v___x_316_ = lean_nat_dec_lt(v_start_313_, v___x_315_);
if (v___x_316_ == 0)
{
lean_object* v___x_317_; 
lean_dec_ref(v_xs_312_);
lean_dec_ref(v_ord_311_);
v___x_317_ = lean_box(0);
return v___x_317_;
}
else
{
uint8_t v___x_318_; 
v___x_318_ = lean_nat_dec_lt(v_start_313_, v_stop_314_);
if (v___x_318_ == 0)
{
lean_object* v___x_319_; 
lean_dec_ref(v_xs_312_);
lean_dec_ref(v_ord_311_);
v___x_319_ = lean_box(0);
return v___x_319_;
}
else
{
lean_object* v___x_320_; 
v___x_320_ = lean_array_fget(v_xs_312_, v_start_313_);
if (v___x_316_ == 0)
{
lean_object* v___x_321_; 
lean_dec_ref(v_xs_312_);
lean_dec_ref(v_ord_311_);
v___x_321_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_321_, 0, v___x_320_);
return v___x_321_;
}
else
{
if (v___x_318_ == 0)
{
lean_object* v___x_322_; 
lean_dec_ref(v_xs_312_);
lean_dec_ref(v_ord_311_);
v___x_322_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_322_, 0, v___x_320_);
return v___x_322_;
}
else
{
lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; uint8_t v___x_326_; 
v___x_323_ = lean_unsigned_to_nat(1u);
v___x_324_ = lean_nat_add(v_start_313_, v___x_323_);
v___x_325_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_326_ = lean_nat_dec_lt(v___x_324_, v_stop_314_);
if (v___x_326_ == 0)
{
lean_object* v___x_327_; 
lean_dec(v___x_324_);
lean_dec_ref(v_xs_312_);
lean_dec_ref(v_ord_311_);
v___x_327_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_327_, 0, v___x_320_);
return v___x_327_;
}
else
{
lean_object* v___f_328_; uint8_t v___x_329_; 
v___f_328_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMinWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_328_, 0, v_ord_311_);
v___x_329_ = lean_nat_dec_le(v_stop_314_, v___x_315_);
if (v___x_329_ == 0)
{
uint8_t v___x_330_; 
v___x_330_ = lean_nat_dec_lt(v___x_324_, v___x_315_);
if (v___x_330_ == 0)
{
lean_object* v___x_331_; 
lean_dec_ref(v___f_328_);
lean_dec(v___x_324_);
lean_dec_ref(v_xs_312_);
v___x_331_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_331_, 0, v___x_320_);
return v___x_331_;
}
else
{
size_t v___x_332_; size_t v___x_333_; lean_object* v___x_334_; lean_object* v___x_335_; 
v___x_332_ = lean_usize_of_nat(v___x_324_);
lean_dec(v___x_324_);
v___x_333_ = lean_usize_of_nat(v___x_315_);
v___x_334_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_325_, v___f_328_, v_xs_312_, v___x_332_, v___x_333_, v___x_320_);
v___x_335_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_335_, 0, v___x_334_);
return v___x_335_;
}
}
else
{
size_t v___x_336_; size_t v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; 
v___x_336_ = lean_usize_of_nat(v___x_324_);
lean_dec(v___x_324_);
v___x_337_ = lean_usize_of_nat(v_stop_314_);
v___x_338_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_325_, v___f_328_, v_xs_312_, v___x_336_, v___x_337_, v___x_320_);
v___x_339_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_339_, 0, v___x_338_);
return v___x_339_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMin_x3f___boxed(lean_object* v_00_u03b1_340_, lean_object* v_ord_341_, lean_object* v_xs_342_, lean_object* v_start_343_, lean_object* v_stop_344_){
_start:
{
lean_object* v_res_345_; 
v_res_345_ = lp_batteries_Array_rangeMin_x3f(v_00_u03b1_340_, v_ord_341_, v_xs_342_, v_start_343_, v_stop_344_);
lean_dec(v_stop_344_);
lean_dec(v_start_343_);
return v_res_345_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinI___redArg(lean_object* v_ord_346_, lean_object* v_inst_347_, lean_object* v_xs_348_, lean_object* v_start_349_, lean_object* v_stop_350_){
_start:
{
lean_object* v___x_351_; uint8_t v___x_352_; 
v___x_351_ = lean_array_get_size(v_xs_348_);
v___x_352_ = lean_nat_dec_lt(v_start_349_, v___x_351_);
if (v___x_352_ == 0)
{
lean_dec_ref(v_xs_348_);
lean_dec_ref(v_ord_346_);
lean_inc(v_inst_347_);
return v_inst_347_;
}
else
{
uint8_t v___x_353_; 
v___x_353_ = lean_nat_dec_lt(v_start_349_, v_stop_350_);
if (v___x_353_ == 0)
{
lean_dec_ref(v_xs_348_);
lean_dec_ref(v_ord_346_);
lean_inc(v_inst_347_);
return v_inst_347_;
}
else
{
lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v___x_357_; uint8_t v___x_358_; 
v___x_354_ = lean_array_fget(v_xs_348_, v_start_349_);
v___x_355_ = lean_unsigned_to_nat(1u);
v___x_356_ = lean_nat_add(v_start_349_, v___x_355_);
v___x_357_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_358_ = lean_nat_dec_lt(v___x_356_, v_stop_350_);
if (v___x_358_ == 0)
{
lean_dec(v___x_356_);
lean_dec_ref(v_xs_348_);
lean_dec_ref(v_ord_346_);
return v___x_354_;
}
else
{
lean_object* v___f_359_; uint8_t v___x_360_; 
v___f_359_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMinWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_359_, 0, v_ord_346_);
v___x_360_ = lean_nat_dec_le(v_stop_350_, v___x_351_);
if (v___x_360_ == 0)
{
uint8_t v___x_361_; 
v___x_361_ = lean_nat_dec_lt(v___x_356_, v___x_351_);
if (v___x_361_ == 0)
{
lean_dec_ref(v___f_359_);
lean_dec(v___x_356_);
lean_dec_ref(v_xs_348_);
return v___x_354_;
}
else
{
size_t v___x_362_; size_t v___x_363_; lean_object* v___x_364_; 
v___x_362_ = lean_usize_of_nat(v___x_356_);
lean_dec(v___x_356_);
v___x_363_ = lean_usize_of_nat(v___x_351_);
v___x_364_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_357_, v___f_359_, v_xs_348_, v___x_362_, v___x_363_, v___x_354_);
return v___x_364_;
}
}
else
{
size_t v___x_365_; size_t v___x_366_; lean_object* v___x_367_; 
v___x_365_ = lean_usize_of_nat(v___x_356_);
lean_dec(v___x_356_);
v___x_366_ = lean_usize_of_nat(v_stop_350_);
v___x_367_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_357_, v___f_359_, v_xs_348_, v___x_365_, v___x_366_, v___x_354_);
return v___x_367_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinI___redArg___boxed(lean_object* v_ord_368_, lean_object* v_inst_369_, lean_object* v_xs_370_, lean_object* v_start_371_, lean_object* v_stop_372_){
_start:
{
lean_object* v_res_373_; 
v_res_373_ = lp_batteries_Array_rangeMinI___redArg(v_ord_368_, v_inst_369_, v_xs_370_, v_start_371_, v_stop_372_);
lean_dec(v_stop_372_);
lean_dec(v_start_371_);
lean_dec(v_inst_369_);
return v_res_373_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinI(lean_object* v_00_u03b1_374_, lean_object* v_ord_375_, lean_object* v_inst_376_, lean_object* v_xs_377_, lean_object* v_start_378_, lean_object* v_stop_379_){
_start:
{
lean_object* v___x_380_; uint8_t v___x_381_; 
v___x_380_ = lean_array_get_size(v_xs_377_);
v___x_381_ = lean_nat_dec_lt(v_start_378_, v___x_380_);
if (v___x_381_ == 0)
{
lean_dec_ref(v_xs_377_);
lean_dec_ref(v_ord_375_);
lean_inc(v_inst_376_);
return v_inst_376_;
}
else
{
uint8_t v___x_382_; 
v___x_382_ = lean_nat_dec_lt(v_start_378_, v_stop_379_);
if (v___x_382_ == 0)
{
lean_dec_ref(v_xs_377_);
lean_dec_ref(v_ord_375_);
lean_inc(v_inst_376_);
return v_inst_376_;
}
else
{
lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; lean_object* v___x_386_; uint8_t v___x_387_; 
v___x_383_ = lean_array_fget(v_xs_377_, v_start_378_);
v___x_384_ = lean_unsigned_to_nat(1u);
v___x_385_ = lean_nat_add(v_start_378_, v___x_384_);
v___x_386_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_387_ = lean_nat_dec_lt(v___x_385_, v_stop_379_);
if (v___x_387_ == 0)
{
lean_dec(v___x_385_);
lean_dec_ref(v_xs_377_);
lean_dec_ref(v_ord_375_);
return v___x_383_;
}
else
{
lean_object* v___f_388_; uint8_t v___x_389_; 
v___f_388_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMinWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_388_, 0, v_ord_375_);
v___x_389_ = lean_nat_dec_le(v_stop_379_, v___x_380_);
if (v___x_389_ == 0)
{
uint8_t v___x_390_; 
v___x_390_ = lean_nat_dec_lt(v___x_385_, v___x_380_);
if (v___x_390_ == 0)
{
lean_dec_ref(v___f_388_);
lean_dec(v___x_385_);
lean_dec_ref(v_xs_377_);
return v___x_383_;
}
else
{
size_t v___x_391_; size_t v___x_392_; lean_object* v___x_393_; 
v___x_391_ = lean_usize_of_nat(v___x_385_);
lean_dec(v___x_385_);
v___x_392_ = lean_usize_of_nat(v___x_380_);
v___x_393_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_386_, v___f_388_, v_xs_377_, v___x_391_, v___x_392_, v___x_383_);
return v___x_393_;
}
}
else
{
size_t v___x_394_; size_t v___x_395_; lean_object* v___x_396_; 
v___x_394_ = lean_usize_of_nat(v___x_385_);
lean_dec(v___x_385_);
v___x_395_ = lean_usize_of_nat(v_stop_379_);
v___x_396_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_386_, v___f_388_, v_xs_377_, v___x_394_, v___x_395_, v___x_383_);
return v___x_396_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMinI___boxed(lean_object* v_00_u03b1_397_, lean_object* v_ord_398_, lean_object* v_inst_399_, lean_object* v_xs_400_, lean_object* v_start_401_, lean_object* v_stop_402_){
_start:
{
lean_object* v_res_403_; 
v_res_403_ = lp_batteries_Array_rangeMinI(v_00_u03b1_397_, v_ord_398_, v_inst_399_, v_xs_400_, v_start_401_, v_stop_402_);
lean_dec(v_stop_402_);
lean_dec(v_start_401_);
lean_dec(v_inst_399_);
return v_res_403_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_minI___redArg(lean_object* v_ord_404_, lean_object* v_inst_405_, lean_object* v_xs_406_, lean_object* v_start_407_, lean_object* v_stop_408_){
_start:
{
lean_object* v___x_409_; uint8_t v___x_410_; 
v___x_409_ = lean_array_get_size(v_xs_406_);
v___x_410_ = lean_nat_dec_lt(v_start_407_, v___x_409_);
if (v___x_410_ == 0)
{
lean_dec_ref(v_xs_406_);
lean_dec_ref(v_ord_404_);
lean_inc(v_inst_405_);
return v_inst_405_;
}
else
{
uint8_t v___x_411_; 
v___x_411_ = lean_nat_dec_lt(v_start_407_, v_stop_408_);
if (v___x_411_ == 0)
{
lean_dec_ref(v_xs_406_);
lean_dec_ref(v_ord_404_);
lean_inc(v_inst_405_);
return v_inst_405_;
}
else
{
lean_object* v___x_412_; lean_object* v___x_413_; lean_object* v___x_414_; lean_object* v___x_415_; uint8_t v___x_416_; 
v___x_412_ = lean_array_fget(v_xs_406_, v_start_407_);
v___x_413_ = lean_unsigned_to_nat(1u);
v___x_414_ = lean_nat_add(v_start_407_, v___x_413_);
v___x_415_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_416_ = lean_nat_dec_lt(v___x_414_, v_stop_408_);
if (v___x_416_ == 0)
{
lean_dec(v___x_414_);
lean_dec_ref(v_xs_406_);
lean_dec_ref(v_ord_404_);
return v___x_412_;
}
else
{
lean_object* v___f_417_; uint8_t v___x_418_; 
v___f_417_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMinWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_417_, 0, v_ord_404_);
v___x_418_ = lean_nat_dec_le(v_stop_408_, v___x_409_);
if (v___x_418_ == 0)
{
uint8_t v___x_419_; 
v___x_419_ = lean_nat_dec_lt(v___x_414_, v___x_409_);
if (v___x_419_ == 0)
{
lean_dec_ref(v___f_417_);
lean_dec(v___x_414_);
lean_dec_ref(v_xs_406_);
return v___x_412_;
}
else
{
size_t v___x_420_; size_t v___x_421_; lean_object* v___x_422_; 
v___x_420_ = lean_usize_of_nat(v___x_414_);
lean_dec(v___x_414_);
v___x_421_ = lean_usize_of_nat(v___x_409_);
v___x_422_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_415_, v___f_417_, v_xs_406_, v___x_420_, v___x_421_, v___x_412_);
return v___x_422_;
}
}
else
{
size_t v___x_423_; size_t v___x_424_; lean_object* v___x_425_; 
v___x_423_ = lean_usize_of_nat(v___x_414_);
lean_dec(v___x_414_);
v___x_424_ = lean_usize_of_nat(v_stop_408_);
v___x_425_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_415_, v___f_417_, v_xs_406_, v___x_423_, v___x_424_, v___x_412_);
return v___x_425_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_minI___redArg___boxed(lean_object* v_ord_426_, lean_object* v_inst_427_, lean_object* v_xs_428_, lean_object* v_start_429_, lean_object* v_stop_430_){
_start:
{
lean_object* v_res_431_; 
v_res_431_ = lp_batteries_Array_minI___redArg(v_ord_426_, v_inst_427_, v_xs_428_, v_start_429_, v_stop_430_);
lean_dec(v_stop_430_);
lean_dec(v_start_429_);
lean_dec(v_inst_427_);
return v_res_431_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_minI(lean_object* v_00_u03b1_432_, lean_object* v_ord_433_, lean_object* v_inst_434_, lean_object* v_xs_435_, lean_object* v_start_436_, lean_object* v_stop_437_){
_start:
{
lean_object* v___x_438_; 
v___x_438_ = lp_batteries_Array_minI___redArg(v_ord_433_, v_inst_434_, v_xs_435_, v_start_436_, v_stop_437_);
return v___x_438_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_minI___boxed(lean_object* v_00_u03b1_439_, lean_object* v_ord_440_, lean_object* v_inst_441_, lean_object* v_xs_442_, lean_object* v_start_443_, lean_object* v_stop_444_){
_start:
{
lean_object* v_res_445_; 
v_res_445_ = lp_batteries_Array_minI(v_00_u03b1_439_, v_ord_440_, v_inst_441_, v_xs_442_, v_start_443_, v_stop_444_);
lean_dec(v_stop_444_);
lean_dec(v_start_443_);
lean_dec(v_inst_441_);
return v_res_445_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxWith___redArg___lam__0(lean_object* v_ord_446_, lean_object* v_x1_447_, lean_object* v_x2_448_){
_start:
{
lean_object* v___x_449_; uint8_t v___x_450_; 
lean_inc(v_x2_448_);
lean_inc(v_x1_447_);
v___x_449_ = lean_apply_2(v_ord_446_, v_x1_447_, v_x2_448_);
v___x_450_ = lean_unbox(v___x_449_);
if (v___x_450_ == 0)
{
lean_dec(v_x1_447_);
return v_x2_448_;
}
else
{
lean_dec(v_x2_448_);
return v_x1_447_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxWith___redArg(lean_object* v_ord_451_, lean_object* v_xs_452_, lean_object* v_d_453_, lean_object* v_start_454_, lean_object* v_stop_455_){
_start:
{
lean_object* v___x_456_; uint8_t v___x_457_; 
v___x_456_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_457_ = lean_nat_dec_lt(v_start_454_, v_stop_455_);
if (v___x_457_ == 0)
{
lean_dec_ref(v_xs_452_);
lean_dec_ref(v_ord_451_);
return v_d_453_;
}
else
{
lean_object* v___f_458_; lean_object* v___x_459_; uint8_t v___x_460_; 
v___f_458_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMaxWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_458_, 0, v_ord_451_);
v___x_459_ = lean_array_get_size(v_xs_452_);
v___x_460_ = lean_nat_dec_le(v_stop_455_, v___x_459_);
if (v___x_460_ == 0)
{
uint8_t v___x_461_; 
v___x_461_ = lean_nat_dec_lt(v_start_454_, v___x_459_);
if (v___x_461_ == 0)
{
lean_dec_ref(v___f_458_);
lean_dec_ref(v_xs_452_);
return v_d_453_;
}
else
{
size_t v___x_462_; size_t v___x_463_; lean_object* v___x_464_; 
v___x_462_ = lean_usize_of_nat(v_start_454_);
v___x_463_ = lean_usize_of_nat(v___x_459_);
v___x_464_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_456_, v___f_458_, v_xs_452_, v___x_462_, v___x_463_, v_d_453_);
return v___x_464_;
}
}
else
{
size_t v___x_465_; size_t v___x_466_; lean_object* v___x_467_; 
v___x_465_ = lean_usize_of_nat(v_start_454_);
v___x_466_ = lean_usize_of_nat(v_stop_455_);
v___x_467_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_456_, v___f_458_, v_xs_452_, v___x_465_, v___x_466_, v_d_453_);
return v___x_467_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxWith___redArg___boxed(lean_object* v_ord_468_, lean_object* v_xs_469_, lean_object* v_d_470_, lean_object* v_start_471_, lean_object* v_stop_472_){
_start:
{
lean_object* v_res_473_; 
v_res_473_ = lp_batteries_Array_rangeMaxWith___redArg(v_ord_468_, v_xs_469_, v_d_470_, v_start_471_, v_stop_472_);
lean_dec(v_stop_472_);
lean_dec(v_start_471_);
return v_res_473_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxWith(lean_object* v_00_u03b1_474_, lean_object* v_ord_475_, lean_object* v_xs_476_, lean_object* v_d_477_, lean_object* v_start_478_, lean_object* v_stop_479_){
_start:
{
lean_object* v___x_480_; uint8_t v___x_481_; 
v___x_480_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_481_ = lean_nat_dec_lt(v_start_478_, v_stop_479_);
if (v___x_481_ == 0)
{
lean_dec_ref(v_xs_476_);
lean_dec_ref(v_ord_475_);
return v_d_477_;
}
else
{
lean_object* v___f_482_; lean_object* v___x_483_; uint8_t v___x_484_; 
v___f_482_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMaxWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_482_, 0, v_ord_475_);
v___x_483_ = lean_array_get_size(v_xs_476_);
v___x_484_ = lean_nat_dec_le(v_stop_479_, v___x_483_);
if (v___x_484_ == 0)
{
uint8_t v___x_485_; 
v___x_485_ = lean_nat_dec_lt(v_start_478_, v___x_483_);
if (v___x_485_ == 0)
{
lean_dec_ref(v___f_482_);
lean_dec_ref(v_xs_476_);
return v_d_477_;
}
else
{
size_t v___x_486_; size_t v___x_487_; lean_object* v___x_488_; 
v___x_486_ = lean_usize_of_nat(v_start_478_);
v___x_487_ = lean_usize_of_nat(v___x_483_);
v___x_488_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_480_, v___f_482_, v_xs_476_, v___x_486_, v___x_487_, v_d_477_);
return v___x_488_;
}
}
else
{
size_t v___x_489_; size_t v___x_490_; lean_object* v___x_491_; 
v___x_489_ = lean_usize_of_nat(v_start_478_);
v___x_490_ = lean_usize_of_nat(v_stop_479_);
v___x_491_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_480_, v___f_482_, v_xs_476_, v___x_489_, v___x_490_, v_d_477_);
return v___x_491_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxWith___boxed(lean_object* v_00_u03b1_492_, lean_object* v_ord_493_, lean_object* v_xs_494_, lean_object* v_d_495_, lean_object* v_start_496_, lean_object* v_stop_497_){
_start:
{
lean_object* v_res_498_; 
v_res_498_ = lp_batteries_Array_rangeMaxWith(v_00_u03b1_492_, v_ord_493_, v_xs_494_, v_d_495_, v_start_496_, v_stop_497_);
lean_dec(v_stop_497_);
lean_dec(v_start_496_);
return v_res_498_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_maxWith___redArg(lean_object* v_ord_499_, lean_object* v_xs_500_, lean_object* v_d_501_, lean_object* v_start_502_, lean_object* v_stop_503_){
_start:
{
lean_object* v___x_504_; uint8_t v___x_505_; 
v___x_504_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_505_ = lean_nat_dec_lt(v_start_502_, v_stop_503_);
if (v___x_505_ == 0)
{
lean_dec_ref(v_xs_500_);
lean_dec_ref(v_ord_499_);
return v_d_501_;
}
else
{
lean_object* v___f_506_; lean_object* v___x_507_; uint8_t v___x_508_; 
v___f_506_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMaxWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_506_, 0, v_ord_499_);
v___x_507_ = lean_array_get_size(v_xs_500_);
v___x_508_ = lean_nat_dec_le(v_stop_503_, v___x_507_);
if (v___x_508_ == 0)
{
uint8_t v___x_509_; 
v___x_509_ = lean_nat_dec_lt(v_start_502_, v___x_507_);
if (v___x_509_ == 0)
{
lean_dec_ref(v___f_506_);
lean_dec_ref(v_xs_500_);
return v_d_501_;
}
else
{
size_t v___x_510_; size_t v___x_511_; lean_object* v___x_512_; 
v___x_510_ = lean_usize_of_nat(v_start_502_);
v___x_511_ = lean_usize_of_nat(v___x_507_);
v___x_512_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_504_, v___f_506_, v_xs_500_, v___x_510_, v___x_511_, v_d_501_);
return v___x_512_;
}
}
else
{
size_t v___x_513_; size_t v___x_514_; lean_object* v___x_515_; 
v___x_513_ = lean_usize_of_nat(v_start_502_);
v___x_514_ = lean_usize_of_nat(v_stop_503_);
v___x_515_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_504_, v___f_506_, v_xs_500_, v___x_513_, v___x_514_, v_d_501_);
return v___x_515_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_maxWith___redArg___boxed(lean_object* v_ord_516_, lean_object* v_xs_517_, lean_object* v_d_518_, lean_object* v_start_519_, lean_object* v_stop_520_){
_start:
{
lean_object* v_res_521_; 
v_res_521_ = lp_batteries_Array_maxWith___redArg(v_ord_516_, v_xs_517_, v_d_518_, v_start_519_, v_stop_520_);
lean_dec(v_stop_520_);
lean_dec(v_start_519_);
return v_res_521_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_maxWith(lean_object* v_00_u03b1_522_, lean_object* v_ord_523_, lean_object* v_xs_524_, lean_object* v_d_525_, lean_object* v_start_526_, lean_object* v_stop_527_){
_start:
{
lean_object* v___x_528_; 
v___x_528_ = lp_batteries_Array_maxWith___redArg(v_ord_523_, v_xs_524_, v_d_525_, v_start_526_, v_stop_527_);
return v___x_528_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_maxWith___boxed(lean_object* v_00_u03b1_529_, lean_object* v_ord_530_, lean_object* v_xs_531_, lean_object* v_d_532_, lean_object* v_start_533_, lean_object* v_stop_534_){
_start:
{
lean_object* v_res_535_; 
v_res_535_ = lp_batteries_Array_maxWith(v_00_u03b1_529_, v_ord_530_, v_xs_531_, v_d_532_, v_start_533_, v_stop_534_);
lean_dec(v_stop_534_);
lean_dec(v_start_533_);
return v_res_535_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxD___redArg(lean_object* v_ord_536_, lean_object* v_xs_537_, lean_object* v_d_538_, lean_object* v_start_539_, lean_object* v_stop_540_){
_start:
{
lean_object* v___x_541_; uint8_t v___x_542_; 
v___x_541_ = lean_array_get_size(v_xs_537_);
v___x_542_ = lean_nat_dec_lt(v_start_539_, v___x_541_);
if (v___x_542_ == 0)
{
lean_dec_ref(v_xs_537_);
lean_dec_ref(v_ord_536_);
lean_inc(v_d_538_);
return v_d_538_;
}
else
{
uint8_t v___x_543_; 
v___x_543_ = lean_nat_dec_lt(v_start_539_, v_stop_540_);
if (v___x_543_ == 0)
{
lean_dec_ref(v_xs_537_);
lean_dec_ref(v_ord_536_);
lean_inc(v_d_538_);
return v_d_538_;
}
else
{
lean_object* v___x_544_; lean_object* v___x_545_; lean_object* v___x_546_; lean_object* v___x_547_; uint8_t v___x_548_; 
v___x_544_ = lean_array_fget(v_xs_537_, v_start_539_);
v___x_545_ = lean_unsigned_to_nat(1u);
v___x_546_ = lean_nat_add(v_start_539_, v___x_545_);
v___x_547_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_548_ = lean_nat_dec_lt(v___x_546_, v_stop_540_);
if (v___x_548_ == 0)
{
lean_dec(v___x_546_);
lean_dec_ref(v_xs_537_);
lean_dec_ref(v_ord_536_);
return v___x_544_;
}
else
{
lean_object* v___f_549_; uint8_t v___x_550_; 
v___f_549_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMaxWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_549_, 0, v_ord_536_);
v___x_550_ = lean_nat_dec_le(v_stop_540_, v___x_541_);
if (v___x_550_ == 0)
{
uint8_t v___x_551_; 
v___x_551_ = lean_nat_dec_lt(v___x_546_, v___x_541_);
if (v___x_551_ == 0)
{
lean_dec_ref(v___f_549_);
lean_dec(v___x_546_);
lean_dec_ref(v_xs_537_);
return v___x_544_;
}
else
{
size_t v___x_552_; size_t v___x_553_; lean_object* v___x_554_; 
v___x_552_ = lean_usize_of_nat(v___x_546_);
lean_dec(v___x_546_);
v___x_553_ = lean_usize_of_nat(v___x_541_);
v___x_554_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_547_, v___f_549_, v_xs_537_, v___x_552_, v___x_553_, v___x_544_);
return v___x_554_;
}
}
else
{
size_t v___x_555_; size_t v___x_556_; lean_object* v___x_557_; 
v___x_555_ = lean_usize_of_nat(v___x_546_);
lean_dec(v___x_546_);
v___x_556_ = lean_usize_of_nat(v_stop_540_);
v___x_557_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_547_, v___f_549_, v_xs_537_, v___x_555_, v___x_556_, v___x_544_);
return v___x_557_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxD___redArg___boxed(lean_object* v_ord_558_, lean_object* v_xs_559_, lean_object* v_d_560_, lean_object* v_start_561_, lean_object* v_stop_562_){
_start:
{
lean_object* v_res_563_; 
v_res_563_ = lp_batteries_Array_rangeMaxD___redArg(v_ord_558_, v_xs_559_, v_d_560_, v_start_561_, v_stop_562_);
lean_dec(v_stop_562_);
lean_dec(v_start_561_);
lean_dec(v_d_560_);
return v_res_563_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxD(lean_object* v_00_u03b1_564_, lean_object* v_ord_565_, lean_object* v_xs_566_, lean_object* v_d_567_, lean_object* v_start_568_, lean_object* v_stop_569_){
_start:
{
lean_object* v___x_570_; uint8_t v___x_571_; 
v___x_570_ = lean_array_get_size(v_xs_566_);
v___x_571_ = lean_nat_dec_lt(v_start_568_, v___x_570_);
if (v___x_571_ == 0)
{
lean_dec_ref(v_xs_566_);
lean_dec_ref(v_ord_565_);
lean_inc(v_d_567_);
return v_d_567_;
}
else
{
uint8_t v___x_572_; 
v___x_572_ = lean_nat_dec_lt(v_start_568_, v_stop_569_);
if (v___x_572_ == 0)
{
lean_dec_ref(v_xs_566_);
lean_dec_ref(v_ord_565_);
lean_inc(v_d_567_);
return v_d_567_;
}
else
{
lean_object* v___x_573_; lean_object* v___x_574_; lean_object* v___x_575_; lean_object* v___x_576_; uint8_t v___x_577_; 
v___x_573_ = lean_array_fget(v_xs_566_, v_start_568_);
v___x_574_ = lean_unsigned_to_nat(1u);
v___x_575_ = lean_nat_add(v_start_568_, v___x_574_);
v___x_576_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_577_ = lean_nat_dec_lt(v___x_575_, v_stop_569_);
if (v___x_577_ == 0)
{
lean_dec(v___x_575_);
lean_dec_ref(v_xs_566_);
lean_dec_ref(v_ord_565_);
return v___x_573_;
}
else
{
lean_object* v___f_578_; uint8_t v___x_579_; 
v___f_578_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMaxWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_578_, 0, v_ord_565_);
v___x_579_ = lean_nat_dec_le(v_stop_569_, v___x_570_);
if (v___x_579_ == 0)
{
uint8_t v___x_580_; 
v___x_580_ = lean_nat_dec_lt(v___x_575_, v___x_570_);
if (v___x_580_ == 0)
{
lean_dec_ref(v___f_578_);
lean_dec(v___x_575_);
lean_dec_ref(v_xs_566_);
return v___x_573_;
}
else
{
size_t v___x_581_; size_t v___x_582_; lean_object* v___x_583_; 
v___x_581_ = lean_usize_of_nat(v___x_575_);
lean_dec(v___x_575_);
v___x_582_ = lean_usize_of_nat(v___x_570_);
v___x_583_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_576_, v___f_578_, v_xs_566_, v___x_581_, v___x_582_, v___x_573_);
return v___x_583_;
}
}
else
{
size_t v___x_584_; size_t v___x_585_; lean_object* v___x_586_; 
v___x_584_ = lean_usize_of_nat(v___x_575_);
lean_dec(v___x_575_);
v___x_585_ = lean_usize_of_nat(v_stop_569_);
v___x_586_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_576_, v___f_578_, v_xs_566_, v___x_584_, v___x_585_, v___x_573_);
return v___x_586_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxD___boxed(lean_object* v_00_u03b1_587_, lean_object* v_ord_588_, lean_object* v_xs_589_, lean_object* v_d_590_, lean_object* v_start_591_, lean_object* v_stop_592_){
_start:
{
lean_object* v_res_593_; 
v_res_593_ = lp_batteries_Array_rangeMaxD(v_00_u03b1_587_, v_ord_588_, v_xs_589_, v_d_590_, v_start_591_, v_stop_592_);
lean_dec(v_stop_592_);
lean_dec(v_start_591_);
lean_dec(v_d_590_);
return v_res_593_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_maxD___redArg(lean_object* v_ord_594_, lean_object* v_xs_595_, lean_object* v_d_596_, lean_object* v_start_597_, lean_object* v_stop_598_){
_start:
{
lean_object* v___x_599_; uint8_t v___x_600_; 
v___x_599_ = lean_array_get_size(v_xs_595_);
v___x_600_ = lean_nat_dec_lt(v_start_597_, v___x_599_);
if (v___x_600_ == 0)
{
lean_dec_ref(v_xs_595_);
lean_dec_ref(v_ord_594_);
lean_inc(v_d_596_);
return v_d_596_;
}
else
{
uint8_t v___x_601_; 
v___x_601_ = lean_nat_dec_lt(v_start_597_, v_stop_598_);
if (v___x_601_ == 0)
{
lean_dec_ref(v_xs_595_);
lean_dec_ref(v_ord_594_);
lean_inc(v_d_596_);
return v_d_596_;
}
else
{
lean_object* v___x_602_; lean_object* v___x_603_; lean_object* v___x_604_; lean_object* v___x_605_; uint8_t v___x_606_; 
v___x_602_ = lean_array_fget(v_xs_595_, v_start_597_);
v___x_603_ = lean_unsigned_to_nat(1u);
v___x_604_ = lean_nat_add(v_start_597_, v___x_603_);
v___x_605_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_606_ = lean_nat_dec_lt(v___x_604_, v_stop_598_);
if (v___x_606_ == 0)
{
lean_dec(v___x_604_);
lean_dec_ref(v_xs_595_);
lean_dec_ref(v_ord_594_);
return v___x_602_;
}
else
{
lean_object* v___f_607_; uint8_t v___x_608_; 
v___f_607_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMaxWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_607_, 0, v_ord_594_);
v___x_608_ = lean_nat_dec_le(v_stop_598_, v___x_599_);
if (v___x_608_ == 0)
{
uint8_t v___x_609_; 
v___x_609_ = lean_nat_dec_lt(v___x_604_, v___x_599_);
if (v___x_609_ == 0)
{
lean_dec_ref(v___f_607_);
lean_dec(v___x_604_);
lean_dec_ref(v_xs_595_);
return v___x_602_;
}
else
{
size_t v___x_610_; size_t v___x_611_; lean_object* v___x_612_; 
v___x_610_ = lean_usize_of_nat(v___x_604_);
lean_dec(v___x_604_);
v___x_611_ = lean_usize_of_nat(v___x_599_);
v___x_612_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_605_, v___f_607_, v_xs_595_, v___x_610_, v___x_611_, v___x_602_);
return v___x_612_;
}
}
else
{
size_t v___x_613_; size_t v___x_614_; lean_object* v___x_615_; 
v___x_613_ = lean_usize_of_nat(v___x_604_);
lean_dec(v___x_604_);
v___x_614_ = lean_usize_of_nat(v_stop_598_);
v___x_615_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_605_, v___f_607_, v_xs_595_, v___x_613_, v___x_614_, v___x_602_);
return v___x_615_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_maxD___redArg___boxed(lean_object* v_ord_616_, lean_object* v_xs_617_, lean_object* v_d_618_, lean_object* v_start_619_, lean_object* v_stop_620_){
_start:
{
lean_object* v_res_621_; 
v_res_621_ = lp_batteries_Array_maxD___redArg(v_ord_616_, v_xs_617_, v_d_618_, v_start_619_, v_stop_620_);
lean_dec(v_stop_620_);
lean_dec(v_start_619_);
lean_dec(v_d_618_);
return v_res_621_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_maxD(lean_object* v_00_u03b1_622_, lean_object* v_ord_623_, lean_object* v_xs_624_, lean_object* v_d_625_, lean_object* v_start_626_, lean_object* v_stop_627_){
_start:
{
lean_object* v___x_628_; 
v___x_628_ = lp_batteries_Array_maxD___redArg(v_ord_623_, v_xs_624_, v_d_625_, v_start_626_, v_stop_627_);
return v___x_628_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_maxD___boxed(lean_object* v_00_u03b1_629_, lean_object* v_ord_630_, lean_object* v_xs_631_, lean_object* v_d_632_, lean_object* v_start_633_, lean_object* v_stop_634_){
_start:
{
lean_object* v_res_635_; 
v_res_635_ = lp_batteries_Array_maxD(v_00_u03b1_629_, v_ord_630_, v_xs_631_, v_d_632_, v_start_633_, v_stop_634_);
lean_dec(v_stop_634_);
lean_dec(v_start_633_);
lean_dec(v_d_632_);
return v_res_635_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMax_x3f___redArg(lean_object* v_ord_636_, lean_object* v_xs_637_, lean_object* v_start_638_, lean_object* v_stop_639_){
_start:
{
lean_object* v___x_640_; uint8_t v___x_641_; 
v___x_640_ = lean_array_get_size(v_xs_637_);
v___x_641_ = lean_nat_dec_lt(v_start_638_, v___x_640_);
if (v___x_641_ == 0)
{
lean_object* v___x_642_; 
lean_dec_ref(v_xs_637_);
lean_dec_ref(v_ord_636_);
v___x_642_ = lean_box(0);
return v___x_642_;
}
else
{
uint8_t v___x_643_; 
v___x_643_ = lean_nat_dec_lt(v_start_638_, v_stop_639_);
if (v___x_643_ == 0)
{
lean_object* v___x_644_; 
lean_dec_ref(v_xs_637_);
lean_dec_ref(v_ord_636_);
v___x_644_ = lean_box(0);
return v___x_644_;
}
else
{
lean_object* v___x_645_; 
v___x_645_ = lean_array_fget(v_xs_637_, v_start_638_);
if (v___x_641_ == 0)
{
lean_object* v___x_646_; 
lean_dec_ref(v_xs_637_);
lean_dec_ref(v_ord_636_);
v___x_646_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_646_, 0, v___x_645_);
return v___x_646_;
}
else
{
if (v___x_643_ == 0)
{
lean_object* v___x_647_; 
lean_dec_ref(v_xs_637_);
lean_dec_ref(v_ord_636_);
v___x_647_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_647_, 0, v___x_645_);
return v___x_647_;
}
else
{
lean_object* v___x_648_; lean_object* v___x_649_; lean_object* v___x_650_; uint8_t v___x_651_; 
v___x_648_ = lean_unsigned_to_nat(1u);
v___x_649_ = lean_nat_add(v_start_638_, v___x_648_);
v___x_650_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_651_ = lean_nat_dec_lt(v___x_649_, v_stop_639_);
if (v___x_651_ == 0)
{
lean_object* v___x_652_; 
lean_dec(v___x_649_);
lean_dec_ref(v_xs_637_);
lean_dec_ref(v_ord_636_);
v___x_652_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_652_, 0, v___x_645_);
return v___x_652_;
}
else
{
lean_object* v___f_653_; uint8_t v___x_654_; 
v___f_653_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMaxWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_653_, 0, v_ord_636_);
v___x_654_ = lean_nat_dec_le(v_stop_639_, v___x_640_);
if (v___x_654_ == 0)
{
uint8_t v___x_655_; 
v___x_655_ = lean_nat_dec_lt(v___x_649_, v___x_640_);
if (v___x_655_ == 0)
{
lean_object* v___x_656_; 
lean_dec_ref(v___f_653_);
lean_dec(v___x_649_);
lean_dec_ref(v_xs_637_);
v___x_656_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_656_, 0, v___x_645_);
return v___x_656_;
}
else
{
size_t v___x_657_; size_t v___x_658_; lean_object* v___x_659_; lean_object* v___x_660_; 
v___x_657_ = lean_usize_of_nat(v___x_649_);
lean_dec(v___x_649_);
v___x_658_ = lean_usize_of_nat(v___x_640_);
v___x_659_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_650_, v___f_653_, v_xs_637_, v___x_657_, v___x_658_, v___x_645_);
v___x_660_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_660_, 0, v___x_659_);
return v___x_660_;
}
}
else
{
size_t v___x_661_; size_t v___x_662_; lean_object* v___x_663_; lean_object* v___x_664_; 
v___x_661_ = lean_usize_of_nat(v___x_649_);
lean_dec(v___x_649_);
v___x_662_ = lean_usize_of_nat(v_stop_639_);
v___x_663_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_650_, v___f_653_, v_xs_637_, v___x_661_, v___x_662_, v___x_645_);
v___x_664_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_664_, 0, v___x_663_);
return v___x_664_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMax_x3f___redArg___boxed(lean_object* v_ord_665_, lean_object* v_xs_666_, lean_object* v_start_667_, lean_object* v_stop_668_){
_start:
{
lean_object* v_res_669_; 
v_res_669_ = lp_batteries_Array_rangeMax_x3f___redArg(v_ord_665_, v_xs_666_, v_start_667_, v_stop_668_);
lean_dec(v_stop_668_);
lean_dec(v_start_667_);
return v_res_669_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMax_x3f(lean_object* v_00_u03b1_670_, lean_object* v_ord_671_, lean_object* v_xs_672_, lean_object* v_start_673_, lean_object* v_stop_674_){
_start:
{
lean_object* v___x_675_; uint8_t v___x_676_; 
v___x_675_ = lean_array_get_size(v_xs_672_);
v___x_676_ = lean_nat_dec_lt(v_start_673_, v___x_675_);
if (v___x_676_ == 0)
{
lean_object* v___x_677_; 
lean_dec_ref(v_xs_672_);
lean_dec_ref(v_ord_671_);
v___x_677_ = lean_box(0);
return v___x_677_;
}
else
{
uint8_t v___x_678_; 
v___x_678_ = lean_nat_dec_lt(v_start_673_, v_stop_674_);
if (v___x_678_ == 0)
{
lean_object* v___x_679_; 
lean_dec_ref(v_xs_672_);
lean_dec_ref(v_ord_671_);
v___x_679_ = lean_box(0);
return v___x_679_;
}
else
{
lean_object* v___x_680_; 
v___x_680_ = lean_array_fget(v_xs_672_, v_start_673_);
if (v___x_676_ == 0)
{
lean_object* v___x_681_; 
lean_dec_ref(v_xs_672_);
lean_dec_ref(v_ord_671_);
v___x_681_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_681_, 0, v___x_680_);
return v___x_681_;
}
else
{
if (v___x_678_ == 0)
{
lean_object* v___x_682_; 
lean_dec_ref(v_xs_672_);
lean_dec_ref(v_ord_671_);
v___x_682_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_682_, 0, v___x_680_);
return v___x_682_;
}
else
{
lean_object* v___x_683_; lean_object* v___x_684_; lean_object* v___x_685_; uint8_t v___x_686_; 
v___x_683_ = lean_unsigned_to_nat(1u);
v___x_684_ = lean_nat_add(v_start_673_, v___x_683_);
v___x_685_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_686_ = lean_nat_dec_lt(v___x_684_, v_stop_674_);
if (v___x_686_ == 0)
{
lean_object* v___x_687_; 
lean_dec(v___x_684_);
lean_dec_ref(v_xs_672_);
lean_dec_ref(v_ord_671_);
v___x_687_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_687_, 0, v___x_680_);
return v___x_687_;
}
else
{
lean_object* v___f_688_; uint8_t v___x_689_; 
v___f_688_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMaxWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_688_, 0, v_ord_671_);
v___x_689_ = lean_nat_dec_le(v_stop_674_, v___x_675_);
if (v___x_689_ == 0)
{
uint8_t v___x_690_; 
v___x_690_ = lean_nat_dec_lt(v___x_684_, v___x_675_);
if (v___x_690_ == 0)
{
lean_object* v___x_691_; 
lean_dec_ref(v___f_688_);
lean_dec(v___x_684_);
lean_dec_ref(v_xs_672_);
v___x_691_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_691_, 0, v___x_680_);
return v___x_691_;
}
else
{
size_t v___x_692_; size_t v___x_693_; lean_object* v___x_694_; lean_object* v___x_695_; 
v___x_692_ = lean_usize_of_nat(v___x_684_);
lean_dec(v___x_684_);
v___x_693_ = lean_usize_of_nat(v___x_675_);
v___x_694_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_685_, v___f_688_, v_xs_672_, v___x_692_, v___x_693_, v___x_680_);
v___x_695_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_695_, 0, v___x_694_);
return v___x_695_;
}
}
else
{
size_t v___x_696_; size_t v___x_697_; lean_object* v___x_698_; lean_object* v___x_699_; 
v___x_696_ = lean_usize_of_nat(v___x_684_);
lean_dec(v___x_684_);
v___x_697_ = lean_usize_of_nat(v_stop_674_);
v___x_698_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_685_, v___f_688_, v_xs_672_, v___x_696_, v___x_697_, v___x_680_);
v___x_699_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_699_, 0, v___x_698_);
return v___x_699_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMax_x3f___boxed(lean_object* v_00_u03b1_700_, lean_object* v_ord_701_, lean_object* v_xs_702_, lean_object* v_start_703_, lean_object* v_stop_704_){
_start:
{
lean_object* v_res_705_; 
v_res_705_ = lp_batteries_Array_rangeMax_x3f(v_00_u03b1_700_, v_ord_701_, v_xs_702_, v_start_703_, v_stop_704_);
lean_dec(v_stop_704_);
lean_dec(v_start_703_);
return v_res_705_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxI___redArg(lean_object* v_ord_706_, lean_object* v_inst_707_, lean_object* v_xs_708_, lean_object* v_start_709_, lean_object* v_stop_710_){
_start:
{
lean_object* v___x_711_; uint8_t v___x_712_; 
v___x_711_ = lean_array_get_size(v_xs_708_);
v___x_712_ = lean_nat_dec_lt(v_start_709_, v___x_711_);
if (v___x_712_ == 0)
{
lean_dec_ref(v_xs_708_);
lean_dec_ref(v_ord_706_);
lean_inc(v_inst_707_);
return v_inst_707_;
}
else
{
uint8_t v___x_713_; 
v___x_713_ = lean_nat_dec_lt(v_start_709_, v_stop_710_);
if (v___x_713_ == 0)
{
lean_dec_ref(v_xs_708_);
lean_dec_ref(v_ord_706_);
lean_inc(v_inst_707_);
return v_inst_707_;
}
else
{
lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; lean_object* v___x_717_; uint8_t v___x_718_; 
v___x_714_ = lean_array_fget(v_xs_708_, v_start_709_);
v___x_715_ = lean_unsigned_to_nat(1u);
v___x_716_ = lean_nat_add(v_start_709_, v___x_715_);
v___x_717_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_718_ = lean_nat_dec_lt(v___x_716_, v_stop_710_);
if (v___x_718_ == 0)
{
lean_dec(v___x_716_);
lean_dec_ref(v_xs_708_);
lean_dec_ref(v_ord_706_);
return v___x_714_;
}
else
{
lean_object* v___f_719_; uint8_t v___x_720_; 
v___f_719_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMaxWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_719_, 0, v_ord_706_);
v___x_720_ = lean_nat_dec_le(v_stop_710_, v___x_711_);
if (v___x_720_ == 0)
{
uint8_t v___x_721_; 
v___x_721_ = lean_nat_dec_lt(v___x_716_, v___x_711_);
if (v___x_721_ == 0)
{
lean_dec_ref(v___f_719_);
lean_dec(v___x_716_);
lean_dec_ref(v_xs_708_);
return v___x_714_;
}
else
{
size_t v___x_722_; size_t v___x_723_; lean_object* v___x_724_; 
v___x_722_ = lean_usize_of_nat(v___x_716_);
lean_dec(v___x_716_);
v___x_723_ = lean_usize_of_nat(v___x_711_);
v___x_724_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_717_, v___f_719_, v_xs_708_, v___x_722_, v___x_723_, v___x_714_);
return v___x_724_;
}
}
else
{
size_t v___x_725_; size_t v___x_726_; lean_object* v___x_727_; 
v___x_725_ = lean_usize_of_nat(v___x_716_);
lean_dec(v___x_716_);
v___x_726_ = lean_usize_of_nat(v_stop_710_);
v___x_727_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_717_, v___f_719_, v_xs_708_, v___x_725_, v___x_726_, v___x_714_);
return v___x_727_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxI___redArg___boxed(lean_object* v_ord_728_, lean_object* v_inst_729_, lean_object* v_xs_730_, lean_object* v_start_731_, lean_object* v_stop_732_){
_start:
{
lean_object* v_res_733_; 
v_res_733_ = lp_batteries_Array_rangeMaxI___redArg(v_ord_728_, v_inst_729_, v_xs_730_, v_start_731_, v_stop_732_);
lean_dec(v_stop_732_);
lean_dec(v_start_731_);
lean_dec(v_inst_729_);
return v_res_733_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxI(lean_object* v_00_u03b1_734_, lean_object* v_ord_735_, lean_object* v_inst_736_, lean_object* v_xs_737_, lean_object* v_start_738_, lean_object* v_stop_739_){
_start:
{
lean_object* v___x_740_; uint8_t v___x_741_; 
v___x_740_ = lean_array_get_size(v_xs_737_);
v___x_741_ = lean_nat_dec_lt(v_start_738_, v___x_740_);
if (v___x_741_ == 0)
{
lean_dec_ref(v_xs_737_);
lean_dec_ref(v_ord_735_);
lean_inc(v_inst_736_);
return v_inst_736_;
}
else
{
uint8_t v___x_742_; 
v___x_742_ = lean_nat_dec_lt(v_start_738_, v_stop_739_);
if (v___x_742_ == 0)
{
lean_dec_ref(v_xs_737_);
lean_dec_ref(v_ord_735_);
lean_inc(v_inst_736_);
return v_inst_736_;
}
else
{
lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; uint8_t v___x_747_; 
v___x_743_ = lean_array_fget(v_xs_737_, v_start_738_);
v___x_744_ = lean_unsigned_to_nat(1u);
v___x_745_ = lean_nat_add(v_start_738_, v___x_744_);
v___x_746_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_747_ = lean_nat_dec_lt(v___x_745_, v_stop_739_);
if (v___x_747_ == 0)
{
lean_dec(v___x_745_);
lean_dec_ref(v_xs_737_);
lean_dec_ref(v_ord_735_);
return v___x_743_;
}
else
{
lean_object* v___f_748_; uint8_t v___x_749_; 
v___f_748_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMaxWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_748_, 0, v_ord_735_);
v___x_749_ = lean_nat_dec_le(v_stop_739_, v___x_740_);
if (v___x_749_ == 0)
{
uint8_t v___x_750_; 
v___x_750_ = lean_nat_dec_lt(v___x_745_, v___x_740_);
if (v___x_750_ == 0)
{
lean_dec_ref(v___f_748_);
lean_dec(v___x_745_);
lean_dec_ref(v_xs_737_);
return v___x_743_;
}
else
{
size_t v___x_751_; size_t v___x_752_; lean_object* v___x_753_; 
v___x_751_ = lean_usize_of_nat(v___x_745_);
lean_dec(v___x_745_);
v___x_752_ = lean_usize_of_nat(v___x_740_);
v___x_753_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_746_, v___f_748_, v_xs_737_, v___x_751_, v___x_752_, v___x_743_);
return v___x_753_;
}
}
else
{
size_t v___x_754_; size_t v___x_755_; lean_object* v___x_756_; 
v___x_754_ = lean_usize_of_nat(v___x_745_);
lean_dec(v___x_745_);
v___x_755_ = lean_usize_of_nat(v_stop_739_);
v___x_756_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_746_, v___f_748_, v_xs_737_, v___x_754_, v___x_755_, v___x_743_);
return v___x_756_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_rangeMaxI___boxed(lean_object* v_00_u03b1_757_, lean_object* v_ord_758_, lean_object* v_inst_759_, lean_object* v_xs_760_, lean_object* v_start_761_, lean_object* v_stop_762_){
_start:
{
lean_object* v_res_763_; 
v_res_763_ = lp_batteries_Array_rangeMaxI(v_00_u03b1_757_, v_ord_758_, v_inst_759_, v_xs_760_, v_start_761_, v_stop_762_);
lean_dec(v_stop_762_);
lean_dec(v_start_761_);
lean_dec(v_inst_759_);
return v_res_763_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_maxI___redArg(lean_object* v_ord_764_, lean_object* v_inst_765_, lean_object* v_xs_766_, lean_object* v_start_767_, lean_object* v_stop_768_){
_start:
{
lean_object* v___x_769_; uint8_t v___x_770_; 
v___x_769_ = lean_array_get_size(v_xs_766_);
v___x_770_ = lean_nat_dec_lt(v_start_767_, v___x_769_);
if (v___x_770_ == 0)
{
lean_dec_ref(v_xs_766_);
lean_dec_ref(v_ord_764_);
lean_inc(v_inst_765_);
return v_inst_765_;
}
else
{
uint8_t v___x_771_; 
v___x_771_ = lean_nat_dec_lt(v_start_767_, v_stop_768_);
if (v___x_771_ == 0)
{
lean_dec_ref(v_xs_766_);
lean_dec_ref(v_ord_764_);
lean_inc(v_inst_765_);
return v_inst_765_;
}
else
{
lean_object* v___x_772_; lean_object* v___x_773_; lean_object* v___x_774_; lean_object* v___x_775_; uint8_t v___x_776_; 
v___x_772_ = lean_array_fget(v_xs_766_, v_start_767_);
v___x_773_ = lean_unsigned_to_nat(1u);
v___x_774_ = lean_nat_add(v_start_767_, v___x_773_);
v___x_775_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_776_ = lean_nat_dec_lt(v___x_774_, v_stop_768_);
if (v___x_776_ == 0)
{
lean_dec(v___x_774_);
lean_dec_ref(v_xs_766_);
lean_dec_ref(v_ord_764_);
return v___x_772_;
}
else
{
lean_object* v___f_777_; uint8_t v___x_778_; 
v___f_777_ = lean_alloc_closure((void*)(lp_batteries_Array_rangeMaxWith___redArg___lam__0), 3, 1);
lean_closure_set(v___f_777_, 0, v_ord_764_);
v___x_778_ = lean_nat_dec_le(v_stop_768_, v___x_769_);
if (v___x_778_ == 0)
{
uint8_t v___x_779_; 
v___x_779_ = lean_nat_dec_lt(v___x_774_, v___x_769_);
if (v___x_779_ == 0)
{
lean_dec_ref(v___f_777_);
lean_dec(v___x_774_);
lean_dec_ref(v_xs_766_);
return v___x_772_;
}
else
{
size_t v___x_780_; size_t v___x_781_; lean_object* v___x_782_; 
v___x_780_ = lean_usize_of_nat(v___x_774_);
lean_dec(v___x_774_);
v___x_781_ = lean_usize_of_nat(v___x_769_);
v___x_782_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_775_, v___f_777_, v_xs_766_, v___x_780_, v___x_781_, v___x_772_);
return v___x_782_;
}
}
else
{
size_t v___x_783_; size_t v___x_784_; lean_object* v___x_785_; 
v___x_783_ = lean_usize_of_nat(v___x_774_);
lean_dec(v___x_774_);
v___x_784_ = lean_usize_of_nat(v_stop_768_);
v___x_785_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v___x_775_, v___f_777_, v_xs_766_, v___x_783_, v___x_784_, v___x_772_);
return v___x_785_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_maxI___redArg___boxed(lean_object* v_ord_786_, lean_object* v_inst_787_, lean_object* v_xs_788_, lean_object* v_start_789_, lean_object* v_stop_790_){
_start:
{
lean_object* v_res_791_; 
v_res_791_ = lp_batteries_Array_maxI___redArg(v_ord_786_, v_inst_787_, v_xs_788_, v_start_789_, v_stop_790_);
lean_dec(v_stop_790_);
lean_dec(v_start_789_);
lean_dec(v_inst_787_);
return v_res_791_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_maxI(lean_object* v_00_u03b1_792_, lean_object* v_ord_793_, lean_object* v_inst_794_, lean_object* v_xs_795_, lean_object* v_start_796_, lean_object* v_stop_797_){
_start:
{
lean_object* v___x_798_; 
v___x_798_ = lp_batteries_Array_maxI___redArg(v_ord_793_, v_inst_794_, v_xs_795_, v_start_796_, v_stop_797_);
return v___x_798_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_maxI___boxed(lean_object* v_00_u03b1_799_, lean_object* v_ord_800_, lean_object* v_inst_801_, lean_object* v_xs_802_, lean_object* v_start_803_, lean_object* v_stop_804_){
_start:
{
lean_object* v_res_805_; 
v_res_805_ = lp_batteries_Array_maxI(v_00_u03b1_799_, v_ord_800_, v_inst_801_, v_xs_802_, v_start_803_, v_stop_804_);
lean_dec(v_stop_804_);
lean_dec(v_start_803_);
lean_dec(v_inst_801_);
return v_res_805_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_setN___redArg(lean_object* v_xs_806_, lean_object* v_i_807_, lean_object* v_v_808_){
_start:
{
lean_object* v___x_809_; 
v___x_809_ = lean_array_fset(v_xs_806_, v_i_807_, v_v_808_);
return v___x_809_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_setN___redArg___boxed(lean_object* v_xs_810_, lean_object* v_i_811_, lean_object* v_v_812_){
_start:
{
lean_object* v_res_813_; 
v_res_813_ = lp_batteries_Array_setN___redArg(v_xs_810_, v_i_811_, v_v_812_);
lean_dec(v_i_811_);
return v_res_813_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_setN(lean_object* v_00_u03b1_814_, lean_object* v_xs_815_, lean_object* v_i_816_, lean_object* v_v_817_, lean_object* v_h_818_){
_start:
{
lean_object* v___x_819_; 
v___x_819_ = lean_array_fset(v_xs_815_, v_i_816_, v_v_817_);
return v___x_819_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_setN___boxed(lean_object* v_00_u03b1_820_, lean_object* v_xs_821_, lean_object* v_i_822_, lean_object* v_v_823_, lean_object* v_h_824_){
_start:
{
lean_object* v_res_825_; 
v_res_825_ = lp_batteries_Array_setN(v_00_u03b1_820_, v_xs_821_, v_i_822_, v_v_823_, v_h_824_);
lean_dec(v_i_822_);
return v_res_825_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg___lam__0___boxed(lean_object* v_start_826_, lean_object* v_acc_827_, lean_object* v_init_828_, lean_object* v_inst_829_, lean_object* v_f_830_, lean_object* v_as_831_, lean_object* v_stop_832_, lean_object* v_next_833_){
_start:
{
size_t v_start_boxed_834_; size_t v_stop_boxed_835_; lean_object* v_res_836_; 
v_start_boxed_834_ = lean_unbox_usize(v_start_826_);
lean_dec(v_start_826_);
v_stop_boxed_835_ = lean_unbox_usize(v_stop_832_);
lean_dec(v_stop_832_);
v_res_836_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg___lam__0(v_start_boxed_834_, v_acc_827_, v_init_828_, v_inst_829_, v_f_830_, v_as_831_, v_stop_boxed_835_, v_next_833_);
return v_res_836_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg(lean_object* v_inst_837_, lean_object* v_f_838_, lean_object* v_init_839_, lean_object* v_as_840_, size_t v_start_841_, size_t v_stop_842_, lean_object* v_acc_843_){
_start:
{
uint8_t v___x_844_; 
v___x_844_ = lean_usize_dec_lt(v_start_841_, v_stop_842_);
if (v___x_844_ == 0)
{
lean_object* v_toApplicative_845_; lean_object* v_toPure_846_; lean_object* v___x_847_; lean_object* v___x_848_; 
lean_dec_ref(v_as_840_);
lean_dec(v_f_838_);
v_toApplicative_845_ = lean_ctor_get(v_inst_837_, 0);
lean_inc_ref(v_toApplicative_845_);
lean_dec_ref(v_inst_837_);
v_toPure_846_ = lean_ctor_get(v_toApplicative_845_, 1);
lean_inc(v_toPure_846_);
lean_dec_ref(v_toApplicative_845_);
v___x_847_ = lean_array_push(v_acc_843_, v_init_839_);
v___x_848_ = lean_apply_2(v_toPure_846_, lean_box(0), v___x_847_);
return v___x_848_;
}
else
{
lean_object* v_toBind_849_; lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___f_852_; lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; 
v_toBind_849_ = lean_ctor_get(v_inst_837_, 1);
lean_inc(v_toBind_849_);
v___x_850_ = lean_box_usize(v_start_841_);
v___x_851_ = lean_box_usize(v_stop_842_);
lean_inc_ref(v_as_840_);
lean_inc(v_f_838_);
lean_inc(v_init_839_);
v___f_852_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg___lam__0___boxed), 8, 7);
lean_closure_set(v___f_852_, 0, v___x_850_);
lean_closure_set(v___f_852_, 1, v_acc_843_);
lean_closure_set(v___f_852_, 2, v_init_839_);
lean_closure_set(v___f_852_, 3, v_inst_837_);
lean_closure_set(v___f_852_, 4, v_f_838_);
lean_closure_set(v___f_852_, 5, v_as_840_);
lean_closure_set(v___f_852_, 6, v___x_851_);
v___x_853_ = lean_array_uget(v_as_840_, v_start_841_);
lean_dec_ref(v_as_840_);
v___x_854_ = lean_apply_2(v_f_838_, v_init_839_, v___x_853_);
v___x_855_ = lean_apply_4(v_toBind_849_, lean_box(0), lean_box(0), v___x_854_, v___f_852_);
return v___x_855_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg___lam__0(size_t v_start_856_, lean_object* v_acc_857_, lean_object* v_init_858_, lean_object* v_inst_859_, lean_object* v_f_860_, lean_object* v_as_861_, size_t v_stop_862_, lean_object* v_next_863_){
_start:
{
size_t v___x_864_; size_t v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; 
v___x_864_ = ((size_t)1ULL);
v___x_865_ = lean_usize_add(v_start_856_, v___x_864_);
v___x_866_ = lean_array_push(v_acc_857_, v_init_858_);
v___x_867_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg(v_inst_859_, v_f_860_, v_next_863_, v_as_861_, v___x_865_, v_stop_862_, v___x_866_);
return v___x_867_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg___boxed(lean_object* v_inst_868_, lean_object* v_f_869_, lean_object* v_init_870_, lean_object* v_as_871_, lean_object* v_start_872_, lean_object* v_stop_873_, lean_object* v_acc_874_){
_start:
{
size_t v_start_boxed_875_; size_t v_stop_boxed_876_; lean_object* v_res_877_; 
v_start_boxed_875_ = lean_unbox_usize(v_start_872_);
lean_dec(v_start_872_);
v_stop_boxed_876_ = lean_unbox_usize(v_stop_873_);
lean_dec(v_stop_873_);
v_res_877_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg(v_inst_868_, v_f_869_, v_init_870_, v_as_871_, v_start_boxed_875_, v_stop_boxed_876_, v_acc_874_);
return v_res_877_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop(lean_object* v_m_878_, lean_object* v_00_u03b2_879_, lean_object* v_00_u03b1_880_, lean_object* v_inst_881_, lean_object* v_f_882_, lean_object* v_init_883_, lean_object* v_as_884_, size_t v_start_885_, size_t v_stop_886_, lean_object* v_h__stop_887_, lean_object* v_acc_888_){
_start:
{
lean_object* v___x_889_; 
v___x_889_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg(v_inst_881_, v_f_882_, v_init_883_, v_as_884_, v_start_885_, v_stop_886_, v_acc_888_);
return v___x_889_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___boxed(lean_object* v_m_890_, lean_object* v_00_u03b2_891_, lean_object* v_00_u03b1_892_, lean_object* v_inst_893_, lean_object* v_f_894_, lean_object* v_init_895_, lean_object* v_as_896_, lean_object* v_start_897_, lean_object* v_stop_898_, lean_object* v_h__stop_899_, lean_object* v_acc_900_){
_start:
{
size_t v_start_boxed_901_; size_t v_stop_boxed_902_; lean_object* v_res_903_; 
v_start_boxed_901_ = lean_unbox_usize(v_start_897_);
lean_dec(v_start_897_);
v_stop_boxed_902_ = lean_unbox_usize(v_stop_898_);
lean_dec(v_stop_898_);
v_res_903_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop(v_m_890_, v_00_u03b2_891_, v_00_u03b1_892_, v_inst_893_, v_f_894_, v_init_895_, v_as_896_, v_start_boxed_901_, v_stop_boxed_902_, v_h__stop_899_, v_acc_900_);
return v_res_903_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast___redArg(lean_object* v_inst_904_, lean_object* v_f_905_, lean_object* v_init_906_, lean_object* v_as_907_, lean_object* v_start_908_, lean_object* v_stop_909_){
_start:
{
lean_object* v___y_911_; lean_object* v___y_912_; lean_object* v___x_920_; lean_object* v___y_922_; uint8_t v___x_924_; 
v___x_920_ = lean_array_get_size(v_as_907_);
v___x_924_ = lean_nat_dec_le(v_stop_909_, v___x_920_);
if (v___x_924_ == 0)
{
lean_dec(v_stop_909_);
v___y_922_ = v___x_920_;
goto v___jp_921_;
}
else
{
v___y_922_ = v_stop_909_;
goto v___jp_921_;
}
v___jp_910_:
{
size_t v___x_913_; size_t v___x_914_; lean_object* v___x_915_; lean_object* v___x_916_; lean_object* v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; 
v___x_913_ = lean_usize_of_nat(v___y_912_);
v___x_914_ = lean_usize_of_nat(v___y_911_);
v___x_915_ = lean_nat_sub(v___y_911_, v___y_912_);
lean_dec(v___y_912_);
lean_dec(v___y_911_);
v___x_916_ = lean_unsigned_to_nat(1u);
v___x_917_ = lean_nat_add(v___x_915_, v___x_916_);
lean_dec(v___x_915_);
v___x_918_ = lean_mk_empty_array_with_capacity(v___x_917_);
lean_dec(v___x_917_);
v___x_919_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg(v_inst_904_, v_f_905_, v_init_906_, v_as_907_, v___x_913_, v___x_914_, v___x_918_);
return v___x_919_;
}
v___jp_921_:
{
uint8_t v___x_923_; 
v___x_923_ = lean_nat_dec_le(v_start_908_, v___x_920_);
if (v___x_923_ == 0)
{
lean_dec(v_start_908_);
v___y_911_ = v___y_922_;
v___y_912_ = v___x_920_;
goto v___jp_910_;
}
else
{
v___y_911_ = v___y_922_;
v___y_912_ = v_start_908_;
goto v___jp_910_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast(lean_object* v_m_925_, lean_object* v_00_u03b2_926_, lean_object* v_00_u03b1_927_, lean_object* v_inst_928_, lean_object* v_f_929_, lean_object* v_init_930_, lean_object* v_as_931_, lean_object* v_start_932_, lean_object* v_stop_933_){
_start:
{
lean_object* v___y_935_; lean_object* v___y_936_; lean_object* v___x_944_; lean_object* v___y_946_; uint8_t v___x_948_; 
v___x_944_ = lean_array_get_size(v_as_931_);
v___x_948_ = lean_nat_dec_le(v_stop_933_, v___x_944_);
if (v___x_948_ == 0)
{
lean_dec(v_stop_933_);
v___y_946_ = v___x_944_;
goto v___jp_945_;
}
else
{
v___y_946_ = v_stop_933_;
goto v___jp_945_;
}
v___jp_934_:
{
size_t v___x_937_; size_t v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; 
v___x_937_ = lean_usize_of_nat(v___y_936_);
v___x_938_ = lean_usize_of_nat(v___y_935_);
v___x_939_ = lean_nat_sub(v___y_935_, v___y_936_);
lean_dec(v___y_936_);
lean_dec(v___y_935_);
v___x_940_ = lean_unsigned_to_nat(1u);
v___x_941_ = lean_nat_add(v___x_939_, v___x_940_);
lean_dec(v___x_939_);
v___x_942_ = lean_mk_empty_array_with_capacity(v___x_941_);
lean_dec(v___x_941_);
v___x_943_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg(v_inst_928_, v_f_929_, v_init_930_, v_as_931_, v___x_937_, v___x_938_, v___x_942_);
return v___x_943_;
}
v___jp_945_:
{
uint8_t v___x_947_; 
v___x_947_ = lean_nat_dec_le(v_start_932_, v___x_944_);
if (v___x_947_ == 0)
{
lean_dec(v_start_932_);
v___y_935_ = v___y_946_;
v___y_936_ = v___x_944_;
goto v___jp_934_;
}
else
{
v___y_935_ = v___y_946_;
v___y_936_ = v_start_932_;
goto v___jp_934_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_scanlM_loop___redArg___lam__0___boxed(lean_object* v_start_949_, lean_object* v_acc_950_, lean_object* v_init_951_, lean_object* v_inst_952_, lean_object* v_f_953_, lean_object* v_as_954_, lean_object* v_stop_955_, lean_object* v_____do__lift_956_){
_start:
{
lean_object* v_res_957_; 
v_res_957_ = lp_batteries_Array_scanlM_loop___redArg___lam__0(v_start_949_, v_acc_950_, v_init_951_, v_inst_952_, v_f_953_, v_as_954_, v_stop_955_, v_____do__lift_956_);
lean_dec(v_start_949_);
return v_res_957_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_scanlM_loop___redArg(lean_object* v_inst_958_, lean_object* v_f_959_, lean_object* v_init_960_, lean_object* v_as_961_, lean_object* v_start_962_, lean_object* v_stop_963_, lean_object* v_acc_964_){
_start:
{
uint8_t v___x_965_; 
v___x_965_ = lean_nat_dec_lt(v_start_962_, v_stop_963_);
if (v___x_965_ == 0)
{
lean_object* v_toApplicative_966_; lean_object* v_toPure_967_; lean_object* v___x_968_; lean_object* v___x_969_; 
lean_dec(v_stop_963_);
lean_dec(v_start_962_);
lean_dec_ref(v_as_961_);
lean_dec(v_f_959_);
v_toApplicative_966_ = lean_ctor_get(v_inst_958_, 0);
lean_inc_ref(v_toApplicative_966_);
lean_dec_ref(v_inst_958_);
v_toPure_967_ = lean_ctor_get(v_toApplicative_966_, 1);
lean_inc(v_toPure_967_);
lean_dec_ref(v_toApplicative_966_);
v___x_968_ = lean_array_push(v_acc_964_, v_init_960_);
v___x_969_ = lean_apply_2(v_toPure_967_, lean_box(0), v___x_968_);
return v___x_969_;
}
else
{
lean_object* v_toBind_970_; lean_object* v___f_971_; lean_object* v___x_972_; lean_object* v___x_973_; lean_object* v___x_974_; 
v_toBind_970_ = lean_ctor_get(v_inst_958_, 1);
lean_inc(v_toBind_970_);
lean_inc_ref(v_as_961_);
lean_inc(v_f_959_);
lean_inc(v_init_960_);
lean_inc(v_start_962_);
v___f_971_ = lean_alloc_closure((void*)(lp_batteries_Array_scanlM_loop___redArg___lam__0___boxed), 8, 7);
lean_closure_set(v___f_971_, 0, v_start_962_);
lean_closure_set(v___f_971_, 1, v_acc_964_);
lean_closure_set(v___f_971_, 2, v_init_960_);
lean_closure_set(v___f_971_, 3, v_inst_958_);
lean_closure_set(v___f_971_, 4, v_f_959_);
lean_closure_set(v___f_971_, 5, v_as_961_);
lean_closure_set(v___f_971_, 6, v_stop_963_);
v___x_972_ = lean_array_fget(v_as_961_, v_start_962_);
lean_dec(v_start_962_);
lean_dec_ref(v_as_961_);
v___x_973_ = lean_apply_2(v_f_959_, v_init_960_, v___x_972_);
v___x_974_ = lean_apply_4(v_toBind_970_, lean_box(0), lean_box(0), v___x_973_, v___f_971_);
return v___x_974_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_scanlM_loop___redArg___lam__0(lean_object* v_start_975_, lean_object* v_acc_976_, lean_object* v_init_977_, lean_object* v_inst_978_, lean_object* v_f_979_, lean_object* v_as_980_, lean_object* v_stop_981_, lean_object* v_____do__lift_982_){
_start:
{
lean_object* v___x_983_; lean_object* v___x_984_; lean_object* v___x_985_; lean_object* v___x_986_; 
v___x_983_ = lean_unsigned_to_nat(1u);
v___x_984_ = lean_nat_add(v_start_975_, v___x_983_);
v___x_985_ = lean_array_push(v_acc_976_, v_init_977_);
v___x_986_ = lp_batteries_Array_scanlM_loop___redArg(v_inst_978_, v_f_979_, v_____do__lift_982_, v_as_980_, v___x_984_, v_stop_981_, v___x_985_);
return v___x_986_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_scanlM_loop(lean_object* v_m_987_, lean_object* v_00_u03b2_988_, lean_object* v_00_u03b1_989_, lean_object* v_inst_990_, lean_object* v_f_991_, lean_object* v_init_992_, lean_object* v_as_993_, lean_object* v_start_994_, lean_object* v_stop_995_, lean_object* v_h__stop_996_, lean_object* v_acc_997_){
_start:
{
lean_object* v___x_998_; 
v___x_998_ = lp_batteries_Array_scanlM_loop___redArg(v_inst_990_, v_f_991_, v_init_992_, v_as_993_, v_start_994_, v_stop_995_, v_acc_997_);
return v___x_998_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg___lam__0___boxed(lean_object* v_startM1_999_, lean_object* v_stop_1000_, lean_object* v_acc_1001_, lean_object* v_inst_1002_, lean_object* v_f_1003_, lean_object* v_as_1004_, lean_object* v_next_1005_){
_start:
{
size_t v_startM1_boxed_1006_; size_t v_stop_boxed_1007_; lean_object* v_res_1008_; 
v_startM1_boxed_1006_ = lean_unbox_usize(v_startM1_999_);
lean_dec(v_startM1_999_);
v_stop_boxed_1007_ = lean_unbox_usize(v_stop_1000_);
lean_dec(v_stop_1000_);
v_res_1008_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg___lam__0(v_startM1_boxed_1006_, v_stop_boxed_1007_, v_acc_1001_, v_inst_1002_, v_f_1003_, v_as_1004_, v_next_1005_);
return v_res_1008_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg(lean_object* v_inst_1009_, lean_object* v_f_1010_, lean_object* v_init_1011_, lean_object* v_as_1012_, size_t v_start_1013_, size_t v_stop_1014_, lean_object* v_acc_1015_){
_start:
{
uint8_t v___x_1016_; 
v___x_1016_ = lean_usize_dec_lt(v_stop_1014_, v_start_1013_);
if (v___x_1016_ == 0)
{
lean_object* v_toApplicative_1017_; lean_object* v_toPure_1018_; lean_object* v___x_1019_; 
lean_dec_ref(v_as_1012_);
lean_dec(v_init_1011_);
lean_dec(v_f_1010_);
v_toApplicative_1017_ = lean_ctor_get(v_inst_1009_, 0);
lean_inc_ref(v_toApplicative_1017_);
lean_dec_ref(v_inst_1009_);
v_toPure_1018_ = lean_ctor_get(v_toApplicative_1017_, 1);
lean_inc(v_toPure_1018_);
lean_dec_ref(v_toApplicative_1017_);
v___x_1019_ = lean_apply_2(v_toPure_1018_, lean_box(0), v_acc_1015_);
return v___x_1019_;
}
else
{
lean_object* v_toBind_1020_; size_t v___x_1021_; size_t v_startM1_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___f_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; 
v_toBind_1020_ = lean_ctor_get(v_inst_1009_, 1);
lean_inc(v_toBind_1020_);
v___x_1021_ = ((size_t)1ULL);
v_startM1_1022_ = lean_usize_sub(v_start_1013_, v___x_1021_);
v___x_1023_ = lean_box_usize(v_startM1_1022_);
v___x_1024_ = lean_box_usize(v_stop_1014_);
lean_inc_ref(v_as_1012_);
lean_inc(v_f_1010_);
v___f_1025_ = lean_alloc_closure((void*)(lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg___lam__0___boxed), 7, 6);
lean_closure_set(v___f_1025_, 0, v___x_1023_);
lean_closure_set(v___f_1025_, 1, v___x_1024_);
lean_closure_set(v___f_1025_, 2, v_acc_1015_);
lean_closure_set(v___f_1025_, 3, v_inst_1009_);
lean_closure_set(v___f_1025_, 4, v_f_1010_);
lean_closure_set(v___f_1025_, 5, v_as_1012_);
v___x_1026_ = lean_array_uget(v_as_1012_, v_startM1_1022_);
lean_dec_ref(v_as_1012_);
v___x_1027_ = lean_apply_2(v_f_1010_, v___x_1026_, v_init_1011_);
v___x_1028_ = lean_apply_4(v_toBind_1020_, lean_box(0), lean_box(0), v___x_1027_, v___f_1025_);
return v___x_1028_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg___lam__0(size_t v_startM1_1029_, size_t v_stop_1030_, lean_object* v_acc_1031_, lean_object* v_inst_1032_, lean_object* v_f_1033_, lean_object* v_as_1034_, lean_object* v_next_1035_){
_start:
{
size_t v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; 
v___x_1036_ = lean_usize_sub(v_startM1_1029_, v_stop_1030_);
lean_inc(v_next_1035_);
v___x_1037_ = lean_array_uset(v_acc_1031_, v___x_1036_, v_next_1035_);
v___x_1038_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg(v_inst_1032_, v_f_1033_, v_next_1035_, v_as_1034_, v_startM1_1029_, v_stop_1030_, v___x_1037_);
return v___x_1038_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg___boxed(lean_object* v_inst_1039_, lean_object* v_f_1040_, lean_object* v_init_1041_, lean_object* v_as_1042_, lean_object* v_start_1043_, lean_object* v_stop_1044_, lean_object* v_acc_1045_){
_start:
{
size_t v_start_boxed_1046_; size_t v_stop_boxed_1047_; lean_object* v_res_1048_; 
v_start_boxed_1046_ = lean_unbox_usize(v_start_1043_);
lean_dec(v_start_1043_);
v_stop_boxed_1047_ = lean_unbox_usize(v_stop_1044_);
lean_dec(v_stop_1044_);
v_res_1048_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg(v_inst_1039_, v_f_1040_, v_init_1041_, v_as_1042_, v_start_boxed_1046_, v_stop_boxed_1047_, v_acc_1045_);
return v_res_1048_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop(lean_object* v_m_1049_, lean_object* v_00_u03b1_1050_, lean_object* v_00_u03b2_1051_, lean_object* v_inst_1052_, lean_object* v_f_1053_, lean_object* v_init_1054_, lean_object* v_as_1055_, size_t v_start_1056_, size_t v_stop_1057_, lean_object* v_h__start_1058_, lean_object* v_acc_1059_, lean_object* v_h__bound_1060_){
_start:
{
lean_object* v___x_1061_; 
v___x_1061_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg(v_inst_1052_, v_f_1053_, v_init_1054_, v_as_1055_, v_start_1056_, v_stop_1057_, v_acc_1059_);
return v___x_1061_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___boxed(lean_object* v_m_1062_, lean_object* v_00_u03b1_1063_, lean_object* v_00_u03b2_1064_, lean_object* v_inst_1065_, lean_object* v_f_1066_, lean_object* v_init_1067_, lean_object* v_as_1068_, lean_object* v_start_1069_, lean_object* v_stop_1070_, lean_object* v_h__start_1071_, lean_object* v_acc_1072_, lean_object* v_h__bound_1073_){
_start:
{
size_t v_start_boxed_1074_; size_t v_stop_boxed_1075_; lean_object* v_res_1076_; 
v_start_boxed_1074_ = lean_unbox_usize(v_start_1069_);
lean_dec(v_start_1069_);
v_stop_boxed_1075_ = lean_unbox_usize(v_stop_1070_);
lean_dec(v_stop_1070_);
v_res_1076_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop(v_m_1062_, v_00_u03b1_1063_, v_00_u03b2_1064_, v_inst_1065_, v_f_1066_, v_init_1067_, v_as_1068_, v_start_boxed_1074_, v_stop_boxed_1075_, v_h__start_1071_, v_acc_1072_, v_h__bound_1073_);
return v_res_1076_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast___redArg(lean_object* v_inst_1077_, lean_object* v_f_1078_, lean_object* v_init_1079_, lean_object* v_as_1080_, lean_object* v_start_1081_, lean_object* v_stop_1082_){
_start:
{
lean_object* v___y_1084_; lean_object* v___y_1085_; lean_object* v___y_1094_; lean_object* v___x_1096_; uint8_t v___x_1097_; 
v___x_1096_ = lean_array_get_size(v_as_1080_);
v___x_1097_ = lean_nat_dec_le(v_start_1081_, v___x_1096_);
if (v___x_1097_ == 0)
{
lean_dec(v_start_1081_);
v___y_1094_ = v___x_1096_;
goto v___jp_1093_;
}
else
{
v___y_1094_ = v_start_1081_;
goto v___jp_1093_;
}
v___jp_1083_:
{
size_t v___x_1086_; size_t v___x_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; lean_object* v___x_1092_; 
v___x_1086_ = lean_usize_of_nat(v___y_1084_);
v___x_1087_ = lean_usize_of_nat(v___y_1085_);
v___x_1088_ = lean_nat_sub(v___y_1084_, v___y_1085_);
lean_dec(v___y_1085_);
lean_dec(v___y_1084_);
v___x_1089_ = lean_unsigned_to_nat(1u);
v___x_1090_ = lean_nat_add(v___x_1088_, v___x_1089_);
lean_dec(v___x_1088_);
lean_inc(v_init_1079_);
v___x_1091_ = lean_mk_array(v___x_1090_, v_init_1079_);
v___x_1092_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg(v_inst_1077_, v_f_1078_, v_init_1079_, v_as_1080_, v___x_1086_, v___x_1087_, v___x_1091_);
return v___x_1092_;
}
v___jp_1093_:
{
uint8_t v___x_1095_; 
v___x_1095_ = lean_nat_dec_le(v_stop_1082_, v___y_1094_);
if (v___x_1095_ == 0)
{
lean_dec(v_stop_1082_);
lean_inc(v___y_1094_);
v___y_1084_ = v___y_1094_;
v___y_1085_ = v___y_1094_;
goto v___jp_1083_;
}
else
{
v___y_1084_ = v___y_1094_;
v___y_1085_ = v_stop_1082_;
goto v___jp_1083_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast(lean_object* v_m_1098_, lean_object* v_00_u03b1_1099_, lean_object* v_00_u03b2_1100_, lean_object* v_inst_1101_, lean_object* v_f_1102_, lean_object* v_init_1103_, lean_object* v_as_1104_, lean_object* v_h__size_1105_, lean_object* v_start_1106_, lean_object* v_stop_1107_){
_start:
{
lean_object* v___y_1109_; lean_object* v___y_1110_; lean_object* v___y_1119_; lean_object* v___x_1121_; uint8_t v___x_1122_; 
v___x_1121_ = lean_array_get_size(v_as_1104_);
v___x_1122_ = lean_nat_dec_le(v_start_1106_, v___x_1121_);
if (v___x_1122_ == 0)
{
lean_dec(v_start_1106_);
v___y_1119_ = v___x_1121_;
goto v___jp_1118_;
}
else
{
v___y_1119_ = v_start_1106_;
goto v___jp_1118_;
}
v___jp_1108_:
{
size_t v___x_1111_; size_t v___x_1112_; lean_object* v___x_1113_; lean_object* v___x_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; 
v___x_1111_ = lean_usize_of_nat(v___y_1109_);
v___x_1112_ = lean_usize_of_nat(v___y_1110_);
v___x_1113_ = lean_nat_sub(v___y_1109_, v___y_1110_);
lean_dec(v___y_1110_);
lean_dec(v___y_1109_);
v___x_1114_ = lean_unsigned_to_nat(1u);
v___x_1115_ = lean_nat_add(v___x_1113_, v___x_1114_);
lean_dec(v___x_1113_);
lean_inc(v_init_1103_);
v___x_1116_ = lean_mk_array(v___x_1115_, v_init_1103_);
v___x_1117_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg(v_inst_1101_, v_f_1102_, v_init_1103_, v_as_1104_, v___x_1111_, v___x_1112_, v___x_1116_);
return v___x_1117_;
}
v___jp_1118_:
{
uint8_t v___x_1120_; 
v___x_1120_ = lean_nat_dec_le(v_stop_1107_, v___y_1119_);
if (v___x_1120_ == 0)
{
lean_dec(v_stop_1107_);
lean_inc(v___y_1119_);
v___y_1109_ = v___y_1119_;
v___y_1110_ = v___y_1119_;
goto v___jp_1108_;
}
else
{
v___y_1109_ = v___y_1119_;
v___y_1110_ = v_stop_1107_;
goto v___jp_1108_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMUnsafe___redArg(lean_object* v_inst_1123_, lean_object* v_f_1124_, lean_object* v_init_1125_, lean_object* v_as_1126_, lean_object* v_start_1127_, lean_object* v_stop_1128_){
_start:
{
lean_object* v___y_1130_; lean_object* v___y_1131_; lean_object* v___y_1140_; lean_object* v___x_1142_; uint8_t v___x_1143_; 
v___x_1142_ = lean_array_get_size(v_as_1126_);
v___x_1143_ = lean_nat_dec_le(v_start_1127_, v___x_1142_);
if (v___x_1143_ == 0)
{
lean_dec(v_start_1127_);
v___y_1140_ = v___x_1142_;
goto v___jp_1139_;
}
else
{
v___y_1140_ = v_start_1127_;
goto v___jp_1139_;
}
v___jp_1129_:
{
size_t v___x_1132_; size_t v___x_1133_; lean_object* v___x_1134_; lean_object* v___x_1135_; lean_object* v___x_1136_; lean_object* v___x_1137_; lean_object* v___x_1138_; 
v___x_1132_ = lean_usize_of_nat(v___y_1130_);
v___x_1133_ = lean_usize_of_nat(v___y_1131_);
v___x_1134_ = lean_nat_sub(v___y_1130_, v___y_1131_);
lean_dec(v___y_1131_);
lean_dec(v___y_1130_);
v___x_1135_ = lean_unsigned_to_nat(1u);
v___x_1136_ = lean_nat_add(v___x_1134_, v___x_1135_);
lean_dec(v___x_1134_);
lean_inc(v_init_1125_);
v___x_1137_ = lean_mk_array(v___x_1136_, v_init_1125_);
v___x_1138_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg(v_inst_1123_, v_f_1124_, v_init_1125_, v_as_1126_, v___x_1132_, v___x_1133_, v___x_1137_);
return v___x_1138_;
}
v___jp_1139_:
{
uint8_t v___x_1141_; 
v___x_1141_ = lean_nat_dec_le(v_stop_1128_, v___y_1140_);
if (v___x_1141_ == 0)
{
lean_dec(v_stop_1128_);
lean_inc(v___y_1140_);
v___y_1130_ = v___y_1140_;
v___y_1131_ = v___y_1140_;
goto v___jp_1129_;
}
else
{
v___y_1130_ = v___y_1140_;
v___y_1131_ = v_stop_1128_;
goto v___jp_1129_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMUnsafe(lean_object* v_m_1144_, lean_object* v_00_u03b1_1145_, lean_object* v_00_u03b2_1146_, lean_object* v_inst_1147_, lean_object* v_f_1148_, lean_object* v_init_1149_, lean_object* v_as_1150_, lean_object* v_start_1151_, lean_object* v_stop_1152_){
_start:
{
lean_object* v___y_1154_; lean_object* v___y_1155_; lean_object* v___y_1164_; lean_object* v___x_1166_; uint8_t v___x_1167_; 
v___x_1166_ = lean_array_get_size(v_as_1150_);
v___x_1167_ = lean_nat_dec_le(v_start_1151_, v___x_1166_);
if (v___x_1167_ == 0)
{
lean_dec(v_start_1151_);
v___y_1164_ = v___x_1166_;
goto v___jp_1163_;
}
else
{
v___y_1164_ = v_start_1151_;
goto v___jp_1163_;
}
v___jp_1153_:
{
size_t v___x_1156_; size_t v___x_1157_; lean_object* v___x_1158_; lean_object* v___x_1159_; lean_object* v___x_1160_; lean_object* v___x_1161_; lean_object* v___x_1162_; 
v___x_1156_ = lean_usize_of_nat(v___y_1154_);
v___x_1157_ = lean_usize_of_nat(v___y_1155_);
v___x_1158_ = lean_nat_sub(v___y_1154_, v___y_1155_);
lean_dec(v___y_1155_);
lean_dec(v___y_1154_);
v___x_1159_ = lean_unsigned_to_nat(1u);
v___x_1160_ = lean_nat_add(v___x_1158_, v___x_1159_);
lean_dec(v___x_1158_);
lean_inc(v_init_1149_);
v___x_1161_ = lean_mk_array(v___x_1160_, v_init_1149_);
v___x_1162_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg(v_inst_1147_, v_f_1148_, v_init_1149_, v_as_1150_, v___x_1156_, v___x_1157_, v___x_1161_);
return v___x_1162_;
}
v___jp_1163_:
{
uint8_t v___x_1165_; 
v___x_1165_ = lean_nat_dec_le(v_stop_1152_, v___y_1164_);
if (v___x_1165_ == 0)
{
lean_dec(v_stop_1152_);
lean_inc(v___y_1164_);
v___y_1154_ = v___y_1164_;
v___y_1155_ = v___y_1164_;
goto v___jp_1153_;
}
else
{
v___y_1154_ = v___y_1164_;
v___y_1155_ = v_stop_1152_;
goto v___jp_1153_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_scanrM_loop___redArg___lam__0___boxed(lean_object* v_acc_1168_, lean_object* v_init_1169_, lean_object* v_inst_1170_, lean_object* v_f_1171_, lean_object* v_as_1172_, lean_object* v_i_1173_, lean_object* v_stop_1174_, lean_object* v_next_1175_){
_start:
{
lean_object* v_res_1176_; 
v_res_1176_ = lp_batteries_Array_scanrM_loop___redArg___lam__0(v_acc_1168_, v_init_1169_, v_inst_1170_, v_f_1171_, v_as_1172_, v_i_1173_, v_stop_1174_, v_next_1175_);
lean_dec(v_i_1173_);
return v_res_1176_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_scanrM_loop___redArg(lean_object* v_inst_1177_, lean_object* v_f_1178_, lean_object* v_init_1179_, lean_object* v_as_1180_, lean_object* v_start_1181_, lean_object* v_stop_1182_, lean_object* v_acc_1183_){
_start:
{
uint8_t v___x_1184_; 
v___x_1184_ = lean_nat_dec_lt(v_stop_1182_, v_start_1181_);
if (v___x_1184_ == 0)
{
lean_object* v_toApplicative_1185_; lean_object* v_toPure_1186_; lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; 
lean_dec(v_stop_1182_);
lean_dec_ref(v_as_1180_);
lean_dec(v_f_1178_);
v_toApplicative_1185_ = lean_ctor_get(v_inst_1177_, 0);
lean_inc_ref(v_toApplicative_1185_);
lean_dec_ref(v_inst_1177_);
v_toPure_1186_ = lean_ctor_get(v_toApplicative_1185_, 1);
lean_inc(v_toPure_1186_);
lean_dec_ref(v_toApplicative_1185_);
v___x_1187_ = lean_array_push(v_acc_1183_, v_init_1179_);
v___x_1188_ = l_Array_reverse___redArg(v___x_1187_);
v___x_1189_ = lean_apply_2(v_toPure_1186_, lean_box(0), v___x_1188_);
return v___x_1189_;
}
else
{
lean_object* v_toBind_1190_; lean_object* v___x_1191_; lean_object* v_i_1192_; lean_object* v___f_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; 
v_toBind_1190_ = lean_ctor_get(v_inst_1177_, 1);
lean_inc(v_toBind_1190_);
v___x_1191_ = lean_unsigned_to_nat(1u);
v_i_1192_ = lean_nat_sub(v_start_1181_, v___x_1191_);
lean_inc(v_i_1192_);
lean_inc_ref(v_as_1180_);
lean_inc(v_f_1178_);
lean_inc(v_init_1179_);
v___f_1193_ = lean_alloc_closure((void*)(lp_batteries_Array_scanrM_loop___redArg___lam__0___boxed), 8, 7);
lean_closure_set(v___f_1193_, 0, v_acc_1183_);
lean_closure_set(v___f_1193_, 1, v_init_1179_);
lean_closure_set(v___f_1193_, 2, v_inst_1177_);
lean_closure_set(v___f_1193_, 3, v_f_1178_);
lean_closure_set(v___f_1193_, 4, v_as_1180_);
lean_closure_set(v___f_1193_, 5, v_i_1192_);
lean_closure_set(v___f_1193_, 6, v_stop_1182_);
v___x_1194_ = lean_array_fget(v_as_1180_, v_i_1192_);
lean_dec(v_i_1192_);
lean_dec_ref(v_as_1180_);
v___x_1195_ = lean_apply_2(v_f_1178_, v___x_1194_, v_init_1179_);
v___x_1196_ = lean_apply_4(v_toBind_1190_, lean_box(0), lean_box(0), v___x_1195_, v___f_1193_);
return v___x_1196_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_scanrM_loop___redArg___lam__0(lean_object* v_acc_1197_, lean_object* v_init_1198_, lean_object* v_inst_1199_, lean_object* v_f_1200_, lean_object* v_as_1201_, lean_object* v_i_1202_, lean_object* v_stop_1203_, lean_object* v_next_1204_){
_start:
{
lean_object* v___x_1205_; lean_object* v___x_1206_; 
v___x_1205_ = lean_array_push(v_acc_1197_, v_init_1198_);
v___x_1206_ = lp_batteries_Array_scanrM_loop___redArg(v_inst_1199_, v_f_1200_, v_next_1204_, v_as_1201_, v_i_1202_, v_stop_1203_, v___x_1205_);
return v___x_1206_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_scanrM_loop___redArg___boxed(lean_object* v_inst_1207_, lean_object* v_f_1208_, lean_object* v_init_1209_, lean_object* v_as_1210_, lean_object* v_start_1211_, lean_object* v_stop_1212_, lean_object* v_acc_1213_){
_start:
{
lean_object* v_res_1214_; 
v_res_1214_ = lp_batteries_Array_scanrM_loop___redArg(v_inst_1207_, v_f_1208_, v_init_1209_, v_as_1210_, v_start_1211_, v_stop_1212_, v_acc_1213_);
lean_dec(v_start_1211_);
return v_res_1214_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_scanrM_loop(lean_object* v_m_1215_, lean_object* v_00_u03b1_1216_, lean_object* v_00_u03b2_1217_, lean_object* v_inst_1218_, lean_object* v_f_1219_, lean_object* v_init_1220_, lean_object* v_as_1221_, lean_object* v_start_1222_, lean_object* v_stop_1223_, lean_object* v_h__start_1224_, lean_object* v_acc_1225_){
_start:
{
lean_object* v___x_1226_; 
v___x_1226_ = lp_batteries_Array_scanrM_loop___redArg(v_inst_1218_, v_f_1219_, v_init_1220_, v_as_1221_, v_start_1222_, v_stop_1223_, v_acc_1225_);
return v___x_1226_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_scanrM_loop___boxed(lean_object* v_m_1227_, lean_object* v_00_u03b1_1228_, lean_object* v_00_u03b2_1229_, lean_object* v_inst_1230_, lean_object* v_f_1231_, lean_object* v_init_1232_, lean_object* v_as_1233_, lean_object* v_start_1234_, lean_object* v_stop_1235_, lean_object* v_h__start_1236_, lean_object* v_acc_1237_){
_start:
{
lean_object* v_res_1238_; 
v_res_1238_ = lp_batteries_Array_scanrM_loop(v_m_1227_, v_00_u03b1_1228_, v_00_u03b2_1229_, v_inst_1230_, v_f_1231_, v_init_1232_, v_as_1233_, v_start_1234_, v_stop_1235_, v_h__start_1236_, v_acc_1237_);
lean_dec(v_start_1234_);
return v_res_1238_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_scanl___redArg___lam__0(lean_object* v_f_1239_, lean_object* v_x1_1240_, lean_object* v_x2_1241_){
_start:
{
lean_object* v___x_1242_; 
v___x_1242_ = lean_apply_2(v_f_1239_, v_x1_1240_, v_x2_1241_);
return v___x_1242_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_scanl___redArg(lean_object* v_f_1243_, lean_object* v_init_1244_, lean_object* v_as_1245_, lean_object* v_start_1246_, lean_object* v_stop_1247_){
_start:
{
lean_object* v___f_1248_; lean_object* v___x_1249_; lean_object* v___y_1251_; lean_object* v___y_1252_; lean_object* v___x_1260_; lean_object* v___y_1262_; uint8_t v___x_1264_; 
v___f_1248_ = lean_alloc_closure((void*)(lp_batteries_Array_scanl___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1248_, 0, v_f_1243_);
v___x_1249_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_1260_ = lean_array_get_size(v_as_1245_);
v___x_1264_ = lean_nat_dec_le(v_stop_1247_, v___x_1260_);
if (v___x_1264_ == 0)
{
lean_dec(v_stop_1247_);
v___y_1262_ = v___x_1260_;
goto v___jp_1261_;
}
else
{
v___y_1262_ = v_stop_1247_;
goto v___jp_1261_;
}
v___jp_1250_:
{
size_t v___x_1253_; size_t v___x_1254_; lean_object* v___x_1255_; lean_object* v___x_1256_; lean_object* v___x_1257_; lean_object* v___x_1258_; lean_object* v___x_1259_; 
v___x_1253_ = lean_usize_of_nat(v___y_1252_);
v___x_1254_ = lean_usize_of_nat(v___y_1251_);
v___x_1255_ = lean_nat_sub(v___y_1251_, v___y_1252_);
lean_dec(v___y_1252_);
lean_dec(v___y_1251_);
v___x_1256_ = lean_unsigned_to_nat(1u);
v___x_1257_ = lean_nat_add(v___x_1255_, v___x_1256_);
lean_dec(v___x_1255_);
v___x_1258_ = lean_mk_empty_array_with_capacity(v___x_1257_);
lean_dec(v___x_1257_);
v___x_1259_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg(v___x_1249_, v___f_1248_, v_init_1244_, v_as_1245_, v___x_1253_, v___x_1254_, v___x_1258_);
return v___x_1259_;
}
v___jp_1261_:
{
uint8_t v___x_1263_; 
v___x_1263_ = lean_nat_dec_le(v_start_1246_, v___x_1260_);
if (v___x_1263_ == 0)
{
lean_dec(v_start_1246_);
v___y_1251_ = v___y_1262_;
v___y_1252_ = v___x_1260_;
goto v___jp_1250_;
}
else
{
v___y_1251_ = v___y_1262_;
v___y_1252_ = v_start_1246_;
goto v___jp_1250_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_scanl(lean_object* v_00_u03b2_1265_, lean_object* v_00_u03b1_1266_, lean_object* v_f_1267_, lean_object* v_init_1268_, lean_object* v_as_1269_, lean_object* v_start_1270_, lean_object* v_stop_1271_){
_start:
{
lean_object* v___f_1272_; lean_object* v___x_1273_; lean_object* v___y_1275_; lean_object* v___y_1276_; lean_object* v___x_1284_; lean_object* v___y_1286_; uint8_t v___x_1288_; 
v___f_1272_ = lean_alloc_closure((void*)(lp_batteries_Array_scanl___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1272_, 0, v_f_1267_);
v___x_1273_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_1284_ = lean_array_get_size(v_as_1269_);
v___x_1288_ = lean_nat_dec_le(v_stop_1271_, v___x_1284_);
if (v___x_1288_ == 0)
{
lean_dec(v_stop_1271_);
v___y_1286_ = v___x_1284_;
goto v___jp_1285_;
}
else
{
v___y_1286_ = v_stop_1271_;
goto v___jp_1285_;
}
v___jp_1274_:
{
size_t v___x_1277_; size_t v___x_1278_; lean_object* v___x_1279_; lean_object* v___x_1280_; lean_object* v___x_1281_; lean_object* v___x_1282_; lean_object* v___x_1283_; 
v___x_1277_ = lean_usize_of_nat(v___y_1276_);
v___x_1278_ = lean_usize_of_nat(v___y_1275_);
v___x_1279_ = lean_nat_sub(v___y_1275_, v___y_1276_);
lean_dec(v___y_1276_);
lean_dec(v___y_1275_);
v___x_1280_ = lean_unsigned_to_nat(1u);
v___x_1281_ = lean_nat_add(v___x_1279_, v___x_1280_);
lean_dec(v___x_1279_);
v___x_1282_ = lean_mk_empty_array_with_capacity(v___x_1281_);
lean_dec(v___x_1281_);
v___x_1283_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg(v___x_1273_, v___f_1272_, v_init_1268_, v_as_1269_, v___x_1277_, v___x_1278_, v___x_1282_);
return v___x_1283_;
}
v___jp_1285_:
{
uint8_t v___x_1287_; 
v___x_1287_ = lean_nat_dec_le(v_start_1270_, v___x_1284_);
if (v___x_1287_ == 0)
{
lean_dec(v_start_1270_);
v___y_1275_ = v___y_1286_;
v___y_1276_ = v___x_1284_;
goto v___jp_1274_;
}
else
{
v___y_1275_ = v___y_1286_;
v___y_1276_ = v_start_1270_;
goto v___jp_1274_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_scanr___redArg(lean_object* v_f_1289_, lean_object* v_init_1290_, lean_object* v_as_1291_, lean_object* v_start_1292_, lean_object* v_stop_1293_){
_start:
{
lean_object* v___f_1294_; lean_object* v___x_1295_; lean_object* v___y_1297_; lean_object* v___y_1298_; lean_object* v___y_1307_; lean_object* v___x_1309_; uint8_t v___x_1310_; 
v___f_1294_ = lean_alloc_closure((void*)(lp_batteries_Array_scanl___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1294_, 0, v_f_1289_);
v___x_1295_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_1309_ = lean_array_get_size(v_as_1291_);
v___x_1310_ = lean_nat_dec_le(v_start_1292_, v___x_1309_);
if (v___x_1310_ == 0)
{
lean_dec(v_start_1292_);
v___y_1307_ = v___x_1309_;
goto v___jp_1306_;
}
else
{
v___y_1307_ = v_start_1292_;
goto v___jp_1306_;
}
v___jp_1296_:
{
size_t v___x_1299_; size_t v___x_1300_; lean_object* v___x_1301_; lean_object* v___x_1302_; lean_object* v___x_1303_; lean_object* v___x_1304_; lean_object* v___x_1305_; 
v___x_1299_ = lean_usize_of_nat(v___y_1297_);
v___x_1300_ = lean_usize_of_nat(v___y_1298_);
v___x_1301_ = lean_nat_sub(v___y_1297_, v___y_1298_);
lean_dec(v___y_1298_);
lean_dec(v___y_1297_);
v___x_1302_ = lean_unsigned_to_nat(1u);
v___x_1303_ = lean_nat_add(v___x_1301_, v___x_1302_);
lean_dec(v___x_1301_);
lean_inc(v_init_1290_);
v___x_1304_ = lean_mk_array(v___x_1303_, v_init_1290_);
v___x_1305_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg(v___x_1295_, v___f_1294_, v_init_1290_, v_as_1291_, v___x_1299_, v___x_1300_, v___x_1304_);
return v___x_1305_;
}
v___jp_1306_:
{
uint8_t v___x_1308_; 
v___x_1308_ = lean_nat_dec_le(v_stop_1293_, v___y_1307_);
if (v___x_1308_ == 0)
{
lean_dec(v_stop_1293_);
lean_inc(v___y_1307_);
v___y_1297_ = v___y_1307_;
v___y_1298_ = v___y_1307_;
goto v___jp_1296_;
}
else
{
v___y_1297_ = v___y_1307_;
v___y_1298_ = v_stop_1293_;
goto v___jp_1296_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Array_scanr(lean_object* v_00_u03b1_1311_, lean_object* v_00_u03b2_1312_, lean_object* v_f_1313_, lean_object* v_init_1314_, lean_object* v_as_1315_, lean_object* v_start_1316_, lean_object* v_stop_1317_){
_start:
{
lean_object* v___f_1318_; lean_object* v___x_1319_; lean_object* v___y_1321_; lean_object* v___y_1322_; lean_object* v___y_1331_; lean_object* v___x_1333_; uint8_t v___x_1334_; 
v___f_1318_ = lean_alloc_closure((void*)(lp_batteries_Array_scanl___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1318_, 0, v_f_1313_);
v___x_1319_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_1333_ = lean_array_get_size(v_as_1315_);
v___x_1334_ = lean_nat_dec_le(v_start_1316_, v___x_1333_);
if (v___x_1334_ == 0)
{
lean_dec(v_start_1316_);
v___y_1331_ = v___x_1333_;
goto v___jp_1330_;
}
else
{
v___y_1331_ = v_start_1316_;
goto v___jp_1330_;
}
v___jp_1320_:
{
size_t v___x_1323_; size_t v___x_1324_; lean_object* v___x_1325_; lean_object* v___x_1326_; lean_object* v___x_1327_; lean_object* v___x_1328_; lean_object* v___x_1329_; 
v___x_1323_ = lean_usize_of_nat(v___y_1321_);
v___x_1324_ = lean_usize_of_nat(v___y_1322_);
v___x_1325_ = lean_nat_sub(v___y_1321_, v___y_1322_);
lean_dec(v___y_1322_);
lean_dec(v___y_1321_);
v___x_1326_ = lean_unsigned_to_nat(1u);
v___x_1327_ = lean_nat_add(v___x_1325_, v___x_1326_);
lean_dec(v___x_1325_);
lean_inc(v_init_1314_);
v___x_1328_ = lean_mk_array(v___x_1327_, v_init_1314_);
v___x_1329_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg(v___x_1319_, v___f_1318_, v_init_1314_, v_as_1315_, v___x_1323_, v___x_1324_, v___x_1328_);
return v___x_1329_;
}
v___jp_1330_:
{
uint8_t v___x_1332_; 
v___x_1332_ = lean_nat_dec_le(v_stop_1317_, v___y_1331_);
if (v___x_1332_ == 0)
{
lean_dec(v_stop_1317_);
lean_inc(v___y_1331_);
v___y_1321_ = v___y_1331_;
v___y_1322_ = v___y_1331_;
goto v___jp_1320_;
}
else
{
v___y_1321_ = v___y_1331_;
v___y_1322_ = v_stop_1317_;
goto v___jp_1320_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Subarray_scanlM___redArg(lean_object* v_inst_1335_, lean_object* v_f_1336_, lean_object* v_init_1337_, lean_object* v_as_1338_){
_start:
{
lean_object* v_array_1339_; lean_object* v_start_1340_; lean_object* v_stop_1341_; lean_object* v___y_1343_; lean_object* v___y_1344_; lean_object* v___x_1352_; lean_object* v___y_1354_; uint8_t v___x_1356_; 
v_array_1339_ = lean_ctor_get(v_as_1338_, 0);
lean_inc_ref(v_array_1339_);
v_start_1340_ = lean_ctor_get(v_as_1338_, 1);
lean_inc(v_start_1340_);
v_stop_1341_ = lean_ctor_get(v_as_1338_, 2);
lean_inc(v_stop_1341_);
lean_dec_ref(v_as_1338_);
v___x_1352_ = lean_array_get_size(v_array_1339_);
v___x_1356_ = lean_nat_dec_le(v_stop_1341_, v___x_1352_);
if (v___x_1356_ == 0)
{
lean_dec(v_stop_1341_);
v___y_1354_ = v___x_1352_;
goto v___jp_1353_;
}
else
{
v___y_1354_ = v_stop_1341_;
goto v___jp_1353_;
}
v___jp_1342_:
{
size_t v___x_1345_; size_t v___x_1346_; lean_object* v___x_1347_; lean_object* v___x_1348_; lean_object* v___x_1349_; lean_object* v___x_1350_; lean_object* v___x_1351_; 
v___x_1345_ = lean_usize_of_nat(v___y_1344_);
v___x_1346_ = lean_usize_of_nat(v___y_1343_);
v___x_1347_ = lean_nat_sub(v___y_1343_, v___y_1344_);
lean_dec(v___y_1344_);
lean_dec(v___y_1343_);
v___x_1348_ = lean_unsigned_to_nat(1u);
v___x_1349_ = lean_nat_add(v___x_1347_, v___x_1348_);
lean_dec(v___x_1347_);
v___x_1350_ = lean_mk_empty_array_with_capacity(v___x_1349_);
lean_dec(v___x_1349_);
v___x_1351_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg(v_inst_1335_, v_f_1336_, v_init_1337_, v_array_1339_, v___x_1345_, v___x_1346_, v___x_1350_);
return v___x_1351_;
}
v___jp_1353_:
{
uint8_t v___x_1355_; 
v___x_1355_ = lean_nat_dec_le(v_start_1340_, v___x_1352_);
if (v___x_1355_ == 0)
{
lean_dec(v_start_1340_);
v___y_1343_ = v___y_1354_;
v___y_1344_ = v___x_1352_;
goto v___jp_1342_;
}
else
{
v___y_1343_ = v___y_1354_;
v___y_1344_ = v_start_1340_;
goto v___jp_1342_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Subarray_scanlM(lean_object* v_m_1357_, lean_object* v_00_u03b2_1358_, lean_object* v_00_u03b1_1359_, lean_object* v_inst_1360_, lean_object* v_f_1361_, lean_object* v_init_1362_, lean_object* v_as_1363_){
_start:
{
lean_object* v_array_1364_; lean_object* v_start_1365_; lean_object* v_stop_1366_; lean_object* v___y_1368_; lean_object* v___y_1369_; lean_object* v___x_1377_; lean_object* v___y_1379_; uint8_t v___x_1381_; 
v_array_1364_ = lean_ctor_get(v_as_1363_, 0);
lean_inc_ref(v_array_1364_);
v_start_1365_ = lean_ctor_get(v_as_1363_, 1);
lean_inc(v_start_1365_);
v_stop_1366_ = lean_ctor_get(v_as_1363_, 2);
lean_inc(v_stop_1366_);
lean_dec_ref(v_as_1363_);
v___x_1377_ = lean_array_get_size(v_array_1364_);
v___x_1381_ = lean_nat_dec_le(v_stop_1366_, v___x_1377_);
if (v___x_1381_ == 0)
{
lean_dec(v_stop_1366_);
v___y_1379_ = v___x_1377_;
goto v___jp_1378_;
}
else
{
v___y_1379_ = v_stop_1366_;
goto v___jp_1378_;
}
v___jp_1367_:
{
size_t v___x_1370_; size_t v___x_1371_; lean_object* v___x_1372_; lean_object* v___x_1373_; lean_object* v___x_1374_; lean_object* v___x_1375_; lean_object* v___x_1376_; 
v___x_1370_ = lean_usize_of_nat(v___y_1369_);
v___x_1371_ = lean_usize_of_nat(v___y_1368_);
v___x_1372_ = lean_nat_sub(v___y_1368_, v___y_1369_);
lean_dec(v___y_1369_);
lean_dec(v___y_1368_);
v___x_1373_ = lean_unsigned_to_nat(1u);
v___x_1374_ = lean_nat_add(v___x_1372_, v___x_1373_);
lean_dec(v___x_1372_);
v___x_1375_ = lean_mk_empty_array_with_capacity(v___x_1374_);
lean_dec(v___x_1374_);
v___x_1376_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg(v_inst_1360_, v_f_1361_, v_init_1362_, v_array_1364_, v___x_1370_, v___x_1371_, v___x_1375_);
return v___x_1376_;
}
v___jp_1378_:
{
uint8_t v___x_1380_; 
v___x_1380_ = lean_nat_dec_le(v_start_1365_, v___x_1377_);
if (v___x_1380_ == 0)
{
lean_dec(v_start_1365_);
v___y_1368_ = v___y_1379_;
v___y_1369_ = v___x_1377_;
goto v___jp_1367_;
}
else
{
v___y_1368_ = v___y_1379_;
v___y_1369_ = v_start_1365_;
goto v___jp_1367_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Subarray_scanrM___redArg(lean_object* v_inst_1382_, lean_object* v_f_1383_, lean_object* v_init_1384_, lean_object* v_as_1385_){
_start:
{
lean_object* v_array_1386_; lean_object* v_start_1387_; lean_object* v_stop_1388_; lean_object* v___y_1390_; lean_object* v___y_1391_; lean_object* v___y_1400_; lean_object* v___x_1402_; uint8_t v___x_1403_; 
v_array_1386_ = lean_ctor_get(v_as_1385_, 0);
lean_inc_ref(v_array_1386_);
v_start_1387_ = lean_ctor_get(v_as_1385_, 1);
lean_inc(v_start_1387_);
v_stop_1388_ = lean_ctor_get(v_as_1385_, 2);
lean_inc(v_stop_1388_);
lean_dec_ref(v_as_1385_);
v___x_1402_ = lean_array_get_size(v_array_1386_);
v___x_1403_ = lean_nat_dec_le(v_start_1387_, v___x_1402_);
if (v___x_1403_ == 0)
{
lean_dec(v_start_1387_);
v___y_1400_ = v___x_1402_;
goto v___jp_1399_;
}
else
{
v___y_1400_ = v_start_1387_;
goto v___jp_1399_;
}
v___jp_1389_:
{
size_t v___x_1392_; size_t v___x_1393_; lean_object* v___x_1394_; lean_object* v___x_1395_; lean_object* v___x_1396_; lean_object* v___x_1397_; lean_object* v___x_1398_; 
v___x_1392_ = lean_usize_of_nat(v___y_1390_);
v___x_1393_ = lean_usize_of_nat(v___y_1391_);
v___x_1394_ = lean_nat_sub(v___y_1390_, v___y_1391_);
lean_dec(v___y_1391_);
lean_dec(v___y_1390_);
v___x_1395_ = lean_unsigned_to_nat(1u);
v___x_1396_ = lean_nat_add(v___x_1394_, v___x_1395_);
lean_dec(v___x_1394_);
lean_inc(v_init_1384_);
v___x_1397_ = lean_mk_array(v___x_1396_, v_init_1384_);
v___x_1398_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg(v_inst_1382_, v_f_1383_, v_init_1384_, v_array_1386_, v___x_1392_, v___x_1393_, v___x_1397_);
return v___x_1398_;
}
v___jp_1399_:
{
uint8_t v___x_1401_; 
v___x_1401_ = lean_nat_dec_le(v_stop_1388_, v___y_1400_);
if (v___x_1401_ == 0)
{
lean_dec(v_stop_1388_);
lean_inc(v___y_1400_);
v___y_1390_ = v___y_1400_;
v___y_1391_ = v___y_1400_;
goto v___jp_1389_;
}
else
{
v___y_1390_ = v___y_1400_;
v___y_1391_ = v_stop_1388_;
goto v___jp_1389_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Subarray_scanrM(lean_object* v_m_1404_, lean_object* v_00_u03b1_1405_, lean_object* v_00_u03b2_1406_, lean_object* v_inst_1407_, lean_object* v_f_1408_, lean_object* v_init_1409_, lean_object* v_as_1410_){
_start:
{
lean_object* v_array_1411_; lean_object* v_start_1412_; lean_object* v_stop_1413_; lean_object* v___y_1415_; lean_object* v___y_1416_; lean_object* v___y_1425_; lean_object* v___x_1427_; uint8_t v___x_1428_; 
v_array_1411_ = lean_ctor_get(v_as_1410_, 0);
lean_inc_ref(v_array_1411_);
v_start_1412_ = lean_ctor_get(v_as_1410_, 1);
lean_inc(v_start_1412_);
v_stop_1413_ = lean_ctor_get(v_as_1410_, 2);
lean_inc(v_stop_1413_);
lean_dec_ref(v_as_1410_);
v___x_1427_ = lean_array_get_size(v_array_1411_);
v___x_1428_ = lean_nat_dec_le(v_start_1412_, v___x_1427_);
if (v___x_1428_ == 0)
{
lean_dec(v_start_1412_);
v___y_1425_ = v___x_1427_;
goto v___jp_1424_;
}
else
{
v___y_1425_ = v_start_1412_;
goto v___jp_1424_;
}
v___jp_1414_:
{
size_t v___x_1417_; size_t v___x_1418_; lean_object* v___x_1419_; lean_object* v___x_1420_; lean_object* v___x_1421_; lean_object* v___x_1422_; lean_object* v___x_1423_; 
v___x_1417_ = lean_usize_of_nat(v___y_1415_);
v___x_1418_ = lean_usize_of_nat(v___y_1416_);
v___x_1419_ = lean_nat_sub(v___y_1415_, v___y_1416_);
lean_dec(v___y_1416_);
lean_dec(v___y_1415_);
v___x_1420_ = lean_unsigned_to_nat(1u);
v___x_1421_ = lean_nat_add(v___x_1419_, v___x_1420_);
lean_dec(v___x_1419_);
lean_inc(v_init_1409_);
v___x_1422_ = lean_mk_array(v___x_1421_, v_init_1409_);
v___x_1423_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg(v_inst_1407_, v_f_1408_, v_init_1409_, v_array_1411_, v___x_1417_, v___x_1418_, v___x_1422_);
return v___x_1423_;
}
v___jp_1424_:
{
uint8_t v___x_1426_; 
v___x_1426_ = lean_nat_dec_le(v_stop_1413_, v___y_1425_);
if (v___x_1426_ == 0)
{
lean_dec(v_stop_1413_);
lean_inc(v___y_1425_);
v___y_1415_ = v___y_1425_;
v___y_1416_ = v___y_1425_;
goto v___jp_1414_;
}
else
{
v___y_1415_ = v___y_1425_;
v___y_1416_ = v_stop_1413_;
goto v___jp_1414_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Subarray_scanl___redArg(lean_object* v_f_1429_, lean_object* v_init_1430_, lean_object* v_as_1431_){
_start:
{
lean_object* v_array_1432_; lean_object* v_start_1433_; lean_object* v_stop_1434_; lean_object* v___f_1435_; lean_object* v___x_1436_; lean_object* v___y_1438_; lean_object* v___y_1439_; lean_object* v___x_1447_; lean_object* v___y_1449_; uint8_t v___x_1451_; 
v_array_1432_ = lean_ctor_get(v_as_1431_, 0);
lean_inc_ref(v_array_1432_);
v_start_1433_ = lean_ctor_get(v_as_1431_, 1);
lean_inc(v_start_1433_);
v_stop_1434_ = lean_ctor_get(v_as_1431_, 2);
lean_inc(v_stop_1434_);
lean_dec_ref(v_as_1431_);
v___f_1435_ = lean_alloc_closure((void*)(lp_batteries_Array_scanl___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1435_, 0, v_f_1429_);
v___x_1436_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_1447_ = lean_array_get_size(v_array_1432_);
v___x_1451_ = lean_nat_dec_le(v_stop_1434_, v___x_1447_);
if (v___x_1451_ == 0)
{
lean_dec(v_stop_1434_);
v___y_1449_ = v___x_1447_;
goto v___jp_1448_;
}
else
{
v___y_1449_ = v_stop_1434_;
goto v___jp_1448_;
}
v___jp_1437_:
{
size_t v___x_1440_; size_t v___x_1441_; lean_object* v___x_1442_; lean_object* v___x_1443_; lean_object* v___x_1444_; lean_object* v___x_1445_; lean_object* v___x_1446_; 
v___x_1440_ = lean_usize_of_nat(v___y_1439_);
v___x_1441_ = lean_usize_of_nat(v___y_1438_);
v___x_1442_ = lean_nat_sub(v___y_1438_, v___y_1439_);
lean_dec(v___y_1439_);
lean_dec(v___y_1438_);
v___x_1443_ = lean_unsigned_to_nat(1u);
v___x_1444_ = lean_nat_add(v___x_1442_, v___x_1443_);
lean_dec(v___x_1442_);
v___x_1445_ = lean_mk_empty_array_with_capacity(v___x_1444_);
lean_dec(v___x_1444_);
v___x_1446_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg(v___x_1436_, v___f_1435_, v_init_1430_, v_array_1432_, v___x_1440_, v___x_1441_, v___x_1445_);
return v___x_1446_;
}
v___jp_1448_:
{
uint8_t v___x_1450_; 
v___x_1450_ = lean_nat_dec_le(v_start_1433_, v___x_1447_);
if (v___x_1450_ == 0)
{
lean_dec(v_start_1433_);
v___y_1438_ = v___y_1449_;
v___y_1439_ = v___x_1447_;
goto v___jp_1437_;
}
else
{
v___y_1438_ = v___y_1449_;
v___y_1439_ = v_start_1433_;
goto v___jp_1437_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Subarray_scanl(lean_object* v_00_u03b2_1452_, lean_object* v_00_u03b1_1453_, lean_object* v_f_1454_, lean_object* v_init_1455_, lean_object* v_as_1456_){
_start:
{
lean_object* v_array_1457_; lean_object* v_start_1458_; lean_object* v_stop_1459_; lean_object* v___f_1460_; lean_object* v___x_1461_; lean_object* v___y_1463_; lean_object* v___y_1464_; lean_object* v___x_1472_; lean_object* v___y_1474_; uint8_t v___x_1476_; 
v_array_1457_ = lean_ctor_get(v_as_1456_, 0);
lean_inc_ref(v_array_1457_);
v_start_1458_ = lean_ctor_get(v_as_1456_, 1);
lean_inc(v_start_1458_);
v_stop_1459_ = lean_ctor_get(v_as_1456_, 2);
lean_inc(v_stop_1459_);
lean_dec_ref(v_as_1456_);
v___f_1460_ = lean_alloc_closure((void*)(lp_batteries_Array_scanl___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1460_, 0, v_f_1454_);
v___x_1461_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_1472_ = lean_array_get_size(v_array_1457_);
v___x_1476_ = lean_nat_dec_le(v_stop_1459_, v___x_1472_);
if (v___x_1476_ == 0)
{
lean_dec(v_stop_1459_);
v___y_1474_ = v___x_1472_;
goto v___jp_1473_;
}
else
{
v___y_1474_ = v_stop_1459_;
goto v___jp_1473_;
}
v___jp_1462_:
{
size_t v___x_1465_; size_t v___x_1466_; lean_object* v___x_1467_; lean_object* v___x_1468_; lean_object* v___x_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; 
v___x_1465_ = lean_usize_of_nat(v___y_1464_);
v___x_1466_ = lean_usize_of_nat(v___y_1463_);
v___x_1467_ = lean_nat_sub(v___y_1463_, v___y_1464_);
lean_dec(v___y_1464_);
lean_dec(v___y_1463_);
v___x_1468_ = lean_unsigned_to_nat(1u);
v___x_1469_ = lean_nat_add(v___x_1467_, v___x_1468_);
lean_dec(v___x_1467_);
v___x_1470_ = lean_mk_empty_array_with_capacity(v___x_1469_);
lean_dec(v___x_1469_);
v___x_1471_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanlMFast_loop___redArg(v___x_1461_, v___f_1460_, v_init_1455_, v_array_1457_, v___x_1465_, v___x_1466_, v___x_1470_);
return v___x_1471_;
}
v___jp_1473_:
{
uint8_t v___x_1475_; 
v___x_1475_ = lean_nat_dec_le(v_start_1458_, v___x_1472_);
if (v___x_1475_ == 0)
{
lean_dec(v_start_1458_);
v___y_1463_ = v___y_1474_;
v___y_1464_ = v___x_1472_;
goto v___jp_1462_;
}
else
{
v___y_1463_ = v___y_1474_;
v___y_1464_ = v_start_1458_;
goto v___jp_1462_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Subarray_scanr___redArg(lean_object* v_f_1477_, lean_object* v_init_1478_, lean_object* v_as_1479_){
_start:
{
lean_object* v_array_1480_; lean_object* v_start_1481_; lean_object* v_stop_1482_; lean_object* v___f_1483_; lean_object* v___x_1484_; lean_object* v___y_1486_; lean_object* v___y_1487_; lean_object* v___y_1496_; lean_object* v___x_1498_; uint8_t v___x_1499_; 
v_array_1480_ = lean_ctor_get(v_as_1479_, 0);
lean_inc_ref(v_array_1480_);
v_start_1481_ = lean_ctor_get(v_as_1479_, 1);
lean_inc(v_start_1481_);
v_stop_1482_ = lean_ctor_get(v_as_1479_, 2);
lean_inc(v_stop_1482_);
lean_dec_ref(v_as_1479_);
v___f_1483_ = lean_alloc_closure((void*)(lp_batteries_Array_scanl___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1483_, 0, v_f_1477_);
v___x_1484_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_1498_ = lean_array_get_size(v_array_1480_);
v___x_1499_ = lean_nat_dec_le(v_start_1481_, v___x_1498_);
if (v___x_1499_ == 0)
{
lean_dec(v_start_1481_);
v___y_1496_ = v___x_1498_;
goto v___jp_1495_;
}
else
{
v___y_1496_ = v_start_1481_;
goto v___jp_1495_;
}
v___jp_1485_:
{
size_t v___x_1488_; size_t v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; lean_object* v___x_1492_; lean_object* v___x_1493_; lean_object* v___x_1494_; 
v___x_1488_ = lean_usize_of_nat(v___y_1486_);
v___x_1489_ = lean_usize_of_nat(v___y_1487_);
v___x_1490_ = lean_nat_sub(v___y_1486_, v___y_1487_);
lean_dec(v___y_1487_);
lean_dec(v___y_1486_);
v___x_1491_ = lean_unsigned_to_nat(1u);
v___x_1492_ = lean_nat_add(v___x_1490_, v___x_1491_);
lean_dec(v___x_1490_);
lean_inc(v_init_1478_);
v___x_1493_ = lean_mk_array(v___x_1492_, v_init_1478_);
v___x_1494_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg(v___x_1484_, v___f_1483_, v_init_1478_, v_array_1480_, v___x_1488_, v___x_1489_, v___x_1493_);
return v___x_1494_;
}
v___jp_1495_:
{
uint8_t v___x_1497_; 
v___x_1497_ = lean_nat_dec_le(v_stop_1482_, v___y_1496_);
if (v___x_1497_ == 0)
{
lean_dec(v_stop_1482_);
lean_inc(v___y_1496_);
v___y_1486_ = v___y_1496_;
v___y_1487_ = v___y_1496_;
goto v___jp_1485_;
}
else
{
v___y_1486_ = v___y_1496_;
v___y_1487_ = v_stop_1482_;
goto v___jp_1485_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Subarray_scanr(lean_object* v_00_u03b1_1500_, lean_object* v_00_u03b2_1501_, lean_object* v_f_1502_, lean_object* v_init_1503_, lean_object* v_as_1504_){
_start:
{
lean_object* v_array_1505_; lean_object* v_start_1506_; lean_object* v_stop_1507_; lean_object* v___f_1508_; lean_object* v___x_1509_; lean_object* v___y_1511_; lean_object* v___y_1512_; lean_object* v___y_1521_; lean_object* v___x_1523_; uint8_t v___x_1524_; 
v_array_1505_ = lean_ctor_get(v_as_1504_, 0);
lean_inc_ref(v_array_1505_);
v_start_1506_ = lean_ctor_get(v_as_1504_, 1);
lean_inc(v_start_1506_);
v_stop_1507_ = lean_ctor_get(v_as_1504_, 2);
lean_inc(v_stop_1507_);
lean_dec_ref(v_as_1504_);
v___f_1508_ = lean_alloc_closure((void*)(lp_batteries_Array_scanl___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1508_, 0, v_f_1502_);
v___x_1509_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v___x_1523_ = lean_array_get_size(v_array_1505_);
v___x_1524_ = lean_nat_dec_le(v_start_1506_, v___x_1523_);
if (v___x_1524_ == 0)
{
lean_dec(v_start_1506_);
v___y_1521_ = v___x_1523_;
goto v___jp_1520_;
}
else
{
v___y_1521_ = v_start_1506_;
goto v___jp_1520_;
}
v___jp_1510_:
{
size_t v___x_1513_; size_t v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1516_; lean_object* v___x_1517_; lean_object* v___x_1518_; lean_object* v___x_1519_; 
v___x_1513_ = lean_usize_of_nat(v___y_1511_);
v___x_1514_ = lean_usize_of_nat(v___y_1512_);
v___x_1515_ = lean_nat_sub(v___y_1511_, v___y_1512_);
lean_dec(v___y_1512_);
lean_dec(v___y_1511_);
v___x_1516_ = lean_unsigned_to_nat(1u);
v___x_1517_ = lean_nat_add(v___x_1515_, v___x_1516_);
lean_dec(v___x_1515_);
lean_inc(v_init_1503_);
v___x_1518_ = lean_mk_array(v___x_1517_, v_init_1503_);
v___x_1519_ = lp_batteries___private_Batteries_Data_Array_Basic_0__Array_scanrMFast_loop___redArg(v___x_1509_, v___f_1508_, v_init_1503_, v_array_1505_, v___x_1513_, v___x_1514_, v___x_1518_);
return v___x_1519_;
}
v___jp_1520_:
{
uint8_t v___x_1522_; 
v___x_1522_ = lean_nat_dec_le(v_stop_1507_, v___y_1521_);
if (v___x_1522_ == 0)
{
lean_dec(v_stop_1507_);
lean_inc(v___y_1521_);
v___y_1511_ = v___y_1521_;
v___y_1512_ = v___y_1521_;
goto v___jp_1510_;
}
else
{
v___y_1511_ = v___y_1521_;
v___y_1512_ = v_stop_1507_;
goto v___jp_1510_;
}
}
}
}
LEAN_EXPORT uint8_t lp_batteries_Subarray_isEmpty___redArg(lean_object* v_as_1525_){
_start:
{
lean_object* v_start_1526_; lean_object* v_stop_1527_; uint8_t v___x_1528_; 
v_start_1526_ = lean_ctor_get(v_as_1525_, 1);
v_stop_1527_ = lean_ctor_get(v_as_1525_, 2);
v___x_1528_ = lean_nat_dec_eq(v_start_1526_, v_stop_1527_);
return v___x_1528_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Subarray_isEmpty___redArg___boxed(lean_object* v_as_1529_){
_start:
{
uint8_t v_res_1530_; lean_object* v_r_1531_; 
v_res_1530_ = lp_batteries_Subarray_isEmpty___redArg(v_as_1529_);
lean_dec_ref(v_as_1529_);
v_r_1531_ = lean_box(v_res_1530_);
return v_r_1531_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Subarray_isEmpty(lean_object* v_00_u03b1_1532_, lean_object* v_as_1533_){
_start:
{
lean_object* v_start_1534_; lean_object* v_stop_1535_; uint8_t v___x_1536_; 
v_start_1534_ = lean_ctor_get(v_as_1533_, 1);
v_stop_1535_ = lean_ctor_get(v_as_1533_, 2);
v___x_1536_ = lean_nat_dec_eq(v_start_1534_, v_stop_1535_);
return v___x_1536_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Subarray_isEmpty___boxed(lean_object* v_00_u03b1_1537_, lean_object* v_as_1538_){
_start:
{
uint8_t v_res_1539_; lean_object* v_r_1540_; 
v_res_1539_ = lp_batteries_Subarray_isEmpty(v_00_u03b1_1537_, v_as_1538_);
lean_dec_ref(v_as_1538_);
v_r_1540_ = lean_box(v_res_1539_);
return v_r_1540_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Subarray_contains___redArg___lam__0(lean_object* v_inst_1541_, lean_object* v_a_1542_, lean_object* v_x_1543_){
_start:
{
lean_object* v___x_1544_; uint8_t v___x_1545_; 
v___x_1544_ = lean_apply_2(v_inst_1541_, v_x_1543_, v_a_1542_);
v___x_1545_ = lean_unbox(v___x_1544_);
return v___x_1545_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Subarray_contains___redArg___lam__0___boxed(lean_object* v_inst_1546_, lean_object* v_a_1547_, lean_object* v_x_1548_){
_start:
{
uint8_t v_res_1549_; lean_object* v_r_1550_; 
v_res_1549_ = lp_batteries_Subarray_contains___redArg___lam__0(v_inst_1546_, v_a_1547_, v_x_1548_);
v_r_1550_ = lean_box(v_res_1549_);
return v_r_1550_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Subarray_contains___redArg(lean_object* v_inst_1551_, lean_object* v_as_1552_, lean_object* v_a_1553_){
_start:
{
lean_object* v___x_1554_; lean_object* v_array_1555_; lean_object* v_start_1556_; lean_object* v_stop_1557_; uint8_t v___x_1558_; 
v___x_1554_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v_array_1555_ = lean_ctor_get(v_as_1552_, 0);
lean_inc_ref(v_array_1555_);
v_start_1556_ = lean_ctor_get(v_as_1552_, 1);
lean_inc(v_start_1556_);
v_stop_1557_ = lean_ctor_get(v_as_1552_, 2);
lean_inc(v_stop_1557_);
lean_dec_ref(v_as_1552_);
v___x_1558_ = lean_nat_dec_lt(v_start_1556_, v_stop_1557_);
if (v___x_1558_ == 0)
{
lean_dec(v_stop_1557_);
lean_dec(v_start_1556_);
lean_dec_ref(v_array_1555_);
lean_dec(v_a_1553_);
lean_dec_ref(v_inst_1551_);
return v___x_1558_;
}
else
{
lean_object* v___f_1559_; lean_object* v___y_1561_; lean_object* v___x_1567_; uint8_t v___x_1568_; 
v___f_1559_ = lean_alloc_closure((void*)(lp_batteries_Subarray_contains___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1559_, 0, v_inst_1551_);
lean_closure_set(v___f_1559_, 1, v_a_1553_);
v___x_1567_ = lean_array_get_size(v_array_1555_);
v___x_1568_ = lean_nat_dec_le(v_stop_1557_, v___x_1567_);
if (v___x_1568_ == 0)
{
lean_dec(v_stop_1557_);
v___y_1561_ = v___x_1567_;
goto v___jp_1560_;
}
else
{
v___y_1561_ = v_stop_1557_;
goto v___jp_1560_;
}
v___jp_1560_:
{
uint8_t v___x_1562_; 
v___x_1562_ = lean_nat_dec_lt(v_start_1556_, v___y_1561_);
if (v___x_1562_ == 0)
{
lean_dec(v___y_1561_);
lean_dec_ref(v___f_1559_);
lean_dec(v_start_1556_);
lean_dec_ref(v_array_1555_);
return v___x_1562_;
}
else
{
size_t v___x_1563_; size_t v___x_1564_; lean_object* v___x_1565_; uint8_t v___x_1566_; 
v___x_1563_ = lean_usize_of_nat(v_start_1556_);
lean_dec(v_start_1556_);
v___x_1564_ = lean_usize_of_nat(v___y_1561_);
lean_dec(v___y_1561_);
v___x_1565_ = l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_box(0), lean_box(0), v___x_1554_, v___f_1559_, v_array_1555_, v___x_1563_, v___x_1564_);
v___x_1566_ = lean_unbox(v___x_1565_);
lean_dec(v___x_1565_);
return v___x_1566_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Subarray_contains___redArg___boxed(lean_object* v_inst_1569_, lean_object* v_as_1570_, lean_object* v_a_1571_){
_start:
{
uint8_t v_res_1572_; lean_object* v_r_1573_; 
v_res_1572_ = lp_batteries_Subarray_contains___redArg(v_inst_1569_, v_as_1570_, v_a_1571_);
v_r_1573_ = lean_box(v_res_1572_);
return v_r_1573_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Subarray_contains(lean_object* v_00_u03b1_1574_, lean_object* v_inst_1575_, lean_object* v_as_1576_, lean_object* v_a_1577_){
_start:
{
lean_object* v___x_1578_; lean_object* v_array_1579_; lean_object* v_start_1580_; lean_object* v_stop_1581_; uint8_t v___x_1582_; 
v___x_1578_ = ((lean_object*)(lp_batteries_Array_equalSet___redArg___closed__9));
v_array_1579_ = lean_ctor_get(v_as_1576_, 0);
lean_inc_ref(v_array_1579_);
v_start_1580_ = lean_ctor_get(v_as_1576_, 1);
lean_inc(v_start_1580_);
v_stop_1581_ = lean_ctor_get(v_as_1576_, 2);
lean_inc(v_stop_1581_);
lean_dec_ref(v_as_1576_);
v___x_1582_ = lean_nat_dec_lt(v_start_1580_, v_stop_1581_);
if (v___x_1582_ == 0)
{
lean_dec(v_stop_1581_);
lean_dec(v_start_1580_);
lean_dec_ref(v_array_1579_);
lean_dec(v_a_1577_);
lean_dec_ref(v_inst_1575_);
return v___x_1582_;
}
else
{
lean_object* v___f_1583_; lean_object* v___y_1585_; lean_object* v___x_1591_; uint8_t v___x_1592_; 
v___f_1583_ = lean_alloc_closure((void*)(lp_batteries_Subarray_contains___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1583_, 0, v_inst_1575_);
lean_closure_set(v___f_1583_, 1, v_a_1577_);
v___x_1591_ = lean_array_get_size(v_array_1579_);
v___x_1592_ = lean_nat_dec_le(v_stop_1581_, v___x_1591_);
if (v___x_1592_ == 0)
{
lean_dec(v_stop_1581_);
v___y_1585_ = v___x_1591_;
goto v___jp_1584_;
}
else
{
v___y_1585_ = v_stop_1581_;
goto v___jp_1584_;
}
v___jp_1584_:
{
uint8_t v___x_1586_; 
v___x_1586_ = lean_nat_dec_lt(v_start_1580_, v___y_1585_);
if (v___x_1586_ == 0)
{
lean_dec(v___y_1585_);
lean_dec_ref(v___f_1583_);
lean_dec(v_start_1580_);
lean_dec_ref(v_array_1579_);
return v___x_1586_;
}
else
{
size_t v___x_1587_; size_t v___x_1588_; lean_object* v___x_1589_; uint8_t v___x_1590_; 
v___x_1587_ = lean_usize_of_nat(v_start_1580_);
lean_dec(v_start_1580_);
v___x_1588_ = lean_usize_of_nat(v___y_1585_);
lean_dec(v___y_1585_);
v___x_1589_ = l___private_Init_Data_Array_Basic_0__Array_anyMUnsafe_any(lean_box(0), lean_box(0), v___x_1578_, v___f_1583_, v_array_1579_, v___x_1587_, v___x_1588_);
v___x_1590_ = lean_unbox(v___x_1589_);
lean_dec(v___x_1589_);
return v___x_1590_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Subarray_contains___boxed(lean_object* v_00_u03b1_1593_, lean_object* v_inst_1594_, lean_object* v_as_1595_, lean_object* v_a_1596_){
_start:
{
uint8_t v_res_1597_; lean_object* v_r_1598_; 
v_res_1597_ = lp_batteries_Subarray_contains(v_00_u03b1_1593_, v_inst_1594_, v_as_1595_, v_a_1596_);
v_r_1598_ = lean_box(v_res_1597_);
return v_r_1598_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Subarray_popHead_x3f___redArg(lean_object* v_as_1599_){
_start:
{
lean_object* v_array_1600_; lean_object* v_start_1601_; lean_object* v_stop_1602_; lean_object* v___x_1604_; uint8_t v_isShared_1605_; uint8_t v_isSharedCheck_1616_; 
v_array_1600_ = lean_ctor_get(v_as_1599_, 0);
v_start_1601_ = lean_ctor_get(v_as_1599_, 1);
v_stop_1602_ = lean_ctor_get(v_as_1599_, 2);
v_isSharedCheck_1616_ = !lean_is_exclusive(v_as_1599_);
if (v_isSharedCheck_1616_ == 0)
{
v___x_1604_ = v_as_1599_;
v_isShared_1605_ = v_isSharedCheck_1616_;
goto v_resetjp_1603_;
}
else
{
lean_inc(v_stop_1602_);
lean_inc(v_start_1601_);
lean_inc(v_array_1600_);
lean_dec(v_as_1599_);
v___x_1604_ = lean_box(0);
v_isShared_1605_ = v_isSharedCheck_1616_;
goto v_resetjp_1603_;
}
v_resetjp_1603_:
{
uint8_t v___x_1606_; 
v___x_1606_ = lean_nat_dec_lt(v_start_1601_, v_stop_1602_);
if (v___x_1606_ == 0)
{
lean_object* v___x_1607_; 
lean_del_object(v___x_1604_);
lean_dec(v_stop_1602_);
lean_dec(v_start_1601_);
lean_dec_ref(v_array_1600_);
v___x_1607_ = lean_box(0);
return v___x_1607_;
}
else
{
lean_object* v_head_1608_; lean_object* v___x_1609_; lean_object* v___x_1610_; lean_object* v_tail_1612_; 
v_head_1608_ = lean_array_fget(v_array_1600_, v_start_1601_);
v___x_1609_ = lean_unsigned_to_nat(1u);
v___x_1610_ = lean_nat_add(v_start_1601_, v___x_1609_);
lean_dec(v_start_1601_);
if (v_isShared_1605_ == 0)
{
lean_ctor_set(v___x_1604_, 1, v___x_1610_);
v_tail_1612_ = v___x_1604_;
goto v_reusejp_1611_;
}
else
{
lean_object* v_reuseFailAlloc_1615_; 
v_reuseFailAlloc_1615_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1615_, 0, v_array_1600_);
lean_ctor_set(v_reuseFailAlloc_1615_, 1, v___x_1610_);
lean_ctor_set(v_reuseFailAlloc_1615_, 2, v_stop_1602_);
v_tail_1612_ = v_reuseFailAlloc_1615_;
goto v_reusejp_1611_;
}
v_reusejp_1611_:
{
lean_object* v___x_1613_; lean_object* v___x_1614_; 
v___x_1613_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1613_, 0, v_head_1608_);
lean_ctor_set(v___x_1613_, 1, v_tail_1612_);
v___x_1614_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1614_, 0, v___x_1613_);
return v___x_1614_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Subarray_popHead_x3f(lean_object* v_00_u03b1_1617_, lean_object* v_as_1618_){
_start:
{
lean_object* v___x_1619_; 
v___x_1619_ = lp_batteries_Subarray_popHead_x3f___redArg(v_as_1618_);
return v___x_1619_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Data_UInt(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Data_Array_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_UInt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Data_Array_Basic(uint8_t builtin) {
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
lean_object* initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_UInt(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Data_Array_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_UInt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Data_Array_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Data_Array_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Data_Array_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
