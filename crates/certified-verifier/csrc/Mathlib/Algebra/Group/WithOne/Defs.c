// Lean compiler output
// Module: Mathlib.Algebra.Group.WithOne.Defs
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.DivInvMonoid public import Mathlib.Basic.Nontrivial.Basic public import Mathlib.Data.Option.Basic public import Mathlib.Tactic.Common public import Mathlib.Tactic.Attr.Core
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
lean_object* l_Option_merge(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_nsmulBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_npowBinRecAuto___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_WithZero_instRepr___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "0"};
static const lean_object* lp_mathlib_WithZero_instRepr___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_WithZero_instRepr___redArg___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_WithZero_instRepr___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_WithZero_instRepr___redArg___lam__0___closed__0_value)}};
static const lean_object* lp_mathlib_WithZero_instRepr___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_WithZero_instRepr___redArg___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_WithZero_instRepr___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "↑"};
static const lean_object* lp_mathlib_WithZero_instRepr___redArg___lam__0___closed__2 = (const lean_object*)&lp_mathlib_WithZero_instRepr___redArg___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_WithZero_instRepr___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_WithZero_instRepr___redArg___lam__0___closed__2_value)}};
static const lean_object* lp_mathlib_WithZero_instRepr___redArg___lam__0___closed__3 = (const lean_object*)&lp_mathlib_WithZero_instRepr___redArg___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instRepr___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instRepr___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instRepr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instRepr(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_WithOne_instRepr___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "1"};
static const lean_object* lp_mathlib_WithOne_instRepr___redArg___lam__0___closed__0 = (const lean_object*)&lp_mathlib_WithOne_instRepr___redArg___lam__0___closed__0_value;
static const lean_ctor_object lp_mathlib_WithOne_instRepr___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_WithOne_instRepr___redArg___lam__0___closed__0_value)}};
static const lean_object* lp_mathlib_WithOne_instRepr___redArg___lam__0___closed__1 = (const lean_object*)&lp_mathlib_WithOne_instRepr___redArg___lam__0___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instRepr___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instRepr___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instRepr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instRepr(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__9___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__9___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__11___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__11___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__11(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__13___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__13(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_WithOne_instMonad___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithOne_instMonad___aux__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithOne_instMonad___closed__0 = (const lean_object*)&lp_mathlib_WithOne_instMonad___closed__0_value;
static const lean_closure_object lp_mathlib_WithOne_instMonad___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithOne_instMonad___aux__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithOne_instMonad___closed__1 = (const lean_object*)&lp_mathlib_WithOne_instMonad___closed__1_value;
static const lean_ctor_object lp_mathlib_WithOne_instMonad___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_WithOne_instMonad___closed__0_value),((lean_object*)&lp_mathlib_WithOne_instMonad___closed__1_value)}};
static const lean_object* lp_mathlib_WithOne_instMonad___closed__2 = (const lean_object*)&lp_mathlib_WithOne_instMonad___closed__2_value;
static const lean_closure_object lp_mathlib_WithOne_instMonad___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithOne_instMonad___aux__5, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithOne_instMonad___closed__3 = (const lean_object*)&lp_mathlib_WithOne_instMonad___closed__3_value;
static const lean_closure_object lp_mathlib_WithOne_instMonad___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithOne_instMonad___aux__7, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithOne_instMonad___closed__4 = (const lean_object*)&lp_mathlib_WithOne_instMonad___closed__4_value;
static const lean_closure_object lp_mathlib_WithOne_instMonad___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithOne_instMonad___aux__9___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithOne_instMonad___closed__5 = (const lean_object*)&lp_mathlib_WithOne_instMonad___closed__5_value;
static const lean_closure_object lp_mathlib_WithOne_instMonad___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithOne_instMonad___aux__11___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithOne_instMonad___closed__6 = (const lean_object*)&lp_mathlib_WithOne_instMonad___closed__6_value;
static const lean_ctor_object lp_mathlib_WithOne_instMonad___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_WithOne_instMonad___closed__2_value),((lean_object*)&lp_mathlib_WithOne_instMonad___closed__3_value),((lean_object*)&lp_mathlib_WithOne_instMonad___closed__4_value),((lean_object*)&lp_mathlib_WithOne_instMonad___closed__5_value),((lean_object*)&lp_mathlib_WithOne_instMonad___closed__6_value)}};
static const lean_object* lp_mathlib_WithOne_instMonad___closed__7 = (const lean_object*)&lp_mathlib_WithOne_instMonad___closed__7_value;
static const lean_closure_object lp_mathlib_WithOne_instMonad___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithOne_instMonad___aux__13, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithOne_instMonad___closed__8 = (const lean_object*)&lp_mathlib_WithOne_instMonad___closed__8_value;
static const lean_ctor_object lp_mathlib_WithOne_instMonad___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_WithOne_instMonad___closed__7_value),((lean_object*)&lp_mathlib_WithOne_instMonad___closed__8_value)}};
static const lean_object* lp_mathlib_WithOne_instMonad___closed__9 = (const lean_object*)&lp_mathlib_WithOne_instMonad___closed__9_value;
LEAN_EXPORT const lean_object* lp_mathlib_WithOne_instMonad = (const lean_object*)&lp_mathlib_WithOne_instMonad___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__3___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__5___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__7___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__7(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__9___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__9___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__9(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__9___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__11___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__11___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__11(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__13___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__13(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_WithZero_instMonad___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithZero_instMonad___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithZero_instMonad___closed__0 = (const lean_object*)&lp_mathlib_WithZero_instMonad___closed__0_value;
static const lean_closure_object lp_mathlib_WithZero_instMonad___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithZero_instMonad___aux__1, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithZero_instMonad___closed__1 = (const lean_object*)&lp_mathlib_WithZero_instMonad___closed__1_value;
static const lean_closure_object lp_mathlib_WithZero_instMonad___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithZero_instMonad___aux__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithZero_instMonad___closed__2 = (const lean_object*)&lp_mathlib_WithZero_instMonad___closed__2_value;
static const lean_ctor_object lp_mathlib_WithZero_instMonad___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_WithZero_instMonad___closed__1_value),((lean_object*)&lp_mathlib_WithZero_instMonad___closed__2_value)}};
static const lean_object* lp_mathlib_WithZero_instMonad___closed__3 = (const lean_object*)&lp_mathlib_WithZero_instMonad___closed__3_value;
static const lean_closure_object lp_mathlib_WithZero_instMonad___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithZero_instMonad___aux__7, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithZero_instMonad___closed__4 = (const lean_object*)&lp_mathlib_WithZero_instMonad___closed__4_value;
static const lean_closure_object lp_mathlib_WithZero_instMonad___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithZero_instMonad___aux__9___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithZero_instMonad___closed__5 = (const lean_object*)&lp_mathlib_WithZero_instMonad___closed__5_value;
static const lean_closure_object lp_mathlib_WithZero_instMonad___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithZero_instMonad___aux__11___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithZero_instMonad___closed__6 = (const lean_object*)&lp_mathlib_WithZero_instMonad___closed__6_value;
static const lean_ctor_object lp_mathlib_WithZero_instMonad___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_WithZero_instMonad___closed__3_value),((lean_object*)&lp_mathlib_WithZero_instMonad___closed__0_value),((lean_object*)&lp_mathlib_WithZero_instMonad___closed__4_value),((lean_object*)&lp_mathlib_WithZero_instMonad___closed__5_value),((lean_object*)&lp_mathlib_WithZero_instMonad___closed__6_value)}};
static const lean_object* lp_mathlib_WithZero_instMonad___closed__7 = (const lean_object*)&lp_mathlib_WithZero_instMonad___closed__7_value;
static const lean_closure_object lp_mathlib_WithZero_instMonad___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithZero_instMonad___aux__13, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithZero_instMonad___closed__8 = (const lean_object*)&lp_mathlib_WithZero_instMonad___closed__8_value;
static const lean_ctor_object lp_mathlib_WithZero_instMonad___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_WithZero_instMonad___closed__7_value),((lean_object*)&lp_mathlib_WithZero_instMonad___closed__8_value)}};
static const lean_object* lp_mathlib_WithZero_instMonad___closed__9 = (const lean_object*)&lp_mathlib_WithZero_instMonad___closed__9_value;
LEAN_EXPORT const lean_object* lp_mathlib_WithZero_instMonad = (const lean_object*)&lp_mathlib_WithZero_instMonad___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instOne(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instZero(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAdd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instInv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instInv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instInv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instNeg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instInvOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instInvOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instNegZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instNegZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_inhabited(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_inhabited(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_coe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_coe(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_coe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_coe(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_WithOne_instCoeTC___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithOne_coe, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_WithOne_instCoeTC___closed__0 = (const lean_object*)&lp_mathlib_WithOne_instCoeTC___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instCoeTC(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instCoeTC___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_WithZero_instCoeTC___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithZero_instCoeTC___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithZero_instCoeTC___closed__0 = (const lean_object*)&lp_mathlib_WithZero_instCoeTC___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instCoeTC(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_recZeroCoe___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_recZeroCoe___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_recZeroCoe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_recZeroCoe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_recOneCoe___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_recOneCoe___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_recOneCoe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_recOneCoe___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unone___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unone___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unone(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unone___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzero_match__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzero_match__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzero___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzero___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_WithOne_Defs_0__WithOne_unone_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_WithOne_Defs_0__WithOne_unone_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_WithOne_Defs_0__WithZero_unzero_match__1_splitter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_WithOne_Defs_0__WithZero_unzero_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMulOneClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddZeroClass(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unoneD___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unoneD___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_WithOne_unoneD___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithOne_unoneD___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithOne_unoneD___redArg___closed__0 = (const lean_object*)&lp_mathlib_WithOne_unoneD___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unoneD___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unoneD___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unoneD(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unoneD___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzeroD___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzeroD___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzeroD(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzeroD___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instRepr___redArg___lam__0(lean_object* v_inst_7_, lean_object* v_o_8_, lean_object* v_x_9_){
_start:
{
if (lean_obj_tag(v_o_8_) == 0)
{
lean_object* v___x_10_; 
lean_dec_ref(v_inst_7_);
v___x_10_ = ((lean_object*)(lp_mathlib_WithZero_instRepr___redArg___lam__0___closed__1));
return v___x_10_;
}
else
{
lean_object* v_val_11_; lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; 
v_val_11_ = lean_ctor_get(v_o_8_, 0);
lean_inc(v_val_11_);
lean_dec_ref_known(v_o_8_, 1);
v___x_12_ = ((lean_object*)(lp_mathlib_WithZero_instRepr___redArg___lam__0___closed__3));
v___x_13_ = lean_unsigned_to_nat(0u);
v___x_14_ = lean_apply_2(v_inst_7_, v_val_11_, v___x_13_);
v___x_15_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_15_, 0, v___x_12_);
lean_ctor_set(v___x_15_, 1, v___x_14_);
return v___x_15_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instRepr___redArg___lam__0___boxed(lean_object* v_inst_16_, lean_object* v_o_17_, lean_object* v_x_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib_WithZero_instRepr___redArg___lam__0(v_inst_16_, v_o_17_, v_x_18_);
lean_dec(v_x_18_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instRepr___redArg(lean_object* v_inst_20_){
_start:
{
lean_object* v___f_21_; 
v___f_21_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_instRepr___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_21_, 0, v_inst_20_);
return v___f_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instRepr(lean_object* v_00_u03b1_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v___f_24_; 
v___f_24_ = lean_alloc_closure((void*)(lp_mathlib_WithZero_instRepr___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_24_, 0, v_inst_23_);
return v___f_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instRepr___redArg___lam__0(lean_object* v_inst_28_, lean_object* v_o_29_, lean_object* v_x_30_){
_start:
{
if (lean_obj_tag(v_o_29_) == 0)
{
lean_object* v___x_31_; 
lean_dec_ref(v_inst_28_);
v___x_31_ = ((lean_object*)(lp_mathlib_WithOne_instRepr___redArg___lam__0___closed__1));
return v___x_31_;
}
else
{
lean_object* v_val_32_; lean_object* v___x_33_; lean_object* v___x_34_; lean_object* v___x_35_; lean_object* v___x_36_; 
v_val_32_ = lean_ctor_get(v_o_29_, 0);
lean_inc(v_val_32_);
lean_dec_ref_known(v_o_29_, 1);
v___x_33_ = ((lean_object*)(lp_mathlib_WithZero_instRepr___redArg___lam__0___closed__3));
v___x_34_ = lean_unsigned_to_nat(0u);
v___x_35_ = lean_apply_2(v_inst_28_, v_val_32_, v___x_34_);
v___x_36_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_36_, 0, v___x_33_);
lean_ctor_set(v___x_36_, 1, v___x_35_);
return v___x_36_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instRepr___redArg___lam__0___boxed(lean_object* v_inst_37_, lean_object* v_o_38_, lean_object* v_x_39_){
_start:
{
lean_object* v_res_40_; 
v_res_40_ = lp_mathlib_WithOne_instRepr___redArg___lam__0(v_inst_37_, v_o_38_, v_x_39_);
lean_dec(v_x_39_);
return v_res_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instRepr___redArg(lean_object* v_inst_41_){
_start:
{
lean_object* v___f_42_; 
v___f_42_ = lean_alloc_closure((void*)(lp_mathlib_WithOne_instRepr___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_42_, 0, v_inst_41_);
return v___f_42_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instRepr(lean_object* v_00_u03b1_43_, lean_object* v_inst_44_){
_start:
{
lean_object* v___f_45_; 
v___f_45_ = lean_alloc_closure((void*)(lp_mathlib_WithOne_instRepr___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_45_, 0, v_inst_44_);
return v___f_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__1___redArg(lean_object* v_f_46_, lean_object* v_a_47_){
_start:
{
if (lean_obj_tag(v_a_47_) == 0)
{
lean_object* v___x_48_; 
lean_dec(v_f_46_);
v___x_48_ = lean_box(0);
return v___x_48_;
}
else
{
lean_object* v_val_49_; lean_object* v___x_51_; uint8_t v_isShared_52_; uint8_t v_isSharedCheck_57_; 
v_val_49_ = lean_ctor_get(v_a_47_, 0);
v_isSharedCheck_57_ = !lean_is_exclusive(v_a_47_);
if (v_isSharedCheck_57_ == 0)
{
v___x_51_ = v_a_47_;
v_isShared_52_ = v_isSharedCheck_57_;
goto v_resetjp_50_;
}
else
{
lean_inc(v_val_49_);
lean_dec(v_a_47_);
v___x_51_ = lean_box(0);
v_isShared_52_ = v_isSharedCheck_57_;
goto v_resetjp_50_;
}
v_resetjp_50_:
{
lean_object* v___x_53_; lean_object* v___x_55_; 
v___x_53_ = lean_apply_1(v_f_46_, v_val_49_);
if (v_isShared_52_ == 0)
{
lean_ctor_set(v___x_51_, 0, v___x_53_);
v___x_55_ = v___x_51_;
goto v_reusejp_54_;
}
else
{
lean_object* v_reuseFailAlloc_56_; 
v_reuseFailAlloc_56_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_56_, 0, v___x_53_);
v___x_55_ = v_reuseFailAlloc_56_;
goto v_reusejp_54_;
}
v_reusejp_54_:
{
return v___x_55_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__1(lean_object* v_00_u03b1_58_, lean_object* v_00_u03b2_59_, lean_object* v_f_60_, lean_object* v_a_61_){
_start:
{
if (lean_obj_tag(v_a_61_) == 0)
{
lean_object* v___x_62_; 
lean_dec(v_f_60_);
v___x_62_ = lean_box(0);
return v___x_62_;
}
else
{
lean_object* v_val_63_; lean_object* v___x_65_; uint8_t v_isShared_66_; uint8_t v_isSharedCheck_71_; 
v_val_63_ = lean_ctor_get(v_a_61_, 0);
v_isSharedCheck_71_ = !lean_is_exclusive(v_a_61_);
if (v_isSharedCheck_71_ == 0)
{
v___x_65_ = v_a_61_;
v_isShared_66_ = v_isSharedCheck_71_;
goto v_resetjp_64_;
}
else
{
lean_inc(v_val_63_);
lean_dec(v_a_61_);
v___x_65_ = lean_box(0);
v_isShared_66_ = v_isSharedCheck_71_;
goto v_resetjp_64_;
}
v_resetjp_64_:
{
lean_object* v___x_67_; lean_object* v___x_69_; 
v___x_67_ = lean_apply_1(v_f_60_, v_val_63_);
if (v_isShared_66_ == 0)
{
lean_ctor_set(v___x_65_, 0, v___x_67_);
v___x_69_ = v___x_65_;
goto v_reusejp_68_;
}
else
{
lean_object* v_reuseFailAlloc_70_; 
v_reuseFailAlloc_70_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_70_, 0, v___x_67_);
v___x_69_ = v_reuseFailAlloc_70_;
goto v_reusejp_68_;
}
v_reusejp_68_:
{
return v___x_69_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__3___redArg(lean_object* v_a_72_, lean_object* v_a_73_){
_start:
{
if (lean_obj_tag(v_a_73_) == 0)
{
lean_object* v___x_74_; 
lean_dec(v_a_72_);
v___x_74_ = lean_box(0);
return v___x_74_;
}
else
{
lean_object* v___x_76_; uint8_t v_isShared_77_; uint8_t v_isSharedCheck_81_; 
v_isSharedCheck_81_ = !lean_is_exclusive(v_a_73_);
if (v_isSharedCheck_81_ == 0)
{
lean_object* v_unused_82_; 
v_unused_82_ = lean_ctor_get(v_a_73_, 0);
lean_dec(v_unused_82_);
v___x_76_ = v_a_73_;
v_isShared_77_ = v_isSharedCheck_81_;
goto v_resetjp_75_;
}
else
{
lean_dec(v_a_73_);
v___x_76_ = lean_box(0);
v_isShared_77_ = v_isSharedCheck_81_;
goto v_resetjp_75_;
}
v_resetjp_75_:
{
lean_object* v___x_79_; 
if (v_isShared_77_ == 0)
{
lean_ctor_set(v___x_76_, 0, v_a_72_);
v___x_79_ = v___x_76_;
goto v_reusejp_78_;
}
else
{
lean_object* v_reuseFailAlloc_80_; 
v_reuseFailAlloc_80_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_80_, 0, v_a_72_);
v___x_79_ = v_reuseFailAlloc_80_;
goto v_reusejp_78_;
}
v_reusejp_78_:
{
return v___x_79_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__3(lean_object* v_00_u03b1_83_, lean_object* v_00_u03b2_84_, lean_object* v_a_85_, lean_object* v_a_86_){
_start:
{
if (lean_obj_tag(v_a_86_) == 0)
{
lean_object* v___x_87_; 
lean_dec(v_a_85_);
v___x_87_ = lean_box(0);
return v___x_87_;
}
else
{
lean_object* v___x_89_; uint8_t v_isShared_90_; uint8_t v_isSharedCheck_94_; 
v_isSharedCheck_94_ = !lean_is_exclusive(v_a_86_);
if (v_isSharedCheck_94_ == 0)
{
lean_object* v_unused_95_; 
v_unused_95_ = lean_ctor_get(v_a_86_, 0);
lean_dec(v_unused_95_);
v___x_89_ = v_a_86_;
v_isShared_90_ = v_isSharedCheck_94_;
goto v_resetjp_88_;
}
else
{
lean_dec(v_a_86_);
v___x_89_ = lean_box(0);
v_isShared_90_ = v_isSharedCheck_94_;
goto v_resetjp_88_;
}
v_resetjp_88_:
{
lean_object* v___x_92_; 
if (v_isShared_90_ == 0)
{
lean_ctor_set(v___x_89_, 0, v_a_85_);
v___x_92_ = v___x_89_;
goto v_reusejp_91_;
}
else
{
lean_object* v_reuseFailAlloc_93_; 
v_reuseFailAlloc_93_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_93_, 0, v_a_85_);
v___x_92_ = v_reuseFailAlloc_93_;
goto v_reusejp_91_;
}
v_reusejp_91_:
{
return v___x_92_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__5___redArg(lean_object* v_val_96_){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_97_, 0, v_val_96_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__5(lean_object* v_00_u03b1_98_, lean_object* v_val_99_){
_start:
{
lean_object* v___x_100_; 
v___x_100_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_100_, 0, v_val_99_);
return v___x_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__7___redArg(lean_object* v_f_101_, lean_object* v_x_102_){
_start:
{
if (lean_obj_tag(v_f_101_) == 0)
{
lean_object* v___x_103_; 
lean_dec_ref(v_x_102_);
v___x_103_ = lean_box(0);
return v___x_103_;
}
else
{
lean_object* v_val_104_; lean_object* v___x_105_; lean_object* v___x_106_; 
v_val_104_ = lean_ctor_get(v_f_101_, 0);
lean_inc(v_val_104_);
lean_dec_ref_known(v_f_101_, 1);
v___x_105_ = lean_box(0);
v___x_106_ = lean_apply_1(v_x_102_, v___x_105_);
if (lean_obj_tag(v___x_106_) == 0)
{
lean_object* v___x_107_; 
lean_dec(v_val_104_);
v___x_107_ = lean_box(0);
return v___x_107_;
}
else
{
lean_object* v_val_108_; lean_object* v___x_110_; uint8_t v_isShared_111_; uint8_t v_isSharedCheck_116_; 
v_val_108_ = lean_ctor_get(v___x_106_, 0);
v_isSharedCheck_116_ = !lean_is_exclusive(v___x_106_);
if (v_isSharedCheck_116_ == 0)
{
v___x_110_ = v___x_106_;
v_isShared_111_ = v_isSharedCheck_116_;
goto v_resetjp_109_;
}
else
{
lean_inc(v_val_108_);
lean_dec(v___x_106_);
v___x_110_ = lean_box(0);
v_isShared_111_ = v_isSharedCheck_116_;
goto v_resetjp_109_;
}
v_resetjp_109_:
{
lean_object* v___x_112_; lean_object* v___x_114_; 
v___x_112_ = lean_apply_1(v_val_104_, v_val_108_);
if (v_isShared_111_ == 0)
{
lean_ctor_set(v___x_110_, 0, v___x_112_);
v___x_114_ = v___x_110_;
goto v_reusejp_113_;
}
else
{
lean_object* v_reuseFailAlloc_115_; 
v_reuseFailAlloc_115_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_115_, 0, v___x_112_);
v___x_114_ = v_reuseFailAlloc_115_;
goto v_reusejp_113_;
}
v_reusejp_113_:
{
return v___x_114_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__7(lean_object* v_00_u03b1_117_, lean_object* v_00_u03b2_118_, lean_object* v_f_119_, lean_object* v_x_120_){
_start:
{
if (lean_obj_tag(v_f_119_) == 0)
{
lean_object* v___x_121_; 
lean_dec_ref(v_x_120_);
v___x_121_ = lean_box(0);
return v___x_121_;
}
else
{
lean_object* v_val_122_; lean_object* v___x_123_; lean_object* v___x_124_; 
v_val_122_ = lean_ctor_get(v_f_119_, 0);
lean_inc(v_val_122_);
lean_dec_ref_known(v_f_119_, 1);
v___x_123_ = lean_box(0);
v___x_124_ = lean_apply_1(v_x_120_, v___x_123_);
if (lean_obj_tag(v___x_124_) == 0)
{
lean_object* v___x_125_; 
lean_dec(v_val_122_);
v___x_125_ = lean_box(0);
return v___x_125_;
}
else
{
lean_object* v_val_126_; lean_object* v___x_128_; uint8_t v_isShared_129_; uint8_t v_isSharedCheck_134_; 
v_val_126_ = lean_ctor_get(v___x_124_, 0);
v_isSharedCheck_134_ = !lean_is_exclusive(v___x_124_);
if (v_isSharedCheck_134_ == 0)
{
v___x_128_ = v___x_124_;
v_isShared_129_ = v_isSharedCheck_134_;
goto v_resetjp_127_;
}
else
{
lean_inc(v_val_126_);
lean_dec(v___x_124_);
v___x_128_ = lean_box(0);
v_isShared_129_ = v_isSharedCheck_134_;
goto v_resetjp_127_;
}
v_resetjp_127_:
{
lean_object* v___x_130_; lean_object* v___x_132_; 
v___x_130_ = lean_apply_1(v_val_122_, v_val_126_);
if (v_isShared_129_ == 0)
{
lean_ctor_set(v___x_128_, 0, v___x_130_);
v___x_132_ = v___x_128_;
goto v_reusejp_131_;
}
else
{
lean_object* v_reuseFailAlloc_133_; 
v_reuseFailAlloc_133_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_133_, 0, v___x_130_);
v___x_132_ = v_reuseFailAlloc_133_;
goto v_reusejp_131_;
}
v_reusejp_131_:
{
return v___x_132_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__9___redArg(lean_object* v_x_135_, lean_object* v_y_136_){
_start:
{
if (lean_obj_tag(v_x_135_) == 0)
{
lean_dec_ref(v_y_136_);
return v_x_135_;
}
else
{
lean_object* v___x_137_; lean_object* v___x_138_; 
v___x_137_ = lean_box(0);
v___x_138_ = lean_apply_1(v_y_136_, v___x_137_);
if (lean_obj_tag(v___x_138_) == 0)
{
lean_object* v___x_139_; 
v___x_139_ = lean_box(0);
return v___x_139_;
}
else
{
lean_dec_ref_known(v___x_138_, 1);
lean_inc_ref(v_x_135_);
return v_x_135_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__9___redArg___boxed(lean_object* v_x_140_, lean_object* v_y_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_mathlib_WithOne_instMonad___aux__9___redArg(v_x_140_, v_y_141_);
lean_dec(v_x_140_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__9(lean_object* v_00_u03b1_143_, lean_object* v_00_u03b2_144_, lean_object* v_x_145_, lean_object* v_y_146_){
_start:
{
if (lean_obj_tag(v_x_145_) == 0)
{
lean_dec_ref(v_y_146_);
return v_x_145_;
}
else
{
lean_object* v___x_147_; lean_object* v___x_148_; 
v___x_147_ = lean_box(0);
v___x_148_ = lean_apply_1(v_y_146_, v___x_147_);
if (lean_obj_tag(v___x_148_) == 0)
{
lean_object* v___x_149_; 
v___x_149_ = lean_box(0);
return v___x_149_;
}
else
{
lean_dec_ref_known(v___x_148_, 1);
lean_inc_ref(v_x_145_);
return v_x_145_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__9___boxed(lean_object* v_00_u03b1_150_, lean_object* v_00_u03b2_151_, lean_object* v_x_152_, lean_object* v_y_153_){
_start:
{
lean_object* v_res_154_; 
v_res_154_ = lp_mathlib_WithOne_instMonad___aux__9(v_00_u03b1_150_, v_00_u03b2_151_, v_x_152_, v_y_153_);
lean_dec(v_x_152_);
return v_res_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__11___redArg(lean_object* v_x_155_, lean_object* v_y_156_){
_start:
{
if (lean_obj_tag(v_x_155_) == 0)
{
lean_object* v___x_157_; 
lean_dec_ref(v_y_156_);
v___x_157_ = lean_box(0);
return v___x_157_;
}
else
{
lean_object* v___x_158_; lean_object* v___x_159_; 
v___x_158_ = lean_box(0);
v___x_159_ = lean_apply_1(v_y_156_, v___x_158_);
return v___x_159_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__11___redArg___boxed(lean_object* v_x_160_, lean_object* v_y_161_){
_start:
{
lean_object* v_res_162_; 
v_res_162_ = lp_mathlib_WithOne_instMonad___aux__11___redArg(v_x_160_, v_y_161_);
lean_dec(v_x_160_);
return v_res_162_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__11(lean_object* v_00_u03b1_163_, lean_object* v_00_u03b2_164_, lean_object* v_x_165_, lean_object* v_y_166_){
_start:
{
if (lean_obj_tag(v_x_165_) == 0)
{
lean_object* v___x_167_; 
lean_dec_ref(v_y_166_);
v___x_167_ = lean_box(0);
return v___x_167_;
}
else
{
lean_object* v___x_168_; lean_object* v___x_169_; 
v___x_168_ = lean_box(0);
v___x_169_ = lean_apply_1(v_y_166_, v___x_168_);
return v___x_169_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__11___boxed(lean_object* v_00_u03b1_170_, lean_object* v_00_u03b2_171_, lean_object* v_x_172_, lean_object* v_y_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_WithOne_instMonad___aux__11(v_00_u03b1_170_, v_00_u03b2_171_, v_x_172_, v_y_173_);
lean_dec(v_x_172_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__13___redArg(lean_object* v_a_175_, lean_object* v_a_176_){
_start:
{
if (lean_obj_tag(v_a_175_) == 0)
{
lean_object* v___x_177_; 
lean_dec_ref(v_a_176_);
v___x_177_ = lean_box(0);
return v___x_177_;
}
else
{
lean_object* v_val_178_; lean_object* v___x_179_; 
v_val_178_ = lean_ctor_get(v_a_175_, 0);
lean_inc(v_val_178_);
lean_dec_ref_known(v_a_175_, 1);
v___x_179_ = lean_apply_1(v_a_176_, v_val_178_);
return v___x_179_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonad___aux__13(lean_object* v_00_u03b1_180_, lean_object* v_00_u03b2_181_, lean_object* v_a_182_, lean_object* v_a_183_){
_start:
{
if (lean_obj_tag(v_a_182_) == 0)
{
lean_object* v___x_184_; 
lean_dec_ref(v_a_183_);
v___x_184_ = lean_box(0);
return v___x_184_;
}
else
{
lean_object* v_val_185_; lean_object* v___x_186_; 
v_val_185_ = lean_ctor_get(v_a_182_, 0);
lean_inc(v_val_185_);
lean_dec_ref_known(v_a_182_, 1);
v___x_186_ = lean_apply_1(v_a_183_, v_val_185_);
return v___x_186_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__1___redArg(lean_object* v_f_207_, lean_object* v_a_208_){
_start:
{
if (lean_obj_tag(v_a_208_) == 0)
{
lean_object* v___x_209_; 
lean_dec(v_f_207_);
v___x_209_ = lean_box(0);
return v___x_209_;
}
else
{
lean_object* v_val_210_; lean_object* v___x_212_; uint8_t v_isShared_213_; uint8_t v_isSharedCheck_218_; 
v_val_210_ = lean_ctor_get(v_a_208_, 0);
v_isSharedCheck_218_ = !lean_is_exclusive(v_a_208_);
if (v_isSharedCheck_218_ == 0)
{
v___x_212_ = v_a_208_;
v_isShared_213_ = v_isSharedCheck_218_;
goto v_resetjp_211_;
}
else
{
lean_inc(v_val_210_);
lean_dec(v_a_208_);
v___x_212_ = lean_box(0);
v_isShared_213_ = v_isSharedCheck_218_;
goto v_resetjp_211_;
}
v_resetjp_211_:
{
lean_object* v___x_214_; lean_object* v___x_216_; 
v___x_214_ = lean_apply_1(v_f_207_, v_val_210_);
if (v_isShared_213_ == 0)
{
lean_ctor_set(v___x_212_, 0, v___x_214_);
v___x_216_ = v___x_212_;
goto v_reusejp_215_;
}
else
{
lean_object* v_reuseFailAlloc_217_; 
v_reuseFailAlloc_217_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_217_, 0, v___x_214_);
v___x_216_ = v_reuseFailAlloc_217_;
goto v_reusejp_215_;
}
v_reusejp_215_:
{
return v___x_216_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__1(lean_object* v_00_u03b1_219_, lean_object* v_00_u03b2_220_, lean_object* v_f_221_, lean_object* v_a_222_){
_start:
{
lean_object* v___x_223_; 
v___x_223_ = lp_mathlib_WithZero_instMonad___aux__1___redArg(v_f_221_, v_a_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__3___redArg(lean_object* v_a_224_, lean_object* v_a_225_){
_start:
{
if (lean_obj_tag(v_a_225_) == 0)
{
lean_object* v___x_226_; 
lean_dec(v_a_224_);
v___x_226_ = lean_box(0);
return v___x_226_;
}
else
{
lean_object* v___x_228_; uint8_t v_isShared_229_; uint8_t v_isSharedCheck_233_; 
v_isSharedCheck_233_ = !lean_is_exclusive(v_a_225_);
if (v_isSharedCheck_233_ == 0)
{
lean_object* v_unused_234_; 
v_unused_234_ = lean_ctor_get(v_a_225_, 0);
lean_dec(v_unused_234_);
v___x_228_ = v_a_225_;
v_isShared_229_ = v_isSharedCheck_233_;
goto v_resetjp_227_;
}
else
{
lean_dec(v_a_225_);
v___x_228_ = lean_box(0);
v_isShared_229_ = v_isSharedCheck_233_;
goto v_resetjp_227_;
}
v_resetjp_227_:
{
lean_object* v___x_231_; 
if (v_isShared_229_ == 0)
{
lean_ctor_set(v___x_228_, 0, v_a_224_);
v___x_231_ = v___x_228_;
goto v_reusejp_230_;
}
else
{
lean_object* v_reuseFailAlloc_232_; 
v_reuseFailAlloc_232_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_232_, 0, v_a_224_);
v___x_231_ = v_reuseFailAlloc_232_;
goto v_reusejp_230_;
}
v_reusejp_230_:
{
return v___x_231_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__3(lean_object* v_00_u03b1_235_, lean_object* v_00_u03b2_236_, lean_object* v_a_237_, lean_object* v_a_238_){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lp_mathlib_WithZero_instMonad___aux__3___redArg(v_a_237_, v_a_238_);
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__5___redArg(lean_object* v_val_240_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_241_, 0, v_val_240_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__5(lean_object* v_00_u03b1_242_, lean_object* v_val_243_){
_start:
{
lean_object* v___x_244_; 
v___x_244_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_244_, 0, v_val_243_);
return v___x_244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__7___redArg(lean_object* v_f_245_, lean_object* v_x_246_){
_start:
{
if (lean_obj_tag(v_f_245_) == 0)
{
lean_object* v___x_247_; 
lean_dec_ref(v_x_246_);
v___x_247_ = lean_box(0);
return v___x_247_;
}
else
{
lean_object* v_val_248_; lean_object* v___x_249_; lean_object* v___x_250_; 
v_val_248_ = lean_ctor_get(v_f_245_, 0);
lean_inc(v_val_248_);
lean_dec_ref_known(v_f_245_, 1);
v___x_249_ = lean_box(0);
v___x_250_ = lean_apply_1(v_x_246_, v___x_249_);
if (lean_obj_tag(v___x_250_) == 0)
{
lean_object* v___x_251_; 
lean_dec(v_val_248_);
v___x_251_ = lean_box(0);
return v___x_251_;
}
else
{
lean_object* v_val_252_; lean_object* v___x_254_; uint8_t v_isShared_255_; uint8_t v_isSharedCheck_260_; 
v_val_252_ = lean_ctor_get(v___x_250_, 0);
v_isSharedCheck_260_ = !lean_is_exclusive(v___x_250_);
if (v_isSharedCheck_260_ == 0)
{
v___x_254_ = v___x_250_;
v_isShared_255_ = v_isSharedCheck_260_;
goto v_resetjp_253_;
}
else
{
lean_inc(v_val_252_);
lean_dec(v___x_250_);
v___x_254_ = lean_box(0);
v_isShared_255_ = v_isSharedCheck_260_;
goto v_resetjp_253_;
}
v_resetjp_253_:
{
lean_object* v___x_256_; lean_object* v___x_258_; 
v___x_256_ = lean_apply_1(v_val_248_, v_val_252_);
if (v_isShared_255_ == 0)
{
lean_ctor_set(v___x_254_, 0, v___x_256_);
v___x_258_ = v___x_254_;
goto v_reusejp_257_;
}
else
{
lean_object* v_reuseFailAlloc_259_; 
v_reuseFailAlloc_259_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_259_, 0, v___x_256_);
v___x_258_ = v_reuseFailAlloc_259_;
goto v_reusejp_257_;
}
v_reusejp_257_:
{
return v___x_258_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__7(lean_object* v_00_u03b1_261_, lean_object* v_00_u03b2_262_, lean_object* v_f_263_, lean_object* v_x_264_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lp_mathlib_WithZero_instMonad___aux__7___redArg(v_f_263_, v_x_264_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__9___redArg(lean_object* v_x_266_, lean_object* v_y_267_){
_start:
{
if (lean_obj_tag(v_x_266_) == 0)
{
lean_dec_ref(v_y_267_);
return v_x_266_;
}
else
{
lean_object* v___x_268_; lean_object* v___x_269_; 
v___x_268_ = lean_box(0);
v___x_269_ = lean_apply_1(v_y_267_, v___x_268_);
if (lean_obj_tag(v___x_269_) == 0)
{
lean_object* v___x_270_; 
v___x_270_ = lean_box(0);
return v___x_270_;
}
else
{
lean_dec_ref_known(v___x_269_, 1);
lean_inc_ref(v_x_266_);
return v_x_266_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__9___redArg___boxed(lean_object* v_x_271_, lean_object* v_y_272_){
_start:
{
lean_object* v_res_273_; 
v_res_273_ = lp_mathlib_WithZero_instMonad___aux__9___redArg(v_x_271_, v_y_272_);
lean_dec(v_x_271_);
return v_res_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__9(lean_object* v_00_u03b1_274_, lean_object* v_00_u03b2_275_, lean_object* v_x_276_, lean_object* v_y_277_){
_start:
{
lean_object* v___x_278_; 
v___x_278_ = lp_mathlib_WithZero_instMonad___aux__9___redArg(v_x_276_, v_y_277_);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__9___boxed(lean_object* v_00_u03b1_279_, lean_object* v_00_u03b2_280_, lean_object* v_x_281_, lean_object* v_y_282_){
_start:
{
lean_object* v_res_283_; 
v_res_283_ = lp_mathlib_WithZero_instMonad___aux__9(v_00_u03b1_279_, v_00_u03b2_280_, v_x_281_, v_y_282_);
lean_dec(v_x_281_);
return v_res_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__11___redArg(lean_object* v_x_284_, lean_object* v_y_285_){
_start:
{
if (lean_obj_tag(v_x_284_) == 0)
{
lean_object* v___x_286_; 
lean_dec_ref(v_y_285_);
v___x_286_ = lean_box(0);
return v___x_286_;
}
else
{
lean_object* v___x_287_; lean_object* v___x_288_; 
v___x_287_ = lean_box(0);
v___x_288_ = lean_apply_1(v_y_285_, v___x_287_);
return v___x_288_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__11___redArg___boxed(lean_object* v_x_289_, lean_object* v_y_290_){
_start:
{
lean_object* v_res_291_; 
v_res_291_ = lp_mathlib_WithZero_instMonad___aux__11___redArg(v_x_289_, v_y_290_);
lean_dec(v_x_289_);
return v_res_291_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__11(lean_object* v_00_u03b1_292_, lean_object* v_00_u03b2_293_, lean_object* v_x_294_, lean_object* v_y_295_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lp_mathlib_WithZero_instMonad___aux__11___redArg(v_x_294_, v_y_295_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__11___boxed(lean_object* v_00_u03b1_297_, lean_object* v_00_u03b2_298_, lean_object* v_x_299_, lean_object* v_y_300_){
_start:
{
lean_object* v_res_301_; 
v_res_301_ = lp_mathlib_WithZero_instMonad___aux__11(v_00_u03b1_297_, v_00_u03b2_298_, v_x_299_, v_y_300_);
lean_dec(v_x_299_);
return v_res_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__13___redArg(lean_object* v_a_302_, lean_object* v_a_303_){
_start:
{
if (lean_obj_tag(v_a_302_) == 0)
{
lean_object* v___x_304_; 
lean_dec_ref(v_a_303_);
v___x_304_ = lean_box(0);
return v___x_304_;
}
else
{
lean_object* v_val_305_; lean_object* v___x_306_; 
v_val_305_ = lean_ctor_get(v_a_302_, 0);
lean_inc(v_val_305_);
lean_dec_ref_known(v_a_302_, 1);
v___x_306_ = lean_apply_1(v_a_303_, v_val_305_);
return v___x_306_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___aux__13(lean_object* v_00_u03b1_307_, lean_object* v_00_u03b2_308_, lean_object* v_a_309_, lean_object* v_a_310_){
_start:
{
lean_object* v___x_311_; 
v___x_311_ = lp_mathlib_WithZero_instMonad___aux__13___redArg(v_a_309_, v_a_310_);
return v___x_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instMonad___lam__0(lean_object* v___y_312_, lean_object* v___y_313_){
_start:
{
lean_object* v___x_314_; 
v___x_314_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_314_, 0, v___y_313_);
return v___x_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instOne(lean_object* v_00_u03b1_335_){
_start:
{
lean_object* v___x_336_; 
v___x_336_ = lean_box(0);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instZero(lean_object* v_00_u03b1_337_){
_start:
{
lean_object* v___x_338_; 
v___x_338_ = lean_box(0);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMul___redArg___lam__0(lean_object* v_inst_339_, lean_object* v_x1_340_, lean_object* v_x2_341_){
_start:
{
lean_object* v___x_342_; 
v___x_342_ = lean_apply_2(v_inst_339_, v_x1_340_, v_x2_341_);
return v___x_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMul___redArg(lean_object* v_inst_343_){
_start:
{
lean_object* v___f_344_; lean_object* v___x_345_; 
v___f_344_ = lean_alloc_closure((void*)(lp_mathlib_WithOne_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_344_, 0, v_inst_343_);
v___x_345_ = lean_alloc_closure((void*)(l_Option_merge), 4, 2);
lean_closure_set(v___x_345_, 0, lean_box(0));
lean_closure_set(v___x_345_, 1, v___f_344_);
return v___x_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMul(lean_object* v_00_u03b1_346_, lean_object* v_inst_347_){
_start:
{
lean_object* v___x_348_; 
v___x_348_ = lp_mathlib_WithOne_instMul___redArg(v_inst_347_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAdd___redArg(lean_object* v_inst_349_){
_start:
{
lean_object* v___f_350_; lean_object* v___x_351_; 
v___f_350_ = lean_alloc_closure((void*)(lp_mathlib_WithOne_instMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_350_, 0, v_inst_349_);
v___x_351_ = lean_alloc_closure((void*)(l_Option_merge), 4, 2);
lean_closure_set(v___x_351_, 0, lean_box(0));
lean_closure_set(v___x_351_, 1, v___f_350_);
return v___x_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAdd(lean_object* v_00_u03b1_352_, lean_object* v_inst_353_){
_start:
{
lean_object* v___x_354_; 
v___x_354_ = lp_mathlib_WithZero_instAdd___redArg(v_inst_353_);
return v___x_354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instInv___redArg___lam__0(lean_object* v_inst_355_, lean_object* v_a_356_){
_start:
{
if (lean_obj_tag(v_a_356_) == 0)
{
lean_dec(v_inst_355_);
return v_a_356_;
}
else
{
lean_object* v_val_357_; lean_object* v___x_359_; uint8_t v_isShared_360_; uint8_t v_isSharedCheck_365_; 
v_val_357_ = lean_ctor_get(v_a_356_, 0);
v_isSharedCheck_365_ = !lean_is_exclusive(v_a_356_);
if (v_isSharedCheck_365_ == 0)
{
v___x_359_ = v_a_356_;
v_isShared_360_ = v_isSharedCheck_365_;
goto v_resetjp_358_;
}
else
{
lean_inc(v_val_357_);
lean_dec(v_a_356_);
v___x_359_ = lean_box(0);
v_isShared_360_ = v_isSharedCheck_365_;
goto v_resetjp_358_;
}
v_resetjp_358_:
{
lean_object* v___x_361_; lean_object* v___x_363_; 
v___x_361_ = lean_apply_1(v_inst_355_, v_val_357_);
if (v_isShared_360_ == 0)
{
lean_ctor_set(v___x_359_, 0, v___x_361_);
v___x_363_ = v___x_359_;
goto v_reusejp_362_;
}
else
{
lean_object* v_reuseFailAlloc_364_; 
v_reuseFailAlloc_364_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_364_, 0, v___x_361_);
v___x_363_ = v_reuseFailAlloc_364_;
goto v_reusejp_362_;
}
v_reusejp_362_:
{
return v___x_363_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instInv___redArg(lean_object* v_inst_366_){
_start:
{
lean_object* v___f_367_; 
v___f_367_ = lean_alloc_closure((void*)(lp_mathlib_WithOne_instInv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_367_, 0, v_inst_366_);
return v___f_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instInv(lean_object* v_00_u03b1_368_, lean_object* v_inst_369_){
_start:
{
lean_object* v___f_370_; 
v___f_370_ = lean_alloc_closure((void*)(lp_mathlib_WithOne_instInv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_370_, 0, v_inst_369_);
return v___f_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instNeg___redArg(lean_object* v_inst_371_){
_start:
{
lean_object* v___f_372_; 
v___f_372_ = lean_alloc_closure((void*)(lp_mathlib_WithOne_instInv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_372_, 0, v_inst_371_);
return v___f_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instNeg(lean_object* v_00_u03b1_373_, lean_object* v_inst_374_){
_start:
{
lean_object* v___f_375_; 
v___f_375_ = lean_alloc_closure((void*)(lp_mathlib_WithOne_instInv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_375_, 0, v_inst_374_);
return v___f_375_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instInvOneClass___redArg(lean_object* v_inst_376_){
_start:
{
lean_object* v___x_377_; lean_object* v___f_378_; lean_object* v___x_379_; 
v___x_377_ = lean_box(0);
v___f_378_ = lean_alloc_closure((void*)(lp_mathlib_WithOne_instInv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_378_, 0, v_inst_376_);
v___x_379_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_379_, 0, v___x_377_);
lean_ctor_set(v___x_379_, 1, v___f_378_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instInvOneClass(lean_object* v_00_u03b1_380_, lean_object* v_inst_381_){
_start:
{
lean_object* v___x_382_; 
v___x_382_ = lp_mathlib_WithOne_instInvOneClass___redArg(v_inst_381_);
return v___x_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instNegZeroClass___redArg(lean_object* v_inst_383_){
_start:
{
lean_object* v___x_384_; lean_object* v___f_385_; lean_object* v___x_386_; 
v___x_384_ = lean_box(0);
v___f_385_ = lean_alloc_closure((void*)(lp_mathlib_WithOne_instInv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_385_, 0, v_inst_383_);
v___x_386_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_386_, 0, v___x_384_);
lean_ctor_set(v___x_386_, 1, v___f_385_);
return v___x_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instNegZeroClass(lean_object* v_00_u03b1_387_, lean_object* v_inst_388_){
_start:
{
lean_object* v___x_389_; 
v___x_389_ = lp_mathlib_WithZero_instNegZeroClass___redArg(v_inst_388_);
return v___x_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_inhabited(lean_object* v_00_u03b1_390_){
_start:
{
lean_object* v___x_391_; 
v___x_391_ = lean_box(0);
return v___x_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_inhabited(lean_object* v_00_u03b1_392_){
_start:
{
lean_object* v___x_393_; 
v___x_393_ = lean_box(0);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_coe___redArg(lean_object* v_val_394_){
_start:
{
lean_object* v___x_395_; 
v___x_395_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_395_, 0, v_val_394_);
return v___x_395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_coe(lean_object* v_00_u03b1_396_, lean_object* v_val_397_){
_start:
{
lean_object* v___x_398_; 
v___x_398_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_398_, 0, v_val_397_);
return v___x_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_coe___redArg(lean_object* v_val_399_){
_start:
{
lean_object* v___x_400_; 
v___x_400_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_400_, 0, v_val_399_);
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_coe(lean_object* v_00_u03b1_401_, lean_object* v_val_402_){
_start:
{
lean_object* v___x_403_; 
v___x_403_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_403_, 0, v_val_402_);
return v___x_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instCoeTC(lean_object* v_00_u03b1_405_){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = ((lean_object*)(lp_mathlib_WithOne_instCoeTC___closed__0));
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instCoeTC___lam__0(lean_object* v___y_407_){
_start:
{
lean_object* v___x_408_; 
v___x_408_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_408_, 0, v___y_407_);
return v___x_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instCoeTC(lean_object* v_00_u03b1_410_){
_start:
{
lean_object* v___f_411_; 
v___f_411_ = ((lean_object*)(lp_mathlib_WithZero_instCoeTC___closed__0));
return v___f_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_recZeroCoe___redArg(lean_object* v_zero_412_, lean_object* v_coe_413_, lean_object* v_x_414_){
_start:
{
if (lean_obj_tag(v_x_414_) == 0)
{
lean_dec(v_coe_413_);
lean_inc(v_zero_412_);
return v_zero_412_;
}
else
{
lean_object* v_val_415_; lean_object* v___x_416_; 
v_val_415_ = lean_ctor_get(v_x_414_, 0);
lean_inc(v_val_415_);
lean_dec_ref_known(v_x_414_, 1);
v___x_416_ = lean_apply_1(v_coe_413_, v_val_415_);
return v___x_416_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_recZeroCoe___redArg___boxed(lean_object* v_zero_417_, lean_object* v_coe_418_, lean_object* v_x_419_){
_start:
{
lean_object* v_res_420_; 
v_res_420_ = lp_mathlib_WithZero_recZeroCoe___redArg(v_zero_417_, v_coe_418_, v_x_419_);
lean_dec(v_zero_417_);
return v_res_420_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_recZeroCoe(lean_object* v_00_u03b1_421_, lean_object* v_motive_422_, lean_object* v_zero_423_, lean_object* v_coe_424_, lean_object* v_x_425_){
_start:
{
lean_object* v___x_426_; 
v___x_426_ = lp_mathlib_WithZero_recZeroCoe___redArg(v_zero_423_, v_coe_424_, v_x_425_);
return v___x_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_recZeroCoe___boxed(lean_object* v_00_u03b1_427_, lean_object* v_motive_428_, lean_object* v_zero_429_, lean_object* v_coe_430_, lean_object* v_x_431_){
_start:
{
lean_object* v_res_432_; 
v_res_432_ = lp_mathlib_WithZero_recZeroCoe(v_00_u03b1_427_, v_motive_428_, v_zero_429_, v_coe_430_, v_x_431_);
lean_dec(v_zero_429_);
return v_res_432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_recOneCoe___redArg(lean_object* v_one_433_, lean_object* v_coe_434_, lean_object* v_x_435_){
_start:
{
if (lean_obj_tag(v_x_435_) == 0)
{
lean_dec(v_coe_434_);
lean_inc(v_one_433_);
return v_one_433_;
}
else
{
lean_object* v_val_436_; lean_object* v___x_437_; 
v_val_436_ = lean_ctor_get(v_x_435_, 0);
lean_inc(v_val_436_);
lean_dec_ref_known(v_x_435_, 1);
v___x_437_ = lean_apply_1(v_coe_434_, v_val_436_);
return v___x_437_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_recOneCoe___redArg___boxed(lean_object* v_one_438_, lean_object* v_coe_439_, lean_object* v_x_440_){
_start:
{
lean_object* v_res_441_; 
v_res_441_ = lp_mathlib_WithOne_recOneCoe___redArg(v_one_438_, v_coe_439_, v_x_440_);
lean_dec(v_one_438_);
return v_res_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_recOneCoe(lean_object* v_00_u03b1_442_, lean_object* v_motive_443_, lean_object* v_one_444_, lean_object* v_coe_445_, lean_object* v_x_446_){
_start:
{
lean_object* v___x_447_; 
v___x_447_ = lp_mathlib_WithOne_recOneCoe___redArg(v_one_444_, v_coe_445_, v_x_446_);
return v___x_447_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_recOneCoe___boxed(lean_object* v_00_u03b1_448_, lean_object* v_motive_449_, lean_object* v_one_450_, lean_object* v_coe_451_, lean_object* v_x_452_){
_start:
{
lean_object* v_res_453_; 
v_res_453_ = lp_mathlib_WithOne_recOneCoe(v_00_u03b1_448_, v_motive_449_, v_one_450_, v_coe_451_, v_x_452_);
lean_dec(v_one_450_);
return v_res_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unone___redArg(lean_object* v_x_454_){
_start:
{
lean_object* v_val_455_; 
v_val_455_ = lean_ctor_get(v_x_454_, 0);
lean_inc(v_val_455_);
return v_val_455_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unone___redArg___boxed(lean_object* v_x_456_){
_start:
{
lean_object* v_res_457_; 
v_res_457_ = lp_mathlib_WithOne_unone___redArg(v_x_456_);
lean_dec(v_x_456_);
return v_res_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unone(lean_object* v_00_u03b1_458_, lean_object* v_x_459_, lean_object* v_x_460_){
_start:
{
lean_object* v_val_461_; 
v_val_461_ = lean_ctor_get(v_x_459_, 0);
lean_inc(v_val_461_);
return v_val_461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unone___boxed(lean_object* v_00_u03b1_462_, lean_object* v_x_463_, lean_object* v_x_464_){
_start:
{
lean_object* v_res_465_; 
v_res_465_ = lp_mathlib_WithOne_unone(v_00_u03b1_462_, v_x_463_, v_x_464_);
lean_dec(v_x_463_);
return v_res_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzero_match__1___redArg(lean_object* v_x_466_, lean_object* v_h__1_467_){
_start:
{
lean_object* v_val_468_; lean_object* v___x_469_; 
v_val_468_ = lean_ctor_get(v_x_466_, 0);
lean_inc(v_val_468_);
lean_dec(v_x_466_);
v___x_469_ = lean_apply_2(v_h__1_467_, v_val_468_, lean_box(0));
return v___x_469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzero_match__1(lean_object* v_00_u03b1_470_, lean_object* v_motive_471_, lean_object* v_x_472_, lean_object* v_x_473_, lean_object* v_h__1_474_){
_start:
{
lean_object* v___x_475_; 
v___x_475_ = lp_mathlib_WithZero_unzero_match__1___redArg(v_x_472_, v_h__1_474_);
return v___x_475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzero___redArg(lean_object* v_x_476_){
_start:
{
lean_object* v_val_477_; 
v_val_477_ = lean_ctor_get(v_x_476_, 0);
lean_inc(v_val_477_);
return v_val_477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzero___redArg___boxed(lean_object* v_x_478_){
_start:
{
lean_object* v_res_479_; 
v_res_479_ = lp_mathlib_WithZero_unzero___redArg(v_x_478_);
lean_dec(v_x_478_);
return v_res_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzero(lean_object* v_00_u03b1_480_, lean_object* v_x_481_, lean_object* v_x_482_){
_start:
{
lean_object* v_val_483_; 
v_val_483_ = lean_ctor_get(v_x_481_, 0);
lean_inc(v_val_483_);
return v_val_483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzero___boxed(lean_object* v_00_u03b1_484_, lean_object* v_x_485_, lean_object* v_x_486_){
_start:
{
lean_object* v_res_487_; 
v_res_487_ = lp_mathlib_WithZero_unzero(v_00_u03b1_484_, v_x_485_, v_x_486_);
lean_dec(v_x_485_);
return v_res_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_WithOne_Defs_0__WithOne_unone_match__1_splitter___redArg(lean_object* v_x_488_, lean_object* v_h__1_489_){
_start:
{
lean_object* v_val_490_; lean_object* v___x_491_; 
v_val_490_ = lean_ctor_get(v_x_488_, 0);
lean_inc(v_val_490_);
lean_dec(v_x_488_);
v___x_491_ = lean_apply_2(v_h__1_489_, v_val_490_, lean_box(0));
return v___x_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_WithOne_Defs_0__WithOne_unone_match__1_splitter(lean_object* v_00_u03b1_492_, lean_object* v_motive_493_, lean_object* v_x_494_, lean_object* v_x_495_, lean_object* v_h__1_496_){
_start:
{
lean_object* v_val_497_; lean_object* v___x_498_; 
v_val_497_ = lean_ctor_get(v_x_494_, 0);
lean_inc(v_val_497_);
lean_dec(v_x_494_);
v___x_498_ = lean_apply_2(v_h__1_496_, v_val_497_, lean_box(0));
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_WithOne_Defs_0__WithZero_unzero_match__1_splitter___redArg(lean_object* v_x_499_, lean_object* v_h__1_500_){
_start:
{
lean_object* v_val_501_; lean_object* v___x_502_; 
v_val_501_ = lean_ctor_get(v_x_499_, 0);
lean_inc(v_val_501_);
lean_dec(v_x_499_);
v___x_502_ = lean_apply_2(v_h__1_500_, v_val_501_, lean_box(0));
return v___x_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Algebra_Group_WithOne_Defs_0__WithZero_unzero_match__1_splitter(lean_object* v_00_u03b1_503_, lean_object* v_motive_504_, lean_object* v_x_505_, lean_object* v_x_506_, lean_object* v_h__1_507_){
_start:
{
lean_object* v_val_508_; lean_object* v___x_509_; 
v_val_508_ = lean_ctor_get(v_x_505_, 0);
lean_inc(v_val_508_);
lean_dec(v_x_505_);
v___x_509_ = lean_apply_2(v_h__1_507_, v_val_508_, lean_box(0));
return v___x_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMulOneClass___redArg(lean_object* v_inst_510_){
_start:
{
lean_object* v___x_511_; lean_object* v___x_512_; lean_object* v___x_513_; 
v___x_511_ = lean_box(0);
v___x_512_ = lp_mathlib_WithOne_instMul___redArg(v_inst_510_);
v___x_513_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_513_, 0, v___x_511_);
lean_ctor_set(v___x_513_, 1, v___x_512_);
return v___x_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMulOneClass(lean_object* v_00_u03b1_514_, lean_object* v_inst_515_){
_start:
{
lean_object* v___x_516_; 
v___x_516_ = lp_mathlib_WithOne_instMulOneClass___redArg(v_inst_515_);
return v___x_516_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddZeroClass___redArg(lean_object* v_inst_517_){
_start:
{
lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_520_; 
v___x_518_ = lean_box(0);
v___x_519_ = lp_mathlib_WithZero_instAdd___redArg(v_inst_517_);
v___x_520_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_520_, 0, v___x_518_);
lean_ctor_set(v___x_520_, 1, v___x_519_);
return v___x_520_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddZeroClass(lean_object* v_00_u03b1_521_, lean_object* v_inst_522_){
_start:
{
lean_object* v___x_523_; 
v___x_523_ = lp_mathlib_WithZero_instAddZeroClass___redArg(v_inst_522_);
return v___x_523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonoid___redArg(lean_object* v_inst_524_){
_start:
{
lean_object* v___x_525_; lean_object* v___x_526_; lean_object* v___x_527_; lean_object* v___x_528_; 
v___x_525_ = lean_box(0);
v___x_526_ = lp_mathlib_WithOne_instMul___redArg(v_inst_524_);
lean_inc_ref(v___x_526_);
v___x_527_ = lean_alloc_closure((void*)(lp_mathlib_npowBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_527_, 0, lean_box(0));
lean_closure_set(v___x_527_, 1, v___x_526_);
lean_closure_set(v___x_527_, 2, v___x_525_);
v___x_528_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_528_, 0, v___x_525_);
lean_ctor_set(v___x_528_, 1, v___x_526_);
lean_ctor_set(v___x_528_, 2, v___x_527_);
return v___x_528_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instMonoid(lean_object* v_00_u03b1_529_, lean_object* v_inst_530_){
_start:
{
lean_object* v___x_531_; 
v___x_531_ = lp_mathlib_WithOne_instMonoid___redArg(v_inst_530_);
return v___x_531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddMonoid___redArg(lean_object* v_inst_532_){
_start:
{
lean_object* v___x_533_; lean_object* v___x_534_; lean_object* v___x_535_; lean_object* v___x_536_; 
v___x_533_ = lean_box(0);
v___x_534_ = lp_mathlib_WithZero_instAdd___redArg(v_inst_532_);
lean_inc_ref(v___x_534_);
v___x_535_ = lean_alloc_closure((void*)(lp_mathlib_nsmulBinRecAuto___boxed), 5, 3);
lean_closure_set(v___x_535_, 0, lean_box(0));
lean_closure_set(v___x_535_, 1, v___x_534_);
lean_closure_set(v___x_535_, 2, v___x_533_);
v___x_536_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_536_, 0, v___x_533_);
lean_ctor_set(v___x_536_, 1, v___x_534_);
lean_ctor_set(v___x_536_, 2, v___x_535_);
return v___x_536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddMonoid(lean_object* v_00_u03b1_537_, lean_object* v_inst_538_){
_start:
{
lean_object* v___x_539_; 
v___x_539_ = lp_mathlib_WithZero_instAddMonoid___redArg(v_inst_538_);
return v___x_539_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instCommMonoid___redArg(lean_object* v_inst_540_){
_start:
{
lean_object* v___x_541_; 
v___x_541_ = lp_mathlib_WithOne_instMonoid___redArg(v_inst_540_);
return v___x_541_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_instCommMonoid(lean_object* v_00_u03b1_542_, lean_object* v_inst_543_){
_start:
{
lean_object* v___x_544_; 
v___x_544_ = lp_mathlib_WithOne_instMonoid___redArg(v_inst_543_);
return v___x_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddCommMonoid___redArg(lean_object* v_inst_545_){
_start:
{
lean_object* v___x_546_; 
v___x_546_ = lp_mathlib_WithZero_instAddMonoid___redArg(v_inst_545_);
return v___x_546_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_instAddCommMonoid(lean_object* v_00_u03b1_547_, lean_object* v_inst_548_){
_start:
{
lean_object* v___x_549_; 
v___x_549_ = lp_mathlib_WithZero_instAddMonoid___redArg(v_inst_548_);
return v___x_549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unoneD___redArg___lam__0(lean_object* v___y_550_){
_start:
{
lean_inc(v___y_550_);
return v___y_550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unoneD___redArg___lam__0___boxed(lean_object* v___y_551_){
_start:
{
lean_object* v_res_552_; 
v_res_552_ = lp_mathlib_WithOne_unoneD___redArg___lam__0(v___y_551_);
lean_dec(v___y_551_);
return v_res_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unoneD___redArg(lean_object* v_d_554_, lean_object* v_x_555_){
_start:
{
lean_object* v___f_556_; lean_object* v___x_557_; 
v___f_556_ = ((lean_object*)(lp_mathlib_WithOne_unoneD___redArg___closed__0));
v___x_557_ = lp_mathlib_WithOne_recOneCoe___redArg(v_d_554_, v___f_556_, v_x_555_);
return v___x_557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unoneD___redArg___boxed(lean_object* v_d_558_, lean_object* v_x_559_){
_start:
{
lean_object* v_res_560_; 
v_res_560_ = lp_mathlib_WithOne_unoneD___redArg(v_d_558_, v_x_559_);
lean_dec(v_d_558_);
return v_res_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unoneD(lean_object* v_00_u03b1_561_, lean_object* v_d_562_, lean_object* v_x_563_){
_start:
{
lean_object* v___x_564_; 
v___x_564_ = lp_mathlib_WithOne_unoneD___redArg(v_d_562_, v_x_563_);
return v___x_564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithOne_unoneD___boxed(lean_object* v_00_u03b1_565_, lean_object* v_d_566_, lean_object* v_x_567_){
_start:
{
lean_object* v_res_568_; 
v_res_568_ = lp_mathlib_WithOne_unoneD(v_00_u03b1_565_, v_d_566_, v_x_567_);
lean_dec(v_d_566_);
return v_res_568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzeroD___redArg(lean_object* v_d_569_, lean_object* v_x_570_){
_start:
{
lean_object* v___f_571_; lean_object* v___x_572_; 
v___f_571_ = ((lean_object*)(lp_mathlib_WithOne_unoneD___redArg___closed__0));
v___x_572_ = lp_mathlib_WithZero_recZeroCoe___redArg(v_d_569_, v___f_571_, v_x_570_);
return v___x_572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzeroD___redArg___boxed(lean_object* v_d_573_, lean_object* v_x_574_){
_start:
{
lean_object* v_res_575_; 
v_res_575_ = lp_mathlib_WithZero_unzeroD___redArg(v_d_573_, v_x_574_);
lean_dec(v_d_573_);
return v_res_575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzeroD(lean_object* v_00_u03b1_576_, lean_object* v_d_577_, lean_object* v_x_578_){
_start:
{
lean_object* v___x_579_; 
v___x_579_ = lp_mathlib_WithZero_unzeroD___redArg(v_d_577_, v_x_578_);
return v___x_579_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithZero_unzeroD___boxed(lean_object* v_00_u03b1_580_, lean_object* v_d_581_, lean_object* v_x_582_){
_start:
{
lean_object* v_res_583_; 
v_res_583_ = lp_mathlib_WithZero_unzeroD(v_00_u03b1_580_, v_d_581_, v_x_582_);
lean_dec(v_d_581_);
return v_res_583_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Basic_Nontrivial_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Option_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_WithOne_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Basic_Nontrivial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Option_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_WithOne_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Basic_Nontrivial_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Option_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Common(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Attr_Core(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_WithOne_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_DivInvMonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Basic_Nontrivial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Option_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Common(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Attr_Core(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_WithOne_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_WithOne_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_WithOne_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
