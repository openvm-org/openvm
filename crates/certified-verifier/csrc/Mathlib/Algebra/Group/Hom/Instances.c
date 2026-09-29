// Lean compiler output
// Module: Mathlib.Algebra.Group.Hom.Instances
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Hom.Basic public import Mathlib.Algebra.Group.InjSurj public import Mathlib.Algebra.Group.Pi.Basic public import Mathlib.Tactic.FastInstance
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
lean_object* lp_mathlib_OneHom_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_instOneOneHom___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_ZeroHom_instAdd___redArg(lean_object*);
lean_object* lp_mathlib_NSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instZeroAddMonoidHom___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_add___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_instNeg___redArg(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_instSub___redArg(lean_object*);
lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_OneHom_instMul___redArg(lean_object*);
lean_object* lp_mathlib_NPow_ofPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(lean_object*);
lean_object* lp_mathlib_OneHom_instInv___redArg(lean_object*);
lean_object* lp_mathlib_OneHom_instDiv___redArg(lean_object*);
lean_object* lp_mathlib_ZPow_ofPow___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instOneMonoidHom___redArg(lean_object*);
lean_object* lp_mathlib_MonoidHom_mul___redArg(lean_object*);
lean_object* lp_mathlib_MonoidHom_instInv___redArg(lean_object*);
lean_object* lp_mathlib_MonoidHom_instDiv___redArg(lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_ZeroHom_instNeg___redArg(lean_object*);
lean_object* lp_mathlib_ZeroHom_instSub___redArg(lean_object*);
lean_object* lp_mathlib_OneHom_id___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instPow___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instPow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instPow(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instPow___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instNSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instNSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instNSMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instNSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instPow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instPow(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instPow___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instNSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instNSMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instNSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instAddCommMonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instAddCommMonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instZPow___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instZPow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instZPow(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instZPow___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instZSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instZSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instZSMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instZSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instZPow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instZPow(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instZPow___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instZSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instZSMul(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instZSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instCommGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instCommGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddCommGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddCommGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instCommGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instCommGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instAddCommGroup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instAddCommGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__3___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__8___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__8(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__1___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__5___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instIntCast___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instIntCast___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instIntCast___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instIntCast(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_flip___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_flip___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_flip(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_flip___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_flip___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_flip(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_flip___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MonoidHom_eval___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OneHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MonoidHom_eval___closed__0 = (const lean_object*)&lp_mathlib_MonoidHom_eval___closed__0_value;
static const lean_closure_object lp_mathlib_MonoidHom_eval___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MonoidHom_flip___redArg___lam__0, .m_arity = 3, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_MonoidHom_eval___closed__0_value)} };
static const lean_object* lp_mathlib_MonoidHom_eval___closed__1 = (const lean_object*)&lp_mathlib_MonoidHom_eval___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_eval(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_eval___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_eval(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_eval___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compHom_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compHom_x27___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compHom_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compHom_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compHom_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compHom_x27___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compHom_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compHom_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compHom___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MonoidHom_compHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MonoidHom_compHom___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MonoidHom_compHom___closed__0 = (const lean_object*)&lp_mathlib_MonoidHom_compHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_flipHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_flipHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_flipHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_flipHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compl_u2082___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compl_u2082___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compl_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compl_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compl_u2082___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compl_u2082___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compl_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compl_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compr_u2082___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compr_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compr_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compr_u2082___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compr_u2082(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compr_u2082___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instPow___redArg___lam__0(lean_object* v_toNPow_1_, lean_object* v_f_2_, lean_object* v_n_3_, lean_object* v___y_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; 
v___x_5_ = lean_apply_1(v_f_2_, v___y_4_);
v___x_6_ = lean_apply_2(v_toNPow_1_, v_n_3_, v___x_5_);
return v___x_6_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instPow___redArg(lean_object* v_inst_7_){
_start:
{
lean_object* v_toNPow_8_; lean_object* v___f_9_; 
v_toNPow_8_ = lean_ctor_get(v_inst_7_, 2);
lean_inc(v_toNPow_8_);
lean_dec_ref(v_inst_7_);
v___f_9_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_instPow___redArg___lam__0), 4, 1);
lean_closure_set(v___f_9_, 0, v_toNPow_8_);
return v___f_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instPow(lean_object* v_M_10_, lean_object* v_N_11_, lean_object* v_inst_12_, lean_object* v_inst_13_){
_start:
{
lean_object* v___x_14_; 
v___x_14_ = lp_mathlib_OneHom_instPow___redArg(v_inst_13_);
return v___x_14_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instPow___boxed(lean_object* v_M_15_, lean_object* v_N_16_, lean_object* v_inst_17_, lean_object* v_inst_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib_OneHom_instPow(v_M_15_, v_N_16_, v_inst_17_, v_inst_18_);
lean_dec(v_inst_17_);
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instNSMul___redArg___lam__0(lean_object* v_toNSMul_20_, lean_object* v_n_21_, lean_object* v_f_22_, lean_object* v___y_23_){
_start:
{
lean_object* v___x_24_; lean_object* v___x_25_; 
v___x_24_ = lean_apply_1(v_f_22_, v___y_23_);
v___x_25_ = lean_apply_2(v_toNSMul_20_, v_n_21_, v___x_24_);
return v___x_25_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instNSMul___redArg(lean_object* v_inst_26_){
_start:
{
lean_object* v_toNSMul_27_; lean_object* v___f_28_; 
v_toNSMul_27_ = lean_ctor_get(v_inst_26_, 2);
lean_inc(v_toNSMul_27_);
lean_dec_ref(v_inst_26_);
v___f_28_ = lean_alloc_closure((void*)(lp_mathlib_ZeroHom_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_28_, 0, v_toNSMul_27_);
return v___f_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instNSMul(lean_object* v_M_29_, lean_object* v_N_30_, lean_object* v_inst_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_ZeroHom_instNSMul___redArg(v_inst_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instNSMul___boxed(lean_object* v_M_34_, lean_object* v_N_35_, lean_object* v_inst_36_, lean_object* v_inst_37_){
_start:
{
lean_object* v_res_38_; 
v_res_38_ = lp_mathlib_ZeroHom_instNSMul(v_M_34_, v_N_35_, v_inst_36_, v_inst_37_);
lean_dec(v_inst_36_);
return v_res_38_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instPow___redArg(lean_object* v_inst_39_){
_start:
{
lean_object* v_toNPow_40_; lean_object* v___f_41_; 
v_toNPow_40_ = lean_ctor_get(v_inst_39_, 2);
lean_inc(v_toNPow_40_);
lean_dec_ref(v_inst_39_);
v___f_41_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_instPow___redArg___lam__0), 4, 1);
lean_closure_set(v___f_41_, 0, v_toNPow_40_);
return v___f_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instPow(lean_object* v_M_42_, lean_object* v_N_43_, lean_object* v_inst_44_, lean_object* v_inst_45_){
_start:
{
lean_object* v___x_46_; 
v___x_46_ = lp_mathlib_MonoidHom_instPow___redArg(v_inst_45_);
return v___x_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instPow___boxed(lean_object* v_M_47_, lean_object* v_N_48_, lean_object* v_inst_49_, lean_object* v_inst_50_){
_start:
{
lean_object* v_res_51_; 
v_res_51_ = lp_mathlib_MonoidHom_instPow(v_M_47_, v_N_48_, v_inst_49_, v_inst_50_);
lean_dec_ref(v_inst_49_);
return v_res_51_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instNSMul___redArg(lean_object* v_inst_52_){
_start:
{
lean_object* v_toNSMul_53_; lean_object* v___f_54_; 
v_toNSMul_53_ = lean_ctor_get(v_inst_52_, 2);
lean_inc(v_toNSMul_53_);
lean_dec_ref(v_inst_52_);
v___f_54_ = lean_alloc_closure((void*)(lp_mathlib_ZeroHom_instNSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_54_, 0, v_toNSMul_53_);
return v___f_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instNSMul(lean_object* v_M_55_, lean_object* v_N_56_, lean_object* v_inst_57_, lean_object* v_inst_58_){
_start:
{
lean_object* v___x_59_; 
v___x_59_ = lp_mathlib_AddMonoidHom_instNSMul___redArg(v_inst_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instNSMul___boxed(lean_object* v_M_60_, lean_object* v_N_61_, lean_object* v_inst_62_, lean_object* v_inst_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_mathlib_AddMonoidHom_instNSMul(v_M_60_, v_N_61_, v_inst_62_, v_inst_63_);
lean_dec_ref(v_inst_62_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instMonoid___redArg(lean_object* v_inst_65_){
_start:
{
lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v_toOne_68_; lean_object* v___f_69_; lean_object* v___x_70_; lean_object* v___x_71_; lean_object* v___f_72_; lean_object* v___x_73_; 
v___x_66_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_65_);
lean_inc_ref(v___x_66_);
v___x_67_ = lp_mathlib_MulOneClass_toMulOne___redArg(v___x_66_);
v_toOne_68_ = lean_ctor_get(v___x_67_, 0);
lean_inc(v_toOne_68_);
lean_dec_ref(v___x_67_);
v___f_69_ = lean_alloc_closure((void*)(lp_mathlib_instOneOneHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_69_, 0, v_toOne_68_);
v___x_70_ = lp_mathlib_OneHom_instMul___redArg(v___x_66_);
v___x_71_ = lp_mathlib_OneHom_instPow___redArg(v_inst_65_);
v___f_72_ = lean_alloc_closure((void*)(lp_mathlib_NPow_ofPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_72_, 0, v___x_71_);
v___x_73_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_73_, 0, v___f_69_);
lean_ctor_set(v___x_73_, 1, v___x_70_);
lean_ctor_set(v___x_73_, 2, v___f_72_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instMonoid(lean_object* v_M_74_, lean_object* v_N_75_, lean_object* v_inst_76_, lean_object* v_inst_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lp_mathlib_OneHom_instMonoid___redArg(v_inst_77_);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instMonoid___boxed(lean_object* v_M_79_, lean_object* v_N_80_, lean_object* v_inst_81_, lean_object* v_inst_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_OneHom_instMonoid(v_M_79_, v_N_80_, v_inst_81_, v_inst_82_);
lean_dec(v_inst_81_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddMonoid___redArg(lean_object* v_inst_84_){
_start:
{
lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v_toZero_87_; lean_object* v___f_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___f_91_; lean_object* v___x_92_; 
v___x_85_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_84_);
lean_inc_ref(v___x_85_);
v___x_86_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_85_);
v_toZero_87_ = lean_ctor_get(v___x_86_, 0);
lean_inc(v_toZero_87_);
lean_dec_ref(v___x_86_);
v___f_88_ = lean_alloc_closure((void*)(lp_mathlib_instOneOneHom___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_88_, 0, v_toZero_87_);
v___x_89_ = lp_mathlib_ZeroHom_instAdd___redArg(v___x_85_);
v___x_90_ = lp_mathlib_ZeroHom_instNSMul___redArg(v_inst_84_);
v___f_91_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_91_, 0, v___x_90_);
v___x_92_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_92_, 0, v___f_88_);
lean_ctor_set(v___x_92_, 1, v___x_89_);
lean_ctor_set(v___x_92_, 2, v___f_91_);
return v___x_92_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddMonoid(lean_object* v_M_93_, lean_object* v_N_94_, lean_object* v_inst_95_, lean_object* v_inst_96_){
_start:
{
lean_object* v___x_97_; 
v___x_97_ = lp_mathlib_ZeroHom_instAddMonoid___redArg(v_inst_96_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddMonoid___boxed(lean_object* v_M_98_, lean_object* v_N_99_, lean_object* v_inst_100_, lean_object* v_inst_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_ZeroHom_instAddMonoid(v_M_98_, v_N_99_, v_inst_100_, v_inst_101_);
lean_dec(v_inst_100_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instCommMonoid___redArg(lean_object* v_inst_103_){
_start:
{
lean_object* v___x_104_; 
v___x_104_ = lp_mathlib_OneHom_instMonoid___redArg(v_inst_103_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instCommMonoid(lean_object* v_M_105_, lean_object* v_N_106_, lean_object* v_inst_107_, lean_object* v_inst_108_){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lp_mathlib_OneHom_instMonoid___redArg(v_inst_108_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instCommMonoid___boxed(lean_object* v_M_110_, lean_object* v_N_111_, lean_object* v_inst_112_, lean_object* v_inst_113_){
_start:
{
lean_object* v_res_114_; 
v_res_114_ = lp_mathlib_OneHom_instCommMonoid(v_M_110_, v_N_111_, v_inst_112_, v_inst_113_);
lean_dec(v_inst_112_);
return v_res_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddCommMonoid___redArg(lean_object* v_inst_115_){
_start:
{
lean_object* v___x_116_; 
v___x_116_ = lp_mathlib_ZeroHom_instAddMonoid___redArg(v_inst_115_);
return v___x_116_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddCommMonoid(lean_object* v_M_117_, lean_object* v_N_118_, lean_object* v_inst_119_, lean_object* v_inst_120_){
_start:
{
lean_object* v___x_121_; 
v___x_121_ = lp_mathlib_ZeroHom_instAddMonoid___redArg(v_inst_120_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddCommMonoid___boxed(lean_object* v_M_122_, lean_object* v_N_123_, lean_object* v_inst_124_, lean_object* v_inst_125_){
_start:
{
lean_object* v_res_126_; 
v_res_126_ = lp_mathlib_ZeroHom_instAddCommMonoid(v_M_122_, v_N_123_, v_inst_124_, v_inst_125_);
lean_dec(v_inst_124_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instCommMonoid___redArg(lean_object* v_inst_127_){
_start:
{
lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___f_132_; lean_object* v___x_133_; 
v___x_128_ = lp_mathlib_Monoid_toMulOneClass___redArg(v_inst_127_);
v___x_129_ = lp_mathlib_instOneMonoidHom___redArg(v___x_128_);
v___x_130_ = lp_mathlib_MonoidHom_mul___redArg(v_inst_127_);
v___x_131_ = lp_mathlib_MonoidHom_instPow___redArg(v_inst_127_);
v___f_132_ = lean_alloc_closure((void*)(lp_mathlib_NPow_ofPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_132_, 0, v___x_131_);
v___x_133_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_133_, 0, v___x_129_);
lean_ctor_set(v___x_133_, 1, v___x_130_);
lean_ctor_set(v___x_133_, 2, v___f_132_);
return v___x_133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instCommMonoid(lean_object* v_M_134_, lean_object* v_N_135_, lean_object* v_inst_136_, lean_object* v_inst_137_){
_start:
{
lean_object* v___x_138_; 
v___x_138_ = lp_mathlib_MonoidHom_instCommMonoid___redArg(v_inst_137_);
return v___x_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instCommMonoid___boxed(lean_object* v_M_139_, lean_object* v_N_140_, lean_object* v_inst_141_, lean_object* v_inst_142_){
_start:
{
lean_object* v_res_143_; 
v_res_143_ = lp_mathlib_MonoidHom_instCommMonoid(v_M_139_, v_N_140_, v_inst_141_, v_inst_142_);
lean_dec_ref(v_inst_141_);
return v_res_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(lean_object* v_inst_144_){
_start:
{
lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___f_149_; lean_object* v___x_150_; 
v___x_145_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_144_);
v___x_146_ = lp_mathlib_instZeroAddMonoidHom___redArg(v___x_145_);
v___x_147_ = lp_mathlib_AddMonoidHom_add___redArg(v_inst_144_);
v___x_148_ = lp_mathlib_AddMonoidHom_instNSMul___redArg(v_inst_144_);
v___f_149_ = lean_alloc_closure((void*)(lp_mathlib_NSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_149_, 0, v___x_148_);
v___x_150_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_150_, 0, v___x_146_);
lean_ctor_set(v___x_150_, 1, v___x_147_);
lean_ctor_set(v___x_150_, 2, v___f_149_);
return v___x_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instAddCommMonoid(lean_object* v_M_151_, lean_object* v_N_152_, lean_object* v_inst_153_, lean_object* v_inst_154_){
_start:
{
lean_object* v___x_155_; 
v___x_155_ = lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(v_inst_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instAddCommMonoid___boxed(lean_object* v_M_156_, lean_object* v_N_157_, lean_object* v_inst_158_, lean_object* v_inst_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib_AddMonoidHom_instAddCommMonoid(v_M_156_, v_N_157_, v_inst_158_, v_inst_159_);
lean_dec_ref(v_inst_158_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instZPow___redArg___lam__0(lean_object* v_toZPow_161_, lean_object* v_f_162_, lean_object* v_n_163_, lean_object* v___y_164_){
_start:
{
lean_object* v___x_165_; lean_object* v___x_166_; 
v___x_165_ = lean_apply_1(v_f_162_, v___y_164_);
v___x_166_ = lean_apply_2(v_toZPow_161_, v_n_163_, v___x_165_);
return v___x_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instZPow___redArg(lean_object* v_inst_167_){
_start:
{
lean_object* v_toZPow_168_; lean_object* v___f_169_; 
v_toZPow_168_ = lean_ctor_get(v_inst_167_, 3);
lean_inc(v_toZPow_168_);
lean_dec_ref(v_inst_167_);
v___f_169_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_instZPow___redArg___lam__0), 4, 1);
lean_closure_set(v___f_169_, 0, v_toZPow_168_);
return v___f_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instZPow(lean_object* v_M_170_, lean_object* v_N_171_, lean_object* v_inst_172_, lean_object* v_inst_173_){
_start:
{
lean_object* v___x_174_; 
v___x_174_ = lp_mathlib_OneHom_instZPow___redArg(v_inst_173_);
return v___x_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instZPow___boxed(lean_object* v_M_175_, lean_object* v_N_176_, lean_object* v_inst_177_, lean_object* v_inst_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_mathlib_OneHom_instZPow(v_M_175_, v_N_176_, v_inst_177_, v_inst_178_);
lean_dec(v_inst_177_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instZSMul___redArg___lam__0(lean_object* v_toZSMul_180_, lean_object* v_n_181_, lean_object* v_f_182_, lean_object* v___y_183_){
_start:
{
lean_object* v___x_184_; lean_object* v___x_185_; 
v___x_184_ = lean_apply_1(v_f_182_, v___y_183_);
v___x_185_ = lean_apply_2(v_toZSMul_180_, v_n_181_, v___x_184_);
return v___x_185_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instZSMul___redArg(lean_object* v_inst_186_){
_start:
{
lean_object* v_toZSMul_187_; lean_object* v___f_188_; 
v_toZSMul_187_ = lean_ctor_get(v_inst_186_, 3);
lean_inc(v_toZSMul_187_);
lean_dec_ref(v_inst_186_);
v___f_188_ = lean_alloc_closure((void*)(lp_mathlib_ZeroHom_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_188_, 0, v_toZSMul_187_);
return v___f_188_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instZSMul(lean_object* v_M_189_, lean_object* v_N_190_, lean_object* v_inst_191_, lean_object* v_inst_192_){
_start:
{
lean_object* v___x_193_; 
v___x_193_ = lp_mathlib_ZeroHom_instZSMul___redArg(v_inst_192_);
return v___x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instZSMul___boxed(lean_object* v_M_194_, lean_object* v_N_195_, lean_object* v_inst_196_, lean_object* v_inst_197_){
_start:
{
lean_object* v_res_198_; 
v_res_198_ = lp_mathlib_ZeroHom_instZSMul(v_M_194_, v_N_195_, v_inst_196_, v_inst_197_);
lean_dec(v_inst_196_);
return v_res_198_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instZPow___redArg(lean_object* v_inst_199_){
_start:
{
lean_object* v_toZPow_200_; lean_object* v___f_201_; 
v_toZPow_200_ = lean_ctor_get(v_inst_199_, 3);
lean_inc(v_toZPow_200_);
lean_dec_ref(v_inst_199_);
v___f_201_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_instZPow___redArg___lam__0), 4, 1);
lean_closure_set(v___f_201_, 0, v_toZPow_200_);
return v___f_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instZPow(lean_object* v_M_202_, lean_object* v_N_203_, lean_object* v_inst_204_, lean_object* v_inst_205_){
_start:
{
lean_object* v___x_206_; 
v___x_206_ = lp_mathlib_MonoidHom_instZPow___redArg(v_inst_205_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instZPow___boxed(lean_object* v_M_207_, lean_object* v_N_208_, lean_object* v_inst_209_, lean_object* v_inst_210_){
_start:
{
lean_object* v_res_211_; 
v_res_211_ = lp_mathlib_MonoidHom_instZPow(v_M_207_, v_N_208_, v_inst_209_, v_inst_210_);
lean_dec_ref(v_inst_209_);
return v_res_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instZSMul___redArg(lean_object* v_inst_212_){
_start:
{
lean_object* v_toZSMul_213_; lean_object* v___f_214_; 
v_toZSMul_213_ = lean_ctor_get(v_inst_212_, 3);
lean_inc(v_toZSMul_213_);
lean_dec_ref(v_inst_212_);
v___f_214_ = lean_alloc_closure((void*)(lp_mathlib_ZeroHom_instZSMul___redArg___lam__0), 4, 1);
lean_closure_set(v___f_214_, 0, v_toZSMul_213_);
return v___f_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instZSMul(lean_object* v_M_215_, lean_object* v_N_216_, lean_object* v_inst_217_, lean_object* v_inst_218_){
_start:
{
lean_object* v___x_219_; 
v___x_219_ = lp_mathlib_AddMonoidHom_instZSMul___redArg(v_inst_218_);
return v___x_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instZSMul___boxed(lean_object* v_M_220_, lean_object* v_N_221_, lean_object* v_inst_222_, lean_object* v_inst_223_){
_start:
{
lean_object* v_res_224_; 
v_res_224_ = lp_mathlib_AddMonoidHom_instZSMul(v_M_220_, v_N_221_, v_inst_222_, v_inst_223_);
lean_dec_ref(v_inst_222_);
return v_res_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instGroup___redArg(lean_object* v_inst_225_){
_start:
{
lean_object* v_toMonoid_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___f_232_; lean_object* v___x_233_; 
v_toMonoid_226_ = lean_ctor_get(v_inst_225_, 0);
lean_inc_ref(v_toMonoid_226_);
v___x_227_ = lp_mathlib_OneHom_instMonoid___redArg(v_toMonoid_226_);
v___x_228_ = lp_mathlib_DivInvOneMonoid_toInvOneClass___redArg(v_inst_225_);
v___x_229_ = lp_mathlib_OneHom_instInv___redArg(v___x_228_);
lean_inc_ref(v_inst_225_);
v___x_230_ = lp_mathlib_OneHom_instDiv___redArg(v_inst_225_);
v___x_231_ = lp_mathlib_OneHom_instZPow___redArg(v_inst_225_);
v___f_232_ = lean_alloc_closure((void*)(lp_mathlib_ZPow_ofPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_232_, 0, v___x_231_);
v___x_233_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_233_, 0, v___x_227_);
lean_ctor_set(v___x_233_, 1, v___x_229_);
lean_ctor_set(v___x_233_, 2, v___x_230_);
lean_ctor_set(v___x_233_, 3, v___f_232_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instGroup(lean_object* v_M_234_, lean_object* v_N_235_, lean_object* v_inst_236_, lean_object* v_inst_237_){
_start:
{
lean_object* v___x_238_; 
v___x_238_ = lp_mathlib_OneHom_instGroup___redArg(v_inst_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instGroup___boxed(lean_object* v_M_239_, lean_object* v_N_240_, lean_object* v_inst_241_, lean_object* v_inst_242_){
_start:
{
lean_object* v_res_243_; 
v_res_243_ = lp_mathlib_OneHom_instGroup(v_M_239_, v_N_240_, v_inst_241_, v_inst_242_);
lean_dec(v_inst_241_);
return v_res_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddGroup___redArg(lean_object* v_inst_244_){
_start:
{
lean_object* v_toAddMonoid_245_; lean_object* v___x_246_; lean_object* v___x_247_; lean_object* v___x_248_; lean_object* v___x_249_; lean_object* v___x_250_; lean_object* v___f_251_; lean_object* v___x_252_; 
v_toAddMonoid_245_ = lean_ctor_get(v_inst_244_, 0);
lean_inc_ref(v_toAddMonoid_245_);
v___x_246_ = lp_mathlib_ZeroHom_instAddMonoid___redArg(v_toAddMonoid_245_);
v___x_247_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_244_);
v___x_248_ = lp_mathlib_ZeroHom_instNeg___redArg(v___x_247_);
lean_inc_ref(v_inst_244_);
v___x_249_ = lp_mathlib_ZeroHom_instSub___redArg(v_inst_244_);
v___x_250_ = lp_mathlib_ZeroHom_instZSMul___redArg(v_inst_244_);
v___f_251_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_251_, 0, v___x_250_);
v___x_252_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_252_, 0, v___x_246_);
lean_ctor_set(v___x_252_, 1, v___x_248_);
lean_ctor_set(v___x_252_, 2, v___x_249_);
lean_ctor_set(v___x_252_, 3, v___f_251_);
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddGroup(lean_object* v_M_253_, lean_object* v_N_254_, lean_object* v_inst_255_, lean_object* v_inst_256_){
_start:
{
lean_object* v___x_257_; 
v___x_257_ = lp_mathlib_ZeroHom_instAddGroup___redArg(v_inst_256_);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddGroup___boxed(lean_object* v_M_258_, lean_object* v_N_259_, lean_object* v_inst_260_, lean_object* v_inst_261_){
_start:
{
lean_object* v_res_262_; 
v_res_262_ = lp_mathlib_ZeroHom_instAddGroup(v_M_258_, v_N_259_, v_inst_260_, v_inst_261_);
lean_dec(v_inst_260_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instCommGroup___redArg(lean_object* v_inst_263_){
_start:
{
lean_object* v___x_264_; 
v___x_264_ = lp_mathlib_OneHom_instGroup___redArg(v_inst_263_);
return v___x_264_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instCommGroup(lean_object* v_M_265_, lean_object* v_N_266_, lean_object* v_inst_267_, lean_object* v_inst_268_){
_start:
{
lean_object* v___x_269_; 
v___x_269_ = lp_mathlib_OneHom_instGroup___redArg(v_inst_268_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OneHom_instCommGroup___boxed(lean_object* v_M_270_, lean_object* v_N_271_, lean_object* v_inst_272_, lean_object* v_inst_273_){
_start:
{
lean_object* v_res_274_; 
v_res_274_ = lp_mathlib_OneHom_instCommGroup(v_M_270_, v_N_271_, v_inst_272_, v_inst_273_);
lean_dec(v_inst_272_);
return v_res_274_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddCommGroup___redArg(lean_object* v_inst_275_){
_start:
{
lean_object* v___x_276_; 
v___x_276_ = lp_mathlib_ZeroHom_instAddGroup___redArg(v_inst_275_);
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddCommGroup(lean_object* v_M_277_, lean_object* v_N_278_, lean_object* v_inst_279_, lean_object* v_inst_280_){
_start:
{
lean_object* v___x_281_; 
v___x_281_ = lp_mathlib_ZeroHom_instAddGroup___redArg(v_inst_280_);
return v___x_281_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ZeroHom_instAddCommGroup___boxed(lean_object* v_M_282_, lean_object* v_N_283_, lean_object* v_inst_284_, lean_object* v_inst_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_mathlib_ZeroHom_instAddCommGroup(v_M_282_, v_N_283_, v_inst_284_, v_inst_285_);
lean_dec(v_inst_284_);
return v_res_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instCommGroup___redArg(lean_object* v_inst_287_){
_start:
{
lean_object* v_toMonoid_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; lean_object* v___f_293_; lean_object* v___x_294_; 
v_toMonoid_288_ = lean_ctor_get(v_inst_287_, 0);
lean_inc_ref(v_toMonoid_288_);
v___x_289_ = lp_mathlib_MonoidHom_instCommMonoid___redArg(v_toMonoid_288_);
v___x_290_ = lp_mathlib_MonoidHom_instInv___redArg(v_inst_287_);
lean_inc_ref(v_inst_287_);
v___x_291_ = lp_mathlib_MonoidHom_instDiv___redArg(v_inst_287_);
v___x_292_ = lp_mathlib_MonoidHom_instZPow___redArg(v_inst_287_);
v___f_293_ = lean_alloc_closure((void*)(lp_mathlib_ZPow_ofPow___redArg___lam__0), 3, 1);
lean_closure_set(v___f_293_, 0, v___x_292_);
v___x_294_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_294_, 0, v___x_289_);
lean_ctor_set(v___x_294_, 1, v___x_290_);
lean_ctor_set(v___x_294_, 2, v___x_291_);
lean_ctor_set(v___x_294_, 3, v___f_293_);
return v___x_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instCommGroup(lean_object* v_M_295_, lean_object* v_N_296_, lean_object* v_inst_297_, lean_object* v_inst_298_){
_start:
{
lean_object* v___x_299_; 
v___x_299_ = lp_mathlib_MonoidHom_instCommGroup___redArg(v_inst_298_);
return v___x_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_instCommGroup___boxed(lean_object* v_M_300_, lean_object* v_N_301_, lean_object* v_inst_302_, lean_object* v_inst_303_){
_start:
{
lean_object* v_res_304_; 
v_res_304_ = lp_mathlib_MonoidHom_instCommGroup(v_M_300_, v_N_301_, v_inst_302_, v_inst_303_);
lean_dec_ref(v_inst_302_);
return v_res_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instAddCommGroup___redArg(lean_object* v_inst_305_){
_start:
{
lean_object* v_toAddMonoid_306_; lean_object* v___x_307_; lean_object* v___x_308_; lean_object* v___x_309_; lean_object* v___x_310_; lean_object* v___f_311_; lean_object* v___x_312_; 
v_toAddMonoid_306_ = lean_ctor_get(v_inst_305_, 0);
lean_inc_ref(v_toAddMonoid_306_);
v___x_307_ = lp_mathlib_AddMonoidHom_instAddCommMonoid___redArg(v_toAddMonoid_306_);
v___x_308_ = lp_mathlib_AddMonoidHom_instNeg___redArg(v_inst_305_);
lean_inc_ref(v_inst_305_);
v___x_309_ = lp_mathlib_AddMonoidHom_instSub___redArg(v_inst_305_);
v___x_310_ = lp_mathlib_AddMonoidHom_instZSMul___redArg(v_inst_305_);
v___f_311_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_311_, 0, v___x_310_);
v___x_312_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_312_, 0, v___x_307_);
lean_ctor_set(v___x_312_, 1, v___x_308_);
lean_ctor_set(v___x_312_, 2, v___x_309_);
lean_ctor_set(v___x_312_, 3, v___f_311_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instAddCommGroup(lean_object* v_M_313_, lean_object* v_N_314_, lean_object* v_inst_315_, lean_object* v_inst_316_){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = lp_mathlib_AddMonoidHom_instAddCommGroup___redArg(v_inst_316_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_instAddCommGroup___boxed(lean_object* v_M_318_, lean_object* v_N_319_, lean_object* v_inst_320_, lean_object* v_inst_321_){
_start:
{
lean_object* v_res_322_; 
v_res_322_ = lp_mathlib_AddMonoidHom_instAddCommGroup(v_M_318_, v_N_319_, v_inst_320_, v_inst_321_);
lean_dec_ref(v_inst_320_);
return v_res_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1___redArg___lam__0(lean_object* v_toZero_323_, lean_object* v_x_324_){
_start:
{
lean_inc(v_toZero_323_);
return v_toZero_323_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1___redArg___lam__0___boxed(lean_object* v_toZero_325_, lean_object* v_x_326_){
_start:
{
lean_object* v_res_327_; 
v_res_327_ = lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1___redArg___lam__0(v_toZero_325_, v_x_326_);
lean_dec(v_x_326_);
lean_dec(v_toZero_325_);
return v_res_327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1___redArg(lean_object* v_inst_328_){
_start:
{
lean_object* v___x_329_; lean_object* v___x_330_; lean_object* v_toZero_331_; lean_object* v___f_332_; 
v___x_329_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_328_);
v___x_330_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_329_);
v_toZero_331_ = lean_ctor_get(v___x_330_, 0);
lean_inc(v_toZero_331_);
lean_dec_ref(v___x_330_);
v___f_332_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_332_, 0, v_toZero_331_);
return v___f_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1___redArg___boxed(lean_object* v_inst_333_){
_start:
{
lean_object* v_res_334_; 
v_res_334_ = lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1___redArg(v_inst_333_);
lean_dec_ref(v_inst_333_);
return v_res_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1(lean_object* v_M_335_, lean_object* v_inst_336_){
_start:
{
lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v_toZero_339_; lean_object* v___f_340_; 
v___x_337_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_336_);
v___x_338_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_337_);
v_toZero_339_ = lean_ctor_get(v___x_338_, 0);
lean_inc(v_toZero_339_);
lean_dec_ref(v___x_338_);
v___f_340_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_340_, 0, v_toZero_339_);
return v___f_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1___boxed(lean_object* v_M_341_, lean_object* v_inst_342_){
_start:
{
lean_object* v_res_343_; 
v_res_343_ = lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1(v_M_341_, v_inst_342_);
lean_dec_ref(v_inst_342_);
return v_res_343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__3___redArg___lam__0(lean_object* v_f_344_, lean_object* v_g_345_, lean_object* v_toAdd_346_, lean_object* v_m_347_){
_start:
{
lean_object* v___x_348_; lean_object* v___x_349_; lean_object* v___x_350_; 
lean_inc(v_m_347_);
v___x_348_ = lean_apply_1(v_f_344_, v_m_347_);
v___x_349_ = lean_apply_1(v_g_345_, v_m_347_);
v___x_350_ = lean_apply_2(v_toAdd_346_, v___x_348_, v___x_349_);
return v___x_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__3___redArg(lean_object* v_inst_351_, lean_object* v_f_352_, lean_object* v_g_353_){
_start:
{
lean_object* v___x_354_; lean_object* v___x_355_; lean_object* v_toAdd_356_; lean_object* v___f_357_; 
v___x_354_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_351_);
v___x_355_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_354_);
v_toAdd_356_ = lean_ctor_get(v___x_355_, 1);
lean_inc(v_toAdd_356_);
lean_dec_ref(v___x_355_);
v___f_357_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__3___redArg___lam__0), 4, 3);
lean_closure_set(v___f_357_, 0, v_f_352_);
lean_closure_set(v___f_357_, 1, v_g_353_);
lean_closure_set(v___f_357_, 2, v_toAdd_356_);
return v___f_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__3___redArg___boxed(lean_object* v_inst_358_, lean_object* v_f_359_, lean_object* v_g_360_){
_start:
{
lean_object* v_res_361_; 
v_res_361_ = lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__3___redArg(v_inst_358_, v_f_359_, v_g_360_);
lean_dec_ref(v_inst_358_);
return v_res_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__3(lean_object* v_M_362_, lean_object* v_inst_363_, lean_object* v_f_364_, lean_object* v_g_365_){
_start:
{
lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v_toAdd_368_; lean_object* v___f_369_; 
v___x_366_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_363_);
v___x_367_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_366_);
v_toAdd_368_ = lean_ctor_get(v___x_367_, 1);
lean_inc(v_toAdd_368_);
lean_dec_ref(v___x_367_);
v___f_369_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__3___redArg___lam__0), 4, 3);
lean_closure_set(v___f_369_, 0, v_f_364_);
lean_closure_set(v___f_369_, 1, v_g_365_);
lean_closure_set(v___f_369_, 2, v_toAdd_368_);
return v___f_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__3___boxed(lean_object* v_M_370_, lean_object* v_inst_371_, lean_object* v_f_372_, lean_object* v_g_373_){
_start:
{
lean_object* v_res_374_; 
v_res_374_ = lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__3(v_M_370_, v_inst_371_, v_f_372_, v_g_373_);
lean_dec_ref(v_inst_371_);
return v_res_374_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__8___redArg___lam__0(lean_object* v_x_375_, lean_object* v_toNSMul_376_, lean_object* v_n_377_, lean_object* v___y_378_){
_start:
{
lean_object* v___x_379_; lean_object* v___x_380_; 
v___x_379_ = lean_apply_1(v_x_375_, v___y_378_);
v___x_380_ = lean_apply_2(v_toNSMul_376_, v_n_377_, v___x_379_);
return v___x_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__8___redArg(lean_object* v_inst_381_, lean_object* v_n_382_, lean_object* v_x_383_){
_start:
{
lean_object* v_toNSMul_384_; lean_object* v___f_385_; 
v_toNSMul_384_ = lean_ctor_get(v_inst_381_, 2);
lean_inc(v_toNSMul_384_);
lean_dec_ref(v_inst_381_);
v___f_385_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__8___redArg___lam__0), 4, 3);
lean_closure_set(v___f_385_, 0, v_x_383_);
lean_closure_set(v___f_385_, 1, v_toNSMul_384_);
lean_closure_set(v___f_385_, 2, v_n_382_);
return v___f_385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__8(lean_object* v_M_386_, lean_object* v_inst_387_, lean_object* v_n_388_, lean_object* v_x_389_){
_start:
{
lean_object* v_toNSMul_390_; lean_object* v___f_391_; 
v_toNSMul_390_ = lean_ctor_get(v_inst_387_, 2);
lean_inc(v_toNSMul_390_);
lean_dec_ref(v_inst_387_);
v___f_391_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__8___redArg___lam__0), 4, 3);
lean_closure_set(v___f_391_, 0, v_x_389_);
lean_closure_set(v___f_391_, 1, v_toNSMul_390_);
lean_closure_set(v___f_391_, 2, v_n_388_);
return v___f_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid___redArg(lean_object* v_inst_392_){
_start:
{
lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v_toZero_395_; lean_object* v___f_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; 
v___x_393_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v_inst_392_);
v___x_394_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_393_);
v_toZero_395_ = lean_ctor_get(v___x_394_, 0);
lean_inc(v_toZero_395_);
lean_dec_ref(v___x_394_);
v___f_396_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__1___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_396_, 0, v_toZero_395_);
lean_inc_ref(v_inst_392_);
v___x_397_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__3___boxed), 4, 2);
lean_closure_set(v___x_397_, 0, lean_box(0));
lean_closure_set(v___x_397_, 1, v_inst_392_);
v___x_398_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommMonoid___aux__8), 4, 2);
lean_closure_set(v___x_398_, 0, lean_box(0));
lean_closure_set(v___x_398_, 1, v_inst_392_);
v___x_399_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_399_, 0, v___f_396_);
lean_ctor_set(v___x_399_, 1, v___x_397_);
lean_ctor_set(v___x_399_, 2, v___x_398_);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommMonoid(lean_object* v_M_400_, lean_object* v_inst_401_){
_start:
{
lean_object* v___x_402_; 
v___x_402_ = lp_mathlib_AddMonoid_End_instAddCommMonoid___redArg(v_inst_401_);
return v___x_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__1___redArg___lam__0(lean_object* v_f_403_, lean_object* v_toNeg_404_, lean_object* v_g_405_){
_start:
{
lean_object* v___x_406_; lean_object* v___x_407_; 
v___x_406_ = lean_apply_1(v_f_403_, v_g_405_);
v___x_407_ = lean_apply_1(v_toNeg_404_, v___x_406_);
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__1___redArg(lean_object* v_inst_408_, lean_object* v_f_409_){
_start:
{
lean_object* v___x_410_; lean_object* v_toNeg_411_; lean_object* v___f_412_; 
v___x_410_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_408_);
v_toNeg_411_ = lean_ctor_get(v___x_410_, 1);
lean_inc(v_toNeg_411_);
lean_dec_ref(v___x_410_);
v___f_412_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommGroup___aux__1___redArg___lam__0), 3, 2);
lean_closure_set(v___f_412_, 0, v_f_409_);
lean_closure_set(v___f_412_, 1, v_toNeg_411_);
return v___f_412_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__1___redArg___boxed(lean_object* v_inst_413_, lean_object* v_f_414_){
_start:
{
lean_object* v_res_415_; 
v_res_415_ = lp_mathlib_AddMonoid_End_instAddCommGroup___aux__1___redArg(v_inst_413_, v_f_414_);
lean_dec_ref(v_inst_413_);
return v_res_415_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__1(lean_object* v_M_416_, lean_object* v_inst_417_, lean_object* v_f_418_){
_start:
{
lean_object* v___x_419_; lean_object* v_toNeg_420_; lean_object* v___f_421_; 
v___x_419_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v_inst_417_);
v_toNeg_420_ = lean_ctor_get(v___x_419_, 1);
lean_inc(v_toNeg_420_);
lean_dec_ref(v___x_419_);
v___f_421_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommGroup___aux__1___redArg___lam__0), 3, 2);
lean_closure_set(v___f_421_, 0, v_f_418_);
lean_closure_set(v___f_421_, 1, v_toNeg_420_);
return v___f_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__1___boxed(lean_object* v_M_422_, lean_object* v_inst_423_, lean_object* v_f_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_mathlib_AddMonoid_End_instAddCommGroup___aux__1(v_M_422_, v_inst_423_, v_f_424_);
lean_dec_ref(v_inst_423_);
return v_res_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__3___redArg___lam__0(lean_object* v_f_426_, lean_object* v_g_427_, lean_object* v_toSub_428_, lean_object* v_x_429_){
_start:
{
lean_object* v___x_430_; lean_object* v___x_431_; lean_object* v___x_432_; 
lean_inc(v_x_429_);
v___x_430_ = lean_apply_1(v_f_426_, v_x_429_);
v___x_431_ = lean_apply_1(v_g_427_, v_x_429_);
v___x_432_ = lean_apply_2(v_toSub_428_, v___x_430_, v___x_431_);
return v___x_432_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__3___redArg(lean_object* v_inst_433_, lean_object* v_f_434_, lean_object* v_g_435_){
_start:
{
lean_object* v_toSub_436_; lean_object* v___f_437_; 
v_toSub_436_ = lean_ctor_get(v_inst_433_, 2);
lean_inc(v_toSub_436_);
lean_dec_ref(v_inst_433_);
v___f_437_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommGroup___aux__3___redArg___lam__0), 4, 3);
lean_closure_set(v___f_437_, 0, v_f_434_);
lean_closure_set(v___f_437_, 1, v_g_435_);
lean_closure_set(v___f_437_, 2, v_toSub_436_);
return v___f_437_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__3(lean_object* v_M_438_, lean_object* v_inst_439_, lean_object* v_f_440_, lean_object* v_g_441_){
_start:
{
lean_object* v_toSub_442_; lean_object* v___f_443_; 
v_toSub_442_ = lean_ctor_get(v_inst_439_, 2);
lean_inc(v_toSub_442_);
lean_dec_ref(v_inst_439_);
v___f_443_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommGroup___aux__3___redArg___lam__0), 4, 3);
lean_closure_set(v___f_443_, 0, v_f_440_);
lean_closure_set(v___f_443_, 1, v_g_441_);
lean_closure_set(v___f_443_, 2, v_toSub_442_);
return v___f_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__5___redArg___lam__0(lean_object* v_x_444_, lean_object* v_toZSMul_445_, lean_object* v_n_446_, lean_object* v___y_447_){
_start:
{
lean_object* v___x_448_; lean_object* v___x_449_; 
v___x_448_ = lean_apply_1(v_x_444_, v___y_447_);
v___x_449_ = lean_apply_2(v_toZSMul_445_, v_n_446_, v___x_448_);
return v___x_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__5___redArg(lean_object* v_inst_450_, lean_object* v_n_451_, lean_object* v_x_452_){
_start:
{
lean_object* v_toZSMul_453_; lean_object* v___f_454_; 
v_toZSMul_453_ = lean_ctor_get(v_inst_450_, 3);
lean_inc(v_toZSMul_453_);
lean_dec_ref(v_inst_450_);
v___f_454_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommGroup___aux__5___redArg___lam__0), 4, 3);
lean_closure_set(v___f_454_, 0, v_x_452_);
lean_closure_set(v___f_454_, 1, v_toZSMul_453_);
lean_closure_set(v___f_454_, 2, v_n_451_);
return v___f_454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___aux__5(lean_object* v_M_455_, lean_object* v_inst_456_, lean_object* v_n_457_, lean_object* v_x_458_){
_start:
{
lean_object* v_toZSMul_459_; lean_object* v___f_460_; 
v_toZSMul_459_ = lean_ctor_get(v_inst_456_, 3);
lean_inc(v_toZSMul_459_);
lean_dec_ref(v_inst_456_);
v___f_460_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommGroup___aux__5___redArg___lam__0), 4, 3);
lean_closure_set(v___f_460_, 0, v_x_458_);
lean_closure_set(v___f_460_, 1, v_toZSMul_459_);
lean_closure_set(v___f_460_, 2, v_n_457_);
return v___f_460_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup___redArg(lean_object* v_inst_461_){
_start:
{
lean_object* v_toAddMonoid_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_466_; lean_object* v___x_467_; 
v_toAddMonoid_462_ = lean_ctor_get(v_inst_461_, 0);
lean_inc_ref(v_toAddMonoid_462_);
v___x_463_ = lp_mathlib_AddMonoid_End_instAddCommMonoid___redArg(v_toAddMonoid_462_);
lean_inc_ref_n(v_inst_461_, 2);
v___x_464_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommGroup___aux__1___boxed), 3, 2);
lean_closure_set(v___x_464_, 0, lean_box(0));
lean_closure_set(v___x_464_, 1, v_inst_461_);
v___x_465_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommGroup___aux__3), 4, 2);
lean_closure_set(v___x_465_, 0, lean_box(0));
lean_closure_set(v___x_465_, 1, v_inst_461_);
v___x_466_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instAddCommGroup___aux__5), 4, 2);
lean_closure_set(v___x_466_, 0, lean_box(0));
lean_closure_set(v___x_466_, 1, v_inst_461_);
v___x_467_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_467_, 0, v___x_463_);
lean_ctor_set(v___x_467_, 1, v___x_464_);
lean_ctor_set(v___x_467_, 2, v___x_465_);
lean_ctor_set(v___x_467_, 3, v___x_466_);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instAddCommGroup(lean_object* v_M_468_, lean_object* v_inst_469_){
_start:
{
lean_object* v___x_470_; 
v___x_470_ = lp_mathlib_AddMonoid_End_instAddCommGroup___redArg(v_inst_469_);
return v___x_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instIntCast___redArg___lam__0(lean_object* v_inst_471_, lean_object* v_z_472_, lean_object* v___y_473_){
_start:
{
lean_object* v_toZSMul_474_; lean_object* v___x_475_; lean_object* v___x_476_; 
v_toZSMul_474_ = lean_ctor_get(v_inst_471_, 3);
lean_inc(v_toZSMul_474_);
lean_dec_ref(v_inst_471_);
v___x_475_ = lp_mathlib_OneHom_id___lam__0(v___y_473_);
v___x_476_ = lean_apply_2(v_toZSMul_474_, v_z_472_, v___x_475_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instIntCast___redArg___lam__0___boxed(lean_object* v_inst_477_, lean_object* v_z_478_, lean_object* v___y_479_){
_start:
{
lean_object* v_res_480_; 
v_res_480_ = lp_mathlib_AddMonoid_End_instIntCast___redArg___lam__0(v_inst_477_, v_z_478_, v___y_479_);
lean_dec(v___y_479_);
return v_res_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instIntCast___redArg(lean_object* v_inst_481_){
_start:
{
lean_object* v___f_482_; 
v___f_482_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instIntCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_482_, 0, v_inst_481_);
return v___f_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoid_End_instIntCast(lean_object* v_M_483_, lean_object* v_inst_484_){
_start:
{
lean_object* v___f_485_; 
v___f_485_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoid_End_instIntCast___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_485_, 0, v_inst_484_);
return v___f_485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_flip___redArg___lam__0(lean_object* v_f_486_, lean_object* v_y_487_, lean_object* v___y_488_){
_start:
{
lean_object* v___x_489_; 
v___x_489_ = lean_apply_2(v_f_486_, v___y_488_, v_y_487_);
return v___x_489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_flip___redArg(lean_object* v_f_490_){
_start:
{
lean_object* v___f_491_; 
v___f_491_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_flip___redArg___lam__0), 3, 1);
lean_closure_set(v___f_491_, 0, v_f_490_);
return v___f_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_flip(lean_object* v_M_492_, lean_object* v_N_493_, lean_object* v_P_494_, lean_object* v_mM_495_, lean_object* v_mN_496_, lean_object* v_mP_497_, lean_object* v_f_498_){
_start:
{
lean_object* v___f_499_; 
v___f_499_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_flip___redArg___lam__0), 3, 1);
lean_closure_set(v___f_499_, 0, v_f_498_);
return v___f_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_flip___boxed(lean_object* v_M_500_, lean_object* v_N_501_, lean_object* v_P_502_, lean_object* v_mM_503_, lean_object* v_mN_504_, lean_object* v_mP_505_, lean_object* v_f_506_){
_start:
{
lean_object* v_res_507_; 
v_res_507_ = lp_mathlib_MonoidHom_flip(v_M_500_, v_N_501_, v_P_502_, v_mM_503_, v_mN_504_, v_mP_505_, v_f_506_);
lean_dec_ref(v_mP_505_);
lean_dec_ref(v_mN_504_);
lean_dec_ref(v_mM_503_);
return v_res_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_flip___redArg(lean_object* v_f_508_){
_start:
{
lean_object* v___f_509_; 
v___f_509_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_flip___redArg___lam__0), 3, 1);
lean_closure_set(v___f_509_, 0, v_f_508_);
return v___f_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_flip(lean_object* v_M_510_, lean_object* v_N_511_, lean_object* v_P_512_, lean_object* v_mM_513_, lean_object* v_mN_514_, lean_object* v_mP_515_, lean_object* v_f_516_){
_start:
{
lean_object* v___f_517_; 
v___f_517_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_flip___redArg___lam__0), 3, 1);
lean_closure_set(v___f_517_, 0, v_f_516_);
return v___f_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_flip___boxed(lean_object* v_M_518_, lean_object* v_N_519_, lean_object* v_P_520_, lean_object* v_mM_521_, lean_object* v_mN_522_, lean_object* v_mP_523_, lean_object* v_f_524_){
_start:
{
lean_object* v_res_525_; 
v_res_525_ = lp_mathlib_AddMonoidHom_flip(v_M_518_, v_N_519_, v_P_520_, v_mM_521_, v_mN_522_, v_mP_523_, v_f_524_);
lean_dec_ref(v_mP_523_);
lean_dec_ref(v_mN_522_);
lean_dec_ref(v_mM_521_);
return v_res_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_eval(lean_object* v_M_529_, lean_object* v_N_530_, lean_object* v_inst_531_, lean_object* v_inst_532_){
_start:
{
lean_object* v___f_533_; 
v___f_533_ = ((lean_object*)(lp_mathlib_MonoidHom_eval___closed__1));
return v___f_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_eval___boxed(lean_object* v_M_534_, lean_object* v_N_535_, lean_object* v_inst_536_, lean_object* v_inst_537_){
_start:
{
lean_object* v_res_538_; 
v_res_538_ = lp_mathlib_MonoidHom_eval(v_M_534_, v_N_535_, v_inst_536_, v_inst_537_);
lean_dec_ref(v_inst_537_);
lean_dec_ref(v_inst_536_);
return v_res_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_eval(lean_object* v_M_539_, lean_object* v_N_540_, lean_object* v_inst_541_, lean_object* v_inst_542_){
_start:
{
lean_object* v___f_543_; 
v___f_543_ = ((lean_object*)(lp_mathlib_MonoidHom_eval___closed__1));
return v___f_543_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_eval___boxed(lean_object* v_M_544_, lean_object* v_N_545_, lean_object* v_inst_546_, lean_object* v_inst_547_){
_start:
{
lean_object* v_res_548_; 
v_res_548_ = lp_mathlib_AddMonoidHom_eval(v_M_544_, v_N_545_, v_inst_546_, v_inst_547_);
lean_dec_ref(v_inst_547_);
lean_dec_ref(v_inst_546_);
return v_res_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compHom_x27___redArg(lean_object* v_inst_549_, lean_object* v_inst_550_, lean_object* v_f_551_){
_start:
{
lean_object* v___x_552_; lean_object* v___f_553_; lean_object* v___f_554_; 
v___x_552_ = lp_mathlib_MonoidHom_eval(lean_box(0), lean_box(0), v_inst_549_, v_inst_550_);
v___f_553_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_553_, 0, v_f_551_);
lean_closure_set(v___f_553_, 1, v___x_552_);
v___f_554_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_flip___redArg___lam__0), 3, 1);
lean_closure_set(v___f_554_, 0, v___f_553_);
return v___f_554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compHom_x27___redArg___boxed(lean_object* v_inst_555_, lean_object* v_inst_556_, lean_object* v_f_557_){
_start:
{
lean_object* v_res_558_; 
v_res_558_ = lp_mathlib_MonoidHom_compHom_x27___redArg(v_inst_555_, v_inst_556_, v_f_557_);
lean_dec_ref(v_inst_556_);
lean_dec_ref(v_inst_555_);
return v_res_558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compHom_x27(lean_object* v_M_559_, lean_object* v_N_560_, lean_object* v_P_561_, lean_object* v_inst_562_, lean_object* v_inst_563_, lean_object* v_inst_564_, lean_object* v_f_565_){
_start:
{
lean_object* v___x_566_; 
v___x_566_ = lp_mathlib_MonoidHom_compHom_x27___redArg(v_inst_563_, v_inst_564_, v_f_565_);
return v___x_566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compHom_x27___boxed(lean_object* v_M_567_, lean_object* v_N_568_, lean_object* v_P_569_, lean_object* v_inst_570_, lean_object* v_inst_571_, lean_object* v_inst_572_, lean_object* v_f_573_){
_start:
{
lean_object* v_res_574_; 
v_res_574_ = lp_mathlib_MonoidHom_compHom_x27(v_M_567_, v_N_568_, v_P_569_, v_inst_570_, v_inst_571_, v_inst_572_, v_f_573_);
lean_dec_ref(v_inst_572_);
lean_dec_ref(v_inst_571_);
lean_dec_ref(v_inst_570_);
return v_res_574_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compHom_x27___redArg(lean_object* v_inst_575_, lean_object* v_inst_576_, lean_object* v_f_577_){
_start:
{
lean_object* v___x_578_; lean_object* v___f_579_; lean_object* v___f_580_; 
v___x_578_ = lp_mathlib_AddMonoidHom_eval(lean_box(0), lean_box(0), v_inst_575_, v_inst_576_);
v___f_579_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_579_, 0, v_f_577_);
lean_closure_set(v___f_579_, 1, v___x_578_);
v___f_580_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_flip___redArg___lam__0), 3, 1);
lean_closure_set(v___f_580_, 0, v___f_579_);
return v___f_580_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compHom_x27___redArg___boxed(lean_object* v_inst_581_, lean_object* v_inst_582_, lean_object* v_f_583_){
_start:
{
lean_object* v_res_584_; 
v_res_584_ = lp_mathlib_AddMonoidHom_compHom_x27___redArg(v_inst_581_, v_inst_582_, v_f_583_);
lean_dec_ref(v_inst_582_);
lean_dec_ref(v_inst_581_);
return v_res_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compHom_x27(lean_object* v_M_585_, lean_object* v_N_586_, lean_object* v_P_587_, lean_object* v_inst_588_, lean_object* v_inst_589_, lean_object* v_inst_590_, lean_object* v_f_591_){
_start:
{
lean_object* v___x_592_; 
v___x_592_ = lp_mathlib_AddMonoidHom_compHom_x27___redArg(v_inst_589_, v_inst_590_, v_f_591_);
return v___x_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compHom_x27___boxed(lean_object* v_M_593_, lean_object* v_N_594_, lean_object* v_P_595_, lean_object* v_inst_596_, lean_object* v_inst_597_, lean_object* v_inst_598_, lean_object* v_f_599_){
_start:
{
lean_object* v_res_600_; 
v_res_600_ = lp_mathlib_AddMonoidHom_compHom_x27(v_M_593_, v_N_594_, v_P_595_, v_inst_596_, v_inst_597_, v_inst_598_, v_f_599_);
lean_dec_ref(v_inst_598_);
lean_dec_ref(v_inst_597_);
lean_dec_ref(v_inst_596_);
return v_res_600_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compHom___lam__0(lean_object* v_g_601_, lean_object* v___y_602_, lean_object* v___y_603_){
_start:
{
lean_object* v___x_604_; 
v___x_604_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___y_602_, v_g_601_, v___y_603_);
return v___x_604_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compHom(lean_object* v_M_606_, lean_object* v_N_607_, lean_object* v_P_608_, lean_object* v_inst_609_, lean_object* v_inst_610_, lean_object* v_inst_611_){
_start:
{
lean_object* v___f_612_; 
v___f_612_ = ((lean_object*)(lp_mathlib_MonoidHom_compHom___closed__0));
return v___f_612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compHom___boxed(lean_object* v_M_613_, lean_object* v_N_614_, lean_object* v_P_615_, lean_object* v_inst_616_, lean_object* v_inst_617_, lean_object* v_inst_618_){
_start:
{
lean_object* v_res_619_; 
v_res_619_ = lp_mathlib_MonoidHom_compHom(v_M_613_, v_N_614_, v_P_615_, v_inst_616_, v_inst_617_, v_inst_618_);
lean_dec_ref(v_inst_618_);
lean_dec_ref(v_inst_617_);
lean_dec_ref(v_inst_616_);
return v_res_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compHom(lean_object* v_M_620_, lean_object* v_N_621_, lean_object* v_P_622_, lean_object* v_inst_623_, lean_object* v_inst_624_, lean_object* v_inst_625_){
_start:
{
lean_object* v___f_626_; 
v___f_626_ = ((lean_object*)(lp_mathlib_MonoidHom_compHom___closed__0));
return v___f_626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compHom___boxed(lean_object* v_M_627_, lean_object* v_N_628_, lean_object* v_P_629_, lean_object* v_inst_630_, lean_object* v_inst_631_, lean_object* v_inst_632_){
_start:
{
lean_object* v_res_633_; 
v_res_633_ = lp_mathlib_AddMonoidHom_compHom(v_M_627_, v_N_628_, v_P_629_, v_inst_630_, v_inst_631_, v_inst_632_);
lean_dec_ref(v_inst_632_);
lean_dec_ref(v_inst_631_);
lean_dec_ref(v_inst_630_);
return v_res_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_flipHom___redArg(lean_object* v_x_634_, lean_object* v_x_635_, lean_object* v_x_636_){
_start:
{
lean_object* v___x_637_; 
v___x_637_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_flip___boxed), 7, 6);
lean_closure_set(v___x_637_, 0, lean_box(0));
lean_closure_set(v___x_637_, 1, lean_box(0));
lean_closure_set(v___x_637_, 2, lean_box(0));
lean_closure_set(v___x_637_, 3, v_x_634_);
lean_closure_set(v___x_637_, 4, v_x_635_);
lean_closure_set(v___x_637_, 5, v_x_636_);
return v___x_637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_flipHom(lean_object* v_M_638_, lean_object* v_N_639_, lean_object* v_P_640_, lean_object* v_x_641_, lean_object* v_x_642_, lean_object* v_x_643_){
_start:
{
lean_object* v___x_644_; 
v___x_644_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_flip___boxed), 7, 6);
lean_closure_set(v___x_644_, 0, lean_box(0));
lean_closure_set(v___x_644_, 1, lean_box(0));
lean_closure_set(v___x_644_, 2, lean_box(0));
lean_closure_set(v___x_644_, 3, v_x_641_);
lean_closure_set(v___x_644_, 4, v_x_642_);
lean_closure_set(v___x_644_, 5, v_x_643_);
return v___x_644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_flipHom___redArg(lean_object* v_x_645_, lean_object* v_x_646_, lean_object* v_x_647_){
_start:
{
lean_object* v___x_648_; 
v___x_648_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_flip___boxed), 7, 6);
lean_closure_set(v___x_648_, 0, lean_box(0));
lean_closure_set(v___x_648_, 1, lean_box(0));
lean_closure_set(v___x_648_, 2, lean_box(0));
lean_closure_set(v___x_648_, 3, v_x_645_);
lean_closure_set(v___x_648_, 4, v_x_646_);
lean_closure_set(v___x_648_, 5, v_x_647_);
return v___x_648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_flipHom(lean_object* v_M_649_, lean_object* v_N_650_, lean_object* v_P_651_, lean_object* v_x_652_, lean_object* v_x_653_, lean_object* v_x_654_){
_start:
{
lean_object* v___x_655_; 
v___x_655_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_flip___boxed), 7, 6);
lean_closure_set(v___x_655_, 0, lean_box(0));
lean_closure_set(v___x_655_, 1, lean_box(0));
lean_closure_set(v___x_655_, 2, lean_box(0));
lean_closure_set(v___x_655_, 3, v_x_652_);
lean_closure_set(v___x_655_, 4, v_x_653_);
lean_closure_set(v___x_655_, 5, v_x_654_);
return v___x_655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compl_u2082___redArg(lean_object* v_inst_656_, lean_object* v_inst_657_, lean_object* v_f_658_, lean_object* v_g_659_){
_start:
{
lean_object* v___x_660_; lean_object* v___f_661_; 
v___x_660_ = lp_mathlib_MonoidHom_compHom_x27___redArg(v_inst_656_, v_inst_657_, v_g_659_);
v___f_661_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_661_, 0, v_f_658_);
lean_closure_set(v___f_661_, 1, v___x_660_);
return v___f_661_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compl_u2082___redArg___boxed(lean_object* v_inst_662_, lean_object* v_inst_663_, lean_object* v_f_664_, lean_object* v_g_665_){
_start:
{
lean_object* v_res_666_; 
v_res_666_ = lp_mathlib_MonoidHom_compl_u2082___redArg(v_inst_662_, v_inst_663_, v_f_664_, v_g_665_);
lean_dec_ref(v_inst_663_);
lean_dec_ref(v_inst_662_);
return v_res_666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compl_u2082(lean_object* v_M_667_, lean_object* v_N_668_, lean_object* v_P_669_, lean_object* v_Q_670_, lean_object* v_inst_671_, lean_object* v_inst_672_, lean_object* v_inst_673_, lean_object* v_inst_674_, lean_object* v_f_675_, lean_object* v_g_676_){
_start:
{
lean_object* v___x_677_; 
v___x_677_ = lp_mathlib_MonoidHom_compl_u2082___redArg(v_inst_672_, v_inst_673_, v_f_675_, v_g_676_);
return v___x_677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compl_u2082___boxed(lean_object* v_M_678_, lean_object* v_N_679_, lean_object* v_P_680_, lean_object* v_Q_681_, lean_object* v_inst_682_, lean_object* v_inst_683_, lean_object* v_inst_684_, lean_object* v_inst_685_, lean_object* v_f_686_, lean_object* v_g_687_){
_start:
{
lean_object* v_res_688_; 
v_res_688_ = lp_mathlib_MonoidHom_compl_u2082(v_M_678_, v_N_679_, v_P_680_, v_Q_681_, v_inst_682_, v_inst_683_, v_inst_684_, v_inst_685_, v_f_686_, v_g_687_);
lean_dec_ref(v_inst_685_);
lean_dec_ref(v_inst_684_);
lean_dec_ref(v_inst_683_);
lean_dec_ref(v_inst_682_);
return v_res_688_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compl_u2082___redArg(lean_object* v_inst_689_, lean_object* v_inst_690_, lean_object* v_f_691_, lean_object* v_g_692_){
_start:
{
lean_object* v___x_693_; lean_object* v___f_694_; 
v___x_693_ = lp_mathlib_AddMonoidHom_compHom_x27___redArg(v_inst_689_, v_inst_690_, v_g_692_);
v___f_694_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_694_, 0, v_f_691_);
lean_closure_set(v___f_694_, 1, v___x_693_);
return v___f_694_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compl_u2082___redArg___boxed(lean_object* v_inst_695_, lean_object* v_inst_696_, lean_object* v_f_697_, lean_object* v_g_698_){
_start:
{
lean_object* v_res_699_; 
v_res_699_ = lp_mathlib_AddMonoidHom_compl_u2082___redArg(v_inst_695_, v_inst_696_, v_f_697_, v_g_698_);
lean_dec_ref(v_inst_696_);
lean_dec_ref(v_inst_695_);
return v_res_699_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compl_u2082(lean_object* v_M_700_, lean_object* v_N_701_, lean_object* v_P_702_, lean_object* v_Q_703_, lean_object* v_inst_704_, lean_object* v_inst_705_, lean_object* v_inst_706_, lean_object* v_inst_707_, lean_object* v_f_708_, lean_object* v_g_709_){
_start:
{
lean_object* v___x_710_; 
v___x_710_ = lp_mathlib_AddMonoidHom_compl_u2082___redArg(v_inst_705_, v_inst_706_, v_f_708_, v_g_709_);
return v___x_710_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compl_u2082___boxed(lean_object* v_M_711_, lean_object* v_N_712_, lean_object* v_P_713_, lean_object* v_Q_714_, lean_object* v_inst_715_, lean_object* v_inst_716_, lean_object* v_inst_717_, lean_object* v_inst_718_, lean_object* v_f_719_, lean_object* v_g_720_){
_start:
{
lean_object* v_res_721_; 
v_res_721_ = lp_mathlib_AddMonoidHom_compl_u2082(v_M_711_, v_N_712_, v_P_713_, v_Q_714_, v_inst_715_, v_inst_716_, v_inst_717_, v_inst_718_, v_f_719_, v_g_720_);
lean_dec_ref(v_inst_718_);
lean_dec_ref(v_inst_717_);
lean_dec_ref(v_inst_716_);
lean_dec_ref(v_inst_715_);
return v_res_721_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compr_u2082___redArg(lean_object* v_f_722_, lean_object* v_g_723_){
_start:
{
lean_object* v___x_724_; lean_object* v___f_725_; 
v___x_724_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_compHom___lam__0), 3, 1);
lean_closure_set(v___x_724_, 0, v_g_723_);
v___f_725_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_725_, 0, v_f_722_);
lean_closure_set(v___f_725_, 1, v___x_724_);
return v___f_725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compr_u2082(lean_object* v_M_726_, lean_object* v_N_727_, lean_object* v_P_728_, lean_object* v_Q_729_, lean_object* v_inst_730_, lean_object* v_inst_731_, lean_object* v_inst_732_, lean_object* v_inst_733_, lean_object* v_f_734_, lean_object* v_g_735_){
_start:
{
lean_object* v___x_736_; 
v___x_736_ = lp_mathlib_MonoidHom_compr_u2082___redArg(v_f_734_, v_g_735_);
return v___x_736_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_compr_u2082___boxed(lean_object* v_M_737_, lean_object* v_N_738_, lean_object* v_P_739_, lean_object* v_Q_740_, lean_object* v_inst_741_, lean_object* v_inst_742_, lean_object* v_inst_743_, lean_object* v_inst_744_, lean_object* v_f_745_, lean_object* v_g_746_){
_start:
{
lean_object* v_res_747_; 
v_res_747_ = lp_mathlib_MonoidHom_compr_u2082(v_M_737_, v_N_738_, v_P_739_, v_Q_740_, v_inst_741_, v_inst_742_, v_inst_743_, v_inst_744_, v_f_745_, v_g_746_);
lean_dec_ref(v_inst_744_);
lean_dec_ref(v_inst_743_);
lean_dec_ref(v_inst_742_);
lean_dec_ref(v_inst_741_);
return v_res_747_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compr_u2082___redArg(lean_object* v_f_748_, lean_object* v_g_749_){
_start:
{
lean_object* v___x_750_; lean_object* v___f_751_; 
v___x_750_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_compHom___lam__0), 3, 1);
lean_closure_set(v___x_750_, 0, v_g_749_);
v___f_751_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_751_, 0, v_f_748_);
lean_closure_set(v___f_751_, 1, v___x_750_);
return v___f_751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compr_u2082(lean_object* v_M_752_, lean_object* v_N_753_, lean_object* v_P_754_, lean_object* v_Q_755_, lean_object* v_inst_756_, lean_object* v_inst_757_, lean_object* v_inst_758_, lean_object* v_inst_759_, lean_object* v_f_760_, lean_object* v_g_761_){
_start:
{
lean_object* v___x_762_; 
v___x_762_ = lp_mathlib_AddMonoidHom_compr_u2082___redArg(v_f_760_, v_g_761_);
return v___x_762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_compr_u2082___boxed(lean_object* v_M_763_, lean_object* v_N_764_, lean_object* v_P_765_, lean_object* v_Q_766_, lean_object* v_inst_767_, lean_object* v_inst_768_, lean_object* v_inst_769_, lean_object* v_inst_770_, lean_object* v_f_771_, lean_object* v_g_772_){
_start:
{
lean_object* v_res_773_; 
v_res_773_ = lp_mathlib_AddMonoidHom_compr_u2082(v_M_763_, v_N_764_, v_P_765_, v_Q_766_, v_inst_767_, v_inst_768_, v_inst_769_, v_inst_770_, v_f_771_, v_g_772_);
lean_dec_ref(v_inst_770_);
lean_dec_ref(v_inst_769_);
lean_dec_ref(v_inst_768_);
lean_dec_ref(v_inst_767_);
return v_res_773_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_InjSurj(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_FastInstance(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Hom_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_FastInstance(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Hom_Instances(builtin);
}
#ifdef __cplusplus
}
#endif
