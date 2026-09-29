// Lean compiler output
// Module: Mathlib.Algebra.Group.Equiv.TypeTags
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.TypeTags.Hom public import Mathlib.Algebra.Group.Equiv.Defs public import Mathlib.Algebra.Notation.Prod public import Mathlib.Tactic.Spread
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
lean_object* lp_mathlib_Multiplicative_toAdd(lean_object*);
lean_object* lp_mathlib_Multiplicative_ofAdd(lean_object*);
lean_object* lp_mathlib_Additive_toMul(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Additive_ofMul(lean_object*);
lean_object* lp_mathlib_AddMonoidHom_toMultiplicative(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidHom_toAdditive(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Additive_add___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
lean_object* lp_mathlib_Multiplicative_mul___redArg(lean_object*);
static lean_once_cell_t lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicative___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicative___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicative___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicative___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicative___lam__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicative___lam__5(lean_object*);
static const lean_closure_object lp_mathlib_AddEquiv_toMultiplicative___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddEquiv_toMultiplicative___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddEquiv_toMultiplicative___closed__0 = (const lean_object*)&lp_mathlib_AddEquiv_toMultiplicative___closed__0_value;
static const lean_closure_object lp_mathlib_AddEquiv_toMultiplicative___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddEquiv_toMultiplicative___lam__5, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddEquiv_toMultiplicative___closed__1 = (const lean_object*)&lp_mathlib_AddEquiv_toMultiplicative___closed__1_value;
static const lean_ctor_object lp_mathlib_AddEquiv_toMultiplicative___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddEquiv_toMultiplicative___closed__1_value),((lean_object*)&lp_mathlib_AddEquiv_toMultiplicative___closed__0_value)}};
static const lean_object* lp_mathlib_AddEquiv_toMultiplicative___closed__2 = (const lean_object*)&lp_mathlib_AddEquiv_toMultiplicative___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicative(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicative___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0;
static lean_once_cell_t lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditive___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditive___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditive___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditive___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditive___lam__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditive___lam__5(lean_object*);
static const lean_closure_object lp_mathlib_MulEquiv_toAdditive___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulEquiv_toAdditive___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulEquiv_toAdditive___closed__0 = (const lean_object*)&lp_mathlib_MulEquiv_toAdditive___closed__0_value;
static const lean_closure_object lp_mathlib_MulEquiv_toAdditive___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulEquiv_toAdditive___lam__5, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulEquiv_toAdditive___closed__1 = (const lean_object*)&lp_mathlib_MulEquiv_toAdditive___closed__1_value;
static const lean_ctor_object lp_mathlib_MulEquiv_toAdditive___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_MulEquiv_toAdditive___closed__1_value),((lean_object*)&lp_mathlib_MulEquiv_toAdditive___closed__0_value)}};
static const lean_object* lp_mathlib_MulEquiv_toAdditive___closed__2 = (const lean_object*)&lp_mathlib_MulEquiv_toAdditive___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditive(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditive___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight___lam__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight___lam__5(lean_object*);
static const lean_closure_object lp_mathlib_AddEquiv_toMultiplicativeRight___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddEquiv_toMultiplicativeRight___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight___closed__0 = (const lean_object*)&lp_mathlib_AddEquiv_toMultiplicativeRight___closed__0_value;
static const lean_closure_object lp_mathlib_AddEquiv_toMultiplicativeRight___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddEquiv_toMultiplicativeRight___lam__5, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight___closed__1 = (const lean_object*)&lp_mathlib_AddEquiv_toMultiplicativeRight___closed__1_value;
static const lean_ctor_object lp_mathlib_AddEquiv_toMultiplicativeRight___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddEquiv_toMultiplicativeRight___closed__1_value),((lean_object*)&lp_mathlib_AddEquiv_toMultiplicativeRight___closed__0_value)}};
static const lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight___closed__2 = (const lean_object*)&lp_mathlib_AddEquiv_toMultiplicativeRight___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditiveLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditiveLeft___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditiveLeft(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditiveLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__5(lean_object*);
static const lean_closure_object lp_mathlib_AddEquiv_toMultiplicativeLeft___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__2, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft___closed__0 = (const lean_object*)&lp_mathlib_AddEquiv_toMultiplicativeLeft___closed__0_value;
static const lean_closure_object lp_mathlib_AddEquiv_toMultiplicativeLeft___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__5, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft___closed__1 = (const lean_object*)&lp_mathlib_AddEquiv_toMultiplicativeLeft___closed__1_value;
static const lean_ctor_object lp_mathlib_AddEquiv_toMultiplicativeLeft___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddEquiv_toMultiplicativeLeft___closed__1_value),((lean_object*)&lp_mathlib_AddEquiv_toMultiplicativeLeft___closed__0_value)}};
static const lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft___closed__2 = (const lean_object*)&lp_mathlib_AddEquiv_toMultiplicativeLeft___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditiveRight___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditiveRight___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditiveRight(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditiveRight___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidEnd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidEnd___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidEnd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidEnd___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_addMonoidEnd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_addMonoidEnd___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_addMonoidEnd(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_addMonoidEnd___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_monoidEndToAdditive___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_monoidEndToAdditive___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_monoidEndToAdditive(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_monoidEndToAdditive___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidEndToMultiplicative___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidEndToMultiplicative___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidEndToMultiplicative(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_addMonoidEndToMultiplicative___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_Monoid_End___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_Monoid_End___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_Monoid_End(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_Monoid_End___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_AddMonoid_End___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_AddMonoid_End___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_AddMonoid_End(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_AddMonoid_End___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MulEquiv_piMultiplicative___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulEquiv_piMultiplicative___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piMultiplicative___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piMultiplicative___lam__1(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MulEquiv_piMultiplicative___lam__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulEquiv_piMultiplicative___lam__2___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piMultiplicative___lam__2(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MulEquiv_piMultiplicative___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulEquiv_piMultiplicative___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulEquiv_piMultiplicative___closed__0 = (const lean_object*)&lp_mathlib_MulEquiv_piMultiplicative___closed__0_value;
static const lean_closure_object lp_mathlib_MulEquiv_piMultiplicative___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulEquiv_piMultiplicative___lam__2, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulEquiv_piMultiplicative___closed__1 = (const lean_object*)&lp_mathlib_MulEquiv_piMultiplicative___closed__1_value;
static const lean_ctor_object lp_mathlib_MulEquiv_piMultiplicative___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_MulEquiv_piMultiplicative___closed__0_value),((lean_object*)&lp_mathlib_MulEquiv_piMultiplicative___closed__1_value)}};
static const lean_object* lp_mathlib_MulEquiv_piMultiplicative___closed__2 = (const lean_object*)&lp_mathlib_MulEquiv_piMultiplicative___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piMultiplicative(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piMultiplicative___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_funMultiplicative___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_funMultiplicative___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_funMultiplicative___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_funMultiplicative(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddEquiv_piAdditive___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddEquiv_piAdditive___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piAdditive___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piAdditive___lam__1(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddEquiv_piAdditive___lam__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddEquiv_piAdditive___lam__2___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piAdditive___lam__2(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddEquiv_piAdditive___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddEquiv_piAdditive___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddEquiv_piAdditive___closed__0 = (const lean_object*)&lp_mathlib_AddEquiv_piAdditive___closed__0_value;
static const lean_closure_object lp_mathlib_AddEquiv_piAdditive___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddEquiv_piAdditive___lam__2, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddEquiv_piAdditive___closed__1 = (const lean_object*)&lp_mathlib_AddEquiv_piAdditive___closed__1_value;
static const lean_ctor_object lp_mathlib_AddEquiv_piAdditive___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddEquiv_piAdditive___closed__0_value),((lean_object*)&lp_mathlib_AddEquiv_piAdditive___closed__1_value)}};
static const lean_object* lp_mathlib_AddEquiv_piAdditive___closed__2 = (const lean_object*)&lp_mathlib_AddEquiv_piAdditive___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piAdditive(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piAdditive___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_funAdditive___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_funAdditive(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddEquiv_additiveMultiplicative___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddEquiv_additiveMultiplicative___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_additiveMultiplicative___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_additiveMultiplicative(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_multiplicativeAdditive___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_multiplicativeAdditive(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMultiplicative__toAdditive___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMultiplicative__toAdditive(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAdditive__toMultiplicative___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAdditive__toMultiplicative(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MulEquiv_prodMultiplicative___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulEquiv_prodMultiplicative___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodMultiplicative___lam__0(lean_object*);
static lean_once_cell_t lp_mathlib_MulEquiv_prodMultiplicative___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulEquiv_prodMultiplicative___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodMultiplicative___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_MulEquiv_prodMultiplicative___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulEquiv_prodMultiplicative___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulEquiv_prodMultiplicative___closed__0 = (const lean_object*)&lp_mathlib_MulEquiv_prodMultiplicative___closed__0_value;
static const lean_closure_object lp_mathlib_MulEquiv_prodMultiplicative___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulEquiv_prodMultiplicative___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulEquiv_prodMultiplicative___closed__1 = (const lean_object*)&lp_mathlib_MulEquiv_prodMultiplicative___closed__1_value;
static const lean_ctor_object lp_mathlib_MulEquiv_prodMultiplicative___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_MulEquiv_prodMultiplicative___closed__0_value),((lean_object*)&lp_mathlib_MulEquiv_prodMultiplicative___closed__1_value)}};
static const lean_object* lp_mathlib_MulEquiv_prodMultiplicative___closed__2 = (const lean_object*)&lp_mathlib_MulEquiv_prodMultiplicative___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodMultiplicative(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodMultiplicative___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_AddEquiv_prodAdditive___lam__0___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddEquiv_prodAdditive___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAdditive___lam__0(lean_object*);
static lean_once_cell_t lp_mathlib_AddEquiv_prodAdditive___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_AddEquiv_prodAdditive___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAdditive___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_AddEquiv_prodAdditive___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddEquiv_prodAdditive___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddEquiv_prodAdditive___closed__0 = (const lean_object*)&lp_mathlib_AddEquiv_prodAdditive___closed__0_value;
static const lean_closure_object lp_mathlib_AddEquiv_prodAdditive___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddEquiv_prodAdditive___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddEquiv_prodAdditive___closed__1 = (const lean_object*)&lp_mathlib_AddEquiv_prodAdditive___closed__1_value;
static const lean_ctor_object lp_mathlib_AddEquiv_prodAdditive___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_AddEquiv_prodAdditive___closed__0_value),((lean_object*)&lp_mathlib_AddEquiv_prodAdditive___closed__1_value)}};
static const lean_object* lp_mathlib_AddEquiv_prodAdditive___closed__2 = (const lean_object*)&lp_mathlib_AddEquiv_prodAdditive___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAdditive(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAdditive___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0(void){
_start:
{
lean_object* v___x_1_; 
v___x_1_ = lp_mathlib_Multiplicative_ofAdd(lean_box(0));
return v___x_1_;
}
}
static lean_object* _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1(void){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lp_mathlib_Multiplicative_toAdd(lean_box(0));
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicative___lam__0(lean_object* v_f_3_, lean_object* v_x_4_){
_start:
{
lean_object* v___x_5_; lean_object* v_toFun_6_; lean_object* v_toFun_7_; lean_object* v___x_8_; lean_object* v_toFun_9_; lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; 
v___x_5_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0);
v_toFun_6_ = lean_ctor_get(v___x_5_, 0);
v_toFun_7_ = lean_ctor_get(v_f_3_, 0);
lean_inc(v_toFun_7_);
lean_dec_ref(v_f_3_);
v___x_8_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1);
v_toFun_9_ = lean_ctor_get(v___x_8_, 0);
lean_inc(v_toFun_6_);
v___x_10_ = lean_apply_1(v_toFun_6_, v_x_4_);
v___x_11_ = lean_apply_1(v_toFun_7_, v___x_10_);
lean_inc(v_toFun_9_);
v___x_12_ = lean_apply_1(v_toFun_9_, v___x_11_);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicative___lam__1(lean_object* v_f_13_, lean_object* v_x_14_){
_start:
{
lean_object* v___x_15_; lean_object* v_toFun_16_; lean_object* v___x_17_; lean_object* v_toFun_18_; lean_object* v___x_19_; lean_object* v_toFun_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; 
v___x_15_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0);
v_toFun_16_ = lean_ctor_get(v___x_15_, 0);
v___x_17_ = lp_mathlib_Equiv_symm___redArg(v_f_13_);
v_toFun_18_ = lean_ctor_get(v___x_17_, 0);
lean_inc(v_toFun_18_);
lean_dec_ref(v___x_17_);
v___x_19_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1);
v_toFun_20_ = lean_ctor_get(v___x_19_, 0);
lean_inc(v_toFun_16_);
v___x_21_ = lean_apply_1(v_toFun_16_, v_x_14_);
v___x_22_ = lean_apply_1(v_toFun_18_, v___x_21_);
lean_inc(v_toFun_20_);
v___x_23_ = lean_apply_1(v_toFun_20_, v___x_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicative___lam__2(lean_object* v_f_24_){
_start:
{
lean_object* v___f_25_; lean_object* v___f_26_; lean_object* v___x_27_; 
lean_inc_ref(v_f_24_);
v___f_25_ = lean_alloc_closure((void*)(lp_mathlib_AddEquiv_toMultiplicative___lam__0), 2, 1);
lean_closure_set(v___f_25_, 0, v_f_24_);
v___f_26_ = lean_alloc_closure((void*)(lp_mathlib_AddEquiv_toMultiplicative___lam__1), 2, 1);
lean_closure_set(v___f_26_, 0, v_f_24_);
v___x_27_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_27_, 0, v___f_25_);
lean_ctor_set(v___x_27_, 1, v___f_26_);
return v___x_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicative___lam__3(lean_object* v_f_28_, lean_object* v_x_29_){
_start:
{
lean_object* v___x_30_; lean_object* v_toFun_31_; lean_object* v_toFun_32_; lean_object* v___x_33_; lean_object* v_toFun_34_; lean_object* v___x_35_; lean_object* v___x_36_; lean_object* v___x_37_; 
v___x_30_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1);
v_toFun_31_ = lean_ctor_get(v___x_30_, 0);
v_toFun_32_ = lean_ctor_get(v_f_28_, 0);
lean_inc(v_toFun_32_);
lean_dec_ref(v_f_28_);
v___x_33_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0);
v_toFun_34_ = lean_ctor_get(v___x_33_, 0);
lean_inc(v_toFun_31_);
v___x_35_ = lean_apply_1(v_toFun_31_, v_x_29_);
v___x_36_ = lean_apply_1(v_toFun_32_, v___x_35_);
lean_inc(v_toFun_34_);
v___x_37_ = lean_apply_1(v_toFun_34_, v___x_36_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicative___lam__4(lean_object* v_f_38_, lean_object* v_x_39_){
_start:
{
lean_object* v___x_40_; lean_object* v_toFun_41_; lean_object* v___x_42_; lean_object* v_toFun_43_; lean_object* v___x_44_; lean_object* v_toFun_45_; lean_object* v___x_46_; lean_object* v___x_47_; lean_object* v___x_48_; 
v___x_40_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1);
v_toFun_41_ = lean_ctor_get(v___x_40_, 0);
v___x_42_ = lp_mathlib_Equiv_symm___redArg(v_f_38_);
v_toFun_43_ = lean_ctor_get(v___x_42_, 0);
lean_inc(v_toFun_43_);
lean_dec_ref(v___x_42_);
v___x_44_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0);
v_toFun_45_ = lean_ctor_get(v___x_44_, 0);
lean_inc(v_toFun_41_);
v___x_46_ = lean_apply_1(v_toFun_41_, v_x_39_);
v___x_47_ = lean_apply_1(v_toFun_43_, v___x_46_);
lean_inc(v_toFun_45_);
v___x_48_ = lean_apply_1(v_toFun_45_, v___x_47_);
return v___x_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicative___lam__5(lean_object* v_f_49_){
_start:
{
lean_object* v___f_50_; lean_object* v___f_51_; lean_object* v___x_52_; 
lean_inc_ref(v_f_49_);
v___f_50_ = lean_alloc_closure((void*)(lp_mathlib_AddEquiv_toMultiplicative___lam__3), 2, 1);
lean_closure_set(v___f_50_, 0, v_f_49_);
v___f_51_ = lean_alloc_closure((void*)(lp_mathlib_AddEquiv_toMultiplicative___lam__4), 2, 1);
lean_closure_set(v___f_51_, 0, v_f_49_);
v___x_52_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_52_, 0, v___f_50_);
lean_ctor_set(v___x_52_, 1, v___f_51_);
return v___x_52_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicative(lean_object* v_G_58_, lean_object* v_H_59_, lean_object* v_inst_60_, lean_object* v_inst_61_){
_start:
{
lean_object* v___x_62_; 
v___x_62_ = ((lean_object*)(lp_mathlib_AddEquiv_toMultiplicative___closed__2));
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicative___boxed(lean_object* v_G_63_, lean_object* v_H_64_, lean_object* v_inst_65_, lean_object* v_inst_66_){
_start:
{
lean_object* v_res_67_; 
v_res_67_ = lp_mathlib_AddEquiv_toMultiplicative(v_G_63_, v_H_64_, v_inst_65_, v_inst_66_);
lean_dec(v_inst_66_);
lean_dec(v_inst_65_);
return v_res_67_;
}
}
static lean_object* _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0(void){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lp_mathlib_Additive_ofMul(lean_box(0));
return v___x_68_;
}
}
static lean_object* _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1(void){
_start:
{
lean_object* v___x_69_; 
v___x_69_ = lp_mathlib_Additive_toMul(lean_box(0));
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditive___lam__0(lean_object* v_f_70_, lean_object* v_x_71_){
_start:
{
lean_object* v___x_72_; lean_object* v_toFun_73_; lean_object* v_toFun_74_; lean_object* v___x_75_; lean_object* v_toFun_76_; lean_object* v___x_77_; lean_object* v___x_78_; lean_object* v___x_79_; 
v___x_72_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0);
v_toFun_73_ = lean_ctor_get(v___x_72_, 0);
v_toFun_74_ = lean_ctor_get(v_f_70_, 0);
lean_inc(v_toFun_74_);
lean_dec_ref(v_f_70_);
v___x_75_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1);
v_toFun_76_ = lean_ctor_get(v___x_75_, 0);
lean_inc(v_toFun_73_);
v___x_77_ = lean_apply_1(v_toFun_73_, v_x_71_);
v___x_78_ = lean_apply_1(v_toFun_74_, v___x_77_);
lean_inc(v_toFun_76_);
v___x_79_ = lean_apply_1(v_toFun_76_, v___x_78_);
return v___x_79_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditive___lam__1(lean_object* v_f_80_, lean_object* v_x_81_){
_start:
{
lean_object* v___x_82_; lean_object* v_toFun_83_; lean_object* v___x_84_; lean_object* v_toFun_85_; lean_object* v___x_86_; lean_object* v_toFun_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; 
v___x_82_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0);
v_toFun_83_ = lean_ctor_get(v___x_82_, 0);
v___x_84_ = lp_mathlib_Equiv_symm___redArg(v_f_80_);
v_toFun_85_ = lean_ctor_get(v___x_84_, 0);
lean_inc(v_toFun_85_);
lean_dec_ref(v___x_84_);
v___x_86_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1);
v_toFun_87_ = lean_ctor_get(v___x_86_, 0);
lean_inc(v_toFun_83_);
v___x_88_ = lean_apply_1(v_toFun_83_, v_x_81_);
v___x_89_ = lean_apply_1(v_toFun_85_, v___x_88_);
lean_inc(v_toFun_87_);
v___x_90_ = lean_apply_1(v_toFun_87_, v___x_89_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditive___lam__2(lean_object* v_f_91_){
_start:
{
lean_object* v___f_92_; lean_object* v___f_93_; lean_object* v___x_94_; 
lean_inc_ref(v_f_91_);
v___f_92_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_toAdditive___lam__0), 2, 1);
lean_closure_set(v___f_92_, 0, v_f_91_);
v___f_93_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_toAdditive___lam__1), 2, 1);
lean_closure_set(v___f_93_, 0, v_f_91_);
v___x_94_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_94_, 0, v___f_92_);
lean_ctor_set(v___x_94_, 1, v___f_93_);
return v___x_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditive___lam__3(lean_object* v_f_95_, lean_object* v_x_96_){
_start:
{
lean_object* v___x_97_; lean_object* v_toFun_98_; lean_object* v_toFun_99_; lean_object* v___x_100_; lean_object* v_toFun_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; 
v___x_97_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1);
v_toFun_98_ = lean_ctor_get(v___x_97_, 0);
v_toFun_99_ = lean_ctor_get(v_f_95_, 0);
lean_inc(v_toFun_99_);
lean_dec_ref(v_f_95_);
v___x_100_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0);
v_toFun_101_ = lean_ctor_get(v___x_100_, 0);
lean_inc(v_toFun_98_);
v___x_102_ = lean_apply_1(v_toFun_98_, v_x_96_);
v___x_103_ = lean_apply_1(v_toFun_99_, v___x_102_);
lean_inc(v_toFun_101_);
v___x_104_ = lean_apply_1(v_toFun_101_, v___x_103_);
return v___x_104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditive___lam__4(lean_object* v_f_105_, lean_object* v_x_106_){
_start:
{
lean_object* v___x_107_; lean_object* v_toFun_108_; lean_object* v___x_109_; lean_object* v_toFun_110_; lean_object* v___x_111_; lean_object* v_toFun_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; 
v___x_107_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1);
v_toFun_108_ = lean_ctor_get(v___x_107_, 0);
v___x_109_ = lp_mathlib_Equiv_symm___redArg(v_f_105_);
v_toFun_110_ = lean_ctor_get(v___x_109_, 0);
lean_inc(v_toFun_110_);
lean_dec_ref(v___x_109_);
v___x_111_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0);
v_toFun_112_ = lean_ctor_get(v___x_111_, 0);
lean_inc(v_toFun_108_);
v___x_113_ = lean_apply_1(v_toFun_108_, v_x_106_);
v___x_114_ = lean_apply_1(v_toFun_110_, v___x_113_);
lean_inc(v_toFun_112_);
v___x_115_ = lean_apply_1(v_toFun_112_, v___x_114_);
return v___x_115_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditive___lam__5(lean_object* v_f_116_){
_start:
{
lean_object* v___f_117_; lean_object* v___f_118_; lean_object* v___x_119_; 
lean_inc_ref(v_f_116_);
v___f_117_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_toAdditive___lam__3), 2, 1);
lean_closure_set(v___f_117_, 0, v_f_116_);
v___f_118_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_toAdditive___lam__4), 2, 1);
lean_closure_set(v___f_118_, 0, v_f_116_);
v___x_119_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_119_, 0, v___f_117_);
lean_ctor_set(v___x_119_, 1, v___f_118_);
return v___x_119_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditive(lean_object* v_G_125_, lean_object* v_H_126_, lean_object* v_inst_127_, lean_object* v_inst_128_){
_start:
{
lean_object* v___x_129_; 
v___x_129_ = ((lean_object*)(lp_mathlib_MulEquiv_toAdditive___closed__2));
return v___x_129_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditive___boxed(lean_object* v_G_130_, lean_object* v_H_131_, lean_object* v_inst_132_, lean_object* v_inst_133_){
_start:
{
lean_object* v_res_134_; 
v_res_134_ = lp_mathlib_MulEquiv_toAdditive(v_G_130_, v_H_131_, v_inst_132_, v_inst_133_);
lean_dec(v_inst_133_);
lean_dec(v_inst_132_);
return v_res_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight___lam__0(lean_object* v_f_135_, lean_object* v_x_136_){
_start:
{
lean_object* v___x_137_; lean_object* v_toFun_138_; lean_object* v_toFun_139_; lean_object* v___x_140_; lean_object* v_toFun_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
v___x_137_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1);
v_toFun_138_ = lean_ctor_get(v___x_137_, 0);
v_toFun_139_ = lean_ctor_get(v_f_135_, 0);
lean_inc(v_toFun_139_);
lean_dec_ref(v_f_135_);
v___x_140_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1);
v_toFun_141_ = lean_ctor_get(v___x_140_, 0);
lean_inc(v_toFun_138_);
v___x_142_ = lean_apply_1(v_toFun_138_, v_x_136_);
v___x_143_ = lean_apply_1(v_toFun_139_, v___x_142_);
lean_inc(v_toFun_141_);
v___x_144_ = lean_apply_1(v_toFun_141_, v___x_143_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight___lam__1(lean_object* v_f_145_, lean_object* v_x_146_){
_start:
{
lean_object* v___x_147_; lean_object* v_toFun_148_; lean_object* v___x_149_; lean_object* v_toFun_150_; lean_object* v___x_151_; lean_object* v_toFun_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; 
v___x_147_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0);
v_toFun_148_ = lean_ctor_get(v___x_147_, 0);
v___x_149_ = lp_mathlib_Equiv_symm___redArg(v_f_145_);
v_toFun_150_ = lean_ctor_get(v___x_149_, 0);
lean_inc(v_toFun_150_);
lean_dec_ref(v___x_149_);
v___x_151_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0);
v_toFun_152_ = lean_ctor_get(v___x_151_, 0);
lean_inc(v_toFun_148_);
v___x_153_ = lean_apply_1(v_toFun_148_, v_x_146_);
v___x_154_ = lean_apply_1(v_toFun_150_, v___x_153_);
lean_inc(v_toFun_152_);
v___x_155_ = lean_apply_1(v_toFun_152_, v___x_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight___lam__2(lean_object* v_f_156_){
_start:
{
lean_object* v___f_157_; lean_object* v___f_158_; lean_object* v___x_159_; 
lean_inc_ref(v_f_156_);
v___f_157_ = lean_alloc_closure((void*)(lp_mathlib_AddEquiv_toMultiplicativeRight___lam__0), 2, 1);
lean_closure_set(v___f_157_, 0, v_f_156_);
v___f_158_ = lean_alloc_closure((void*)(lp_mathlib_AddEquiv_toMultiplicativeRight___lam__1), 2, 1);
lean_closure_set(v___f_158_, 0, v_f_156_);
v___x_159_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_159_, 0, v___f_157_);
lean_ctor_set(v___x_159_, 1, v___f_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight___lam__3(lean_object* v_f_160_, lean_object* v_x_161_){
_start:
{
lean_object* v___x_162_; lean_object* v_toFun_163_; lean_object* v_toFun_164_; lean_object* v___x_165_; lean_object* v_toFun_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; 
v___x_162_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0);
v_toFun_163_ = lean_ctor_get(v___x_162_, 0);
v_toFun_164_ = lean_ctor_get(v_f_160_, 0);
lean_inc(v_toFun_164_);
lean_dec_ref(v_f_160_);
v___x_165_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0);
v_toFun_166_ = lean_ctor_get(v___x_165_, 0);
lean_inc(v_toFun_163_);
v___x_167_ = lean_apply_1(v_toFun_163_, v_x_161_);
v___x_168_ = lean_apply_1(v_toFun_164_, v___x_167_);
lean_inc(v_toFun_166_);
v___x_169_ = lean_apply_1(v_toFun_166_, v___x_168_);
return v___x_169_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight___lam__4(lean_object* v_f_170_, lean_object* v_x_171_){
_start:
{
lean_object* v___x_172_; lean_object* v_toFun_173_; lean_object* v___x_174_; lean_object* v_toFun_175_; lean_object* v___x_176_; lean_object* v_toFun_177_; lean_object* v___x_178_; lean_object* v___x_179_; lean_object* v___x_180_; 
v___x_172_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1);
v_toFun_173_ = lean_ctor_get(v___x_172_, 0);
v___x_174_ = lp_mathlib_Equiv_symm___redArg(v_f_170_);
v_toFun_175_ = lean_ctor_get(v___x_174_, 0);
lean_inc(v_toFun_175_);
lean_dec_ref(v___x_174_);
v___x_176_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1);
v_toFun_177_ = lean_ctor_get(v___x_176_, 0);
lean_inc(v_toFun_173_);
v___x_178_ = lean_apply_1(v_toFun_173_, v_x_171_);
v___x_179_ = lean_apply_1(v_toFun_175_, v___x_178_);
lean_inc(v_toFun_177_);
v___x_180_ = lean_apply_1(v_toFun_177_, v___x_179_);
return v___x_180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight___lam__5(lean_object* v_f_181_){
_start:
{
lean_object* v___f_182_; lean_object* v___f_183_; lean_object* v___x_184_; 
lean_inc_ref(v_f_181_);
v___f_182_ = lean_alloc_closure((void*)(lp_mathlib_AddEquiv_toMultiplicativeRight___lam__3), 2, 1);
lean_closure_set(v___f_182_, 0, v_f_181_);
v___f_183_ = lean_alloc_closure((void*)(lp_mathlib_AddEquiv_toMultiplicativeRight___lam__4), 2, 1);
lean_closure_set(v___f_183_, 0, v_f_181_);
v___x_184_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_184_, 0, v___f_182_);
lean_ctor_set(v___x_184_, 1, v___f_183_);
return v___x_184_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight(lean_object* v_G_190_, lean_object* v_H_191_, lean_object* v_inst_192_, lean_object* v_inst_193_){
_start:
{
lean_object* v___x_194_; 
v___x_194_ = ((lean_object*)(lp_mathlib_AddEquiv_toMultiplicativeRight___closed__2));
return v___x_194_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeRight___boxed(lean_object* v_G_195_, lean_object* v_H_196_, lean_object* v_inst_197_, lean_object* v_inst_198_){
_start:
{
lean_object* v_res_199_; 
v_res_199_ = lp_mathlib_AddEquiv_toMultiplicativeRight(v_G_195_, v_H_196_, v_inst_197_, v_inst_198_);
lean_dec(v_inst_198_);
lean_dec(v_inst_197_);
return v_res_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditiveLeft___redArg(lean_object* v_inst_200_, lean_object* v_inst_201_){
_start:
{
lean_object* v___x_202_; lean_object* v___x_203_; 
v___x_202_ = lp_mathlib_AddEquiv_toMultiplicativeRight(lean_box(0), lean_box(0), v_inst_200_, v_inst_201_);
v___x_203_ = lp_mathlib_Equiv_symm___redArg(v___x_202_);
return v___x_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditiveLeft___redArg___boxed(lean_object* v_inst_204_, lean_object* v_inst_205_){
_start:
{
lean_object* v_res_206_; 
v_res_206_ = lp_mathlib_MulEquiv_toAdditiveLeft___redArg(v_inst_204_, v_inst_205_);
lean_dec(v_inst_205_);
lean_dec(v_inst_204_);
return v_res_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditiveLeft(lean_object* v_G_207_, lean_object* v_H_208_, lean_object* v_inst_209_, lean_object* v_inst_210_){
_start:
{
lean_object* v___x_211_; lean_object* v___x_212_; 
v___x_211_ = lp_mathlib_AddEquiv_toMultiplicativeRight(lean_box(0), lean_box(0), v_inst_209_, v_inst_210_);
v___x_212_ = lp_mathlib_Equiv_symm___redArg(v___x_211_);
return v___x_212_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditiveLeft___boxed(lean_object* v_G_213_, lean_object* v_H_214_, lean_object* v_inst_215_, lean_object* v_inst_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_mathlib_MulEquiv_toAdditiveLeft(v_G_213_, v_H_214_, v_inst_215_, v_inst_216_);
lean_dec(v_inst_216_);
lean_dec(v_inst_215_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__0(lean_object* v_f_218_, lean_object* v_x_219_){
_start:
{
lean_object* v___x_220_; lean_object* v_toFun_221_; lean_object* v_toFun_222_; lean_object* v___x_223_; lean_object* v_toFun_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; 
v___x_220_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0);
v_toFun_221_ = lean_ctor_get(v___x_220_, 0);
v_toFun_222_ = lean_ctor_get(v_f_218_, 0);
lean_inc(v_toFun_222_);
lean_dec_ref(v_f_218_);
v___x_223_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0);
v_toFun_224_ = lean_ctor_get(v___x_223_, 0);
lean_inc(v_toFun_221_);
v___x_225_ = lean_apply_1(v_toFun_221_, v_x_219_);
v___x_226_ = lean_apply_1(v_toFun_222_, v___x_225_);
lean_inc(v_toFun_224_);
v___x_227_ = lean_apply_1(v_toFun_224_, v___x_226_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__1(lean_object* v_f_228_, lean_object* v_x_229_){
_start:
{
lean_object* v___x_230_; lean_object* v_toFun_231_; lean_object* v___x_232_; lean_object* v_toFun_233_; lean_object* v___x_234_; lean_object* v_toFun_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; 
v___x_230_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1);
v_toFun_231_ = lean_ctor_get(v___x_230_, 0);
v___x_232_ = lp_mathlib_Equiv_symm___redArg(v_f_228_);
v_toFun_233_ = lean_ctor_get(v___x_232_, 0);
lean_inc(v_toFun_233_);
lean_dec_ref(v___x_232_);
v___x_234_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1);
v_toFun_235_ = lean_ctor_get(v___x_234_, 0);
lean_inc(v_toFun_231_);
v___x_236_ = lean_apply_1(v_toFun_231_, v_x_229_);
v___x_237_ = lean_apply_1(v_toFun_233_, v___x_236_);
lean_inc(v_toFun_235_);
v___x_238_ = lean_apply_1(v_toFun_235_, v___x_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__2(lean_object* v_f_239_){
_start:
{
lean_object* v___f_240_; lean_object* v___f_241_; lean_object* v___x_242_; 
lean_inc_ref(v_f_239_);
v___f_240_ = lean_alloc_closure((void*)(lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__0), 2, 1);
lean_closure_set(v___f_240_, 0, v_f_239_);
v___f_241_ = lean_alloc_closure((void*)(lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__1), 2, 1);
lean_closure_set(v___f_241_, 0, v_f_239_);
v___x_242_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_242_, 0, v___f_240_);
lean_ctor_set(v___x_242_, 1, v___f_241_);
return v___x_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__3(lean_object* v_f_243_, lean_object* v_x_244_){
_start:
{
lean_object* v___x_245_; lean_object* v_toFun_246_; lean_object* v_toFun_247_; lean_object* v___x_248_; lean_object* v_toFun_249_; lean_object* v___x_250_; lean_object* v___x_251_; lean_object* v___x_252_; 
v___x_245_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1);
v_toFun_246_ = lean_ctor_get(v___x_245_, 0);
v_toFun_247_ = lean_ctor_get(v_f_243_, 0);
lean_inc(v_toFun_247_);
lean_dec_ref(v_f_243_);
v___x_248_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1);
v_toFun_249_ = lean_ctor_get(v___x_248_, 0);
lean_inc(v_toFun_246_);
v___x_250_ = lean_apply_1(v_toFun_246_, v_x_244_);
v___x_251_ = lean_apply_1(v_toFun_247_, v___x_250_);
lean_inc(v_toFun_249_);
v___x_252_ = lean_apply_1(v_toFun_249_, v___x_251_);
return v___x_252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__4(lean_object* v_f_253_, lean_object* v_x_254_){
_start:
{
lean_object* v___x_255_; lean_object* v_toFun_256_; lean_object* v___x_257_; lean_object* v_toFun_258_; lean_object* v___x_259_; lean_object* v_toFun_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; 
v___x_255_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0);
v_toFun_256_ = lean_ctor_get(v___x_255_, 0);
v___x_257_ = lp_mathlib_Equiv_symm___redArg(v_f_253_);
v_toFun_258_ = lean_ctor_get(v___x_257_, 0);
lean_inc(v_toFun_258_);
lean_dec_ref(v___x_257_);
v___x_259_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0);
v_toFun_260_ = lean_ctor_get(v___x_259_, 0);
lean_inc(v_toFun_256_);
v___x_261_ = lean_apply_1(v_toFun_256_, v_x_254_);
v___x_262_ = lean_apply_1(v_toFun_258_, v___x_261_);
lean_inc(v_toFun_260_);
v___x_263_ = lean_apply_1(v_toFun_260_, v___x_262_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__5(lean_object* v_f_264_){
_start:
{
lean_object* v___f_265_; lean_object* v___f_266_; lean_object* v___x_267_; 
lean_inc_ref(v_f_264_);
v___f_265_ = lean_alloc_closure((void*)(lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__3), 2, 1);
lean_closure_set(v___f_265_, 0, v_f_264_);
v___f_266_ = lean_alloc_closure((void*)(lp_mathlib_AddEquiv_toMultiplicativeLeft___lam__4), 2, 1);
lean_closure_set(v___f_266_, 0, v_f_264_);
v___x_267_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_267_, 0, v___f_265_);
lean_ctor_set(v___x_267_, 1, v___f_266_);
return v___x_267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft(lean_object* v_G_273_, lean_object* v_H_274_, lean_object* v_inst_275_, lean_object* v_inst_276_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = ((lean_object*)(lp_mathlib_AddEquiv_toMultiplicativeLeft___closed__2));
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toMultiplicativeLeft___boxed(lean_object* v_G_278_, lean_object* v_H_279_, lean_object* v_inst_280_, lean_object* v_inst_281_){
_start:
{
lean_object* v_res_282_; 
v_res_282_ = lp_mathlib_AddEquiv_toMultiplicativeLeft(v_G_278_, v_H_279_, v_inst_280_, v_inst_281_);
lean_dec(v_inst_281_);
lean_dec(v_inst_280_);
return v_res_282_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditiveRight___redArg(lean_object* v_inst_283_, lean_object* v_inst_284_){
_start:
{
lean_object* v___x_285_; lean_object* v___x_286_; 
v___x_285_ = lp_mathlib_AddEquiv_toMultiplicativeLeft(lean_box(0), lean_box(0), v_inst_283_, v_inst_284_);
v___x_286_ = lp_mathlib_Equiv_symm___redArg(v___x_285_);
return v___x_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditiveRight___redArg___boxed(lean_object* v_inst_287_, lean_object* v_inst_288_){
_start:
{
lean_object* v_res_289_; 
v_res_289_ = lp_mathlib_MulEquiv_toAdditiveRight___redArg(v_inst_287_, v_inst_288_);
lean_dec(v_inst_288_);
lean_dec(v_inst_287_);
return v_res_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditiveRight(lean_object* v_G_290_, lean_object* v_H_291_, lean_object* v_inst_292_, lean_object* v_inst_293_){
_start:
{
lean_object* v___x_294_; lean_object* v___x_295_; 
v___x_294_ = lp_mathlib_AddEquiv_toMultiplicativeLeft(lean_box(0), lean_box(0), v_inst_292_, v_inst_293_);
v___x_295_ = lp_mathlib_Equiv_symm___redArg(v___x_294_);
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toAdditiveRight___boxed(lean_object* v_G_296_, lean_object* v_H_297_, lean_object* v_inst_298_, lean_object* v_inst_299_){
_start:
{
lean_object* v_res_300_; 
v_res_300_ = lp_mathlib_MulEquiv_toAdditiveRight(v_G_296_, v_H_297_, v_inst_298_, v_inst_299_);
lean_dec(v_inst_299_);
lean_dec(v_inst_298_);
return v_res_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidEnd___redArg(lean_object* v_inst_301_){
_start:
{
lean_object* v___x_302_; 
v___x_302_ = lp_mathlib_MonoidHom_toAdditive(lean_box(0), lean_box(0), v_inst_301_, v_inst_301_);
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidEnd___redArg___boxed(lean_object* v_inst_303_){
_start:
{
lean_object* v_res_304_; 
v_res_304_ = lp_mathlib_MulEquiv_monoidEnd___redArg(v_inst_303_);
lean_dec_ref(v_inst_303_);
return v_res_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidEnd(lean_object* v_M_305_, lean_object* v_inst_306_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = lp_mathlib_MonoidHom_toAdditive(lean_box(0), lean_box(0), v_inst_306_, v_inst_306_);
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_monoidEnd___boxed(lean_object* v_M_308_, lean_object* v_inst_309_){
_start:
{
lean_object* v_res_310_; 
v_res_310_ = lp_mathlib_MulEquiv_monoidEnd(v_M_308_, v_inst_309_);
lean_dec_ref(v_inst_309_);
return v_res_310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_addMonoidEnd___redArg(lean_object* v_inst_311_){
_start:
{
lean_object* v___x_312_; 
v___x_312_ = lp_mathlib_AddMonoidHom_toMultiplicative(lean_box(0), lean_box(0), v_inst_311_, v_inst_311_);
return v___x_312_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_addMonoidEnd___redArg___boxed(lean_object* v_inst_313_){
_start:
{
lean_object* v_res_314_; 
v_res_314_ = lp_mathlib_MulEquiv_addMonoidEnd___redArg(v_inst_313_);
lean_dec_ref(v_inst_313_);
return v_res_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_addMonoidEnd(lean_object* v_A_315_, lean_object* v_inst_316_){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = lp_mathlib_AddMonoidHom_toMultiplicative(lean_box(0), lean_box(0), v_inst_316_, v_inst_316_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_addMonoidEnd___boxed(lean_object* v_A_318_, lean_object* v_inst_319_){
_start:
{
lean_object* v_res_320_; 
v_res_320_ = lp_mathlib_MulEquiv_addMonoidEnd(v_A_318_, v_inst_319_);
lean_dec_ref(v_inst_319_);
return v_res_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_monoidEndToAdditive___redArg(lean_object* v_inst_321_){
_start:
{
lean_object* v___x_322_; 
v___x_322_ = lp_mathlib_MonoidHom_toAdditive(lean_box(0), lean_box(0), v_inst_321_, v_inst_321_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_monoidEndToAdditive___redArg___boxed(lean_object* v_inst_323_){
_start:
{
lean_object* v_res_324_; 
v_res_324_ = lp_mathlib_monoidEndToAdditive___redArg(v_inst_323_);
lean_dec_ref(v_inst_323_);
return v_res_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_monoidEndToAdditive(lean_object* v_M_325_, lean_object* v_inst_326_){
_start:
{
lean_object* v___x_327_; 
v___x_327_ = lp_mathlib_MonoidHom_toAdditive(lean_box(0), lean_box(0), v_inst_326_, v_inst_326_);
return v___x_327_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_monoidEndToAdditive___boxed(lean_object* v_M_328_, lean_object* v_inst_329_){
_start:
{
lean_object* v_res_330_; 
v_res_330_ = lp_mathlib_monoidEndToAdditive(v_M_328_, v_inst_329_);
lean_dec_ref(v_inst_329_);
return v_res_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidEndToMultiplicative___redArg(lean_object* v_inst_331_){
_start:
{
lean_object* v___x_332_; 
v___x_332_ = lp_mathlib_AddMonoidHom_toMultiplicative(lean_box(0), lean_box(0), v_inst_331_, v_inst_331_);
return v___x_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidEndToMultiplicative___redArg___boxed(lean_object* v_inst_333_){
_start:
{
lean_object* v_res_334_; 
v_res_334_ = lp_mathlib_addMonoidEndToMultiplicative___redArg(v_inst_333_);
lean_dec_ref(v_inst_333_);
return v_res_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidEndToMultiplicative(lean_object* v_A_335_, lean_object* v_inst_336_){
_start:
{
lean_object* v___x_337_; 
v___x_337_ = lp_mathlib_AddMonoidHom_toMultiplicative(lean_box(0), lean_box(0), v_inst_336_, v_inst_336_);
return v___x_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_addMonoidEndToMultiplicative___boxed(lean_object* v_A_338_, lean_object* v_inst_339_){
_start:
{
lean_object* v_res_340_; 
v_res_340_ = lp_mathlib_addMonoidEndToMultiplicative(v_A_338_, v_inst_339_);
lean_dec_ref(v_inst_339_);
return v_res_340_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_Monoid_End___redArg(lean_object* v_inst_341_){
_start:
{
lean_object* v___x_342_; 
v___x_342_ = lp_mathlib_MonoidHom_toAdditive(lean_box(0), lean_box(0), v_inst_341_, v_inst_341_);
return v___x_342_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_Monoid_End___redArg___boxed(lean_object* v_inst_343_){
_start:
{
lean_object* v_res_344_; 
v_res_344_ = lp_mathlib_MulEquiv_Monoid_End___redArg(v_inst_343_);
lean_dec_ref(v_inst_343_);
return v_res_344_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_Monoid_End(lean_object* v_M_345_, lean_object* v_inst_346_){
_start:
{
lean_object* v___x_347_; 
v___x_347_ = lp_mathlib_MonoidHom_toAdditive(lean_box(0), lean_box(0), v_inst_346_, v_inst_346_);
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_Monoid_End___boxed(lean_object* v_M_348_, lean_object* v_inst_349_){
_start:
{
lean_object* v_res_350_; 
v_res_350_ = lp_mathlib_MulEquiv_Monoid_End(v_M_348_, v_inst_349_);
lean_dec_ref(v_inst_349_);
return v_res_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_AddMonoid_End___redArg(lean_object* v_inst_351_){
_start:
{
lean_object* v___x_352_; 
v___x_352_ = lp_mathlib_AddMonoidHom_toMultiplicative(lean_box(0), lean_box(0), v_inst_351_, v_inst_351_);
return v___x_352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_AddMonoid_End___redArg___boxed(lean_object* v_inst_353_){
_start:
{
lean_object* v_res_354_; 
v_res_354_ = lp_mathlib_MulEquiv_AddMonoid_End___redArg(v_inst_353_);
lean_dec_ref(v_inst_353_);
return v_res_354_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_AddMonoid_End(lean_object* v_A_355_, lean_object* v_inst_356_){
_start:
{
lean_object* v___x_357_; 
v___x_357_ = lp_mathlib_AddMonoidHom_toMultiplicative(lean_box(0), lean_box(0), v_inst_356_, v_inst_356_);
return v___x_357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_AddMonoid_End___boxed(lean_object* v_A_358_, lean_object* v_inst_359_){
_start:
{
lean_object* v_res_360_; 
v_res_360_ = lp_mathlib_MulEquiv_AddMonoid_End(v_A_358_, v_inst_359_);
lean_dec_ref(v_inst_359_);
return v_res_360_;
}
}
static lean_object* _init_lp_mathlib_MulEquiv_piMultiplicative___lam__0___closed__0(void){
_start:
{
lean_object* v___x_361_; 
v___x_361_ = lp_mathlib_Multiplicative_toAdd(lean_box(0));
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piMultiplicative___lam__0(lean_object* v_x_362_, lean_object* v_i_363_){
_start:
{
lean_object* v___x_364_; lean_object* v_toFun_365_; lean_object* v___x_366_; lean_object* v_toFun_367_; lean_object* v___x_368_; lean_object* v___x_369_; 
v___x_364_ = lean_obj_once(&lp_mathlib_MulEquiv_piMultiplicative___lam__0___closed__0, &lp_mathlib_MulEquiv_piMultiplicative___lam__0___closed__0_once, _init_lp_mathlib_MulEquiv_piMultiplicative___lam__0___closed__0);
v_toFun_365_ = lean_ctor_get(v___x_364_, 0);
v___x_366_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0);
v_toFun_367_ = lean_ctor_get(v___x_366_, 0);
lean_inc(v_toFun_365_);
v___x_368_ = lean_apply_2(v_toFun_365_, v_x_362_, v_i_363_);
lean_inc(v_toFun_367_);
v___x_369_ = lean_apply_1(v_toFun_367_, v___x_368_);
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piMultiplicative___lam__1(lean_object* v_x_370_, lean_object* v_i_371_){
_start:
{
lean_object* v___x_372_; lean_object* v_toFun_373_; lean_object* v___x_374_; lean_object* v___x_375_; 
v___x_372_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1);
v_toFun_373_ = lean_ctor_get(v___x_372_, 0);
v___x_374_ = lean_apply_1(v_x_370_, v_i_371_);
lean_inc(v_toFun_373_);
v___x_375_ = lean_apply_1(v_toFun_373_, v___x_374_);
return v___x_375_;
}
}
static lean_object* _init_lp_mathlib_MulEquiv_piMultiplicative___lam__2___closed__0(void){
_start:
{
lean_object* v___x_376_; 
v___x_376_ = lp_mathlib_Multiplicative_ofAdd(lean_box(0));
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piMultiplicative___lam__2(lean_object* v_x_377_, lean_object* v___y_378_){
_start:
{
lean_object* v___x_379_; lean_object* v_toFun_380_; lean_object* v___f_381_; lean_object* v___x_382_; 
v___x_379_ = lean_obj_once(&lp_mathlib_MulEquiv_piMultiplicative___lam__2___closed__0, &lp_mathlib_MulEquiv_piMultiplicative___lam__2___closed__0_once, _init_lp_mathlib_MulEquiv_piMultiplicative___lam__2___closed__0);
v_toFun_380_ = lean_ctor_get(v___x_379_, 0);
v___f_381_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_piMultiplicative___lam__1), 2, 1);
lean_closure_set(v___f_381_, 0, v_x_377_);
lean_inc(v_toFun_380_);
v___x_382_ = lean_apply_2(v_toFun_380_, v___f_381_, v___y_378_);
return v___x_382_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piMultiplicative(lean_object* v_00_u03b9_388_, lean_object* v_K_389_, lean_object* v_inst_390_){
_start:
{
lean_object* v___x_391_; 
v___x_391_ = ((lean_object*)(lp_mathlib_MulEquiv_piMultiplicative___closed__2));
return v___x_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_piMultiplicative___boxed(lean_object* v_00_u03b9_392_, lean_object* v_K_393_, lean_object* v_inst_394_){
_start:
{
lean_object* v_res_395_; 
v_res_395_ = lp_mathlib_MulEquiv_piMultiplicative(v_00_u03b9_392_, v_K_393_, v_inst_394_);
lean_dec(v_inst_394_);
return v_res_395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_funMultiplicative___redArg___lam__0(lean_object* v_inst_396_, lean_object* v_i_397_, lean_object* v___y_398_, lean_object* v___y_399_){
_start:
{
lean_object* v___x_400_; 
v___x_400_ = lean_apply_2(v_inst_396_, v___y_398_, v___y_399_);
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_funMultiplicative___redArg___lam__0___boxed(lean_object* v_inst_401_, lean_object* v_i_402_, lean_object* v___y_403_, lean_object* v___y_404_){
_start:
{
lean_object* v_res_405_; 
v_res_405_ = lp_mathlib_MulEquiv_funMultiplicative___redArg___lam__0(v_inst_401_, v_i_402_, v___y_403_, v___y_404_);
lean_dec(v_i_402_);
return v_res_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_funMultiplicative___redArg(lean_object* v_inst_406_){
_start:
{
lean_object* v___f_407_; lean_object* v___x_408_; 
v___f_407_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_funMultiplicative___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_407_, 0, v_inst_406_);
v___x_408_ = lp_mathlib_MulEquiv_piMultiplicative(lean_box(0), lean_box(0), v___f_407_);
lean_dec_ref(v___f_407_);
return v___x_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_funMultiplicative(lean_object* v_00_u03b9_409_, lean_object* v_G_410_, lean_object* v_inst_411_){
_start:
{
lean_object* v___f_412_; lean_object* v___x_413_; 
v___f_412_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_funMultiplicative___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_412_, 0, v_inst_411_);
v___x_413_ = lp_mathlib_MulEquiv_piMultiplicative(lean_box(0), lean_box(0), v___f_412_);
lean_dec_ref(v___f_412_);
return v___x_413_;
}
}
static lean_object* _init_lp_mathlib_AddEquiv_piAdditive___lam__0___closed__0(void){
_start:
{
lean_object* v___x_414_; 
v___x_414_ = lp_mathlib_Additive_toMul(lean_box(0));
return v___x_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piAdditive___lam__0(lean_object* v_x_415_, lean_object* v_i_416_){
_start:
{
lean_object* v___x_417_; lean_object* v_toFun_418_; lean_object* v___x_419_; lean_object* v_toFun_420_; lean_object* v___x_421_; lean_object* v___x_422_; 
v___x_417_ = lean_obj_once(&lp_mathlib_AddEquiv_piAdditive___lam__0___closed__0, &lp_mathlib_AddEquiv_piAdditive___lam__0___closed__0_once, _init_lp_mathlib_AddEquiv_piAdditive___lam__0___closed__0);
v_toFun_418_ = lean_ctor_get(v___x_417_, 0);
v___x_419_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0);
v_toFun_420_ = lean_ctor_get(v___x_419_, 0);
lean_inc(v_toFun_418_);
v___x_421_ = lean_apply_2(v_toFun_418_, v_x_415_, v_i_416_);
lean_inc(v_toFun_420_);
v___x_422_ = lean_apply_1(v_toFun_420_, v___x_421_);
return v___x_422_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piAdditive___lam__1(lean_object* v_x_423_, lean_object* v_i_424_){
_start:
{
lean_object* v___x_425_; lean_object* v_toFun_426_; lean_object* v___x_427_; lean_object* v___x_428_; 
v___x_425_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1);
v_toFun_426_ = lean_ctor_get(v___x_425_, 0);
v___x_427_ = lean_apply_1(v_x_423_, v_i_424_);
lean_inc(v_toFun_426_);
v___x_428_ = lean_apply_1(v_toFun_426_, v___x_427_);
return v___x_428_;
}
}
static lean_object* _init_lp_mathlib_AddEquiv_piAdditive___lam__2___closed__0(void){
_start:
{
lean_object* v___x_429_; 
v___x_429_ = lp_mathlib_Additive_ofMul(lean_box(0));
return v___x_429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piAdditive___lam__2(lean_object* v_x_430_, lean_object* v___y_431_){
_start:
{
lean_object* v___x_432_; lean_object* v_toFun_433_; lean_object* v___f_434_; lean_object* v___x_435_; 
v___x_432_ = lean_obj_once(&lp_mathlib_AddEquiv_piAdditive___lam__2___closed__0, &lp_mathlib_AddEquiv_piAdditive___lam__2___closed__0_once, _init_lp_mathlib_AddEquiv_piAdditive___lam__2___closed__0);
v_toFun_433_ = lean_ctor_get(v___x_432_, 0);
v___f_434_ = lean_alloc_closure((void*)(lp_mathlib_AddEquiv_piAdditive___lam__1), 2, 1);
lean_closure_set(v___f_434_, 0, v_x_430_);
lean_inc(v_toFun_433_);
v___x_435_ = lean_apply_2(v_toFun_433_, v___f_434_, v___y_431_);
return v___x_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piAdditive(lean_object* v_00_u03b9_441_, lean_object* v_K_442_, lean_object* v_inst_443_){
_start:
{
lean_object* v___x_444_; 
v___x_444_ = ((lean_object*)(lp_mathlib_AddEquiv_piAdditive___closed__2));
return v___x_444_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_piAdditive___boxed(lean_object* v_00_u03b9_445_, lean_object* v_K_446_, lean_object* v_inst_447_){
_start:
{
lean_object* v_res_448_; 
v_res_448_ = lp_mathlib_AddEquiv_piAdditive(v_00_u03b9_445_, v_K_446_, v_inst_447_);
lean_dec(v_inst_447_);
return v_res_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_funAdditive___redArg(lean_object* v_inst_449_){
_start:
{
lean_object* v___f_450_; lean_object* v___x_451_; 
v___f_450_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_funMultiplicative___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_450_, 0, v_inst_449_);
v___x_451_ = lp_mathlib_AddEquiv_piAdditive(lean_box(0), lean_box(0), v___f_450_);
lean_dec_ref(v___f_450_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_funAdditive(lean_object* v_00_u03b9_452_, lean_object* v_G_453_, lean_object* v_inst_454_){
_start:
{
lean_object* v___f_455_; lean_object* v___x_456_; 
v___f_455_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_funMultiplicative___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_455_, 0, v_inst_454_);
v___x_456_ = lp_mathlib_AddEquiv_piAdditive(lean_box(0), lean_box(0), v___f_455_);
lean_dec_ref(v___f_455_);
return v___x_456_;
}
}
static lean_object* _init_lp_mathlib_AddEquiv_additiveMultiplicative___redArg___closed__0(void){
_start:
{
lean_object* v___x_457_; 
v___x_457_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_additiveMultiplicative___redArg(lean_object* v_inst_458_){
_start:
{
lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; lean_object* v_toFun_462_; lean_object* v___x_463_; lean_object* v___x_464_; 
lean_inc(v_inst_458_);
v___x_459_ = lp_mathlib_Multiplicative_mul___redArg(v_inst_458_);
v___x_460_ = lp_mathlib_AddEquiv_toMultiplicativeRight(lean_box(0), lean_box(0), v___x_459_, v_inst_458_);
lean_dec(v_inst_458_);
lean_dec(v___x_459_);
v___x_461_ = lp_mathlib_Equiv_symm___redArg(v___x_460_);
v_toFun_462_ = lean_ctor_get(v___x_461_, 0);
lean_inc(v_toFun_462_);
lean_dec_ref(v___x_461_);
v___x_463_ = lean_obj_once(&lp_mathlib_AddEquiv_additiveMultiplicative___redArg___closed__0, &lp_mathlib_AddEquiv_additiveMultiplicative___redArg___closed__0_once, _init_lp_mathlib_AddEquiv_additiveMultiplicative___redArg___closed__0);
v___x_464_ = lean_apply_1(v_toFun_462_, v___x_463_);
return v___x_464_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_additiveMultiplicative(lean_object* v_G_465_, lean_object* v_inst_466_){
_start:
{
lean_object* v___x_467_; 
v___x_467_ = lp_mathlib_AddEquiv_additiveMultiplicative___redArg(v_inst_466_);
return v___x_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_multiplicativeAdditive___redArg(lean_object* v_inst_468_){
_start:
{
lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v_toFun_471_; lean_object* v___x_472_; lean_object* v___x_473_; 
lean_inc(v_inst_468_);
v___x_469_ = lp_mathlib_Additive_add___redArg(v_inst_468_);
v___x_470_ = lp_mathlib_AddEquiv_toMultiplicativeLeft(lean_box(0), lean_box(0), v___x_469_, v_inst_468_);
lean_dec(v_inst_468_);
lean_dec(v___x_469_);
v_toFun_471_ = lean_ctor_get(v___x_470_, 0);
lean_inc(v_toFun_471_);
lean_dec_ref(v___x_470_);
v___x_472_ = lean_obj_once(&lp_mathlib_AddEquiv_additiveMultiplicative___redArg___closed__0, &lp_mathlib_AddEquiv_additiveMultiplicative___redArg___closed__0_once, _init_lp_mathlib_AddEquiv_additiveMultiplicative___redArg___closed__0);
v___x_473_ = lean_apply_1(v_toFun_471_, v___x_472_);
return v___x_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_multiplicativeAdditive(lean_object* v_H_474_, lean_object* v_inst_475_){
_start:
{
lean_object* v___x_476_; 
v___x_476_ = lp_mathlib_MulEquiv_multiplicativeAdditive___redArg(v_inst_475_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMultiplicative__toAdditive___redArg(lean_object* v_inst_477_){
_start:
{
lean_object* v___x_478_; 
v___x_478_ = lp_mathlib_MulEquiv_multiplicativeAdditive___redArg(v_inst_477_);
return v___x_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMultiplicative__toAdditive(lean_object* v_H_479_, lean_object* v_inst_480_){
_start:
{
lean_object* v___x_481_; 
v___x_481_ = lp_mathlib_MulEquiv_multiplicativeAdditive___redArg(v_inst_480_);
return v___x_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAdditive__toMultiplicative___redArg(lean_object* v_inst_482_){
_start:
{
lean_object* v___x_483_; 
v___x_483_ = lp_mathlib_AddEquiv_additiveMultiplicative___redArg(v_inst_482_);
return v___x_483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAdditive__toMultiplicative(lean_object* v_G_484_, lean_object* v_inst_485_){
_start:
{
lean_object* v___x_486_; 
v___x_486_ = lp_mathlib_AddEquiv_additiveMultiplicative___redArg(v_inst_485_);
return v___x_486_;
}
}
static lean_object* _init_lp_mathlib_MulEquiv_prodMultiplicative___lam__0___closed__0(void){
_start:
{
lean_object* v___x_487_; 
v___x_487_ = lp_mathlib_Multiplicative_toAdd(lean_box(0));
return v___x_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodMultiplicative___lam__0(lean_object* v_x_488_){
_start:
{
lean_object* v___x_489_; lean_object* v_toFun_490_; lean_object* v___x_491_; lean_object* v_fst_492_; lean_object* v_snd_493_; lean_object* v___x_495_; uint8_t v_isShared_496_; uint8_t v_isSharedCheck_505_; 
v___x_489_ = lean_obj_once(&lp_mathlib_MulEquiv_prodMultiplicative___lam__0___closed__0, &lp_mathlib_MulEquiv_prodMultiplicative___lam__0___closed__0_once, _init_lp_mathlib_MulEquiv_prodMultiplicative___lam__0___closed__0);
v_toFun_490_ = lean_ctor_get(v___x_489_, 0);
lean_inc(v_toFun_490_);
v___x_491_ = lean_apply_1(v_toFun_490_, v_x_488_);
v_fst_492_ = lean_ctor_get(v___x_491_, 0);
v_snd_493_ = lean_ctor_get(v___x_491_, 1);
v_isSharedCheck_505_ = !lean_is_exclusive(v___x_491_);
if (v_isSharedCheck_505_ == 0)
{
v___x_495_ = v___x_491_;
v_isShared_496_ = v_isSharedCheck_505_;
goto v_resetjp_494_;
}
else
{
lean_inc(v_snd_493_);
lean_inc(v_fst_492_);
lean_dec(v___x_491_);
v___x_495_ = lean_box(0);
v_isShared_496_ = v_isSharedCheck_505_;
goto v_resetjp_494_;
}
v_resetjp_494_:
{
lean_object* v___x_497_; lean_object* v_toFun_498_; lean_object* v_toFun_499_; lean_object* v___x_500_; lean_object* v___x_501_; lean_object* v___x_503_; 
v___x_497_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__0);
v_toFun_498_ = lean_ctor_get(v___x_497_, 0);
v_toFun_499_ = lean_ctor_get(v___x_497_, 0);
lean_inc(v_toFun_498_);
v___x_500_ = lean_apply_1(v_toFun_498_, v_fst_492_);
lean_inc(v_toFun_499_);
v___x_501_ = lean_apply_1(v_toFun_499_, v_snd_493_);
if (v_isShared_496_ == 0)
{
lean_ctor_set(v___x_495_, 1, v___x_501_);
lean_ctor_set(v___x_495_, 0, v___x_500_);
v___x_503_ = v___x_495_;
goto v_reusejp_502_;
}
else
{
lean_object* v_reuseFailAlloc_504_; 
v_reuseFailAlloc_504_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_504_, 0, v___x_500_);
lean_ctor_set(v_reuseFailAlloc_504_, 1, v___x_501_);
v___x_503_ = v_reuseFailAlloc_504_;
goto v_reusejp_502_;
}
v_reusejp_502_:
{
return v___x_503_;
}
}
}
}
static lean_object* _init_lp_mathlib_MulEquiv_prodMultiplicative___lam__1___closed__0(void){
_start:
{
lean_object* v___x_506_; 
v___x_506_ = lp_mathlib_Multiplicative_ofAdd(lean_box(0));
return v___x_506_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodMultiplicative___lam__1(lean_object* v_x_507_){
_start:
{
lean_object* v_fst_508_; lean_object* v_snd_509_; lean_object* v___x_511_; uint8_t v_isShared_512_; uint8_t v_isSharedCheck_524_; 
v_fst_508_ = lean_ctor_get(v_x_507_, 0);
v_snd_509_ = lean_ctor_get(v_x_507_, 1);
v_isSharedCheck_524_ = !lean_is_exclusive(v_x_507_);
if (v_isSharedCheck_524_ == 0)
{
v___x_511_ = v_x_507_;
v_isShared_512_ = v_isSharedCheck_524_;
goto v_resetjp_510_;
}
else
{
lean_inc(v_snd_509_);
lean_inc(v_fst_508_);
lean_dec(v_x_507_);
v___x_511_ = lean_box(0);
v_isShared_512_ = v_isSharedCheck_524_;
goto v_resetjp_510_;
}
v_resetjp_510_:
{
lean_object* v___x_513_; lean_object* v_toFun_514_; lean_object* v_toFun_515_; lean_object* v___x_516_; lean_object* v_toFun_517_; lean_object* v___x_518_; lean_object* v___x_519_; lean_object* v___x_521_; 
v___x_513_ = lean_obj_once(&lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1, &lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1_once, _init_lp_mathlib_AddEquiv_toMultiplicative___lam__0___closed__1);
v_toFun_514_ = lean_ctor_get(v___x_513_, 0);
v_toFun_515_ = lean_ctor_get(v___x_513_, 0);
v___x_516_ = lean_obj_once(&lp_mathlib_MulEquiv_prodMultiplicative___lam__1___closed__0, &lp_mathlib_MulEquiv_prodMultiplicative___lam__1___closed__0_once, _init_lp_mathlib_MulEquiv_prodMultiplicative___lam__1___closed__0);
v_toFun_517_ = lean_ctor_get(v___x_516_, 0);
lean_inc(v_toFun_514_);
v___x_518_ = lean_apply_1(v_toFun_514_, v_fst_508_);
lean_inc(v_toFun_515_);
v___x_519_ = lean_apply_1(v_toFun_515_, v_snd_509_);
if (v_isShared_512_ == 0)
{
lean_ctor_set(v___x_511_, 1, v___x_519_);
lean_ctor_set(v___x_511_, 0, v___x_518_);
v___x_521_ = v___x_511_;
goto v_reusejp_520_;
}
else
{
lean_object* v_reuseFailAlloc_523_; 
v_reuseFailAlloc_523_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_523_, 0, v___x_518_);
lean_ctor_set(v_reuseFailAlloc_523_, 1, v___x_519_);
v___x_521_ = v_reuseFailAlloc_523_;
goto v_reusejp_520_;
}
v_reusejp_520_:
{
lean_object* v___x_522_; 
lean_inc(v_toFun_517_);
v___x_522_ = lean_apply_1(v_toFun_517_, v___x_521_);
return v___x_522_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodMultiplicative(lean_object* v_G_530_, lean_object* v_H_531_, lean_object* v_inst_532_, lean_object* v_inst_533_){
_start:
{
lean_object* v___x_534_; 
v___x_534_ = ((lean_object*)(lp_mathlib_MulEquiv_prodMultiplicative___closed__2));
return v___x_534_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_prodMultiplicative___boxed(lean_object* v_G_535_, lean_object* v_H_536_, lean_object* v_inst_537_, lean_object* v_inst_538_){
_start:
{
lean_object* v_res_539_; 
v_res_539_ = lp_mathlib_MulEquiv_prodMultiplicative(v_G_535_, v_H_536_, v_inst_537_, v_inst_538_);
lean_dec(v_inst_538_);
lean_dec(v_inst_537_);
return v_res_539_;
}
}
static lean_object* _init_lp_mathlib_AddEquiv_prodAdditive___lam__0___closed__0(void){
_start:
{
lean_object* v___x_540_; 
v___x_540_ = lp_mathlib_Additive_toMul(lean_box(0));
return v___x_540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAdditive___lam__0(lean_object* v_x_541_){
_start:
{
lean_object* v___x_542_; lean_object* v_toFun_543_; lean_object* v___x_544_; lean_object* v_fst_545_; lean_object* v_snd_546_; lean_object* v___x_548_; uint8_t v_isShared_549_; uint8_t v_isSharedCheck_558_; 
v___x_542_ = lean_obj_once(&lp_mathlib_AddEquiv_prodAdditive___lam__0___closed__0, &lp_mathlib_AddEquiv_prodAdditive___lam__0___closed__0_once, _init_lp_mathlib_AddEquiv_prodAdditive___lam__0___closed__0);
v_toFun_543_ = lean_ctor_get(v___x_542_, 0);
lean_inc(v_toFun_543_);
v___x_544_ = lean_apply_1(v_toFun_543_, v_x_541_);
v_fst_545_ = lean_ctor_get(v___x_544_, 0);
v_snd_546_ = lean_ctor_get(v___x_544_, 1);
v_isSharedCheck_558_ = !lean_is_exclusive(v___x_544_);
if (v_isSharedCheck_558_ == 0)
{
v___x_548_ = v___x_544_;
v_isShared_549_ = v_isSharedCheck_558_;
goto v_resetjp_547_;
}
else
{
lean_inc(v_snd_546_);
lean_inc(v_fst_545_);
lean_dec(v___x_544_);
v___x_548_ = lean_box(0);
v_isShared_549_ = v_isSharedCheck_558_;
goto v_resetjp_547_;
}
v_resetjp_547_:
{
lean_object* v___x_550_; lean_object* v_toFun_551_; lean_object* v_toFun_552_; lean_object* v___x_553_; lean_object* v___x_554_; lean_object* v___x_556_; 
v___x_550_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__0);
v_toFun_551_ = lean_ctor_get(v___x_550_, 0);
v_toFun_552_ = lean_ctor_get(v___x_550_, 0);
lean_inc(v_toFun_551_);
v___x_553_ = lean_apply_1(v_toFun_551_, v_fst_545_);
lean_inc(v_toFun_552_);
v___x_554_ = lean_apply_1(v_toFun_552_, v_snd_546_);
if (v_isShared_549_ == 0)
{
lean_ctor_set(v___x_548_, 1, v___x_554_);
lean_ctor_set(v___x_548_, 0, v___x_553_);
v___x_556_ = v___x_548_;
goto v_reusejp_555_;
}
else
{
lean_object* v_reuseFailAlloc_557_; 
v_reuseFailAlloc_557_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_557_, 0, v___x_553_);
lean_ctor_set(v_reuseFailAlloc_557_, 1, v___x_554_);
v___x_556_ = v_reuseFailAlloc_557_;
goto v_reusejp_555_;
}
v_reusejp_555_:
{
return v___x_556_;
}
}
}
}
static lean_object* _init_lp_mathlib_AddEquiv_prodAdditive___lam__1___closed__0(void){
_start:
{
lean_object* v___x_559_; 
v___x_559_ = lp_mathlib_Additive_ofMul(lean_box(0));
return v___x_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAdditive___lam__1(lean_object* v_x_560_){
_start:
{
lean_object* v_fst_561_; lean_object* v_snd_562_; lean_object* v___x_564_; uint8_t v_isShared_565_; uint8_t v_isSharedCheck_577_; 
v_fst_561_ = lean_ctor_get(v_x_560_, 0);
v_snd_562_ = lean_ctor_get(v_x_560_, 1);
v_isSharedCheck_577_ = !lean_is_exclusive(v_x_560_);
if (v_isSharedCheck_577_ == 0)
{
v___x_564_ = v_x_560_;
v_isShared_565_ = v_isSharedCheck_577_;
goto v_resetjp_563_;
}
else
{
lean_inc(v_snd_562_);
lean_inc(v_fst_561_);
lean_dec(v_x_560_);
v___x_564_ = lean_box(0);
v_isShared_565_ = v_isSharedCheck_577_;
goto v_resetjp_563_;
}
v_resetjp_563_:
{
lean_object* v___x_566_; lean_object* v_toFun_567_; lean_object* v_toFun_568_; lean_object* v___x_569_; lean_object* v_toFun_570_; lean_object* v___x_571_; lean_object* v___x_572_; lean_object* v___x_574_; 
v___x_566_ = lean_obj_once(&lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1, &lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1_once, _init_lp_mathlib_MulEquiv_toAdditive___lam__0___closed__1);
v_toFun_567_ = lean_ctor_get(v___x_566_, 0);
v_toFun_568_ = lean_ctor_get(v___x_566_, 0);
v___x_569_ = lean_obj_once(&lp_mathlib_AddEquiv_prodAdditive___lam__1___closed__0, &lp_mathlib_AddEquiv_prodAdditive___lam__1___closed__0_once, _init_lp_mathlib_AddEquiv_prodAdditive___lam__1___closed__0);
v_toFun_570_ = lean_ctor_get(v___x_569_, 0);
lean_inc(v_toFun_567_);
v___x_571_ = lean_apply_1(v_toFun_567_, v_fst_561_);
lean_inc(v_toFun_568_);
v___x_572_ = lean_apply_1(v_toFun_568_, v_snd_562_);
if (v_isShared_565_ == 0)
{
lean_ctor_set(v___x_564_, 1, v___x_572_);
lean_ctor_set(v___x_564_, 0, v___x_571_);
v___x_574_ = v___x_564_;
goto v_reusejp_573_;
}
else
{
lean_object* v_reuseFailAlloc_576_; 
v_reuseFailAlloc_576_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_576_, 0, v___x_571_);
lean_ctor_set(v_reuseFailAlloc_576_, 1, v___x_572_);
v___x_574_ = v_reuseFailAlloc_576_;
goto v_reusejp_573_;
}
v_reusejp_573_:
{
lean_object* v___x_575_; 
lean_inc(v_toFun_570_);
v___x_575_ = lean_apply_1(v_toFun_570_, v___x_574_);
return v___x_575_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAdditive(lean_object* v_G_583_, lean_object* v_H_584_, lean_object* v_inst_585_, lean_object* v_inst_586_){
_start:
{
lean_object* v___x_587_; 
v___x_587_ = ((lean_object*)(lp_mathlib_AddEquiv_prodAdditive___closed__2));
return v___x_587_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_prodAdditive___boxed(lean_object* v_G_588_, lean_object* v_H_589_, lean_object* v_inst_590_, lean_object* v_inst_591_){
_start:
{
lean_object* v_res_592_; 
v_res_592_ = lp_mathlib_AddEquiv_prodAdditive(v_G_588_, v_H_589_, v_inst_590_, v_inst_591_);
lean_dec(v_inst_591_);
lean_dec(v_inst_590_);
return v_res_592_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_TypeTags(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Equiv_TypeTags(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_TypeTags(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_TypeTags(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Equiv_TypeTags(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Equiv_TypeTags(builtin);
}
#ifdef __cplusplus
}
#endif
