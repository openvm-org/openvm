// Lean compiler output
// Module: Mathlib.Algebra.Group.Submonoid.Operations
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Action.Faithful public import Mathlib.Algebra.Group.Pi.Lemmas public import Mathlib.Algebra.Group.Prod public import Mathlib.Algebra.Group.Submonoid.Basic public import Mathlib.Algebra.Group.Submonoid.MulAction public import Mathlib.Algebra.Group.TypeTags.Basic
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
lean_object* lp_mathlib_GaloisInsertion_monotoneIntro___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_subtypeProdEquivProd(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubmonoidClass_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOneClass_toMulOne___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_EquivLike_toEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_image___redArg(lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toAddSubmonoid___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toAddSubmonoid___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_Submonoid_toAddSubmonoid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submonoid_toAddSubmonoid___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submonoid_toAddSubmonoid___closed__0 = (const lean_object*)&lp_mathlib_Submonoid_toAddSubmonoid___closed__0_value;
static const lean_closure_object lp_mathlib_Submonoid_toAddSubmonoid___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submonoid_toAddSubmonoid___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submonoid_toAddSubmonoid___closed__1 = (const lean_object*)&lp_mathlib_Submonoid_toAddSubmonoid___closed__1_value;
static const lean_ctor_object lp_mathlib_Submonoid_toAddSubmonoid___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Submonoid_toAddSubmonoid___closed__0_value),((lean_object*)&lp_mathlib_Submonoid_toAddSubmonoid___closed__1_value)}};
static const lean_object* lp_mathlib_Submonoid_toAddSubmonoid___closed__2 = (const lean_object*)&lp_mathlib_Submonoid_toAddSubmonoid___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toAddSubmonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toAddSubmonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toSubmonoid_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toSubmonoid_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toSubmonoid_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toSubmonoid_x27___boxed(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_AddSubmonoid_toSubmonoid___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Submonoid_toAddSubmonoid___closed__1_value),((lean_object*)&lp_mathlib_Submonoid_toAddSubmonoid___closed__0_value)}};
static const lean_object* lp_mathlib_AddSubmonoid_toSubmonoid___closed__0 = (const lean_object*)&lp_mathlib_AddSubmonoid_toSubmonoid___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toSubmonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toSubmonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toAddSubmonoid_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toAddSubmonoid_x27___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toAddSubmonoid_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toAddSubmonoid_x27___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_comap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_comap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_comap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_gciMapComap___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_gciMapComap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_gciMapComap___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_gciMapComap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_giMapComap___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_giMapComap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_giMapComap___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_giMapComap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_topEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_topEquiv___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Submonoid_topEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Submonoid_topEquiv___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Submonoid_topEquiv___closed__0 = (const lean_object*)&lp_mathlib_Submonoid_topEquiv___closed__0_value;
static const lean_ctor_object lp_mathlib_Submonoid_topEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Submonoid_topEquiv___closed__0_value),((lean_object*)&lp_mathlib_Submonoid_topEquiv___closed__0_value)}};
static const lean_object* lp_mathlib_Submonoid_topEquiv___closed__1 = (const lean_object*)&lp_mathlib_Submonoid_topEquiv___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_topEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_topEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_topEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_topEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Submonoid_prodEquiv___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Submonoid_prodEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_prodEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_prodEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LibraryNote_range__copy__pattern;
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mrange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mrange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mrange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mrange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MonoidHom_domRestrict___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubmonoidClass_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MonoidHom_domRestrict___redArg___closed__0 = (const lean_object*)&lp_mathlib_MonoidHom_domRestrict___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MonoidHom_domRestrictHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MonoidHom_domRestrict___redArg, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MonoidHom_domRestrictHom___closed__0 = (const lean_object*)&lp_mathlib_MonoidHom_domRestrictHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrictHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrictHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_AddMonoidHom_domRestrictHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_AddMonoidHom_domRestrict___redArg, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_AddMonoidHom_domRestrictHom___closed__0 = (const lean_object*)&lp_mathlib_AddMonoidHom_domRestrictHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_restrictHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_restrictHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_restrictHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_restrictHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_codRestrict___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_codRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_codRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_codRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_codRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_codRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_codRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mrangeRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mrangeRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mrangeRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mrangeRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mrangeRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mrangeRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mker(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mker___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mker(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mker___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MonoidHom_decidableMemMker___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_decidableMemMker___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MonoidHom_decidableMemMker(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_decidableMemMker___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddMonoidHom_decidableMemMker___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_decidableMemMker___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_AddMonoidHom_decidableMemMker(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_decidableMemMker___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_submonoidComap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_submonoidComap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_submonoidComap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubmonoidComap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubmonoidComap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubmonoidComap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_submonoidMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_submonoidMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_submonoidMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubmonoidMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubmonoidMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubmonoidMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Submonoid_inclusion___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MonoidHom_codRestrict___redArg___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_MonoidHom_domRestrict___redArg___closed__0_value)} };
static const lean_object* lp_mathlib_Submonoid_inclusion___closed__0 = (const lean_object*)&lp_mathlib_Submonoid_inclusion___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_inclusion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_inclusion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_pi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_restrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_restrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_restrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_restrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_restrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_restrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_MulEquiv_submonoidCongr___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_MulEquiv_submonoidCongr___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_submonoidCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_submonoidCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubmonoidCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubmonoidCongr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ofLeftInverse_x27___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ofLeftInverse_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ofLeftInverse_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ofLeftInverse_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ofLeftInverse_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ofLeftInverse_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ofLeftInverse_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_submonoidMap___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_submonoidMap___redArg___lam__1(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MulEquiv_submonoidMap___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulEquiv_submonoidMap___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulEquiv_submonoidMap___redArg___closed__0 = (const lean_object*)&lp_mathlib_MulEquiv_submonoidMap___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_MulEquiv_submonoidMap___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_MulEquiv_submonoidMap___redArg___lam__1, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MulEquiv_submonoidMap___redArg___closed__1 = (const lean_object*)&lp_mathlib_MulEquiv_submonoidMap___redArg___closed__1_value;
static const lean_ctor_object lp_mathlib_MulEquiv_submonoidMap___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_MulEquiv_submonoidMap___redArg___closed__0_value),((lean_object*)&lp_mathlib_MulEquiv_submonoidMap___redArg___closed__1_value)}};
static const lean_object* lp_mathlib_MulEquiv_submonoidMap___redArg___closed__2 = (const lean_object*)&lp_mathlib_MulEquiv_submonoidMap___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_submonoidMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_submonoidMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_submonoidMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubmonoidMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubmonoidMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubmonoidMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toAddSubmonoid___lam__0(lean_object* v_S_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_box(0);
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toAddSubmonoid___lam__1(lean_object* v_S_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_box(0);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toAddSubmonoid(lean_object* v_M_10_, lean_object* v_inst_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = ((lean_object*)(lp_mathlib_Submonoid_toAddSubmonoid___closed__2));
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toAddSubmonoid___boxed(lean_object* v_M_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_mathlib_Submonoid_toAddSubmonoid(v_M_13_, v_inst_14_);
lean_dec_ref(v_inst_14_);
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toSubmonoid_x27___redArg(lean_object* v_inst_16_){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; 
v___x_17_ = lp_mathlib_Submonoid_toAddSubmonoid(lean_box(0), v_inst_16_);
v___x_18_ = lp_mathlib_Equiv_symm___redArg(v___x_17_);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toSubmonoid_x27___redArg___boxed(lean_object* v_inst_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_AddSubmonoid_toSubmonoid_x27___redArg(v_inst_19_);
lean_dec_ref(v_inst_19_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toSubmonoid_x27(lean_object* v_M_21_, lean_object* v_inst_22_){
_start:
{
lean_object* v___x_23_; lean_object* v___x_24_; 
v___x_23_ = lp_mathlib_Submonoid_toAddSubmonoid(lean_box(0), v_inst_22_);
v___x_24_ = lp_mathlib_Equiv_symm___redArg(v___x_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toSubmonoid_x27___boxed(lean_object* v_M_25_, lean_object* v_inst_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_AddSubmonoid_toSubmonoid_x27(v_M_25_, v_inst_26_);
lean_dec_ref(v_inst_26_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toSubmonoid(lean_object* v_A_31_, lean_object* v_inst_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = ((lean_object*)(lp_mathlib_AddSubmonoid_toSubmonoid___closed__0));
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_toSubmonoid___boxed(lean_object* v_A_34_, lean_object* v_inst_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_AddSubmonoid_toSubmonoid(v_A_34_, v_inst_35_);
lean_dec_ref(v_inst_35_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toAddSubmonoid_x27___redArg(lean_object* v_inst_37_){
_start:
{
lean_object* v___x_38_; lean_object* v___x_39_; 
v___x_38_ = lp_mathlib_AddSubmonoid_toSubmonoid(lean_box(0), v_inst_37_);
v___x_39_ = lp_mathlib_Equiv_symm___redArg(v___x_38_);
return v___x_39_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toAddSubmonoid_x27___redArg___boxed(lean_object* v_inst_40_){
_start:
{
lean_object* v_res_41_; 
v_res_41_ = lp_mathlib_Submonoid_toAddSubmonoid_x27___redArg(v_inst_40_);
lean_dec_ref(v_inst_40_);
return v_res_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toAddSubmonoid_x27(lean_object* v_A_42_, lean_object* v_inst_43_){
_start:
{
lean_object* v___x_44_; lean_object* v___x_45_; 
v___x_44_ = lp_mathlib_AddSubmonoid_toSubmonoid(lean_box(0), v_inst_43_);
v___x_45_ = lp_mathlib_Equiv_symm___redArg(v___x_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_toAddSubmonoid_x27___boxed(lean_object* v_A_46_, lean_object* v_inst_47_){
_start:
{
lean_object* v_res_48_; 
v_res_48_ = lp_mathlib_Submonoid_toAddSubmonoid_x27(v_A_46_, v_inst_47_);
lean_dec_ref(v_inst_47_);
return v_res_48_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_comap(lean_object* v_M_49_, lean_object* v_N_50_, lean_object* v_inst_51_, lean_object* v_inst_52_, lean_object* v_F_53_, lean_object* v_inst_54_, lean_object* v_mc_55_, lean_object* v_f_56_, lean_object* v_S_57_){
_start:
{
lean_object* v___x_58_; 
v___x_58_ = lean_box(0);
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_comap___boxed(lean_object* v_M_59_, lean_object* v_N_60_, lean_object* v_inst_61_, lean_object* v_inst_62_, lean_object* v_F_63_, lean_object* v_inst_64_, lean_object* v_mc_65_, lean_object* v_f_66_, lean_object* v_S_67_){
_start:
{
lean_object* v_res_68_; 
v_res_68_ = lp_mathlib_Submonoid_comap(v_M_59_, v_N_60_, v_inst_61_, v_inst_62_, v_F_63_, v_inst_64_, v_mc_65_, v_f_66_, v_S_67_);
lean_dec(v_f_66_);
lean_dec(v_inst_64_);
lean_dec_ref(v_inst_62_);
lean_dec_ref(v_inst_61_);
return v_res_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_comap(lean_object* v_M_69_, lean_object* v_N_70_, lean_object* v_inst_71_, lean_object* v_inst_72_, lean_object* v_F_73_, lean_object* v_inst_74_, lean_object* v_mc_75_, lean_object* v_f_76_, lean_object* v_S_77_){
_start:
{
lean_object* v___x_78_; 
v___x_78_ = lean_box(0);
return v___x_78_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_comap___boxed(lean_object* v_M_79_, lean_object* v_N_80_, lean_object* v_inst_81_, lean_object* v_inst_82_, lean_object* v_F_83_, lean_object* v_inst_84_, lean_object* v_mc_85_, lean_object* v_f_86_, lean_object* v_S_87_){
_start:
{
lean_object* v_res_88_; 
v_res_88_ = lp_mathlib_AddSubmonoid_comap(v_M_79_, v_N_80_, v_inst_81_, v_inst_82_, v_F_83_, v_inst_84_, v_mc_85_, v_f_86_, v_S_87_);
lean_dec(v_f_86_);
lean_dec(v_inst_84_);
lean_dec_ref(v_inst_82_);
lean_dec_ref(v_inst_81_);
return v_res_88_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_map(lean_object* v_M_89_, lean_object* v_N_90_, lean_object* v_inst_91_, lean_object* v_inst_92_, lean_object* v_F_93_, lean_object* v_inst_94_, lean_object* v_mc_95_, lean_object* v_f_96_, lean_object* v_S_97_){
_start:
{
lean_object* v___x_98_; 
v___x_98_ = lean_box(0);
return v___x_98_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_map___boxed(lean_object* v_M_99_, lean_object* v_N_100_, lean_object* v_inst_101_, lean_object* v_inst_102_, lean_object* v_F_103_, lean_object* v_inst_104_, lean_object* v_mc_105_, lean_object* v_f_106_, lean_object* v_S_107_){
_start:
{
lean_object* v_res_108_; 
v_res_108_ = lp_mathlib_Submonoid_map(v_M_99_, v_N_100_, v_inst_101_, v_inst_102_, v_F_103_, v_inst_104_, v_mc_105_, v_f_106_, v_S_107_);
lean_dec(v_f_106_);
lean_dec(v_inst_104_);
lean_dec_ref(v_inst_102_);
lean_dec_ref(v_inst_101_);
return v_res_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_map(lean_object* v_M_109_, lean_object* v_N_110_, lean_object* v_inst_111_, lean_object* v_inst_112_, lean_object* v_F_113_, lean_object* v_inst_114_, lean_object* v_mc_115_, lean_object* v_f_116_, lean_object* v_S_117_){
_start:
{
lean_object* v___x_118_; 
v___x_118_ = lean_box(0);
return v___x_118_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_map___boxed(lean_object* v_M_119_, lean_object* v_N_120_, lean_object* v_inst_121_, lean_object* v_inst_122_, lean_object* v_F_123_, lean_object* v_inst_124_, lean_object* v_mc_125_, lean_object* v_f_126_, lean_object* v_S_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_mathlib_AddSubmonoid_map(v_M_119_, v_N_120_, v_inst_121_, v_inst_122_, v_F_123_, v_inst_124_, v_mc_125_, v_f_126_, v_S_127_);
lean_dec(v_f_126_);
lean_dec(v_inst_124_);
lean_dec_ref(v_inst_122_);
lean_dec_ref(v_inst_121_);
return v_res_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_gciMapComap___redArg(lean_object* v_inst_129_, lean_object* v_inst_130_, lean_object* v_inst_131_, lean_object* v_f_132_){
_start:
{
lean_object* v___x_133_; lean_object* v___f_134_; 
v___x_133_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_comap___boxed), 9, 8);
lean_closure_set(v___x_133_, 0, lean_box(0));
lean_closure_set(v___x_133_, 1, lean_box(0));
lean_closure_set(v___x_133_, 2, v_inst_129_);
lean_closure_set(v___x_133_, 3, v_inst_130_);
lean_closure_set(v___x_133_, 4, lean_box(0));
lean_closure_set(v___x_133_, 5, v_inst_131_);
lean_closure_set(v___x_133_, 6, lean_box(0));
lean_closure_set(v___x_133_, 7, v_f_132_);
v___f_134_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_monotoneIntro___redArg___lam__0), 3, 1);
lean_closure_set(v___f_134_, 0, v___x_133_);
return v___f_134_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_gciMapComap(lean_object* v_M_135_, lean_object* v_N_136_, lean_object* v_inst_137_, lean_object* v_inst_138_, lean_object* v_F_139_, lean_object* v_inst_140_, lean_object* v_mc_141_, lean_object* v_f_142_, lean_object* v_hf_143_){
_start:
{
lean_object* v___x_144_; 
v___x_144_ = lp_mathlib_Submonoid_gciMapComap___redArg(v_inst_137_, v_inst_138_, v_inst_140_, v_f_142_);
return v___x_144_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_gciMapComap___redArg(lean_object* v_inst_145_, lean_object* v_inst_146_, lean_object* v_inst_147_, lean_object* v_f_148_){
_start:
{
lean_object* v___x_149_; lean_object* v___f_150_; 
v___x_149_ = lean_alloc_closure((void*)(lp_mathlib_AddSubmonoid_comap___boxed), 9, 8);
lean_closure_set(v___x_149_, 0, lean_box(0));
lean_closure_set(v___x_149_, 1, lean_box(0));
lean_closure_set(v___x_149_, 2, v_inst_145_);
lean_closure_set(v___x_149_, 3, v_inst_146_);
lean_closure_set(v___x_149_, 4, lean_box(0));
lean_closure_set(v___x_149_, 5, v_inst_147_);
lean_closure_set(v___x_149_, 6, lean_box(0));
lean_closure_set(v___x_149_, 7, v_f_148_);
v___f_150_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_monotoneIntro___redArg___lam__0), 3, 1);
lean_closure_set(v___f_150_, 0, v___x_149_);
return v___f_150_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_gciMapComap(lean_object* v_M_151_, lean_object* v_N_152_, lean_object* v_inst_153_, lean_object* v_inst_154_, lean_object* v_F_155_, lean_object* v_inst_156_, lean_object* v_mc_157_, lean_object* v_f_158_, lean_object* v_hf_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lp_mathlib_AddSubmonoid_gciMapComap___redArg(v_inst_153_, v_inst_154_, v_inst_156_, v_f_158_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_giMapComap___redArg(lean_object* v_inst_161_, lean_object* v_inst_162_, lean_object* v_inst_163_, lean_object* v_f_164_){
_start:
{
lean_object* v___x_165_; lean_object* v___f_166_; 
v___x_165_ = lean_alloc_closure((void*)(lp_mathlib_Submonoid_map___boxed), 9, 8);
lean_closure_set(v___x_165_, 0, lean_box(0));
lean_closure_set(v___x_165_, 1, lean_box(0));
lean_closure_set(v___x_165_, 2, v_inst_161_);
lean_closure_set(v___x_165_, 3, v_inst_162_);
lean_closure_set(v___x_165_, 4, lean_box(0));
lean_closure_set(v___x_165_, 5, v_inst_163_);
lean_closure_set(v___x_165_, 6, lean_box(0));
lean_closure_set(v___x_165_, 7, v_f_164_);
v___f_166_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_monotoneIntro___redArg___lam__0), 3, 1);
lean_closure_set(v___f_166_, 0, v___x_165_);
return v___f_166_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_giMapComap(lean_object* v_M_167_, lean_object* v_N_168_, lean_object* v_inst_169_, lean_object* v_inst_170_, lean_object* v_F_171_, lean_object* v_inst_172_, lean_object* v_mc_173_, lean_object* v_f_174_, lean_object* v_hf_175_){
_start:
{
lean_object* v___x_176_; 
v___x_176_ = lp_mathlib_Submonoid_giMapComap___redArg(v_inst_169_, v_inst_170_, v_inst_172_, v_f_174_);
return v___x_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_giMapComap___redArg(lean_object* v_inst_177_, lean_object* v_inst_178_, lean_object* v_inst_179_, lean_object* v_f_180_){
_start:
{
lean_object* v___x_181_; lean_object* v___f_182_; 
v___x_181_ = lean_alloc_closure((void*)(lp_mathlib_AddSubmonoid_map___boxed), 9, 8);
lean_closure_set(v___x_181_, 0, lean_box(0));
lean_closure_set(v___x_181_, 1, lean_box(0));
lean_closure_set(v___x_181_, 2, v_inst_177_);
lean_closure_set(v___x_181_, 3, v_inst_178_);
lean_closure_set(v___x_181_, 4, lean_box(0));
lean_closure_set(v___x_181_, 5, v_inst_179_);
lean_closure_set(v___x_181_, 6, lean_box(0));
lean_closure_set(v___x_181_, 7, v_f_180_);
v___f_182_ = lean_alloc_closure((void*)(lp_mathlib_GaloisInsertion_monotoneIntro___redArg___lam__0), 3, 1);
lean_closure_set(v___f_182_, 0, v___x_181_);
return v___f_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_giMapComap(lean_object* v_M_183_, lean_object* v_N_184_, lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_F_187_, lean_object* v_inst_188_, lean_object* v_mc_189_, lean_object* v_f_190_, lean_object* v_hf_191_){
_start:
{
lean_object* v___x_192_; 
v___x_192_ = lp_mathlib_AddSubmonoid_giMapComap___redArg(v_inst_185_, v_inst_186_, v_inst_188_, v_f_190_);
return v___x_192_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_topEquiv___lam__0(lean_object* v_x_193_){
_start:
{
lean_inc(v_x_193_);
return v_x_193_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_topEquiv___lam__0___boxed(lean_object* v_x_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_mathlib_Submonoid_topEquiv___lam__0(v_x_194_);
lean_dec(v_x_194_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_topEquiv(lean_object* v_M_199_, lean_object* v_inst_200_){
_start:
{
lean_object* v___x_201_; 
v___x_201_ = ((lean_object*)(lp_mathlib_Submonoid_topEquiv___closed__1));
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_topEquiv___boxed(lean_object* v_M_202_, lean_object* v_inst_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib_Submonoid_topEquiv(v_M_202_, v_inst_203_);
lean_dec_ref(v_inst_203_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_topEquiv(lean_object* v_M_205_, lean_object* v_inst_206_){
_start:
{
lean_object* v___x_207_; 
v___x_207_ = ((lean_object*)(lp_mathlib_Submonoid_topEquiv___closed__1));
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_topEquiv___boxed(lean_object* v_M_208_, lean_object* v_inst_209_){
_start:
{
lean_object* v_res_210_; 
v_res_210_ = lp_mathlib_AddSubmonoid_topEquiv(v_M_208_, v_inst_209_);
lean_dec_ref(v_inst_209_);
return v_res_210_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_prod(lean_object* v_N_211_, lean_object* v_inst_212_, lean_object* v_M_213_, lean_object* v_inst_214_, lean_object* v_s_215_, lean_object* v_t_216_){
_start:
{
lean_object* v___x_217_; 
v___x_217_ = lean_box(0);
return v___x_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_prod___boxed(lean_object* v_N_218_, lean_object* v_inst_219_, lean_object* v_M_220_, lean_object* v_inst_221_, lean_object* v_s_222_, lean_object* v_t_223_){
_start:
{
lean_object* v_res_224_; 
v_res_224_ = lp_mathlib_Submonoid_prod(v_N_218_, v_inst_219_, v_M_220_, v_inst_221_, v_s_222_, v_t_223_);
lean_dec_ref(v_inst_221_);
lean_dec_ref(v_inst_219_);
return v_res_224_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_prod(lean_object* v_N_225_, lean_object* v_inst_226_, lean_object* v_M_227_, lean_object* v_inst_228_, lean_object* v_s_229_, lean_object* v_t_230_){
_start:
{
lean_object* v___x_231_; 
v___x_231_ = lean_box(0);
return v___x_231_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_prod___boxed(lean_object* v_N_232_, lean_object* v_inst_233_, lean_object* v_M_234_, lean_object* v_inst_235_, lean_object* v_s_236_, lean_object* v_t_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_mathlib_AddSubmonoid_prod(v_N_232_, v_inst_233_, v_M_234_, v_inst_235_, v_s_236_, v_t_237_);
lean_dec_ref(v_inst_235_);
lean_dec_ref(v_inst_233_);
return v_res_238_;
}
}
static lean_object* _init_lp_mathlib_Submonoid_prodEquiv___closed__0(void){
_start:
{
lean_object* v___x_239_; 
v___x_239_ = lp_mathlib_Equiv_subtypeProdEquivProd(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_prodEquiv(lean_object* v_N_240_, lean_object* v_inst_241_, lean_object* v_M_242_, lean_object* v_inst_243_, lean_object* v_s_244_, lean_object* v_t_245_){
_start:
{
lean_object* v___x_246_; 
v___x_246_ = lean_obj_once(&lp_mathlib_Submonoid_prodEquiv___closed__0, &lp_mathlib_Submonoid_prodEquiv___closed__0_once, _init_lp_mathlib_Submonoid_prodEquiv___closed__0);
return v___x_246_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_prodEquiv___boxed(lean_object* v_N_247_, lean_object* v_inst_248_, lean_object* v_M_249_, lean_object* v_inst_250_, lean_object* v_s_251_, lean_object* v_t_252_){
_start:
{
lean_object* v_res_253_; 
v_res_253_ = lp_mathlib_Submonoid_prodEquiv(v_N_247_, v_inst_248_, v_M_249_, v_inst_250_, v_s_251_, v_t_252_);
lean_dec_ref(v_inst_250_);
lean_dec_ref(v_inst_248_);
return v_res_253_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_prodEquiv(lean_object* v_N_254_, lean_object* v_inst_255_, lean_object* v_M_256_, lean_object* v_inst_257_, lean_object* v_s_258_, lean_object* v_t_259_){
_start:
{
lean_object* v___x_260_; 
v___x_260_ = lean_obj_once(&lp_mathlib_Submonoid_prodEquiv___closed__0, &lp_mathlib_Submonoid_prodEquiv___closed__0_once, _init_lp_mathlib_Submonoid_prodEquiv___closed__0);
return v___x_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_prodEquiv___boxed(lean_object* v_N_261_, lean_object* v_inst_262_, lean_object* v_M_263_, lean_object* v_inst_264_, lean_object* v_s_265_, lean_object* v_t_266_){
_start:
{
lean_object* v_res_267_; 
v_res_267_ = lp_mathlib_AddSubmonoid_prodEquiv(v_N_261_, v_inst_262_, v_M_263_, v_inst_264_, v_s_265_, v_t_266_);
lean_dec_ref(v_inst_264_);
lean_dec_ref(v_inst_262_);
return v_res_267_;
}
}
static lean_object* _init_lp_mathlib_LibraryNote_range__copy__pattern(void){
_start:
{
lean_object* v___x_268_; 
v___x_268_ = lean_box(0);
return v___x_268_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mrange(lean_object* v_M_269_, lean_object* v_N_270_, lean_object* v_inst_271_, lean_object* v_inst_272_, lean_object* v_F_273_, lean_object* v_inst_274_, lean_object* v_mc_275_, lean_object* v_f_276_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = lean_box(0);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mrange___boxed(lean_object* v_M_278_, lean_object* v_N_279_, lean_object* v_inst_280_, lean_object* v_inst_281_, lean_object* v_F_282_, lean_object* v_inst_283_, lean_object* v_mc_284_, lean_object* v_f_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_mathlib_MonoidHom_mrange(v_M_278_, v_N_279_, v_inst_280_, v_inst_281_, v_F_282_, v_inst_283_, v_mc_284_, v_f_285_);
lean_dec(v_f_285_);
lean_dec(v_inst_283_);
lean_dec_ref(v_inst_281_);
lean_dec_ref(v_inst_280_);
return v_res_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mrange(lean_object* v_M_287_, lean_object* v_N_288_, lean_object* v_inst_289_, lean_object* v_inst_290_, lean_object* v_F_291_, lean_object* v_inst_292_, lean_object* v_mc_293_, lean_object* v_f_294_){
_start:
{
lean_object* v___x_295_; 
v___x_295_ = lean_box(0);
return v___x_295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mrange___boxed(lean_object* v_M_296_, lean_object* v_N_297_, lean_object* v_inst_298_, lean_object* v_inst_299_, lean_object* v_F_300_, lean_object* v_inst_301_, lean_object* v_mc_302_, lean_object* v_f_303_){
_start:
{
lean_object* v_res_304_; 
v_res_304_ = lp_mathlib_AddMonoidHom_mrange(v_M_296_, v_N_297_, v_inst_298_, v_inst_299_, v_F_300_, v_inst_301_, v_mc_302_, v_f_303_);
lean_dec(v_f_303_);
lean_dec(v_inst_301_);
lean_dec_ref(v_inst_299_);
lean_dec_ref(v_inst_298_);
return v_res_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrict___redArg(lean_object* v_f_306_){
_start:
{
lean_object* v___f_307_; lean_object* v___f_308_; 
v___f_307_ = ((lean_object*)(lp_mathlib_MonoidHom_domRestrict___redArg___closed__0));
v___f_308_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_308_, 0, v___f_307_);
lean_closure_set(v___f_308_, 1, v_f_306_);
return v___f_308_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrict(lean_object* v_M_309_, lean_object* v_inst_310_, lean_object* v_N_311_, lean_object* v_S_312_, lean_object* v_inst_313_, lean_object* v_inst_314_, lean_object* v_inst_315_, lean_object* v_f_316_, lean_object* v_s_317_){
_start:
{
lean_object* v___x_318_; 
v___x_318_ = lp_mathlib_MonoidHom_domRestrict___redArg(v_f_316_);
return v___x_318_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrict___boxed(lean_object* v_M_319_, lean_object* v_inst_320_, lean_object* v_N_321_, lean_object* v_S_322_, lean_object* v_inst_323_, lean_object* v_inst_324_, lean_object* v_inst_325_, lean_object* v_f_326_, lean_object* v_s_327_){
_start:
{
lean_object* v_res_328_; 
v_res_328_ = lp_mathlib_MonoidHom_domRestrict(v_M_319_, v_inst_320_, v_N_321_, v_S_322_, v_inst_323_, v_inst_324_, v_inst_325_, v_f_326_, v_s_327_);
lean_dec(v_s_327_);
lean_dec_ref(v_inst_323_);
lean_dec_ref(v_inst_320_);
return v_res_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrict___redArg(lean_object* v_f_329_){
_start:
{
lean_object* v___f_330_; lean_object* v___f_331_; 
v___f_330_ = ((lean_object*)(lp_mathlib_MonoidHom_domRestrict___redArg___closed__0));
v___f_331_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_331_, 0, v___f_330_);
lean_closure_set(v___f_331_, 1, v_f_329_);
return v___f_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrict(lean_object* v_M_332_, lean_object* v_inst_333_, lean_object* v_N_334_, lean_object* v_S_335_, lean_object* v_inst_336_, lean_object* v_inst_337_, lean_object* v_inst_338_, lean_object* v_f_339_, lean_object* v_s_340_){
_start:
{
lean_object* v___x_341_; 
v___x_341_ = lp_mathlib_AddMonoidHom_domRestrict___redArg(v_f_339_);
return v___x_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrict___boxed(lean_object* v_M_342_, lean_object* v_inst_343_, lean_object* v_N_344_, lean_object* v_S_345_, lean_object* v_inst_346_, lean_object* v_inst_347_, lean_object* v_inst_348_, lean_object* v_f_349_, lean_object* v_s_350_){
_start:
{
lean_object* v_res_351_; 
v_res_351_ = lp_mathlib_AddMonoidHom_domRestrict(v_M_342_, v_inst_343_, v_N_344_, v_S_345_, v_inst_346_, v_inst_347_, v_inst_348_, v_f_349_, v_s_350_);
lean_dec(v_s_350_);
lean_dec_ref(v_inst_346_);
lean_dec_ref(v_inst_343_);
return v_res_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrictHom(lean_object* v_M_353_, lean_object* v_inst_354_, lean_object* v_S_355_, lean_object* v_inst_356_, lean_object* v_inst_357_, lean_object* v_M_x27_358_, lean_object* v_A_359_, lean_object* v_inst_360_){
_start:
{
lean_object* v___f_361_; 
v___f_361_ = ((lean_object*)(lp_mathlib_MonoidHom_domRestrictHom___closed__0));
return v___f_361_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrictHom___boxed(lean_object* v_M_362_, lean_object* v_inst_363_, lean_object* v_S_364_, lean_object* v_inst_365_, lean_object* v_inst_366_, lean_object* v_M_x27_367_, lean_object* v_A_368_, lean_object* v_inst_369_){
_start:
{
lean_object* v_res_370_; 
v_res_370_ = lp_mathlib_MonoidHom_domRestrictHom(v_M_362_, v_inst_363_, v_S_364_, v_inst_365_, v_inst_366_, v_M_x27_367_, v_A_368_, v_inst_369_);
lean_dec_ref(v_inst_369_);
lean_dec(v_M_x27_367_);
lean_dec_ref(v_inst_363_);
return v_res_370_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHom(lean_object* v_M_372_, lean_object* v_inst_373_, lean_object* v_S_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_M_x27_377_, lean_object* v_A_378_, lean_object* v_inst_379_){
_start:
{
lean_object* v___f_380_; 
v___f_380_ = ((lean_object*)(lp_mathlib_AddMonoidHom_domRestrictHom___closed__0));
return v___f_380_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHom___boxed(lean_object* v_M_381_, lean_object* v_inst_382_, lean_object* v_S_383_, lean_object* v_inst_384_, lean_object* v_inst_385_, lean_object* v_M_x27_386_, lean_object* v_A_387_, lean_object* v_inst_388_){
_start:
{
lean_object* v_res_389_; 
v_res_389_ = lp_mathlib_AddMonoidHom_domRestrictHom(v_M_381_, v_inst_382_, v_S_383_, v_inst_384_, v_inst_385_, v_M_x27_386_, v_A_387_, v_inst_388_);
lean_dec_ref(v_inst_388_);
lean_dec(v_M_x27_386_);
lean_dec_ref(v_inst_382_);
return v_res_389_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_restrictHom(lean_object* v_M_390_, lean_object* v_inst_391_, lean_object* v_S_392_, lean_object* v_inst_393_, lean_object* v_inst_394_, lean_object* v_M_x27_395_, lean_object* v_A_396_, lean_object* v_inst_397_){
_start:
{
lean_object* v___f_398_; 
v___f_398_ = ((lean_object*)(lp_mathlib_MonoidHom_domRestrictHom___closed__0));
return v___f_398_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_restrictHom___boxed(lean_object* v_M_399_, lean_object* v_inst_400_, lean_object* v_S_401_, lean_object* v_inst_402_, lean_object* v_inst_403_, lean_object* v_M_x27_404_, lean_object* v_A_405_, lean_object* v_inst_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_mathlib_MonoidHom_restrictHom(v_M_399_, v_inst_400_, v_S_401_, v_inst_402_, v_inst_403_, v_M_x27_404_, v_A_405_, v_inst_406_);
lean_dec_ref(v_inst_406_);
lean_dec(v_M_x27_404_);
lean_dec_ref(v_inst_400_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_restrictHom(lean_object* v_M_408_, lean_object* v_inst_409_, lean_object* v_S_410_, lean_object* v_inst_411_, lean_object* v_inst_412_, lean_object* v_M_x27_413_, lean_object* v_A_414_, lean_object* v_inst_415_){
_start:
{
lean_object* v___f_416_; 
v___f_416_ = ((lean_object*)(lp_mathlib_AddMonoidHom_domRestrictHom___closed__0));
return v___f_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_restrictHom___boxed(lean_object* v_M_417_, lean_object* v_inst_418_, lean_object* v_S_419_, lean_object* v_inst_420_, lean_object* v_inst_421_, lean_object* v_M_x27_422_, lean_object* v_A_423_, lean_object* v_inst_424_){
_start:
{
lean_object* v_res_425_; 
v_res_425_ = lp_mathlib_AddMonoidHom_restrictHom(v_M_417_, v_inst_418_, v_S_419_, v_inst_420_, v_inst_421_, v_M_x27_422_, v_A_423_, v_inst_424_);
lean_dec_ref(v_inst_424_);
lean_dec(v_M_x27_422_);
lean_dec_ref(v_inst_418_);
return v_res_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_codRestrict___redArg___lam__0(lean_object* v_f_426_, lean_object* v_n_427_){
_start:
{
lean_object* v___x_428_; 
v___x_428_ = lean_apply_1(v_f_426_, v_n_427_);
return v___x_428_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_codRestrict___redArg(lean_object* v_f_429_){
_start:
{
lean_object* v___f_430_; 
v___f_430_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_430_, 0, v_f_429_);
return v___f_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_codRestrict(lean_object* v_M_431_, lean_object* v_N_432_, lean_object* v_inst_433_, lean_object* v_inst_434_, lean_object* v_S_435_, lean_object* v_inst_436_, lean_object* v_inst_437_, lean_object* v_f_438_, lean_object* v_s_439_, lean_object* v_h_440_){
_start:
{
lean_object* v___f_441_; 
v___f_441_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_441_, 0, v_f_438_);
return v___f_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_codRestrict___boxed(lean_object* v_M_442_, lean_object* v_N_443_, lean_object* v_inst_444_, lean_object* v_inst_445_, lean_object* v_S_446_, lean_object* v_inst_447_, lean_object* v_inst_448_, lean_object* v_f_449_, lean_object* v_s_450_, lean_object* v_h_451_){
_start:
{
lean_object* v_res_452_; 
v_res_452_ = lp_mathlib_MonoidHom_codRestrict(v_M_442_, v_N_443_, v_inst_444_, v_inst_445_, v_S_446_, v_inst_447_, v_inst_448_, v_f_449_, v_s_450_, v_h_451_);
lean_dec(v_s_450_);
lean_dec_ref(v_inst_445_);
lean_dec_ref(v_inst_444_);
return v_res_452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_codRestrict___redArg(lean_object* v_f_453_){
_start:
{
lean_object* v___f_454_; 
v___f_454_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_454_, 0, v_f_453_);
return v___f_454_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_codRestrict(lean_object* v_M_455_, lean_object* v_N_456_, lean_object* v_inst_457_, lean_object* v_inst_458_, lean_object* v_S_459_, lean_object* v_inst_460_, lean_object* v_inst_461_, lean_object* v_f_462_, lean_object* v_s_463_, lean_object* v_h_464_){
_start:
{
lean_object* v___f_465_; 
v___f_465_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_465_, 0, v_f_462_);
return v___f_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_codRestrict___boxed(lean_object* v_M_466_, lean_object* v_N_467_, lean_object* v_inst_468_, lean_object* v_inst_469_, lean_object* v_S_470_, lean_object* v_inst_471_, lean_object* v_inst_472_, lean_object* v_f_473_, lean_object* v_s_474_, lean_object* v_h_475_){
_start:
{
lean_object* v_res_476_; 
v_res_476_ = lp_mathlib_AddMonoidHom_codRestrict(v_M_466_, v_N_467_, v_inst_468_, v_inst_469_, v_S_470_, v_inst_471_, v_inst_472_, v_f_473_, v_s_474_, v_h_475_);
lean_dec(v_s_474_);
lean_dec_ref(v_inst_469_);
lean_dec_ref(v_inst_468_);
return v_res_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mrangeRestrict___redArg(lean_object* v_f_477_){
_start:
{
lean_object* v___f_478_; 
v___f_478_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_478_, 0, v_f_477_);
return v___f_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mrangeRestrict(lean_object* v_M_479_, lean_object* v_inst_480_, lean_object* v_N_481_, lean_object* v_inst_482_, lean_object* v_f_483_){
_start:
{
lean_object* v___f_484_; 
v___f_484_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_484_, 0, v_f_483_);
return v___f_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mrangeRestrict___boxed(lean_object* v_M_485_, lean_object* v_inst_486_, lean_object* v_N_487_, lean_object* v_inst_488_, lean_object* v_f_489_){
_start:
{
lean_object* v_res_490_; 
v_res_490_ = lp_mathlib_MonoidHom_mrangeRestrict(v_M_485_, v_inst_486_, v_N_487_, v_inst_488_, v_f_489_);
lean_dec_ref(v_inst_488_);
lean_dec_ref(v_inst_486_);
return v_res_490_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mrangeRestrict___redArg(lean_object* v_f_491_){
_start:
{
lean_object* v___f_492_; 
v___f_492_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_492_, 0, v_f_491_);
return v___f_492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mrangeRestrict(lean_object* v_M_493_, lean_object* v_inst_494_, lean_object* v_N_495_, lean_object* v_inst_496_, lean_object* v_f_497_){
_start:
{
lean_object* v___f_498_; 
v___f_498_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_498_, 0, v_f_497_);
return v___f_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mrangeRestrict___boxed(lean_object* v_M_499_, lean_object* v_inst_500_, lean_object* v_N_501_, lean_object* v_inst_502_, lean_object* v_f_503_){
_start:
{
lean_object* v_res_504_; 
v_res_504_ = lp_mathlib_AddMonoidHom_mrangeRestrict(v_M_499_, v_inst_500_, v_N_501_, v_inst_502_, v_f_503_);
lean_dec_ref(v_inst_502_);
lean_dec_ref(v_inst_500_);
return v_res_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mker(lean_object* v_M_505_, lean_object* v_N_506_, lean_object* v_inst_507_, lean_object* v_inst_508_, lean_object* v_F_509_, lean_object* v_inst_510_, lean_object* v_mc_511_, lean_object* v_f_512_){
_start:
{
lean_object* v___x_513_; 
v___x_513_ = lean_box(0);
return v___x_513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_mker___boxed(lean_object* v_M_514_, lean_object* v_N_515_, lean_object* v_inst_516_, lean_object* v_inst_517_, lean_object* v_F_518_, lean_object* v_inst_519_, lean_object* v_mc_520_, lean_object* v_f_521_){
_start:
{
lean_object* v_res_522_; 
v_res_522_ = lp_mathlib_MonoidHom_mker(v_M_514_, v_N_515_, v_inst_516_, v_inst_517_, v_F_518_, v_inst_519_, v_mc_520_, v_f_521_);
lean_dec(v_f_521_);
lean_dec(v_inst_519_);
lean_dec_ref(v_inst_517_);
lean_dec_ref(v_inst_516_);
return v_res_522_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mker(lean_object* v_M_523_, lean_object* v_N_524_, lean_object* v_inst_525_, lean_object* v_inst_526_, lean_object* v_F_527_, lean_object* v_inst_528_, lean_object* v_mc_529_, lean_object* v_f_530_){
_start:
{
lean_object* v___x_531_; 
v___x_531_ = lean_box(0);
return v___x_531_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_mker___boxed(lean_object* v_M_532_, lean_object* v_N_533_, lean_object* v_inst_534_, lean_object* v_inst_535_, lean_object* v_F_536_, lean_object* v_inst_537_, lean_object* v_mc_538_, lean_object* v_f_539_){
_start:
{
lean_object* v_res_540_; 
v_res_540_ = lp_mathlib_AddMonoidHom_mker(v_M_532_, v_N_533_, v_inst_534_, v_inst_535_, v_F_536_, v_inst_537_, v_mc_538_, v_f_539_);
lean_dec(v_f_539_);
lean_dec(v_inst_537_);
lean_dec_ref(v_inst_535_);
lean_dec_ref(v_inst_534_);
return v_res_540_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MonoidHom_decidableMemMker___redArg(lean_object* v_inst_541_, lean_object* v_inst_542_, lean_object* v_inst_543_, lean_object* v_f_544_, lean_object* v_x_545_){
_start:
{
lean_object* v___x_546_; lean_object* v_toOne_547_; lean_object* v___x_548_; lean_object* v___x_549_; uint8_t v___x_550_; 
v___x_546_ = lp_mathlib_MulOneClass_toMulOne___redArg(v_inst_541_);
v_toOne_547_ = lean_ctor_get(v___x_546_, 0);
lean_inc(v_toOne_547_);
lean_dec_ref(v___x_546_);
v___x_548_ = lean_apply_2(v_inst_542_, v_f_544_, v_x_545_);
v___x_549_ = lean_apply_2(v_inst_543_, v___x_548_, v_toOne_547_);
v___x_550_ = lean_unbox(v___x_549_);
return v___x_550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_decidableMemMker___redArg___boxed(lean_object* v_inst_551_, lean_object* v_inst_552_, lean_object* v_inst_553_, lean_object* v_f_554_, lean_object* v_x_555_){
_start:
{
uint8_t v_res_556_; lean_object* v_r_557_; 
v_res_556_ = lp_mathlib_MonoidHom_decidableMemMker___redArg(v_inst_551_, v_inst_552_, v_inst_553_, v_f_554_, v_x_555_);
v_r_557_ = lean_box(v_res_556_);
return v_r_557_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MonoidHom_decidableMemMker(lean_object* v_M_558_, lean_object* v_N_559_, lean_object* v_inst_560_, lean_object* v_inst_561_, lean_object* v_F_562_, lean_object* v_inst_563_, lean_object* v_mc_564_, lean_object* v_inst_565_, lean_object* v_f_566_, lean_object* v_x_567_){
_start:
{
uint8_t v___x_568_; 
v___x_568_ = lp_mathlib_MonoidHom_decidableMemMker___redArg(v_inst_561_, v_inst_563_, v_inst_565_, v_f_566_, v_x_567_);
return v___x_568_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_decidableMemMker___boxed(lean_object* v_M_569_, lean_object* v_N_570_, lean_object* v_inst_571_, lean_object* v_inst_572_, lean_object* v_F_573_, lean_object* v_inst_574_, lean_object* v_mc_575_, lean_object* v_inst_576_, lean_object* v_f_577_, lean_object* v_x_578_){
_start:
{
uint8_t v_res_579_; lean_object* v_r_580_; 
v_res_579_ = lp_mathlib_MonoidHom_decidableMemMker(v_M_569_, v_N_570_, v_inst_571_, v_inst_572_, v_F_573_, v_inst_574_, v_mc_575_, v_inst_576_, v_f_577_, v_x_578_);
lean_dec_ref(v_inst_571_);
v_r_580_ = lean_box(v_res_579_);
return v_r_580_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddMonoidHom_decidableMemMker___redArg(lean_object* v_inst_581_, lean_object* v_inst_582_, lean_object* v_inst_583_, lean_object* v_f_584_, lean_object* v_x_585_){
_start:
{
lean_object* v___x_586_; lean_object* v_toZero_587_; lean_object* v___x_588_; lean_object* v___x_589_; uint8_t v___x_590_; 
v___x_586_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v_inst_581_);
v_toZero_587_ = lean_ctor_get(v___x_586_, 0);
lean_inc(v_toZero_587_);
lean_dec_ref(v___x_586_);
v___x_588_ = lean_apply_2(v_inst_582_, v_f_584_, v_x_585_);
v___x_589_ = lean_apply_2(v_inst_583_, v___x_588_, v_toZero_587_);
v___x_590_ = lean_unbox(v___x_589_);
return v___x_590_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_decidableMemMker___redArg___boxed(lean_object* v_inst_591_, lean_object* v_inst_592_, lean_object* v_inst_593_, lean_object* v_f_594_, lean_object* v_x_595_){
_start:
{
uint8_t v_res_596_; lean_object* v_r_597_; 
v_res_596_ = lp_mathlib_AddMonoidHom_decidableMemMker___redArg(v_inst_591_, v_inst_592_, v_inst_593_, v_f_594_, v_x_595_);
v_r_597_ = lean_box(v_res_596_);
return v_r_597_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_AddMonoidHom_decidableMemMker(lean_object* v_M_598_, lean_object* v_N_599_, lean_object* v_inst_600_, lean_object* v_inst_601_, lean_object* v_F_602_, lean_object* v_inst_603_, lean_object* v_mc_604_, lean_object* v_inst_605_, lean_object* v_f_606_, lean_object* v_x_607_){
_start:
{
uint8_t v___x_608_; 
v___x_608_ = lp_mathlib_AddMonoidHom_decidableMemMker___redArg(v_inst_601_, v_inst_603_, v_inst_605_, v_f_606_, v_x_607_);
return v___x_608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_decidableMemMker___boxed(lean_object* v_M_609_, lean_object* v_N_610_, lean_object* v_inst_611_, lean_object* v_inst_612_, lean_object* v_F_613_, lean_object* v_inst_614_, lean_object* v_mc_615_, lean_object* v_inst_616_, lean_object* v_f_617_, lean_object* v_x_618_){
_start:
{
uint8_t v_res_619_; lean_object* v_r_620_; 
v_res_619_ = lp_mathlib_AddMonoidHom_decidableMemMker(v_M_609_, v_N_610_, v_inst_611_, v_inst_612_, v_F_613_, v_inst_614_, v_mc_615_, v_inst_616_, v_f_617_, v_x_618_);
lean_dec_ref(v_inst_611_);
v_r_620_ = lean_box(v_res_619_);
return v_r_620_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0(lean_object* v_f_621_, lean_object* v_x_622_){
_start:
{
lean_object* v___x_623_; 
v___x_623_ = lean_apply_1(v_f_621_, v_x_622_);
return v___x_623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_submonoidComap___redArg(lean_object* v_f_624_){
_start:
{
lean_object* v___f_625_; 
v___f_625_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_625_, 0, v_f_624_);
return v___f_625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_submonoidComap(lean_object* v_M_626_, lean_object* v_N_627_, lean_object* v_inst_628_, lean_object* v_inst_629_, lean_object* v_f_630_, lean_object* v_N_x27_631_){
_start:
{
lean_object* v___f_632_; 
v___f_632_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_632_, 0, v_f_630_);
return v___f_632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_submonoidComap___boxed(lean_object* v_M_633_, lean_object* v_N_634_, lean_object* v_inst_635_, lean_object* v_inst_636_, lean_object* v_f_637_, lean_object* v_N_x27_638_){
_start:
{
lean_object* v_res_639_; 
v_res_639_ = lp_mathlib_MonoidHom_submonoidComap(v_M_633_, v_N_634_, v_inst_635_, v_inst_636_, v_f_637_, v_N_x27_638_);
lean_dec_ref(v_inst_636_);
lean_dec_ref(v_inst_635_);
return v_res_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubmonoidComap___redArg(lean_object* v_f_640_){
_start:
{
lean_object* v___f_641_; 
v___f_641_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_641_, 0, v_f_640_);
return v___f_641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubmonoidComap(lean_object* v_M_642_, lean_object* v_N_643_, lean_object* v_inst_644_, lean_object* v_inst_645_, lean_object* v_f_646_, lean_object* v_N_x27_647_){
_start:
{
lean_object* v___f_648_; 
v___f_648_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_648_, 0, v_f_646_);
return v___f_648_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubmonoidComap___boxed(lean_object* v_M_649_, lean_object* v_N_650_, lean_object* v_inst_651_, lean_object* v_inst_652_, lean_object* v_f_653_, lean_object* v_N_x27_654_){
_start:
{
lean_object* v_res_655_; 
v_res_655_ = lp_mathlib_AddMonoidHom_addSubmonoidComap(v_M_649_, v_N_650_, v_inst_651_, v_inst_652_, v_f_653_, v_N_x27_654_);
lean_dec_ref(v_inst_652_);
lean_dec_ref(v_inst_651_);
return v_res_655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_submonoidMap___redArg(lean_object* v_f_656_){
_start:
{
lean_object* v___f_657_; 
v___f_657_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_657_, 0, v_f_656_);
return v___f_657_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_submonoidMap(lean_object* v_M_658_, lean_object* v_N_659_, lean_object* v_inst_660_, lean_object* v_inst_661_, lean_object* v_f_662_, lean_object* v_M_x27_663_){
_start:
{
lean_object* v___f_664_; 
v___f_664_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_664_, 0, v_f_662_);
return v___f_664_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_submonoidMap___boxed(lean_object* v_M_665_, lean_object* v_N_666_, lean_object* v_inst_667_, lean_object* v_inst_668_, lean_object* v_f_669_, lean_object* v_M_x27_670_){
_start:
{
lean_object* v_res_671_; 
v_res_671_ = lp_mathlib_MonoidHom_submonoidMap(v_M_665_, v_N_666_, v_inst_667_, v_inst_668_, v_f_669_, v_M_x27_670_);
lean_dec_ref(v_inst_668_);
lean_dec_ref(v_inst_667_);
return v_res_671_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubmonoidMap___redArg(lean_object* v_f_672_){
_start:
{
lean_object* v___f_673_; 
v___f_673_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_673_, 0, v_f_672_);
return v___f_673_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubmonoidMap(lean_object* v_M_674_, lean_object* v_N_675_, lean_object* v_inst_676_, lean_object* v_inst_677_, lean_object* v_f_678_, lean_object* v_M_x27_679_){
_start:
{
lean_object* v___f_680_; 
v___f_680_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_submonoidComap___redArg___lam__0), 2, 1);
lean_closure_set(v___f_680_, 0, v_f_678_);
return v___f_680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_addSubmonoidMap___boxed(lean_object* v_M_681_, lean_object* v_N_682_, lean_object* v_inst_683_, lean_object* v_inst_684_, lean_object* v_f_685_, lean_object* v_M_x27_686_){
_start:
{
lean_object* v_res_687_; 
v_res_687_ = lp_mathlib_AddMonoidHom_addSubmonoidMap(v_M_681_, v_N_682_, v_inst_683_, v_inst_684_, v_f_685_, v_M_x27_686_);
lean_dec_ref(v_inst_684_);
lean_dec_ref(v_inst_683_);
return v_res_687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_inclusion(lean_object* v_M_690_, lean_object* v_inst_691_, lean_object* v_S_692_, lean_object* v_T_693_, lean_object* v_h_694_){
_start:
{
lean_object* v___f_695_; 
v___f_695_ = ((lean_object*)(lp_mathlib_Submonoid_inclusion___closed__0));
return v___f_695_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_inclusion___boxed(lean_object* v_M_696_, lean_object* v_inst_697_, lean_object* v_S_698_, lean_object* v_T_699_, lean_object* v_h_700_){
_start:
{
lean_object* v_res_701_; 
v_res_701_ = lp_mathlib_Submonoid_inclusion(v_M_696_, v_inst_697_, v_S_698_, v_T_699_, v_h_700_);
lean_dec_ref(v_inst_697_);
return v_res_701_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_inclusion(lean_object* v_M_702_, lean_object* v_inst_703_, lean_object* v_S_704_, lean_object* v_T_705_, lean_object* v_h_706_){
_start:
{
lean_object* v___f_707_; 
v___f_707_ = ((lean_object*)(lp_mathlib_Submonoid_inclusion___closed__0));
return v___f_707_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_inclusion___boxed(lean_object* v_M_708_, lean_object* v_inst_709_, lean_object* v_S_710_, lean_object* v_T_711_, lean_object* v_h_712_){
_start:
{
lean_object* v_res_713_; 
v_res_713_ = lp_mathlib_AddSubmonoid_inclusion(v_M_708_, v_inst_709_, v_S_710_, v_T_711_, v_h_712_);
lean_dec_ref(v_inst_709_);
return v_res_713_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_pi(lean_object* v_00_u03b9_714_, lean_object* v_M_715_, lean_object* v_inst_716_, lean_object* v_I_717_, lean_object* v_S_718_){
_start:
{
lean_object* v___x_719_; 
v___x_719_ = lean_box(0);
return v___x_719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_pi___boxed(lean_object* v_00_u03b9_720_, lean_object* v_M_721_, lean_object* v_inst_722_, lean_object* v_I_723_, lean_object* v_S_724_){
_start:
{
lean_object* v_res_725_; 
v_res_725_ = lp_mathlib_Submonoid_pi(v_00_u03b9_720_, v_M_721_, v_inst_722_, v_I_723_, v_S_724_);
lean_dec_ref(v_S_724_);
lean_dec_ref(v_inst_722_);
return v_res_725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_pi(lean_object* v_00_u03b9_726_, lean_object* v_M_727_, lean_object* v_inst_728_, lean_object* v_I_729_, lean_object* v_S_730_){
_start:
{
lean_object* v___x_731_; 
v___x_731_ = lean_box(0);
return v___x_731_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubmonoid_pi___boxed(lean_object* v_00_u03b9_732_, lean_object* v_M_733_, lean_object* v_inst_734_, lean_object* v_I_735_, lean_object* v_S_736_){
_start:
{
lean_object* v_res_737_; 
v_res_737_ = lp_mathlib_AddSubmonoid_pi(v_00_u03b9_732_, v_M_733_, v_inst_734_, v_I_735_, v_S_736_);
lean_dec_ref(v_S_736_);
lean_dec_ref(v_inst_734_);
return v_res_737_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_restrict___redArg(lean_object* v_f_738_){
_start:
{
lean_object* v___x_739_; lean_object* v___f_740_; 
v___x_739_ = lp_mathlib_MonoidHom_domRestrict___redArg(v_f_738_);
v___f_740_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_740_, 0, v___x_739_);
return v___f_740_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_restrict(lean_object* v_M_741_, lean_object* v_N_742_, lean_object* v_inst_743_, lean_object* v_inst_744_, lean_object* v_M_x27_745_, lean_object* v_N_x27_746_, lean_object* v_f_747_, lean_object* v_h_748_){
_start:
{
lean_object* v___x_749_; 
v___x_749_ = lp_mathlib_MonoidHom_restrict___redArg(v_f_747_);
return v___x_749_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_restrict___boxed(lean_object* v_M_750_, lean_object* v_N_751_, lean_object* v_inst_752_, lean_object* v_inst_753_, lean_object* v_M_x27_754_, lean_object* v_N_x27_755_, lean_object* v_f_756_, lean_object* v_h_757_){
_start:
{
lean_object* v_res_758_; 
v_res_758_ = lp_mathlib_MonoidHom_restrict(v_M_750_, v_N_751_, v_inst_752_, v_inst_753_, v_M_x27_754_, v_N_x27_755_, v_f_756_, v_h_757_);
lean_dec_ref(v_inst_753_);
lean_dec_ref(v_inst_752_);
return v_res_758_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_restrict___redArg(lean_object* v_f_759_){
_start:
{
lean_object* v___x_760_; lean_object* v___f_761_; 
v___x_760_ = lp_mathlib_AddMonoidHom_domRestrict___redArg(v_f_759_);
v___f_761_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_761_, 0, v___x_760_);
return v___f_761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_restrict(lean_object* v_M_762_, lean_object* v_N_763_, lean_object* v_inst_764_, lean_object* v_inst_765_, lean_object* v_M_x27_766_, lean_object* v_N_x27_767_, lean_object* v_f_768_, lean_object* v_h_769_){
_start:
{
lean_object* v___x_770_; 
v___x_770_ = lp_mathlib_AddMonoidHom_restrict___redArg(v_f_768_);
return v___x_770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_restrict___boxed(lean_object* v_M_771_, lean_object* v_N_772_, lean_object* v_inst_773_, lean_object* v_inst_774_, lean_object* v_M_x27_775_, lean_object* v_N_x27_776_, lean_object* v_f_777_, lean_object* v_h_778_){
_start:
{
lean_object* v_res_779_; 
v_res_779_ = lp_mathlib_AddMonoidHom_restrict(v_M_771_, v_N_772_, v_inst_773_, v_inst_774_, v_M_x27_775_, v_N_x27_776_, v_f_777_, v_h_778_);
lean_dec_ref(v_inst_774_);
lean_dec_ref(v_inst_773_);
return v_res_779_;
}
}
static lean_object* _init_lp_mathlib_MulEquiv_submonoidCongr___closed__0(void){
_start:
{
lean_object* v___x_780_; 
v___x_780_ = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return v___x_780_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_submonoidCongr(lean_object* v_M_781_, lean_object* v_inst_782_, lean_object* v_S_783_, lean_object* v_T_784_, lean_object* v_h_785_){
_start:
{
lean_object* v___x_786_; 
v___x_786_ = lean_obj_once(&lp_mathlib_MulEquiv_submonoidCongr___closed__0, &lp_mathlib_MulEquiv_submonoidCongr___closed__0_once, _init_lp_mathlib_MulEquiv_submonoidCongr___closed__0);
return v___x_786_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_submonoidCongr___boxed(lean_object* v_M_787_, lean_object* v_inst_788_, lean_object* v_S_789_, lean_object* v_T_790_, lean_object* v_h_791_){
_start:
{
lean_object* v_res_792_; 
v_res_792_ = lp_mathlib_MulEquiv_submonoidCongr(v_M_787_, v_inst_788_, v_S_789_, v_T_790_, v_h_791_);
lean_dec_ref(v_inst_788_);
return v_res_792_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubmonoidCongr(lean_object* v_M_793_, lean_object* v_inst_794_, lean_object* v_S_795_, lean_object* v_T_796_, lean_object* v_h_797_){
_start:
{
lean_object* v___x_798_; 
v___x_798_ = lean_obj_once(&lp_mathlib_MulEquiv_submonoidCongr___closed__0, &lp_mathlib_MulEquiv_submonoidCongr___closed__0_once, _init_lp_mathlib_MulEquiv_submonoidCongr___closed__0);
return v___x_798_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubmonoidCongr___boxed(lean_object* v_M_799_, lean_object* v_inst_800_, lean_object* v_S_801_, lean_object* v_T_802_, lean_object* v_h_803_){
_start:
{
lean_object* v_res_804_; 
v_res_804_ = lp_mathlib_AddEquiv_addSubmonoidCongr(v_M_799_, v_inst_800_, v_S_801_, v_T_802_, v_h_803_);
lean_dec_ref(v_inst_800_);
return v_res_804_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ofLeftInverse_x27___redArg___lam__0(lean_object* v_f_805_, lean_object* v___y_806_){
_start:
{
lean_object* v___x_807_; 
v___x_807_ = lean_apply_1(v_f_805_, v___y_806_);
return v___x_807_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ofLeftInverse_x27___redArg(lean_object* v_f_808_, lean_object* v_g_809_){
_start:
{
lean_object* v___f_810_; lean_object* v___f_811_; lean_object* v___x_812_; lean_object* v___x_813_; 
v___f_810_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_ofLeftInverse_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_810_, 0, v_f_808_);
v___f_811_ = ((lean_object*)(lp_mathlib_MonoidHom_domRestrict___redArg___closed__0));
v___x_812_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_812_, 0, lean_box(0));
lean_closure_set(v___x_812_, 1, lean_box(0));
lean_closure_set(v___x_812_, 2, lean_box(0));
lean_closure_set(v___x_812_, 3, v_g_809_);
lean_closure_set(v___x_812_, 4, v___f_811_);
v___x_813_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_813_, 0, v___f_810_);
lean_ctor_set(v___x_813_, 1, v___x_812_);
return v___x_813_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ofLeftInverse_x27(lean_object* v_M_814_, lean_object* v_N_815_, lean_object* v_inst_816_, lean_object* v_inst_817_, lean_object* v_f_818_, lean_object* v_g_819_, lean_object* v_h_820_){
_start:
{
lean_object* v___x_821_; 
v___x_821_ = lp_mathlib_MulEquiv_ofLeftInverse_x27___redArg(v_f_818_, v_g_819_);
return v___x_821_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ofLeftInverse_x27___boxed(lean_object* v_M_822_, lean_object* v_N_823_, lean_object* v_inst_824_, lean_object* v_inst_825_, lean_object* v_f_826_, lean_object* v_g_827_, lean_object* v_h_828_){
_start:
{
lean_object* v_res_829_; 
v_res_829_ = lp_mathlib_MulEquiv_ofLeftInverse_x27(v_M_822_, v_N_823_, v_inst_824_, v_inst_825_, v_f_826_, v_g_827_, v_h_828_);
lean_dec_ref(v_inst_825_);
lean_dec_ref(v_inst_824_);
return v_res_829_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ofLeftInverse_x27___redArg(lean_object* v_f_830_, lean_object* v_g_831_){
_start:
{
lean_object* v___f_832_; lean_object* v___f_833_; lean_object* v___x_834_; lean_object* v___x_835_; 
v___f_832_ = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_ofLeftInverse_x27___redArg___lam__0), 2, 1);
lean_closure_set(v___f_832_, 0, v_f_830_);
v___f_833_ = ((lean_object*)(lp_mathlib_MonoidHom_domRestrict___redArg___closed__0));
v___x_834_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_834_, 0, lean_box(0));
lean_closure_set(v___x_834_, 1, lean_box(0));
lean_closure_set(v___x_834_, 2, lean_box(0));
lean_closure_set(v___x_834_, 3, v_g_831_);
lean_closure_set(v___x_834_, 4, v___f_833_);
v___x_835_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_835_, 0, v___f_832_);
lean_ctor_set(v___x_835_, 1, v___x_834_);
return v___x_835_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ofLeftInverse_x27(lean_object* v_M_836_, lean_object* v_N_837_, lean_object* v_inst_838_, lean_object* v_inst_839_, lean_object* v_f_840_, lean_object* v_g_841_, lean_object* v_h_842_){
_start:
{
lean_object* v___x_843_; 
v___x_843_ = lp_mathlib_AddEquiv_ofLeftInverse_x27___redArg(v_f_840_, v_g_841_);
return v___x_843_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ofLeftInverse_x27___boxed(lean_object* v_M_844_, lean_object* v_N_845_, lean_object* v_inst_846_, lean_object* v_inst_847_, lean_object* v_f_848_, lean_object* v_g_849_, lean_object* v_h_850_){
_start:
{
lean_object* v_res_851_; 
v_res_851_ = lp_mathlib_AddEquiv_ofLeftInverse_x27(v_M_844_, v_N_845_, v_inst_846_, v_inst_847_, v_f_848_, v_g_849_, v_h_850_);
lean_dec_ref(v_inst_847_);
lean_dec_ref(v_inst_846_);
return v_res_851_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_submonoidMap___redArg___lam__0(lean_object* v_f_852_, lean_object* v___y_853_){
_start:
{
lean_object* v_toFun_854_; lean_object* v___x_855_; 
v_toFun_854_ = lean_ctor_get(v_f_852_, 0);
lean_inc(v_toFun_854_);
lean_dec_ref(v_f_852_);
v___x_855_ = lean_apply_1(v_toFun_854_, v___y_853_);
return v___x_855_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_submonoidMap___redArg___lam__1(lean_object* v_f_856_, lean_object* v___y_857_){
_start:
{
lean_object* v_invFun_858_; lean_object* v___x_859_; 
v_invFun_858_ = lean_ctor_get(v_f_856_, 1);
lean_inc(v_invFun_858_);
lean_dec_ref(v_f_856_);
v___x_859_ = lean_apply_1(v_invFun_858_, v___y_857_);
return v___x_859_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_submonoidMap___redArg(lean_object* v_e_865_){
_start:
{
lean_object* v___x_866_; lean_object* v___x_867_; lean_object* v___x_868_; 
v___x_866_ = ((lean_object*)(lp_mathlib_MulEquiv_submonoidMap___redArg___closed__2));
v___x_867_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_866_, v_e_865_);
v___x_868_ = lp_mathlib_Equiv_image___redArg(v___x_867_);
return v___x_868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_submonoidMap(lean_object* v_M_869_, lean_object* v_N_870_, lean_object* v_inst_871_, lean_object* v_inst_872_, lean_object* v_e_873_, lean_object* v_S_874_){
_start:
{
lean_object* v___x_875_; 
v___x_875_ = lp_mathlib_MulEquiv_submonoidMap___redArg(v_e_873_);
return v___x_875_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_submonoidMap___boxed(lean_object* v_M_876_, lean_object* v_N_877_, lean_object* v_inst_878_, lean_object* v_inst_879_, lean_object* v_e_880_, lean_object* v_S_881_){
_start:
{
lean_object* v_res_882_; 
v_res_882_ = lp_mathlib_MulEquiv_submonoidMap(v_M_876_, v_N_877_, v_inst_878_, v_inst_879_, v_e_880_, v_S_881_);
lean_dec_ref(v_inst_879_);
lean_dec_ref(v_inst_878_);
return v_res_882_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubmonoidMap___redArg(lean_object* v_e_883_){
_start:
{
lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; 
v___x_884_ = ((lean_object*)(lp_mathlib_MulEquiv_submonoidMap___redArg___closed__2));
v___x_885_ = lp_mathlib_EquivLike_toEquiv___redArg(v___x_884_, v_e_883_);
v___x_886_ = lp_mathlib_Equiv_image___redArg(v___x_885_);
return v___x_886_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubmonoidMap(lean_object* v_M_887_, lean_object* v_N_888_, lean_object* v_inst_889_, lean_object* v_inst_890_, lean_object* v_e_891_, lean_object* v_S_892_){
_start:
{
lean_object* v___x_893_; 
v___x_893_ = lp_mathlib_AddEquiv_addSubmonoidMap___redArg(v_e_891_);
return v___x_893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_addSubmonoidMap___boxed(lean_object* v_M_894_, lean_object* v_N_895_, lean_object* v_inst_896_, lean_object* v_inst_897_, lean_object* v_e_898_, lean_object* v_S_899_){
_start:
{
lean_object* v_res_900_; 
v_res_900_ = lp_mathlib_AddEquiv_addSubmonoidMap(v_M_894_, v_N_895_, v_inst_896_, v_inst_897_, v_e_898_, v_S_899_);
lean_dec_ref(v_inst_897_);
lean_dec_ref(v_inst_896_);
return v_res_900_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Faithful(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulAction(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Action_Faithful(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_LibraryNote_range__copy__pattern = _init_lp_mathlib_LibraryNote_range__copy__pattern();
lean_mark_persistent(lp_mathlib_LibraryNote_range__copy__pattern);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Action_Faithful(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulAction(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Action_Faithful(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulAction(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
}
#ifdef __cplusplus
}
#endif
