// Lean compiler output
// Module: Mathlib.Data.SetLike.Basic
// Imports: public import Init public meta import Init public import Mathlib.Tactic.Monotonicity.Attr public import Mathlib.Tactic.SetLike public import Mathlib.Data.Set.Basic
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
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_SubExpr_Pos_push(lean_object*, lean_object*);
lean_object* l_Lean_Expr_bindingBody_x21(lean_object*);
lean_object* lean_expr_instantiate1(lean_object*, lean_object*);
uint8_t l_Lean_Expr_binderInfo(lean_object*);
lean_object* l_Lean_Expr_bindingDomain_x21(lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_delab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
lean_object* l_Lean_SubExpr_Pos_pushNaryArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_failure___redArg();
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOf(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_instCoeTCSet(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_instMembership(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_instCoeSortType(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00SetLike_delabSubtypeSetLike_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00SetLike_delabSubtypeSetLike_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00SetLike_delabSubtypeSetLike_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00SetLike_delabSubtypeSetLike_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4___redArg(lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_delab___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__0 = (const lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "coeSortNotation"};
static const lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__1 = (const lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__1_value;
static const lean_ctor_object lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(24, 190, 27, 248, 164, 200, 2, 94)}};
static const lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__2 = (const lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "↥"};
static const lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__3 = (const lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__3_value;
static const lean_string_object lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "SetLike"};
static const lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__4 = (const lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "instMembership"};
static const lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__5 = (const lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__5_value;
static const lean_ctor_object lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(146, 248, 10, 158, 176, 176, 178, 2)}};
static const lean_ctor_object lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__6_value_aux_0),((lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(142, 227, 62, 155, 191, 114, 172, 231)}};
static const lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__6 = (const lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__6_value;
static const lean_string_object lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Membership"};
static const lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__7 = (const lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__7_value;
static const lean_string_object lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "mem"};
static const lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__8 = (const lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__8_value;
static const lean_ctor_object lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(205, 217, 109, 94, 255, 55, 82, 109)}};
static const lean_ctor_object lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__9_value_aux_0),((lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__8_value),LEAN_SCALAR_PTR_LITERAL(224, 90, 126, 237, 128, 148, 153, 69)}};
static const lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__9 = (const lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_SetLike_delabSubtypeSetLike___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___closed__0 = (const lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___closed__0_value;
static const lean_closure_object lp_mathlib_SetLike_delabSubtypeSetLike___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___closed__1 = (const lean_object*)&lp_mathlib_SetLike_delabSubtypeSetLike___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_SetLike_delabSubtypeSetLike(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__0 = (const lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__0_value;
static const lean_string_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__1 = (const lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__1_value;
static const lean_string_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__2 = (const lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__2_value;
static const lean_string_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__3 = (const lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__3_value;
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__4 = (const lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__4_value;
static const lean_array_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__5 = (const lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__5_value;
static const lean_string_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__6 = (const lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__6_value;
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__7 = (const lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__7_value;
static const lean_string_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__8 = (const lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__8_value;
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__9 = (const lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__9_value;
static const lean_string_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "simp"};
static const lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__10 = (const lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__10_value;
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(50, 13, 241, 145, 67, 153, 105, 177)}};
static const lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__11 = (const lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__11_value;
static lean_once_cell_t lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__12;
static lean_once_cell_t lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__13;
static const lean_string_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__14 = (const lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__14_value;
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__15_value_aux_0),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__15_value_aux_1),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__15_value_aux_2),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__15 = (const lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__15_value;
static const lean_ctor_object lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__9_value),((lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__5_value)}};
static const lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__16 = (const lean_object*)&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__16_value;
static lean_once_cell_t lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__17;
static lean_once_cell_t lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__18;
static lean_once_cell_t lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__19;
static lean_once_cell_t lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__20;
static lean_once_cell_t lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__21;
static lean_once_cell_t lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__22;
static lean_once_cell_t lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__23;
static lean_once_cell_t lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__24;
static lean_once_cell_t lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__25;
static lean_once_cell_t lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__26;
static lean_once_cell_t lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__27;
static lean_once_cell_t lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__28;
static lean_once_cell_t lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__29;
static lean_once_cell_t lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__30;
LEAN_EXPORT lean_object* lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_LE_ofSetLike(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_PartialOrder_ofSetLike___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_PartialOrder_ofSetLike___closed__0 = (const lean_object*)&lp_mathlib_PartialOrder_ofSetLike___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_instSubtypeSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_instSubtype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SetLike_instCoeTCSet(lean_object* v_A_1_, lean_object* v_B_2_, lean_object* v_i_3_){
_start:
{
lean_object* v___x_4_; 
v___x_4_ = lean_box(0);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_instMembership(lean_object* v_A_5_, lean_object* v_B_6_, lean_object* v_i_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lean_box(0);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_instCoeSortType(lean_object* v_A_9_, lean_object* v_B_10_, lean_object* v_i_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_box(0);
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00SetLike_delabSubtypeSetLike_spec__0___redArg(lean_object* v___y_13_){
_start:
{
lean_object* v_subExpr_15_; lean_object* v_expr_16_; lean_object* v___x_17_; 
v_subExpr_15_ = lean_ctor_get(v___y_13_, 3);
v_expr_16_ = lean_ctor_get(v_subExpr_15_, 0);
lean_inc_ref(v_expr_16_);
v___x_17_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_17_, 0, v_expr_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00SetLike_delabSubtypeSetLike_spec__0___redArg___boxed(lean_object* v___y_18_, lean_object* v___y_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00SetLike_delabSubtypeSetLike_spec__0___redArg(v___y_18_);
lean_dec_ref(v___y_18_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00SetLike_delabSubtypeSetLike_spec__0(lean_object* v___y_21_, lean_object* v___y_22_, lean_object* v___y_23_, lean_object* v___y_24_, lean_object* v___y_25_, lean_object* v___y_26_){
_start:
{
lean_object* v___x_28_; 
v___x_28_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00SetLike_delabSubtypeSetLike_spec__0___redArg(v___y_21_);
return v___x_28_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00SetLike_delabSubtypeSetLike_spec__0___boxed(lean_object* v___y_29_, lean_object* v___y_30_, lean_object* v___y_31_, lean_object* v___y_32_, lean_object* v___y_33_, lean_object* v___y_34_, lean_object* v___y_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00SetLike_delabSubtypeSetLike_spec__0(v___y_29_, v___y_30_, v___y_31_, v___y_32_, v___y_33_, v___y_34_);
lean_dec(v___y_34_);
lean_dec_ref(v___y_33_);
lean_dec(v___y_32_);
lean_dec_ref(v___y_31_);
lean_dec(v___y_30_);
lean_dec_ref(v___y_29_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___redArg(lean_object* v_child_37_, lean_object* v_childIdx_38_, lean_object* v_x_39_, lean_object* v___y_40_, lean_object* v___y_41_, lean_object* v___y_42_, lean_object* v___y_43_, lean_object* v___y_44_, lean_object* v___y_45_){
_start:
{
lean_object* v_subExpr_47_; lean_object* v_optionsPerPos_48_; lean_object* v_currNamespace_49_; lean_object* v_openDecls_50_; uint8_t v_inPattern_51_; lean_object* v_depth_52_; lean_object* v_lctxInitIndices_53_; lean_object* v_pos_54_; lean_object* v___x_55_; lean_object* v___x_56_; lean_object* v___x_57_; lean_object* v___x_58_; 
v_subExpr_47_ = lean_ctor_get(v___y_40_, 3);
v_optionsPerPos_48_ = lean_ctor_get(v___y_40_, 0);
v_currNamespace_49_ = lean_ctor_get(v___y_40_, 1);
v_openDecls_50_ = lean_ctor_get(v___y_40_, 2);
v_inPattern_51_ = lean_ctor_get_uint8(v___y_40_, sizeof(void*)*6);
v_depth_52_ = lean_ctor_get(v___y_40_, 4);
v_lctxInitIndices_53_ = lean_ctor_get(v___y_40_, 5);
v_pos_54_ = lean_ctor_get(v_subExpr_47_, 1);
v___x_55_ = l_Lean_SubExpr_Pos_push(v_pos_54_, v_childIdx_38_);
v___x_56_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_56_, 0, v_child_37_);
lean_ctor_set(v___x_56_, 1, v___x_55_);
lean_inc(v_lctxInitIndices_53_);
lean_inc(v_depth_52_);
lean_inc(v_openDecls_50_);
lean_inc(v_currNamespace_49_);
lean_inc(v_optionsPerPos_48_);
v___x_57_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_57_, 0, v_optionsPerPos_48_);
lean_ctor_set(v___x_57_, 1, v_currNamespace_49_);
lean_ctor_set(v___x_57_, 2, v_openDecls_50_);
lean_ctor_set(v___x_57_, 3, v___x_56_);
lean_ctor_set(v___x_57_, 4, v_depth_52_);
lean_ctor_set(v___x_57_, 5, v_lctxInitIndices_53_);
lean_ctor_set_uint8(v___x_57_, sizeof(void*)*6, v_inPattern_51_);
lean_inc(v___y_45_);
lean_inc_ref(v___y_44_);
lean_inc(v___y_43_);
lean_inc_ref(v___y_42_);
lean_inc(v___y_41_);
v___x_58_ = lean_apply_7(v_x_39_, v___x_57_, v___y_41_, v___y_42_, v___y_43_, v___y_44_, v___y_45_, lean_box(0));
return v___x_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___redArg___boxed(lean_object* v_child_59_, lean_object* v_childIdx_60_, lean_object* v_x_61_, lean_object* v___y_62_, lean_object* v___y_63_, lean_object* v___y_64_, lean_object* v___y_65_, lean_object* v___y_66_, lean_object* v___y_67_, lean_object* v___y_68_){
_start:
{
lean_object* v_res_69_; 
v_res_69_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___redArg(v_child_59_, v_childIdx_60_, v_x_61_, v___y_62_, v___y_63_, v___y_64_, v___y_65_, v___y_66_, v___y_67_);
lean_dec(v___y_67_);
lean_dec_ref(v___y_66_);
lean_dec(v___y_65_);
lean_dec_ref(v___y_64_);
lean_dec(v___y_63_);
lean_dec_ref(v___y_62_);
return v_res_69_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3___redArg(lean_object* v_x_70_, lean_object* v___y_71_, lean_object* v___y_72_, lean_object* v___y_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_){
_start:
{
lean_object* v___x_78_; lean_object* v_a_79_; lean_object* v___x_80_; lean_object* v___x_81_; lean_object* v___x_82_; 
v___x_78_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00SetLike_delabSubtypeSetLike_spec__0___redArg(v___y_71_);
v_a_79_ = lean_ctor_get(v___x_78_, 0);
lean_inc(v_a_79_);
lean_dec_ref(v___x_78_);
v___x_80_ = l_Lean_Expr_appArg_x21(v_a_79_);
lean_dec(v_a_79_);
v___x_81_ = lean_unsigned_to_nat(1u);
v___x_82_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___redArg(v___x_80_, v___x_81_, v_x_70_, v___y_71_, v___y_72_, v___y_73_, v___y_74_, v___y_75_, v___y_76_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3___redArg___boxed(lean_object* v_x_83_, lean_object* v___y_84_, lean_object* v___y_85_, lean_object* v___y_86_, lean_object* v___y_87_, lean_object* v___y_88_, lean_object* v___y_89_, lean_object* v___y_90_){
_start:
{
lean_object* v_res_91_; 
v_res_91_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3___redArg(v_x_83_, v___y_84_, v___y_85_, v___y_86_, v___y_87_, v___y_88_, v___y_89_);
lean_dec(v___y_89_);
lean_dec_ref(v___y_88_);
lean_dec(v___y_87_);
lean_dec_ref(v___y_86_);
lean_dec(v___y_85_);
lean_dec_ref(v___y_84_);
return v_res_91_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4___redArg___lam__0(lean_object* v_k_92_, lean_object* v___y_93_, lean_object* v___y_94_, lean_object* v_b_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_){
_start:
{
lean_object* v___x_101_; 
lean_inc(v___y_99_);
lean_inc_ref(v___y_98_);
lean_inc(v___y_97_);
lean_inc_ref(v___y_96_);
lean_inc(v___y_94_);
lean_inc_ref(v___y_93_);
v___x_101_ = lean_apply_8(v_k_92_, v_b_95_, v___y_93_, v___y_94_, v___y_96_, v___y_97_, v___y_98_, v___y_99_, lean_box(0));
return v___x_101_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4___redArg___lam__0___boxed(lean_object* v_k_102_, lean_object* v___y_103_, lean_object* v___y_104_, lean_object* v_b_105_, lean_object* v___y_106_, lean_object* v___y_107_, lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_){
_start:
{
lean_object* v_res_111_; 
v_res_111_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4___redArg___lam__0(v_k_102_, v___y_103_, v___y_104_, v_b_105_, v___y_106_, v___y_107_, v___y_108_, v___y_109_);
lean_dec(v___y_109_);
lean_dec_ref(v___y_108_);
lean_dec(v___y_107_);
lean_dec_ref(v___y_106_);
lean_dec(v___y_104_);
lean_dec_ref(v___y_103_);
return v_res_111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4___redArg(lean_object* v_name_112_, uint8_t v_bi_113_, lean_object* v_type_114_, lean_object* v_k_115_, uint8_t v_kind_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_){
_start:
{
lean_object* v___f_124_; lean_object* v___x_125_; 
lean_inc(v___y_118_);
lean_inc_ref(v___y_117_);
v___f_124_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4___redArg___lam__0___boxed), 9, 3);
lean_closure_set(v___f_124_, 0, v_k_115_);
lean_closure_set(v___f_124_, 1, v___y_117_);
lean_closure_set(v___f_124_, 2, v___y_118_);
v___x_125_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalDeclImp(lean_box(0), v_name_112_, v_bi_113_, v_type_114_, v___f_124_, v_kind_116_, v___y_119_, v___y_120_, v___y_121_, v___y_122_);
if (lean_obj_tag(v___x_125_) == 0)
{
return v___x_125_;
}
else
{
lean_object* v_a_126_; lean_object* v___x_128_; uint8_t v_isShared_129_; uint8_t v_isSharedCheck_133_; 
v_a_126_ = lean_ctor_get(v___x_125_, 0);
v_isSharedCheck_133_ = !lean_is_exclusive(v___x_125_);
if (v_isSharedCheck_133_ == 0)
{
v___x_128_ = v___x_125_;
v_isShared_129_ = v_isSharedCheck_133_;
goto v_resetjp_127_;
}
else
{
lean_inc(v_a_126_);
lean_dec(v___x_125_);
v___x_128_ = lean_box(0);
v_isShared_129_ = v_isSharedCheck_133_;
goto v_resetjp_127_;
}
v_resetjp_127_:
{
lean_object* v___x_131_; 
if (v_isShared_129_ == 0)
{
v___x_131_ = v___x_128_;
goto v_reusejp_130_;
}
else
{
lean_object* v_reuseFailAlloc_132_; 
v_reuseFailAlloc_132_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_132_, 0, v_a_126_);
v___x_131_ = v_reuseFailAlloc_132_;
goto v_reusejp_130_;
}
v_reusejp_130_:
{
return v___x_131_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4___redArg___boxed(lean_object* v_name_134_, lean_object* v_bi_135_, lean_object* v_type_136_, lean_object* v_k_137_, lean_object* v_kind_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_, lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_){
_start:
{
uint8_t v_bi_boxed_146_; uint8_t v_kind_boxed_147_; lean_object* v_res_148_; 
v_bi_boxed_146_ = lean_unbox(v_bi_135_);
v_kind_boxed_147_ = lean_unbox(v_kind_138_);
v_res_148_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4___redArg(v_name_134_, v_bi_boxed_146_, v_type_136_, v_k_137_, v_kind_boxed_147_, v___y_139_, v___y_140_, v___y_141_, v___y_142_, v___y_143_, v___y_144_);
lean_dec(v___y_144_);
lean_dec_ref(v___y_143_);
lean_dec(v___y_142_);
lean_dec_ref(v___y_141_);
lean_dec(v___y_140_);
lean_dec_ref(v___y_139_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__8___redArg(lean_object* v___x_149_, lean_object* v_child_150_, lean_object* v_childIdx_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_){
_start:
{
lean_object* v_subExpr_159_; lean_object* v_optionsPerPos_160_; lean_object* v_currNamespace_161_; lean_object* v_openDecls_162_; uint8_t v_inPattern_163_; lean_object* v_depth_164_; lean_object* v_lctxInitIndices_165_; lean_object* v_pos_166_; lean_object* v___x_167_; lean_object* v___x_168_; lean_object* v___x_169_; lean_object* v___x_170_; 
v_subExpr_159_ = lean_ctor_get(v___y_152_, 3);
v_optionsPerPos_160_ = lean_ctor_get(v___y_152_, 0);
v_currNamespace_161_ = lean_ctor_get(v___y_152_, 1);
v_openDecls_162_ = lean_ctor_get(v___y_152_, 2);
v_inPattern_163_ = lean_ctor_get_uint8(v___y_152_, sizeof(void*)*6);
v_depth_164_ = lean_ctor_get(v___y_152_, 4);
v_lctxInitIndices_165_ = lean_ctor_get(v___y_152_, 5);
v_pos_166_ = lean_ctor_get(v_subExpr_159_, 1);
v___x_167_ = l_Lean_SubExpr_Pos_push(v_pos_166_, v_childIdx_151_);
v___x_168_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_168_, 0, v_child_150_);
lean_ctor_set(v___x_168_, 1, v___x_167_);
lean_inc(v_lctxInitIndices_165_);
lean_inc(v_depth_164_);
lean_inc(v_openDecls_162_);
lean_inc(v_currNamespace_161_);
lean_inc(v_optionsPerPos_160_);
v___x_169_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_169_, 0, v_optionsPerPos_160_);
lean_ctor_set(v___x_169_, 1, v_currNamespace_161_);
lean_ctor_set(v___x_169_, 2, v_openDecls_162_);
lean_ctor_set(v___x_169_, 3, v___x_168_);
lean_ctor_set(v___x_169_, 4, v_depth_164_);
lean_ctor_set(v___x_169_, 5, v_lctxInitIndices_165_);
lean_ctor_set_uint8(v___x_169_, sizeof(void*)*6, v_inPattern_163_);
lean_inc(v___y_157_);
lean_inc_ref(v___y_156_);
lean_inc(v___y_155_);
lean_inc_ref(v___y_154_);
lean_inc(v___y_153_);
v___x_170_ = lean_apply_7(v___x_149_, v___x_169_, v___y_153_, v___y_154_, v___y_155_, v___y_156_, v___y_157_, lean_box(0));
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__8___redArg___boxed(lean_object* v___x_171_, lean_object* v_child_172_, lean_object* v_childIdx_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_){
_start:
{
lean_object* v_res_181_; 
v_res_181_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__8___redArg(v___x_171_, v_child_172_, v_childIdx_173_, v___y_174_, v___y_175_, v___y_176_, v___y_177_, v___y_178_, v___y_179_);
lean_dec(v___y_179_);
lean_dec_ref(v___y_178_);
lean_dec(v___y_177_);
lean_dec_ref(v___y_176_);
lean_dec(v___y_175_);
lean_dec_ref(v___y_174_);
return v_res_181_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3___redArg___lam__0(lean_object* v_v_182_, lean_object* v_a_183_, lean_object* v_x_184_, lean_object* v_fvar_185_, lean_object* v___y_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_, lean_object* v___y_190_, lean_object* v___y_191_){
_start:
{
lean_object* v___x_193_; 
lean_inc(v___y_191_);
lean_inc_ref(v___y_190_);
lean_inc(v___y_189_);
lean_inc_ref(v___y_188_);
lean_inc(v___y_187_);
lean_inc_ref(v___y_186_);
lean_inc_ref(v_fvar_185_);
v___x_193_ = lean_apply_8(v_v_182_, v_fvar_185_, v___y_186_, v___y_187_, v___y_188_, v___y_189_, v___y_190_, v___y_191_, lean_box(0));
if (lean_obj_tag(v___x_193_) == 0)
{
lean_object* v_a_194_; lean_object* v___x_195_; lean_object* v___x_196_; lean_object* v___x_197_; lean_object* v___x_198_; lean_object* v___x_199_; 
v_a_194_ = lean_ctor_get(v___x_193_, 0);
lean_inc(v_a_194_);
lean_dec_ref_known(v___x_193_, 1);
v___x_195_ = l_Lean_Expr_bindingBody_x21(v_a_183_);
v___x_196_ = lean_expr_instantiate1(v___x_195_, v_fvar_185_);
lean_dec_ref(v_fvar_185_);
lean_dec_ref(v___x_195_);
v___x_197_ = lean_unsigned_to_nat(1u);
v___x_198_ = lean_apply_1(v_x_184_, v_a_194_);
v___x_199_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__8___redArg(v___x_198_, v___x_196_, v___x_197_, v___y_186_, v___y_187_, v___y_188_, v___y_189_, v___y_190_, v___y_191_);
return v___x_199_;
}
else
{
lean_object* v_a_200_; lean_object* v___x_202_; uint8_t v_isShared_203_; uint8_t v_isSharedCheck_207_; 
lean_dec_ref(v_fvar_185_);
lean_dec_ref(v_x_184_);
v_a_200_ = lean_ctor_get(v___x_193_, 0);
v_isSharedCheck_207_ = !lean_is_exclusive(v___x_193_);
if (v_isSharedCheck_207_ == 0)
{
v___x_202_ = v___x_193_;
v_isShared_203_ = v_isSharedCheck_207_;
goto v_resetjp_201_;
}
else
{
lean_inc(v_a_200_);
lean_dec(v___x_193_);
v___x_202_ = lean_box(0);
v_isShared_203_ = v_isSharedCheck_207_;
goto v_resetjp_201_;
}
v_resetjp_201_:
{
lean_object* v___x_205_; 
if (v_isShared_203_ == 0)
{
v___x_205_ = v___x_202_;
goto v_reusejp_204_;
}
else
{
lean_object* v_reuseFailAlloc_206_; 
v_reuseFailAlloc_206_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_206_, 0, v_a_200_);
v___x_205_ = v_reuseFailAlloc_206_;
goto v_reusejp_204_;
}
v_reusejp_204_:
{
return v___x_205_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3___redArg___lam__0___boxed(lean_object* v_v_208_, lean_object* v_a_209_, lean_object* v_x_210_, lean_object* v_fvar_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_, lean_object* v___y_216_, lean_object* v___y_217_, lean_object* v___y_218_){
_start:
{
lean_object* v_res_219_; 
v_res_219_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3___redArg___lam__0(v_v_208_, v_a_209_, v_x_210_, v_fvar_211_, v___y_212_, v___y_213_, v___y_214_, v___y_215_, v___y_216_, v___y_217_);
lean_dec(v___y_217_);
lean_dec_ref(v___y_216_);
lean_dec(v___y_215_);
lean_dec_ref(v___y_214_);
lean_dec(v___y_213_);
lean_dec_ref(v___y_212_);
lean_dec_ref(v_a_209_);
return v_res_219_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3___redArg(lean_object* v_n_220_, lean_object* v_v_221_, lean_object* v_x_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_, lean_object* v___y_226_, lean_object* v___y_227_, lean_object* v___y_228_){
_start:
{
lean_object* v___x_230_; lean_object* v_a_231_; lean_object* v___f_232_; uint8_t v___x_233_; lean_object* v___x_234_; uint8_t v___x_235_; lean_object* v___x_236_; 
v___x_230_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00SetLike_delabSubtypeSetLike_spec__0___redArg(v___y_223_);
v_a_231_ = lean_ctor_get(v___x_230_, 0);
lean_inc_n(v_a_231_, 2);
lean_dec_ref(v___x_230_);
v___f_232_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3___redArg___lam__0___boxed), 11, 3);
lean_closure_set(v___f_232_, 0, v_v_221_);
lean_closure_set(v___f_232_, 1, v_a_231_);
lean_closure_set(v___f_232_, 2, v_x_222_);
v___x_233_ = l_Lean_Expr_binderInfo(v_a_231_);
v___x_234_ = l_Lean_Expr_bindingDomain_x21(v_a_231_);
lean_dec(v_a_231_);
v___x_235_ = 0;
v___x_236_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4___redArg(v_n_220_, v___x_233_, v___x_234_, v___f_232_, v___x_235_, v___y_223_, v___y_224_, v___y_225_, v___y_226_, v___y_227_, v___y_228_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3___redArg___boxed(lean_object* v_n_237_, lean_object* v_v_238_, lean_object* v_x_239_, lean_object* v___y_240_, lean_object* v___y_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_, lean_object* v___y_245_, lean_object* v___y_246_){
_start:
{
lean_object* v_res_247_; 
v_res_247_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3___redArg(v_n_237_, v_v_238_, v_x_239_, v___y_240_, v___y_241_, v___y_242_, v___y_243_, v___y_244_, v___y_245_);
lean_dec(v___y_245_);
lean_dec_ref(v___y_244_);
lean_dec(v___y_243_);
lean_dec_ref(v___y_242_);
lean_dec(v___y_241_);
lean_dec_ref(v___y_240_);
return v_res_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___lam__1(lean_object* v_x_248_, lean_object* v___y_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_, lean_object* v___y_253_, lean_object* v___y_254_){
_start:
{
lean_object* v___x_256_; lean_object* v___x_257_; 
v___x_256_ = lean_box(0);
v___x_257_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_257_, 0, v___x_256_);
return v___x_257_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___lam__1___boxed(lean_object* v_x_258_, lean_object* v___y_259_, lean_object* v___y_260_, lean_object* v___y_261_, lean_object* v___y_262_, lean_object* v___y_263_, lean_object* v___y_264_, lean_object* v___y_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___lam__1(v_x_258_, v___y_259_, v___y_260_, v___y_261_, v___y_262_, v___y_263_, v___y_264_);
lean_dec(v___y_264_);
lean_dec_ref(v___y_263_);
lean_dec(v___y_262_);
lean_dec_ref(v___y_261_);
lean_dec(v___y_260_);
lean_dec_ref(v___y_259_);
lean_dec_ref(v_x_258_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___lam__0(lean_object* v_x_267_, lean_object* v_x_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_){
_start:
{
lean_object* v___x_276_; 
lean_inc(v___y_274_);
lean_inc_ref(v___y_273_);
lean_inc(v___y_272_);
lean_inc_ref(v___y_271_);
lean_inc(v___y_270_);
lean_inc_ref(v___y_269_);
v___x_276_ = lean_apply_7(v_x_267_, v___y_269_, v___y_270_, v___y_271_, v___y_272_, v___y_273_, v___y_274_, lean_box(0));
return v___x_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___lam__0___boxed(lean_object* v_x_277_, lean_object* v_x_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_, lean_object* v___y_283_, lean_object* v___y_284_, lean_object* v___y_285_){
_start:
{
lean_object* v_res_286_; 
v_res_286_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___lam__0(v_x_277_, v_x_278_, v___y_279_, v___y_280_, v___y_281_, v___y_282_, v___y_283_, v___y_284_);
lean_dec(v___y_284_);
lean_dec_ref(v___y_283_);
lean_dec(v___y_282_);
lean_dec_ref(v___y_281_);
lean_dec(v___y_280_);
lean_dec_ref(v___y_279_);
return v_res_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg(lean_object* v_n_288_, lean_object* v_x_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_){
_start:
{
lean_object* v___f_297_; lean_object* v___f_298_; lean_object* v___x_299_; 
v___f_297_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___lam__0___boxed), 9, 1);
lean_closure_set(v___f_297_, 0, v_x_289_);
v___f_298_ = ((lean_object*)(lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___closed__0));
v___x_299_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3___redArg(v_n_288_, v___f_298_, v___f_297_, v___y_290_, v___y_291_, v___y_292_, v___y_293_, v___y_294_, v___y_295_);
return v___x_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg___boxed(lean_object* v_n_300_, lean_object* v_x_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_){
_start:
{
lean_object* v_res_309_; 
v_res_309_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg(v_n_300_, v_x_301_, v___y_302_, v___y_303_, v___y_304_, v___y_305_, v___y_306_, v___y_307_);
lean_dec(v___y_307_);
lean_dec_ref(v___y_306_);
lean_dec(v___y_305_);
lean_dec_ref(v___y_304_);
lean_dec(v___y_303_);
lean_dec_ref(v___y_302_);
return v_res_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2(lean_object* v_00_u03b1_310_, lean_object* v_n_311_, lean_object* v_x_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_, lean_object* v___y_317_, lean_object* v___y_318_){
_start:
{
lean_object* v___x_320_; 
v___x_320_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___redArg(v_n_311_, v_x_312_, v___y_313_, v___y_314_, v___y_315_, v___y_316_, v___y_317_, v___y_318_);
return v___x_320_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___boxed(lean_object* v_00_u03b1_321_, lean_object* v_n_322_, lean_object* v_x_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_, lean_object* v___y_330_){
_start:
{
lean_object* v_res_331_; 
v_res_331_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2(v_00_u03b1_321_, v_n_322_, v_x_323_, v___y_324_, v___y_325_, v___y_326_, v___y_327_, v___y_328_, v___y_329_);
lean_dec(v___y_329_);
lean_dec_ref(v___y_328_);
lean_dec(v___y_327_);
lean_dec_ref(v___y_326_);
lean_dec(v___y_325_);
lean_dec_ref(v___y_324_);
return v_res_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1_spec__1___redArg(lean_object* v___y_332_){
_start:
{
lean_object* v_subExpr_334_; lean_object* v_pos_335_; lean_object* v___x_336_; 
v_subExpr_334_ = lean_ctor_get(v___y_332_, 3);
v_pos_335_ = lean_ctor_get(v_subExpr_334_, 1);
lean_inc(v_pos_335_);
v___x_336_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_336_, 0, v_pos_335_);
return v___x_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1_spec__1___redArg___boxed(lean_object* v___y_337_, lean_object* v___y_338_){
_start:
{
lean_object* v_res_339_; 
v_res_339_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1_spec__1___redArg(v___y_337_);
lean_dec_ref(v___y_337_);
return v_res_339_;
}
}
static lean_object* _init_lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_340_; lean_object* v_dummy_341_; 
v___x_340_ = lean_box(0);
v_dummy_341_ = l_Lean_Expr_sort___override(v___x_340_);
return v_dummy_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___redArg(lean_object* v_argIdx_342_, lean_object* v_x_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_){
_start:
{
lean_object* v___x_351_; lean_object* v_a_352_; lean_object* v___x_353_; lean_object* v_a_354_; lean_object* v_optionsPerPos_355_; lean_object* v_currNamespace_356_; lean_object* v_openDecls_357_; uint8_t v_inPattern_358_; lean_object* v_depth_359_; lean_object* v_lctxInitIndices_360_; lean_object* v_nargs_361_; lean_object* v___x_362_; lean_object* v_dummy_363_; lean_object* v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v_args_367_; lean_object* v___x_368_; lean_object* v_newPos_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; 
v___x_351_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00SetLike_delabSubtypeSetLike_spec__0___redArg(v___y_344_);
v_a_352_ = lean_ctor_get(v___x_351_, 0);
lean_inc(v_a_352_);
lean_dec_ref(v___x_351_);
v___x_353_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1_spec__1___redArg(v___y_344_);
v_a_354_ = lean_ctor_get(v___x_353_, 0);
lean_inc(v_a_354_);
lean_dec_ref(v___x_353_);
v_optionsPerPos_355_ = lean_ctor_get(v___y_344_, 0);
v_currNamespace_356_ = lean_ctor_get(v___y_344_, 1);
v_openDecls_357_ = lean_ctor_get(v___y_344_, 2);
v_inPattern_358_ = lean_ctor_get_uint8(v___y_344_, sizeof(void*)*6);
v_depth_359_ = lean_ctor_get(v___y_344_, 4);
v_lctxInitIndices_360_ = lean_ctor_get(v___y_344_, 5);
v_nargs_361_ = l_Lean_Expr_getAppNumArgs(v_a_352_);
v___x_362_ = l_Lean_instInhabitedExpr;
v_dummy_363_ = lean_obj_once(&lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___redArg___closed__0, &lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___redArg___closed__0);
lean_inc(v_nargs_361_);
v___x_364_ = lean_mk_array(v_nargs_361_, v_dummy_363_);
v___x_365_ = lean_unsigned_to_nat(1u);
v___x_366_ = lean_nat_sub(v_nargs_361_, v___x_365_);
lean_dec(v_nargs_361_);
v_args_367_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_352_, v___x_364_, v___x_366_);
v___x_368_ = lean_array_get_size(v_args_367_);
v_newPos_369_ = l_Lean_SubExpr_Pos_pushNaryArg(v___x_368_, v_argIdx_342_, v_a_354_);
lean_dec(v_a_354_);
v___x_370_ = lean_array_get(v___x_362_, v_args_367_, v_argIdx_342_);
lean_dec_ref(v_args_367_);
v___x_371_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_371_, 0, v___x_370_);
lean_ctor_set(v___x_371_, 1, v_newPos_369_);
lean_inc(v_lctxInitIndices_360_);
lean_inc(v_depth_359_);
lean_inc(v_openDecls_357_);
lean_inc(v_currNamespace_356_);
lean_inc(v_optionsPerPos_355_);
v___x_372_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_372_, 0, v_optionsPerPos_355_);
lean_ctor_set(v___x_372_, 1, v_currNamespace_356_);
lean_ctor_set(v___x_372_, 2, v_openDecls_357_);
lean_ctor_set(v___x_372_, 3, v___x_371_);
lean_ctor_set(v___x_372_, 4, v_depth_359_);
lean_ctor_set(v___x_372_, 5, v_lctxInitIndices_360_);
lean_ctor_set_uint8(v___x_372_, sizeof(void*)*6, v_inPattern_358_);
lean_inc(v___y_349_);
lean_inc_ref(v___y_348_);
lean_inc(v___y_347_);
lean_inc_ref(v___y_346_);
lean_inc(v___y_345_);
v___x_373_ = lean_apply_7(v_x_343_, v___x_372_, v___y_345_, v___y_346_, v___y_347_, v___y_348_, v___y_349_, lean_box(0));
return v___x_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___redArg___boxed(lean_object* v_argIdx_374_, lean_object* v_x_375_, lean_object* v___y_376_, lean_object* v___y_377_, lean_object* v___y_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_){
_start:
{
lean_object* v_res_383_; 
v_res_383_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___redArg(v_argIdx_374_, v_x_375_, v___y_376_, v___y_377_, v___y_378_, v___y_379_, v___y_380_, v___y_381_);
lean_dec(v___y_381_);
lean_dec_ref(v___y_380_);
lean_dec(v___y_379_);
lean_dec_ref(v___y_378_);
lean_dec(v___y_377_);
lean_dec_ref(v___y_376_);
lean_dec(v_argIdx_374_);
return v_res_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1(lean_object* v_00_u03b1_384_, lean_object* v_argIdx_385_, lean_object* v_x_386_, lean_object* v___y_387_, lean_object* v___y_388_, lean_object* v___y_389_, lean_object* v___y_390_, lean_object* v___y_391_, lean_object* v___y_392_){
_start:
{
lean_object* v___x_394_; 
v___x_394_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___redArg(v_argIdx_385_, v_x_386_, v___y_387_, v___y_388_, v___y_389_, v___y_390_, v___y_391_, v___y_392_);
return v___x_394_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___boxed(lean_object* v_00_u03b1_395_, lean_object* v_argIdx_396_, lean_object* v_x_397_, lean_object* v___y_398_, lean_object* v___y_399_, lean_object* v___y_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_, lean_object* v___y_404_){
_start:
{
lean_object* v_res_405_; 
v_res_405_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1(v_00_u03b1_395_, v_argIdx_396_, v_x_397_, v___y_398_, v___y_399_, v___y_400_, v___y_401_, v___y_402_, v___y_403_);
lean_dec(v___y_403_);
lean_dec_ref(v___y_402_);
lean_dec(v___y_401_);
lean_dec_ref(v___y_400_);
lean_dec(v___y_399_);
lean_dec_ref(v___y_398_);
lean_dec(v_argIdx_396_);
return v_res_405_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___lam__0(lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_, lean_object* v___y_426_){
_start:
{
lean_object* v___x_428_; lean_object* v_a_429_; lean_object* v_dummy_430_; lean_object* v_nargs_431_; lean_object* v___x_432_; lean_object* v___x_433_; lean_object* v___x_434_; lean_object* v___x_435_; lean_object* v___x_436_; lean_object* v___x_437_; uint8_t v___x_438_; 
v___x_428_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00SetLike_delabSubtypeSetLike_spec__0___redArg(v___y_421_);
v_a_429_ = lean_ctor_get(v___x_428_, 0);
lean_inc(v_a_429_);
lean_dec_ref(v___x_428_);
v_dummy_430_ = lean_obj_once(&lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___redArg___closed__0, &lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___redArg___closed__0_once, _init_lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___redArg___closed__0);
v_nargs_431_ = l_Lean_Expr_getAppNumArgs(v_a_429_);
lean_inc(v_nargs_431_);
v___x_432_ = lean_mk_array(v_nargs_431_, v_dummy_430_);
v___x_433_ = lean_unsigned_to_nat(1u);
v___x_434_ = lean_nat_sub(v_nargs_431_, v___x_433_);
lean_dec(v_nargs_431_);
v___x_435_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_429_, v___x_432_, v___x_434_);
v___x_436_ = lean_array_get_size(v___x_435_);
v___x_437_ = lean_unsigned_to_nat(2u);
v___x_438_ = lean_nat_dec_eq(v___x_436_, v___x_437_);
if (v___x_438_ == 0)
{
lean_object* v___x_439_; 
lean_dec_ref(v___x_435_);
v___x_439_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_439_;
}
else
{
lean_object* v___x_440_; 
v___x_440_ = lean_array_fget(v___x_435_, v___x_433_);
lean_dec_ref(v___x_435_);
if (lean_obj_tag(v___x_440_) == 6)
{
lean_object* v_binderName_441_; lean_object* v_body_442_; lean_object* v___y_444_; lean_object* v___x_493_; uint8_t v___x_494_; 
v_binderName_441_ = lean_ctor_get(v___x_440_, 0);
lean_inc(v_binderName_441_);
v_body_442_ = lean_ctor_get(v___x_440_, 2);
lean_inc_ref(v_body_442_);
lean_dec_ref_known(v___x_440_, 3);
v___x_493_ = ((lean_object*)(lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__9));
v___x_494_ = l_Lean_Expr_isAppOf(v_body_442_, v___x_493_);
if (v___x_494_ == 0)
{
lean_object* v___x_495_; 
v___x_495_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_495_) == 0)
{
lean_dec_ref_known(v___x_495_, 1);
goto v___jp_464_;
}
else
{
lean_object* v_a_496_; lean_object* v___x_498_; uint8_t v_isShared_499_; uint8_t v_isSharedCheck_503_; 
lean_dec_ref(v_body_442_);
lean_dec(v_binderName_441_);
v_a_496_ = lean_ctor_get(v___x_495_, 0);
v_isSharedCheck_503_ = !lean_is_exclusive(v___x_495_);
if (v_isSharedCheck_503_ == 0)
{
v___x_498_ = v___x_495_;
v_isShared_499_ = v_isSharedCheck_503_;
goto v_resetjp_497_;
}
else
{
lean_inc(v_a_496_);
lean_dec(v___x_495_);
v___x_498_ = lean_box(0);
v_isShared_499_ = v_isSharedCheck_503_;
goto v_resetjp_497_;
}
v_resetjp_497_:
{
lean_object* v___x_501_; 
if (v_isShared_499_ == 0)
{
v___x_501_ = v___x_498_;
goto v_reusejp_500_;
}
else
{
lean_object* v_reuseFailAlloc_502_; 
v_reuseFailAlloc_502_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_502_, 0, v_a_496_);
v___x_501_ = v_reuseFailAlloc_502_;
goto v_reusejp_500_;
}
v_reusejp_500_:
{
return v___x_501_;
}
}
}
}
else
{
goto v___jp_464_;
}
v___jp_443_:
{
lean_object* v___x_445_; lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; 
v___x_445_ = ((lean_object*)(lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__0));
v___x_446_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1___boxed), 10, 3);
lean_closure_set(v___x_446_, 0, lean_box(0));
lean_closure_set(v___x_446_, 1, v___y_444_);
lean_closure_set(v___x_446_, 2, v___x_445_);
v___x_447_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2___boxed), 10, 3);
lean_closure_set(v___x_447_, 0, lean_box(0));
lean_closure_set(v___x_447_, 1, v_binderName_441_);
lean_closure_set(v___x_447_, 2, v___x_446_);
v___x_448_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3___redArg(v___x_447_, v___y_421_, v___y_422_, v___y_423_, v___y_424_, v___y_425_, v___y_426_);
if (lean_obj_tag(v___x_448_) == 0)
{
lean_object* v_a_449_; lean_object* v___x_451_; uint8_t v_isShared_452_; uint8_t v_isSharedCheck_463_; 
v_a_449_ = lean_ctor_get(v___x_448_, 0);
v_isSharedCheck_463_ = !lean_is_exclusive(v___x_448_);
if (v_isSharedCheck_463_ == 0)
{
v___x_451_ = v___x_448_;
v_isShared_452_ = v_isSharedCheck_463_;
goto v_resetjp_450_;
}
else
{
lean_inc(v_a_449_);
lean_dec(v___x_448_);
v___x_451_ = lean_box(0);
v_isShared_452_ = v_isSharedCheck_463_;
goto v_resetjp_450_;
}
v_resetjp_450_:
{
lean_object* v_ref_453_; uint8_t v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; lean_object* v___x_459_; lean_object* v___x_461_; 
v_ref_453_ = lean_ctor_get(v___y_425_, 5);
v___x_454_ = 0;
v___x_455_ = l_Lean_SourceInfo_fromRef(v_ref_453_, v___x_454_);
v___x_456_ = ((lean_object*)(lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__2));
v___x_457_ = ((lean_object*)(lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__3));
lean_inc(v___x_455_);
v___x_458_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_458_, 0, v___x_455_);
lean_ctor_set(v___x_458_, 1, v___x_457_);
v___x_459_ = l_Lean_Syntax_node2(v___x_455_, v___x_456_, v___x_458_, v_a_449_);
if (v_isShared_452_ == 0)
{
lean_ctor_set(v___x_451_, 0, v___x_459_);
v___x_461_ = v___x_451_;
goto v_reusejp_460_;
}
else
{
lean_object* v_reuseFailAlloc_462_; 
v_reuseFailAlloc_462_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_462_, 0, v___x_459_);
v___x_461_ = v_reuseFailAlloc_462_;
goto v_reusejp_460_;
}
v_reusejp_460_:
{
return v___x_461_;
}
}
}
else
{
return v___x_448_;
}
}
v___jp_464_:
{
lean_object* v_nargs_465_; lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; uint8_t v___x_471_; 
v_nargs_465_ = l_Lean_Expr_getAppNumArgs(v_body_442_);
lean_inc(v_nargs_465_);
v___x_466_ = lean_mk_array(v_nargs_465_, v_dummy_430_);
v___x_467_ = lean_nat_sub(v_nargs_465_, v___x_433_);
lean_dec(v_nargs_465_);
v___x_468_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_body_442_, v___x_466_, v___x_467_);
v___x_469_ = lean_array_get_size(v___x_468_);
v___x_470_ = lean_unsigned_to_nat(5u);
v___x_471_ = lean_nat_dec_eq(v___x_469_, v___x_470_);
if (v___x_471_ == 0)
{
lean_object* v___x_472_; 
lean_dec_ref(v___x_468_);
lean_dec(v_binderName_441_);
v___x_472_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_472_;
}
else
{
lean_object* v___x_473_; lean_object* v___x_474_; 
v___x_473_ = lean_unsigned_to_nat(4u);
v___x_474_ = lean_array_fget(v___x_468_, v___x_473_);
if (lean_obj_tag(v___x_474_) == 0)
{
lean_object* v_deBruijnIndex_475_; lean_object* v___x_476_; uint8_t v___x_477_; 
v_deBruijnIndex_475_ = lean_ctor_get(v___x_474_, 0);
lean_inc(v_deBruijnIndex_475_);
lean_dec_ref_known(v___x_474_, 1);
v___x_476_ = lean_unsigned_to_nat(0u);
v___x_477_ = lean_nat_dec_eq(v_deBruijnIndex_475_, v___x_476_);
lean_dec(v_deBruijnIndex_475_);
if (v___x_477_ == 0)
{
lean_object* v___x_478_; 
lean_dec_ref(v___x_468_);
lean_dec(v_binderName_441_);
v___x_478_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_478_;
}
else
{
lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; uint8_t v___x_482_; 
v___x_479_ = lean_array_fget(v___x_468_, v___x_437_);
lean_dec_ref(v___x_468_);
v___x_480_ = ((lean_object*)(lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___closed__6));
v___x_481_ = lean_unsigned_to_nat(3u);
v___x_482_ = l_Lean_Expr_isAppOfArity(v___x_479_, v___x_480_, v___x_481_);
lean_dec(v___x_479_);
if (v___x_482_ == 0)
{
lean_object* v___x_483_; 
v___x_483_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
if (lean_obj_tag(v___x_483_) == 0)
{
lean_dec_ref_known(v___x_483_, 1);
v___y_444_ = v___x_481_;
goto v___jp_443_;
}
else
{
lean_object* v_a_484_; lean_object* v___x_486_; uint8_t v_isShared_487_; uint8_t v_isSharedCheck_491_; 
lean_dec(v_binderName_441_);
v_a_484_ = lean_ctor_get(v___x_483_, 0);
v_isSharedCheck_491_ = !lean_is_exclusive(v___x_483_);
if (v_isSharedCheck_491_ == 0)
{
v___x_486_ = v___x_483_;
v_isShared_487_ = v_isSharedCheck_491_;
goto v_resetjp_485_;
}
else
{
lean_inc(v_a_484_);
lean_dec(v___x_483_);
v___x_486_ = lean_box(0);
v_isShared_487_ = v_isSharedCheck_491_;
goto v_resetjp_485_;
}
v_resetjp_485_:
{
lean_object* v___x_489_; 
if (v_isShared_487_ == 0)
{
v___x_489_ = v___x_486_;
goto v_reusejp_488_;
}
else
{
lean_object* v_reuseFailAlloc_490_; 
v_reuseFailAlloc_490_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_490_, 0, v_a_484_);
v___x_489_ = v_reuseFailAlloc_490_;
goto v_reusejp_488_;
}
v_reusejp_488_:
{
return v___x_489_;
}
}
}
}
else
{
v___y_444_ = v___x_481_;
goto v___jp_443_;
}
}
}
else
{
lean_object* v___x_492_; 
lean_dec(v___x_474_);
lean_dec_ref(v___x_468_);
lean_dec(v_binderName_441_);
v___x_492_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_492_;
}
}
}
}
else
{
lean_object* v___x_504_; 
lean_dec(v___x_440_);
v___x_504_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_504_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___lam__0___boxed(lean_object* v___y_505_, lean_object* v___y_506_, lean_object* v___y_507_, lean_object* v___y_508_, lean_object* v___y_509_, lean_object* v___y_510_, lean_object* v___y_511_){
_start:
{
lean_object* v_res_512_; 
v_res_512_ = lp_mathlib_SetLike_delabSubtypeSetLike___lam__0(v___y_505_, v___y_506_, v___y_507_, v___y_508_, v___y_509_, v___y_510_);
lean_dec(v___y_510_);
lean_dec_ref(v___y_509_);
lean_dec(v___y_508_);
lean_dec_ref(v___y_507_);
lean_dec(v___y_506_);
lean_dec_ref(v___y_505_);
return v_res_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_delabSubtypeSetLike(lean_object* v_a_515_, lean_object* v_a_516_, lean_object* v_a_517_, lean_object* v_a_518_, lean_object* v_a_519_, lean_object* v_a_520_){
_start:
{
lean_object* v___f_522_; lean_object* v___x_523_; lean_object* v___x_524_; 
v___f_522_ = ((lean_object*)(lp_mathlib_SetLike_delabSubtypeSetLike___closed__0));
v___x_523_ = ((lean_object*)(lp_mathlib_SetLike_delabSubtypeSetLike___closed__1));
v___x_524_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_523_, v___f_522_, v_a_515_, v_a_516_, v_a_517_, v_a_518_, v_a_519_, v_a_520_);
return v___x_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_delabSubtypeSetLike___boxed(lean_object* v_a_525_, lean_object* v_a_526_, lean_object* v_a_527_, lean_object* v_a_528_, lean_object* v_a_529_, lean_object* v_a_530_, lean_object* v_a_531_){
_start:
{
lean_object* v_res_532_; 
v_res_532_ = lp_mathlib_SetLike_delabSubtypeSetLike(v_a_525_, v_a_526_, v_a_527_, v_a_528_, v_a_529_, v_a_530_);
lean_dec(v_a_530_);
lean_dec_ref(v_a_529_);
lean_dec(v_a_528_);
lean_dec_ref(v_a_527_);
lean_dec(v_a_526_);
lean_dec_ref(v_a_525_);
return v_res_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1_spec__1(lean_object* v___y_533_, lean_object* v___y_534_, lean_object* v___y_535_, lean_object* v___y_536_, lean_object* v___y_537_, lean_object* v___y_538_){
_start:
{
lean_object* v___x_540_; 
v___x_540_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1_spec__1___redArg(v___y_533_);
return v___x_540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1_spec__1___boxed(lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_, lean_object* v___y_545_, lean_object* v___y_546_, lean_object* v___y_547_){
_start:
{
lean_object* v_res_548_; 
v_res_548_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00SetLike_delabSubtypeSetLike_spec__1_spec__1(v___y_541_, v___y_542_, v___y_543_, v___y_544_, v___y_545_, v___y_546_);
lean_dec(v___y_546_);
lean_dec_ref(v___y_545_);
lean_dec(v___y_544_);
lean_dec_ref(v___y_543_);
lean_dec(v___y_542_);
lean_dec_ref(v___y_541_);
return v_res_548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5(lean_object* v_00_u03b1_549_, lean_object* v_child_550_, lean_object* v_childIdx_551_, lean_object* v_x_552_, lean_object* v___y_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_, lean_object* v___y_558_){
_start:
{
lean_object* v___x_560_; 
v___x_560_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___redArg(v_child_550_, v_childIdx_551_, v_x_552_, v___y_553_, v___y_554_, v___y_555_, v___y_556_, v___y_557_, v___y_558_);
return v___x_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___boxed(lean_object* v_00_u03b1_561_, lean_object* v_child_562_, lean_object* v_childIdx_563_, lean_object* v_x_564_, lean_object* v___y_565_, lean_object* v___y_566_, lean_object* v___y_567_, lean_object* v___y_568_, lean_object* v___y_569_, lean_object* v___y_570_, lean_object* v___y_571_){
_start:
{
lean_object* v_res_572_; 
v_res_572_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5(v_00_u03b1_561_, v_child_562_, v_childIdx_563_, v_x_564_, v___y_565_, v___y_566_, v___y_567_, v___y_568_, v___y_569_, v___y_570_);
lean_dec(v___y_570_);
lean_dec_ref(v___y_569_);
lean_dec(v___y_568_);
lean_dec_ref(v___y_567_);
lean_dec(v___y_566_);
lean_dec_ref(v___y_565_);
return v_res_572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3(lean_object* v_00_u03b1_573_, lean_object* v_x_574_, lean_object* v___y_575_, lean_object* v___y_576_, lean_object* v___y_577_, lean_object* v___y_578_, lean_object* v___y_579_, lean_object* v___y_580_){
_start:
{
lean_object* v___x_582_; 
v___x_582_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3___redArg(v_x_574_, v___y_575_, v___y_576_, v___y_577_, v___y_578_, v___y_579_, v___y_580_);
return v___x_582_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3___boxed(lean_object* v_00_u03b1_583_, lean_object* v_x_584_, lean_object* v___y_585_, lean_object* v___y_586_, lean_object* v___y_587_, lean_object* v___y_588_, lean_object* v___y_589_, lean_object* v___y_590_, lean_object* v___y_591_){
_start:
{
lean_object* v_res_592_; 
v_res_592_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3(v_00_u03b1_583_, v_x_584_, v___y_585_, v___y_586_, v___y_587_, v___y_588_, v___y_589_, v___y_590_);
lean_dec(v___y_590_);
lean_dec_ref(v___y_589_);
lean_dec(v___y_588_);
lean_dec_ref(v___y_587_);
lean_dec(v___y_586_);
lean_dec_ref(v___y_585_);
return v_res_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4(lean_object* v_00_u03b1_593_, lean_object* v_name_594_, uint8_t v_bi_595_, lean_object* v_type_596_, lean_object* v_k_597_, uint8_t v_kind_598_, lean_object* v___y_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_, lean_object* v___y_603_, lean_object* v___y_604_){
_start:
{
lean_object* v___x_606_; 
v___x_606_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4___redArg(v_name_594_, v_bi_595_, v_type_596_, v_k_597_, v_kind_598_, v___y_599_, v___y_600_, v___y_601_, v___y_602_, v___y_603_, v___y_604_);
return v___x_606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4___boxed(lean_object* v_00_u03b1_607_, lean_object* v_name_608_, lean_object* v_bi_609_, lean_object* v_type_610_, lean_object* v_k_611_, lean_object* v_kind_612_, lean_object* v___y_613_, lean_object* v___y_614_, lean_object* v___y_615_, lean_object* v___y_616_, lean_object* v___y_617_, lean_object* v___y_618_, lean_object* v___y_619_){
_start:
{
uint8_t v_bi_boxed_620_; uint8_t v_kind_boxed_621_; lean_object* v_res_622_; 
v_bi_boxed_620_ = lean_unbox(v_bi_609_);
v_kind_boxed_621_ = lean_unbox(v_kind_612_);
v_res_622_ = lp_mathlib_Lean_Meta_withLocalDecl___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__4(v_00_u03b1_607_, v_name_608_, v_bi_boxed_620_, v_type_610_, v_k_611_, v_kind_boxed_621_, v___y_613_, v___y_614_, v___y_615_, v___y_616_, v___y_617_, v___y_618_);
lean_dec(v___y_618_);
lean_dec_ref(v___y_617_);
lean_dec(v___y_616_);
lean_dec_ref(v___y_615_);
lean_dec(v___y_614_);
lean_dec_ref(v___y_613_);
return v_res_622_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__8(lean_object* v_00_u03b1_623_, lean_object* v___x_624_, lean_object* v_child_625_, lean_object* v_childIdx_626_, lean_object* v___y_627_, lean_object* v___y_628_, lean_object* v___y_629_, lean_object* v___y_630_, lean_object* v___y_631_, lean_object* v___y_632_){
_start:
{
lean_object* v___x_634_; 
v___x_634_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__8___redArg(v___x_624_, v_child_625_, v_childIdx_626_, v___y_627_, v___y_628_, v___y_629_, v___y_630_, v___y_631_, v___y_632_);
return v___x_634_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__8___boxed(lean_object* v_00_u03b1_635_, lean_object* v___x_636_, lean_object* v_child_637_, lean_object* v_childIdx_638_, lean_object* v___y_639_, lean_object* v___y_640_, lean_object* v___y_641_, lean_object* v___y_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_){
_start:
{
lean_object* v_res_646_; 
v_res_646_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00SetLike_delabSubtypeSetLike_spec__3_spec__5___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3_spec__8(v_00_u03b1_635_, v___x_636_, v_child_637_, v_childIdx_638_, v___y_639_, v___y_640_, v___y_641_, v___y_642_, v___y_643_, v___y_644_);
lean_dec(v___y_644_);
lean_dec_ref(v___y_643_);
lean_dec(v___y_642_);
lean_dec_ref(v___y_641_);
lean_dec(v___y_640_);
lean_dec_ref(v___y_639_);
return v_res_646_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3(lean_object* v_00_u03b1_647_, lean_object* v_00_u03b2_648_, lean_object* v_n_649_, lean_object* v_v_650_, lean_object* v_x_651_, lean_object* v___y_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_){
_start:
{
lean_object* v___x_659_; 
v___x_659_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3___redArg(v_n_649_, v_v_650_, v_x_651_, v___y_652_, v___y_653_, v___y_654_, v___y_655_, v___y_656_, v___y_657_);
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3___boxed(lean_object* v_00_u03b1_660_, lean_object* v_00_u03b2_661_, lean_object* v_n_662_, lean_object* v_v_663_, lean_object* v_x_664_, lean_object* v___y_665_, lean_object* v___y_666_, lean_object* v___y_667_, lean_object* v___y_668_, lean_object* v___y_669_, lean_object* v___y_670_, lean_object* v___y_671_){
_start:
{
lean_object* v_res_672_; 
v_res_672_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody_x27___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withBindingBody___at___00SetLike_delabSubtypeSetLike_spec__2_spec__3(v_00_u03b1_660_, v_00_u03b2_661_, v_n_662_, v_v_663_, v_x_664_, v___y_665_, v___y_666_, v___y_667_, v___y_668_, v___y_669_, v___y_670_);
lean_dec(v___y_670_);
lean_dec_ref(v___y_669_);
lean_dec(v___y_668_);
lean_dec_ref(v___y_667_);
lean_dec(v___y_666_);
lean_dec_ref(v___y_665_);
return v_res_672_;
}
}
static lean_object* _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__12(void){
_start:
{
lean_object* v___x_699_; lean_object* v___x_700_; 
v___x_699_ = ((lean_object*)(lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__10));
v___x_700_ = l_Lean_mkAtom(v___x_699_);
return v___x_700_;
}
}
static lean_object* _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__13(void){
_start:
{
lean_object* v___x_701_; lean_object* v___x_702_; lean_object* v___x_703_; 
v___x_701_ = lean_obj_once(&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__12, &lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__12_once, _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__12);
v___x_702_ = ((lean_object*)(lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__5));
v___x_703_ = lean_array_push(v___x_702_, v___x_701_);
return v___x_703_;
}
}
static lean_object* _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__17(void){
_start:
{
lean_object* v___x_714_; lean_object* v___x_715_; lean_object* v___x_716_; 
v___x_714_ = ((lean_object*)(lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__16));
v___x_715_ = ((lean_object*)(lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__5));
v___x_716_ = lean_array_push(v___x_715_, v___x_714_);
return v___x_716_;
}
}
static lean_object* _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__18(void){
_start:
{
lean_object* v___x_717_; lean_object* v___x_718_; lean_object* v___x_719_; lean_object* v___x_720_; 
v___x_717_ = lean_obj_once(&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__17, &lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__17_once, _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__17);
v___x_718_ = ((lean_object*)(lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__15));
v___x_719_ = lean_box(2);
v___x_720_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_720_, 0, v___x_719_);
lean_ctor_set(v___x_720_, 1, v___x_718_);
lean_ctor_set(v___x_720_, 2, v___x_717_);
return v___x_720_;
}
}
static lean_object* _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__19(void){
_start:
{
lean_object* v___x_721_; lean_object* v___x_722_; lean_object* v___x_723_; 
v___x_721_ = lean_obj_once(&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__18, &lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__18_once, _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__18);
v___x_722_ = lean_obj_once(&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__13, &lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__13_once, _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__13);
v___x_723_ = lean_array_push(v___x_722_, v___x_721_);
return v___x_723_;
}
}
static lean_object* _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__20(void){
_start:
{
lean_object* v___x_724_; lean_object* v___x_725_; lean_object* v___x_726_; 
v___x_724_ = ((lean_object*)(lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__16));
v___x_725_ = lean_obj_once(&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__19, &lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__19_once, _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__19);
v___x_726_ = lean_array_push(v___x_725_, v___x_724_);
return v___x_726_;
}
}
static lean_object* _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__21(void){
_start:
{
lean_object* v___x_727_; lean_object* v___x_728_; lean_object* v___x_729_; 
v___x_727_ = ((lean_object*)(lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__16));
v___x_728_ = lean_obj_once(&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__20, &lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__20_once, _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__20);
v___x_729_ = lean_array_push(v___x_728_, v___x_727_);
return v___x_729_;
}
}
static lean_object* _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__22(void){
_start:
{
lean_object* v___x_730_; lean_object* v___x_731_; lean_object* v___x_732_; 
v___x_730_ = ((lean_object*)(lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__16));
v___x_731_ = lean_obj_once(&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__21, &lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__21_once, _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__21);
v___x_732_ = lean_array_push(v___x_731_, v___x_730_);
return v___x_732_;
}
}
static lean_object* _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__23(void){
_start:
{
lean_object* v___x_733_; lean_object* v___x_734_; lean_object* v___x_735_; 
v___x_733_ = ((lean_object*)(lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__16));
v___x_734_ = lean_obj_once(&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__22, &lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__22_once, _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__22);
v___x_735_ = lean_array_push(v___x_734_, v___x_733_);
return v___x_735_;
}
}
static lean_object* _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__24(void){
_start:
{
lean_object* v___x_736_; lean_object* v___x_737_; lean_object* v___x_738_; lean_object* v___x_739_; 
v___x_736_ = lean_obj_once(&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__23, &lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__23_once, _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__23);
v___x_737_ = ((lean_object*)(lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__11));
v___x_738_ = lean_box(2);
v___x_739_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_739_, 0, v___x_738_);
lean_ctor_set(v___x_739_, 1, v___x_737_);
lean_ctor_set(v___x_739_, 2, v___x_736_);
return v___x_739_;
}
}
static lean_object* _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__25(void){
_start:
{
lean_object* v___x_740_; lean_object* v___x_741_; lean_object* v___x_742_; 
v___x_740_ = lean_obj_once(&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__24, &lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__24_once, _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__24);
v___x_741_ = ((lean_object*)(lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__5));
v___x_742_ = lean_array_push(v___x_741_, v___x_740_);
return v___x_742_;
}
}
static lean_object* _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__26(void){
_start:
{
lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; 
v___x_743_ = lean_obj_once(&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__25, &lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__25_once, _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__25);
v___x_744_ = ((lean_object*)(lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__9));
v___x_745_ = lean_box(2);
v___x_746_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_746_, 0, v___x_745_);
lean_ctor_set(v___x_746_, 1, v___x_744_);
lean_ctor_set(v___x_746_, 2, v___x_743_);
return v___x_746_;
}
}
static lean_object* _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__27(void){
_start:
{
lean_object* v___x_747_; lean_object* v___x_748_; lean_object* v___x_749_; 
v___x_747_ = lean_obj_once(&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__26, &lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__26_once, _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__26);
v___x_748_ = ((lean_object*)(lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__5));
v___x_749_ = lean_array_push(v___x_748_, v___x_747_);
return v___x_749_;
}
}
static lean_object* _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__28(void){
_start:
{
lean_object* v___x_750_; lean_object* v___x_751_; lean_object* v___x_752_; lean_object* v___x_753_; 
v___x_750_ = lean_obj_once(&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__27, &lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__27_once, _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__27);
v___x_751_ = ((lean_object*)(lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__7));
v___x_752_ = lean_box(2);
v___x_753_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_753_, 0, v___x_752_);
lean_ctor_set(v___x_753_, 1, v___x_751_);
lean_ctor_set(v___x_753_, 2, v___x_750_);
return v___x_753_;
}
}
static lean_object* _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__29(void){
_start:
{
lean_object* v___x_754_; lean_object* v___x_755_; lean_object* v___x_756_; 
v___x_754_ = lean_obj_once(&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__28, &lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__28_once, _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__28);
v___x_755_ = ((lean_object*)(lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__5));
v___x_756_ = lean_array_push(v___x_755_, v___x_754_);
return v___x_756_;
}
}
static lean_object* _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__30(void){
_start:
{
lean_object* v___x_757_; lean_object* v___x_758_; lean_object* v___x_759_; lean_object* v___x_760_; 
v___x_757_ = lean_obj_once(&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__29, &lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__29_once, _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__29);
v___x_758_ = ((lean_object*)(lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__4));
v___x_759_ = lean_box(2);
v___x_760_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_760_, 0, v___x_759_);
lean_ctor_set(v___x_760_, 1, v___x_758_);
lean_ctor_set(v___x_760_, 2, v___x_757_);
return v___x_760_;
}
}
static lean_object* _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1(void){
_start:
{
lean_object* v___x_761_; 
v___x_761_ = lean_obj_once(&lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__30, &lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__30_once, _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1___closed__30);
return v___x_761_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LE_ofSetLike(lean_object* v_A_762_, lean_object* v_B_763_, lean_object* v_inst_764_){
_start:
{
lean_object* v___x_765_; 
v___x_765_ = lean_box(0);
return v___x_765_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PartialOrder_ofSetLike(lean_object* v_A_769_, lean_object* v_B_770_, lean_object* v_inst_771_){
_start:
{
lean_object* v___x_772_; 
v___x_772_ = ((lean_object*)(lp_mathlib_PartialOrder_ofSetLike___closed__0));
return v___x_772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_instSubtypeSet(lean_object* v_X_773_, lean_object* v_p_774_){
_start:
{
lean_object* v___x_775_; 
v___x_775_ = lean_box(0);
return v___x_775_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SetLike_instSubtype(lean_object* v_X_776_, lean_object* v_S_777_, lean_object* v_inst_778_, lean_object* v_p_779_){
_start:
{
lean_object* v___x_780_; 
v___x_780_ = lean_box(0);
return v___x_780_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_SetLike(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_SetLike_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_SetLike(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_SetLike_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1 = _init_lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1();
lean_mark_persistent(lp_mathlib_SetLike_exists__not__mem__of__ne__top___auto__1);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_SetLike(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_SetLike_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Monotonicity_Attr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_SetLike(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_SetLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_SetLike_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_SetLike_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
