// Lean compiler output
// Module: Mathlib.Lean.Meta.Simp
// Imports: public import Init public meta import Init public import Mathlib.Init public import Lean.Elab.Tactic.Simp public import Lean.Meta.DiscrTree
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
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__2___boxed(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_mkConst(lean_object*, lean_object*);
uint8_t l_Lean_Meta_SimpTheorems_isLemma(lean_object*, lean_object*);
uint8_t l_Lean_Meta_SimpTheorems_isDeclToUnfold(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
uint8_t lean_usize_dec_lt(size_t, size_t);
size_t lean_array_size(lean_object*);
uint8_t l_Lean_LocalDecl_isAuxDecl(lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
lean_object* l_Lean_Meta_isProp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Meta_getSimpTheorems___redArg(lean_object*);
lean_object* l_Lean_Meta_SimpTheorems_addConst(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_empty(lean_object*);
lean_object* l_Lean_Meta_getSimpCongrTheorems___redArg(lean_object*);
extern lean_object* l_Lean_Options_empty;
lean_object* l_Lean_Meta_Simp_mkContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_simp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getSimpExtension_x3f(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SimpExtension_getTheorems___redArg(lean_object*, lean_object*);
lean_object* l_List_reverse___redArg(lean_object*);
lean_object* lean_array_to_list(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__6(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_SimpTheoremsArray_eraseTheorem(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_Simprocs_erase(lean_object*, lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
uint8_t lean_string_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_mkApp3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkForallFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLambdaFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEqTrans(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEqSymm(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_Meta_simpExtensionMapRef;
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkOfEqTrue(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_getSimprocs___redArg(lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Meta_Simp_mkMethods(lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Meta_Simp_SimpM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_Trie_foldValuesM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_toList___redArg(lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Elab_Tactic_elabSimpArgs(lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkExpectedTypeHint(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkEqMP(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_Origin_key___boxed(lean_object*);
lean_object* l_Lean_instToFormatName__lean___lam__0(lean_object*);
lean_object* l_instToFormatArray___redArg___lam__0(lean_object*, lean_object*);
lean_object* l_Lean_Meta_instToFormatSimpTheorem___lam__0(lean_object*);
lean_object* l_instToFormatProd___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_foldlMAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_format___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList___redArg___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_Lean_PHashSet_toList___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_PHashSet_toList___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_PHashSet_toList___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_PHashSet_toList___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "pre:\n"};
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__0_value)}};
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__1_value;
static const lean_array_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__2_value;
static const lean_closure_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__0, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__3 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__3_value;
static const lean_closure_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__1___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__4 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__4_value;
static const lean_closure_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__2___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__5 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__5_value;
static const lean_closure_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__3, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__6 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__6_value;
static const lean_closure_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__4___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__7 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__7_value;
static const lean_closure_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__5___boxed, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__8 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__8_value;
static const lean_closure_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Id_instMonad___lam__6, .m_arity = 4, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__9 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__3_value),((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__4_value)}};
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__10 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__10_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*5 + 0, .m_other = 5, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__10_value),((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__5_value),((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__6_value),((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__7_value),((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__8_value)}};
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__11 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__11_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__11_value),((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__9_value)}};
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__12 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__12_value;
static const lean_string_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "\npost:\n"};
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__13 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__13_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__13_value)}};
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__14 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__14_value;
static const lean_string_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "\nlemmaNames:\n"};
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__15 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__15_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__15_value)}};
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__16 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__16_value;
static const lean_string_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "\ntoUnfold: "};
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__17 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__17_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__17_value)}};
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__18 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__18_value;
static const lean_string_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "\nerased: "};
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__19 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__19_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__19_value)}};
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__20 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__20_value;
static const lean_string_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "\ntoUnfoldThms: "};
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__21 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__21_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__21_value)}};
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__22 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__22_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__0_value;
static const lean_closure_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_Origin_key___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__1_value;
static const lean_closure_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_Meta_instToFormatSimpTheorem___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__2_value;
static const lean_closure_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_instToFormatName__lean___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__3 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__3_value;
static const lean_closure_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instToFormatArray___redArg___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__3_value)} };
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__4 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__4_value;
static const lean_closure_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_instToFormatProd___redArg___lam__0, .m_arity = 3, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__3_value),((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__4_value)} };
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__5 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__5_value;
static const lean_closure_object lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*6, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3, .m_arity = 7, .m_num_fixed = 6, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__0_value),((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__0_value),((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__2_value),((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__1_value),((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__3_value),((lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__5_value)} };
static const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__6 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__6_value;
LEAN_EXPORT const lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___closed__6_value;
static const lean_string_object lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "True"};
static const lean_object* lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__0_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__0_value),LEAN_SCALAR_PTR_LITERAL(78, 21, 103, 131, 118, 13, 187, 164)}};
static const lean_object* lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__1 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__1_value;
static const lean_string_object lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "intro"};
static const lean_object* lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__0_value),LEAN_SCALAR_PTR_LITERAL(78, 21, 103, 131, 118, 13, 187, 164)}};
static const lean_ctor_object lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__2_value),LEAN_SCALAR_PTR_LITERAL(177, 152, 123, 219, 220, 182, 189, 250)}};
static const lean_object* lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__3 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__3_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_Result_ofTrue(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_Result_ofTrue___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__1_spec__4(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__2_spec__3(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_Meta_Simp_getPropHyps___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Meta_Simp_getPropHyps___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_getPropHyps___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_getPropHyps(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_getPropHyps___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__1___closed__0;
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__1___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__1___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__1(lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__2___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__2___closed__0;
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__2___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_Meta_simpTheoremsOfNames_spec__3(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_Meta_simpTheoremsOfNames_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_Meta_simpTheoremsOfNames_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_Meta_simpTheoremsOfNames_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__0;
static lean_once_cell_t lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__2;
static lean_once_cell_t lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__3;
static lean_once_cell_t lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__4;
static lean_once_cell_t lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__5;
static const lean_string_object lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "eq_self"};
static const lean_object* lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__6 = (const lean_object*)&lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__6_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__6_value),LEAN_SCALAR_PTR_LITERAL(224, 148, 98, 216, 254, 239, 13, 169)}};
static const lean_object* lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__7 = (const lean_object*)&lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__7_value;
static const lean_string_object lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "iff_self"};
static const lean_object* lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__8 = (const lean_object*)&lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__8_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__8_value),LEAN_SCALAR_PTR_LITERAL(79, 255, 41, 65, 134, 196, 244, 123)}};
static const lean_object* lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__9 = (const lean_object*)&lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__9_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__10 = (const lean_object*)&lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__10_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__7_value),((lean_object*)&lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__10_value)}};
static const lean_object* lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__11 = (const lean_object*)&lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpTheoremsOfNames(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpTheoremsOfNames___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_Context_ofNames(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_Context_ofNames___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_Meta_Simp_Context_ofArgs___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Meta_Simp_Context_ofArgs___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_Simp_Context_ofArgs___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_Context_ofArgs(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_Context_ofArgs___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_Meta_simpOnlyNames___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_Meta_simpOnlyNames___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_simpOnlyNames___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_simpOnlyNames___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_simpOnlyNames___closed__1;
static lean_once_cell_t lp_mathlib_Lean_Meta_simpOnlyNames___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_simpOnlyNames___closed__2;
static lean_once_cell_t lp_mathlib_Lean_Meta_simpOnlyNames___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_simpOnlyNames___closed__3;
static lean_once_cell_t lp_mathlib_Lean_Meta_simpOnlyNames___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_simpOnlyNames___closed__4;
static lean_once_cell_t lp_mathlib_Lean_Meta_simpOnlyNames___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_simpOnlyNames___closed__5;
static lean_once_cell_t lp_mathlib_Lean_Meta_simpOnlyNames___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_simpOnlyNames___closed__6;
static lean_once_cell_t lp_mathlib_Lean_Meta_simpOnlyNames___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_simpOnlyNames___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpOnlyNames(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpOnlyNames___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpType___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1___redArg(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_simpEq_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_simpEq_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_simpEq_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_simpEq_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Lean_Meta_simpEq___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "simpEq expecting Eq"};
static const lean_object* lp_mathlib_Lean_Meta_simpEq___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Lean_Meta_simpEq___lam__0___closed__0_value;
static lean_once_cell_t lp_mathlib_Lean_Meta_simpEq___lam__0___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Meta_simpEq___lam__0___closed__1;
static const lean_string_object lp_mathlib_Lean_Meta_simpEq___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Eq"};
static const lean_object* lp_mathlib_Lean_Meta_simpEq___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Lean_Meta_simpEq___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib_Lean_Meta_simpEq___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Lean_Meta_simpEq___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(143, 37, 101, 248, 9, 246, 191, 223)}};
static const lean_object* lp_mathlib_Lean_Meta_simpEq___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Lean_Meta_simpEq___lam__0___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpEq___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpEq___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_simpEq_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_simpEq_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Meta_SimpTheorems_contains(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_SimpTheorems_contains___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_isInSimpSet(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_isInSimpSet___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__10___redArg(lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__10___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__4(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Lean_Meta_getAllSimpDecls_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0___redArg___closed__0 = (const lean_object*)&lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getAllSimpDecls(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getAllSimpDecls___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5_spec__8___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5_spec__8___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__10(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__10___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Lean_Meta_getAllSimpAttrs_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Lean_Meta_getAllSimpAttrs_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_getAllSimpAttrs_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_getAllSimpAttrs_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getAllSimpAttrs(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getAllSimpAttrs___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Lean_Meta_getAllSimpAttrs_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Lean_Meta_getAllSimpAttrs_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_SMap_switch___at___00Lean_Meta_Simp_withoutTheorems_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_SMap_switch___at___00Lean_Meta_Simp_withoutTheorems_spec__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_withoutTheorems___redArg___lam__0(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_withoutTheorems___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_Simp_withoutTheorems_spec__0___redArg(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_Simp_withoutTheorems_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_withoutTheorems___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_withoutTheorems___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_withoutTheorems(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_withoutTheorems___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_Simp_withoutTheorems_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_Simp_withoutTheorems_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList___redArg___lam__0(lean_object* v_x_1_){
_start:
{
lean_object* v_fst_2_; 
v_fst_2_ = lean_ctor_get(v_x_1_, 0);
lean_inc(v_fst_2_);
return v_fst_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList___redArg___lam__0___boxed(lean_object* v_x_3_){
_start:
{
lean_object* v_res_4_; 
v_res_4_ = lp_mathlib_Lean_PHashSet_toList___redArg___lam__0(v_x_3_);
lean_dec_ref(v_x_3_);
return v_res_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList___redArg(lean_object* v_s_6_){
_start:
{
lean_object* v___f_7_; lean_object* v___x_8_; lean_object* v___x_9_; lean_object* v___x_10_; 
v___f_7_ = ((lean_object*)(lp_mathlib_Lean_PHashSet_toList___redArg___closed__0));
v___x_8_ = l_Lean_PersistentHashMap_toList___redArg(v_s_6_);
v___x_9_ = lean_box(0);
v___x_10_ = l_List_mapTR_loop___redArg(v___f_7_, v___x_8_, v___x_9_);
return v___x_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList(lean_object* v_00_u03b1_11_, lean_object* v_inst_12_, lean_object* v_inst_13_, lean_object* v_s_14_){
_start:
{
lean_object* v___x_15_; 
v___x_15_ = lp_mathlib_Lean_PHashSet_toList___redArg(v_s_14_);
return v___x_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList___boxed(lean_object* v_00_u03b1_16_, lean_object* v_inst_17_, lean_object* v_inst_18_, lean_object* v_s_19_){
_start:
{
lean_object* v_res_20_; 
v_res_20_ = lp_mathlib_Lean_PHashSet_toList(v_00_u03b1_16_, v_inst_17_, v_inst_18_, v_s_19_);
lean_dec_ref(v_inst_18_);
lean_dec_ref(v_inst_17_);
return v_res_20_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__0(lean_object* v_x1_21_, lean_object* v_x2_22_){
_start:
{
lean_object* v___x_23_; 
v___x_23_ = lean_array_push(v_x1_21_, v_x2_22_);
return v___x_23_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__2(lean_object* v___x_24_, lean_object* v___f_25_, lean_object* v_s_26_, lean_object* v_x_27_, lean_object* v_t_28_){
_start:
{
lean_object* v___x_29_; 
v___x_29_ = l_Lean_Meta_DiscrTree_Trie_foldValuesM___redArg(v___x_24_, v___f_25_, v_s_26_, v_t_28_);
return v___x_29_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__2___boxed(lean_object* v___x_30_, lean_object* v___f_31_, lean_object* v_s_32_, lean_object* v_x_33_, lean_object* v_t_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__2(v___x_30_, v___f_31_, v_s_32_, v_x_33_, v_t_34_);
lean_dec(v_x_33_);
return v_res_35_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3(lean_object* v___f_75_, lean_object* v___f_76_, lean_object* v___f_77_, lean_object* v___f_78_, lean_object* v___f_79_, lean_object* v___f_80_, lean_object* v_s_81_){
_start:
{
lean_object* v_pre_82_; lean_object* v_post_83_; lean_object* v_lemmaNames_84_; lean_object* v_toUnfold_85_; lean_object* v_erased_86_; lean_object* v_toUnfoldThms_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___f_91_; lean_object* v___f_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; 
v_pre_82_ = lean_ctor_get(v_s_81_, 0);
lean_inc_ref(v_pre_82_);
v_post_83_ = lean_ctor_get(v_s_81_, 1);
lean_inc_ref(v_post_83_);
v_lemmaNames_84_ = lean_ctor_get(v_s_81_, 2);
lean_inc_ref(v_lemmaNames_84_);
v_toUnfold_85_ = lean_ctor_get(v_s_81_, 3);
lean_inc_ref(v_toUnfold_85_);
v_erased_86_ = lean_ctor_get(v_s_81_, 4);
lean_inc_ref(v_erased_86_);
v_toUnfoldThms_87_ = lean_ctor_get(v_s_81_, 5);
lean_inc_ref(v_toUnfoldThms_87_);
lean_dec_ref(v_s_81_);
v___x_88_ = ((lean_object*)(lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__1));
v___x_89_ = ((lean_object*)(lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__2));
v___x_90_ = ((lean_object*)(lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__12));
v___f_91_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__2___boxed), 5, 2);
lean_closure_set(v___f_91_, 0, v___x_90_);
lean_closure_set(v___f_91_, 1, v___f_75_);
v___f_92_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__2___boxed), 5, 2);
lean_closure_set(v___f_92_, 0, v___x_90_);
lean_closure_set(v___f_92_, 1, v___f_76_);
v___x_93_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_90_, v___f_92_, v_pre_82_, v___x_89_);
v___x_94_ = lean_array_to_list(v___x_93_);
lean_inc_ref(v___f_77_);
v___x_95_ = l_List_format___redArg(v___f_77_, v___x_94_);
v___x_96_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_88_);
lean_ctor_set(v___x_96_, 1, v___x_95_);
v___x_97_ = ((lean_object*)(lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__14));
v___x_98_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_98_, 0, v___x_96_);
lean_ctor_set(v___x_98_, 1, v___x_97_);
v___x_99_ = l_Lean_PersistentHashMap_foldlMAux___redArg(v___x_90_, v___f_91_, v_post_83_, v___x_89_);
v___x_100_ = lean_array_to_list(v___x_99_);
v___x_101_ = l_List_format___redArg(v___f_77_, v___x_100_);
v___x_102_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_102_, 0, v___x_98_);
lean_ctor_set(v___x_102_, 1, v___x_101_);
v___x_103_ = ((lean_object*)(lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__16));
v___x_104_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_104_, 0, v___x_102_);
lean_ctor_set(v___x_104_, 1, v___x_103_);
v___x_105_ = lp_mathlib_Lean_PHashSet_toList___redArg(v_lemmaNames_84_);
v___x_106_ = lean_box(0);
lean_inc_ref(v___f_78_);
v___x_107_ = l_List_mapTR_loop___redArg(v___f_78_, v___x_105_, v___x_106_);
lean_inc_ref_n(v___f_79_, 2);
v___x_108_ = l_List_format___redArg(v___f_79_, v___x_107_);
v___x_109_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_109_, 0, v___x_104_);
lean_ctor_set(v___x_109_, 1, v___x_108_);
v___x_110_ = ((lean_object*)(lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__18));
v___x_111_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_111_, 0, v___x_109_);
lean_ctor_set(v___x_111_, 1, v___x_110_);
v___x_112_ = lp_mathlib_Lean_PHashSet_toList___redArg(v_toUnfold_85_);
v___x_113_ = l_List_format___redArg(v___f_79_, v___x_112_);
v___x_114_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_114_, 0, v___x_111_);
lean_ctor_set(v___x_114_, 1, v___x_113_);
v___x_115_ = ((lean_object*)(lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__20));
v___x_116_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_116_, 0, v___x_114_);
lean_ctor_set(v___x_116_, 1, v___x_115_);
v___x_117_ = lp_mathlib_Lean_PHashSet_toList___redArg(v_erased_86_);
v___x_118_ = l_List_mapTR_loop___redArg(v___f_78_, v___x_117_, v___x_106_);
v___x_119_ = l_List_format___redArg(v___f_79_, v___x_118_);
v___x_120_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_120_, 0, v___x_116_);
lean_ctor_set(v___x_120_, 1, v___x_119_);
v___x_121_ = ((lean_object*)(lp_mathlib_Lean_Meta_Simp_instToFormatSimpTheorems__mathlib___lam__3___closed__22));
v___x_122_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_122_, 0, v___x_120_);
lean_ctor_set(v___x_122_, 1, v___x_121_);
v___x_123_ = l_Lean_PersistentHashMap_toList___redArg(v_toUnfoldThms_87_);
v___x_124_ = l_List_format___redArg(v___f_80_, v___x_123_);
v___x_125_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_125_, 0, v___x_122_);
lean_ctor_set(v___x_125_, 1, v___x_124_);
return v___x_125_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__4(void){
_start:
{
lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; 
v___x_149_ = lean_box(0);
v___x_150_ = ((lean_object*)(lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__3));
v___x_151_ = l_Lean_mkConst(v___x_150_, v___x_149_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_Result_ofTrue(lean_object* v_r_152_, lean_object* v_a_153_, lean_object* v_a_154_, lean_object* v_a_155_, lean_object* v_a_156_){
_start:
{
lean_object* v_a_159_; lean_object* v_expr_162_; lean_object* v_proof_x3f_163_; lean_object* v___x_164_; uint8_t v___x_165_; 
v_expr_162_ = lean_ctor_get(v_r_152_, 0);
lean_inc_ref(v_expr_162_);
v_proof_x3f_163_ = lean_ctor_get(v_r_152_, 1);
lean_inc(v_proof_x3f_163_);
lean_dec_ref(v_r_152_);
v___x_164_ = ((lean_object*)(lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__1));
v___x_165_ = l_Lean_Expr_isConstOf(v_expr_162_, v___x_164_);
lean_dec_ref(v_expr_162_);
if (v___x_165_ == 0)
{
lean_object* v___x_166_; lean_object* v___x_167_; 
lean_dec(v_proof_x3f_163_);
v___x_166_ = lean_box(0);
v___x_167_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_167_, 0, v___x_166_);
return v___x_167_;
}
else
{
if (lean_obj_tag(v_proof_x3f_163_) == 0)
{
lean_object* v___x_168_; 
v___x_168_ = lean_obj_once(&lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__4, &lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__4_once, _init_lp_mathlib_Lean_Meta_Simp_Result_ofTrue___closed__4);
v_a_159_ = v___x_168_;
goto v___jp_158_;
}
else
{
lean_object* v_val_169_; lean_object* v___x_170_; 
v_val_169_ = lean_ctor_get(v_proof_x3f_163_, 0);
lean_inc(v_val_169_);
lean_dec_ref_known(v_proof_x3f_163_, 1);
v___x_170_ = l_Lean_Meta_mkOfEqTrue(v_val_169_, v_a_153_, v_a_154_, v_a_155_, v_a_156_);
if (lean_obj_tag(v___x_170_) == 0)
{
lean_object* v_a_171_; 
v_a_171_ = lean_ctor_get(v___x_170_, 0);
lean_inc(v_a_171_);
lean_dec_ref_known(v___x_170_, 1);
v_a_159_ = v_a_171_;
goto v___jp_158_;
}
else
{
lean_object* v_a_172_; lean_object* v___x_174_; uint8_t v_isShared_175_; uint8_t v_isSharedCheck_179_; 
v_a_172_ = lean_ctor_get(v___x_170_, 0);
v_isSharedCheck_179_ = !lean_is_exclusive(v___x_170_);
if (v_isSharedCheck_179_ == 0)
{
v___x_174_ = v___x_170_;
v_isShared_175_ = v_isSharedCheck_179_;
goto v_resetjp_173_;
}
else
{
lean_inc(v_a_172_);
lean_dec(v___x_170_);
v___x_174_ = lean_box(0);
v_isShared_175_ = v_isSharedCheck_179_;
goto v_resetjp_173_;
}
v_resetjp_173_:
{
lean_object* v___x_177_; 
if (v_isShared_175_ == 0)
{
v___x_177_ = v___x_174_;
goto v_reusejp_176_;
}
else
{
lean_object* v_reuseFailAlloc_178_; 
v_reuseFailAlloc_178_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_178_, 0, v_a_172_);
v___x_177_ = v_reuseFailAlloc_178_;
goto v_reusejp_176_;
}
v_reusejp_176_:
{
return v___x_177_;
}
}
}
}
}
v___jp_158_:
{
lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_160_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_160_, 0, v_a_159_);
v___x_161_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_161_, 0, v___x_160_);
return v___x_161_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_Result_ofTrue___boxed(lean_object* v_r_180_, lean_object* v_a_181_, lean_object* v_a_182_, lean_object* v_a_183_, lean_object* v_a_184_, lean_object* v_a_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib_Lean_Meta_Simp_Result_ofTrue(v_r_180_, v_a_181_, v_a_182_, v_a_183_, v_a_184_);
lean_dec(v_a_184_);
lean_dec_ref(v_a_183_);
lean_dec(v_a_182_);
lean_dec_ref(v_a_181_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__1_spec__4(lean_object* v_as_187_, size_t v_sz_188_, size_t v_i_189_, lean_object* v_b_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_){
_start:
{
uint8_t v___x_196_; 
v___x_196_ = lean_usize_dec_lt(v_i_189_, v_sz_188_);
if (v___x_196_ == 0)
{
lean_object* v___x_197_; 
v___x_197_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_197_, 0, v_b_190_);
return v___x_197_;
}
else
{
lean_object* v_snd_198_; lean_object* v___x_200_; uint8_t v_isShared_201_; uint8_t v_isSharedCheck_228_; 
v_snd_198_ = lean_ctor_get(v_b_190_, 1);
v_isSharedCheck_228_ = !lean_is_exclusive(v_b_190_);
if (v_isSharedCheck_228_ == 0)
{
lean_object* v_unused_229_; 
v_unused_229_ = lean_ctor_get(v_b_190_, 0);
lean_dec(v_unused_229_);
v___x_200_ = v_b_190_;
v_isShared_201_ = v_isSharedCheck_228_;
goto v_resetjp_199_;
}
else
{
lean_inc(v_snd_198_);
lean_dec(v_b_190_);
v___x_200_ = lean_box(0);
v_isShared_201_ = v_isSharedCheck_228_;
goto v_resetjp_199_;
}
v_resetjp_199_:
{
lean_object* v___x_202_; lean_object* v_a_204_; lean_object* v_a_211_; 
v___x_202_ = lean_box(0);
v_a_211_ = lean_array_uget_borrowed(v_as_187_, v_i_189_);
if (lean_obj_tag(v_a_211_) == 0)
{
v_a_204_ = v_snd_198_;
goto v___jp_203_;
}
else
{
lean_object* v_val_212_; uint8_t v___x_213_; 
v_val_212_ = lean_ctor_get(v_a_211_, 0);
v___x_213_ = l_Lean_LocalDecl_isAuxDecl(v_val_212_);
if (v___x_213_ == 0)
{
lean_object* v___x_214_; lean_object* v___x_215_; 
v___x_214_ = l_Lean_LocalDecl_type(v_val_212_);
v___x_215_ = l_Lean_Meta_isProp(v___x_214_, v___y_191_, v___y_192_, v___y_193_, v___y_194_);
if (lean_obj_tag(v___x_215_) == 0)
{
lean_object* v_a_216_; uint8_t v___x_217_; 
v_a_216_ = lean_ctor_get(v___x_215_, 0);
lean_inc(v_a_216_);
lean_dec_ref_known(v___x_215_, 1);
v___x_217_ = lean_unbox(v_a_216_);
lean_dec(v_a_216_);
if (v___x_217_ == 0)
{
v_a_204_ = v_snd_198_;
goto v___jp_203_;
}
else
{
lean_object* v___x_218_; lean_object* v___x_219_; 
v___x_218_ = l_Lean_LocalDecl_fvarId(v_val_212_);
v___x_219_ = lean_array_push(v_snd_198_, v___x_218_);
v_a_204_ = v___x_219_;
goto v___jp_203_;
}
}
else
{
lean_object* v_a_220_; lean_object* v___x_222_; uint8_t v_isShared_223_; uint8_t v_isSharedCheck_227_; 
lean_del_object(v___x_200_);
lean_dec(v_snd_198_);
v_a_220_ = lean_ctor_get(v___x_215_, 0);
v_isSharedCheck_227_ = !lean_is_exclusive(v___x_215_);
if (v_isSharedCheck_227_ == 0)
{
v___x_222_ = v___x_215_;
v_isShared_223_ = v_isSharedCheck_227_;
goto v_resetjp_221_;
}
else
{
lean_inc(v_a_220_);
lean_dec(v___x_215_);
v___x_222_ = lean_box(0);
v_isShared_223_ = v_isSharedCheck_227_;
goto v_resetjp_221_;
}
v_resetjp_221_:
{
lean_object* v___x_225_; 
if (v_isShared_223_ == 0)
{
v___x_225_ = v___x_222_;
goto v_reusejp_224_;
}
else
{
lean_object* v_reuseFailAlloc_226_; 
v_reuseFailAlloc_226_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_226_, 0, v_a_220_);
v___x_225_ = v_reuseFailAlloc_226_;
goto v_reusejp_224_;
}
v_reusejp_224_:
{
return v___x_225_;
}
}
}
}
else
{
v_a_204_ = v_snd_198_;
goto v___jp_203_;
}
}
v___jp_203_:
{
lean_object* v___x_206_; 
if (v_isShared_201_ == 0)
{
lean_ctor_set(v___x_200_, 1, v_a_204_);
lean_ctor_set(v___x_200_, 0, v___x_202_);
v___x_206_ = v___x_200_;
goto v_reusejp_205_;
}
else
{
lean_object* v_reuseFailAlloc_210_; 
v_reuseFailAlloc_210_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_210_, 0, v___x_202_);
lean_ctor_set(v_reuseFailAlloc_210_, 1, v_a_204_);
v___x_206_ = v_reuseFailAlloc_210_;
goto v_reusejp_205_;
}
v_reusejp_205_:
{
size_t v___x_207_; size_t v___x_208_; 
v___x_207_ = ((size_t)1ULL);
v___x_208_ = lean_usize_add(v_i_189_, v___x_207_);
v_i_189_ = v___x_208_;
v_b_190_ = v___x_206_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__1_spec__4___boxed(lean_object* v_as_230_, lean_object* v_sz_231_, lean_object* v_i_232_, lean_object* v_b_233_, lean_object* v___y_234_, lean_object* v___y_235_, lean_object* v___y_236_, lean_object* v___y_237_, lean_object* v___y_238_){
_start:
{
size_t v_sz_boxed_239_; size_t v_i_boxed_240_; lean_object* v_res_241_; 
v_sz_boxed_239_ = lean_unbox_usize(v_sz_231_);
lean_dec(v_sz_231_);
v_i_boxed_240_ = lean_unbox_usize(v_i_232_);
lean_dec(v_i_232_);
v_res_241_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__1_spec__4(v_as_230_, v_sz_boxed_239_, v_i_boxed_240_, v_b_233_, v___y_234_, v___y_235_, v___y_236_, v___y_237_);
lean_dec(v___y_237_);
lean_dec_ref(v___y_236_);
lean_dec(v___y_235_);
lean_dec_ref(v___y_234_);
lean_dec_ref(v_as_230_);
return v_res_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__1(lean_object* v_as_242_, size_t v_sz_243_, size_t v_i_244_, lean_object* v_b_245_, lean_object* v___y_246_, lean_object* v___y_247_, lean_object* v___y_248_, lean_object* v___y_249_){
_start:
{
uint8_t v___x_251_; 
v___x_251_ = lean_usize_dec_lt(v_i_244_, v_sz_243_);
if (v___x_251_ == 0)
{
lean_object* v___x_252_; 
v___x_252_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_252_, 0, v_b_245_);
return v___x_252_;
}
else
{
lean_object* v_snd_253_; lean_object* v___x_255_; uint8_t v_isShared_256_; uint8_t v_isSharedCheck_283_; 
v_snd_253_ = lean_ctor_get(v_b_245_, 1);
v_isSharedCheck_283_ = !lean_is_exclusive(v_b_245_);
if (v_isSharedCheck_283_ == 0)
{
lean_object* v_unused_284_; 
v_unused_284_ = lean_ctor_get(v_b_245_, 0);
lean_dec(v_unused_284_);
v___x_255_ = v_b_245_;
v_isShared_256_ = v_isSharedCheck_283_;
goto v_resetjp_254_;
}
else
{
lean_inc(v_snd_253_);
lean_dec(v_b_245_);
v___x_255_ = lean_box(0);
v_isShared_256_ = v_isSharedCheck_283_;
goto v_resetjp_254_;
}
v_resetjp_254_:
{
lean_object* v___x_257_; lean_object* v_a_259_; lean_object* v_a_266_; 
v___x_257_ = lean_box(0);
v_a_266_ = lean_array_uget_borrowed(v_as_242_, v_i_244_);
if (lean_obj_tag(v_a_266_) == 0)
{
v_a_259_ = v_snd_253_;
goto v___jp_258_;
}
else
{
lean_object* v_val_267_; uint8_t v___x_268_; 
v_val_267_ = lean_ctor_get(v_a_266_, 0);
v___x_268_ = l_Lean_LocalDecl_isAuxDecl(v_val_267_);
if (v___x_268_ == 0)
{
lean_object* v___x_269_; lean_object* v___x_270_; 
v___x_269_ = l_Lean_LocalDecl_type(v_val_267_);
v___x_270_ = l_Lean_Meta_isProp(v___x_269_, v___y_246_, v___y_247_, v___y_248_, v___y_249_);
if (lean_obj_tag(v___x_270_) == 0)
{
lean_object* v_a_271_; uint8_t v___x_272_; 
v_a_271_ = lean_ctor_get(v___x_270_, 0);
lean_inc(v_a_271_);
lean_dec_ref_known(v___x_270_, 1);
v___x_272_ = lean_unbox(v_a_271_);
lean_dec(v_a_271_);
if (v___x_272_ == 0)
{
v_a_259_ = v_snd_253_;
goto v___jp_258_;
}
else
{
lean_object* v___x_273_; lean_object* v___x_274_; 
v___x_273_ = l_Lean_LocalDecl_fvarId(v_val_267_);
v___x_274_ = lean_array_push(v_snd_253_, v___x_273_);
v_a_259_ = v___x_274_;
goto v___jp_258_;
}
}
else
{
lean_object* v_a_275_; lean_object* v___x_277_; uint8_t v_isShared_278_; uint8_t v_isSharedCheck_282_; 
lean_del_object(v___x_255_);
lean_dec(v_snd_253_);
v_a_275_ = lean_ctor_get(v___x_270_, 0);
v_isSharedCheck_282_ = !lean_is_exclusive(v___x_270_);
if (v_isSharedCheck_282_ == 0)
{
v___x_277_ = v___x_270_;
v_isShared_278_ = v_isSharedCheck_282_;
goto v_resetjp_276_;
}
else
{
lean_inc(v_a_275_);
lean_dec(v___x_270_);
v___x_277_ = lean_box(0);
v_isShared_278_ = v_isSharedCheck_282_;
goto v_resetjp_276_;
}
v_resetjp_276_:
{
lean_object* v___x_280_; 
if (v_isShared_278_ == 0)
{
v___x_280_ = v___x_277_;
goto v_reusejp_279_;
}
else
{
lean_object* v_reuseFailAlloc_281_; 
v_reuseFailAlloc_281_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_281_, 0, v_a_275_);
v___x_280_ = v_reuseFailAlloc_281_;
goto v_reusejp_279_;
}
v_reusejp_279_:
{
return v___x_280_;
}
}
}
}
else
{
v_a_259_ = v_snd_253_;
goto v___jp_258_;
}
}
v___jp_258_:
{
lean_object* v___x_261_; 
if (v_isShared_256_ == 0)
{
lean_ctor_set(v___x_255_, 1, v_a_259_);
lean_ctor_set(v___x_255_, 0, v___x_257_);
v___x_261_ = v___x_255_;
goto v_reusejp_260_;
}
else
{
lean_object* v_reuseFailAlloc_265_; 
v_reuseFailAlloc_265_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_265_, 0, v___x_257_);
lean_ctor_set(v_reuseFailAlloc_265_, 1, v_a_259_);
v___x_261_ = v_reuseFailAlloc_265_;
goto v_reusejp_260_;
}
v_reusejp_260_:
{
size_t v___x_262_; size_t v___x_263_; lean_object* v___x_264_; 
v___x_262_ = ((size_t)1ULL);
v___x_263_ = lean_usize_add(v_i_244_, v___x_262_);
v___x_264_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__1_spec__4(v_as_242_, v_sz_243_, v___x_263_, v___x_261_, v___y_246_, v___y_247_, v___y_248_, v___y_249_);
return v___x_264_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__1___boxed(lean_object* v_as_285_, lean_object* v_sz_286_, lean_object* v_i_287_, lean_object* v_b_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_){
_start:
{
size_t v_sz_boxed_294_; size_t v_i_boxed_295_; lean_object* v_res_296_; 
v_sz_boxed_294_ = lean_unbox_usize(v_sz_286_);
lean_dec(v_sz_286_);
v_i_boxed_295_ = lean_unbox_usize(v_i_287_);
lean_dec(v_i_287_);
v_res_296_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__1(v_as_285_, v_sz_boxed_294_, v_i_boxed_295_, v_b_288_, v___y_289_, v___y_290_, v___y_291_, v___y_292_);
lean_dec(v___y_292_);
lean_dec_ref(v___y_291_);
lean_dec(v___y_290_);
lean_dec_ref(v___y_289_);
lean_dec_ref(v_as_285_);
return v_res_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__2_spec__3(lean_object* v_as_297_, size_t v_sz_298_, size_t v_i_299_, lean_object* v_b_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_){
_start:
{
uint8_t v___x_306_; 
v___x_306_ = lean_usize_dec_lt(v_i_299_, v_sz_298_);
if (v___x_306_ == 0)
{
lean_object* v___x_307_; 
v___x_307_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_307_, 0, v_b_300_);
return v___x_307_;
}
else
{
lean_object* v_snd_308_; lean_object* v___x_310_; uint8_t v_isShared_311_; uint8_t v_isSharedCheck_338_; 
v_snd_308_ = lean_ctor_get(v_b_300_, 1);
v_isSharedCheck_338_ = !lean_is_exclusive(v_b_300_);
if (v_isSharedCheck_338_ == 0)
{
lean_object* v_unused_339_; 
v_unused_339_ = lean_ctor_get(v_b_300_, 0);
lean_dec(v_unused_339_);
v___x_310_ = v_b_300_;
v_isShared_311_ = v_isSharedCheck_338_;
goto v_resetjp_309_;
}
else
{
lean_inc(v_snd_308_);
lean_dec(v_b_300_);
v___x_310_ = lean_box(0);
v_isShared_311_ = v_isSharedCheck_338_;
goto v_resetjp_309_;
}
v_resetjp_309_:
{
lean_object* v___x_312_; lean_object* v_a_314_; lean_object* v_a_321_; 
v___x_312_ = lean_box(0);
v_a_321_ = lean_array_uget_borrowed(v_as_297_, v_i_299_);
if (lean_obj_tag(v_a_321_) == 0)
{
v_a_314_ = v_snd_308_;
goto v___jp_313_;
}
else
{
lean_object* v_val_322_; uint8_t v___x_323_; 
v_val_322_ = lean_ctor_get(v_a_321_, 0);
v___x_323_ = l_Lean_LocalDecl_isAuxDecl(v_val_322_);
if (v___x_323_ == 0)
{
lean_object* v___x_324_; lean_object* v___x_325_; 
v___x_324_ = l_Lean_LocalDecl_type(v_val_322_);
v___x_325_ = l_Lean_Meta_isProp(v___x_324_, v___y_301_, v___y_302_, v___y_303_, v___y_304_);
if (lean_obj_tag(v___x_325_) == 0)
{
lean_object* v_a_326_; uint8_t v___x_327_; 
v_a_326_ = lean_ctor_get(v___x_325_, 0);
lean_inc(v_a_326_);
lean_dec_ref_known(v___x_325_, 1);
v___x_327_ = lean_unbox(v_a_326_);
lean_dec(v_a_326_);
if (v___x_327_ == 0)
{
v_a_314_ = v_snd_308_;
goto v___jp_313_;
}
else
{
lean_object* v___x_328_; lean_object* v___x_329_; 
v___x_328_ = l_Lean_LocalDecl_fvarId(v_val_322_);
v___x_329_ = lean_array_push(v_snd_308_, v___x_328_);
v_a_314_ = v___x_329_;
goto v___jp_313_;
}
}
else
{
lean_object* v_a_330_; lean_object* v___x_332_; uint8_t v_isShared_333_; uint8_t v_isSharedCheck_337_; 
lean_del_object(v___x_310_);
lean_dec(v_snd_308_);
v_a_330_ = lean_ctor_get(v___x_325_, 0);
v_isSharedCheck_337_ = !lean_is_exclusive(v___x_325_);
if (v_isSharedCheck_337_ == 0)
{
v___x_332_ = v___x_325_;
v_isShared_333_ = v_isSharedCheck_337_;
goto v_resetjp_331_;
}
else
{
lean_inc(v_a_330_);
lean_dec(v___x_325_);
v___x_332_ = lean_box(0);
v_isShared_333_ = v_isSharedCheck_337_;
goto v_resetjp_331_;
}
v_resetjp_331_:
{
lean_object* v___x_335_; 
if (v_isShared_333_ == 0)
{
v___x_335_ = v___x_332_;
goto v_reusejp_334_;
}
else
{
lean_object* v_reuseFailAlloc_336_; 
v_reuseFailAlloc_336_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_336_, 0, v_a_330_);
v___x_335_ = v_reuseFailAlloc_336_;
goto v_reusejp_334_;
}
v_reusejp_334_:
{
return v___x_335_;
}
}
}
}
else
{
v_a_314_ = v_snd_308_;
goto v___jp_313_;
}
}
v___jp_313_:
{
lean_object* v___x_316_; 
if (v_isShared_311_ == 0)
{
lean_ctor_set(v___x_310_, 1, v_a_314_);
lean_ctor_set(v___x_310_, 0, v___x_312_);
v___x_316_ = v___x_310_;
goto v_reusejp_315_;
}
else
{
lean_object* v_reuseFailAlloc_320_; 
v_reuseFailAlloc_320_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_320_, 0, v___x_312_);
lean_ctor_set(v_reuseFailAlloc_320_, 1, v_a_314_);
v___x_316_ = v_reuseFailAlloc_320_;
goto v_reusejp_315_;
}
v_reusejp_315_:
{
size_t v___x_317_; size_t v___x_318_; 
v___x_317_ = ((size_t)1ULL);
v___x_318_ = lean_usize_add(v_i_299_, v___x_317_);
v_i_299_ = v___x_318_;
v_b_300_ = v___x_316_;
goto _start;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__2_spec__3___boxed(lean_object* v_as_340_, lean_object* v_sz_341_, lean_object* v_i_342_, lean_object* v_b_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_){
_start:
{
size_t v_sz_boxed_349_; size_t v_i_boxed_350_; lean_object* v_res_351_; 
v_sz_boxed_349_ = lean_unbox_usize(v_sz_341_);
lean_dec(v_sz_341_);
v_i_boxed_350_ = lean_unbox_usize(v_i_342_);
lean_dec(v_i_342_);
v_res_351_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__2_spec__3(v_as_340_, v_sz_boxed_349_, v_i_boxed_350_, v_b_343_, v___y_344_, v___y_345_, v___y_346_, v___y_347_);
lean_dec(v___y_347_);
lean_dec_ref(v___y_346_);
lean_dec(v___y_345_);
lean_dec_ref(v___y_344_);
lean_dec_ref(v_as_340_);
return v_res_351_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__2(lean_object* v_as_352_, size_t v_sz_353_, size_t v_i_354_, lean_object* v_b_355_, lean_object* v___y_356_, lean_object* v___y_357_, lean_object* v___y_358_, lean_object* v___y_359_){
_start:
{
uint8_t v___x_361_; 
v___x_361_ = lean_usize_dec_lt(v_i_354_, v_sz_353_);
if (v___x_361_ == 0)
{
lean_object* v___x_362_; 
v___x_362_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_362_, 0, v_b_355_);
return v___x_362_;
}
else
{
lean_object* v_snd_363_; lean_object* v___x_365_; uint8_t v_isShared_366_; uint8_t v_isSharedCheck_393_; 
v_snd_363_ = lean_ctor_get(v_b_355_, 1);
v_isSharedCheck_393_ = !lean_is_exclusive(v_b_355_);
if (v_isSharedCheck_393_ == 0)
{
lean_object* v_unused_394_; 
v_unused_394_ = lean_ctor_get(v_b_355_, 0);
lean_dec(v_unused_394_);
v___x_365_ = v_b_355_;
v_isShared_366_ = v_isSharedCheck_393_;
goto v_resetjp_364_;
}
else
{
lean_inc(v_snd_363_);
lean_dec(v_b_355_);
v___x_365_ = lean_box(0);
v_isShared_366_ = v_isSharedCheck_393_;
goto v_resetjp_364_;
}
v_resetjp_364_:
{
lean_object* v___x_367_; lean_object* v_a_369_; lean_object* v_a_376_; 
v___x_367_ = lean_box(0);
v_a_376_ = lean_array_uget_borrowed(v_as_352_, v_i_354_);
if (lean_obj_tag(v_a_376_) == 0)
{
v_a_369_ = v_snd_363_;
goto v___jp_368_;
}
else
{
lean_object* v_val_377_; uint8_t v___x_378_; 
v_val_377_ = lean_ctor_get(v_a_376_, 0);
v___x_378_ = l_Lean_LocalDecl_isAuxDecl(v_val_377_);
if (v___x_378_ == 0)
{
lean_object* v___x_379_; lean_object* v___x_380_; 
v___x_379_ = l_Lean_LocalDecl_type(v_val_377_);
v___x_380_ = l_Lean_Meta_isProp(v___x_379_, v___y_356_, v___y_357_, v___y_358_, v___y_359_);
if (lean_obj_tag(v___x_380_) == 0)
{
lean_object* v_a_381_; uint8_t v___x_382_; 
v_a_381_ = lean_ctor_get(v___x_380_, 0);
lean_inc(v_a_381_);
lean_dec_ref_known(v___x_380_, 1);
v___x_382_ = lean_unbox(v_a_381_);
lean_dec(v_a_381_);
if (v___x_382_ == 0)
{
v_a_369_ = v_snd_363_;
goto v___jp_368_;
}
else
{
lean_object* v___x_383_; lean_object* v___x_384_; 
v___x_383_ = l_Lean_LocalDecl_fvarId(v_val_377_);
v___x_384_ = lean_array_push(v_snd_363_, v___x_383_);
v_a_369_ = v___x_384_;
goto v___jp_368_;
}
}
else
{
lean_object* v_a_385_; lean_object* v___x_387_; uint8_t v_isShared_388_; uint8_t v_isSharedCheck_392_; 
lean_del_object(v___x_365_);
lean_dec(v_snd_363_);
v_a_385_ = lean_ctor_get(v___x_380_, 0);
v_isSharedCheck_392_ = !lean_is_exclusive(v___x_380_);
if (v_isSharedCheck_392_ == 0)
{
v___x_387_ = v___x_380_;
v_isShared_388_ = v_isSharedCheck_392_;
goto v_resetjp_386_;
}
else
{
lean_inc(v_a_385_);
lean_dec(v___x_380_);
v___x_387_ = lean_box(0);
v_isShared_388_ = v_isSharedCheck_392_;
goto v_resetjp_386_;
}
v_resetjp_386_:
{
lean_object* v___x_390_; 
if (v_isShared_388_ == 0)
{
v___x_390_ = v___x_387_;
goto v_reusejp_389_;
}
else
{
lean_object* v_reuseFailAlloc_391_; 
v_reuseFailAlloc_391_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_391_, 0, v_a_385_);
v___x_390_ = v_reuseFailAlloc_391_;
goto v_reusejp_389_;
}
v_reusejp_389_:
{
return v___x_390_;
}
}
}
}
else
{
v_a_369_ = v_snd_363_;
goto v___jp_368_;
}
}
v___jp_368_:
{
lean_object* v___x_371_; 
if (v_isShared_366_ == 0)
{
lean_ctor_set(v___x_365_, 1, v_a_369_);
lean_ctor_set(v___x_365_, 0, v___x_367_);
v___x_371_ = v___x_365_;
goto v_reusejp_370_;
}
else
{
lean_object* v_reuseFailAlloc_375_; 
v_reuseFailAlloc_375_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_375_, 0, v___x_367_);
lean_ctor_set(v_reuseFailAlloc_375_, 1, v_a_369_);
v___x_371_ = v_reuseFailAlloc_375_;
goto v_reusejp_370_;
}
v_reusejp_370_:
{
size_t v___x_372_; size_t v___x_373_; lean_object* v___x_374_; 
v___x_372_ = ((size_t)1ULL);
v___x_373_ = lean_usize_add(v_i_354_, v___x_372_);
v___x_374_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__2_spec__3(v_as_352_, v_sz_353_, v___x_373_, v___x_371_, v___y_356_, v___y_357_, v___y_358_, v___y_359_);
return v___x_374_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__2___boxed(lean_object* v_as_395_, lean_object* v_sz_396_, lean_object* v_i_397_, lean_object* v_b_398_, lean_object* v___y_399_, lean_object* v___y_400_, lean_object* v___y_401_, lean_object* v___y_402_, lean_object* v___y_403_){
_start:
{
size_t v_sz_boxed_404_; size_t v_i_boxed_405_; lean_object* v_res_406_; 
v_sz_boxed_404_ = lean_unbox_usize(v_sz_396_);
lean_dec(v_sz_396_);
v_i_boxed_405_ = lean_unbox_usize(v_i_397_);
lean_dec(v_i_397_);
v_res_406_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__2(v_as_395_, v_sz_boxed_404_, v_i_boxed_405_, v_b_398_, v___y_399_, v___y_400_, v___y_401_, v___y_402_);
lean_dec(v___y_402_);
lean_dec_ref(v___y_401_);
lean_dec(v___y_400_);
lean_dec_ref(v___y_399_);
lean_dec_ref(v_as_395_);
return v_res_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0(lean_object* v_init_407_, lean_object* v_n_408_, lean_object* v_b_409_, lean_object* v___y_410_, lean_object* v___y_411_, lean_object* v___y_412_, lean_object* v___y_413_){
_start:
{
if (lean_obj_tag(v_n_408_) == 0)
{
lean_object* v_cs_415_; lean_object* v___x_416_; lean_object* v___x_417_; size_t v_sz_418_; size_t v___x_419_; lean_object* v___x_420_; 
v_cs_415_ = lean_ctor_get(v_n_408_, 0);
v___x_416_ = lean_box(0);
v___x_417_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_417_, 0, v___x_416_);
lean_ctor_set(v___x_417_, 1, v_b_409_);
v_sz_418_ = lean_array_size(v_cs_415_);
v___x_419_ = ((size_t)0ULL);
v___x_420_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__1(v_init_407_, v_cs_415_, v_sz_418_, v___x_419_, v___x_417_, v___y_410_, v___y_411_, v___y_412_, v___y_413_);
if (lean_obj_tag(v___x_420_) == 0)
{
lean_object* v_a_421_; lean_object* v___x_423_; uint8_t v_isShared_424_; uint8_t v_isSharedCheck_435_; 
v_a_421_ = lean_ctor_get(v___x_420_, 0);
v_isSharedCheck_435_ = !lean_is_exclusive(v___x_420_);
if (v_isSharedCheck_435_ == 0)
{
v___x_423_ = v___x_420_;
v_isShared_424_ = v_isSharedCheck_435_;
goto v_resetjp_422_;
}
else
{
lean_inc(v_a_421_);
lean_dec(v___x_420_);
v___x_423_ = lean_box(0);
v_isShared_424_ = v_isSharedCheck_435_;
goto v_resetjp_422_;
}
v_resetjp_422_:
{
lean_object* v_fst_425_; 
v_fst_425_ = lean_ctor_get(v_a_421_, 0);
if (lean_obj_tag(v_fst_425_) == 0)
{
lean_object* v_snd_426_; lean_object* v___x_427_; lean_object* v___x_429_; 
v_snd_426_ = lean_ctor_get(v_a_421_, 1);
lean_inc(v_snd_426_);
lean_dec(v_a_421_);
v___x_427_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_427_, 0, v_snd_426_);
if (v_isShared_424_ == 0)
{
lean_ctor_set(v___x_423_, 0, v___x_427_);
v___x_429_ = v___x_423_;
goto v_reusejp_428_;
}
else
{
lean_object* v_reuseFailAlloc_430_; 
v_reuseFailAlloc_430_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_430_, 0, v___x_427_);
v___x_429_ = v_reuseFailAlloc_430_;
goto v_reusejp_428_;
}
v_reusejp_428_:
{
return v___x_429_;
}
}
else
{
lean_object* v_val_431_; lean_object* v___x_433_; 
lean_inc_ref(v_fst_425_);
lean_dec(v_a_421_);
v_val_431_ = lean_ctor_get(v_fst_425_, 0);
lean_inc(v_val_431_);
lean_dec_ref_known(v_fst_425_, 1);
if (v_isShared_424_ == 0)
{
lean_ctor_set(v___x_423_, 0, v_val_431_);
v___x_433_ = v___x_423_;
goto v_reusejp_432_;
}
else
{
lean_object* v_reuseFailAlloc_434_; 
v_reuseFailAlloc_434_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_434_, 0, v_val_431_);
v___x_433_ = v_reuseFailAlloc_434_;
goto v_reusejp_432_;
}
v_reusejp_432_:
{
return v___x_433_;
}
}
}
}
else
{
lean_object* v_a_436_; lean_object* v___x_438_; uint8_t v_isShared_439_; uint8_t v_isSharedCheck_443_; 
v_a_436_ = lean_ctor_get(v___x_420_, 0);
v_isSharedCheck_443_ = !lean_is_exclusive(v___x_420_);
if (v_isSharedCheck_443_ == 0)
{
v___x_438_ = v___x_420_;
v_isShared_439_ = v_isSharedCheck_443_;
goto v_resetjp_437_;
}
else
{
lean_inc(v_a_436_);
lean_dec(v___x_420_);
v___x_438_ = lean_box(0);
v_isShared_439_ = v_isSharedCheck_443_;
goto v_resetjp_437_;
}
v_resetjp_437_:
{
lean_object* v___x_441_; 
if (v_isShared_439_ == 0)
{
v___x_441_ = v___x_438_;
goto v_reusejp_440_;
}
else
{
lean_object* v_reuseFailAlloc_442_; 
v_reuseFailAlloc_442_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_442_, 0, v_a_436_);
v___x_441_ = v_reuseFailAlloc_442_;
goto v_reusejp_440_;
}
v_reusejp_440_:
{
return v___x_441_;
}
}
}
}
else
{
lean_object* v_vs_444_; lean_object* v___x_445_; lean_object* v___x_446_; size_t v_sz_447_; size_t v___x_448_; lean_object* v___x_449_; 
v_vs_444_ = lean_ctor_get(v_n_408_, 0);
v___x_445_ = lean_box(0);
v___x_446_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_446_, 0, v___x_445_);
lean_ctor_set(v___x_446_, 1, v_b_409_);
v_sz_447_ = lean_array_size(v_vs_444_);
v___x_448_ = ((size_t)0ULL);
v___x_449_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__2(v_vs_444_, v_sz_447_, v___x_448_, v___x_446_, v___y_410_, v___y_411_, v___y_412_, v___y_413_);
if (lean_obj_tag(v___x_449_) == 0)
{
lean_object* v_a_450_; lean_object* v___x_452_; uint8_t v_isShared_453_; uint8_t v_isSharedCheck_464_; 
v_a_450_ = lean_ctor_get(v___x_449_, 0);
v_isSharedCheck_464_ = !lean_is_exclusive(v___x_449_);
if (v_isSharedCheck_464_ == 0)
{
v___x_452_ = v___x_449_;
v_isShared_453_ = v_isSharedCheck_464_;
goto v_resetjp_451_;
}
else
{
lean_inc(v_a_450_);
lean_dec(v___x_449_);
v___x_452_ = lean_box(0);
v_isShared_453_ = v_isSharedCheck_464_;
goto v_resetjp_451_;
}
v_resetjp_451_:
{
lean_object* v_fst_454_; 
v_fst_454_ = lean_ctor_get(v_a_450_, 0);
if (lean_obj_tag(v_fst_454_) == 0)
{
lean_object* v_snd_455_; lean_object* v___x_456_; lean_object* v___x_458_; 
v_snd_455_ = lean_ctor_get(v_a_450_, 1);
lean_inc(v_snd_455_);
lean_dec(v_a_450_);
v___x_456_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_456_, 0, v_snd_455_);
if (v_isShared_453_ == 0)
{
lean_ctor_set(v___x_452_, 0, v___x_456_);
v___x_458_ = v___x_452_;
goto v_reusejp_457_;
}
else
{
lean_object* v_reuseFailAlloc_459_; 
v_reuseFailAlloc_459_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_459_, 0, v___x_456_);
v___x_458_ = v_reuseFailAlloc_459_;
goto v_reusejp_457_;
}
v_reusejp_457_:
{
return v___x_458_;
}
}
else
{
lean_object* v_val_460_; lean_object* v___x_462_; 
lean_inc_ref(v_fst_454_);
lean_dec(v_a_450_);
v_val_460_ = lean_ctor_get(v_fst_454_, 0);
lean_inc(v_val_460_);
lean_dec_ref_known(v_fst_454_, 1);
if (v_isShared_453_ == 0)
{
lean_ctor_set(v___x_452_, 0, v_val_460_);
v___x_462_ = v___x_452_;
goto v_reusejp_461_;
}
else
{
lean_object* v_reuseFailAlloc_463_; 
v_reuseFailAlloc_463_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_463_, 0, v_val_460_);
v___x_462_ = v_reuseFailAlloc_463_;
goto v_reusejp_461_;
}
v_reusejp_461_:
{
return v___x_462_;
}
}
}
}
else
{
lean_object* v_a_465_; lean_object* v___x_467_; uint8_t v_isShared_468_; uint8_t v_isSharedCheck_472_; 
v_a_465_ = lean_ctor_get(v___x_449_, 0);
v_isSharedCheck_472_ = !lean_is_exclusive(v___x_449_);
if (v_isSharedCheck_472_ == 0)
{
v___x_467_ = v___x_449_;
v_isShared_468_ = v_isSharedCheck_472_;
goto v_resetjp_466_;
}
else
{
lean_inc(v_a_465_);
lean_dec(v___x_449_);
v___x_467_ = lean_box(0);
v_isShared_468_ = v_isSharedCheck_472_;
goto v_resetjp_466_;
}
v_resetjp_466_:
{
lean_object* v___x_470_; 
if (v_isShared_468_ == 0)
{
v___x_470_ = v___x_467_;
goto v_reusejp_469_;
}
else
{
lean_object* v_reuseFailAlloc_471_; 
v_reuseFailAlloc_471_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_471_, 0, v_a_465_);
v___x_470_ = v_reuseFailAlloc_471_;
goto v_reusejp_469_;
}
v_reusejp_469_:
{
return v___x_470_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__1(lean_object* v_init_473_, lean_object* v_as_474_, size_t v_sz_475_, size_t v_i_476_, lean_object* v_b_477_, lean_object* v___y_478_, lean_object* v___y_479_, lean_object* v___y_480_, lean_object* v___y_481_){
_start:
{
uint8_t v___x_483_; 
v___x_483_ = lean_usize_dec_lt(v_i_476_, v_sz_475_);
if (v___x_483_ == 0)
{
lean_object* v___x_484_; 
v___x_484_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_484_, 0, v_b_477_);
return v___x_484_;
}
else
{
lean_object* v_snd_485_; lean_object* v___x_487_; uint8_t v_isShared_488_; uint8_t v_isSharedCheck_519_; 
v_snd_485_ = lean_ctor_get(v_b_477_, 1);
v_isSharedCheck_519_ = !lean_is_exclusive(v_b_477_);
if (v_isSharedCheck_519_ == 0)
{
lean_object* v_unused_520_; 
v_unused_520_ = lean_ctor_get(v_b_477_, 0);
lean_dec(v_unused_520_);
v___x_487_ = v_b_477_;
v_isShared_488_ = v_isSharedCheck_519_;
goto v_resetjp_486_;
}
else
{
lean_inc(v_snd_485_);
lean_dec(v_b_477_);
v___x_487_ = lean_box(0);
v_isShared_488_ = v_isSharedCheck_519_;
goto v_resetjp_486_;
}
v_resetjp_486_:
{
lean_object* v_a_489_; lean_object* v___x_490_; 
v_a_489_ = lean_array_uget_borrowed(v_as_474_, v_i_476_);
lean_inc(v_snd_485_);
v___x_490_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0(v_init_473_, v_a_489_, v_snd_485_, v___y_478_, v___y_479_, v___y_480_, v___y_481_);
if (lean_obj_tag(v___x_490_) == 0)
{
lean_object* v_a_491_; lean_object* v___x_493_; uint8_t v_isShared_494_; uint8_t v_isSharedCheck_510_; 
v_a_491_ = lean_ctor_get(v___x_490_, 0);
v_isSharedCheck_510_ = !lean_is_exclusive(v___x_490_);
if (v_isSharedCheck_510_ == 0)
{
v___x_493_ = v___x_490_;
v_isShared_494_ = v_isSharedCheck_510_;
goto v_resetjp_492_;
}
else
{
lean_inc(v_a_491_);
lean_dec(v___x_490_);
v___x_493_ = lean_box(0);
v_isShared_494_ = v_isSharedCheck_510_;
goto v_resetjp_492_;
}
v_resetjp_492_:
{
if (lean_obj_tag(v_a_491_) == 0)
{
lean_object* v___x_495_; lean_object* v___x_497_; 
v___x_495_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_495_, 0, v_a_491_);
if (v_isShared_488_ == 0)
{
lean_ctor_set(v___x_487_, 0, v___x_495_);
v___x_497_ = v___x_487_;
goto v_reusejp_496_;
}
else
{
lean_object* v_reuseFailAlloc_501_; 
v_reuseFailAlloc_501_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_501_, 0, v___x_495_);
lean_ctor_set(v_reuseFailAlloc_501_, 1, v_snd_485_);
v___x_497_ = v_reuseFailAlloc_501_;
goto v_reusejp_496_;
}
v_reusejp_496_:
{
lean_object* v___x_499_; 
if (v_isShared_494_ == 0)
{
lean_ctor_set(v___x_493_, 0, v___x_497_);
v___x_499_ = v___x_493_;
goto v_reusejp_498_;
}
else
{
lean_object* v_reuseFailAlloc_500_; 
v_reuseFailAlloc_500_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_500_, 0, v___x_497_);
v___x_499_ = v_reuseFailAlloc_500_;
goto v_reusejp_498_;
}
v_reusejp_498_:
{
return v___x_499_;
}
}
}
else
{
lean_object* v_a_502_; lean_object* v___x_503_; lean_object* v___x_505_; 
lean_del_object(v___x_493_);
lean_dec(v_snd_485_);
v_a_502_ = lean_ctor_get(v_a_491_, 0);
lean_inc(v_a_502_);
lean_dec_ref_known(v_a_491_, 1);
v___x_503_ = lean_box(0);
if (v_isShared_488_ == 0)
{
lean_ctor_set(v___x_487_, 1, v_a_502_);
lean_ctor_set(v___x_487_, 0, v___x_503_);
v___x_505_ = v___x_487_;
goto v_reusejp_504_;
}
else
{
lean_object* v_reuseFailAlloc_509_; 
v_reuseFailAlloc_509_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_509_, 0, v___x_503_);
lean_ctor_set(v_reuseFailAlloc_509_, 1, v_a_502_);
v___x_505_ = v_reuseFailAlloc_509_;
goto v_reusejp_504_;
}
v_reusejp_504_:
{
size_t v___x_506_; size_t v___x_507_; 
v___x_506_ = ((size_t)1ULL);
v___x_507_ = lean_usize_add(v_i_476_, v___x_506_);
v_i_476_ = v___x_507_;
v_b_477_ = v___x_505_;
goto _start;
}
}
}
}
else
{
lean_object* v_a_511_; lean_object* v___x_513_; uint8_t v_isShared_514_; uint8_t v_isSharedCheck_518_; 
lean_del_object(v___x_487_);
lean_dec(v_snd_485_);
v_a_511_ = lean_ctor_get(v___x_490_, 0);
v_isSharedCheck_518_ = !lean_is_exclusive(v___x_490_);
if (v_isSharedCheck_518_ == 0)
{
v___x_513_ = v___x_490_;
v_isShared_514_ = v_isSharedCheck_518_;
goto v_resetjp_512_;
}
else
{
lean_inc(v_a_511_);
lean_dec(v___x_490_);
v___x_513_ = lean_box(0);
v_isShared_514_ = v_isSharedCheck_518_;
goto v_resetjp_512_;
}
v_resetjp_512_:
{
lean_object* v___x_516_; 
if (v_isShared_514_ == 0)
{
v___x_516_ = v___x_513_;
goto v_reusejp_515_;
}
else
{
lean_object* v_reuseFailAlloc_517_; 
v_reuseFailAlloc_517_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_517_, 0, v_a_511_);
v___x_516_ = v_reuseFailAlloc_517_;
goto v_reusejp_515_;
}
v_reusejp_515_:
{
return v___x_516_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__1___boxed(lean_object* v_init_521_, lean_object* v_as_522_, lean_object* v_sz_523_, lean_object* v_i_524_, lean_object* v_b_525_, lean_object* v___y_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_, lean_object* v___y_530_){
_start:
{
size_t v_sz_boxed_531_; size_t v_i_boxed_532_; lean_object* v_res_533_; 
v_sz_boxed_531_ = lean_unbox_usize(v_sz_523_);
lean_dec(v_sz_523_);
v_i_boxed_532_ = lean_unbox_usize(v_i_524_);
lean_dec(v_i_524_);
v_res_533_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0_spec__1(v_init_521_, v_as_522_, v_sz_boxed_531_, v_i_boxed_532_, v_b_525_, v___y_526_, v___y_527_, v___y_528_, v___y_529_);
lean_dec(v___y_529_);
lean_dec_ref(v___y_528_);
lean_dec(v___y_527_);
lean_dec_ref(v___y_526_);
lean_dec_ref(v_as_522_);
lean_dec_ref(v_init_521_);
return v_res_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0___boxed(lean_object* v_init_534_, lean_object* v_n_535_, lean_object* v_b_536_, lean_object* v___y_537_, lean_object* v___y_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_){
_start:
{
lean_object* v_res_542_; 
v_res_542_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0(v_init_534_, v_n_535_, v_b_536_, v___y_537_, v___y_538_, v___y_539_, v___y_540_);
lean_dec(v___y_540_);
lean_dec_ref(v___y_539_);
lean_dec(v___y_538_);
lean_dec_ref(v___y_537_);
lean_dec_ref(v_n_535_);
lean_dec_ref(v_init_534_);
return v_res_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0(lean_object* v_t_543_, lean_object* v_init_544_, lean_object* v___y_545_, lean_object* v___y_546_, lean_object* v___y_547_, lean_object* v___y_548_){
_start:
{
lean_object* v_root_550_; lean_object* v_tail_551_; lean_object* v___x_552_; 
v_root_550_ = lean_ctor_get(v_t_543_, 0);
v_tail_551_ = lean_ctor_get(v_t_543_, 1);
lean_inc_ref(v_init_544_);
v___x_552_ = lp_mathlib_Lean_PersistentArray_forInAux___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__0(v_init_544_, v_root_550_, v_init_544_, v___y_545_, v___y_546_, v___y_547_, v___y_548_);
lean_dec_ref(v_init_544_);
if (lean_obj_tag(v___x_552_) == 0)
{
lean_object* v_a_553_; lean_object* v___x_555_; uint8_t v_isShared_556_; uint8_t v_isSharedCheck_589_; 
v_a_553_ = lean_ctor_get(v___x_552_, 0);
v_isSharedCheck_589_ = !lean_is_exclusive(v___x_552_);
if (v_isSharedCheck_589_ == 0)
{
v___x_555_ = v___x_552_;
v_isShared_556_ = v_isSharedCheck_589_;
goto v_resetjp_554_;
}
else
{
lean_inc(v_a_553_);
lean_dec(v___x_552_);
v___x_555_ = lean_box(0);
v_isShared_556_ = v_isSharedCheck_589_;
goto v_resetjp_554_;
}
v_resetjp_554_:
{
if (lean_obj_tag(v_a_553_) == 0)
{
lean_object* v_a_557_; lean_object* v___x_559_; 
v_a_557_ = lean_ctor_get(v_a_553_, 0);
lean_inc(v_a_557_);
lean_dec_ref_known(v_a_553_, 1);
if (v_isShared_556_ == 0)
{
lean_ctor_set(v___x_555_, 0, v_a_557_);
v___x_559_ = v___x_555_;
goto v_reusejp_558_;
}
else
{
lean_object* v_reuseFailAlloc_560_; 
v_reuseFailAlloc_560_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_560_, 0, v_a_557_);
v___x_559_ = v_reuseFailAlloc_560_;
goto v_reusejp_558_;
}
v_reusejp_558_:
{
return v___x_559_;
}
}
else
{
lean_object* v_a_561_; lean_object* v___x_562_; lean_object* v___x_563_; size_t v_sz_564_; size_t v___x_565_; lean_object* v___x_566_; 
lean_del_object(v___x_555_);
v_a_561_ = lean_ctor_get(v_a_553_, 0);
lean_inc(v_a_561_);
lean_dec_ref_known(v_a_553_, 1);
v___x_562_ = lean_box(0);
v___x_563_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_563_, 0, v___x_562_);
lean_ctor_set(v___x_563_, 1, v_a_561_);
v_sz_564_ = lean_array_size(v_tail_551_);
v___x_565_ = ((size_t)0ULL);
v___x_566_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0_spec__1(v_tail_551_, v_sz_564_, v___x_565_, v___x_563_, v___y_545_, v___y_546_, v___y_547_, v___y_548_);
if (lean_obj_tag(v___x_566_) == 0)
{
lean_object* v_a_567_; lean_object* v___x_569_; uint8_t v_isShared_570_; uint8_t v_isSharedCheck_580_; 
v_a_567_ = lean_ctor_get(v___x_566_, 0);
v_isSharedCheck_580_ = !lean_is_exclusive(v___x_566_);
if (v_isSharedCheck_580_ == 0)
{
v___x_569_ = v___x_566_;
v_isShared_570_ = v_isSharedCheck_580_;
goto v_resetjp_568_;
}
else
{
lean_inc(v_a_567_);
lean_dec(v___x_566_);
v___x_569_ = lean_box(0);
v_isShared_570_ = v_isSharedCheck_580_;
goto v_resetjp_568_;
}
v_resetjp_568_:
{
lean_object* v_fst_571_; 
v_fst_571_ = lean_ctor_get(v_a_567_, 0);
if (lean_obj_tag(v_fst_571_) == 0)
{
lean_object* v_snd_572_; lean_object* v___x_574_; 
v_snd_572_ = lean_ctor_get(v_a_567_, 1);
lean_inc(v_snd_572_);
lean_dec(v_a_567_);
if (v_isShared_570_ == 0)
{
lean_ctor_set(v___x_569_, 0, v_snd_572_);
v___x_574_ = v___x_569_;
goto v_reusejp_573_;
}
else
{
lean_object* v_reuseFailAlloc_575_; 
v_reuseFailAlloc_575_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_575_, 0, v_snd_572_);
v___x_574_ = v_reuseFailAlloc_575_;
goto v_reusejp_573_;
}
v_reusejp_573_:
{
return v___x_574_;
}
}
else
{
lean_object* v_val_576_; lean_object* v___x_578_; 
lean_inc_ref(v_fst_571_);
lean_dec(v_a_567_);
v_val_576_ = lean_ctor_get(v_fst_571_, 0);
lean_inc(v_val_576_);
lean_dec_ref_known(v_fst_571_, 1);
if (v_isShared_570_ == 0)
{
lean_ctor_set(v___x_569_, 0, v_val_576_);
v___x_578_ = v___x_569_;
goto v_reusejp_577_;
}
else
{
lean_object* v_reuseFailAlloc_579_; 
v_reuseFailAlloc_579_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_579_, 0, v_val_576_);
v___x_578_ = v_reuseFailAlloc_579_;
goto v_reusejp_577_;
}
v_reusejp_577_:
{
return v___x_578_;
}
}
}
}
else
{
lean_object* v_a_581_; lean_object* v___x_583_; uint8_t v_isShared_584_; uint8_t v_isSharedCheck_588_; 
v_a_581_ = lean_ctor_get(v___x_566_, 0);
v_isSharedCheck_588_ = !lean_is_exclusive(v___x_566_);
if (v_isSharedCheck_588_ == 0)
{
v___x_583_ = v___x_566_;
v_isShared_584_ = v_isSharedCheck_588_;
goto v_resetjp_582_;
}
else
{
lean_inc(v_a_581_);
lean_dec(v___x_566_);
v___x_583_ = lean_box(0);
v_isShared_584_ = v_isSharedCheck_588_;
goto v_resetjp_582_;
}
v_resetjp_582_:
{
lean_object* v___x_586_; 
if (v_isShared_584_ == 0)
{
v___x_586_ = v___x_583_;
goto v_reusejp_585_;
}
else
{
lean_object* v_reuseFailAlloc_587_; 
v_reuseFailAlloc_587_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_587_, 0, v_a_581_);
v___x_586_ = v_reuseFailAlloc_587_;
goto v_reusejp_585_;
}
v_reusejp_585_:
{
return v___x_586_;
}
}
}
}
}
}
else
{
lean_object* v_a_590_; lean_object* v___x_592_; uint8_t v_isShared_593_; uint8_t v_isSharedCheck_597_; 
v_a_590_ = lean_ctor_get(v___x_552_, 0);
v_isSharedCheck_597_ = !lean_is_exclusive(v___x_552_);
if (v_isSharedCheck_597_ == 0)
{
v___x_592_ = v___x_552_;
v_isShared_593_ = v_isSharedCheck_597_;
goto v_resetjp_591_;
}
else
{
lean_inc(v_a_590_);
lean_dec(v___x_552_);
v___x_592_ = lean_box(0);
v_isShared_593_ = v_isSharedCheck_597_;
goto v_resetjp_591_;
}
v_resetjp_591_:
{
lean_object* v___x_595_; 
if (v_isShared_593_ == 0)
{
v___x_595_ = v___x_592_;
goto v_reusejp_594_;
}
else
{
lean_object* v_reuseFailAlloc_596_; 
v_reuseFailAlloc_596_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_596_, 0, v_a_590_);
v___x_595_ = v_reuseFailAlloc_596_;
goto v_reusejp_594_;
}
v_reusejp_594_:
{
return v___x_595_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0___boxed(lean_object* v_t_598_, lean_object* v_init_599_, lean_object* v___y_600_, lean_object* v___y_601_, lean_object* v___y_602_, lean_object* v___y_603_, lean_object* v___y_604_){
_start:
{
lean_object* v_res_605_; 
v_res_605_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0(v_t_598_, v_init_599_, v___y_600_, v___y_601_, v___y_602_, v___y_603_);
lean_dec(v___y_603_);
lean_dec_ref(v___y_602_);
lean_dec(v___y_601_);
lean_dec_ref(v___y_600_);
lean_dec_ref(v_t_598_);
return v_res_605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_getPropHyps(lean_object* v_a_608_, lean_object* v_a_609_, lean_object* v_a_610_, lean_object* v_a_611_){
_start:
{
lean_object* v_lctx_613_; lean_object* v_decls_614_; lean_object* v_result_615_; lean_object* v___x_616_; 
v_lctx_613_ = lean_ctor_get(v_a_608_, 2);
v_decls_614_ = lean_ctor_get(v_lctx_613_, 1);
v_result_615_ = ((lean_object*)(lp_mathlib_Lean_Meta_Simp_getPropHyps___closed__0));
v___x_616_ = lp_mathlib_Lean_PersistentArray_forIn___at___00Lean_Meta_Simp_getPropHyps_spec__0(v_decls_614_, v_result_615_, v_a_608_, v_a_609_, v_a_610_, v_a_611_);
return v___x_616_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_getPropHyps___boxed(lean_object* v_a_617_, lean_object* v_a_618_, lean_object* v_a_619_, lean_object* v_a_620_, lean_object* v_a_621_){
_start:
{
lean_object* v_res_622_; 
v_res_622_ = lp_mathlib_Lean_Meta_Simp_getPropHyps(v_a_617_, v_a_618_, v_a_619_, v_a_620_);
lean_dec(v_a_620_);
lean_dec_ref(v_a_619_);
lean_dec(v_a_618_);
lean_dec_ref(v_a_617_);
return v_res_622_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__1___closed__0(void){
_start:
{
lean_object* v___x_623_; 
v___x_623_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_623_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__1___closed__1(void){
_start:
{
lean_object* v___x_624_; lean_object* v___x_625_; 
v___x_624_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__1___closed__0, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__1___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__1___closed__0);
v___x_625_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_625_, 0, v___x_624_);
return v___x_625_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__1(lean_object* v_00_u03b2_626_){
_start:
{
lean_object* v___x_627_; 
v___x_627_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__1___closed__1, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__1___closed__1_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__1___closed__1);
return v___x_627_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__2___closed__0(void){
_start:
{
lean_object* v___x_628_; 
v___x_628_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_628_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__2___closed__1(void){
_start:
{
lean_object* v___x_629_; lean_object* v___x_630_; 
v___x_629_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__2___closed__0, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__2___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__2___closed__0);
v___x_630_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_630_, 0, v___x_629_);
return v___x_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__2(lean_object* v_00_u03b2_631_){
_start:
{
lean_object* v___x_632_; 
v___x_632_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__2___closed__1, &lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__2___closed__1_once, _init_lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__2___closed__1);
return v___x_632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_Meta_simpTheoremsOfNames_spec__3(uint8_t v_simpOnly_633_, lean_object* v_x_634_, lean_object* v_x_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_){
_start:
{
if (lean_obj_tag(v_x_635_) == 0)
{
lean_object* v___x_641_; 
v___x_641_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_641_, 0, v_x_634_);
return v___x_641_;
}
else
{
lean_object* v_head_642_; lean_object* v_tail_643_; uint8_t v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; 
v_head_642_ = lean_ctor_get(v_x_635_, 0);
lean_inc(v_head_642_);
v_tail_643_ = lean_ctor_get(v_x_635_, 1);
lean_inc(v_tail_643_);
lean_dec_ref_known(v_x_635_, 2);
v___x_644_ = 0;
v___x_645_ = lean_unsigned_to_nat(1000u);
v___x_646_ = l_Lean_Meta_SimpTheorems_addConst(v_x_634_, v_head_642_, v_simpOnly_633_, v___x_644_, v___x_645_, v___y_636_, v___y_637_, v___y_638_, v___y_639_);
if (lean_obj_tag(v___x_646_) == 0)
{
lean_object* v_a_647_; 
v_a_647_ = lean_ctor_get(v___x_646_, 0);
lean_inc(v_a_647_);
lean_dec_ref_known(v___x_646_, 1);
v_x_634_ = v_a_647_;
v_x_635_ = v_tail_643_;
goto _start;
}
else
{
lean_dec(v_tail_643_);
return v___x_646_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_Meta_simpTheoremsOfNames_spec__3___boxed(lean_object* v_simpOnly_649_, lean_object* v_x_650_, lean_object* v_x_651_, lean_object* v___y_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_){
_start:
{
uint8_t v_simpOnly_boxed_657_; lean_object* v_res_658_; 
v_simpOnly_boxed_657_ = lean_unbox(v_simpOnly_649_);
v_res_658_ = lp_mathlib_List_foldlM___at___00Lean_Meta_simpTheoremsOfNames_spec__3(v_simpOnly_boxed_657_, v_x_650_, v_x_651_, v___y_652_, v___y_653_, v___y_654_, v___y_655_);
lean_dec(v___y_655_);
lean_dec_ref(v___y_654_);
lean_dec(v___y_653_);
lean_dec_ref(v___y_652_);
return v_res_658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_Meta_simpTheoremsOfNames_spec__0(lean_object* v_x_659_, lean_object* v_x_660_, lean_object* v___y_661_, lean_object* v___y_662_, lean_object* v___y_663_, lean_object* v___y_664_){
_start:
{
if (lean_obj_tag(v_x_660_) == 0)
{
lean_object* v___x_666_; 
v___x_666_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_666_, 0, v_x_659_);
return v___x_666_;
}
else
{
lean_object* v_head_667_; lean_object* v_tail_668_; uint8_t v___x_669_; uint8_t v___x_670_; lean_object* v___x_671_; lean_object* v___x_672_; 
v_head_667_ = lean_ctor_get(v_x_660_, 0);
lean_inc(v_head_667_);
v_tail_668_ = lean_ctor_get(v_x_660_, 1);
lean_inc(v_tail_668_);
lean_dec_ref_known(v_x_660_, 2);
v___x_669_ = 1;
v___x_670_ = 0;
v___x_671_ = lean_unsigned_to_nat(1000u);
v___x_672_ = l_Lean_Meta_SimpTheorems_addConst(v_x_659_, v_head_667_, v___x_669_, v___x_670_, v___x_671_, v___y_661_, v___y_662_, v___y_663_, v___y_664_);
if (lean_obj_tag(v___x_672_) == 0)
{
lean_object* v_a_673_; 
v_a_673_ = lean_ctor_get(v___x_672_, 0);
lean_inc(v_a_673_);
lean_dec_ref_known(v___x_672_, 1);
v_x_659_ = v_a_673_;
v_x_660_ = v_tail_668_;
goto _start;
}
else
{
lean_dec(v_tail_668_);
return v___x_672_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_Meta_simpTheoremsOfNames_spec__0___boxed(lean_object* v_x_675_, lean_object* v_x_676_, lean_object* v___y_677_, lean_object* v___y_678_, lean_object* v___y_679_, lean_object* v___y_680_, lean_object* v___y_681_){
_start:
{
lean_object* v_res_682_; 
v_res_682_ = lp_mathlib_List_foldlM___at___00Lean_Meta_simpTheoremsOfNames_spec__0(v_x_675_, v_x_676_, v___y_677_, v___y_678_, v___y_679_, v___y_680_);
lean_dec(v___y_680_);
lean_dec_ref(v___y_679_);
lean_dec(v___y_678_);
lean_dec_ref(v___y_677_);
return v_res_682_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__0(void){
_start:
{
lean_object* v___x_683_; 
v___x_683_ = l_Lean_Meta_DiscrTree_empty(lean_box(0));
return v___x_683_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__1(void){
_start:
{
lean_object* v___x_684_; 
v___x_684_ = lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__1(lean_box(0));
return v___x_684_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__2(void){
_start:
{
lean_object* v___x_685_; 
v___x_685_ = lp_mathlib_Lean_PersistentHashMap_empty___at___00Lean_Meta_simpTheoremsOfNames_spec__2(lean_box(0));
return v___x_685_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__3(void){
_start:
{
lean_object* v___x_686_; 
v___x_686_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_686_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__4(void){
_start:
{
lean_object* v___x_687_; lean_object* v___x_688_; 
v___x_687_ = lean_obj_once(&lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__3, &lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__3_once, _init_lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__3);
v___x_688_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_688_, 0, v___x_687_);
return v___x_688_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__5(void){
_start:
{
lean_object* v___x_689_; lean_object* v___x_690_; lean_object* v___x_691_; lean_object* v___x_692_; lean_object* v___x_693_; 
v___x_689_ = lean_obj_once(&lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__4, &lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__4_once, _init_lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__4);
v___x_690_ = lean_obj_once(&lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__2, &lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__2_once, _init_lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__2);
v___x_691_ = lean_obj_once(&lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__1, &lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__1_once, _init_lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__1);
v___x_692_ = lean_obj_once(&lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__0, &lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__0_once, _init_lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__0);
v___x_693_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_693_, 0, v___x_692_);
lean_ctor_set(v___x_693_, 1, v___x_692_);
lean_ctor_set(v___x_693_, 2, v___x_691_);
lean_ctor_set(v___x_693_, 3, v___x_690_);
lean_ctor_set(v___x_693_, 4, v___x_691_);
lean_ctor_set(v___x_693_, 5, v___x_689_);
return v___x_693_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpTheoremsOfNames(lean_object* v_lemmas_706_, uint8_t v_simpOnly_707_, lean_object* v_a_708_, lean_object* v_a_709_, lean_object* v_a_710_, lean_object* v_a_711_){
_start:
{
if (v_simpOnly_707_ == 0)
{
lean_object* v___x_713_; 
v___x_713_ = l_Lean_Meta_getSimpTheorems___redArg(v_a_711_);
if (lean_obj_tag(v___x_713_) == 0)
{
lean_object* v_a_714_; lean_object* v___x_715_; 
v_a_714_ = lean_ctor_get(v___x_713_, 0);
lean_inc(v_a_714_);
lean_dec_ref_known(v___x_713_, 1);
v___x_715_ = lp_mathlib_List_foldlM___at___00Lean_Meta_simpTheoremsOfNames_spec__0(v_a_714_, v_lemmas_706_, v_a_708_, v_a_709_, v_a_710_, v_a_711_);
return v___x_715_;
}
else
{
lean_dec(v_lemmas_706_);
return v___x_713_;
}
}
else
{
lean_object* v___x_716_; lean_object* v___x_717_; lean_object* v___x_718_; 
v___x_716_ = lean_obj_once(&lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__5, &lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__5_once, _init_lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__5);
v___x_717_ = ((lean_object*)(lp_mathlib_Lean_Meta_simpTheoremsOfNames___closed__11));
v___x_718_ = lp_mathlib_List_foldlM___at___00Lean_Meta_simpTheoremsOfNames_spec__3(v_simpOnly_707_, v___x_716_, v___x_717_, v_a_708_, v_a_709_, v_a_710_, v_a_711_);
if (lean_obj_tag(v___x_718_) == 0)
{
lean_object* v_a_719_; lean_object* v___x_720_; 
v_a_719_ = lean_ctor_get(v___x_718_, 0);
lean_inc(v_a_719_);
lean_dec_ref_known(v___x_718_, 1);
v___x_720_ = lp_mathlib_List_foldlM___at___00Lean_Meta_simpTheoremsOfNames_spec__0(v_a_719_, v_lemmas_706_, v_a_708_, v_a_709_, v_a_710_, v_a_711_);
return v___x_720_;
}
else
{
lean_dec(v_lemmas_706_);
return v___x_718_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpTheoremsOfNames___boxed(lean_object* v_lemmas_721_, lean_object* v_simpOnly_722_, lean_object* v_a_723_, lean_object* v_a_724_, lean_object* v_a_725_, lean_object* v_a_726_, lean_object* v_a_727_){
_start:
{
uint8_t v_simpOnly_boxed_728_; lean_object* v_res_729_; 
v_simpOnly_boxed_728_ = lean_unbox(v_simpOnly_722_);
v_res_729_ = lp_mathlib_Lean_Meta_simpTheoremsOfNames(v_lemmas_721_, v_simpOnly_boxed_728_, v_a_723_, v_a_724_, v_a_725_, v_a_726_);
lean_dec(v_a_726_);
lean_dec_ref(v_a_725_);
lean_dec(v_a_724_);
lean_dec_ref(v_a_723_);
return v_res_729_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_Context_ofNames(lean_object* v_lemmas_730_, uint8_t v_simpOnly_731_, lean_object* v_config_732_, lean_object* v_a_733_, lean_object* v_a_734_, lean_object* v_a_735_, lean_object* v_a_736_){
_start:
{
lean_object* v___x_738_; 
v___x_738_ = lp_mathlib_Lean_Meta_simpTheoremsOfNames(v_lemmas_730_, v_simpOnly_731_, v_a_733_, v_a_734_, v_a_735_, v_a_736_);
if (lean_obj_tag(v___x_738_) == 0)
{
lean_object* v_a_739_; lean_object* v___x_740_; 
v_a_739_ = lean_ctor_get(v___x_738_, 0);
lean_inc(v_a_739_);
lean_dec_ref_known(v___x_738_, 1);
v___x_740_ = l_Lean_Meta_getSimpCongrTheorems___redArg(v_a_736_);
if (lean_obj_tag(v___x_740_) == 0)
{
lean_object* v_a_741_; lean_object* v___x_742_; lean_object* v___x_743_; lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_746_; 
v_a_741_ = lean_ctor_get(v___x_740_, 0);
lean_inc(v_a_741_);
lean_dec_ref_known(v___x_740_, 1);
v___x_742_ = lean_unsigned_to_nat(1u);
v___x_743_ = lean_mk_empty_array_with_capacity(v___x_742_);
v___x_744_ = lean_array_push(v___x_743_, v_a_739_);
v___x_745_ = l_Lean_Options_empty;
v___x_746_ = l_Lean_Meta_Simp_mkContext___redArg(v_config_732_, v___x_744_, v_a_741_, v___x_745_, v_a_733_, v_a_735_, v_a_736_);
return v___x_746_;
}
else
{
lean_object* v_a_747_; lean_object* v___x_749_; uint8_t v_isShared_750_; uint8_t v_isSharedCheck_754_; 
lean_dec(v_a_739_);
lean_dec_ref(v_config_732_);
v_a_747_ = lean_ctor_get(v___x_740_, 0);
v_isSharedCheck_754_ = !lean_is_exclusive(v___x_740_);
if (v_isSharedCheck_754_ == 0)
{
v___x_749_ = v___x_740_;
v_isShared_750_ = v_isSharedCheck_754_;
goto v_resetjp_748_;
}
else
{
lean_inc(v_a_747_);
lean_dec(v___x_740_);
v___x_749_ = lean_box(0);
v_isShared_750_ = v_isSharedCheck_754_;
goto v_resetjp_748_;
}
v_resetjp_748_:
{
lean_object* v___x_752_; 
if (v_isShared_750_ == 0)
{
v___x_752_ = v___x_749_;
goto v_reusejp_751_;
}
else
{
lean_object* v_reuseFailAlloc_753_; 
v_reuseFailAlloc_753_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_753_, 0, v_a_747_);
v___x_752_ = v_reuseFailAlloc_753_;
goto v_reusejp_751_;
}
v_reusejp_751_:
{
return v___x_752_;
}
}
}
}
else
{
lean_object* v_a_755_; lean_object* v___x_757_; uint8_t v_isShared_758_; uint8_t v_isSharedCheck_762_; 
lean_dec_ref(v_config_732_);
v_a_755_ = lean_ctor_get(v___x_738_, 0);
v_isSharedCheck_762_ = !lean_is_exclusive(v___x_738_);
if (v_isSharedCheck_762_ == 0)
{
v___x_757_ = v___x_738_;
v_isShared_758_ = v_isSharedCheck_762_;
goto v_resetjp_756_;
}
else
{
lean_inc(v_a_755_);
lean_dec(v___x_738_);
v___x_757_ = lean_box(0);
v_isShared_758_ = v_isSharedCheck_762_;
goto v_resetjp_756_;
}
v_resetjp_756_:
{
lean_object* v___x_760_; 
if (v_isShared_758_ == 0)
{
v___x_760_ = v___x_757_;
goto v_reusejp_759_;
}
else
{
lean_object* v_reuseFailAlloc_761_; 
v_reuseFailAlloc_761_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_761_, 0, v_a_755_);
v___x_760_ = v_reuseFailAlloc_761_;
goto v_reusejp_759_;
}
v_reusejp_759_:
{
return v___x_760_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_Context_ofNames___boxed(lean_object* v_lemmas_763_, lean_object* v_simpOnly_764_, lean_object* v_config_765_, lean_object* v_a_766_, lean_object* v_a_767_, lean_object* v_a_768_, lean_object* v_a_769_, lean_object* v_a_770_){
_start:
{
uint8_t v_simpOnly_boxed_771_; lean_object* v_res_772_; 
v_simpOnly_boxed_771_ = lean_unbox(v_simpOnly_764_);
v_res_772_ = lp_mathlib_Lean_Meta_Simp_Context_ofNames(v_lemmas_763_, v_simpOnly_boxed_771_, v_config_765_, v_a_766_, v_a_767_, v_a_768_, v_a_769_);
lean_dec(v_a_769_);
lean_dec_ref(v_a_768_);
lean_dec(v_a_767_);
lean_dec_ref(v_a_766_);
return v_res_772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_Context_ofArgs(lean_object* v_args_775_, lean_object* v_config_776_, lean_object* v_a_777_, lean_object* v_a_778_, lean_object* v_a_779_, lean_object* v_a_780_, lean_object* v_a_781_, lean_object* v_a_782_, lean_object* v_a_783_, lean_object* v_a_784_){
_start:
{
lean_object* v___x_786_; 
v___x_786_ = l_Lean_Meta_getSimpTheorems___redArg(v_a_784_);
if (lean_obj_tag(v___x_786_) == 0)
{
lean_object* v_a_787_; lean_object* v___x_788_; 
v_a_787_ = lean_ctor_get(v___x_786_, 0);
lean_inc(v_a_787_);
lean_dec_ref_known(v___x_786_, 1);
v___x_788_ = l_Lean_Meta_getSimpCongrTheorems___redArg(v_a_784_);
if (lean_obj_tag(v___x_788_) == 0)
{
lean_object* v_a_789_; lean_object* v___x_790_; lean_object* v___x_791_; lean_object* v___x_792_; lean_object* v___x_793_; lean_object* v___x_794_; 
v_a_789_ = lean_ctor_get(v___x_788_, 0);
lean_inc(v_a_789_);
lean_dec_ref_known(v___x_788_, 1);
v___x_790_ = lean_unsigned_to_nat(1u);
v___x_791_ = lean_mk_empty_array_with_capacity(v___x_790_);
v___x_792_ = lean_array_push(v___x_791_, v_a_787_);
v___x_793_ = l_Lean_Options_empty;
v___x_794_ = l_Lean_Meta_Simp_mkContext___redArg(v_config_776_, v___x_792_, v_a_789_, v___x_793_, v_a_781_, v_a_783_, v_a_784_);
if (lean_obj_tag(v___x_794_) == 0)
{
lean_object* v_a_795_; lean_object* v___x_796_; uint8_t v___x_797_; uint8_t v___x_798_; lean_object* v___x_799_; 
v_a_795_ = lean_ctor_get(v___x_794_, 0);
lean_inc(v_a_795_);
lean_dec_ref_known(v___x_794_, 1);
v___x_796_ = ((lean_object*)(lp_mathlib_Lean_Meta_Simp_Context_ofArgs___closed__0));
v___x_797_ = 0;
v___x_798_ = 0;
v___x_799_ = l_Lean_Elab_Tactic_elabSimpArgs(v_args_775_, v_a_795_, v___x_796_, v___x_797_, v___x_798_, v___x_797_, v_a_777_, v_a_778_, v_a_779_, v_a_780_, v_a_781_, v_a_782_, v_a_783_, v_a_784_);
if (lean_obj_tag(v___x_799_) == 0)
{
lean_object* v_a_800_; lean_object* v___x_802_; uint8_t v_isShared_803_; uint8_t v_isSharedCheck_808_; 
v_a_800_ = lean_ctor_get(v___x_799_, 0);
v_isSharedCheck_808_ = !lean_is_exclusive(v___x_799_);
if (v_isSharedCheck_808_ == 0)
{
v___x_802_ = v___x_799_;
v_isShared_803_ = v_isSharedCheck_808_;
goto v_resetjp_801_;
}
else
{
lean_inc(v_a_800_);
lean_dec(v___x_799_);
v___x_802_ = lean_box(0);
v_isShared_803_ = v_isSharedCheck_808_;
goto v_resetjp_801_;
}
v_resetjp_801_:
{
lean_object* v_ctx_804_; lean_object* v___x_806_; 
v_ctx_804_ = lean_ctor_get(v_a_800_, 0);
lean_inc_ref(v_ctx_804_);
lean_dec(v_a_800_);
if (v_isShared_803_ == 0)
{
lean_ctor_set(v___x_802_, 0, v_ctx_804_);
v___x_806_ = v___x_802_;
goto v_reusejp_805_;
}
else
{
lean_object* v_reuseFailAlloc_807_; 
v_reuseFailAlloc_807_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_807_, 0, v_ctx_804_);
v___x_806_ = v_reuseFailAlloc_807_;
goto v_reusejp_805_;
}
v_reusejp_805_:
{
return v___x_806_;
}
}
}
else
{
lean_object* v_a_809_; lean_object* v___x_811_; uint8_t v_isShared_812_; uint8_t v_isSharedCheck_816_; 
v_a_809_ = lean_ctor_get(v___x_799_, 0);
v_isSharedCheck_816_ = !lean_is_exclusive(v___x_799_);
if (v_isSharedCheck_816_ == 0)
{
v___x_811_ = v___x_799_;
v_isShared_812_ = v_isSharedCheck_816_;
goto v_resetjp_810_;
}
else
{
lean_inc(v_a_809_);
lean_dec(v___x_799_);
v___x_811_ = lean_box(0);
v_isShared_812_ = v_isSharedCheck_816_;
goto v_resetjp_810_;
}
v_resetjp_810_:
{
lean_object* v___x_814_; 
if (v_isShared_812_ == 0)
{
v___x_814_ = v___x_811_;
goto v_reusejp_813_;
}
else
{
lean_object* v_reuseFailAlloc_815_; 
v_reuseFailAlloc_815_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_815_, 0, v_a_809_);
v___x_814_ = v_reuseFailAlloc_815_;
goto v_reusejp_813_;
}
v_reusejp_813_:
{
return v___x_814_;
}
}
}
}
else
{
lean_dec(v_args_775_);
return v___x_794_;
}
}
else
{
lean_object* v_a_817_; lean_object* v___x_819_; uint8_t v_isShared_820_; uint8_t v_isSharedCheck_824_; 
lean_dec(v_a_787_);
lean_dec_ref(v_config_776_);
lean_dec(v_args_775_);
v_a_817_ = lean_ctor_get(v___x_788_, 0);
v_isSharedCheck_824_ = !lean_is_exclusive(v___x_788_);
if (v_isSharedCheck_824_ == 0)
{
v___x_819_ = v___x_788_;
v_isShared_820_ = v_isSharedCheck_824_;
goto v_resetjp_818_;
}
else
{
lean_inc(v_a_817_);
lean_dec(v___x_788_);
v___x_819_ = lean_box(0);
v_isShared_820_ = v_isSharedCheck_824_;
goto v_resetjp_818_;
}
v_resetjp_818_:
{
lean_object* v___x_822_; 
if (v_isShared_820_ == 0)
{
v___x_822_ = v___x_819_;
goto v_reusejp_821_;
}
else
{
lean_object* v_reuseFailAlloc_823_; 
v_reuseFailAlloc_823_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_823_, 0, v_a_817_);
v___x_822_ = v_reuseFailAlloc_823_;
goto v_reusejp_821_;
}
v_reusejp_821_:
{
return v___x_822_;
}
}
}
}
else
{
lean_object* v_a_825_; lean_object* v___x_827_; uint8_t v_isShared_828_; uint8_t v_isSharedCheck_832_; 
lean_dec_ref(v_config_776_);
lean_dec(v_args_775_);
v_a_825_ = lean_ctor_get(v___x_786_, 0);
v_isSharedCheck_832_ = !lean_is_exclusive(v___x_786_);
if (v_isSharedCheck_832_ == 0)
{
v___x_827_ = v___x_786_;
v_isShared_828_ = v_isSharedCheck_832_;
goto v_resetjp_826_;
}
else
{
lean_inc(v_a_825_);
lean_dec(v___x_786_);
v___x_827_ = lean_box(0);
v_isShared_828_ = v_isSharedCheck_832_;
goto v_resetjp_826_;
}
v_resetjp_826_:
{
lean_object* v___x_830_; 
if (v_isShared_828_ == 0)
{
v___x_830_ = v___x_827_;
goto v_reusejp_829_;
}
else
{
lean_object* v_reuseFailAlloc_831_; 
v_reuseFailAlloc_831_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_831_, 0, v_a_825_);
v___x_830_ = v_reuseFailAlloc_831_;
goto v_reusejp_829_;
}
v_reusejp_829_:
{
return v___x_830_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_Context_ofArgs___boxed(lean_object* v_args_833_, lean_object* v_config_834_, lean_object* v_a_835_, lean_object* v_a_836_, lean_object* v_a_837_, lean_object* v_a_838_, lean_object* v_a_839_, lean_object* v_a_840_, lean_object* v_a_841_, lean_object* v_a_842_, lean_object* v_a_843_){
_start:
{
lean_object* v_res_844_; 
v_res_844_ = lp_mathlib_Lean_Meta_Simp_Context_ofArgs(v_args_833_, v_config_834_, v_a_835_, v_a_836_, v_a_837_, v_a_838_, v_a_839_, v_a_840_, v_a_841_, v_a_842_);
lean_dec(v_a_842_);
lean_dec_ref(v_a_841_);
lean_dec(v_a_840_);
lean_dec_ref(v_a_839_);
lean_dec(v_a_838_);
lean_dec_ref(v_a_837_);
lean_dec(v_a_836_);
lean_dec_ref(v_a_835_);
return v_res_844_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_simpOnlyNames___closed__1(void){
_start:
{
lean_object* v___x_847_; 
v___x_847_ = l_Lean_PersistentHashMap_mkEmptyEntriesArray(lean_box(0), lean_box(0));
return v___x_847_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_simpOnlyNames___closed__2(void){
_start:
{
lean_object* v___x_848_; lean_object* v___x_849_; 
v___x_848_ = lean_obj_once(&lp_mathlib_Lean_Meta_simpOnlyNames___closed__1, &lp_mathlib_Lean_Meta_simpOnlyNames___closed__1_once, _init_lp_mathlib_Lean_Meta_simpOnlyNames___closed__1);
v___x_849_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_849_, 0, v___x_848_);
return v___x_849_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_simpOnlyNames___closed__3(void){
_start:
{
lean_object* v___x_850_; lean_object* v___x_851_; lean_object* v___x_852_; 
v___x_850_ = lean_unsigned_to_nat(0u);
v___x_851_ = lean_obj_once(&lp_mathlib_Lean_Meta_simpOnlyNames___closed__2, &lp_mathlib_Lean_Meta_simpOnlyNames___closed__2_once, _init_lp_mathlib_Lean_Meta_simpOnlyNames___closed__2);
v___x_852_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_852_, 0, v___x_851_);
lean_ctor_set(v___x_852_, 1, v___x_850_);
return v___x_852_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_simpOnlyNames___closed__4(void){
_start:
{
lean_object* v___x_853_; lean_object* v___x_854_; lean_object* v___x_855_; 
v___x_853_ = lean_unsigned_to_nat(32u);
v___x_854_ = lean_mk_empty_array_with_capacity(v___x_853_);
v___x_855_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_855_, 0, v___x_854_);
return v___x_855_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_simpOnlyNames___closed__5(void){
_start:
{
size_t v___x_856_; lean_object* v___x_857_; lean_object* v___x_858_; lean_object* v___x_859_; lean_object* v___x_860_; lean_object* v___x_861_; 
v___x_856_ = ((size_t)5ULL);
v___x_857_ = lean_unsigned_to_nat(0u);
v___x_858_ = lean_unsigned_to_nat(32u);
v___x_859_ = lean_mk_empty_array_with_capacity(v___x_858_);
v___x_860_ = lean_obj_once(&lp_mathlib_Lean_Meta_simpOnlyNames___closed__4, &lp_mathlib_Lean_Meta_simpOnlyNames___closed__4_once, _init_lp_mathlib_Lean_Meta_simpOnlyNames___closed__4);
v___x_861_ = lean_alloc_ctor(0, 4, sizeof(size_t)*1);
lean_ctor_set(v___x_861_, 0, v___x_860_);
lean_ctor_set(v___x_861_, 1, v___x_859_);
lean_ctor_set(v___x_861_, 2, v___x_857_);
lean_ctor_set(v___x_861_, 3, v___x_857_);
lean_ctor_set_usize(v___x_861_, 4, v___x_856_);
return v___x_861_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_simpOnlyNames___closed__6(void){
_start:
{
lean_object* v___x_862_; lean_object* v___x_863_; lean_object* v___x_864_; 
v___x_862_ = lean_obj_once(&lp_mathlib_Lean_Meta_simpOnlyNames___closed__5, &lp_mathlib_Lean_Meta_simpOnlyNames___closed__5_once, _init_lp_mathlib_Lean_Meta_simpOnlyNames___closed__5);
v___x_863_ = lean_obj_once(&lp_mathlib_Lean_Meta_simpOnlyNames___closed__2, &lp_mathlib_Lean_Meta_simpOnlyNames___closed__2_once, _init_lp_mathlib_Lean_Meta_simpOnlyNames___closed__2);
v___x_864_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_864_, 0, v___x_863_);
lean_ctor_set(v___x_864_, 1, v___x_863_);
lean_ctor_set(v___x_864_, 2, v___x_863_);
lean_ctor_set(v___x_864_, 3, v___x_862_);
return v___x_864_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_simpOnlyNames___closed__7(void){
_start:
{
lean_object* v___x_865_; lean_object* v___x_866_; lean_object* v___x_867_; 
v___x_865_ = lean_obj_once(&lp_mathlib_Lean_Meta_simpOnlyNames___closed__6, &lp_mathlib_Lean_Meta_simpOnlyNames___closed__6_once, _init_lp_mathlib_Lean_Meta_simpOnlyNames___closed__6);
v___x_866_ = lean_obj_once(&lp_mathlib_Lean_Meta_simpOnlyNames___closed__3, &lp_mathlib_Lean_Meta_simpOnlyNames___closed__3_once, _init_lp_mathlib_Lean_Meta_simpOnlyNames___closed__3);
v___x_867_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_867_, 0, v___x_866_);
lean_ctor_set(v___x_867_, 1, v___x_865_);
return v___x_867_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpOnlyNames(lean_object* v_lemmas_868_, lean_object* v_e_869_, lean_object* v_config_870_, lean_object* v_a_871_, lean_object* v_a_872_, lean_object* v_a_873_, lean_object* v_a_874_){
_start:
{
uint8_t v___x_876_; lean_object* v___x_877_; 
v___x_876_ = 1;
v___x_877_ = lp_mathlib_Lean_Meta_Simp_Context_ofNames(v_lemmas_868_, v___x_876_, v_config_870_, v_a_871_, v_a_872_, v_a_873_, v_a_874_);
if (lean_obj_tag(v___x_877_) == 0)
{
lean_object* v_a_878_; lean_object* v___x_879_; lean_object* v___x_880_; lean_object* v___x_881_; lean_object* v___x_882_; 
v_a_878_ = lean_ctor_get(v___x_877_, 0);
lean_inc(v_a_878_);
lean_dec_ref_known(v___x_877_, 1);
v___x_879_ = ((lean_object*)(lp_mathlib_Lean_Meta_simpOnlyNames___closed__0));
v___x_880_ = lean_box(0);
v___x_881_ = lean_obj_once(&lp_mathlib_Lean_Meta_simpOnlyNames___closed__7, &lp_mathlib_Lean_Meta_simpOnlyNames___closed__7_once, _init_lp_mathlib_Lean_Meta_simpOnlyNames___closed__7);
v___x_882_ = l_Lean_Meta_simp(v_e_869_, v_a_878_, v___x_879_, v___x_880_, v___x_881_, v_a_871_, v_a_872_, v_a_873_, v_a_874_);
if (lean_obj_tag(v___x_882_) == 0)
{
lean_object* v_a_883_; lean_object* v___x_885_; uint8_t v_isShared_886_; uint8_t v_isSharedCheck_891_; 
v_a_883_ = lean_ctor_get(v___x_882_, 0);
v_isSharedCheck_891_ = !lean_is_exclusive(v___x_882_);
if (v_isSharedCheck_891_ == 0)
{
v___x_885_ = v___x_882_;
v_isShared_886_ = v_isSharedCheck_891_;
goto v_resetjp_884_;
}
else
{
lean_inc(v_a_883_);
lean_dec(v___x_882_);
v___x_885_ = lean_box(0);
v_isShared_886_ = v_isSharedCheck_891_;
goto v_resetjp_884_;
}
v_resetjp_884_:
{
lean_object* v_fst_887_; lean_object* v___x_889_; 
v_fst_887_ = lean_ctor_get(v_a_883_, 0);
lean_inc(v_fst_887_);
lean_dec(v_a_883_);
if (v_isShared_886_ == 0)
{
lean_ctor_set(v___x_885_, 0, v_fst_887_);
v___x_889_ = v___x_885_;
goto v_reusejp_888_;
}
else
{
lean_object* v_reuseFailAlloc_890_; 
v_reuseFailAlloc_890_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_890_, 0, v_fst_887_);
v___x_889_ = v_reuseFailAlloc_890_;
goto v_reusejp_888_;
}
v_reusejp_888_:
{
return v___x_889_;
}
}
}
else
{
lean_object* v_a_892_; lean_object* v___x_894_; uint8_t v_isShared_895_; uint8_t v_isSharedCheck_899_; 
v_a_892_ = lean_ctor_get(v___x_882_, 0);
v_isSharedCheck_899_ = !lean_is_exclusive(v___x_882_);
if (v_isSharedCheck_899_ == 0)
{
v___x_894_ = v___x_882_;
v_isShared_895_ = v_isSharedCheck_899_;
goto v_resetjp_893_;
}
else
{
lean_inc(v_a_892_);
lean_dec(v___x_882_);
v___x_894_ = lean_box(0);
v_isShared_895_ = v_isSharedCheck_899_;
goto v_resetjp_893_;
}
v_resetjp_893_:
{
lean_object* v___x_897_; 
if (v_isShared_895_ == 0)
{
v___x_897_ = v___x_894_;
goto v_reusejp_896_;
}
else
{
lean_object* v_reuseFailAlloc_898_; 
v_reuseFailAlloc_898_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_898_, 0, v_a_892_);
v___x_897_ = v_reuseFailAlloc_898_;
goto v_reusejp_896_;
}
v_reusejp_896_:
{
return v___x_897_;
}
}
}
}
else
{
lean_object* v_a_900_; lean_object* v___x_902_; uint8_t v_isShared_903_; uint8_t v_isSharedCheck_907_; 
lean_dec_ref(v_e_869_);
v_a_900_ = lean_ctor_get(v___x_877_, 0);
v_isSharedCheck_907_ = !lean_is_exclusive(v___x_877_);
if (v_isSharedCheck_907_ == 0)
{
v___x_902_ = v___x_877_;
v_isShared_903_ = v_isSharedCheck_907_;
goto v_resetjp_901_;
}
else
{
lean_inc(v_a_900_);
lean_dec(v___x_877_);
v___x_902_ = lean_box(0);
v_isShared_903_ = v_isSharedCheck_907_;
goto v_resetjp_901_;
}
v_resetjp_901_:
{
lean_object* v___x_905_; 
if (v_isShared_903_ == 0)
{
v___x_905_ = v___x_902_;
goto v_reusejp_904_;
}
else
{
lean_object* v_reuseFailAlloc_906_; 
v_reuseFailAlloc_906_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_906_, 0, v_a_900_);
v___x_905_ = v_reuseFailAlloc_906_;
goto v_reusejp_904_;
}
v_reusejp_904_:
{
return v___x_905_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpOnlyNames___boxed(lean_object* v_lemmas_908_, lean_object* v_e_909_, lean_object* v_config_910_, lean_object* v_a_911_, lean_object* v_a_912_, lean_object* v_a_913_, lean_object* v_a_914_, lean_object* v_a_915_){
_start:
{
lean_object* v_res_916_; 
v_res_916_ = lp_mathlib_Lean_Meta_simpOnlyNames(v_lemmas_908_, v_e_909_, v_config_910_, v_a_911_, v_a_912_, v_a_913_, v_a_914_);
lean_dec(v_a_914_);
lean_dec_ref(v_a_913_);
lean_dec(v_a_912_);
lean_dec_ref(v_a_911_);
return v_res_916_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpType(lean_object* v_S_917_, lean_object* v_e_918_, lean_object* v_type_x3f_919_, lean_object* v_a_920_, lean_object* v_a_921_, lean_object* v_a_922_, lean_object* v_a_923_){
_start:
{
lean_object* v_a_926_; 
if (lean_obj_tag(v_type_x3f_919_) == 0)
{
lean_object* v___x_945_; 
lean_inc(v_a_923_);
lean_inc_ref(v_a_922_);
lean_inc(v_a_921_);
lean_inc_ref(v_a_920_);
lean_inc_ref(v_e_918_);
v___x_945_ = lean_infer_type(v_e_918_, v_a_920_, v_a_921_, v_a_922_, v_a_923_);
if (lean_obj_tag(v___x_945_) == 0)
{
lean_object* v_a_946_; 
v_a_946_ = lean_ctor_get(v___x_945_, 0);
lean_inc(v_a_946_);
lean_dec_ref_known(v___x_945_, 1);
v_a_926_ = v_a_946_;
goto v___jp_925_;
}
else
{
lean_dec_ref(v_e_918_);
lean_dec_ref(v_S_917_);
return v___x_945_;
}
}
else
{
lean_object* v_val_947_; 
v_val_947_ = lean_ctor_get(v_type_x3f_919_, 0);
lean_inc(v_val_947_);
lean_dec_ref_known(v_type_x3f_919_, 1);
v_a_926_ = v_val_947_;
goto v___jp_925_;
}
v___jp_925_:
{
lean_object* v___x_927_; 
lean_inc(v_a_923_);
lean_inc_ref(v_a_922_);
lean_inc(v_a_921_);
lean_inc_ref(v_a_920_);
v___x_927_ = lean_apply_6(v_S_917_, v_a_926_, v_a_920_, v_a_921_, v_a_922_, v_a_923_, lean_box(0));
if (lean_obj_tag(v___x_927_) == 0)
{
lean_object* v_a_928_; lean_object* v_proof_x3f_929_; 
v_a_928_ = lean_ctor_get(v___x_927_, 0);
lean_inc(v_a_928_);
lean_dec_ref_known(v___x_927_, 1);
v_proof_x3f_929_ = lean_ctor_get(v_a_928_, 1);
if (lean_obj_tag(v_proof_x3f_929_) == 0)
{
lean_object* v_expr_930_; lean_object* v___x_931_; 
v_expr_930_ = lean_ctor_get(v_a_928_, 0);
lean_inc_ref(v_expr_930_);
lean_dec(v_a_928_);
v___x_931_ = l_Lean_Meta_mkExpectedTypeHint(v_e_918_, v_expr_930_, v_a_920_, v_a_921_, v_a_922_, v_a_923_);
return v___x_931_;
}
else
{
lean_object* v_expr_932_; lean_object* v_val_933_; lean_object* v___x_934_; 
lean_inc_ref(v_proof_x3f_929_);
v_expr_932_ = lean_ctor_get(v_a_928_, 0);
lean_inc_ref(v_expr_932_);
lean_dec(v_a_928_);
v_val_933_ = lean_ctor_get(v_proof_x3f_929_, 0);
lean_inc(v_val_933_);
lean_dec_ref_known(v_proof_x3f_929_, 1);
v___x_934_ = l_Lean_Meta_mkEqMP(v_val_933_, v_e_918_, v_a_920_, v_a_921_, v_a_922_, v_a_923_);
if (lean_obj_tag(v___x_934_) == 0)
{
lean_object* v_a_935_; lean_object* v___x_936_; 
v_a_935_ = lean_ctor_get(v___x_934_, 0);
lean_inc(v_a_935_);
lean_dec_ref_known(v___x_934_, 1);
v___x_936_ = l_Lean_Meta_mkExpectedTypeHint(v_a_935_, v_expr_932_, v_a_920_, v_a_921_, v_a_922_, v_a_923_);
return v___x_936_;
}
else
{
lean_dec_ref(v_expr_932_);
return v___x_934_;
}
}
}
else
{
lean_object* v_a_937_; lean_object* v___x_939_; uint8_t v_isShared_940_; uint8_t v_isSharedCheck_944_; 
lean_dec_ref(v_e_918_);
v_a_937_ = lean_ctor_get(v___x_927_, 0);
v_isSharedCheck_944_ = !lean_is_exclusive(v___x_927_);
if (v_isSharedCheck_944_ == 0)
{
v___x_939_ = v___x_927_;
v_isShared_940_ = v_isSharedCheck_944_;
goto v_resetjp_938_;
}
else
{
lean_inc(v_a_937_);
lean_dec(v___x_927_);
v___x_939_ = lean_box(0);
v_isShared_940_ = v_isSharedCheck_944_;
goto v_resetjp_938_;
}
v_resetjp_938_:
{
lean_object* v___x_942_; 
if (v_isShared_940_ == 0)
{
v___x_942_ = v___x_939_;
goto v_reusejp_941_;
}
else
{
lean_object* v_reuseFailAlloc_943_; 
v_reuseFailAlloc_943_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_943_, 0, v_a_937_);
v___x_942_ = v_reuseFailAlloc_943_;
goto v_reusejp_941_;
}
v_reusejp_941_:
{
return v___x_942_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpType___boxed(lean_object* v_S_948_, lean_object* v_e_949_, lean_object* v_type_x3f_950_, lean_object* v_a_951_, lean_object* v_a_952_, lean_object* v_a_953_, lean_object* v_a_954_, lean_object* v_a_955_){
_start:
{
lean_object* v_res_956_; 
v_res_956_ = lp_mathlib_Lean_Meta_simpType(v_S_948_, v_e_949_, v_type_x3f_950_, v_a_951_, v_a_952_, v_a_953_, v_a_954_);
lean_dec(v_a_954_);
lean_dec_ref(v_a_953_);
lean_dec(v_a_952_);
lean_dec_ref(v_a_951_);
return v_res_956_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1___redArg___lam__0(lean_object* v_k_957_, lean_object* v_b_958_, lean_object* v_c_959_, lean_object* v___y_960_, lean_object* v___y_961_, lean_object* v___y_962_, lean_object* v___y_963_){
_start:
{
lean_object* v___x_965_; 
lean_inc(v___y_963_);
lean_inc_ref(v___y_962_);
lean_inc(v___y_961_);
lean_inc_ref(v___y_960_);
v___x_965_ = lean_apply_7(v_k_957_, v_b_958_, v_c_959_, v___y_960_, v___y_961_, v___y_962_, v___y_963_, lean_box(0));
return v___x_965_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1___redArg___lam__0___boxed(lean_object* v_k_966_, lean_object* v_b_967_, lean_object* v_c_968_, lean_object* v___y_969_, lean_object* v___y_970_, lean_object* v___y_971_, lean_object* v___y_972_, lean_object* v___y_973_){
_start:
{
lean_object* v_res_974_; 
v_res_974_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1___redArg___lam__0(v_k_966_, v_b_967_, v_c_968_, v___y_969_, v___y_970_, v___y_971_, v___y_972_);
lean_dec(v___y_972_);
lean_dec_ref(v___y_971_);
lean_dec(v___y_970_);
lean_dec_ref(v___y_969_);
return v_res_974_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1___redArg(lean_object* v_type_975_, lean_object* v_k_976_, uint8_t v_cleanupAnnotations_977_, lean_object* v___y_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_){
_start:
{
lean_object* v___f_983_; uint8_t v___x_984_; lean_object* v___x_985_; lean_object* v___x_986_; 
v___f_983_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1___redArg___lam__0___boxed), 8, 1);
lean_closure_set(v___f_983_, 0, v_k_976_);
v___x_984_ = 0;
v___x_985_ = lean_box(0);
v___x_986_ = l___private_Lean_Meta_Basic_0__Lean_Meta_forallTelescopeReducingAuxAux(lean_box(0), v___x_984_, v___x_985_, v_type_975_, v___f_983_, v_cleanupAnnotations_977_, v___x_984_, v___y_978_, v___y_979_, v___y_980_, v___y_981_);
if (lean_obj_tag(v___x_986_) == 0)
{
lean_object* v_a_987_; lean_object* v___x_989_; uint8_t v_isShared_990_; uint8_t v_isSharedCheck_994_; 
v_a_987_ = lean_ctor_get(v___x_986_, 0);
v_isSharedCheck_994_ = !lean_is_exclusive(v___x_986_);
if (v_isSharedCheck_994_ == 0)
{
v___x_989_ = v___x_986_;
v_isShared_990_ = v_isSharedCheck_994_;
goto v_resetjp_988_;
}
else
{
lean_inc(v_a_987_);
lean_dec(v___x_986_);
v___x_989_ = lean_box(0);
v_isShared_990_ = v_isSharedCheck_994_;
goto v_resetjp_988_;
}
v_resetjp_988_:
{
lean_object* v___x_992_; 
if (v_isShared_990_ == 0)
{
v___x_992_ = v___x_989_;
goto v_reusejp_991_;
}
else
{
lean_object* v_reuseFailAlloc_993_; 
v_reuseFailAlloc_993_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_993_, 0, v_a_987_);
v___x_992_ = v_reuseFailAlloc_993_;
goto v_reusejp_991_;
}
v_reusejp_991_:
{
return v___x_992_;
}
}
}
else
{
lean_object* v_a_995_; lean_object* v___x_997_; uint8_t v_isShared_998_; uint8_t v_isSharedCheck_1002_; 
v_a_995_ = lean_ctor_get(v___x_986_, 0);
v_isSharedCheck_1002_ = !lean_is_exclusive(v___x_986_);
if (v_isSharedCheck_1002_ == 0)
{
v___x_997_ = v___x_986_;
v_isShared_998_ = v_isSharedCheck_1002_;
goto v_resetjp_996_;
}
else
{
lean_inc(v_a_995_);
lean_dec(v___x_986_);
v___x_997_ = lean_box(0);
v_isShared_998_ = v_isSharedCheck_1002_;
goto v_resetjp_996_;
}
v_resetjp_996_:
{
lean_object* v___x_1000_; 
if (v_isShared_998_ == 0)
{
v___x_1000_ = v___x_997_;
goto v_reusejp_999_;
}
else
{
lean_object* v_reuseFailAlloc_1001_; 
v_reuseFailAlloc_1001_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1001_, 0, v_a_995_);
v___x_1000_ = v_reuseFailAlloc_1001_;
goto v_reusejp_999_;
}
v_reusejp_999_:
{
return v___x_1000_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1___redArg___boxed(lean_object* v_type_1003_, lean_object* v_k_1004_, lean_object* v_cleanupAnnotations_1005_, lean_object* v___y_1006_, lean_object* v___y_1007_, lean_object* v___y_1008_, lean_object* v___y_1009_, lean_object* v___y_1010_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_1011_; lean_object* v_res_1012_; 
v_cleanupAnnotations_boxed_1011_ = lean_unbox(v_cleanupAnnotations_1005_);
v_res_1012_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1___redArg(v_type_1003_, v_k_1004_, v_cleanupAnnotations_boxed_1011_, v___y_1006_, v___y_1007_, v___y_1008_, v___y_1009_);
lean_dec(v___y_1009_);
lean_dec_ref(v___y_1008_);
lean_dec(v___y_1007_);
lean_dec_ref(v___y_1006_);
return v_res_1012_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1(lean_object* v_00_u03b1_1013_, lean_object* v_type_1014_, lean_object* v_k_1015_, uint8_t v_cleanupAnnotations_1016_, lean_object* v___y_1017_, lean_object* v___y_1018_, lean_object* v___y_1019_, lean_object* v___y_1020_){
_start:
{
lean_object* v___x_1022_; 
v___x_1022_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1___redArg(v_type_1014_, v_k_1015_, v_cleanupAnnotations_1016_, v___y_1017_, v___y_1018_, v___y_1019_, v___y_1020_);
return v___x_1022_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1___boxed(lean_object* v_00_u03b1_1023_, lean_object* v_type_1024_, lean_object* v_k_1025_, lean_object* v_cleanupAnnotations_1026_, lean_object* v___y_1027_, lean_object* v___y_1028_, lean_object* v___y_1029_, lean_object* v___y_1030_, lean_object* v___y_1031_){
_start:
{
uint8_t v_cleanupAnnotations_boxed_1032_; lean_object* v_res_1033_; 
v_cleanupAnnotations_boxed_1032_ = lean_unbox(v_cleanupAnnotations_1026_);
v_res_1033_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1(v_00_u03b1_1023_, v_type_1024_, v_k_1025_, v_cleanupAnnotations_boxed_1032_, v___y_1027_, v___y_1028_, v___y_1029_, v___y_1030_);
lean_dec(v___y_1030_);
lean_dec_ref(v___y_1029_);
lean_dec(v___y_1028_);
lean_dec_ref(v___y_1027_);
return v_res_1033_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_simpEq_spec__0_spec__0(lean_object* v_msgData_1034_, lean_object* v___y_1035_, lean_object* v___y_1036_, lean_object* v___y_1037_, lean_object* v___y_1038_){
_start:
{
lean_object* v___x_1040_; lean_object* v_env_1041_; lean_object* v___x_1042_; lean_object* v_mctx_1043_; lean_object* v_lctx_1044_; lean_object* v_options_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; 
v___x_1040_ = lean_st_ref_get(v___y_1038_);
v_env_1041_ = lean_ctor_get(v___x_1040_, 0);
lean_inc_ref(v_env_1041_);
lean_dec(v___x_1040_);
v___x_1042_ = lean_st_ref_get(v___y_1036_);
v_mctx_1043_ = lean_ctor_get(v___x_1042_, 0);
lean_inc_ref(v_mctx_1043_);
lean_dec(v___x_1042_);
v_lctx_1044_ = lean_ctor_get(v___y_1035_, 2);
v_options_1045_ = lean_ctor_get(v___y_1037_, 2);
lean_inc_ref(v_options_1045_);
lean_inc_ref(v_lctx_1044_);
v___x_1046_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1046_, 0, v_env_1041_);
lean_ctor_set(v___x_1046_, 1, v_mctx_1043_);
lean_ctor_set(v___x_1046_, 2, v_lctx_1044_);
lean_ctor_set(v___x_1046_, 3, v_options_1045_);
v___x_1047_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_1047_, 0, v___x_1046_);
lean_ctor_set(v___x_1047_, 1, v_msgData_1034_);
v___x_1048_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1048_, 0, v___x_1047_);
return v___x_1048_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_simpEq_spec__0_spec__0___boxed(lean_object* v_msgData_1049_, lean_object* v___y_1050_, lean_object* v___y_1051_, lean_object* v___y_1052_, lean_object* v___y_1053_, lean_object* v___y_1054_){
_start:
{
lean_object* v_res_1055_; 
v_res_1055_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_simpEq_spec__0_spec__0(v_msgData_1049_, v___y_1050_, v___y_1051_, v___y_1052_, v___y_1053_);
lean_dec(v___y_1053_);
lean_dec_ref(v___y_1052_);
lean_dec(v___y_1051_);
lean_dec_ref(v___y_1050_);
return v_res_1055_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_simpEq_spec__0___redArg(lean_object* v_msg_1056_, lean_object* v___y_1057_, lean_object* v___y_1058_, lean_object* v___y_1059_, lean_object* v___y_1060_){
_start:
{
lean_object* v_ref_1062_; lean_object* v___x_1063_; lean_object* v_a_1064_; lean_object* v___x_1066_; uint8_t v_isShared_1067_; uint8_t v_isSharedCheck_1072_; 
v_ref_1062_ = lean_ctor_get(v___y_1059_, 5);
v___x_1063_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_Meta_simpEq_spec__0_spec__0(v_msg_1056_, v___y_1057_, v___y_1058_, v___y_1059_, v___y_1060_);
v_a_1064_ = lean_ctor_get(v___x_1063_, 0);
v_isSharedCheck_1072_ = !lean_is_exclusive(v___x_1063_);
if (v_isSharedCheck_1072_ == 0)
{
v___x_1066_ = v___x_1063_;
v_isShared_1067_ = v_isSharedCheck_1072_;
goto v_resetjp_1065_;
}
else
{
lean_inc(v_a_1064_);
lean_dec(v___x_1063_);
v___x_1066_ = lean_box(0);
v_isShared_1067_ = v_isSharedCheck_1072_;
goto v_resetjp_1065_;
}
v_resetjp_1065_:
{
lean_object* v___x_1068_; lean_object* v___x_1070_; 
lean_inc(v_ref_1062_);
v___x_1068_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1068_, 0, v_ref_1062_);
lean_ctor_set(v___x_1068_, 1, v_a_1064_);
if (v_isShared_1067_ == 0)
{
lean_ctor_set_tag(v___x_1066_, 1);
lean_ctor_set(v___x_1066_, 0, v___x_1068_);
v___x_1070_ = v___x_1066_;
goto v_reusejp_1069_;
}
else
{
lean_object* v_reuseFailAlloc_1071_; 
v_reuseFailAlloc_1071_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1071_, 0, v___x_1068_);
v___x_1070_ = v_reuseFailAlloc_1071_;
goto v_reusejp_1069_;
}
v_reusejp_1069_:
{
return v___x_1070_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_simpEq_spec__0___redArg___boxed(lean_object* v_msg_1073_, lean_object* v___y_1074_, lean_object* v___y_1075_, lean_object* v___y_1076_, lean_object* v___y_1077_, lean_object* v___y_1078_){
_start:
{
lean_object* v_res_1079_; 
v_res_1079_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_simpEq_spec__0___redArg(v_msg_1073_, v___y_1074_, v___y_1075_, v___y_1076_, v___y_1077_);
lean_dec(v___y_1077_);
lean_dec_ref(v___y_1076_);
lean_dec(v___y_1075_);
lean_dec_ref(v___y_1074_);
return v_res_1079_;
}
}
static lean_object* _init_lp_mathlib_Lean_Meta_simpEq___lam__0___closed__1(void){
_start:
{
lean_object* v___x_1081_; lean_object* v___x_1082_; 
v___x_1081_ = ((lean_object*)(lp_mathlib_Lean_Meta_simpEq___lam__0___closed__0));
v___x_1082_ = l_Lean_stringToMessageData(v___x_1081_);
return v___x_1082_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpEq___lam__0(lean_object* v_S_1086_, lean_object* v_pf_1087_, lean_object* v_fvars_1088_, lean_object* v_type_1089_, lean_object* v___y_1090_, lean_object* v___y_1091_, lean_object* v___y_1092_, lean_object* v___y_1093_){
_start:
{
if (lean_obj_tag(v_type_1089_) == 5)
{
lean_object* v_fn_1098_; 
v_fn_1098_ = lean_ctor_get(v_type_1089_, 0);
lean_inc_ref(v_fn_1098_);
if (lean_obj_tag(v_fn_1098_) == 5)
{
lean_object* v_fn_1099_; 
v_fn_1099_ = lean_ctor_get(v_fn_1098_, 0);
lean_inc_ref(v_fn_1099_);
if (lean_obj_tag(v_fn_1099_) == 5)
{
lean_object* v_fn_1100_; 
v_fn_1100_ = lean_ctor_get(v_fn_1099_, 0);
lean_inc_ref(v_fn_1100_);
if (lean_obj_tag(v_fn_1100_) == 4)
{
lean_object* v_declName_1101_; 
v_declName_1101_ = lean_ctor_get(v_fn_1100_, 0);
lean_inc(v_declName_1101_);
if (lean_obj_tag(v_declName_1101_) == 1)
{
lean_object* v_pre_1102_; 
v_pre_1102_ = lean_ctor_get(v_declName_1101_, 0);
if (lean_obj_tag(v_pre_1102_) == 0)
{
lean_object* v_arg_1103_; lean_object* v_arg_1104_; lean_object* v_arg_1105_; lean_object* v_us_1106_; lean_object* v_str_1107_; lean_object* v___x_1108_; uint8_t v___x_1109_; 
v_arg_1103_ = lean_ctor_get(v_type_1089_, 1);
lean_inc_ref(v_arg_1103_);
lean_dec_ref_known(v_type_1089_, 2);
v_arg_1104_ = lean_ctor_get(v_fn_1098_, 1);
lean_inc_ref(v_arg_1104_);
lean_dec_ref_known(v_fn_1098_, 2);
v_arg_1105_ = lean_ctor_get(v_fn_1099_, 1);
lean_inc_ref(v_arg_1105_);
lean_dec_ref_known(v_fn_1099_, 2);
v_us_1106_ = lean_ctor_get(v_fn_1100_, 1);
lean_inc(v_us_1106_);
lean_dec_ref_known(v_fn_1100_, 2);
v_str_1107_ = lean_ctor_get(v_declName_1101_, 1);
lean_inc_ref(v_str_1107_);
lean_dec_ref_known(v_declName_1101_, 2);
v___x_1108_ = ((lean_object*)(lp_mathlib_Lean_Meta_simpEq___lam__0___closed__2));
v___x_1109_ = lean_string_dec_eq(v_str_1107_, v___x_1108_);
lean_dec_ref(v_str_1107_);
if (v___x_1109_ == 0)
{
lean_dec(v_us_1106_);
lean_dec_ref(v_arg_1105_);
lean_dec_ref(v_arg_1104_);
lean_dec_ref(v_arg_1103_);
lean_dec_ref(v_pf_1087_);
lean_dec_ref(v_S_1086_);
goto v___jp_1095_;
}
else
{
if (lean_obj_tag(v_us_1106_) == 1)
{
lean_object* v_tail_1110_; 
v_tail_1110_ = lean_ctor_get(v_us_1106_, 1);
if (lean_obj_tag(v_tail_1110_) == 0)
{
lean_object* v___x_1111_; 
lean_inc_ref(v_S_1086_);
lean_inc(v___y_1093_);
lean_inc_ref(v___y_1092_);
lean_inc(v___y_1091_);
lean_inc_ref(v___y_1090_);
v___x_1111_ = lean_apply_6(v_S_1086_, v_arg_1104_, v___y_1090_, v___y_1091_, v___y_1092_, v___y_1093_, lean_box(0));
if (lean_obj_tag(v___x_1111_) == 0)
{
lean_object* v_a_1112_; lean_object* v_expr_1113_; lean_object* v_proof_x3f_1114_; lean_object* v___x_1115_; 
v_a_1112_ = lean_ctor_get(v___x_1111_, 0);
lean_inc(v_a_1112_);
lean_dec_ref_known(v___x_1111_, 1);
v_expr_1113_ = lean_ctor_get(v_a_1112_, 0);
lean_inc_ref(v_expr_1113_);
v_proof_x3f_1114_ = lean_ctor_get(v_a_1112_, 1);
lean_inc(v_proof_x3f_1114_);
lean_dec(v_a_1112_);
lean_inc(v___y_1093_);
lean_inc_ref(v___y_1092_);
lean_inc(v___y_1091_);
lean_inc_ref(v___y_1090_);
v___x_1115_ = lean_apply_6(v_S_1086_, v_arg_1103_, v___y_1090_, v___y_1091_, v___y_1092_, v___y_1093_, lean_box(0));
if (lean_obj_tag(v___x_1115_) == 0)
{
lean_object* v_a_1116_; lean_object* v_expr_1117_; lean_object* v_proof_x3f_1118_; lean_object* v_pf_x27_1120_; lean_object* v___y_1121_; lean_object* v___y_1122_; lean_object* v___y_1123_; lean_object* v___y_1124_; lean_object* v_pf_x27_1159_; lean_object* v___y_1160_; lean_object* v___y_1161_; lean_object* v___y_1162_; lean_object* v___y_1163_; lean_object* v___x_1175_; 
v_a_1116_ = lean_ctor_get(v___x_1115_, 0);
lean_inc(v_a_1116_);
lean_dec_ref_known(v___x_1115_, 1);
v_expr_1117_ = lean_ctor_get(v_a_1116_, 0);
lean_inc_ref(v_expr_1117_);
v_proof_x3f_1118_ = lean_ctor_get(v_a_1116_, 1);
lean_inc(v_proof_x3f_1118_);
lean_dec(v_a_1116_);
v___x_1175_ = l_Lean_mkAppN(v_pf_1087_, v_fvars_1088_);
if (lean_obj_tag(v_proof_x3f_1114_) == 1)
{
lean_object* v_val_1176_; lean_object* v___x_1177_; 
v_val_1176_ = lean_ctor_get(v_proof_x3f_1114_, 0);
lean_inc(v_val_1176_);
lean_dec_ref_known(v_proof_x3f_1114_, 1);
v___x_1177_ = l_Lean_Meta_mkEqSymm(v_val_1176_, v___y_1090_, v___y_1091_, v___y_1092_, v___y_1093_);
if (lean_obj_tag(v___x_1177_) == 0)
{
lean_object* v_a_1178_; lean_object* v___x_1179_; 
v_a_1178_ = lean_ctor_get(v___x_1177_, 0);
lean_inc(v_a_1178_);
lean_dec_ref_known(v___x_1177_, 1);
v___x_1179_ = l_Lean_Meta_mkEqTrans(v_a_1178_, v___x_1175_, v___y_1090_, v___y_1091_, v___y_1092_, v___y_1093_);
if (lean_obj_tag(v___x_1179_) == 0)
{
lean_object* v_a_1180_; 
v_a_1180_ = lean_ctor_get(v___x_1179_, 0);
lean_inc(v_a_1180_);
lean_dec_ref_known(v___x_1179_, 1);
v_pf_x27_1159_ = v_a_1180_;
v___y_1160_ = v___y_1090_;
v___y_1161_ = v___y_1091_;
v___y_1162_ = v___y_1092_;
v___y_1163_ = v___y_1093_;
goto v___jp_1158_;
}
else
{
lean_object* v_a_1181_; lean_object* v___x_1183_; uint8_t v_isShared_1184_; uint8_t v_isSharedCheck_1188_; 
lean_dec(v_proof_x3f_1118_);
lean_dec_ref(v_expr_1117_);
lean_dec_ref(v_expr_1113_);
lean_dec_ref_known(v_us_1106_, 2);
lean_dec_ref(v_arg_1105_);
v_a_1181_ = lean_ctor_get(v___x_1179_, 0);
v_isSharedCheck_1188_ = !lean_is_exclusive(v___x_1179_);
if (v_isSharedCheck_1188_ == 0)
{
v___x_1183_ = v___x_1179_;
v_isShared_1184_ = v_isSharedCheck_1188_;
goto v_resetjp_1182_;
}
else
{
lean_inc(v_a_1181_);
lean_dec(v___x_1179_);
v___x_1183_ = lean_box(0);
v_isShared_1184_ = v_isSharedCheck_1188_;
goto v_resetjp_1182_;
}
v_resetjp_1182_:
{
lean_object* v___x_1186_; 
if (v_isShared_1184_ == 0)
{
v___x_1186_ = v___x_1183_;
goto v_reusejp_1185_;
}
else
{
lean_object* v_reuseFailAlloc_1187_; 
v_reuseFailAlloc_1187_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1187_, 0, v_a_1181_);
v___x_1186_ = v_reuseFailAlloc_1187_;
goto v_reusejp_1185_;
}
v_reusejp_1185_:
{
return v___x_1186_;
}
}
}
}
else
{
lean_object* v_a_1189_; lean_object* v___x_1191_; uint8_t v_isShared_1192_; uint8_t v_isSharedCheck_1196_; 
lean_dec_ref(v___x_1175_);
lean_dec(v_proof_x3f_1118_);
lean_dec_ref(v_expr_1117_);
lean_dec_ref(v_expr_1113_);
lean_dec_ref_known(v_us_1106_, 2);
lean_dec_ref(v_arg_1105_);
v_a_1189_ = lean_ctor_get(v___x_1177_, 0);
v_isSharedCheck_1196_ = !lean_is_exclusive(v___x_1177_);
if (v_isSharedCheck_1196_ == 0)
{
v___x_1191_ = v___x_1177_;
v_isShared_1192_ = v_isSharedCheck_1196_;
goto v_resetjp_1190_;
}
else
{
lean_inc(v_a_1189_);
lean_dec(v___x_1177_);
v___x_1191_ = lean_box(0);
v_isShared_1192_ = v_isSharedCheck_1196_;
goto v_resetjp_1190_;
}
v_resetjp_1190_:
{
lean_object* v___x_1194_; 
if (v_isShared_1192_ == 0)
{
v___x_1194_ = v___x_1191_;
goto v_reusejp_1193_;
}
else
{
lean_object* v_reuseFailAlloc_1195_; 
v_reuseFailAlloc_1195_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1195_, 0, v_a_1189_);
v___x_1194_ = v_reuseFailAlloc_1195_;
goto v_reusejp_1193_;
}
v_reusejp_1193_:
{
return v___x_1194_;
}
}
}
}
else
{
lean_dec(v_proof_x3f_1114_);
v_pf_x27_1159_ = v___x_1175_;
v___y_1160_ = v___y_1090_;
v___y_1161_ = v___y_1091_;
v___y_1162_ = v___y_1092_;
v___y_1163_ = v___y_1093_;
goto v___jp_1158_;
}
v___jp_1119_:
{
lean_object* v___x_1125_; lean_object* v___x_1126_; lean_object* v___x_1127_; uint8_t v___x_1128_; uint8_t v___x_1129_; lean_object* v___x_1130_; 
v___x_1125_ = ((lean_object*)(lp_mathlib_Lean_Meta_simpEq___lam__0___closed__3));
v___x_1126_ = l_Lean_mkConst(v___x_1125_, v_us_1106_);
v___x_1127_ = l_Lean_mkApp3(v___x_1126_, v_arg_1105_, v_expr_1113_, v_expr_1117_);
v___x_1128_ = 0;
v___x_1129_ = 1;
v___x_1130_ = l_Lean_Meta_mkForallFVars(v_fvars_1088_, v___x_1127_, v___x_1128_, v___x_1109_, v___x_1109_, v___x_1129_, v___y_1121_, v___y_1122_, v___y_1123_, v___y_1124_);
if (lean_obj_tag(v___x_1130_) == 0)
{
lean_object* v_a_1131_; lean_object* v___x_1132_; 
v_a_1131_ = lean_ctor_get(v___x_1130_, 0);
lean_inc(v_a_1131_);
lean_dec_ref_known(v___x_1130_, 1);
v___x_1132_ = l_Lean_Meta_mkLambdaFVars(v_fvars_1088_, v_pf_x27_1120_, v___x_1128_, v___x_1109_, v___x_1128_, v___x_1109_, v___x_1129_, v___y_1121_, v___y_1122_, v___y_1123_, v___y_1124_);
if (lean_obj_tag(v___x_1132_) == 0)
{
lean_object* v_a_1133_; lean_object* v___x_1135_; uint8_t v_isShared_1136_; uint8_t v_isSharedCheck_1141_; 
v_a_1133_ = lean_ctor_get(v___x_1132_, 0);
v_isSharedCheck_1141_ = !lean_is_exclusive(v___x_1132_);
if (v_isSharedCheck_1141_ == 0)
{
v___x_1135_ = v___x_1132_;
v_isShared_1136_ = v_isSharedCheck_1141_;
goto v_resetjp_1134_;
}
else
{
lean_inc(v_a_1133_);
lean_dec(v___x_1132_);
v___x_1135_ = lean_box(0);
v_isShared_1136_ = v_isSharedCheck_1141_;
goto v_resetjp_1134_;
}
v_resetjp_1134_:
{
lean_object* v___x_1137_; lean_object* v___x_1139_; 
v___x_1137_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1137_, 0, v_a_1131_);
lean_ctor_set(v___x_1137_, 1, v_a_1133_);
if (v_isShared_1136_ == 0)
{
lean_ctor_set(v___x_1135_, 0, v___x_1137_);
v___x_1139_ = v___x_1135_;
goto v_reusejp_1138_;
}
else
{
lean_object* v_reuseFailAlloc_1140_; 
v_reuseFailAlloc_1140_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1140_, 0, v___x_1137_);
v___x_1139_ = v_reuseFailAlloc_1140_;
goto v_reusejp_1138_;
}
v_reusejp_1138_:
{
return v___x_1139_;
}
}
}
else
{
lean_object* v_a_1142_; lean_object* v___x_1144_; uint8_t v_isShared_1145_; uint8_t v_isSharedCheck_1149_; 
lean_dec(v_a_1131_);
v_a_1142_ = lean_ctor_get(v___x_1132_, 0);
v_isSharedCheck_1149_ = !lean_is_exclusive(v___x_1132_);
if (v_isSharedCheck_1149_ == 0)
{
v___x_1144_ = v___x_1132_;
v_isShared_1145_ = v_isSharedCheck_1149_;
goto v_resetjp_1143_;
}
else
{
lean_inc(v_a_1142_);
lean_dec(v___x_1132_);
v___x_1144_ = lean_box(0);
v_isShared_1145_ = v_isSharedCheck_1149_;
goto v_resetjp_1143_;
}
v_resetjp_1143_:
{
lean_object* v___x_1147_; 
if (v_isShared_1145_ == 0)
{
v___x_1147_ = v___x_1144_;
goto v_reusejp_1146_;
}
else
{
lean_object* v_reuseFailAlloc_1148_; 
v_reuseFailAlloc_1148_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1148_, 0, v_a_1142_);
v___x_1147_ = v_reuseFailAlloc_1148_;
goto v_reusejp_1146_;
}
v_reusejp_1146_:
{
return v___x_1147_;
}
}
}
}
else
{
lean_object* v_a_1150_; lean_object* v___x_1152_; uint8_t v_isShared_1153_; uint8_t v_isSharedCheck_1157_; 
lean_dec_ref(v_pf_x27_1120_);
v_a_1150_ = lean_ctor_get(v___x_1130_, 0);
v_isSharedCheck_1157_ = !lean_is_exclusive(v___x_1130_);
if (v_isSharedCheck_1157_ == 0)
{
v___x_1152_ = v___x_1130_;
v_isShared_1153_ = v_isSharedCheck_1157_;
goto v_resetjp_1151_;
}
else
{
lean_inc(v_a_1150_);
lean_dec(v___x_1130_);
v___x_1152_ = lean_box(0);
v_isShared_1153_ = v_isSharedCheck_1157_;
goto v_resetjp_1151_;
}
v_resetjp_1151_:
{
lean_object* v___x_1155_; 
if (v_isShared_1153_ == 0)
{
v___x_1155_ = v___x_1152_;
goto v_reusejp_1154_;
}
else
{
lean_object* v_reuseFailAlloc_1156_; 
v_reuseFailAlloc_1156_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1156_, 0, v_a_1150_);
v___x_1155_ = v_reuseFailAlloc_1156_;
goto v_reusejp_1154_;
}
v_reusejp_1154_:
{
return v___x_1155_;
}
}
}
}
v___jp_1158_:
{
if (lean_obj_tag(v_proof_x3f_1118_) == 1)
{
lean_object* v_val_1164_; lean_object* v___x_1165_; 
v_val_1164_ = lean_ctor_get(v_proof_x3f_1118_, 0);
lean_inc(v_val_1164_);
lean_dec_ref_known(v_proof_x3f_1118_, 1);
v___x_1165_ = l_Lean_Meta_mkEqTrans(v_pf_x27_1159_, v_val_1164_, v___y_1160_, v___y_1161_, v___y_1162_, v___y_1163_);
if (lean_obj_tag(v___x_1165_) == 0)
{
lean_object* v_a_1166_; 
v_a_1166_ = lean_ctor_get(v___x_1165_, 0);
lean_inc(v_a_1166_);
lean_dec_ref_known(v___x_1165_, 1);
v_pf_x27_1120_ = v_a_1166_;
v___y_1121_ = v___y_1160_;
v___y_1122_ = v___y_1161_;
v___y_1123_ = v___y_1162_;
v___y_1124_ = v___y_1163_;
goto v___jp_1119_;
}
else
{
lean_object* v_a_1167_; lean_object* v___x_1169_; uint8_t v_isShared_1170_; uint8_t v_isSharedCheck_1174_; 
lean_dec_ref(v_expr_1117_);
lean_dec_ref(v_expr_1113_);
lean_dec_ref_known(v_us_1106_, 2);
lean_dec_ref(v_arg_1105_);
v_a_1167_ = lean_ctor_get(v___x_1165_, 0);
v_isSharedCheck_1174_ = !lean_is_exclusive(v___x_1165_);
if (v_isSharedCheck_1174_ == 0)
{
v___x_1169_ = v___x_1165_;
v_isShared_1170_ = v_isSharedCheck_1174_;
goto v_resetjp_1168_;
}
else
{
lean_inc(v_a_1167_);
lean_dec(v___x_1165_);
v___x_1169_ = lean_box(0);
v_isShared_1170_ = v_isSharedCheck_1174_;
goto v_resetjp_1168_;
}
v_resetjp_1168_:
{
lean_object* v___x_1172_; 
if (v_isShared_1170_ == 0)
{
v___x_1172_ = v___x_1169_;
goto v_reusejp_1171_;
}
else
{
lean_object* v_reuseFailAlloc_1173_; 
v_reuseFailAlloc_1173_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1173_, 0, v_a_1167_);
v___x_1172_ = v_reuseFailAlloc_1173_;
goto v_reusejp_1171_;
}
v_reusejp_1171_:
{
return v___x_1172_;
}
}
}
}
else
{
lean_dec(v_proof_x3f_1118_);
v_pf_x27_1120_ = v_pf_x27_1159_;
v___y_1121_ = v___y_1160_;
v___y_1122_ = v___y_1161_;
v___y_1123_ = v___y_1162_;
v___y_1124_ = v___y_1163_;
goto v___jp_1119_;
}
}
}
else
{
lean_object* v_a_1197_; lean_object* v___x_1199_; uint8_t v_isShared_1200_; uint8_t v_isSharedCheck_1204_; 
lean_dec(v_proof_x3f_1114_);
lean_dec_ref(v_expr_1113_);
lean_dec_ref_known(v_us_1106_, 2);
lean_dec_ref(v_arg_1105_);
lean_dec_ref(v_pf_1087_);
v_a_1197_ = lean_ctor_get(v___x_1115_, 0);
v_isSharedCheck_1204_ = !lean_is_exclusive(v___x_1115_);
if (v_isSharedCheck_1204_ == 0)
{
v___x_1199_ = v___x_1115_;
v_isShared_1200_ = v_isSharedCheck_1204_;
goto v_resetjp_1198_;
}
else
{
lean_inc(v_a_1197_);
lean_dec(v___x_1115_);
v___x_1199_ = lean_box(0);
v_isShared_1200_ = v_isSharedCheck_1204_;
goto v_resetjp_1198_;
}
v_resetjp_1198_:
{
lean_object* v___x_1202_; 
if (v_isShared_1200_ == 0)
{
v___x_1202_ = v___x_1199_;
goto v_reusejp_1201_;
}
else
{
lean_object* v_reuseFailAlloc_1203_; 
v_reuseFailAlloc_1203_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1203_, 0, v_a_1197_);
v___x_1202_ = v_reuseFailAlloc_1203_;
goto v_reusejp_1201_;
}
v_reusejp_1201_:
{
return v___x_1202_;
}
}
}
}
else
{
lean_object* v_a_1205_; lean_object* v___x_1207_; uint8_t v_isShared_1208_; uint8_t v_isSharedCheck_1212_; 
lean_dec_ref_known(v_us_1106_, 2);
lean_dec_ref(v_arg_1105_);
lean_dec_ref(v_arg_1103_);
lean_dec_ref(v_pf_1087_);
lean_dec_ref(v_S_1086_);
v_a_1205_ = lean_ctor_get(v___x_1111_, 0);
v_isSharedCheck_1212_ = !lean_is_exclusive(v___x_1111_);
if (v_isSharedCheck_1212_ == 0)
{
v___x_1207_ = v___x_1111_;
v_isShared_1208_ = v_isSharedCheck_1212_;
goto v_resetjp_1206_;
}
else
{
lean_inc(v_a_1205_);
lean_dec(v___x_1111_);
v___x_1207_ = lean_box(0);
v_isShared_1208_ = v_isSharedCheck_1212_;
goto v_resetjp_1206_;
}
v_resetjp_1206_:
{
lean_object* v___x_1210_; 
if (v_isShared_1208_ == 0)
{
v___x_1210_ = v___x_1207_;
goto v_reusejp_1209_;
}
else
{
lean_object* v_reuseFailAlloc_1211_; 
v_reuseFailAlloc_1211_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1211_, 0, v_a_1205_);
v___x_1210_ = v_reuseFailAlloc_1211_;
goto v_reusejp_1209_;
}
v_reusejp_1209_:
{
return v___x_1210_;
}
}
}
}
else
{
lean_dec_ref_known(v_us_1106_, 2);
lean_dec_ref(v_arg_1105_);
lean_dec_ref(v_arg_1104_);
lean_dec_ref(v_arg_1103_);
lean_dec_ref(v_pf_1087_);
lean_dec_ref(v_S_1086_);
goto v___jp_1095_;
}
}
else
{
lean_dec(v_us_1106_);
lean_dec_ref(v_arg_1105_);
lean_dec_ref(v_arg_1104_);
lean_dec_ref(v_arg_1103_);
lean_dec_ref(v_pf_1087_);
lean_dec_ref(v_S_1086_);
goto v___jp_1095_;
}
}
}
else
{
lean_dec_ref_known(v_declName_1101_, 2);
lean_dec_ref_known(v_fn_1100_, 2);
lean_dec_ref_known(v_fn_1099_, 2);
lean_dec_ref_known(v_fn_1098_, 2);
lean_dec_ref_known(v_type_1089_, 2);
lean_dec_ref(v_pf_1087_);
lean_dec_ref(v_S_1086_);
goto v___jp_1095_;
}
}
else
{
lean_dec(v_declName_1101_);
lean_dec_ref_known(v_fn_1100_, 2);
lean_dec_ref_known(v_fn_1099_, 2);
lean_dec_ref_known(v_fn_1098_, 2);
lean_dec_ref_known(v_type_1089_, 2);
lean_dec_ref(v_pf_1087_);
lean_dec_ref(v_S_1086_);
goto v___jp_1095_;
}
}
else
{
lean_dec_ref(v_fn_1100_);
lean_dec_ref_known(v_fn_1099_, 2);
lean_dec_ref_known(v_fn_1098_, 2);
lean_dec_ref_known(v_type_1089_, 2);
lean_dec_ref(v_pf_1087_);
lean_dec_ref(v_S_1086_);
goto v___jp_1095_;
}
}
else
{
lean_dec_ref(v_fn_1099_);
lean_dec_ref_known(v_fn_1098_, 2);
lean_dec_ref_known(v_type_1089_, 2);
lean_dec_ref(v_pf_1087_);
lean_dec_ref(v_S_1086_);
goto v___jp_1095_;
}
}
else
{
lean_dec_ref_known(v_type_1089_, 2);
lean_dec_ref(v_fn_1098_);
lean_dec_ref(v_pf_1087_);
lean_dec_ref(v_S_1086_);
goto v___jp_1095_;
}
}
else
{
lean_dec_ref(v_type_1089_);
lean_dec_ref(v_pf_1087_);
lean_dec_ref(v_S_1086_);
goto v___jp_1095_;
}
v___jp_1095_:
{
lean_object* v___x_1096_; lean_object* v___x_1097_; 
v___x_1096_ = lean_obj_once(&lp_mathlib_Lean_Meta_simpEq___lam__0___closed__1, &lp_mathlib_Lean_Meta_simpEq___lam__0___closed__1_once, _init_lp_mathlib_Lean_Meta_simpEq___lam__0___closed__1);
v___x_1097_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_simpEq_spec__0___redArg(v___x_1096_, v___y_1090_, v___y_1091_, v___y_1092_, v___y_1093_);
return v___x_1097_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpEq___lam__0___boxed(lean_object* v_S_1213_, lean_object* v_pf_1214_, lean_object* v_fvars_1215_, lean_object* v_type_1216_, lean_object* v___y_1217_, lean_object* v___y_1218_, lean_object* v___y_1219_, lean_object* v___y_1220_, lean_object* v___y_1221_){
_start:
{
lean_object* v_res_1222_; 
v_res_1222_ = lp_mathlib_Lean_Meta_simpEq___lam__0(v_S_1213_, v_pf_1214_, v_fvars_1215_, v_type_1216_, v___y_1217_, v___y_1218_, v___y_1219_, v___y_1220_);
lean_dec(v___y_1220_);
lean_dec_ref(v___y_1219_);
lean_dec(v___y_1218_);
lean_dec_ref(v___y_1217_);
lean_dec_ref(v_fvars_1215_);
return v_res_1222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpEq(lean_object* v_S_1223_, lean_object* v_type_1224_, lean_object* v_pf_1225_, lean_object* v_a_1226_, lean_object* v_a_1227_, lean_object* v_a_1228_, lean_object* v_a_1229_){
_start:
{
lean_object* v___f_1231_; uint8_t v___x_1232_; lean_object* v___x_1233_; 
v___f_1231_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_simpEq___lam__0___boxed), 9, 2);
lean_closure_set(v___f_1231_, 0, v_S_1223_);
lean_closure_set(v___f_1231_, 1, v_pf_1225_);
v___x_1232_ = 0;
v___x_1233_ = lp_mathlib_Lean_Meta_forallTelescope___at___00Lean_Meta_simpEq_spec__1___redArg(v_type_1224_, v___f_1231_, v___x_1232_, v_a_1226_, v_a_1227_, v_a_1228_, v_a_1229_);
return v___x_1233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_simpEq___boxed(lean_object* v_S_1234_, lean_object* v_type_1235_, lean_object* v_pf_1236_, lean_object* v_a_1237_, lean_object* v_a_1238_, lean_object* v_a_1239_, lean_object* v_a_1240_, lean_object* v_a_1241_){
_start:
{
lean_object* v_res_1242_; 
v_res_1242_ = lp_mathlib_Lean_Meta_simpEq(v_S_1234_, v_type_1235_, v_pf_1236_, v_a_1237_, v_a_1238_, v_a_1239_, v_a_1240_);
lean_dec(v_a_1240_);
lean_dec_ref(v_a_1239_);
lean_dec(v_a_1238_);
lean_dec_ref(v_a_1237_);
return v_res_1242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_simpEq_spec__0(lean_object* v_00_u03b1_1243_, lean_object* v_msg_1244_, lean_object* v___y_1245_, lean_object* v___y_1246_, lean_object* v___y_1247_, lean_object* v___y_1248_){
_start:
{
lean_object* v___x_1250_; 
v___x_1250_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_simpEq_spec__0___redArg(v_msg_1244_, v___y_1245_, v___y_1246_, v___y_1247_, v___y_1248_);
return v___x_1250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_Meta_simpEq_spec__0___boxed(lean_object* v_00_u03b1_1251_, lean_object* v_msg_1252_, lean_object* v___y_1253_, lean_object* v___y_1254_, lean_object* v___y_1255_, lean_object* v___y_1256_, lean_object* v___y_1257_){
_start:
{
lean_object* v_res_1258_; 
v_res_1258_ = lp_mathlib_Lean_throwError___at___00Lean_Meta_simpEq_spec__0(v_00_u03b1_1251_, v_msg_1252_, v___y_1253_, v___y_1254_, v___y_1255_, v___y_1256_);
lean_dec(v___y_1256_);
lean_dec_ref(v___y_1255_);
lean_dec(v___y_1254_);
lean_dec_ref(v___y_1253_);
return v_res_1258_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Meta_SimpTheorems_contains(lean_object* v_d_1259_, lean_object* v_declName_1260_){
_start:
{
uint8_t v___x_1261_; uint8_t v___x_1262_; lean_object* v___x_1263_; uint8_t v___x_1264_; 
v___x_1261_ = 1;
v___x_1262_ = 0;
lean_inc(v_declName_1260_);
v___x_1263_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_1263_, 0, v_declName_1260_);
lean_ctor_set_uint8(v___x_1263_, sizeof(void*)*1, v___x_1261_);
lean_ctor_set_uint8(v___x_1263_, sizeof(void*)*1 + 1, v___x_1262_);
v___x_1264_ = l_Lean_Meta_SimpTheorems_isLemma(v_d_1259_, v___x_1263_);
lean_dec_ref_known(v___x_1263_, 1);
if (v___x_1264_ == 0)
{
uint8_t v___x_1265_; 
v___x_1265_ = l_Lean_Meta_SimpTheorems_isDeclToUnfold(v_d_1259_, v_declName_1260_);
lean_dec(v_declName_1260_);
return v___x_1265_;
}
else
{
lean_dec(v_declName_1260_);
return v___x_1264_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_SimpTheorems_contains___boxed(lean_object* v_d_1266_, lean_object* v_declName_1267_){
_start:
{
uint8_t v_res_1268_; lean_object* v_r_1269_; 
v_res_1268_ = lp_mathlib_Lean_Meta_SimpTheorems_contains(v_d_1266_, v_declName_1267_);
lean_dec_ref(v_d_1266_);
v_r_1269_ = lean_box(v_res_1268_);
return v_r_1269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_isInSimpSet(lean_object* v_simpAttr_1270_, lean_object* v_decl_1271_, lean_object* v_a_1272_, lean_object* v_a_1273_){
_start:
{
lean_object* v___x_1275_; 
v___x_1275_ = l_Lean_Meta_getSimpExtension_x3f(v_simpAttr_1270_, v_a_1272_, v_a_1273_);
if (lean_obj_tag(v___x_1275_) == 0)
{
lean_object* v_a_1276_; lean_object* v___x_1278_; uint8_t v_isShared_1279_; uint8_t v_isSharedCheck_1305_; 
v_a_1276_ = lean_ctor_get(v___x_1275_, 0);
v_isSharedCheck_1305_ = !lean_is_exclusive(v___x_1275_);
if (v_isSharedCheck_1305_ == 0)
{
v___x_1278_ = v___x_1275_;
v_isShared_1279_ = v_isSharedCheck_1305_;
goto v_resetjp_1277_;
}
else
{
lean_inc(v_a_1276_);
lean_dec(v___x_1275_);
v___x_1278_ = lean_box(0);
v_isShared_1279_ = v_isSharedCheck_1305_;
goto v_resetjp_1277_;
}
v_resetjp_1277_:
{
if (lean_obj_tag(v_a_1276_) == 1)
{
lean_object* v_val_1280_; lean_object* v___x_1281_; 
lean_del_object(v___x_1278_);
v_val_1280_ = lean_ctor_get(v_a_1276_, 0);
lean_inc(v_val_1280_);
lean_dec_ref_known(v_a_1276_, 1);
v___x_1281_ = l_Lean_Meta_SimpExtension_getTheorems___redArg(v_val_1280_, v_a_1273_);
lean_dec(v_val_1280_);
if (lean_obj_tag(v___x_1281_) == 0)
{
lean_object* v_a_1282_; lean_object* v___x_1284_; uint8_t v_isShared_1285_; uint8_t v_isSharedCheck_1291_; 
v_a_1282_ = lean_ctor_get(v___x_1281_, 0);
v_isSharedCheck_1291_ = !lean_is_exclusive(v___x_1281_);
if (v_isSharedCheck_1291_ == 0)
{
v___x_1284_ = v___x_1281_;
v_isShared_1285_ = v_isSharedCheck_1291_;
goto v_resetjp_1283_;
}
else
{
lean_inc(v_a_1282_);
lean_dec(v___x_1281_);
v___x_1284_ = lean_box(0);
v_isShared_1285_ = v_isSharedCheck_1291_;
goto v_resetjp_1283_;
}
v_resetjp_1283_:
{
uint8_t v___x_1286_; lean_object* v___x_1287_; lean_object* v___x_1289_; 
v___x_1286_ = lp_mathlib_Lean_Meta_SimpTheorems_contains(v_a_1282_, v_decl_1271_);
lean_dec(v_a_1282_);
v___x_1287_ = lean_box(v___x_1286_);
if (v_isShared_1285_ == 0)
{
lean_ctor_set(v___x_1284_, 0, v___x_1287_);
v___x_1289_ = v___x_1284_;
goto v_reusejp_1288_;
}
else
{
lean_object* v_reuseFailAlloc_1290_; 
v_reuseFailAlloc_1290_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1290_, 0, v___x_1287_);
v___x_1289_ = v_reuseFailAlloc_1290_;
goto v_reusejp_1288_;
}
v_reusejp_1288_:
{
return v___x_1289_;
}
}
}
else
{
lean_object* v_a_1292_; lean_object* v___x_1294_; uint8_t v_isShared_1295_; uint8_t v_isSharedCheck_1299_; 
lean_dec(v_decl_1271_);
v_a_1292_ = lean_ctor_get(v___x_1281_, 0);
v_isSharedCheck_1299_ = !lean_is_exclusive(v___x_1281_);
if (v_isSharedCheck_1299_ == 0)
{
v___x_1294_ = v___x_1281_;
v_isShared_1295_ = v_isSharedCheck_1299_;
goto v_resetjp_1293_;
}
else
{
lean_inc(v_a_1292_);
lean_dec(v___x_1281_);
v___x_1294_ = lean_box(0);
v_isShared_1295_ = v_isSharedCheck_1299_;
goto v_resetjp_1293_;
}
v_resetjp_1293_:
{
lean_object* v___x_1297_; 
if (v_isShared_1295_ == 0)
{
v___x_1297_ = v___x_1294_;
goto v_reusejp_1296_;
}
else
{
lean_object* v_reuseFailAlloc_1298_; 
v_reuseFailAlloc_1298_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1298_, 0, v_a_1292_);
v___x_1297_ = v_reuseFailAlloc_1298_;
goto v_reusejp_1296_;
}
v_reusejp_1296_:
{
return v___x_1297_;
}
}
}
}
else
{
uint8_t v___x_1300_; lean_object* v___x_1301_; lean_object* v___x_1303_; 
lean_dec(v_a_1276_);
lean_dec(v_decl_1271_);
v___x_1300_ = 0;
v___x_1301_ = lean_box(v___x_1300_);
if (v_isShared_1279_ == 0)
{
lean_ctor_set(v___x_1278_, 0, v___x_1301_);
v___x_1303_ = v___x_1278_;
goto v_reusejp_1302_;
}
else
{
lean_object* v_reuseFailAlloc_1304_; 
v_reuseFailAlloc_1304_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1304_, 0, v___x_1301_);
v___x_1303_ = v_reuseFailAlloc_1304_;
goto v_reusejp_1302_;
}
v_reusejp_1302_:
{
return v___x_1303_;
}
}
}
}
else
{
lean_object* v_a_1306_; lean_object* v___x_1308_; uint8_t v_isShared_1309_; uint8_t v_isSharedCheck_1313_; 
lean_dec(v_decl_1271_);
v_a_1306_ = lean_ctor_get(v___x_1275_, 0);
v_isSharedCheck_1313_ = !lean_is_exclusive(v___x_1275_);
if (v_isSharedCheck_1313_ == 0)
{
v___x_1308_ = v___x_1275_;
v_isShared_1309_ = v_isSharedCheck_1313_;
goto v_resetjp_1307_;
}
else
{
lean_inc(v_a_1306_);
lean_dec(v___x_1275_);
v___x_1308_ = lean_box(0);
v_isShared_1309_ = v_isSharedCheck_1313_;
goto v_resetjp_1307_;
}
v_resetjp_1307_:
{
lean_object* v___x_1311_; 
if (v_isShared_1309_ == 0)
{
v___x_1311_ = v___x_1308_;
goto v_reusejp_1310_;
}
else
{
lean_object* v_reuseFailAlloc_1312_; 
v_reuseFailAlloc_1312_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1312_, 0, v_a_1306_);
v___x_1311_ = v_reuseFailAlloc_1312_;
goto v_reusejp_1310_;
}
v_reusejp_1310_:
{
return v___x_1311_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_isInSimpSet___boxed(lean_object* v_simpAttr_1314_, lean_object* v_decl_1315_, lean_object* v_a_1316_, lean_object* v_a_1317_, lean_object* v_a_1318_){
_start:
{
lean_object* v_res_1319_; 
v_res_1319_ = lp_mathlib_Lean_Meta_isInSimpSet(v_simpAttr_1314_, v_decl_1315_, v_a_1316_, v_a_1317_);
lean_dec(v_a_1317_);
lean_dec_ref(v_a_1316_);
lean_dec(v_simpAttr_1314_);
return v_res_1319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg(lean_object* v_f_1320_, lean_object* v_keys_1321_, lean_object* v_vals_1322_, lean_object* v_i_1323_, lean_object* v_acc_1324_){
_start:
{
lean_object* v___x_1325_; uint8_t v___x_1326_; 
v___x_1325_ = lean_array_get_size(v_keys_1321_);
v___x_1326_ = lean_nat_dec_lt(v_i_1323_, v___x_1325_);
if (v___x_1326_ == 0)
{
lean_dec(v_i_1323_);
lean_dec(v_f_1320_);
return v_acc_1324_;
}
else
{
lean_object* v_k_1327_; lean_object* v_v_1328_; lean_object* v___x_1329_; lean_object* v___x_1330_; lean_object* v___x_1331_; 
v_k_1327_ = lean_array_fget_borrowed(v_keys_1321_, v_i_1323_);
v_v_1328_ = lean_array_fget_borrowed(v_vals_1322_, v_i_1323_);
lean_inc(v_f_1320_);
lean_inc(v_v_1328_);
lean_inc(v_k_1327_);
v___x_1329_ = lean_apply_3(v_f_1320_, v_acc_1324_, v_k_1327_, v_v_1328_);
v___x_1330_ = lean_unsigned_to_nat(1u);
v___x_1331_ = lean_nat_add(v_i_1323_, v___x_1330_);
lean_dec(v_i_1323_);
v_i_1323_ = v___x_1331_;
v_acc_1324_ = v___x_1329_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg___boxed(lean_object* v_f_1333_, lean_object* v_keys_1334_, lean_object* v_vals_1335_, lean_object* v_i_1336_, lean_object* v_acc_1337_){
_start:
{
lean_object* v_res_1338_; 
v_res_1338_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg(v_f_1333_, v_keys_1334_, v_vals_1335_, v_i_1336_, v_acc_1337_);
lean_dec_ref(v_vals_1335_);
lean_dec_ref(v_keys_1334_);
return v_res_1338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8___redArg(lean_object* v_f_1339_, lean_object* v_x_1340_, lean_object* v_x_1341_){
_start:
{
if (lean_obj_tag(v_x_1340_) == 0)
{
lean_object* v_es_1342_; lean_object* v___x_1343_; lean_object* v___x_1344_; uint8_t v___x_1345_; 
v_es_1342_ = lean_ctor_get(v_x_1340_, 0);
v___x_1343_ = lean_unsigned_to_nat(0u);
v___x_1344_ = lean_array_get_size(v_es_1342_);
v___x_1345_ = lean_nat_dec_lt(v___x_1343_, v___x_1344_);
if (v___x_1345_ == 0)
{
lean_dec(v_f_1339_);
return v_x_1341_;
}
else
{
uint8_t v___x_1346_; 
v___x_1346_ = lean_nat_dec_le(v___x_1344_, v___x_1344_);
if (v___x_1346_ == 0)
{
if (v___x_1345_ == 0)
{
lean_dec(v_f_1339_);
return v_x_1341_;
}
else
{
size_t v___x_1347_; size_t v___x_1348_; lean_object* v___x_1349_; 
v___x_1347_ = ((size_t)0ULL);
v___x_1348_ = lean_usize_of_nat(v___x_1344_);
v___x_1349_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__10___redArg(v_f_1339_, v_es_1342_, v___x_1347_, v___x_1348_, v_x_1341_);
return v___x_1349_;
}
}
else
{
size_t v___x_1350_; size_t v___x_1351_; lean_object* v___x_1352_; 
v___x_1350_ = ((size_t)0ULL);
v___x_1351_ = lean_usize_of_nat(v___x_1344_);
v___x_1352_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__10___redArg(v_f_1339_, v_es_1342_, v___x_1350_, v___x_1351_, v_x_1341_);
return v___x_1352_;
}
}
}
else
{
lean_object* v_ks_1353_; lean_object* v_vs_1354_; lean_object* v___x_1355_; lean_object* v___x_1356_; 
v_ks_1353_ = lean_ctor_get(v_x_1340_, 0);
v_vs_1354_ = lean_ctor_get(v_x_1340_, 1);
v___x_1355_ = lean_unsigned_to_nat(0u);
v___x_1356_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg(v_f_1339_, v_ks_1353_, v_vs_1354_, v___x_1355_, v_x_1341_);
return v___x_1356_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__10___redArg(lean_object* v_f_1357_, lean_object* v_as_1358_, size_t v_i_1359_, size_t v_stop_1360_, lean_object* v_b_1361_){
_start:
{
lean_object* v___y_1363_; uint8_t v___x_1367_; 
v___x_1367_ = lean_usize_dec_eq(v_i_1359_, v_stop_1360_);
if (v___x_1367_ == 0)
{
lean_object* v___x_1368_; 
v___x_1368_ = lean_array_uget_borrowed(v_as_1358_, v_i_1359_);
switch(lean_obj_tag(v___x_1368_))
{
case 0:
{
lean_object* v_key_1369_; lean_object* v_val_1370_; lean_object* v___x_1371_; 
v_key_1369_ = lean_ctor_get(v___x_1368_, 0);
v_val_1370_ = lean_ctor_get(v___x_1368_, 1);
lean_inc(v_f_1357_);
lean_inc(v_val_1370_);
lean_inc(v_key_1369_);
v___x_1371_ = lean_apply_3(v_f_1357_, v_b_1361_, v_key_1369_, v_val_1370_);
v___y_1363_ = v___x_1371_;
goto v___jp_1362_;
}
case 1:
{
lean_object* v_node_1372_; lean_object* v___x_1373_; 
v_node_1372_ = lean_ctor_get(v___x_1368_, 0);
lean_inc(v_f_1357_);
v___x_1373_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8___redArg(v_f_1357_, v_node_1372_, v_b_1361_);
v___y_1363_ = v___x_1373_;
goto v___jp_1362_;
}
default: 
{
v___y_1363_ = v_b_1361_;
goto v___jp_1362_;
}
}
}
else
{
lean_dec(v_f_1357_);
return v_b_1361_;
}
v___jp_1362_:
{
size_t v___x_1364_; size_t v___x_1365_; 
v___x_1364_ = ((size_t)1ULL);
v___x_1365_ = lean_usize_add(v_i_1359_, v___x_1364_);
v_i_1359_ = v___x_1365_;
v_b_1361_ = v___y_1363_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__10___redArg___boxed(lean_object* v_f_1374_, lean_object* v_as_1375_, lean_object* v_i_1376_, lean_object* v_stop_1377_, lean_object* v_b_1378_){
_start:
{
size_t v_i_boxed_1379_; size_t v_stop_boxed_1380_; lean_object* v_res_1381_; 
v_i_boxed_1379_ = lean_unbox_usize(v_i_1376_);
lean_dec(v_i_1376_);
v_stop_boxed_1380_ = lean_unbox_usize(v_stop_1377_);
lean_dec(v_stop_1377_);
v_res_1381_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__10___redArg(v_f_1374_, v_as_1375_, v_i_boxed_1379_, v_stop_boxed_1380_, v_b_1378_);
lean_dec_ref(v_as_1375_);
return v_res_1381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8___redArg___boxed(lean_object* v_f_1382_, lean_object* v_x_1383_, lean_object* v_x_1384_){
_start:
{
lean_object* v_res_1385_; 
v_res_1385_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8___redArg(v_f_1382_, v_x_1383_, v_x_1384_);
lean_dec_ref(v_x_1383_);
return v_res_1385_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5___redArg___lam__0(lean_object* v_f_1386_, lean_object* v_x1_1387_, lean_object* v_x2_1388_, lean_object* v_x3_1389_){
_start:
{
lean_object* v___x_1390_; 
v___x_1390_ = lean_apply_3(v_f_1386_, v_x1_1387_, v_x2_1388_, v_x3_1389_);
return v___x_1390_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5___redArg(lean_object* v_map_1391_, lean_object* v_f_1392_, lean_object* v_init_1393_){
_start:
{
lean_object* v___f_1394_; lean_object* v___x_1395_; 
v___f_1394_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1394_, 0, v_f_1392_);
v___x_1395_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8___redArg(v___f_1394_, v_map_1391_, v_init_1393_);
return v___x_1395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5___redArg___boxed(lean_object* v_map_1396_, lean_object* v_f_1397_, lean_object* v_init_1398_){
_start:
{
lean_object* v_res_1399_; 
v_res_1399_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5___redArg(v_map_1396_, v_f_1397_, v_init_1398_);
lean_dec_ref(v_map_1396_);
return v_res_1399_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3___redArg___lam__0(lean_object* v_ps_1400_, lean_object* v_k_1401_, lean_object* v_v_1402_){
_start:
{
lean_object* v___x_1403_; lean_object* v___x_1404_; 
v___x_1403_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1403_, 0, v_k_1401_);
lean_ctor_set(v___x_1403_, 1, v_v_1402_);
v___x_1404_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1404_, 0, v___x_1403_);
lean_ctor_set(v___x_1404_, 1, v_ps_1400_);
return v___x_1404_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3___redArg(lean_object* v_m_1406_){
_start:
{
lean_object* v___f_1407_; lean_object* v___x_1408_; lean_object* v___x_1409_; 
v___f_1407_ = ((lean_object*)(lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3___redArg___closed__0));
v___x_1408_ = lean_box(0);
v___x_1409_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5___redArg(v_m_1406_, v___f_1407_, v___x_1408_);
return v___x_1409_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3___redArg___boxed(lean_object* v_m_1410_){
_start:
{
lean_object* v_res_1411_; 
v_res_1411_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3___redArg(v_m_1410_);
lean_dec_ref(v_m_1410_);
return v_res_1411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__4(lean_object* v_a_1412_, lean_object* v_a_1413_){
_start:
{
if (lean_obj_tag(v_a_1412_) == 0)
{
lean_object* v___x_1414_; 
v___x_1414_ = l_List_reverse___redArg(v_a_1413_);
return v___x_1414_;
}
else
{
lean_object* v_head_1415_; lean_object* v_tail_1416_; lean_object* v___x_1418_; uint8_t v_isShared_1419_; uint8_t v_isSharedCheck_1425_; 
v_head_1415_ = lean_ctor_get(v_a_1412_, 0);
v_tail_1416_ = lean_ctor_get(v_a_1412_, 1);
v_isSharedCheck_1425_ = !lean_is_exclusive(v_a_1412_);
if (v_isSharedCheck_1425_ == 0)
{
v___x_1418_ = v_a_1412_;
v_isShared_1419_ = v_isSharedCheck_1425_;
goto v_resetjp_1417_;
}
else
{
lean_inc(v_tail_1416_);
lean_inc(v_head_1415_);
lean_dec(v_a_1412_);
v___x_1418_ = lean_box(0);
v_isShared_1419_ = v_isSharedCheck_1425_;
goto v_resetjp_1417_;
}
v_resetjp_1417_:
{
lean_object* v_fst_1420_; lean_object* v___x_1422_; 
v_fst_1420_ = lean_ctor_get(v_head_1415_, 0);
lean_inc(v_fst_1420_);
lean_dec(v_head_1415_);
if (v_isShared_1419_ == 0)
{
lean_ctor_set(v___x_1418_, 1, v_a_1413_);
lean_ctor_set(v___x_1418_, 0, v_fst_1420_);
v___x_1422_ = v___x_1418_;
goto v_reusejp_1421_;
}
else
{
lean_object* v_reuseFailAlloc_1424_; 
v_reuseFailAlloc_1424_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1424_, 0, v_fst_1420_);
lean_ctor_set(v_reuseFailAlloc_1424_, 1, v_a_1413_);
v___x_1422_ = v_reuseFailAlloc_1424_;
goto v_reusejp_1421_;
}
v_reusejp_1421_:
{
v_a_1412_ = v_tail_1416_;
v_a_1413_ = v___x_1422_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1(lean_object* v_s_1426_){
_start:
{
lean_object* v___x_1427_; lean_object* v___x_1428_; lean_object* v___x_1429_; 
v___x_1427_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3___redArg(v_s_1426_);
v___x_1428_ = lean_box(0);
v___x_1429_ = lp_mathlib_List_mapTR_loop___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__4(v___x_1427_, v___x_1428_);
return v___x_1429_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1___boxed(lean_object* v_s_1430_){
_start:
{
lean_object* v_res_1431_; 
v_res_1431_ = lp_mathlib_Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1(v_s_1430_);
lean_dec_ref(v_s_1430_);
return v_res_1431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_filterMapTR_go___at___00Lean_Meta_getAllSimpDecls_spec__2(lean_object* v_a_1432_, lean_object* v_a_1433_){
_start:
{
if (lean_obj_tag(v_a_1432_) == 0)
{
lean_object* v___x_1434_; 
v___x_1434_ = lean_array_to_list(v_a_1433_);
return v___x_1434_;
}
else
{
lean_object* v_head_1435_; 
v_head_1435_ = lean_ctor_get(v_a_1432_, 0);
if (lean_obj_tag(v_head_1435_) == 0)
{
uint8_t v_post_1436_; 
v_post_1436_ = lean_ctor_get_uint8(v_head_1435_, sizeof(void*)*1);
if (v_post_1436_ == 1)
{
uint8_t v_inv_1437_; 
v_inv_1437_ = lean_ctor_get_uint8(v_head_1435_, sizeof(void*)*1 + 1);
if (v_inv_1437_ == 0)
{
lean_object* v_tail_1438_; lean_object* v_declName_1439_; lean_object* v___x_1440_; 
lean_inc_ref(v_head_1435_);
v_tail_1438_ = lean_ctor_get(v_a_1432_, 1);
lean_inc(v_tail_1438_);
lean_dec_ref_known(v_a_1432_, 2);
v_declName_1439_ = lean_ctor_get(v_head_1435_, 0);
lean_inc(v_declName_1439_);
lean_dec_ref_known(v_head_1435_, 1);
v___x_1440_ = lean_array_push(v_a_1433_, v_declName_1439_);
v_a_1432_ = v_tail_1438_;
v_a_1433_ = v___x_1440_;
goto _start;
}
else
{
lean_object* v_tail_1442_; 
v_tail_1442_ = lean_ctor_get(v_a_1432_, 1);
lean_inc(v_tail_1442_);
lean_dec_ref_known(v_a_1432_, 2);
v_a_1432_ = v_tail_1442_;
goto _start;
}
}
else
{
lean_object* v_tail_1444_; 
v_tail_1444_ = lean_ctor_get(v_a_1432_, 1);
lean_inc(v_tail_1444_);
lean_dec_ref_known(v_a_1432_, 2);
v_a_1432_ = v_tail_1444_;
goto _start;
}
}
else
{
lean_object* v_tail_1446_; 
v_tail_1446_ = lean_ctor_get(v_a_1432_, 1);
lean_inc(v_tail_1446_);
lean_dec_ref_known(v_a_1432_, 2);
v_a_1432_ = v_tail_1446_;
goto _start;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1___redArg___lam__0(lean_object* v_f_1448_, lean_object* v_x1_1449_, lean_object* v_x2_1450_, lean_object* v_x3_1451_){
_start:
{
lean_object* v___x_1452_; 
v___x_1452_ = lean_apply_3(v_f_1448_, v_x1_1449_, v_x2_1450_, v_x3_1451_);
return v___x_1452_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1___redArg(lean_object* v_map_1453_, lean_object* v_f_1454_, lean_object* v_init_1455_){
_start:
{
lean_object* v___f_1456_; lean_object* v___x_1457_; 
v___f_1456_ = lean_alloc_closure((void*)(lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1___redArg___lam__0), 4, 1);
lean_closure_set(v___f_1456_, 0, v_f_1454_);
v___x_1457_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8___redArg(v___f_1456_, v_map_1453_, v_init_1455_);
return v___x_1457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_map_1458_, lean_object* v_f_1459_, lean_object* v_init_1460_){
_start:
{
lean_object* v_res_1461_; 
v_res_1461_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1___redArg(v_map_1458_, v_f_1459_, v_init_1460_);
lean_dec_ref(v_map_1458_);
return v_res_1461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0___redArg___lam__0(lean_object* v_ps_1462_, lean_object* v_k_1463_, lean_object* v_v_1464_){
_start:
{
lean_object* v___x_1465_; lean_object* v___x_1466_; 
v___x_1465_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1465_, 0, v_k_1463_);
lean_ctor_set(v___x_1465_, 1, v_v_1464_);
v___x_1466_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1466_, 0, v___x_1465_);
lean_ctor_set(v___x_1466_, 1, v_ps_1462_);
return v___x_1466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0___redArg(lean_object* v_m_1468_){
_start:
{
lean_object* v___f_1469_; lean_object* v___x_1470_; lean_object* v___x_1471_; 
v___f_1469_ = ((lean_object*)(lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0___redArg___closed__0));
v___x_1470_ = lean_box(0);
v___x_1471_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1___redArg(v_m_1468_, v___f_1469_, v___x_1470_);
return v___x_1471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0___redArg___boxed(lean_object* v_m_1472_){
_start:
{
lean_object* v_res_1473_; 
v_res_1473_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0___redArg(v_m_1472_);
lean_dec_ref(v_m_1472_);
return v_res_1473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_mapTR_loop___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__1(lean_object* v_a_1474_, lean_object* v_a_1475_){
_start:
{
if (lean_obj_tag(v_a_1474_) == 0)
{
lean_object* v___x_1476_; 
v___x_1476_ = l_List_reverse___redArg(v_a_1475_);
return v___x_1476_;
}
else
{
lean_object* v_head_1477_; lean_object* v_tail_1478_; lean_object* v___x_1480_; uint8_t v_isShared_1481_; uint8_t v_isSharedCheck_1487_; 
v_head_1477_ = lean_ctor_get(v_a_1474_, 0);
v_tail_1478_ = lean_ctor_get(v_a_1474_, 1);
v_isSharedCheck_1487_ = !lean_is_exclusive(v_a_1474_);
if (v_isSharedCheck_1487_ == 0)
{
v___x_1480_ = v_a_1474_;
v_isShared_1481_ = v_isSharedCheck_1487_;
goto v_resetjp_1479_;
}
else
{
lean_inc(v_tail_1478_);
lean_inc(v_head_1477_);
lean_dec(v_a_1474_);
v___x_1480_ = lean_box(0);
v_isShared_1481_ = v_isSharedCheck_1487_;
goto v_resetjp_1479_;
}
v_resetjp_1479_:
{
lean_object* v_fst_1482_; lean_object* v___x_1484_; 
v_fst_1482_ = lean_ctor_get(v_head_1477_, 0);
lean_inc(v_fst_1482_);
lean_dec(v_head_1477_);
if (v_isShared_1481_ == 0)
{
lean_ctor_set(v___x_1480_, 1, v_a_1475_);
lean_ctor_set(v___x_1480_, 0, v_fst_1482_);
v___x_1484_ = v___x_1480_;
goto v_reusejp_1483_;
}
else
{
lean_object* v_reuseFailAlloc_1486_; 
v_reuseFailAlloc_1486_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1486_, 0, v_fst_1482_);
lean_ctor_set(v_reuseFailAlloc_1486_, 1, v_a_1475_);
v___x_1484_ = v_reuseFailAlloc_1486_;
goto v_reusejp_1483_;
}
v_reusejp_1483_:
{
v_a_1474_ = v_tail_1478_;
v_a_1475_ = v___x_1484_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0(lean_object* v_s_1488_){
_start:
{
lean_object* v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; 
v___x_1489_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0___redArg(v_s_1488_);
v___x_1490_ = lean_box(0);
v___x_1491_ = lp_mathlib_List_mapTR_loop___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__1(v___x_1489_, v___x_1490_);
return v___x_1491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0___boxed(lean_object* v_s_1492_){
_start:
{
lean_object* v_res_1493_; 
v_res_1493_ = lp_mathlib_Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0(v_s_1492_);
lean_dec_ref(v_s_1492_);
return v_res_1493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getAllSimpDecls(lean_object* v_simpAttr_1494_, lean_object* v_a_1495_, lean_object* v_a_1496_){
_start:
{
lean_object* v___x_1498_; 
v___x_1498_ = l_Lean_Meta_getSimpExtension_x3f(v_simpAttr_1494_, v_a_1495_, v_a_1496_);
if (lean_obj_tag(v___x_1498_) == 0)
{
lean_object* v_a_1499_; lean_object* v___x_1501_; uint8_t v_isShared_1502_; uint8_t v_isSharedCheck_1532_; 
v_a_1499_ = lean_ctor_get(v___x_1498_, 0);
v_isSharedCheck_1532_ = !lean_is_exclusive(v___x_1498_);
if (v_isSharedCheck_1532_ == 0)
{
v___x_1501_ = v___x_1498_;
v_isShared_1502_ = v_isSharedCheck_1532_;
goto v_resetjp_1500_;
}
else
{
lean_inc(v_a_1499_);
lean_dec(v___x_1498_);
v___x_1501_ = lean_box(0);
v_isShared_1502_ = v_isSharedCheck_1532_;
goto v_resetjp_1500_;
}
v_resetjp_1500_:
{
if (lean_obj_tag(v_a_1499_) == 1)
{
lean_object* v_val_1503_; lean_object* v___x_1504_; 
lean_del_object(v___x_1501_);
v_val_1503_ = lean_ctor_get(v_a_1499_, 0);
lean_inc(v_val_1503_);
lean_dec_ref_known(v_a_1499_, 1);
v___x_1504_ = l_Lean_Meta_SimpExtension_getTheorems___redArg(v_val_1503_, v_a_1496_);
lean_dec(v_val_1503_);
if (lean_obj_tag(v___x_1504_) == 0)
{
lean_object* v_a_1505_; lean_object* v___x_1507_; uint8_t v_isShared_1508_; uint8_t v_isSharedCheck_1519_; 
v_a_1505_ = lean_ctor_get(v___x_1504_, 0);
v_isSharedCheck_1519_ = !lean_is_exclusive(v___x_1504_);
if (v_isSharedCheck_1519_ == 0)
{
v___x_1507_ = v___x_1504_;
v_isShared_1508_ = v_isSharedCheck_1519_;
goto v_resetjp_1506_;
}
else
{
lean_inc(v_a_1505_);
lean_dec(v___x_1504_);
v___x_1507_ = lean_box(0);
v_isShared_1508_ = v_isSharedCheck_1519_;
goto v_resetjp_1506_;
}
v_resetjp_1506_:
{
lean_object* v_lemmaNames_1509_; lean_object* v_toUnfold_1510_; lean_object* v___x_1511_; lean_object* v___x_1512_; lean_object* v___x_1513_; lean_object* v___x_1514_; lean_object* v___x_1515_; lean_object* v___x_1517_; 
v_lemmaNames_1509_ = lean_ctor_get(v_a_1505_, 2);
lean_inc_ref(v_lemmaNames_1509_);
v_toUnfold_1510_ = lean_ctor_get(v_a_1505_, 3);
lean_inc_ref(v_toUnfold_1510_);
lean_dec(v_a_1505_);
v___x_1511_ = lp_mathlib_Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0(v_toUnfold_1510_);
lean_dec_ref(v_toUnfold_1510_);
v___x_1512_ = lp_mathlib_Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1(v_lemmaNames_1509_);
lean_dec_ref(v_lemmaNames_1509_);
v___x_1513_ = ((lean_object*)(lp_mathlib_Lean_Meta_Simp_getPropHyps___closed__0));
v___x_1514_ = lp_mathlib_List_filterMapTR_go___at___00Lean_Meta_getAllSimpDecls_spec__2(v___x_1512_, v___x_1513_);
v___x_1515_ = l_List_appendTR___redArg(v___x_1511_, v___x_1514_);
if (v_isShared_1508_ == 0)
{
lean_ctor_set(v___x_1507_, 0, v___x_1515_);
v___x_1517_ = v___x_1507_;
goto v_reusejp_1516_;
}
else
{
lean_object* v_reuseFailAlloc_1518_; 
v_reuseFailAlloc_1518_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1518_, 0, v___x_1515_);
v___x_1517_ = v_reuseFailAlloc_1518_;
goto v_reusejp_1516_;
}
v_reusejp_1516_:
{
return v___x_1517_;
}
}
}
else
{
lean_object* v_a_1520_; lean_object* v___x_1522_; uint8_t v_isShared_1523_; uint8_t v_isSharedCheck_1527_; 
v_a_1520_ = lean_ctor_get(v___x_1504_, 0);
v_isSharedCheck_1527_ = !lean_is_exclusive(v___x_1504_);
if (v_isSharedCheck_1527_ == 0)
{
v___x_1522_ = v___x_1504_;
v_isShared_1523_ = v_isSharedCheck_1527_;
goto v_resetjp_1521_;
}
else
{
lean_inc(v_a_1520_);
lean_dec(v___x_1504_);
v___x_1522_ = lean_box(0);
v_isShared_1523_ = v_isSharedCheck_1527_;
goto v_resetjp_1521_;
}
v_resetjp_1521_:
{
lean_object* v___x_1525_; 
if (v_isShared_1523_ == 0)
{
v___x_1525_ = v___x_1522_;
goto v_reusejp_1524_;
}
else
{
lean_object* v_reuseFailAlloc_1526_; 
v_reuseFailAlloc_1526_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1526_, 0, v_a_1520_);
v___x_1525_ = v_reuseFailAlloc_1526_;
goto v_reusejp_1524_;
}
v_reusejp_1524_:
{
return v___x_1525_;
}
}
}
}
else
{
lean_object* v___x_1528_; lean_object* v___x_1530_; 
lean_dec(v_a_1499_);
v___x_1528_ = lean_box(0);
if (v_isShared_1502_ == 0)
{
lean_ctor_set(v___x_1501_, 0, v___x_1528_);
v___x_1530_ = v___x_1501_;
goto v_reusejp_1529_;
}
else
{
lean_object* v_reuseFailAlloc_1531_; 
v_reuseFailAlloc_1531_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1531_, 0, v___x_1528_);
v___x_1530_ = v_reuseFailAlloc_1531_;
goto v_reusejp_1529_;
}
v_reusejp_1529_:
{
return v___x_1530_;
}
}
}
}
else
{
lean_object* v_a_1533_; lean_object* v___x_1535_; uint8_t v_isShared_1536_; uint8_t v_isSharedCheck_1540_; 
v_a_1533_ = lean_ctor_get(v___x_1498_, 0);
v_isSharedCheck_1540_ = !lean_is_exclusive(v___x_1498_);
if (v_isSharedCheck_1540_ == 0)
{
v___x_1535_ = v___x_1498_;
v_isShared_1536_ = v_isSharedCheck_1540_;
goto v_resetjp_1534_;
}
else
{
lean_inc(v_a_1533_);
lean_dec(v___x_1498_);
v___x_1535_ = lean_box(0);
v_isShared_1536_ = v_isSharedCheck_1540_;
goto v_resetjp_1534_;
}
v_resetjp_1534_:
{
lean_object* v___x_1538_; 
if (v_isShared_1536_ == 0)
{
v___x_1538_ = v___x_1535_;
goto v_reusejp_1537_;
}
else
{
lean_object* v_reuseFailAlloc_1539_; 
v_reuseFailAlloc_1539_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1539_, 0, v_a_1533_);
v___x_1538_ = v_reuseFailAlloc_1539_;
goto v_reusejp_1537_;
}
v_reusejp_1537_:
{
return v___x_1538_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getAllSimpDecls___boxed(lean_object* v_simpAttr_1541_, lean_object* v_a_1542_, lean_object* v_a_1543_, lean_object* v_a_1544_){
_start:
{
lean_object* v_res_1545_; 
v_res_1545_ = lp_mathlib_Lean_Meta_getAllSimpDecls(v_simpAttr_1541_, v_a_1542_, v_a_1543_);
lean_dec(v_a_1543_);
lean_dec_ref(v_a_1542_);
lean_dec(v_simpAttr_1541_);
return v_res_1545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0(lean_object* v_00_u03b2_1546_, lean_object* v_m_1547_){
_start:
{
lean_object* v___x_1548_; 
v___x_1548_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0___redArg(v_m_1547_);
return v___x_1548_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0___boxed(lean_object* v_00_u03b2_1549_, lean_object* v_m_1550_){
_start:
{
lean_object* v_res_1551_; 
v_res_1551_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0(v_00_u03b2_1549_, v_m_1550_);
lean_dec_ref(v_m_1550_);
return v_res_1551_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3(lean_object* v_00_u03b2_1552_, lean_object* v_m_1553_){
_start:
{
lean_object* v___x_1554_; 
v___x_1554_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3___redArg(v_m_1553_);
return v___x_1554_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3___boxed(lean_object* v_00_u03b2_1555_, lean_object* v_m_1556_){
_start:
{
lean_object* v_res_1557_; 
v_res_1557_ = lp_mathlib_Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3(v_00_u03b2_1555_, v_m_1556_);
lean_dec_ref(v_m_1556_);
return v_res_1557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1(lean_object* v_00_u03c3_1558_, lean_object* v_00_u03b2_1559_, lean_object* v_map_1560_, lean_object* v_f_1561_, lean_object* v_init_1562_){
_start:
{
lean_object* v___x_1563_; 
v___x_1563_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1___redArg(v_map_1560_, v_f_1561_, v_init_1562_);
return v___x_1563_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03c3_1564_, lean_object* v_00_u03b2_1565_, lean_object* v_map_1566_, lean_object* v_f_1567_, lean_object* v_init_1568_){
_start:
{
lean_object* v_res_1569_; 
v_res_1569_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1(v_00_u03c3_1564_, v_00_u03b2_1565_, v_map_1566_, v_f_1567_, v_init_1568_);
lean_dec_ref(v_map_1566_);
return v_res_1569_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5(lean_object* v_00_u03c3_1570_, lean_object* v_00_u03b2_1571_, lean_object* v_map_1572_, lean_object* v_f_1573_, lean_object* v_init_1574_){
_start:
{
lean_object* v___x_1575_; 
v___x_1575_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5___redArg(v_map_1572_, v_f_1573_, v_init_1574_);
return v___x_1575_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5___boxed(lean_object* v_00_u03c3_1576_, lean_object* v_00_u03b2_1577_, lean_object* v_map_1578_, lean_object* v_f_1579_, lean_object* v_init_1580_){
_start:
{
lean_object* v_res_1581_; 
v_res_1581_ = lp_mathlib_Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5(v_00_u03c3_1576_, v_00_u03b2_1577_, v_map_1578_, v_f_1579_, v_init_1580_);
lean_dec_ref(v_map_1578_);
return v_res_1581_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4___redArg(lean_object* v_map_1582_, lean_object* v_f_1583_, lean_object* v_init_1584_){
_start:
{
lean_object* v___x_1585_; 
v___x_1585_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8___redArg(v_f_1583_, v_map_1582_, v_init_1584_);
return v___x_1585_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4___redArg___boxed(lean_object* v_map_1586_, lean_object* v_f_1587_, lean_object* v_init_1588_){
_start:
{
lean_object* v_res_1589_; 
v_res_1589_ = lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4___redArg(v_map_1586_, v_f_1587_, v_init_1588_);
lean_dec_ref(v_map_1586_);
return v_res_1589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4(lean_object* v_00_u03c3_1590_, lean_object* v_00_u03b2_1591_, lean_object* v_map_1592_, lean_object* v_f_1593_, lean_object* v_init_1594_){
_start:
{
lean_object* v___x_1595_; 
v___x_1595_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8___redArg(v_f_1593_, v_map_1592_, v_init_1594_);
return v___x_1595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4___boxed(lean_object* v_00_u03c3_1596_, lean_object* v_00_u03b2_1597_, lean_object* v_map_1598_, lean_object* v_f_1599_, lean_object* v_init_1600_){
_start:
{
lean_object* v_res_1601_; 
v_res_1601_ = lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4(v_00_u03c3_1596_, v_00_u03b2_1597_, v_map_1598_, v_f_1599_, v_init_1600_);
lean_dec_ref(v_map_1598_);
return v_res_1601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5_spec__8___redArg(lean_object* v_map_1602_, lean_object* v_f_1603_, lean_object* v_init_1604_){
_start:
{
lean_object* v___x_1605_; 
v___x_1605_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8___redArg(v_f_1603_, v_map_1602_, v_init_1604_);
return v___x_1605_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5_spec__8___redArg___boxed(lean_object* v_map_1606_, lean_object* v_f_1607_, lean_object* v_init_1608_){
_start:
{
lean_object* v_res_1609_; 
v_res_1609_ = lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5_spec__8___redArg(v_map_1606_, v_f_1607_, v_init_1608_);
lean_dec_ref(v_map_1606_);
return v_res_1609_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5_spec__8(lean_object* v_00_u03c3_1610_, lean_object* v_00_u03b2_1611_, lean_object* v_map_1612_, lean_object* v_f_1613_, lean_object* v_init_1614_){
_start:
{
lean_object* v___x_1615_; 
v___x_1615_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8___redArg(v_f_1613_, v_map_1612_, v_init_1614_);
return v___x_1615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5_spec__8___boxed(lean_object* v_00_u03c3_1616_, lean_object* v_00_u03b2_1617_, lean_object* v_map_1618_, lean_object* v_f_1619_, lean_object* v_init_1620_){
_start:
{
lean_object* v_res_1621_; 
v_res_1621_ = lp_mathlib_Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__1_spec__3_spec__5_spec__8(v_00_u03c3_1616_, v_00_u03b2_1617_, v_map_1618_, v_f_1619_, v_init_1620_);
lean_dec_ref(v_map_1618_);
return v_res_1621_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8(lean_object* v_00_u03c3_1622_, lean_object* v_00_u03b1_1623_, lean_object* v_00_u03b2_1624_, lean_object* v_f_1625_, lean_object* v_x_1626_, lean_object* v_x_1627_){
_start:
{
lean_object* v___x_1628_; 
v___x_1628_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8___redArg(v_f_1625_, v_x_1626_, v_x_1627_);
return v___x_1628_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8___boxed(lean_object* v_00_u03c3_1629_, lean_object* v_00_u03b1_1630_, lean_object* v_00_u03b2_1631_, lean_object* v_f_1632_, lean_object* v_x_1633_, lean_object* v_x_1634_){
_start:
{
lean_object* v_res_1635_; 
v_res_1635_ = lp_mathlib_Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8(v_00_u03c3_1629_, v_00_u03b1_1630_, v_00_u03b2_1631_, v_f_1632_, v_x_1633_, v_x_1634_);
lean_dec_ref(v_x_1633_);
return v_res_1635_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__10(lean_object* v_00_u03b1_1636_, lean_object* v_00_u03b2_1637_, lean_object* v_00_u03c3_1638_, lean_object* v_f_1639_, lean_object* v_as_1640_, size_t v_i_1641_, size_t v_stop_1642_, lean_object* v_b_1643_){
_start:
{
lean_object* v___x_1644_; 
v___x_1644_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__10___redArg(v_f_1639_, v_as_1640_, v_i_1641_, v_stop_1642_, v_b_1643_);
return v___x_1644_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__10___boxed(lean_object* v_00_u03b1_1645_, lean_object* v_00_u03b2_1646_, lean_object* v_00_u03c3_1647_, lean_object* v_f_1648_, lean_object* v_as_1649_, lean_object* v_i_1650_, lean_object* v_stop_1651_, lean_object* v_b_1652_){
_start:
{
size_t v_i_boxed_1653_; size_t v_stop_boxed_1654_; lean_object* v_res_1655_; 
v_i_boxed_1653_ = lean_unbox_usize(v_i_1650_);
lean_dec(v_i_1650_);
v_stop_boxed_1654_ = lean_unbox_usize(v_stop_1651_);
lean_dec(v_stop_1651_);
v_res_1655_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__10(v_00_u03b1_1645_, v_00_u03b2_1646_, v_00_u03c3_1647_, v_f_1648_, v_as_1649_, v_i_boxed_1653_, v_stop_boxed_1654_, v_b_1652_);
lean_dec_ref(v_as_1649_);
return v_res_1655_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__11(lean_object* v_00_u03c3_1656_, lean_object* v_00_u03b1_1657_, lean_object* v_00_u03b2_1658_, lean_object* v_f_1659_, lean_object* v_keys_1660_, lean_object* v_vals_1661_, lean_object* v_heq_1662_, lean_object* v_i_1663_, lean_object* v_acc_1664_){
_start:
{
lean_object* v___x_1665_; 
v___x_1665_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___redArg(v_f_1659_, v_keys_1660_, v_vals_1661_, v_i_1663_, v_acc_1664_);
return v___x_1665_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__11___boxed(lean_object* v_00_u03c3_1666_, lean_object* v_00_u03b1_1667_, lean_object* v_00_u03b2_1668_, lean_object* v_f_1669_, lean_object* v_keys_1670_, lean_object* v_vals_1671_, lean_object* v_heq_1672_, lean_object* v_i_1673_, lean_object* v_acc_1674_){
_start:
{
lean_object* v_res_1675_; 
v_res_1675_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_foldlMAux_traverse___at___00Lean_PersistentHashMap_foldlMAux___at___00Lean_PersistentHashMap_foldlM___at___00Lean_PersistentHashMap_foldl___at___00Lean_PersistentHashMap_toList___at___00Lean_PHashSet_toList___at___00Lean_Meta_getAllSimpDecls_spec__0_spec__0_spec__1_spec__4_spec__8_spec__11(v_00_u03c3_1666_, v_00_u03b1_1667_, v_00_u03b2_1668_, v_f_1669_, v_keys_1670_, v_vals_1671_, v_heq_1672_, v_i_1673_, v_acc_1674_);
lean_dec_ref(v_vals_1671_);
lean_dec_ref(v_keys_1670_);
return v_res_1675_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Lean_Meta_getAllSimpAttrs_spec__0___redArg(lean_object* v_decl_1676_, lean_object* v_a_1677_, lean_object* v_a_1678_, lean_object* v___y_1679_){
_start:
{
if (lean_obj_tag(v_a_1677_) == 0)
{
lean_object* v___x_1681_; lean_object* v___x_1682_; 
lean_dec(v_decl_1676_);
v___x_1681_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1681_, 0, v_a_1678_);
v___x_1682_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1682_, 0, v___x_1681_);
return v___x_1682_;
}
else
{
lean_object* v_key_1683_; lean_object* v_value_1684_; lean_object* v_tail_1685_; lean_object* v___x_1686_; 
v_key_1683_ = lean_ctor_get(v_a_1677_, 0);
lean_inc(v_key_1683_);
v_value_1684_ = lean_ctor_get(v_a_1677_, 1);
lean_inc(v_value_1684_);
v_tail_1685_ = lean_ctor_get(v_a_1677_, 2);
lean_inc(v_tail_1685_);
lean_dec_ref_known(v_a_1677_, 3);
v___x_1686_ = l_Lean_Meta_SimpExtension_getTheorems___redArg(v_value_1684_, v___y_1679_);
lean_dec(v_value_1684_);
if (lean_obj_tag(v___x_1686_) == 0)
{
lean_object* v_a_1687_; uint8_t v___x_1688_; 
v_a_1687_ = lean_ctor_get(v___x_1686_, 0);
lean_inc(v_a_1687_);
lean_dec_ref_known(v___x_1686_, 1);
lean_inc(v_decl_1676_);
v___x_1688_ = lp_mathlib_Lean_Meta_SimpTheorems_contains(v_a_1687_, v_decl_1676_);
lean_dec(v_a_1687_);
if (v___x_1688_ == 0)
{
lean_dec(v_key_1683_);
v_a_1677_ = v_tail_1685_;
goto _start;
}
else
{
lean_object* v___x_1690_; 
v___x_1690_ = lean_array_push(v_a_1678_, v_key_1683_);
v_a_1677_ = v_tail_1685_;
v_a_1678_ = v___x_1690_;
goto _start;
}
}
else
{
lean_object* v_a_1692_; lean_object* v___x_1694_; uint8_t v_isShared_1695_; uint8_t v_isSharedCheck_1699_; 
lean_dec(v_tail_1685_);
lean_dec(v_key_1683_);
lean_dec_ref(v_a_1678_);
lean_dec(v_decl_1676_);
v_a_1692_ = lean_ctor_get(v___x_1686_, 0);
v_isSharedCheck_1699_ = !lean_is_exclusive(v___x_1686_);
if (v_isSharedCheck_1699_ == 0)
{
v___x_1694_ = v___x_1686_;
v_isShared_1695_ = v_isSharedCheck_1699_;
goto v_resetjp_1693_;
}
else
{
lean_inc(v_a_1692_);
lean_dec(v___x_1686_);
v___x_1694_ = lean_box(0);
v_isShared_1695_ = v_isSharedCheck_1699_;
goto v_resetjp_1693_;
}
v_resetjp_1693_:
{
lean_object* v___x_1697_; 
if (v_isShared_1695_ == 0)
{
v___x_1697_ = v___x_1694_;
goto v_reusejp_1696_;
}
else
{
lean_object* v_reuseFailAlloc_1698_; 
v_reuseFailAlloc_1698_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1698_, 0, v_a_1692_);
v___x_1697_ = v_reuseFailAlloc_1698_;
goto v_reusejp_1696_;
}
v_reusejp_1696_:
{
return v___x_1697_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Lean_Meta_getAllSimpAttrs_spec__0___redArg___boxed(lean_object* v_decl_1700_, lean_object* v_a_1701_, lean_object* v_a_1702_, lean_object* v___y_1703_, lean_object* v___y_1704_){
_start:
{
lean_object* v_res_1705_; 
v_res_1705_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Lean_Meta_getAllSimpAttrs_spec__0___redArg(v_decl_1700_, v_a_1701_, v_a_1702_, v___y_1703_);
lean_dec(v___y_1703_);
return v_res_1705_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_getAllSimpAttrs_spec__1(lean_object* v_decl_1706_, lean_object* v_as_1707_, size_t v_sz_1708_, size_t v_i_1709_, lean_object* v_b_1710_, lean_object* v___y_1711_, lean_object* v___y_1712_){
_start:
{
uint8_t v___x_1714_; 
v___x_1714_ = lean_usize_dec_lt(v_i_1709_, v_sz_1708_);
if (v___x_1714_ == 0)
{
lean_object* v___x_1715_; 
lean_dec(v_decl_1706_);
v___x_1715_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1715_, 0, v_b_1710_);
return v___x_1715_;
}
else
{
lean_object* v_a_1716_; lean_object* v___x_1717_; 
v_a_1716_ = lean_array_uget_borrowed(v_as_1707_, v_i_1709_);
lean_inc(v_a_1716_);
lean_inc(v_decl_1706_);
v___x_1717_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Lean_Meta_getAllSimpAttrs_spec__0___redArg(v_decl_1706_, v_a_1716_, v_b_1710_, v___y_1712_);
if (lean_obj_tag(v___x_1717_) == 0)
{
lean_object* v_a_1718_; lean_object* v___x_1720_; uint8_t v_isShared_1721_; uint8_t v_isSharedCheck_1730_; 
v_a_1718_ = lean_ctor_get(v___x_1717_, 0);
v_isSharedCheck_1730_ = !lean_is_exclusive(v___x_1717_);
if (v_isSharedCheck_1730_ == 0)
{
v___x_1720_ = v___x_1717_;
v_isShared_1721_ = v_isSharedCheck_1730_;
goto v_resetjp_1719_;
}
else
{
lean_inc(v_a_1718_);
lean_dec(v___x_1717_);
v___x_1720_ = lean_box(0);
v_isShared_1721_ = v_isSharedCheck_1730_;
goto v_resetjp_1719_;
}
v_resetjp_1719_:
{
if (lean_obj_tag(v_a_1718_) == 0)
{
lean_object* v_a_1722_; lean_object* v___x_1724_; 
lean_dec(v_decl_1706_);
v_a_1722_ = lean_ctor_get(v_a_1718_, 0);
lean_inc(v_a_1722_);
lean_dec_ref_known(v_a_1718_, 1);
if (v_isShared_1721_ == 0)
{
lean_ctor_set(v___x_1720_, 0, v_a_1722_);
v___x_1724_ = v___x_1720_;
goto v_reusejp_1723_;
}
else
{
lean_object* v_reuseFailAlloc_1725_; 
v_reuseFailAlloc_1725_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1725_, 0, v_a_1722_);
v___x_1724_ = v_reuseFailAlloc_1725_;
goto v_reusejp_1723_;
}
v_reusejp_1723_:
{
return v___x_1724_;
}
}
else
{
lean_object* v_a_1726_; size_t v___x_1727_; size_t v___x_1728_; 
lean_del_object(v___x_1720_);
v_a_1726_ = lean_ctor_get(v_a_1718_, 0);
lean_inc(v_a_1726_);
lean_dec_ref_known(v_a_1718_, 1);
v___x_1727_ = ((size_t)1ULL);
v___x_1728_ = lean_usize_add(v_i_1709_, v___x_1727_);
v_i_1709_ = v___x_1728_;
v_b_1710_ = v_a_1726_;
goto _start;
}
}
}
else
{
lean_object* v_a_1731_; lean_object* v___x_1733_; uint8_t v_isShared_1734_; uint8_t v_isSharedCheck_1738_; 
lean_dec(v_decl_1706_);
v_a_1731_ = lean_ctor_get(v___x_1717_, 0);
v_isSharedCheck_1738_ = !lean_is_exclusive(v___x_1717_);
if (v_isSharedCheck_1738_ == 0)
{
v___x_1733_ = v___x_1717_;
v_isShared_1734_ = v_isSharedCheck_1738_;
goto v_resetjp_1732_;
}
else
{
lean_inc(v_a_1731_);
lean_dec(v___x_1717_);
v___x_1733_ = lean_box(0);
v_isShared_1734_ = v_isSharedCheck_1738_;
goto v_resetjp_1732_;
}
v_resetjp_1732_:
{
lean_object* v___x_1736_; 
if (v_isShared_1734_ == 0)
{
v___x_1736_ = v___x_1733_;
goto v_reusejp_1735_;
}
else
{
lean_object* v_reuseFailAlloc_1737_; 
v_reuseFailAlloc_1737_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1737_, 0, v_a_1731_);
v___x_1736_ = v_reuseFailAlloc_1737_;
goto v_reusejp_1735_;
}
v_reusejp_1735_:
{
return v___x_1736_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_getAllSimpAttrs_spec__1___boxed(lean_object* v_decl_1739_, lean_object* v_as_1740_, lean_object* v_sz_1741_, lean_object* v_i_1742_, lean_object* v_b_1743_, lean_object* v___y_1744_, lean_object* v___y_1745_, lean_object* v___y_1746_){
_start:
{
size_t v_sz_boxed_1747_; size_t v_i_boxed_1748_; lean_object* v_res_1749_; 
v_sz_boxed_1747_ = lean_unbox_usize(v_sz_1741_);
lean_dec(v_sz_1741_);
v_i_boxed_1748_ = lean_unbox_usize(v_i_1742_);
lean_dec(v_i_1742_);
v_res_1749_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_getAllSimpAttrs_spec__1(v_decl_1739_, v_as_1740_, v_sz_boxed_1747_, v_i_boxed_1748_, v_b_1743_, v___y_1744_, v___y_1745_);
lean_dec(v___y_1745_);
lean_dec_ref(v___y_1744_);
lean_dec_ref(v_as_1740_);
return v_res_1749_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getAllSimpAttrs(lean_object* v_decl_1750_, lean_object* v_a_1751_, lean_object* v_a_1752_){
_start:
{
lean_object* v___x_1754_; lean_object* v___x_1755_; lean_object* v_buckets_1756_; lean_object* v_simpAttrs_1757_; size_t v_sz_1758_; size_t v___x_1759_; lean_object* v___x_1760_; 
v___x_1754_ = l_Lean_Meta_simpExtensionMapRef;
v___x_1755_ = lean_st_ref_get(v___x_1754_);
v_buckets_1756_ = lean_ctor_get(v___x_1755_, 1);
lean_inc_ref(v_buckets_1756_);
lean_dec(v___x_1755_);
v_simpAttrs_1757_ = ((lean_object*)(lp_mathlib_Lean_Meta_Simp_getPropHyps___closed__0));
v_sz_1758_ = lean_array_size(v_buckets_1756_);
v___x_1759_ = ((size_t)0ULL);
v___x_1760_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_getAllSimpAttrs_spec__1(v_decl_1750_, v_buckets_1756_, v_sz_1758_, v___x_1759_, v_simpAttrs_1757_, v_a_1751_, v_a_1752_);
lean_dec_ref(v_buckets_1756_);
return v___x_1760_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_getAllSimpAttrs___boxed(lean_object* v_decl_1761_, lean_object* v_a_1762_, lean_object* v_a_1763_, lean_object* v_a_1764_){
_start:
{
lean_object* v_res_1765_; 
v_res_1765_ = lp_mathlib_Lean_Meta_getAllSimpAttrs(v_decl_1761_, v_a_1762_, v_a_1763_);
lean_dec(v_a_1763_);
lean_dec_ref(v_a_1762_);
return v_res_1765_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Lean_Meta_getAllSimpAttrs_spec__0(lean_object* v_decl_1766_, lean_object* v_a_1767_, lean_object* v_a_1768_, lean_object* v___y_1769_, lean_object* v___y_1770_){
_start:
{
lean_object* v___x_1772_; 
v___x_1772_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Lean_Meta_getAllSimpAttrs_spec__0___redArg(v_decl_1766_, v_a_1767_, v_a_1768_, v___y_1770_);
return v___x_1772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Lean_Meta_getAllSimpAttrs_spec__0___boxed(lean_object* v_decl_1773_, lean_object* v_a_1774_, lean_object* v_a_1775_, lean_object* v___y_1776_, lean_object* v___y_1777_, lean_object* v___y_1778_){
_start:
{
lean_object* v_res_1779_; 
v_res_1779_ = lp_mathlib___private_Std_Data_DHashMap_Internal_AssocList_Basic_0__Std_DHashMap_Internal_AssocList_forInStep_go___at___00Lean_Meta_getAllSimpAttrs_spec__0(v_decl_1773_, v_a_1774_, v_a_1775_, v___y_1776_, v___y_1777_);
lean_dec(v___y_1777_);
lean_dec_ref(v___y_1776_);
return v_res_1779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_SMap_switch___at___00Lean_Meta_Simp_withoutTheorems_spec__1___redArg(lean_object* v_m_1780_){
_start:
{
uint8_t v_stage_u2081_1781_; 
v_stage_u2081_1781_ = lean_ctor_get_uint8(v_m_1780_, sizeof(void*)*2);
if (v_stage_u2081_1781_ == 0)
{
return v_m_1780_;
}
else
{
lean_object* v_map_u2081_1782_; lean_object* v_map_u2082_1783_; lean_object* v___x_1785_; uint8_t v_isShared_1786_; uint8_t v_isSharedCheck_1791_; 
v_map_u2081_1782_ = lean_ctor_get(v_m_1780_, 0);
v_map_u2082_1783_ = lean_ctor_get(v_m_1780_, 1);
v_isSharedCheck_1791_ = !lean_is_exclusive(v_m_1780_);
if (v_isSharedCheck_1791_ == 0)
{
v___x_1785_ = v_m_1780_;
v_isShared_1786_ = v_isSharedCheck_1791_;
goto v_resetjp_1784_;
}
else
{
lean_inc(v_map_u2082_1783_);
lean_inc(v_map_u2081_1782_);
lean_dec(v_m_1780_);
v___x_1785_ = lean_box(0);
v_isShared_1786_ = v_isSharedCheck_1791_;
goto v_resetjp_1784_;
}
v_resetjp_1784_:
{
uint8_t v___x_1787_; lean_object* v___x_1789_; 
v___x_1787_ = 0;
if (v_isShared_1786_ == 0)
{
v___x_1789_ = v___x_1785_;
goto v_reusejp_1788_;
}
else
{
lean_object* v_reuseFailAlloc_1790_; 
v_reuseFailAlloc_1790_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v_reuseFailAlloc_1790_, 0, v_map_u2081_1782_);
lean_ctor_set(v_reuseFailAlloc_1790_, 1, v_map_u2082_1783_);
v___x_1789_ = v_reuseFailAlloc_1790_;
goto v_reusejp_1788_;
}
v_reusejp_1788_:
{
lean_ctor_set_uint8(v___x_1789_, sizeof(void*)*2, v___x_1787_);
return v___x_1789_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_SMap_switch___at___00Lean_Meta_Simp_withoutTheorems_spec__1(lean_object* v_00_u03b2_1792_, lean_object* v_m_1793_){
_start:
{
lean_object* v___x_1794_; 
v___x_1794_ = lp_mathlib_Lean_SMap_switch___at___00Lean_Meta_Simp_withoutTheorems_spec__1___redArg(v_m_1793_);
return v___x_1794_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_withoutTheorems___redArg___lam__0(lean_object* v_a_1795_, uint8_t v_stage_u2081_1796_, lean_object* v_map_u2082_1797_, lean_object* v_a_x3f_1798_){
_start:
{
lean_object* v___x_1800_; lean_object* v_cache_1801_; lean_object* v_congrCache_1802_; lean_object* v_dsimpCache_1803_; lean_object* v_usedTheorems_1804_; lean_object* v_numSteps_1805_; lean_object* v_diag_1806_; lean_object* v___x_1808_; uint8_t v_isShared_1809_; uint8_t v_isSharedCheck_1825_; 
v___x_1800_ = lean_st_ref_take(v_a_1795_);
v_cache_1801_ = lean_ctor_get(v___x_1800_, 0);
v_congrCache_1802_ = lean_ctor_get(v___x_1800_, 1);
v_dsimpCache_1803_ = lean_ctor_get(v___x_1800_, 2);
v_usedTheorems_1804_ = lean_ctor_get(v___x_1800_, 3);
v_numSteps_1805_ = lean_ctor_get(v___x_1800_, 4);
v_diag_1806_ = lean_ctor_get(v___x_1800_, 5);
v_isSharedCheck_1825_ = !lean_is_exclusive(v___x_1800_);
if (v_isSharedCheck_1825_ == 0)
{
v___x_1808_ = v___x_1800_;
v_isShared_1809_ = v_isSharedCheck_1825_;
goto v_resetjp_1807_;
}
else
{
lean_inc(v_diag_1806_);
lean_inc(v_numSteps_1805_);
lean_inc(v_usedTheorems_1804_);
lean_inc(v_dsimpCache_1803_);
lean_inc(v_congrCache_1802_);
lean_inc(v_cache_1801_);
lean_dec(v___x_1800_);
v___x_1808_ = lean_box(0);
v_isShared_1809_ = v_isSharedCheck_1825_;
goto v_resetjp_1807_;
}
v_resetjp_1807_:
{
lean_object* v_map_u2081_1810_; lean_object* v___x_1812_; uint8_t v_isShared_1813_; uint8_t v_isSharedCheck_1823_; 
v_map_u2081_1810_ = lean_ctor_get(v_cache_1801_, 0);
v_isSharedCheck_1823_ = !lean_is_exclusive(v_cache_1801_);
if (v_isSharedCheck_1823_ == 0)
{
lean_object* v_unused_1824_; 
v_unused_1824_ = lean_ctor_get(v_cache_1801_, 1);
lean_dec(v_unused_1824_);
v___x_1812_ = v_cache_1801_;
v_isShared_1813_ = v_isSharedCheck_1823_;
goto v_resetjp_1811_;
}
else
{
lean_inc(v_map_u2081_1810_);
lean_dec(v_cache_1801_);
v___x_1812_ = lean_box(0);
v_isShared_1813_ = v_isSharedCheck_1823_;
goto v_resetjp_1811_;
}
v_resetjp_1811_:
{
lean_object* v___x_1815_; 
if (v_isShared_1813_ == 0)
{
lean_ctor_set(v___x_1812_, 1, v_map_u2082_1797_);
v___x_1815_ = v___x_1812_;
goto v_reusejp_1814_;
}
else
{
lean_object* v_reuseFailAlloc_1822_; 
v_reuseFailAlloc_1822_ = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(v_reuseFailAlloc_1822_, 0, v_map_u2081_1810_);
lean_ctor_set(v_reuseFailAlloc_1822_, 1, v_map_u2082_1797_);
v___x_1815_ = v_reuseFailAlloc_1822_;
goto v_reusejp_1814_;
}
v_reusejp_1814_:
{
lean_object* v___x_1817_; 
lean_ctor_set_uint8(v___x_1815_, sizeof(void*)*2, v_stage_u2081_1796_);
if (v_isShared_1809_ == 0)
{
lean_ctor_set(v___x_1808_, 0, v___x_1815_);
v___x_1817_ = v___x_1808_;
goto v_reusejp_1816_;
}
else
{
lean_object* v_reuseFailAlloc_1821_; 
v_reuseFailAlloc_1821_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_1821_, 0, v___x_1815_);
lean_ctor_set(v_reuseFailAlloc_1821_, 1, v_congrCache_1802_);
lean_ctor_set(v_reuseFailAlloc_1821_, 2, v_dsimpCache_1803_);
lean_ctor_set(v_reuseFailAlloc_1821_, 3, v_usedTheorems_1804_);
lean_ctor_set(v_reuseFailAlloc_1821_, 4, v_numSteps_1805_);
lean_ctor_set(v_reuseFailAlloc_1821_, 5, v_diag_1806_);
v___x_1817_ = v_reuseFailAlloc_1821_;
goto v_reusejp_1816_;
}
v_reusejp_1816_:
{
lean_object* v___x_1818_; lean_object* v___x_1819_; lean_object* v___x_1820_; 
v___x_1818_ = lean_st_ref_set(v_a_1795_, v___x_1817_);
v___x_1819_ = lean_box(0);
v___x_1820_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1820_, 0, v___x_1819_);
return v___x_1820_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_withoutTheorems___redArg___lam__0___boxed(lean_object* v_a_1826_, lean_object* v_stage_u2081_1827_, lean_object* v_map_u2082_1828_, lean_object* v_a_x3f_1829_, lean_object* v___y_1830_){
_start:
{
uint8_t v_stage_u2081_boxed_1831_; lean_object* v_res_1832_; 
v_stage_u2081_boxed_1831_ = lean_unbox(v_stage_u2081_1827_);
v_res_1832_ = lp_mathlib_Lean_Meta_Simp_withoutTheorems___redArg___lam__0(v_a_1826_, v_stage_u2081_boxed_1831_, v_map_u2082_1828_, v_a_x3f_1829_);
lean_dec(v_a_x3f_1829_);
lean_dec(v_a_1826_);
return v_res_1832_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_Simp_withoutTheorems_spec__0___redArg(lean_object* v_as_1833_, size_t v_sz_1834_, size_t v_i_1835_, lean_object* v_b_1836_){
_start:
{
uint8_t v___x_1838_; 
v___x_1838_ = lean_usize_dec_lt(v_i_1835_, v_sz_1834_);
if (v___x_1838_ == 0)
{
lean_object* v___x_1839_; 
v___x_1839_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_1839_, 0, v_b_1836_);
return v___x_1839_;
}
else
{
lean_object* v_fst_1840_; lean_object* v_snd_1841_; lean_object* v___x_1843_; uint8_t v_isShared_1844_; uint8_t v_isSharedCheck_1862_; 
v_fst_1840_ = lean_ctor_get(v_b_1836_, 0);
v_snd_1841_ = lean_ctor_get(v_b_1836_, 1);
v_isSharedCheck_1862_ = !lean_is_exclusive(v_b_1836_);
if (v_isSharedCheck_1862_ == 0)
{
v___x_1843_ = v_b_1836_;
v_isShared_1844_ = v_isSharedCheck_1862_;
goto v_resetjp_1842_;
}
else
{
lean_inc(v_snd_1841_);
lean_inc(v_fst_1840_);
lean_dec(v_b_1836_);
v___x_1843_ = lean_box(0);
v_isShared_1844_ = v_isSharedCheck_1862_;
goto v_resetjp_1842_;
}
v_resetjp_1842_:
{
lean_object* v_a_1845_; uint8_t v___x_1846_; lean_object* v___x_1847_; lean_object* v___x_1848_; lean_object* v___x_1849_; lean_object* v___x_1850_; lean_object* v___x_1851_; lean_object* v___x_1852_; lean_object* v___x_1853_; lean_object* v___x_1854_; lean_object* v___x_1855_; lean_object* v___x_1857_; 
v_a_1845_ = lean_array_uget_borrowed(v_as_1833_, v_i_1835_);
v___x_1846_ = 0;
lean_inc_n(v_a_1845_, 5);
v___x_1847_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_1847_, 0, v_a_1845_);
lean_ctor_set_uint8(v___x_1847_, sizeof(void*)*1, v___x_1838_);
lean_ctor_set_uint8(v___x_1847_, sizeof(void*)*1 + 1, v___x_1846_);
v___x_1848_ = l_Lean_Meta_SimpTheoremsArray_eraseTheorem(v_fst_1840_, v___x_1847_);
v___x_1849_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_1849_, 0, v_a_1845_);
lean_ctor_set_uint8(v___x_1849_, sizeof(void*)*1, v___x_1838_);
lean_ctor_set_uint8(v___x_1849_, sizeof(void*)*1 + 1, v___x_1838_);
v___x_1850_ = l_Lean_Meta_SimpTheoremsArray_eraseTheorem(v___x_1848_, v___x_1849_);
v___x_1851_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_1851_, 0, v_a_1845_);
lean_ctor_set_uint8(v___x_1851_, sizeof(void*)*1, v___x_1846_);
lean_ctor_set_uint8(v___x_1851_, sizeof(void*)*1 + 1, v___x_1846_);
v___x_1852_ = l_Lean_Meta_SimpTheoremsArray_eraseTheorem(v___x_1850_, v___x_1851_);
v___x_1853_ = lean_alloc_ctor(0, 1, 2);
lean_ctor_set(v___x_1853_, 0, v_a_1845_);
lean_ctor_set_uint8(v___x_1853_, sizeof(void*)*1, v___x_1846_);
lean_ctor_set_uint8(v___x_1853_, sizeof(void*)*1 + 1, v___x_1838_);
v___x_1854_ = l_Lean_Meta_SimpTheoremsArray_eraseTheorem(v___x_1852_, v___x_1853_);
v___x_1855_ = l_Lean_Meta_Simp_Simprocs_erase(v_snd_1841_, v_a_1845_);
if (v_isShared_1844_ == 0)
{
lean_ctor_set(v___x_1843_, 1, v___x_1855_);
lean_ctor_set(v___x_1843_, 0, v___x_1854_);
v___x_1857_ = v___x_1843_;
goto v_reusejp_1856_;
}
else
{
lean_object* v_reuseFailAlloc_1861_; 
v_reuseFailAlloc_1861_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1861_, 0, v___x_1854_);
lean_ctor_set(v_reuseFailAlloc_1861_, 1, v___x_1855_);
v___x_1857_ = v_reuseFailAlloc_1861_;
goto v_reusejp_1856_;
}
v_reusejp_1856_:
{
size_t v___x_1858_; size_t v___x_1859_; 
v___x_1858_ = ((size_t)1ULL);
v___x_1859_ = lean_usize_add(v_i_1835_, v___x_1858_);
v_i_1835_ = v___x_1859_;
v_b_1836_ = v___x_1857_;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_Simp_withoutTheorems_spec__0___redArg___boxed(lean_object* v_as_1863_, lean_object* v_sz_1864_, lean_object* v_i_1865_, lean_object* v_b_1866_, lean_object* v___y_1867_){
_start:
{
size_t v_sz_boxed_1868_; size_t v_i_boxed_1869_; lean_object* v_res_1870_; 
v_sz_boxed_1868_ = lean_unbox_usize(v_sz_1864_);
lean_dec(v_sz_1864_);
v_i_boxed_1869_ = lean_unbox_usize(v_i_1865_);
lean_dec(v_i_1865_);
v_res_1870_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_Simp_withoutTheorems_spec__0___redArg(v_as_1863_, v_sz_boxed_1868_, v_i_boxed_1869_, v_b_1866_);
lean_dec_ref(v_as_1863_);
return v_res_1870_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_withoutTheorems___redArg(lean_object* v_declNames_1871_, lean_object* v_e_1872_, lean_object* v_a_1873_, lean_object* v_a_1874_, lean_object* v_a_1875_, lean_object* v_a_1876_, lean_object* v_a_1877_, lean_object* v_a_1878_, lean_object* v_a_1879_){
_start:
{
lean_object* v___x_1881_; 
v___x_1881_ = l_Lean_Meta_Simp_getSimprocs___redArg(v_a_1879_);
if (lean_obj_tag(v___x_1881_) == 0)
{
lean_object* v_a_1882_; lean_object* v_config_1883_; lean_object* v_userConfig_1884_; lean_object* v_zetaDeltaSet_1885_; lean_object* v_initUsedZetaDelta_1886_; lean_object* v_metaConfig_1887_; lean_object* v_indexConfig_1888_; uint32_t v_maxDischargeDepth_1889_; lean_object* v_simpTheorems_1890_; lean_object* v_congrTheorems_1891_; lean_object* v_parent_x3f_1892_; uint32_t v_dischargeDepth_1893_; lean_object* v_lctxInitIndices_1894_; uint8_t v_inDSimp_1895_; lean_object* v___x_1896_; size_t v_sz_1897_; size_t v___x_1898_; lean_object* v___x_1899_; 
v_a_1882_ = lean_ctor_get(v___x_1881_, 0);
lean_inc(v_a_1882_);
lean_dec_ref_known(v___x_1881_, 1);
v_config_1883_ = lean_ctor_get(v_a_1874_, 0);
v_userConfig_1884_ = lean_ctor_get(v_a_1874_, 1);
v_zetaDeltaSet_1885_ = lean_ctor_get(v_a_1874_, 2);
v_initUsedZetaDelta_1886_ = lean_ctor_get(v_a_1874_, 3);
v_metaConfig_1887_ = lean_ctor_get(v_a_1874_, 4);
v_indexConfig_1888_ = lean_ctor_get(v_a_1874_, 5);
v_maxDischargeDepth_1889_ = lean_ctor_get_uint32(v_a_1874_, sizeof(void*)*10);
v_simpTheorems_1890_ = lean_ctor_get(v_a_1874_, 6);
v_congrTheorems_1891_ = lean_ctor_get(v_a_1874_, 7);
v_parent_x3f_1892_ = lean_ctor_get(v_a_1874_, 8);
v_dischargeDepth_1893_ = lean_ctor_get_uint32(v_a_1874_, sizeof(void*)*10 + 4);
v_lctxInitIndices_1894_ = lean_ctor_get(v_a_1874_, 9);
v_inDSimp_1895_ = lean_ctor_get_uint8(v_a_1874_, sizeof(void*)*10 + 8);
lean_inc_ref(v_simpTheorems_1890_);
v___x_1896_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1896_, 0, v_simpTheorems_1890_);
lean_ctor_set(v___x_1896_, 1, v_a_1882_);
v_sz_1897_ = lean_array_size(v_declNames_1871_);
v___x_1898_ = ((size_t)0ULL);
v___x_1899_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_Simp_withoutTheorems_spec__0___redArg(v_declNames_1871_, v_sz_1897_, v___x_1898_, v___x_1896_);
if (lean_obj_tag(v___x_1899_) == 0)
{
lean_object* v_a_1900_; lean_object* v___x_1901_; lean_object* v___x_1902_; lean_object* v___x_1903_; lean_object* v_cache_1904_; lean_object* v_congrCache_1905_; lean_object* v_dsimpCache_1906_; lean_object* v_usedTheorems_1907_; lean_object* v_numSteps_1908_; lean_object* v_diag_1909_; lean_object* v___x_1911_; uint8_t v_isShared_1912_; uint8_t v_isSharedCheck_1964_; 
v_a_1900_ = lean_ctor_get(v___x_1899_, 0);
lean_inc(v_a_1900_);
lean_dec_ref_known(v___x_1899_, 1);
v___x_1901_ = lean_st_ref_get(v_a_1875_);
v___x_1902_ = lean_st_ref_get(v_a_1875_);
v___x_1903_ = lean_st_ref_take(v_a_1875_);
v_cache_1904_ = lean_ctor_get(v___x_1903_, 0);
v_congrCache_1905_ = lean_ctor_get(v___x_1903_, 1);
v_dsimpCache_1906_ = lean_ctor_get(v___x_1903_, 2);
v_usedTheorems_1907_ = lean_ctor_get(v___x_1903_, 3);
v_numSteps_1908_ = lean_ctor_get(v___x_1903_, 4);
v_diag_1909_ = lean_ctor_get(v___x_1903_, 5);
v_isSharedCheck_1964_ = !lean_is_exclusive(v___x_1903_);
if (v_isSharedCheck_1964_ == 0)
{
v___x_1911_ = v___x_1903_;
v_isShared_1912_ = v_isSharedCheck_1964_;
goto v_resetjp_1910_;
}
else
{
lean_inc(v_diag_1909_);
lean_inc(v_numSteps_1908_);
lean_inc(v_usedTheorems_1907_);
lean_inc(v_dsimpCache_1906_);
lean_inc(v_congrCache_1905_);
lean_inc(v_cache_1904_);
lean_dec(v___x_1903_);
v___x_1911_ = lean_box(0);
v_isShared_1912_ = v_isSharedCheck_1964_;
goto v_resetjp_1910_;
}
v_resetjp_1910_:
{
lean_object* v___x_1913_; lean_object* v___x_1915_; 
v___x_1913_ = lp_mathlib_Lean_SMap_switch___at___00Lean_Meta_Simp_withoutTheorems_spec__1___redArg(v_cache_1904_);
if (v_isShared_1912_ == 0)
{
lean_ctor_set(v___x_1911_, 0, v___x_1913_);
v___x_1915_ = v___x_1911_;
goto v_reusejp_1914_;
}
else
{
lean_object* v_reuseFailAlloc_1963_; 
v_reuseFailAlloc_1963_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v_reuseFailAlloc_1963_, 0, v___x_1913_);
lean_ctor_set(v_reuseFailAlloc_1963_, 1, v_congrCache_1905_);
lean_ctor_set(v_reuseFailAlloc_1963_, 2, v_dsimpCache_1906_);
lean_ctor_set(v_reuseFailAlloc_1963_, 3, v_usedTheorems_1907_);
lean_ctor_set(v_reuseFailAlloc_1963_, 4, v_numSteps_1908_);
lean_ctor_set(v_reuseFailAlloc_1963_, 5, v_diag_1909_);
v___x_1915_ = v_reuseFailAlloc_1963_;
goto v_reusejp_1914_;
}
v_reusejp_1914_:
{
lean_object* v___x_1916_; lean_object* v___x_1917_; lean_object* v_cache_1918_; lean_object* v_cache_1919_; lean_object* v_fst_1920_; lean_object* v_snd_1921_; lean_object* v_discharge_x3f_1922_; uint8_t v_wellBehavedDischarge_1923_; lean_object* v_map_u2082_1924_; uint8_t v_stage_u2081_1925_; lean_object* v___x_1926_; lean_object* v___x_1927_; lean_object* v___x_1928_; lean_object* v___x_1929_; lean_object* v___x_1930_; lean_object* v___x_1931_; 
v___x_1916_ = lean_st_ref_set(v_a_1875_, v___x_1915_);
v___x_1917_ = lean_st_ref_get(v_a_1875_);
v_cache_1918_ = lean_ctor_get(v___x_1901_, 0);
lean_inc_ref(v_cache_1918_);
lean_dec(v___x_1901_);
v_cache_1919_ = lean_ctor_get(v___x_1902_, 0);
lean_inc_ref(v_cache_1919_);
lean_dec(v___x_1902_);
v_fst_1920_ = lean_ctor_get(v_a_1900_, 0);
lean_inc(v_fst_1920_);
v_snd_1921_ = lean_ctor_get(v_a_1900_, 1);
lean_inc(v_snd_1921_);
lean_dec(v_a_1900_);
v_discharge_x3f_1922_ = lean_ctor_get(v_a_1873_, 4);
v_wellBehavedDischarge_1923_ = lean_ctor_get_uint8(v_a_1873_, sizeof(void*)*5);
v_map_u2082_1924_ = lean_ctor_get(v_cache_1918_, 1);
lean_inc_ref(v_map_u2082_1924_);
lean_dec_ref(v_cache_1918_);
v_stage_u2081_1925_ = lean_ctor_get_uint8(v_cache_1919_, sizeof(void*)*2);
lean_dec_ref(v_cache_1919_);
v___x_1926_ = lean_unsigned_to_nat(1u);
v___x_1927_ = lean_mk_empty_array_with_capacity(v___x_1926_);
v___x_1928_ = lean_array_push(v___x_1927_, v_snd_1921_);
lean_inc_ref(v_discharge_x3f_1922_);
v___x_1929_ = l_Lean_Meta_Simp_mkMethods(v___x_1928_, v_discharge_x3f_1922_, v_wellBehavedDischarge_1923_);
lean_inc(v_lctxInitIndices_1894_);
lean_inc(v_parent_x3f_1892_);
lean_inc_ref(v_congrTheorems_1891_);
lean_inc_ref(v_indexConfig_1888_);
lean_inc_ref(v_metaConfig_1887_);
lean_inc(v_initUsedZetaDelta_1886_);
lean_inc(v_zetaDeltaSet_1885_);
lean_inc_ref(v_userConfig_1884_);
lean_inc_ref(v_config_1883_);
v___x_1930_ = lean_alloc_ctor(0, 10, 9);
lean_ctor_set(v___x_1930_, 0, v_config_1883_);
lean_ctor_set(v___x_1930_, 1, v_userConfig_1884_);
lean_ctor_set(v___x_1930_, 2, v_zetaDeltaSet_1885_);
lean_ctor_set(v___x_1930_, 3, v_initUsedZetaDelta_1886_);
lean_ctor_set(v___x_1930_, 4, v_metaConfig_1887_);
lean_ctor_set(v___x_1930_, 5, v_indexConfig_1888_);
lean_ctor_set(v___x_1930_, 6, v_fst_1920_);
lean_ctor_set(v___x_1930_, 7, v_congrTheorems_1891_);
lean_ctor_set(v___x_1930_, 8, v_parent_x3f_1892_);
lean_ctor_set(v___x_1930_, 9, v_lctxInitIndices_1894_);
lean_ctor_set_uint32(v___x_1930_, sizeof(void*)*10, v_maxDischargeDepth_1889_);
lean_ctor_set_uint32(v___x_1930_, sizeof(void*)*10 + 4, v_dischargeDepth_1893_);
lean_ctor_set_uint8(v___x_1930_, sizeof(void*)*10 + 8, v_inDSimp_1895_);
v___x_1931_ = l_Lean_Meta_Simp_SimpM_run___redArg(v___x_1930_, v___x_1917_, v___x_1929_, v_e_1872_, v_a_1876_, v_a_1877_, v_a_1878_, v_a_1879_);
if (lean_obj_tag(v___x_1931_) == 0)
{
lean_object* v_a_1932_; lean_object* v___x_1934_; uint8_t v_isShared_1935_; uint8_t v_isSharedCheck_1951_; 
v_a_1932_ = lean_ctor_get(v___x_1931_, 0);
v_isSharedCheck_1951_ = !lean_is_exclusive(v___x_1931_);
if (v_isSharedCheck_1951_ == 0)
{
v___x_1934_ = v___x_1931_;
v_isShared_1935_ = v_isSharedCheck_1951_;
goto v_resetjp_1933_;
}
else
{
lean_inc(v_a_1932_);
lean_dec(v___x_1931_);
v___x_1934_ = lean_box(0);
v_isShared_1935_ = v_isSharedCheck_1951_;
goto v_resetjp_1933_;
}
v_resetjp_1933_:
{
lean_object* v_fst_1936_; lean_object* v_snd_1937_; lean_object* v___x_1938_; lean_object* v___x_1940_; 
v_fst_1936_ = lean_ctor_get(v_a_1932_, 0);
lean_inc_n(v_fst_1936_, 2);
v_snd_1937_ = lean_ctor_get(v_a_1932_, 1);
lean_inc(v_snd_1937_);
lean_dec(v_a_1932_);
v___x_1938_ = lean_st_ref_set(v_a_1875_, v_snd_1937_);
if (v_isShared_1935_ == 0)
{
lean_ctor_set_tag(v___x_1934_, 1);
lean_ctor_set(v___x_1934_, 0, v_fst_1936_);
v___x_1940_ = v___x_1934_;
goto v_reusejp_1939_;
}
else
{
lean_object* v_reuseFailAlloc_1950_; 
v_reuseFailAlloc_1950_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1950_, 0, v_fst_1936_);
v___x_1940_ = v_reuseFailAlloc_1950_;
goto v_reusejp_1939_;
}
v_reusejp_1939_:
{
lean_object* v___x_1941_; lean_object* v___x_1943_; uint8_t v_isShared_1944_; uint8_t v_isSharedCheck_1948_; 
v___x_1941_ = lp_mathlib_Lean_Meta_Simp_withoutTheorems___redArg___lam__0(v_a_1875_, v_stage_u2081_1925_, v_map_u2082_1924_, v___x_1940_);
lean_dec_ref(v___x_1940_);
v_isSharedCheck_1948_ = !lean_is_exclusive(v___x_1941_);
if (v_isSharedCheck_1948_ == 0)
{
lean_object* v_unused_1949_; 
v_unused_1949_ = lean_ctor_get(v___x_1941_, 0);
lean_dec(v_unused_1949_);
v___x_1943_ = v___x_1941_;
v_isShared_1944_ = v_isSharedCheck_1948_;
goto v_resetjp_1942_;
}
else
{
lean_dec(v___x_1941_);
v___x_1943_ = lean_box(0);
v_isShared_1944_ = v_isSharedCheck_1948_;
goto v_resetjp_1942_;
}
v_resetjp_1942_:
{
lean_object* v___x_1946_; 
if (v_isShared_1944_ == 0)
{
lean_ctor_set(v___x_1943_, 0, v_fst_1936_);
v___x_1946_ = v___x_1943_;
goto v_reusejp_1945_;
}
else
{
lean_object* v_reuseFailAlloc_1947_; 
v_reuseFailAlloc_1947_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1947_, 0, v_fst_1936_);
v___x_1946_ = v_reuseFailAlloc_1947_;
goto v_reusejp_1945_;
}
v_reusejp_1945_:
{
return v___x_1946_;
}
}
}
}
}
else
{
lean_object* v_a_1952_; lean_object* v___x_1953_; lean_object* v___x_1954_; lean_object* v___x_1956_; uint8_t v_isShared_1957_; uint8_t v_isSharedCheck_1961_; 
v_a_1952_ = lean_ctor_get(v___x_1931_, 0);
lean_inc(v_a_1952_);
lean_dec_ref_known(v___x_1931_, 1);
v___x_1953_ = lean_box(0);
v___x_1954_ = lp_mathlib_Lean_Meta_Simp_withoutTheorems___redArg___lam__0(v_a_1875_, v_stage_u2081_1925_, v_map_u2082_1924_, v___x_1953_);
v_isSharedCheck_1961_ = !lean_is_exclusive(v___x_1954_);
if (v_isSharedCheck_1961_ == 0)
{
lean_object* v_unused_1962_; 
v_unused_1962_ = lean_ctor_get(v___x_1954_, 0);
lean_dec(v_unused_1962_);
v___x_1956_ = v___x_1954_;
v_isShared_1957_ = v_isSharedCheck_1961_;
goto v_resetjp_1955_;
}
else
{
lean_dec(v___x_1954_);
v___x_1956_ = lean_box(0);
v_isShared_1957_ = v_isSharedCheck_1961_;
goto v_resetjp_1955_;
}
v_resetjp_1955_:
{
lean_object* v___x_1959_; 
if (v_isShared_1957_ == 0)
{
lean_ctor_set_tag(v___x_1956_, 1);
lean_ctor_set(v___x_1956_, 0, v_a_1952_);
v___x_1959_ = v___x_1956_;
goto v_reusejp_1958_;
}
else
{
lean_object* v_reuseFailAlloc_1960_; 
v_reuseFailAlloc_1960_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1960_, 0, v_a_1952_);
v___x_1959_ = v_reuseFailAlloc_1960_;
goto v_reusejp_1958_;
}
v_reusejp_1958_:
{
return v___x_1959_;
}
}
}
}
}
}
else
{
lean_object* v_a_1965_; lean_object* v___x_1967_; uint8_t v_isShared_1968_; uint8_t v_isSharedCheck_1972_; 
lean_dec_ref(v_e_1872_);
v_a_1965_ = lean_ctor_get(v___x_1899_, 0);
v_isSharedCheck_1972_ = !lean_is_exclusive(v___x_1899_);
if (v_isSharedCheck_1972_ == 0)
{
v___x_1967_ = v___x_1899_;
v_isShared_1968_ = v_isSharedCheck_1972_;
goto v_resetjp_1966_;
}
else
{
lean_inc(v_a_1965_);
lean_dec(v___x_1899_);
v___x_1967_ = lean_box(0);
v_isShared_1968_ = v_isSharedCheck_1972_;
goto v_resetjp_1966_;
}
v_resetjp_1966_:
{
lean_object* v___x_1970_; 
if (v_isShared_1968_ == 0)
{
v___x_1970_ = v___x_1967_;
goto v_reusejp_1969_;
}
else
{
lean_object* v_reuseFailAlloc_1971_; 
v_reuseFailAlloc_1971_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1971_, 0, v_a_1965_);
v___x_1970_ = v_reuseFailAlloc_1971_;
goto v_reusejp_1969_;
}
v_reusejp_1969_:
{
return v___x_1970_;
}
}
}
}
else
{
lean_object* v_a_1973_; lean_object* v___x_1975_; uint8_t v_isShared_1976_; uint8_t v_isSharedCheck_1980_; 
lean_dec_ref(v_e_1872_);
v_a_1973_ = lean_ctor_get(v___x_1881_, 0);
v_isSharedCheck_1980_ = !lean_is_exclusive(v___x_1881_);
if (v_isSharedCheck_1980_ == 0)
{
v___x_1975_ = v___x_1881_;
v_isShared_1976_ = v_isSharedCheck_1980_;
goto v_resetjp_1974_;
}
else
{
lean_inc(v_a_1973_);
lean_dec(v___x_1881_);
v___x_1975_ = lean_box(0);
v_isShared_1976_ = v_isSharedCheck_1980_;
goto v_resetjp_1974_;
}
v_resetjp_1974_:
{
lean_object* v___x_1978_; 
if (v_isShared_1976_ == 0)
{
v___x_1978_ = v___x_1975_;
goto v_reusejp_1977_;
}
else
{
lean_object* v_reuseFailAlloc_1979_; 
v_reuseFailAlloc_1979_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1979_, 0, v_a_1973_);
v___x_1978_ = v_reuseFailAlloc_1979_;
goto v_reusejp_1977_;
}
v_reusejp_1977_:
{
return v___x_1978_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_withoutTheorems___redArg___boxed(lean_object* v_declNames_1981_, lean_object* v_e_1982_, lean_object* v_a_1983_, lean_object* v_a_1984_, lean_object* v_a_1985_, lean_object* v_a_1986_, lean_object* v_a_1987_, lean_object* v_a_1988_, lean_object* v_a_1989_, lean_object* v_a_1990_){
_start:
{
lean_object* v_res_1991_; 
v_res_1991_ = lp_mathlib_Lean_Meta_Simp_withoutTheorems___redArg(v_declNames_1981_, v_e_1982_, v_a_1983_, v_a_1984_, v_a_1985_, v_a_1986_, v_a_1987_, v_a_1988_, v_a_1989_);
lean_dec(v_a_1989_);
lean_dec_ref(v_a_1988_);
lean_dec(v_a_1987_);
lean_dec_ref(v_a_1986_);
lean_dec(v_a_1985_);
lean_dec_ref(v_a_1984_);
lean_dec(v_a_1983_);
lean_dec_ref(v_declNames_1981_);
return v_res_1991_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_withoutTheorems(lean_object* v_00_u03b1_1992_, lean_object* v_declNames_1993_, lean_object* v_e_1994_, lean_object* v_a_1995_, lean_object* v_a_1996_, lean_object* v_a_1997_, lean_object* v_a_1998_, lean_object* v_a_1999_, lean_object* v_a_2000_, lean_object* v_a_2001_){
_start:
{
lean_object* v___x_2003_; 
v___x_2003_ = lp_mathlib_Lean_Meta_Simp_withoutTheorems___redArg(v_declNames_1993_, v_e_1994_, v_a_1995_, v_a_1996_, v_a_1997_, v_a_1998_, v_a_1999_, v_a_2000_, v_a_2001_);
return v___x_2003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_Simp_withoutTheorems___boxed(lean_object* v_00_u03b1_2004_, lean_object* v_declNames_2005_, lean_object* v_e_2006_, lean_object* v_a_2007_, lean_object* v_a_2008_, lean_object* v_a_2009_, lean_object* v_a_2010_, lean_object* v_a_2011_, lean_object* v_a_2012_, lean_object* v_a_2013_, lean_object* v_a_2014_){
_start:
{
lean_object* v_res_2015_; 
v_res_2015_ = lp_mathlib_Lean_Meta_Simp_withoutTheorems(v_00_u03b1_2004_, v_declNames_2005_, v_e_2006_, v_a_2007_, v_a_2008_, v_a_2009_, v_a_2010_, v_a_2011_, v_a_2012_, v_a_2013_);
lean_dec(v_a_2013_);
lean_dec_ref(v_a_2012_);
lean_dec(v_a_2011_);
lean_dec_ref(v_a_2010_);
lean_dec(v_a_2009_);
lean_dec_ref(v_a_2008_);
lean_dec(v_a_2007_);
lean_dec_ref(v_declNames_2005_);
return v_res_2015_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_Simp_withoutTheorems_spec__0(lean_object* v_as_2016_, size_t v_sz_2017_, size_t v_i_2018_, lean_object* v_b_2019_, lean_object* v___y_2020_, lean_object* v___y_2021_, lean_object* v___y_2022_, lean_object* v___y_2023_, lean_object* v___y_2024_, lean_object* v___y_2025_, lean_object* v___y_2026_){
_start:
{
lean_object* v___x_2028_; 
v___x_2028_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_Simp_withoutTheorems_spec__0___redArg(v_as_2016_, v_sz_2017_, v_i_2018_, v_b_2019_);
return v___x_2028_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_Simp_withoutTheorems_spec__0___boxed(lean_object* v_as_2029_, lean_object* v_sz_2030_, lean_object* v_i_2031_, lean_object* v_b_2032_, lean_object* v___y_2033_, lean_object* v___y_2034_, lean_object* v___y_2035_, lean_object* v___y_2036_, lean_object* v___y_2037_, lean_object* v___y_2038_, lean_object* v___y_2039_, lean_object* v___y_2040_){
_start:
{
size_t v_sz_boxed_2041_; size_t v_i_boxed_2042_; lean_object* v_res_2043_; 
v_sz_boxed_2041_ = lean_unbox_usize(v_sz_2030_);
lean_dec(v_sz_2030_);
v_i_boxed_2042_ = lean_unbox_usize(v_i_2031_);
lean_dec(v_i_2031_);
v_res_2043_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_Meta_Simp_withoutTheorems_spec__0(v_as_2029_, v_sz_boxed_2041_, v_i_boxed_2042_, v_b_2032_, v___y_2033_, v___y_2034_, v___y_2035_, v___y_2036_, v___y_2037_, v___y_2038_, v___y_2039_);
lean_dec(v___y_2039_);
lean_dec_ref(v___y_2038_);
lean_dec(v___y_2037_);
lean_dec_ref(v___y_2036_);
lean_dec(v___y_2035_);
lean_dec_ref(v___y_2034_);
lean_dec(v___y_2033_);
lean_dec_ref(v_as_2029_);
return v_res_2043_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Simp(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_DiscrTree(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta_Simp(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_DiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Lean_Meta_Simp(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Simp(uint8_t builtin);
lean_object* initialize_Lean_Meta_DiscrTree(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_Meta_Simp(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_DiscrTree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Lean_Meta_Simp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Lean_Meta_Simp(builtin);
}
#ifdef __cplusplus
}
#endif
