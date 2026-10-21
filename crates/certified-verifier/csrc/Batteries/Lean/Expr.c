// Lean compiler output
// Module: Batteries.Lean.Expr
// Imports: public import Init public meta import Init public import Lean.Elab.Term public import Lean.Elab.Binders
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
lean_object* lean_st_ref_take(lean_object*);
uint64_t l_Lean_instHashableMVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqMVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_shift_right(size_t, size_t);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_addLocalVarInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_mkSort(lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs_x27(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_array_set(lean_object*, lean_object*, lean_object*);
lean_object* l_mkPanicMessageWithDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
extern lean_object* l_Int_instInhabited;
lean_object* lean_panic_fn_borrowed(lean_object*, lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Int_negOfNat(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Lean_mkAppN(lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_mvarId_x21(lean_object*);
lean_object* l_Lean_Core_withFreshMacroScope___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Expr_getAppFn_x27(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_TermElabM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__2_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__3___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_batteries_Lean_Expr_toSyntax___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\?"};
static const lean_object* lp_batteries_Lean_Expr_toSyntax___lam__0___closed__0 = (const lean_object*)&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__0_value;
static const lean_string_object lp_batteries_Lean_Expr_toSyntax___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "a"};
static const lean_object* lp_batteries_Lean_Expr_toSyntax___lam__0___closed__1 = (const lean_object*)&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__1_value;
static lean_once_cell_t lp_batteries_Lean_Expr_toSyntax___lam__0___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Expr_toSyntax___lam__0___closed__2;
static const lean_ctor_object lp_batteries_Lean_Expr_toSyntax___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(247, 80, 99, 121, 74, 33, 203, 108)}};
static const lean_object* lp_batteries_Lean_Expr_toSyntax___lam__0___closed__3 = (const lean_object*)&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__3_value;
static const lean_string_object lp_batteries_Lean_Expr_toSyntax___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_batteries_Lean_Expr_toSyntax___lam__0___closed__4 = (const lean_object*)&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__4_value;
static const lean_string_object lp_batteries_Lean_Expr_toSyntax___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_batteries_Lean_Expr_toSyntax___lam__0___closed__5 = (const lean_object*)&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__5_value;
static const lean_string_object lp_batteries_Lean_Expr_toSyntax___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_batteries_Lean_Expr_toSyntax___lam__0___closed__6 = (const lean_object*)&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__6_value;
static const lean_string_object lp_batteries_Lean_Expr_toSyntax___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "syntheticHole"};
static const lean_object* lp_batteries_Lean_Expr_toSyntax___lam__0___closed__7 = (const lean_object*)&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__7_value;
static const lean_ctor_object lp_batteries_Lean_Expr_toSyntax___lam__0___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Lean_Expr_toSyntax___lam__0___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__8_value_aux_0),((lean_object*)&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__5_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_batteries_Lean_Expr_toSyntax___lam__0___closed__8_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__8_value_aux_1),((lean_object*)&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_batteries_Lean_Expr_toSyntax___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__8_value_aux_2),((lean_object*)&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__7_value),LEAN_SCALAR_PTR_LITERAL(218, 189, 67, 60, 211, 196, 112, 165)}};
static const lean_object* lp_batteries_Lean_Expr_toSyntax___lam__0___closed__8 = (const lean_object*)&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__8_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_toSyntax___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_toSyntax___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_toSyntax(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_toSyntax___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__3(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__2_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_withApp_x27_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_withApp_x27_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_batteries_Lean_Expr_withApp_x27___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Expr_withApp_x27___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_withApp_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_withApp_x27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_getAppArgs_x27___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_getAppArgs_x27___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Lean_Expr_getAppArgs_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_Expr_getAppArgs_x27___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_Expr_getAppArgs_x27___closed__0 = (const lean_object*)&lp_batteries_Lean_Expr_getAppArgs_x27___closed__0_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_getAppArgs_x27(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_traverseApp_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_traverseApp_x27___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_traverseApp_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_traverseApp_x27___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_traverseApp_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_traverseApp_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_withAppRev_x27_go___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_withAppRev_x27_go(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_withAppRev_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_withAppRev_x27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_getAppRevArgs_x27(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_getRevArgD_x27(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_getRevArgD_x27___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_getArgD_x27(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_getArgD_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_Expr_isAppOf_x27(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_isAppOf_x27___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Lean_Expr_natLit_x21_spec__0(lean_object*);
static const lean_string_object lp_batteries_Lean_Expr_natLit_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "Batteries.Lean.Expr"};
static const lean_object* lp_batteries_Lean_Expr_natLit_x21___closed__0 = (const lean_object*)&lp_batteries_Lean_Expr_natLit_x21___closed__0_value;
static const lean_string_object lp_batteries_Lean_Expr_natLit_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Lean.Expr.natLit!"};
static const lean_object* lp_batteries_Lean_Expr_natLit_x21___closed__1 = (const lean_object*)&lp_batteries_Lean_Expr_natLit_x21___closed__1_value;
static const lean_string_object lp_batteries_Lean_Expr_natLit_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "nat literal expected"};
static const lean_object* lp_batteries_Lean_Expr_natLit_x21___closed__2 = (const lean_object*)&lp_batteries_Lean_Expr_natLit_x21___closed__2_value;
static lean_once_cell_t lp_batteries_Lean_Expr_natLit_x21___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Expr_natLit_x21___closed__3;
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_natLit_x21(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_natLit_x21___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Lean_Expr_intLit_x21_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nat_cast___at___00Lean_Expr_intLit_x21_spec__1(lean_object*);
static const lean_string_object lp_batteries_Lean_Expr_intLit_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Int"};
static const lean_object* lp_batteries_Lean_Expr_intLit_x21___closed__0 = (const lean_object*)&lp_batteries_Lean_Expr_intLit_x21___closed__0_value;
static const lean_string_object lp_batteries_Lean_Expr_intLit_x21___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ofNat"};
static const lean_object* lp_batteries_Lean_Expr_intLit_x21___closed__1 = (const lean_object*)&lp_batteries_Lean_Expr_intLit_x21___closed__1_value;
static const lean_ctor_object lp_batteries_Lean_Expr_intLit_x21___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_Expr_intLit_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_batteries_Lean_Expr_intLit_x21___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Expr_intLit_x21___closed__2_value_aux_0),((lean_object*)&lp_batteries_Lean_Expr_intLit_x21___closed__1_value),LEAN_SCALAR_PTR_LITERAL(192, 66, 133, 102, 95, 170, 134, 92)}};
static const lean_object* lp_batteries_Lean_Expr_intLit_x21___closed__2 = (const lean_object*)&lp_batteries_Lean_Expr_intLit_x21___closed__2_value;
static const lean_string_object lp_batteries_Lean_Expr_intLit_x21___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "negOfNat"};
static const lean_object* lp_batteries_Lean_Expr_intLit_x21___closed__3 = (const lean_object*)&lp_batteries_Lean_Expr_intLit_x21___closed__3_value;
static const lean_ctor_object lp_batteries_Lean_Expr_intLit_x21___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_Expr_intLit_x21___closed__0_value),LEAN_SCALAR_PTR_LITERAL(61, 25, 98, 154, 117, 127, 69, 97)}};
static const lean_ctor_object lp_batteries_Lean_Expr_intLit_x21___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Expr_intLit_x21___closed__4_value_aux_0),((lean_object*)&lp_batteries_Lean_Expr_intLit_x21___closed__3_value),LEAN_SCALAR_PTR_LITERAL(100, 231, 152, 184, 84, 220, 144, 243)}};
static const lean_object* lp_batteries_Lean_Expr_intLit_x21___closed__4 = (const lean_object*)&lp_batteries_Lean_Expr_intLit_x21___closed__4_value;
static const lean_string_object lp_batteries_Lean_Expr_intLit_x21___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 18, .m_capacity = 18, .m_length = 17, .m_data = "Lean.Expr.intLit!"};
static const lean_object* lp_batteries_Lean_Expr_intLit_x21___closed__5 = (const lean_object*)&lp_batteries_Lean_Expr_intLit_x21___closed__5_value;
static const lean_string_object lp_batteries_Lean_Expr_intLit_x21___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "not a raw integer literal"};
static const lean_object* lp_batteries_Lean_Expr_intLit_x21___closed__6 = (const lean_object*)&lp_batteries_Lean_Expr_intLit_x21___closed__6_value;
static lean_once_cell_t lp_batteries_Lean_Expr_intLit_x21___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_batteries_Lean_Expr_intLit_x21___closed__7;
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_intLit_x21(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_intLit_x21___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__0___boxed(lean_object*);
static const lean_string_object lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "hole"};
static const lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__0 = (const lean_object*)&lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__0_value;
static const lean_string_object lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__1 = (const lean_object*)&lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__1_value;
static const lean_string_object lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__2 = (const lean_object*)&lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__2_value;
static const lean_ctor_object lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__3 = (const lean_object*)&lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__3_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1(uint8_t, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__0 = (const lean_object*)&lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__0_value;
static const lean_string_object lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__1 = (const lean_object*)&lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__1_value;
static const lean_ctor_object lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__2_value_aux_0),((lean_object*)&lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__1_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__2 = (const lean_object*)&lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__2_value;
static const lean_array_object lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__3 = (const lean_object*)&lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__3_value;
static const lean_ctor_object lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*8 + 16, .m_other = 8, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__0_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__3_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 1, 0, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(1, 0, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__4 = (const lean_object*)&lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__4_value;
static const lean_ctor_object lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__5 = (const lean_object*)&lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__5_value;
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__2_spec__3___redArg(lean_object* v_x_1_, lean_object* v_x_2_, lean_object* v_x_3_, lean_object* v_x_4_){
_start:
{
lean_object* v_ks_5_; lean_object* v_vs_6_; lean_object* v___x_8_; uint8_t v_isShared_9_; uint8_t v_isSharedCheck_30_; 
v_ks_5_ = lean_ctor_get(v_x_1_, 0);
v_vs_6_ = lean_ctor_get(v_x_1_, 1);
v_isSharedCheck_30_ = !lean_is_exclusive(v_x_1_);
if (v_isSharedCheck_30_ == 0)
{
v___x_8_ = v_x_1_;
v_isShared_9_ = v_isSharedCheck_30_;
goto v_resetjp_7_;
}
else
{
lean_inc(v_vs_6_);
lean_inc(v_ks_5_);
lean_dec(v_x_1_);
v___x_8_ = lean_box(0);
v_isShared_9_ = v_isSharedCheck_30_;
goto v_resetjp_7_;
}
v_resetjp_7_:
{
lean_object* v___x_10_; uint8_t v___x_11_; 
v___x_10_ = lean_array_get_size(v_ks_5_);
v___x_11_ = lean_nat_dec_lt(v_x_2_, v___x_10_);
if (v___x_11_ == 0)
{
lean_object* v___x_12_; lean_object* v___x_13_; lean_object* v___x_15_; 
lean_dec(v_x_2_);
v___x_12_ = lean_array_push(v_ks_5_, v_x_3_);
v___x_13_ = lean_array_push(v_vs_6_, v_x_4_);
if (v_isShared_9_ == 0)
{
lean_ctor_set(v___x_8_, 1, v___x_13_);
lean_ctor_set(v___x_8_, 0, v___x_12_);
v___x_15_ = v___x_8_;
goto v_reusejp_14_;
}
else
{
lean_object* v_reuseFailAlloc_16_; 
v_reuseFailAlloc_16_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_16_, 0, v___x_12_);
lean_ctor_set(v_reuseFailAlloc_16_, 1, v___x_13_);
v___x_15_ = v_reuseFailAlloc_16_;
goto v_reusejp_14_;
}
v_reusejp_14_:
{
return v___x_15_;
}
}
else
{
lean_object* v_k_x27_17_; uint8_t v___x_18_; 
v_k_x27_17_ = lean_array_fget_borrowed(v_ks_5_, v_x_2_);
v___x_18_ = l_Lean_instBEqMVarId_beq(v_x_3_, v_k_x27_17_);
if (v___x_18_ == 0)
{
lean_object* v___x_20_; 
if (v_isShared_9_ == 0)
{
v___x_20_ = v___x_8_;
goto v_reusejp_19_;
}
else
{
lean_object* v_reuseFailAlloc_24_; 
v_reuseFailAlloc_24_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_24_, 0, v_ks_5_);
lean_ctor_set(v_reuseFailAlloc_24_, 1, v_vs_6_);
v___x_20_ = v_reuseFailAlloc_24_;
goto v_reusejp_19_;
}
v_reusejp_19_:
{
lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_21_ = lean_unsigned_to_nat(1u);
v___x_22_ = lean_nat_add(v_x_2_, v___x_21_);
lean_dec(v_x_2_);
v_x_1_ = v___x_20_;
v_x_2_ = v___x_22_;
goto _start;
}
}
else
{
lean_object* v___x_25_; lean_object* v___x_26_; lean_object* v___x_28_; 
v___x_25_ = lean_array_fset(v_ks_5_, v_x_2_, v_x_3_);
v___x_26_ = lean_array_fset(v_vs_6_, v_x_2_, v_x_4_);
lean_dec(v_x_2_);
if (v_isShared_9_ == 0)
{
lean_ctor_set(v___x_8_, 1, v___x_26_);
lean_ctor_set(v___x_8_, 0, v___x_25_);
v___x_28_ = v___x_8_;
goto v_reusejp_27_;
}
else
{
lean_object* v_reuseFailAlloc_29_; 
v_reuseFailAlloc_29_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_29_, 0, v___x_25_);
lean_ctor_set(v_reuseFailAlloc_29_, 1, v___x_26_);
v___x_28_ = v_reuseFailAlloc_29_;
goto v_reusejp_27_;
}
v_reusejp_27_:
{
return v___x_28_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__2___redArg(lean_object* v_n_31_, lean_object* v_k_32_, lean_object* v_v_33_){
_start:
{
lean_object* v___x_34_; lean_object* v___x_35_; 
v___x_34_ = lean_unsigned_to_nat(0u);
v___x_35_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__2_spec__3___redArg(v_n_31_, v___x_34_, v_k_32_, v_v_33_);
return v___x_35_;
}
}
static lean_object* _init_lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_36_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1___redArg(lean_object* v_x_37_, size_t v_x_38_, size_t v_x_39_, lean_object* v_x_40_, lean_object* v_x_41_){
_start:
{
if (lean_obj_tag(v_x_37_) == 0)
{
lean_object* v_es_42_; size_t v___x_43_; size_t v___x_44_; lean_object* v_j_45_; lean_object* v___x_46_; uint8_t v___x_47_; 
v_es_42_ = lean_ctor_get(v_x_37_, 0);
v___x_43_ = ((size_t)31ULL);
v___x_44_ = lean_usize_land(v_x_38_, v___x_43_);
v_j_45_ = lean_usize_to_nat(v___x_44_);
v___x_46_ = lean_array_get_size(v_es_42_);
v___x_47_ = lean_nat_dec_lt(v_j_45_, v___x_46_);
if (v___x_47_ == 0)
{
lean_dec(v_j_45_);
lean_dec(v_x_41_);
lean_dec(v_x_40_);
return v_x_37_;
}
else
{
lean_object* v___x_49_; uint8_t v_isShared_50_; uint8_t v_isSharedCheck_86_; 
lean_inc_ref(v_es_42_);
v_isSharedCheck_86_ = !lean_is_exclusive(v_x_37_);
if (v_isSharedCheck_86_ == 0)
{
lean_object* v_unused_87_; 
v_unused_87_ = lean_ctor_get(v_x_37_, 0);
lean_dec(v_unused_87_);
v___x_49_ = v_x_37_;
v_isShared_50_ = v_isSharedCheck_86_;
goto v_resetjp_48_;
}
else
{
lean_dec(v_x_37_);
v___x_49_ = lean_box(0);
v_isShared_50_ = v_isSharedCheck_86_;
goto v_resetjp_48_;
}
v_resetjp_48_:
{
lean_object* v_v_51_; lean_object* v___x_52_; lean_object* v_xs_x27_53_; lean_object* v___y_55_; 
v_v_51_ = lean_array_fget(v_es_42_, v_j_45_);
v___x_52_ = lean_box(0);
v_xs_x27_53_ = lean_array_fset(v_es_42_, v_j_45_, v___x_52_);
switch(lean_obj_tag(v_v_51_))
{
case 0:
{
lean_object* v_key_60_; lean_object* v_val_61_; lean_object* v___x_63_; uint8_t v_isShared_64_; uint8_t v_isSharedCheck_71_; 
v_key_60_ = lean_ctor_get(v_v_51_, 0);
v_val_61_ = lean_ctor_get(v_v_51_, 1);
v_isSharedCheck_71_ = !lean_is_exclusive(v_v_51_);
if (v_isSharedCheck_71_ == 0)
{
v___x_63_ = v_v_51_;
v_isShared_64_ = v_isSharedCheck_71_;
goto v_resetjp_62_;
}
else
{
lean_inc(v_val_61_);
lean_inc(v_key_60_);
lean_dec(v_v_51_);
v___x_63_ = lean_box(0);
v_isShared_64_ = v_isSharedCheck_71_;
goto v_resetjp_62_;
}
v_resetjp_62_:
{
uint8_t v___x_65_; 
v___x_65_ = l_Lean_instBEqMVarId_beq(v_x_40_, v_key_60_);
if (v___x_65_ == 0)
{
lean_object* v___x_66_; lean_object* v___x_67_; 
lean_del_object(v___x_63_);
v___x_66_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_60_, v_val_61_, v_x_40_, v_x_41_);
v___x_67_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_67_, 0, v___x_66_);
v___y_55_ = v___x_67_;
goto v___jp_54_;
}
else
{
lean_object* v___x_69_; 
lean_dec(v_val_61_);
lean_dec(v_key_60_);
if (v_isShared_64_ == 0)
{
lean_ctor_set(v___x_63_, 1, v_x_41_);
lean_ctor_set(v___x_63_, 0, v_x_40_);
v___x_69_ = v___x_63_;
goto v_reusejp_68_;
}
else
{
lean_object* v_reuseFailAlloc_70_; 
v_reuseFailAlloc_70_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_70_, 0, v_x_40_);
lean_ctor_set(v_reuseFailAlloc_70_, 1, v_x_41_);
v___x_69_ = v_reuseFailAlloc_70_;
goto v_reusejp_68_;
}
v_reusejp_68_:
{
v___y_55_ = v___x_69_;
goto v___jp_54_;
}
}
}
}
case 1:
{
lean_object* v_node_72_; lean_object* v___x_74_; uint8_t v_isShared_75_; uint8_t v_isSharedCheck_84_; 
v_node_72_ = lean_ctor_get(v_v_51_, 0);
v_isSharedCheck_84_ = !lean_is_exclusive(v_v_51_);
if (v_isSharedCheck_84_ == 0)
{
v___x_74_ = v_v_51_;
v_isShared_75_ = v_isSharedCheck_84_;
goto v_resetjp_73_;
}
else
{
lean_inc(v_node_72_);
lean_dec(v_v_51_);
v___x_74_ = lean_box(0);
v_isShared_75_ = v_isSharedCheck_84_;
goto v_resetjp_73_;
}
v_resetjp_73_:
{
size_t v___x_76_; size_t v___x_77_; size_t v___x_78_; size_t v___x_79_; lean_object* v___x_80_; lean_object* v___x_82_; 
v___x_76_ = ((size_t)5ULL);
v___x_77_ = lean_usize_shift_right(v_x_38_, v___x_76_);
v___x_78_ = ((size_t)1ULL);
v___x_79_ = lean_usize_add(v_x_39_, v___x_78_);
v___x_80_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1___redArg(v_node_72_, v___x_77_, v___x_79_, v_x_40_, v_x_41_);
if (v_isShared_75_ == 0)
{
lean_ctor_set(v___x_74_, 0, v___x_80_);
v___x_82_ = v___x_74_;
goto v_reusejp_81_;
}
else
{
lean_object* v_reuseFailAlloc_83_; 
v_reuseFailAlloc_83_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_83_, 0, v___x_80_);
v___x_82_ = v_reuseFailAlloc_83_;
goto v_reusejp_81_;
}
v_reusejp_81_:
{
v___y_55_ = v___x_82_;
goto v___jp_54_;
}
}
}
default: 
{
lean_object* v___x_85_; 
v___x_85_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_85_, 0, v_x_40_);
lean_ctor_set(v___x_85_, 1, v_x_41_);
v___y_55_ = v___x_85_;
goto v___jp_54_;
}
}
v___jp_54_:
{
lean_object* v___x_56_; lean_object* v___x_58_; 
v___x_56_ = lean_array_fset(v_xs_x27_53_, v_j_45_, v___y_55_);
lean_dec(v_j_45_);
if (v_isShared_50_ == 0)
{
lean_ctor_set(v___x_49_, 0, v___x_56_);
v___x_58_ = v___x_49_;
goto v_reusejp_57_;
}
else
{
lean_object* v_reuseFailAlloc_59_; 
v_reuseFailAlloc_59_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_59_, 0, v___x_56_);
v___x_58_ = v_reuseFailAlloc_59_;
goto v_reusejp_57_;
}
v_reusejp_57_:
{
return v___x_58_;
}
}
}
}
}
else
{
lean_object* v_ks_88_; lean_object* v_vs_89_; lean_object* v___x_91_; uint8_t v_isShared_92_; uint8_t v_isSharedCheck_109_; 
v_ks_88_ = lean_ctor_get(v_x_37_, 0);
v_vs_89_ = lean_ctor_get(v_x_37_, 1);
v_isSharedCheck_109_ = !lean_is_exclusive(v_x_37_);
if (v_isSharedCheck_109_ == 0)
{
v___x_91_ = v_x_37_;
v_isShared_92_ = v_isSharedCheck_109_;
goto v_resetjp_90_;
}
else
{
lean_inc(v_vs_89_);
lean_inc(v_ks_88_);
lean_dec(v_x_37_);
v___x_91_ = lean_box(0);
v_isShared_92_ = v_isSharedCheck_109_;
goto v_resetjp_90_;
}
v_resetjp_90_:
{
lean_object* v___x_94_; 
if (v_isShared_92_ == 0)
{
v___x_94_ = v___x_91_;
goto v_reusejp_93_;
}
else
{
lean_object* v_reuseFailAlloc_108_; 
v_reuseFailAlloc_108_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_108_, 0, v_ks_88_);
lean_ctor_set(v_reuseFailAlloc_108_, 1, v_vs_89_);
v___x_94_ = v_reuseFailAlloc_108_;
goto v_reusejp_93_;
}
v_reusejp_93_:
{
lean_object* v_newNode_95_; uint8_t v___y_97_; size_t v___x_103_; uint8_t v___x_104_; 
v_newNode_95_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__2___redArg(v___x_94_, v_x_40_, v_x_41_);
v___x_103_ = ((size_t)7ULL);
v___x_104_ = lean_usize_dec_le(v___x_103_, v_x_39_);
if (v___x_104_ == 0)
{
lean_object* v___x_105_; lean_object* v___x_106_; uint8_t v___x_107_; 
v___x_105_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_95_);
v___x_106_ = lean_unsigned_to_nat(4u);
v___x_107_ = lean_nat_dec_lt(v___x_105_, v___x_106_);
lean_dec(v___x_105_);
v___y_97_ = v___x_107_;
goto v___jp_96_;
}
else
{
v___y_97_ = v___x_104_;
goto v___jp_96_;
}
v___jp_96_:
{
if (v___y_97_ == 0)
{
lean_object* v_ks_98_; lean_object* v_vs_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; 
v_ks_98_ = lean_ctor_get(v_newNode_95_, 0);
lean_inc_ref(v_ks_98_);
v_vs_99_ = lean_ctor_get(v_newNode_95_, 1);
lean_inc_ref(v_vs_99_);
lean_dec_ref(v_newNode_95_);
v___x_100_ = lean_unsigned_to_nat(0u);
v___x_101_ = lean_obj_once(&lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1___redArg___closed__0, &lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1___redArg___closed__0_once, _init_lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1___redArg___closed__0);
v___x_102_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__3___redArg(v_x_39_, v_ks_98_, v_vs_99_, v___x_100_, v___x_101_);
lean_dec_ref(v_vs_99_);
lean_dec_ref(v_ks_98_);
return v___x_102_;
}
else
{
return v_newNode_95_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__3___redArg(size_t v_depth_110_, lean_object* v_keys_111_, lean_object* v_vals_112_, lean_object* v_i_113_, lean_object* v_entries_114_){
_start:
{
lean_object* v___x_115_; uint8_t v___x_116_; 
v___x_115_ = lean_array_get_size(v_keys_111_);
v___x_116_ = lean_nat_dec_lt(v_i_113_, v___x_115_);
if (v___x_116_ == 0)
{
lean_dec(v_i_113_);
return v_entries_114_;
}
else
{
lean_object* v_k_117_; lean_object* v_v_118_; uint64_t v___x_119_; size_t v_h_120_; size_t v___x_121_; lean_object* v___x_122_; size_t v___x_123_; size_t v___x_124_; size_t v___x_125_; size_t v_h_126_; lean_object* v___x_127_; lean_object* v___x_128_; 
v_k_117_ = lean_array_fget_borrowed(v_keys_111_, v_i_113_);
v_v_118_ = lean_array_fget_borrowed(v_vals_112_, v_i_113_);
v___x_119_ = l_Lean_instHashableMVarId_hash(v_k_117_);
v_h_120_ = lean_uint64_to_usize(v___x_119_);
v___x_121_ = ((size_t)5ULL);
v___x_122_ = lean_unsigned_to_nat(1u);
v___x_123_ = ((size_t)1ULL);
v___x_124_ = lean_usize_sub(v_depth_110_, v___x_123_);
v___x_125_ = lean_usize_mul(v___x_121_, v___x_124_);
v_h_126_ = lean_usize_shift_right(v_h_120_, v___x_125_);
v___x_127_ = lean_nat_add(v_i_113_, v___x_122_);
lean_dec(v_i_113_);
lean_inc(v_v_118_);
lean_inc(v_k_117_);
v___x_128_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1___redArg(v_entries_114_, v_h_126_, v_depth_110_, v_k_117_, v_v_118_);
v_i_113_ = v___x_127_;
v_entries_114_ = v___x_128_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__3___redArg___boxed(lean_object* v_depth_130_, lean_object* v_keys_131_, lean_object* v_vals_132_, lean_object* v_i_133_, lean_object* v_entries_134_){
_start:
{
size_t v_depth_boxed_135_; lean_object* v_res_136_; 
v_depth_boxed_135_ = lean_unbox_usize(v_depth_130_);
lean_dec(v_depth_130_);
v_res_136_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__3___redArg(v_depth_boxed_135_, v_keys_131_, v_vals_132_, v_i_133_, v_entries_134_);
lean_dec_ref(v_vals_132_);
lean_dec_ref(v_keys_131_);
return v_res_136_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1___redArg___boxed(lean_object* v_x_137_, lean_object* v_x_138_, lean_object* v_x_139_, lean_object* v_x_140_, lean_object* v_x_141_){
_start:
{
size_t v_x_3631__boxed_142_; size_t v_x_3632__boxed_143_; lean_object* v_res_144_; 
v_x_3631__boxed_142_ = lean_unbox_usize(v_x_138_);
lean_dec(v_x_138_);
v_x_3632__boxed_143_ = lean_unbox_usize(v_x_139_);
lean_dec(v_x_139_);
v_res_144_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1___redArg(v_x_137_, v_x_3631__boxed_142_, v_x_3632__boxed_143_, v_x_140_, v_x_141_);
return v_res_144_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0___redArg(lean_object* v_x_145_, lean_object* v_x_146_, lean_object* v_x_147_){
_start:
{
uint64_t v___x_148_; size_t v___x_149_; size_t v___x_150_; lean_object* v___x_151_; 
v___x_148_ = l_Lean_instHashableMVarId_hash(v_x_146_);
v___x_149_ = lean_uint64_to_usize(v___x_148_);
v___x_150_ = ((size_t)1ULL);
v___x_151_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1___redArg(v_x_145_, v___x_149_, v___x_150_, v_x_146_, v_x_147_);
return v___x_151_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0___redArg(lean_object* v_mvarId_152_, lean_object* v_val_153_, lean_object* v___y_154_){
_start:
{
lean_object* v___x_156_; lean_object* v_mctx_157_; lean_object* v_cache_158_; lean_object* v_zetaDeltaFVarIds_159_; lean_object* v_postponed_160_; lean_object* v_diag_161_; lean_object* v___x_163_; uint8_t v_isShared_164_; uint8_t v_isSharedCheck_189_; 
v___x_156_ = lean_st_ref_take(v___y_154_);
v_mctx_157_ = lean_ctor_get(v___x_156_, 0);
v_cache_158_ = lean_ctor_get(v___x_156_, 1);
v_zetaDeltaFVarIds_159_ = lean_ctor_get(v___x_156_, 2);
v_postponed_160_ = lean_ctor_get(v___x_156_, 3);
v_diag_161_ = lean_ctor_get(v___x_156_, 4);
v_isSharedCheck_189_ = !lean_is_exclusive(v___x_156_);
if (v_isSharedCheck_189_ == 0)
{
v___x_163_ = v___x_156_;
v_isShared_164_ = v_isSharedCheck_189_;
goto v_resetjp_162_;
}
else
{
lean_inc(v_diag_161_);
lean_inc(v_postponed_160_);
lean_inc(v_zetaDeltaFVarIds_159_);
lean_inc(v_cache_158_);
lean_inc(v_mctx_157_);
lean_dec(v___x_156_);
v___x_163_ = lean_box(0);
v_isShared_164_ = v_isSharedCheck_189_;
goto v_resetjp_162_;
}
v_resetjp_162_:
{
lean_object* v_depth_165_; lean_object* v_levelAssignDepth_166_; lean_object* v_lmvarCounter_167_; lean_object* v_mvarCounter_168_; lean_object* v_lDecls_169_; lean_object* v_decls_170_; lean_object* v_userNames_171_; lean_object* v_lAssignment_172_; lean_object* v_eAssignment_173_; lean_object* v_dAssignment_174_; lean_object* v___x_176_; uint8_t v_isShared_177_; uint8_t v_isSharedCheck_188_; 
v_depth_165_ = lean_ctor_get(v_mctx_157_, 0);
v_levelAssignDepth_166_ = lean_ctor_get(v_mctx_157_, 1);
v_lmvarCounter_167_ = lean_ctor_get(v_mctx_157_, 2);
v_mvarCounter_168_ = lean_ctor_get(v_mctx_157_, 3);
v_lDecls_169_ = lean_ctor_get(v_mctx_157_, 4);
v_decls_170_ = lean_ctor_get(v_mctx_157_, 5);
v_userNames_171_ = lean_ctor_get(v_mctx_157_, 6);
v_lAssignment_172_ = lean_ctor_get(v_mctx_157_, 7);
v_eAssignment_173_ = lean_ctor_get(v_mctx_157_, 8);
v_dAssignment_174_ = lean_ctor_get(v_mctx_157_, 9);
v_isSharedCheck_188_ = !lean_is_exclusive(v_mctx_157_);
if (v_isSharedCheck_188_ == 0)
{
v___x_176_ = v_mctx_157_;
v_isShared_177_ = v_isSharedCheck_188_;
goto v_resetjp_175_;
}
else
{
lean_inc(v_dAssignment_174_);
lean_inc(v_eAssignment_173_);
lean_inc(v_lAssignment_172_);
lean_inc(v_userNames_171_);
lean_inc(v_decls_170_);
lean_inc(v_lDecls_169_);
lean_inc(v_mvarCounter_168_);
lean_inc(v_lmvarCounter_167_);
lean_inc(v_levelAssignDepth_166_);
lean_inc(v_depth_165_);
lean_dec(v_mctx_157_);
v___x_176_ = lean_box(0);
v_isShared_177_ = v_isSharedCheck_188_;
goto v_resetjp_175_;
}
v_resetjp_175_:
{
lean_object* v___x_178_; lean_object* v___x_180_; 
v___x_178_ = lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0___redArg(v_eAssignment_173_, v_mvarId_152_, v_val_153_);
if (v_isShared_177_ == 0)
{
lean_ctor_set(v___x_176_, 8, v___x_178_);
v___x_180_ = v___x_176_;
goto v_reusejp_179_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v_depth_165_);
lean_ctor_set(v_reuseFailAlloc_187_, 1, v_levelAssignDepth_166_);
lean_ctor_set(v_reuseFailAlloc_187_, 2, v_lmvarCounter_167_);
lean_ctor_set(v_reuseFailAlloc_187_, 3, v_mvarCounter_168_);
lean_ctor_set(v_reuseFailAlloc_187_, 4, v_lDecls_169_);
lean_ctor_set(v_reuseFailAlloc_187_, 5, v_decls_170_);
lean_ctor_set(v_reuseFailAlloc_187_, 6, v_userNames_171_);
lean_ctor_set(v_reuseFailAlloc_187_, 7, v_lAssignment_172_);
lean_ctor_set(v_reuseFailAlloc_187_, 8, v___x_178_);
lean_ctor_set(v_reuseFailAlloc_187_, 9, v_dAssignment_174_);
v___x_180_ = v_reuseFailAlloc_187_;
goto v_reusejp_179_;
}
v_reusejp_179_:
{
lean_object* v___x_182_; 
if (v_isShared_164_ == 0)
{
lean_ctor_set(v___x_163_, 0, v___x_180_);
v___x_182_ = v___x_163_;
goto v_reusejp_181_;
}
else
{
lean_object* v_reuseFailAlloc_186_; 
v_reuseFailAlloc_186_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_186_, 0, v___x_180_);
lean_ctor_set(v_reuseFailAlloc_186_, 1, v_cache_158_);
lean_ctor_set(v_reuseFailAlloc_186_, 2, v_zetaDeltaFVarIds_159_);
lean_ctor_set(v_reuseFailAlloc_186_, 3, v_postponed_160_);
lean_ctor_set(v_reuseFailAlloc_186_, 4, v_diag_161_);
v___x_182_ = v_reuseFailAlloc_186_;
goto v_reusejp_181_;
}
v_reusejp_181_:
{
lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; 
v___x_183_ = lean_st_ref_set(v___y_154_, v___x_182_);
v___x_184_ = lean_box(0);
v___x_185_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_185_, 0, v___x_184_);
return v___x_185_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0___redArg___boxed(lean_object* v_mvarId_190_, lean_object* v_val_191_, lean_object* v___y_192_, lean_object* v___y_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_batteries_Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0___redArg(v_mvarId_190_, v_val_191_, v___y_192_);
lean_dec(v___y_192_);
return v_res_194_;
}
}
static lean_object* _init_lp_batteries_Lean_Expr_toSyntax___lam__0___closed__2(void){
_start:
{
lean_object* v___x_197_; lean_object* v___x_198_; 
v___x_197_ = ((lean_object*)(lp_batteries_Lean_Expr_toSyntax___lam__0___closed__1));
v___x_198_ = l_String_toRawSubstring_x27(v___x_197_);
return v___x_198_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_toSyntax___lam__0(lean_object* v_e_210_, lean_object* v_a_211_, lean_object* v_a_212_, lean_object* v_a_213_, lean_object* v_a_214_, lean_object* v___y_215_, lean_object* v___y_216_){
_start:
{
lean_object* v_ref_218_; lean_object* v_quotContext_219_; lean_object* v_currMacroScope_220_; lean_object* v___x_221_; 
v_ref_218_ = lean_ctor_get(v___y_215_, 5);
v_quotContext_219_ = lean_ctor_get(v___y_215_, 10);
v_currMacroScope_220_ = lean_ctor_get(v___y_215_, 11);
lean_inc(v___y_216_);
lean_inc_ref(v___y_215_);
lean_inc(v_a_212_);
lean_inc_ref(v_a_211_);
lean_inc_ref(v_e_210_);
v___x_221_ = lean_infer_type(v_e_210_, v_a_211_, v_a_212_, v___y_215_, v___y_216_);
if (lean_obj_tag(v___x_221_) == 0)
{
lean_object* v_a_222_; uint8_t v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; lean_object* v___x_226_; lean_object* v___x_227_; lean_object* v___x_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; lean_object* v___x_234_; uint8_t v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; 
v_a_222_ = lean_ctor_get(v___x_221_, 0);
lean_inc(v_a_222_);
lean_dec_ref_known(v___x_221_, 1);
v___x_223_ = 0;
v___x_224_ = l_Lean_SourceInfo_fromRef(v_ref_218_, v___x_223_);
v___x_225_ = ((lean_object*)(lp_batteries_Lean_Expr_toSyntax___lam__0___closed__0));
lean_inc_n(v___x_224_, 2);
v___x_226_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_226_, 0, v___x_224_);
lean_ctor_set(v___x_226_, 1, v___x_225_);
v___x_227_ = lean_obj_once(&lp_batteries_Lean_Expr_toSyntax___lam__0___closed__2, &lp_batteries_Lean_Expr_toSyntax___lam__0___closed__2_once, _init_lp_batteries_Lean_Expr_toSyntax___lam__0___closed__2);
v___x_228_ = ((lean_object*)(lp_batteries_Lean_Expr_toSyntax___lam__0___closed__3));
lean_inc(v_currMacroScope_220_);
lean_inc(v_quotContext_219_);
v___x_229_ = l_Lean_addMacroScope(v_quotContext_219_, v___x_228_, v_currMacroScope_220_);
v___x_230_ = lean_box(0);
v___x_231_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_231_, 0, v___x_224_);
lean_ctor_set(v___x_231_, 1, v___x_227_);
lean_ctor_set(v___x_231_, 2, v___x_229_);
lean_ctor_set(v___x_231_, 3, v___x_230_);
v___x_232_ = ((lean_object*)(lp_batteries_Lean_Expr_toSyntax___lam__0___closed__8));
v___x_233_ = l_Lean_Syntax_node2(v___x_224_, v___x_232_, v___x_226_, v___x_231_);
v___x_234_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_234_, 0, v_a_222_);
v___x_235_ = 1;
v___x_236_ = lean_box(0);
lean_inc(v___x_233_);
v___x_237_ = l_Lean_Elab_Term_elabTermEnsuringType(v___x_233_, v___x_234_, v___x_235_, v___x_235_, v___x_236_, v_a_213_, v_a_214_, v_a_211_, v_a_212_, v___y_215_, v___y_216_);
lean_dec(v___y_216_);
lean_dec_ref(v___y_215_);
if (lean_obj_tag(v___x_237_) == 0)
{
lean_object* v_a_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_242_; uint8_t v_isShared_243_; uint8_t v_isSharedCheck_247_; 
v_a_238_ = lean_ctor_get(v___x_237_, 0);
lean_inc(v_a_238_);
lean_dec_ref_known(v___x_237_, 1);
v___x_239_ = l_Lean_Expr_mvarId_x21(v_a_238_);
lean_dec(v_a_238_);
v___x_240_ = lp_batteries_Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0___redArg(v___x_239_, v_e_210_, v_a_212_);
v_isSharedCheck_247_ = !lean_is_exclusive(v___x_240_);
if (v_isSharedCheck_247_ == 0)
{
lean_object* v_unused_248_; 
v_unused_248_ = lean_ctor_get(v___x_240_, 0);
lean_dec(v_unused_248_);
v___x_242_ = v___x_240_;
v_isShared_243_ = v_isSharedCheck_247_;
goto v_resetjp_241_;
}
else
{
lean_dec(v___x_240_);
v___x_242_ = lean_box(0);
v_isShared_243_ = v_isSharedCheck_247_;
goto v_resetjp_241_;
}
v_resetjp_241_:
{
lean_object* v___x_245_; 
if (v_isShared_243_ == 0)
{
lean_ctor_set(v___x_242_, 0, v___x_233_);
v___x_245_ = v___x_242_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_246_; 
v_reuseFailAlloc_246_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_246_, 0, v___x_233_);
v___x_245_ = v_reuseFailAlloc_246_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
return v___x_245_;
}
}
}
else
{
lean_object* v_a_249_; lean_object* v___x_251_; uint8_t v_isShared_252_; uint8_t v_isSharedCheck_256_; 
lean_dec(v___x_233_);
lean_dec_ref(v_e_210_);
v_a_249_ = lean_ctor_get(v___x_237_, 0);
v_isSharedCheck_256_ = !lean_is_exclusive(v___x_237_);
if (v_isSharedCheck_256_ == 0)
{
v___x_251_ = v___x_237_;
v_isShared_252_ = v_isSharedCheck_256_;
goto v_resetjp_250_;
}
else
{
lean_inc(v_a_249_);
lean_dec(v___x_237_);
v___x_251_ = lean_box(0);
v_isShared_252_ = v_isSharedCheck_256_;
goto v_resetjp_250_;
}
v_resetjp_250_:
{
lean_object* v___x_254_; 
if (v_isShared_252_ == 0)
{
v___x_254_ = v___x_251_;
goto v_reusejp_253_;
}
else
{
lean_object* v_reuseFailAlloc_255_; 
v_reuseFailAlloc_255_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_255_, 0, v_a_249_);
v___x_254_ = v_reuseFailAlloc_255_;
goto v_reusejp_253_;
}
v_reusejp_253_:
{
return v___x_254_;
}
}
}
}
else
{
lean_object* v_a_257_; lean_object* v___x_259_; uint8_t v_isShared_260_; uint8_t v_isSharedCheck_264_; 
lean_dec(v___y_216_);
lean_dec_ref(v___y_215_);
lean_dec_ref(v_e_210_);
v_a_257_ = lean_ctor_get(v___x_221_, 0);
v_isSharedCheck_264_ = !lean_is_exclusive(v___x_221_);
if (v_isSharedCheck_264_ == 0)
{
v___x_259_ = v___x_221_;
v_isShared_260_ = v_isSharedCheck_264_;
goto v_resetjp_258_;
}
else
{
lean_inc(v_a_257_);
lean_dec(v___x_221_);
v___x_259_ = lean_box(0);
v_isShared_260_ = v_isSharedCheck_264_;
goto v_resetjp_258_;
}
v_resetjp_258_:
{
lean_object* v___x_262_; 
if (v_isShared_260_ == 0)
{
v___x_262_ = v___x_259_;
goto v_reusejp_261_;
}
else
{
lean_object* v_reuseFailAlloc_263_; 
v_reuseFailAlloc_263_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_263_, 0, v_a_257_);
v___x_262_ = v_reuseFailAlloc_263_;
goto v_reusejp_261_;
}
v_reusejp_261_:
{
return v___x_262_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_toSyntax___lam__0___boxed(lean_object* v_e_265_, lean_object* v_a_266_, lean_object* v_a_267_, lean_object* v_a_268_, lean_object* v_a_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_){
_start:
{
lean_object* v_res_273_; 
v_res_273_ = lp_batteries_Lean_Expr_toSyntax___lam__0(v_e_265_, v_a_266_, v_a_267_, v_a_268_, v_a_269_, v___y_270_, v___y_271_);
lean_dec(v_a_269_);
lean_dec_ref(v_a_268_);
lean_dec(v_a_267_);
lean_dec_ref(v_a_266_);
return v_res_273_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_toSyntax(lean_object* v_e_274_, lean_object* v_a_275_, lean_object* v_a_276_, lean_object* v_a_277_, lean_object* v_a_278_, lean_object* v_a_279_, lean_object* v_a_280_){
_start:
{
lean_object* v___f_282_; lean_object* v___x_283_; 
lean_inc(v_a_276_);
lean_inc_ref(v_a_275_);
lean_inc(v_a_278_);
lean_inc_ref(v_a_277_);
v___f_282_ = lean_alloc_closure((void*)(lp_batteries_Lean_Expr_toSyntax___lam__0___boxed), 8, 5);
lean_closure_set(v___f_282_, 0, v_e_274_);
lean_closure_set(v___f_282_, 1, v_a_277_);
lean_closure_set(v___f_282_, 2, v_a_278_);
lean_closure_set(v___f_282_, 3, v_a_275_);
lean_closure_set(v___f_282_, 4, v_a_276_);
v___x_283_ = l_Lean_Core_withFreshMacroScope___redArg(v___f_282_, v_a_279_, v_a_280_);
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_toSyntax___boxed(lean_object* v_e_284_, lean_object* v_a_285_, lean_object* v_a_286_, lean_object* v_a_287_, lean_object* v_a_288_, lean_object* v_a_289_, lean_object* v_a_290_, lean_object* v_a_291_){
_start:
{
lean_object* v_res_292_; 
v_res_292_ = lp_batteries_Lean_Expr_toSyntax(v_e_284_, v_a_285_, v_a_286_, v_a_287_, v_a_288_, v_a_289_, v_a_290_);
lean_dec(v_a_290_);
lean_dec_ref(v_a_289_);
lean_dec(v_a_288_);
lean_dec_ref(v_a_287_);
lean_dec(v_a_286_);
lean_dec_ref(v_a_285_);
return v_res_292_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0(lean_object* v_mvarId_293_, lean_object* v_val_294_, lean_object* v___y_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_){
_start:
{
lean_object* v___x_302_; 
v___x_302_ = lp_batteries_Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0___redArg(v_mvarId_293_, v_val_294_, v___y_298_);
return v___x_302_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0___boxed(lean_object* v_mvarId_303_, lean_object* v_val_304_, lean_object* v___y_305_, lean_object* v___y_306_, lean_object* v___y_307_, lean_object* v___y_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_){
_start:
{
lean_object* v_res_312_; 
v_res_312_ = lp_batteries_Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0(v_mvarId_303_, v_val_304_, v___y_305_, v___y_306_, v___y_307_, v___y_308_, v___y_309_, v___y_310_);
lean_dec(v___y_310_);
lean_dec_ref(v___y_309_);
lean_dec(v___y_308_);
lean_dec_ref(v___y_307_);
lean_dec(v___y_306_);
lean_dec_ref(v___y_305_);
return v_res_312_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0(lean_object* v_00_u03b2_313_, lean_object* v_x_314_, lean_object* v_x_315_, lean_object* v_x_316_){
_start:
{
lean_object* v___x_317_; 
v___x_317_ = lp_batteries_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0___redArg(v_x_314_, v_x_315_, v_x_316_);
return v___x_317_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1(lean_object* v_00_u03b2_318_, lean_object* v_x_319_, size_t v_x_320_, size_t v_x_321_, lean_object* v_x_322_, lean_object* v_x_323_){
_start:
{
lean_object* v___x_324_; 
v___x_324_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1___redArg(v_x_319_, v_x_320_, v_x_321_, v_x_322_, v_x_323_);
return v___x_324_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1___boxed(lean_object* v_00_u03b2_325_, lean_object* v_x_326_, lean_object* v_x_327_, lean_object* v_x_328_, lean_object* v_x_329_, lean_object* v_x_330_){
_start:
{
size_t v_x_4027__boxed_331_; size_t v_x_4028__boxed_332_; lean_object* v_res_333_; 
v_x_4027__boxed_331_ = lean_unbox_usize(v_x_327_);
lean_dec(v_x_327_);
v_x_4028__boxed_332_ = lean_unbox_usize(v_x_328_);
lean_dec(v_x_328_);
v_res_333_ = lp_batteries_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1(v_00_u03b2_325_, v_x_326_, v_x_4027__boxed_331_, v_x_4028__boxed_332_, v_x_329_, v_x_330_);
return v_res_333_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__2(lean_object* v_00_u03b2_334_, lean_object* v_n_335_, lean_object* v_k_336_, lean_object* v_v_337_){
_start:
{
lean_object* v___x_338_; 
v___x_338_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__2___redArg(v_n_335_, v_k_336_, v_v_337_);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__3(lean_object* v_00_u03b2_339_, size_t v_depth_340_, lean_object* v_keys_341_, lean_object* v_vals_342_, lean_object* v_heq_343_, lean_object* v_i_344_, lean_object* v_entries_345_){
_start:
{
lean_object* v___x_346_; 
v___x_346_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__3___redArg(v_depth_340_, v_keys_341_, v_vals_342_, v_i_344_, v_entries_345_);
return v___x_346_;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__3___boxed(lean_object* v_00_u03b2_347_, lean_object* v_depth_348_, lean_object* v_keys_349_, lean_object* v_vals_350_, lean_object* v_heq_351_, lean_object* v_i_352_, lean_object* v_entries_353_){
_start:
{
size_t v_depth_boxed_354_; lean_object* v_res_355_; 
v_depth_boxed_354_ = lean_unbox_usize(v_depth_348_);
lean_dec(v_depth_348_);
v_res_355_ = lp_batteries___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__3(v_00_u03b2_347_, v_depth_boxed_354_, v_keys_349_, v_vals_350_, v_heq_351_, v_i_352_, v_entries_353_);
lean_dec_ref(v_vals_350_);
lean_dec_ref(v_keys_349_);
return v_res_355_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__2_spec__3(lean_object* v_00_u03b2_356_, lean_object* v_x_357_, lean_object* v_x_358_, lean_object* v_x_359_, lean_object* v_x_360_){
_start:
{
lean_object* v___x_361_; 
v___x_361_ = lp_batteries_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_Expr_toSyntax_spec__0_spec__0_spec__1_spec__2_spec__3___redArg(v_x_357_, v_x_358_, v_x_359_, v_x_360_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_withApp_x27_go___redArg(lean_object* v_k_362_, lean_object* v_a_363_, lean_object* v_a_364_, lean_object* v_a_365_){
_start:
{
switch(lean_obj_tag(v_a_363_))
{
case 10:
{
lean_object* v_expr_366_; 
v_expr_366_ = lean_ctor_get(v_a_363_, 1);
lean_inc_ref(v_expr_366_);
lean_dec_ref_known(v_a_363_, 2);
v_a_363_ = v_expr_366_;
goto _start;
}
case 5:
{
lean_object* v_fn_368_; lean_object* v_arg_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; 
v_fn_368_ = lean_ctor_get(v_a_363_, 0);
lean_inc_ref(v_fn_368_);
v_arg_369_ = lean_ctor_get(v_a_363_, 1);
lean_inc_ref(v_arg_369_);
lean_dec_ref_known(v_a_363_, 2);
v___x_370_ = lean_array_set(v_a_364_, v_a_365_, v_arg_369_);
v___x_371_ = lean_unsigned_to_nat(1u);
v___x_372_ = lean_nat_sub(v_a_365_, v___x_371_);
lean_dec(v_a_365_);
v_a_363_ = v_fn_368_;
v_a_364_ = v___x_370_;
v_a_365_ = v___x_372_;
goto _start;
}
default: 
{
lean_object* v___x_374_; 
lean_dec(v_a_365_);
v___x_374_ = lean_apply_2(v_k_362_, v_a_363_, v_a_364_);
return v___x_374_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_withApp_x27_go(lean_object* v_00_u03b1_375_, lean_object* v_k_376_, lean_object* v_a_377_, lean_object* v_a_378_, lean_object* v_a_379_){
_start:
{
lean_object* v___x_380_; 
v___x_380_ = lp_batteries_Lean_Expr_withApp_x27_go___redArg(v_k_376_, v_a_377_, v_a_378_, v_a_379_);
return v___x_380_;
}
}
static lean_object* _init_lp_batteries_Lean_Expr_withApp_x27___redArg___closed__0(void){
_start:
{
lean_object* v___x_381_; lean_object* v_dummy_382_; 
v___x_381_ = lean_box(0);
v_dummy_382_ = l_Lean_mkSort(v___x_381_);
return v_dummy_382_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_withApp_x27___redArg(lean_object* v_e_383_, lean_object* v_k_384_){
_start:
{
lean_object* v_dummy_385_; lean_object* v_nargs_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; lean_object* v___x_390_; 
v_dummy_385_ = lean_obj_once(&lp_batteries_Lean_Expr_withApp_x27___redArg___closed__0, &lp_batteries_Lean_Expr_withApp_x27___redArg___closed__0_once, _init_lp_batteries_Lean_Expr_withApp_x27___redArg___closed__0);
v_nargs_386_ = l_Lean_Expr_getAppNumArgs_x27(v_e_383_);
lean_inc(v_nargs_386_);
v___x_387_ = lean_mk_array(v_nargs_386_, v_dummy_385_);
v___x_388_ = lean_unsigned_to_nat(1u);
v___x_389_ = lean_nat_sub(v_nargs_386_, v___x_388_);
lean_dec(v_nargs_386_);
v___x_390_ = lp_batteries_Lean_Expr_withApp_x27_go___redArg(v_k_384_, v_e_383_, v___x_387_, v___x_389_);
return v___x_390_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_withApp_x27(lean_object* v_00_u03b1_391_, lean_object* v_e_392_, lean_object* v_k_393_){
_start:
{
lean_object* v_dummy_394_; lean_object* v_nargs_395_; lean_object* v___x_396_; lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; 
v_dummy_394_ = lean_obj_once(&lp_batteries_Lean_Expr_withApp_x27___redArg___closed__0, &lp_batteries_Lean_Expr_withApp_x27___redArg___closed__0_once, _init_lp_batteries_Lean_Expr_withApp_x27___redArg___closed__0);
v_nargs_395_ = l_Lean_Expr_getAppNumArgs_x27(v_e_392_);
lean_inc(v_nargs_395_);
v___x_396_ = lean_mk_array(v_nargs_395_, v_dummy_394_);
v___x_397_ = lean_unsigned_to_nat(1u);
v___x_398_ = lean_nat_sub(v_nargs_395_, v___x_397_);
lean_dec(v_nargs_395_);
v___x_399_ = lp_batteries_Lean_Expr_withApp_x27_go___redArg(v_k_393_, v_e_392_, v___x_396_, v___x_398_);
return v___x_399_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_getAppArgs_x27___lam__0(lean_object* v_x_400_, lean_object* v_as_401_){
_start:
{
lean_inc_ref(v_as_401_);
return v_as_401_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_getAppArgs_x27___lam__0___boxed(lean_object* v_x_402_, lean_object* v_as_403_){
_start:
{
lean_object* v_res_404_; 
v_res_404_ = lp_batteries_Lean_Expr_getAppArgs_x27___lam__0(v_x_402_, v_as_403_);
lean_dec_ref(v_as_403_);
lean_dec_ref(v_x_402_);
return v_res_404_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_getAppArgs_x27(lean_object* v_e_406_){
_start:
{
lean_object* v___f_407_; lean_object* v_dummy_408_; lean_object* v_nargs_409_; lean_object* v___x_410_; lean_object* v___x_411_; lean_object* v___x_412_; lean_object* v___x_413_; 
v___f_407_ = ((lean_object*)(lp_batteries_Lean_Expr_getAppArgs_x27___closed__0));
v_dummy_408_ = lean_obj_once(&lp_batteries_Lean_Expr_withApp_x27___redArg___closed__0, &lp_batteries_Lean_Expr_withApp_x27___redArg___closed__0_once, _init_lp_batteries_Lean_Expr_withApp_x27___redArg___closed__0);
v_nargs_409_ = l_Lean_Expr_getAppNumArgs_x27(v_e_406_);
lean_inc(v_nargs_409_);
v___x_410_ = lean_mk_array(v_nargs_409_, v_dummy_408_);
v___x_411_ = lean_unsigned_to_nat(1u);
v___x_412_ = lean_nat_sub(v_nargs_409_, v___x_411_);
lean_dec(v_nargs_409_);
v___x_413_ = lp_batteries_Lean_Expr_withApp_x27_go___redArg(v___f_407_, v_e_406_, v___x_410_, v___x_412_);
return v___x_413_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_traverseApp_x27___redArg___lam__0(lean_object* v_____do__lift_414_, lean_object* v_toPure_415_, lean_object* v_____do__lift_416_){
_start:
{
lean_object* v___x_417_; lean_object* v___x_418_; 
v___x_417_ = l_Lean_mkAppN(v_____do__lift_414_, v_____do__lift_416_);
v___x_418_ = lean_apply_2(v_toPure_415_, lean_box(0), v___x_417_);
return v___x_418_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_traverseApp_x27___redArg___lam__0___boxed(lean_object* v_____do__lift_419_, lean_object* v_toPure_420_, lean_object* v_____do__lift_421_){
_start:
{
lean_object* v_res_422_; 
v_res_422_ = lp_batteries_Lean_Expr_traverseApp_x27___redArg___lam__0(v_____do__lift_419_, v_toPure_420_, v_____do__lift_421_);
lean_dec_ref(v_____do__lift_421_);
return v_res_422_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_traverseApp_x27___redArg___lam__1(lean_object* v_toPure_423_, lean_object* v_args_424_, lean_object* v_inst_425_, lean_object* v_f_426_, lean_object* v_toBind_427_, lean_object* v_____do__lift_428_){
_start:
{
lean_object* v___f_429_; size_t v_sz_430_; size_t v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; 
v___f_429_ = lean_alloc_closure((void*)(lp_batteries_Lean_Expr_traverseApp_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_429_, 0, v_____do__lift_428_);
lean_closure_set(v___f_429_, 1, v_toPure_423_);
v_sz_430_ = lean_array_size(v_args_424_);
v___x_431_ = ((size_t)0ULL);
v___x_432_ = l___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map(lean_box(0), lean_box(0), lean_box(0), v_inst_425_, v_f_426_, v_sz_430_, v___x_431_, v_args_424_);
v___x_433_ = lean_apply_4(v_toBind_427_, lean_box(0), lean_box(0), v___x_432_, v___f_429_);
return v___x_433_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_traverseApp_x27___redArg___lam__2(lean_object* v_toPure_434_, lean_object* v_inst_435_, lean_object* v_f_436_, lean_object* v_toBind_437_, lean_object* v_fn_438_, lean_object* v_args_439_){
_start:
{
lean_object* v___f_440_; lean_object* v___x_441_; lean_object* v___x_442_; 
lean_inc(v_toBind_437_);
lean_inc(v_f_436_);
v___f_440_ = lean_alloc_closure((void*)(lp_batteries_Lean_Expr_traverseApp_x27___redArg___lam__1), 6, 5);
lean_closure_set(v___f_440_, 0, v_toPure_434_);
lean_closure_set(v___f_440_, 1, v_args_439_);
lean_closure_set(v___f_440_, 2, v_inst_435_);
lean_closure_set(v___f_440_, 3, v_f_436_);
lean_closure_set(v___f_440_, 4, v_toBind_437_);
v___x_441_ = lean_apply_1(v_f_436_, v_fn_438_);
v___x_442_ = lean_apply_4(v_toBind_437_, lean_box(0), lean_box(0), v___x_441_, v___f_440_);
return v___x_442_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_traverseApp_x27___redArg(lean_object* v_inst_443_, lean_object* v_f_444_, lean_object* v_e_445_){
_start:
{
lean_object* v_toApplicative_446_; lean_object* v_toBind_447_; lean_object* v_toPure_448_; lean_object* v___f_449_; lean_object* v_dummy_450_; lean_object* v_nargs_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; 
v_toApplicative_446_ = lean_ctor_get(v_inst_443_, 0);
v_toBind_447_ = lean_ctor_get(v_inst_443_, 1);
lean_inc(v_toBind_447_);
v_toPure_448_ = lean_ctor_get(v_toApplicative_446_, 1);
lean_inc(v_toPure_448_);
v___f_449_ = lean_alloc_closure((void*)(lp_batteries_Lean_Expr_traverseApp_x27___redArg___lam__2), 6, 4);
lean_closure_set(v___f_449_, 0, v_toPure_448_);
lean_closure_set(v___f_449_, 1, v_inst_443_);
lean_closure_set(v___f_449_, 2, v_f_444_);
lean_closure_set(v___f_449_, 3, v_toBind_447_);
v_dummy_450_ = lean_obj_once(&lp_batteries_Lean_Expr_withApp_x27___redArg___closed__0, &lp_batteries_Lean_Expr_withApp_x27___redArg___closed__0_once, _init_lp_batteries_Lean_Expr_withApp_x27___redArg___closed__0);
v_nargs_451_ = l_Lean_Expr_getAppNumArgs_x27(v_e_445_);
lean_inc(v_nargs_451_);
v___x_452_ = lean_mk_array(v_nargs_451_, v_dummy_450_);
v___x_453_ = lean_unsigned_to_nat(1u);
v___x_454_ = lean_nat_sub(v_nargs_451_, v___x_453_);
lean_dec(v_nargs_451_);
v___x_455_ = lp_batteries_Lean_Expr_withApp_x27_go___redArg(v___f_449_, v_e_445_, v___x_452_, v___x_454_);
return v___x_455_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_traverseApp_x27(lean_object* v_m_456_, lean_object* v_inst_457_, lean_object* v_f_458_, lean_object* v_e_459_){
_start:
{
lean_object* v___x_460_; 
v___x_460_ = lp_batteries_Lean_Expr_traverseApp_x27___redArg(v_inst_457_, v_f_458_, v_e_459_);
return v___x_460_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_withAppRev_x27_go___redArg(lean_object* v_k_461_, lean_object* v_a_462_, lean_object* v_a_463_){
_start:
{
switch(lean_obj_tag(v_a_462_))
{
case 10:
{
lean_object* v_expr_464_; 
v_expr_464_ = lean_ctor_get(v_a_462_, 1);
lean_inc_ref(v_expr_464_);
lean_dec_ref_known(v_a_462_, 2);
v_a_462_ = v_expr_464_;
goto _start;
}
case 5:
{
lean_object* v_fn_466_; lean_object* v_arg_467_; lean_object* v___x_468_; 
v_fn_466_ = lean_ctor_get(v_a_462_, 0);
lean_inc_ref(v_fn_466_);
v_arg_467_ = lean_ctor_get(v_a_462_, 1);
lean_inc_ref(v_arg_467_);
lean_dec_ref_known(v_a_462_, 2);
v___x_468_ = lean_array_push(v_a_463_, v_arg_467_);
v_a_462_ = v_fn_466_;
v_a_463_ = v___x_468_;
goto _start;
}
default: 
{
lean_object* v___x_470_; 
v___x_470_ = lean_apply_2(v_k_461_, v_a_462_, v_a_463_);
return v___x_470_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_withAppRev_x27_go(lean_object* v_00_u03b1_471_, lean_object* v_k_472_, lean_object* v_a_473_, lean_object* v_a_474_){
_start:
{
lean_object* v___x_475_; 
v___x_475_ = lp_batteries_Lean_Expr_withAppRev_x27_go___redArg(v_k_472_, v_a_473_, v_a_474_);
return v___x_475_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_withAppRev_x27___redArg(lean_object* v_e_476_, lean_object* v_k_477_){
_start:
{
lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; 
v___x_478_ = l_Lean_Expr_getAppNumArgs_x27(v_e_476_);
v___x_479_ = lean_mk_empty_array_with_capacity(v___x_478_);
lean_dec(v___x_478_);
v___x_480_ = lp_batteries_Lean_Expr_withAppRev_x27_go___redArg(v_k_477_, v_e_476_, v___x_479_);
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_withAppRev_x27(lean_object* v_00_u03b1_481_, lean_object* v_e_482_, lean_object* v_k_483_){
_start:
{
lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; 
v___x_484_ = l_Lean_Expr_getAppNumArgs_x27(v_e_482_);
v___x_485_ = lean_mk_empty_array_with_capacity(v___x_484_);
lean_dec(v___x_484_);
v___x_486_ = lp_batteries_Lean_Expr_withAppRev_x27_go___redArg(v_k_483_, v_e_482_, v___x_485_);
return v___x_486_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_getAppRevArgs_x27(lean_object* v_e_487_){
_start:
{
lean_object* v___f_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; 
v___f_488_ = ((lean_object*)(lp_batteries_Lean_Expr_getAppArgs_x27___closed__0));
v___x_489_ = l_Lean_Expr_getAppNumArgs_x27(v_e_487_);
v___x_490_ = lean_mk_empty_array_with_capacity(v___x_489_);
lean_dec(v___x_489_);
v___x_491_ = lp_batteries_Lean_Expr_withAppRev_x27_go___redArg(v___f_488_, v_e_487_, v___x_490_);
return v___x_491_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_getRevArgD_x27(lean_object* v_x_492_, lean_object* v_x_493_, lean_object* v_x_494_){
_start:
{
switch(lean_obj_tag(v_x_492_))
{
case 10:
{
lean_object* v_expr_495_; 
v_expr_495_ = lean_ctor_get(v_x_492_, 1);
v_x_492_ = v_expr_495_;
goto _start;
}
case 5:
{
lean_object* v_fn_497_; lean_object* v_arg_498_; lean_object* v_zero_499_; uint8_t v_isZero_500_; 
v_fn_497_ = lean_ctor_get(v_x_492_, 0);
v_arg_498_ = lean_ctor_get(v_x_492_, 1);
v_zero_499_ = lean_unsigned_to_nat(0u);
v_isZero_500_ = lean_nat_dec_eq(v_x_493_, v_zero_499_);
if (v_isZero_500_ == 1)
{
lean_dec(v_x_493_);
lean_inc_ref(v_arg_498_);
return v_arg_498_;
}
else
{
lean_object* v_one_501_; lean_object* v_n_502_; 
v_one_501_ = lean_unsigned_to_nat(1u);
v_n_502_ = lean_nat_sub(v_x_493_, v_one_501_);
lean_dec(v_x_493_);
v_x_492_ = v_fn_497_;
v_x_493_ = v_n_502_;
goto _start;
}
}
default: 
{
lean_dec(v_x_493_);
lean_inc_ref(v_x_494_);
return v_x_494_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_getRevArgD_x27___boxed(lean_object* v_x_504_, lean_object* v_x_505_, lean_object* v_x_506_){
_start:
{
lean_object* v_res_507_; 
v_res_507_ = lp_batteries_Lean_Expr_getRevArgD_x27(v_x_504_, v_x_505_, v_x_506_);
lean_dec_ref(v_x_506_);
lean_dec_ref(v_x_504_);
return v_res_507_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_getArgD_x27(lean_object* v_e_508_, lean_object* v_i_509_, lean_object* v_v_u2080_510_, lean_object* v_n_511_){
_start:
{
lean_object* v___x_512_; lean_object* v___x_513_; lean_object* v___x_514_; lean_object* v___x_515_; 
v___x_512_ = lean_nat_sub(v_n_511_, v_i_509_);
v___x_513_ = lean_unsigned_to_nat(1u);
v___x_514_ = lean_nat_sub(v___x_512_, v___x_513_);
lean_dec(v___x_512_);
v___x_515_ = lp_batteries_Lean_Expr_getRevArgD_x27(v_e_508_, v___x_514_, v_v_u2080_510_);
return v___x_515_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_getArgD_x27___boxed(lean_object* v_e_516_, lean_object* v_i_517_, lean_object* v_v_u2080_518_, lean_object* v_n_519_){
_start:
{
lean_object* v_res_520_; 
v_res_520_ = lp_batteries_Lean_Expr_getArgD_x27(v_e_516_, v_i_517_, v_v_u2080_518_, v_n_519_);
lean_dec(v_n_519_);
lean_dec_ref(v_v_u2080_518_);
lean_dec(v_i_517_);
lean_dec_ref(v_e_516_);
return v_res_520_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_Expr_isAppOf_x27(lean_object* v_e_521_, lean_object* v_n_522_){
_start:
{
lean_object* v___x_523_; 
v___x_523_ = l_Lean_Expr_getAppFn_x27(v_e_521_);
if (lean_obj_tag(v___x_523_) == 4)
{
lean_object* v_declName_524_; uint8_t v___x_525_; 
v_declName_524_ = lean_ctor_get(v___x_523_, 0);
lean_inc(v_declName_524_);
lean_dec_ref_known(v___x_523_, 2);
v___x_525_ = lean_name_eq(v_declName_524_, v_n_522_);
lean_dec(v_declName_524_);
return v___x_525_;
}
else
{
uint8_t v___x_526_; 
lean_dec_ref(v___x_523_);
v___x_526_ = 0;
return v___x_526_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_isAppOf_x27___boxed(lean_object* v_e_527_, lean_object* v_n_528_){
_start:
{
uint8_t v_res_529_; lean_object* v_r_530_; 
v_res_529_ = lp_batteries_Lean_Expr_isAppOf_x27(v_e_527_, v_n_528_);
lean_dec(v_n_528_);
lean_dec_ref(v_e_527_);
v_r_530_ = lean_box(v_res_529_);
return v_r_530_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Lean_Expr_natLit_x21_spec__0(lean_object* v_msg_531_){
_start:
{
lean_object* v___x_532_; lean_object* v___x_533_; 
v___x_532_ = lean_unsigned_to_nat(0u);
v___x_533_ = lean_panic_fn_borrowed(v___x_532_, v_msg_531_);
return v___x_533_;
}
}
static lean_object* _init_lp_batteries_Lean_Expr_natLit_x21___closed__3(void){
_start:
{
lean_object* v___x_537_; lean_object* v___x_538_; lean_object* v___x_539_; lean_object* v___x_540_; lean_object* v___x_541_; lean_object* v___x_542_; 
v___x_537_ = ((lean_object*)(lp_batteries_Lean_Expr_natLit_x21___closed__2));
v___x_538_ = lean_unsigned_to_nat(30u);
v___x_539_ = lean_unsigned_to_nat(94u);
v___x_540_ = ((lean_object*)(lp_batteries_Lean_Expr_natLit_x21___closed__1));
v___x_541_ = ((lean_object*)(lp_batteries_Lean_Expr_natLit_x21___closed__0));
v___x_542_ = l_mkPanicMessageWithDecl(v___x_541_, v___x_540_, v___x_539_, v___x_538_, v___x_537_);
return v___x_542_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_natLit_x21(lean_object* v_x_543_){
_start:
{
if (lean_obj_tag(v_x_543_) == 9)
{
lean_object* v_a_547_; 
v_a_547_ = lean_ctor_get(v_x_543_, 0);
if (lean_obj_tag(v_a_547_) == 0)
{
lean_object* v_val_548_; 
v_val_548_ = lean_ctor_get(v_a_547_, 0);
lean_inc(v_val_548_);
return v_val_548_;
}
else
{
goto v___jp_544_;
}
}
else
{
goto v___jp_544_;
}
v___jp_544_:
{
lean_object* v___x_545_; lean_object* v___x_546_; 
v___x_545_ = lean_obj_once(&lp_batteries_Lean_Expr_natLit_x21___closed__3, &lp_batteries_Lean_Expr_natLit_x21___closed__3_once, _init_lp_batteries_Lean_Expr_natLit_x21___closed__3);
v___x_546_ = lp_batteries_panic___at___00Lean_Expr_natLit_x21_spec__0(v___x_545_);
return v___x_546_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_natLit_x21___boxed(lean_object* v_x_549_){
_start:
{
lean_object* v_res_550_; 
v_res_550_ = lp_batteries_Lean_Expr_natLit_x21(v_x_549_);
lean_dec_ref(v_x_549_);
return v_res_550_;
}
}
LEAN_EXPORT lean_object* lp_batteries_panic___at___00Lean_Expr_intLit_x21_spec__0(lean_object* v_msg_551_){
_start:
{
lean_object* v___x_552_; lean_object* v___x_553_; 
v___x_552_ = l_Int_instInhabited;
v___x_553_ = lean_panic_fn_borrowed(v___x_552_, v_msg_551_);
return v___x_553_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nat_cast___at___00Lean_Expr_intLit_x21_spec__1(lean_object* v_a_554_){
_start:
{
lean_object* v___x_555_; 
v___x_555_ = lean_nat_to_int(v_a_554_);
return v___x_555_;
}
}
static lean_object* _init_lp_batteries_Lean_Expr_intLit_x21___closed__7(void){
_start:
{
lean_object* v___x_567_; lean_object* v___x_568_; lean_object* v___x_569_; lean_object* v___x_570_; lean_object* v___x_571_; lean_object* v___x_572_; 
v___x_567_ = ((lean_object*)(lp_batteries_Lean_Expr_intLit_x21___closed__6));
v___x_568_ = lean_unsigned_to_nat(4u);
v___x_569_ = lean_unsigned_to_nat(104u);
v___x_570_ = ((lean_object*)(lp_batteries_Lean_Expr_intLit_x21___closed__5));
v___x_571_ = ((lean_object*)(lp_batteries_Lean_Expr_natLit_x21___closed__0));
v___x_572_ = l_mkPanicMessageWithDecl(v___x_571_, v___x_570_, v___x_569_, v___x_568_, v___x_567_);
return v___x_572_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_intLit_x21(lean_object* v_e_573_){
_start:
{
lean_object* v___x_574_; lean_object* v___x_575_; uint8_t v___x_576_; 
v___x_574_ = ((lean_object*)(lp_batteries_Lean_Expr_intLit_x21___closed__2));
v___x_575_ = lean_unsigned_to_nat(1u);
v___x_576_ = l_Lean_Expr_isAppOfArity(v_e_573_, v___x_574_, v___x_575_);
if (v___x_576_ == 0)
{
lean_object* v___x_577_; uint8_t v___x_578_; 
v___x_577_ = ((lean_object*)(lp_batteries_Lean_Expr_intLit_x21___closed__4));
v___x_578_ = l_Lean_Expr_isAppOfArity(v_e_573_, v___x_577_, v___x_575_);
if (v___x_578_ == 0)
{
lean_object* v___x_579_; lean_object* v___x_580_; 
v___x_579_ = lean_obj_once(&lp_batteries_Lean_Expr_intLit_x21___closed__7, &lp_batteries_Lean_Expr_intLit_x21___closed__7_once, _init_lp_batteries_Lean_Expr_intLit_x21___closed__7);
v___x_580_ = lp_batteries_panic___at___00Lean_Expr_intLit_x21_spec__0(v___x_579_);
return v___x_580_;
}
else
{
lean_object* v___x_581_; lean_object* v___x_582_; lean_object* v___x_583_; 
v___x_581_ = l_Lean_Expr_appArg_x21(v_e_573_);
v___x_582_ = lp_batteries_Lean_Expr_natLit_x21(v___x_581_);
lean_dec_ref(v___x_581_);
v___x_583_ = l_Int_negOfNat(v___x_582_);
lean_dec(v___x_582_);
return v___x_583_;
}
}
else
{
lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; 
v___x_584_ = l_Lean_Expr_appArg_x21(v_e_573_);
v___x_585_ = lp_batteries_Lean_Expr_natLit_x21(v___x_584_);
lean_dec_ref(v___x_584_);
v___x_586_ = lean_nat_to_int(v___x_585_);
return v___x_586_;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_intLit_x21___boxed(lean_object* v_e_587_){
_start:
{
lean_object* v_res_588_; 
v_res_588_ = lp_batteries_Lean_Expr_intLit_x21(v_e_587_);
lean_dec_ref(v_e_587_);
return v_res_588_;
}
}
LEAN_EXPORT uint8_t lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__0(lean_object* v_x_589_){
_start:
{
uint8_t v___x_590_; 
v___x_590_ = 0;
return v___x_590_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__0___boxed(lean_object* v_x_591_){
_start:
{
uint8_t v_res_592_; lean_object* v_r_593_; 
v_res_592_ = lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__0(v_x_591_);
lean_dec(v_x_591_);
v_r_593_ = lean_box(v_res_592_);
return v_r_593_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1(uint8_t v___x_599_, lean_object* v___x_600_, lean_object* v_tk_601_, uint8_t v___x_602_, lean_object* v_fvar_603_, lean_object* v___y_604_, lean_object* v___y_605_, lean_object* v___y_606_, lean_object* v___y_607_, lean_object* v___y_608_, lean_object* v___y_609_){
_start:
{
if (v___x_599_ == 0)
{
lean_object* v___x_611_; lean_object* v___x_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; 
v___x_611_ = lean_box(0);
v___x_612_ = l_Lean_SourceInfo_fromRef(v___x_611_, v___x_599_);
v___x_613_ = ((lean_object*)(lp_batteries_Lean_Expr_toSyntax___lam__0___closed__5));
v___x_614_ = ((lean_object*)(lp_batteries_Lean_Expr_toSyntax___lam__0___closed__6));
v___x_615_ = ((lean_object*)(lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__0));
v___x_616_ = l_Lean_Name_mkStr4(v___x_600_, v___x_613_, v___x_614_, v___x_615_);
v___x_617_ = l_Lean_SourceInfo_fromRef(v_tk_601_, v___x_602_);
v___x_618_ = ((lean_object*)(lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__1));
v___x_619_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_619_, 0, v___x_617_);
lean_ctor_set(v___x_619_, 1, v___x_618_);
v___x_620_ = l_Lean_Syntax_node1(v___x_612_, v___x_616_, v___x_619_);
v___x_621_ = l_Lean_Elab_Term_addLocalVarInfo(v___x_620_, v_fvar_603_, v___y_604_, v___y_605_, v___y_606_, v___y_607_, v___y_608_, v___y_609_);
return v___x_621_;
}
else
{
lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; uint8_t v___x_625_; 
v___x_622_ = lean_unsigned_to_nat(0u);
v___x_623_ = l_Lean_Syntax_getArg(v_tk_601_, v___x_622_);
v___x_624_ = ((lean_object*)(lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__3));
lean_inc(v___x_623_);
v___x_625_ = l_Lean_Syntax_isOfKind(v___x_623_, v___x_624_);
if (v___x_625_ == 0)
{
lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; 
lean_dec(v___x_623_);
v___x_626_ = lean_box(0);
v___x_627_ = l_Lean_SourceInfo_fromRef(v___x_626_, v___x_625_);
v___x_628_ = ((lean_object*)(lp_batteries_Lean_Expr_toSyntax___lam__0___closed__5));
v___x_629_ = ((lean_object*)(lp_batteries_Lean_Expr_toSyntax___lam__0___closed__6));
v___x_630_ = ((lean_object*)(lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__0));
v___x_631_ = l_Lean_Name_mkStr4(v___x_600_, v___x_628_, v___x_629_, v___x_630_);
v___x_632_ = l_Lean_SourceInfo_fromRef(v_tk_601_, v___x_602_);
v___x_633_ = ((lean_object*)(lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___closed__1));
v___x_634_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_634_, 0, v___x_632_);
lean_ctor_set(v___x_634_, 1, v___x_633_);
v___x_635_ = l_Lean_Syntax_node1(v___x_627_, v___x_631_, v___x_634_);
v___x_636_ = l_Lean_Elab_Term_addLocalVarInfo(v___x_635_, v_fvar_603_, v___y_604_, v___y_605_, v___y_606_, v___y_607_, v___y_608_, v___y_609_);
return v___x_636_;
}
else
{
lean_object* v___x_637_; 
lean_dec_ref(v___x_600_);
v___x_637_ = l_Lean_Elab_Term_addLocalVarInfo(v___x_623_, v_fvar_603_, v___y_604_, v___y_605_, v___y_606_, v___y_607_, v___y_608_, v___y_609_);
return v___x_637_;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___boxed(lean_object* v___x_638_, lean_object* v___x_639_, lean_object* v_tk_640_, lean_object* v___x_641_, lean_object* v_fvar_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_){
_start:
{
uint8_t v___x_4940__boxed_650_; uint8_t v___x_4942__boxed_651_; lean_object* v_res_652_; 
v___x_4940__boxed_650_ = lean_unbox(v___x_638_);
v___x_4942__boxed_651_ = lean_unbox(v___x_641_);
v_res_652_ = lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1(v___x_4940__boxed_650_, v___x_639_, v_tk_640_, v___x_4942__boxed_651_, v_fvar_642_, v___y_643_, v___y_644_, v___y_645_, v___y_646_, v___y_647_, v___y_648_);
lean_dec(v___y_648_);
lean_dec_ref(v___y_647_);
lean_dec(v___y_646_);
lean_dec_ref(v___y_645_);
lean_dec(v___y_644_);
lean_dec_ref(v___y_643_);
lean_dec(v_tk_640_);
return v_res_652_;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent(lean_object* v_fvar_671_, lean_object* v_tk_672_, lean_object* v_a_673_, lean_object* v_a_674_, lean_object* v_a_675_, lean_object* v_a_676_){
_start:
{
lean_object* v___x_678_; lean_object* v___x_679_; uint8_t v___x_680_; uint8_t v___x_681_; lean_object* v___x_682_; lean_object* v___x_683_; lean_object* v___y_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___x_687_; 
v___x_678_ = ((lean_object*)(lp_batteries_Lean_Expr_toSyntax___lam__0___closed__4));
v___x_679_ = ((lean_object*)(lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__2));
lean_inc(v_tk_672_);
v___x_680_ = l_Lean_Syntax_isOfKind(v_tk_672_, v___x_679_);
v___x_681_ = 1;
v___x_682_ = lean_box(v___x_680_);
v___x_683_ = lean_box(v___x_681_);
v___y_684_ = lean_alloc_closure((void*)(lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___lam__1___boxed), 12, 5);
lean_closure_set(v___y_684_, 0, v___x_682_);
lean_closure_set(v___y_684_, 1, v___x_678_);
lean_closure_set(v___y_684_, 2, v_tk_672_);
lean_closure_set(v___y_684_, 3, v___x_683_);
lean_closure_set(v___y_684_, 4, v_fvar_671_);
v___x_685_ = ((lean_object*)(lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__4));
v___x_686_ = ((lean_object*)(lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___closed__5));
v___x_687_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___y_684_, v___x_685_, v___x_686_, v_a_673_, v_a_674_, v_a_675_, v_a_676_);
if (lean_obj_tag(v___x_687_) == 0)
{
lean_object* v___x_689_; uint8_t v_isShared_690_; uint8_t v_isSharedCheck_695_; 
v_isSharedCheck_695_ = !lean_is_exclusive(v___x_687_);
if (v_isSharedCheck_695_ == 0)
{
lean_object* v_unused_696_; 
v_unused_696_ = lean_ctor_get(v___x_687_, 0);
lean_dec(v_unused_696_);
v___x_689_ = v___x_687_;
v_isShared_690_ = v_isSharedCheck_695_;
goto v_resetjp_688_;
}
else
{
lean_dec(v___x_687_);
v___x_689_ = lean_box(0);
v_isShared_690_ = v_isSharedCheck_695_;
goto v_resetjp_688_;
}
v_resetjp_688_:
{
lean_object* v___x_691_; lean_object* v___x_693_; 
v___x_691_ = lean_box(0);
if (v_isShared_690_ == 0)
{
lean_ctor_set(v___x_689_, 0, v___x_691_);
v___x_693_ = v___x_689_;
goto v_reusejp_692_;
}
else
{
lean_object* v_reuseFailAlloc_694_; 
v_reuseFailAlloc_694_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_694_, 0, v___x_691_);
v___x_693_ = v_reuseFailAlloc_694_;
goto v_reusejp_692_;
}
v_reusejp_692_:
{
return v___x_693_;
}
}
}
else
{
lean_object* v_a_697_; lean_object* v___x_699_; uint8_t v_isShared_700_; uint8_t v_isSharedCheck_704_; 
v_a_697_ = lean_ctor_get(v___x_687_, 0);
v_isSharedCheck_704_ = !lean_is_exclusive(v___x_687_);
if (v_isSharedCheck_704_ == 0)
{
v___x_699_ = v___x_687_;
v_isShared_700_ = v_isSharedCheck_704_;
goto v_resetjp_698_;
}
else
{
lean_inc(v_a_697_);
lean_dec(v___x_687_);
v___x_699_ = lean_box(0);
v_isShared_700_ = v_isSharedCheck_704_;
goto v_resetjp_698_;
}
v_resetjp_698_:
{
lean_object* v___x_702_; 
if (v_isShared_700_ == 0)
{
v___x_702_ = v___x_699_;
goto v_reusejp_701_;
}
else
{
lean_object* v_reuseFailAlloc_703_; 
v_reuseFailAlloc_703_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_703_, 0, v_a_697_);
v___x_702_ = v_reuseFailAlloc_703_;
goto v_reusejp_701_;
}
v_reusejp_701_:
{
return v___x_702_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent___boxed(lean_object* v_fvar_705_, lean_object* v_tk_706_, lean_object* v_a_707_, lean_object* v_a_708_, lean_object* v_a_709_, lean_object* v_a_710_, lean_object* v_a_711_){
_start:
{
lean_object* v_res_712_; 
v_res_712_ = lp_batteries_Lean_Expr_addLocalVarInfoForBinderIdent(v_fvar_705_, v_tk_706_, v_a_707_, v_a_708_, v_a_709_, v_a_710_);
lean_dec(v_a_710_);
lean_dec_ref(v_a_709_);
lean_dec(v_a_708_);
lean_dec_ref(v_a_707_);
return v_res_712_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Term(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Binders(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_batteries_Batteries_Lean_Expr(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Term(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Binders(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_batteries_Batteries_Lean_Expr(uint8_t builtin) {
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
lean_object* initialize_Lean_Elab_Term(uint8_t builtin);
lean_object* initialize_Lean_Elab_Binders(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_Expr(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Term(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Binders(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Lean_Expr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_batteries_Batteries_Lean_Expr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_batteries_Batteries_Lean_Expr(builtin);
}
#ifdef __cplusplus
}
#endif
