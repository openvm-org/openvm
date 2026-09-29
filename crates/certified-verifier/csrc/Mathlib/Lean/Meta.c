// Lean compiler output
// Module: Mathlib.Lean.Meta
// Imports: public import Init public meta import Init public import Mathlib.Init public import Lean.Elab.Term public import Lean.Elab.Tactic.Basic public import Lean.Meta.Tactic.Assert public import Lean.Meta.Tactic.Clear
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
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_evalTactic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_TermElabM_run___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_getType(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_saveState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_SavedState_restore___redArg(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Elab_isAbortTacticException(lean_object*);
lean_object* l_Lean_Elab_Tactic_getUnsolvedGoals(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Exception_isInterrupt(lean_object*);
uint8_t l_Lean_Exception_isRuntime(lean_object*);
lean_object* l_Lean_Meta_intro1Core(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_define(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_let(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_let___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__0 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__0_value;
static const lean_string_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__1 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__1_value;
static const lean_string_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__2 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__2_value;
static const lean_string_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "refine"};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__3 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__3_value;
static const lean_ctor_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(49, 130, 130, 160, 131, 48, 178, 245)}};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__4 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__4_value;
static const lean_string_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__5 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__5_value;
static const lean_string_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "anonymousCtor"};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__6 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__6_value;
static const lean_ctor_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(56, 53, 154, 97, 179, 232, 94, 186)}};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__7 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__7_value;
static const lean_string_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟨"};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__8 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__8_value;
static const lean_string_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__9 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__9_value;
static const lean_ctor_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__10 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__10_value;
static const lean_string_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "syntheticHole"};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__11 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__11_value;
static const lean_ctor_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__12_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__12_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__12_value_aux_0),((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__12_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__12_value_aux_1),((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__12_value_aux_2),((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(218, 189, 67, 60, 211, 196, 112, 165)}};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__12 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__12_value;
static const lean_string_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "\?"};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__13 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__13_value;
static const lean_string_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "_"};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__14 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__14_value;
static const lean_string_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__15 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__15_value;
static const lean_string_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⟩"};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__16 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__16_value;
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_existsi_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_existsi_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_existsi_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_existsi_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__5_spec__6___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__6___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__0___boxed(lean_object*);
static const lean_string_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 22, .m_capacity = 22, .m_length = 21, .m_data = "expected two subgoals"};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__0 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__0_value;
static lean_once_cell_t lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__1;
static const lean_closure_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__2 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__2_value;
static const lean_closure_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___boxed, .m_arity = 9, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__3 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__3_value;
static const lean_array_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__4 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__4_value;
static const lean_ctor_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*8 + 16, .m_other = 8, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__2_value),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__4_value),LEAN_SCALAR_PTR_LITERAL(1, 1, 1, 1, 0, 0, 0, 0),LEAN_SCALAR_PTR_LITERAL(1, 0, 1, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__5 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__5_value;
static const lean_ctor_object lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*7 + 0, .m_other = 7, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__6 = (const lean_object*)&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_existsi(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_existsi___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_existsi_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_existsi_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__6(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_MVarId_intros_x21_run(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_MVarId_intros_x21_run___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_mathlib_Lean_MVarId_intros_x21___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_MVarId_intros_x21___closed__0 = (const lean_object*)&lp_mathlib_Lean_MVarId_intros_x21___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_intros_x21(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_intros_x21___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_getType_x27_x27_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_getType_x27_x27_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_getType_x27_x27_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_getType_x27_x27_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_getType_x27_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_getType_x27_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore_x27___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_run__for___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_run__for___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Lean_Elab_Tactic_run__for___redArg___lam__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 8, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(1, 0, 0, 0, 0, 0, 0, 0)}};
static const lean_object* lp_mathlib_Lean_Elab_Tactic_run__for___redArg___lam__1___closed__0 = (const lean_object*)&lp_mathlib_Lean_Elab_Tactic_run__for___redArg___lam__1___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_run__for___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_run__for___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_run__for___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_run__for___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_run__for(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_run__for___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_let(lean_object* v_g_1_, lean_object* v_h_2_, lean_object* v_v_3_, lean_object* v_t_4_, lean_object* v_a_5_, lean_object* v_a_6_, lean_object* v_a_7_, lean_object* v_a_8_){
_start:
{
lean_object* v_a_11_; 
if (lean_obj_tag(v_t_4_) == 0)
{
lean_object* v___x_24_; 
lean_inc(v_a_8_);
lean_inc_ref(v_a_7_);
lean_inc(v_a_6_);
lean_inc_ref(v_a_5_);
lean_inc_ref(v_v_3_);
v___x_24_ = lean_infer_type(v_v_3_, v_a_5_, v_a_6_, v_a_7_, v_a_8_);
if (lean_obj_tag(v___x_24_) == 0)
{
lean_object* v_a_25_; 
v_a_25_ = lean_ctor_get(v___x_24_, 0);
lean_inc(v_a_25_);
lean_dec_ref_known(v___x_24_, 1);
v_a_11_ = v_a_25_;
goto v___jp_10_;
}
else
{
lean_object* v_a_26_; lean_object* v___x_28_; uint8_t v_isShared_29_; uint8_t v_isSharedCheck_33_; 
lean_dec_ref(v_v_3_);
lean_dec(v_h_2_);
lean_dec(v_g_1_);
v_a_26_ = lean_ctor_get(v___x_24_, 0);
v_isSharedCheck_33_ = !lean_is_exclusive(v___x_24_);
if (v_isSharedCheck_33_ == 0)
{
v___x_28_ = v___x_24_;
v_isShared_29_ = v_isSharedCheck_33_;
goto v_resetjp_27_;
}
else
{
lean_inc(v_a_26_);
lean_dec(v___x_24_);
v___x_28_ = lean_box(0);
v_isShared_29_ = v_isSharedCheck_33_;
goto v_resetjp_27_;
}
v_resetjp_27_:
{
lean_object* v___x_31_; 
if (v_isShared_29_ == 0)
{
v___x_31_ = v___x_28_;
goto v_reusejp_30_;
}
else
{
lean_object* v_reuseFailAlloc_32_; 
v_reuseFailAlloc_32_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_32_, 0, v_a_26_);
v___x_31_ = v_reuseFailAlloc_32_;
goto v_reusejp_30_;
}
v_reusejp_30_:
{
return v___x_31_;
}
}
}
}
else
{
lean_object* v_val_34_; 
v_val_34_ = lean_ctor_get(v_t_4_, 0);
lean_inc(v_val_34_);
lean_dec_ref_known(v_t_4_, 1);
v_a_11_ = v_val_34_;
goto v___jp_10_;
}
v___jp_10_:
{
lean_object* v___x_12_; 
v___x_12_ = l_Lean_MVarId_define(v_g_1_, v_h_2_, v_a_11_, v_v_3_, v_a_5_, v_a_6_, v_a_7_, v_a_8_);
if (lean_obj_tag(v___x_12_) == 0)
{
lean_object* v_a_13_; uint8_t v___x_14_; lean_object* v___x_15_; 
v_a_13_ = lean_ctor_get(v___x_12_, 0);
lean_inc(v_a_13_);
lean_dec_ref_known(v___x_12_, 1);
v___x_14_ = 1;
v___x_15_ = l_Lean_Meta_intro1Core(v_a_13_, v___x_14_, v_a_5_, v_a_6_, v_a_7_, v_a_8_);
return v___x_15_;
}
else
{
lean_object* v_a_16_; lean_object* v___x_18_; uint8_t v_isShared_19_; uint8_t v_isSharedCheck_23_; 
v_a_16_ = lean_ctor_get(v___x_12_, 0);
v_isSharedCheck_23_ = !lean_is_exclusive(v___x_12_);
if (v_isSharedCheck_23_ == 0)
{
v___x_18_ = v___x_12_;
v_isShared_19_ = v_isSharedCheck_23_;
goto v_resetjp_17_;
}
else
{
lean_inc(v_a_16_);
lean_dec(v___x_12_);
v___x_18_ = lean_box(0);
v_isShared_19_ = v_isSharedCheck_23_;
goto v_resetjp_17_;
}
v_resetjp_17_:
{
lean_object* v___x_21_; 
if (v_isShared_19_ == 0)
{
v___x_21_ = v___x_18_;
goto v_reusejp_20_;
}
else
{
lean_object* v_reuseFailAlloc_22_; 
v_reuseFailAlloc_22_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_22_, 0, v_a_16_);
v___x_21_ = v_reuseFailAlloc_22_;
goto v_reusejp_20_;
}
v_reusejp_20_:
{
return v___x_21_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_let___boxed(lean_object* v_g_35_, lean_object* v_h_36_, lean_object* v_v_37_, lean_object* v_t_38_, lean_object* v_a_39_, lean_object* v_a_40_, lean_object* v_a_41_, lean_object* v_a_42_, lean_object* v_a_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_Lean_MVarId_let(v_g_35_, v_h_36_, v_v_37_, v_t_38_, v_a_39_, v_a_40_, v_a_41_, v_a_42_);
lean_dec(v_a_42_);
lean_dec_ref(v_a_41_);
lean_dec(v_a_40_);
lean_dec_ref(v_a_39_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1(lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_){
_start:
{
lean_object* v_ref_84_; uint8_t v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; lean_object* v___x_88_; lean_object* v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; lean_object* v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v_ref_84_ = lean_ctor_get(v___y_81_, 5);
v___x_85_ = 0;
v___x_86_ = l_Lean_SourceInfo_fromRef(v_ref_84_, v___x_85_);
v___x_87_ = ((lean_object*)(lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__3));
v___x_88_ = ((lean_object*)(lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__4));
lean_inc_n(v___x_86_, 9);
v___x_89_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_89_, 0, v___x_86_);
lean_ctor_set(v___x_89_, 1, v___x_87_);
v___x_90_ = ((lean_object*)(lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__7));
v___x_91_ = ((lean_object*)(lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__8));
v___x_92_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_92_, 0, v___x_86_);
lean_ctor_set(v___x_92_, 1, v___x_91_);
v___x_93_ = ((lean_object*)(lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__10));
v___x_94_ = ((lean_object*)(lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__12));
v___x_95_ = ((lean_object*)(lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__13));
v___x_96_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_96_, 0, v___x_86_);
lean_ctor_set(v___x_96_, 1, v___x_95_);
v___x_97_ = ((lean_object*)(lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__14));
v___x_98_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_98_, 0, v___x_86_);
lean_ctor_set(v___x_98_, 1, v___x_97_);
v___x_99_ = l_Lean_Syntax_node2(v___x_86_, v___x_94_, v___x_96_, v___x_98_);
v___x_100_ = ((lean_object*)(lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__15));
v___x_101_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_101_, 0, v___x_86_);
lean_ctor_set(v___x_101_, 1, v___x_100_);
lean_inc(v___x_99_);
v___x_102_ = l_Lean_Syntax_node3(v___x_86_, v___x_93_, v___x_99_, v___x_101_, v___x_99_);
v___x_103_ = ((lean_object*)(lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___closed__16));
v___x_104_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_104_, 0, v___x_86_);
lean_ctor_set(v___x_104_, 1, v___x_103_);
v___x_105_ = l_Lean_Syntax_node3(v___x_86_, v___x_90_, v___x_92_, v___x_102_, v___x_104_);
v___x_106_ = l_Lean_Syntax_node2(v___x_86_, v___x_88_, v___x_89_, v___x_105_);
v___x_107_ = l_Lean_Elab_Tactic_evalTactic(v___x_106_, v___y_75_, v___y_76_, v___y_77_, v___y_78_, v___y_79_, v___y_80_, v___y_81_, v___y_82_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1___boxed(lean_object* v___y_108_, lean_object* v___y_109_, lean_object* v___y_110_, lean_object* v___y_111_, lean_object* v___y_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_){
_start:
{
lean_object* v_res_117_; 
v_res_117_ = lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__1(v___y_108_, v___y_109_, v___y_110_, v___y_111_, v___y_112_, v___y_113_, v___y_114_, v___y_115_);
lean_dec(v___y_115_);
lean_dec_ref(v___y_114_);
lean_dec(v___y_113_);
lean_dec_ref(v___y_112_);
lean_dec(v___y_111_);
lean_dec_ref(v___y_110_);
lean_dec(v___y_109_);
lean_dec_ref(v___y_108_);
return v_res_117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_existsi_spec__0_spec__0(lean_object* v_msgData_118_, lean_object* v___y_119_, lean_object* v___y_120_, lean_object* v___y_121_, lean_object* v___y_122_){
_start:
{
lean_object* v___x_124_; lean_object* v_env_125_; lean_object* v___x_126_; lean_object* v_mctx_127_; lean_object* v_lctx_128_; lean_object* v_options_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; 
v___x_124_ = lean_st_ref_get(v___y_122_);
v_env_125_ = lean_ctor_get(v___x_124_, 0);
lean_inc_ref(v_env_125_);
lean_dec(v___x_124_);
v___x_126_ = lean_st_ref_get(v___y_120_);
v_mctx_127_ = lean_ctor_get(v___x_126_, 0);
lean_inc_ref(v_mctx_127_);
lean_dec(v___x_126_);
v_lctx_128_ = lean_ctor_get(v___y_119_, 2);
v_options_129_ = lean_ctor_get(v___y_121_, 2);
lean_inc_ref(v_options_129_);
lean_inc_ref(v_lctx_128_);
v___x_130_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_130_, 0, v_env_125_);
lean_ctor_set(v___x_130_, 1, v_mctx_127_);
lean_ctor_set(v___x_130_, 2, v_lctx_128_);
lean_ctor_set(v___x_130_, 3, v_options_129_);
v___x_131_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_131_, 0, v___x_130_);
lean_ctor_set(v___x_131_, 1, v_msgData_118_);
v___x_132_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_132_, 0, v___x_131_);
return v___x_132_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_existsi_spec__0_spec__0___boxed(lean_object* v_msgData_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_, lean_object* v___y_138_){
_start:
{
lean_object* v_res_139_; 
v_res_139_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_existsi_spec__0_spec__0(v_msgData_133_, v___y_134_, v___y_135_, v___y_136_, v___y_137_);
lean_dec(v___y_137_);
lean_dec_ref(v___y_136_);
lean_dec(v___y_135_);
lean_dec_ref(v___y_134_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_existsi_spec__0___redArg(lean_object* v_msg_140_, lean_object* v___y_141_, lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v___y_144_){
_start:
{
lean_object* v_ref_146_; lean_object* v___x_147_; lean_object* v_a_148_; lean_object* v___x_150_; uint8_t v_isShared_151_; uint8_t v_isSharedCheck_156_; 
v_ref_146_ = lean_ctor_get(v___y_143_, 5);
v___x_147_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Lean_MVarId_existsi_spec__0_spec__0(v_msg_140_, v___y_141_, v___y_142_, v___y_143_, v___y_144_);
v_a_148_ = lean_ctor_get(v___x_147_, 0);
v_isSharedCheck_156_ = !lean_is_exclusive(v___x_147_);
if (v_isSharedCheck_156_ == 0)
{
v___x_150_ = v___x_147_;
v_isShared_151_ = v_isSharedCheck_156_;
goto v_resetjp_149_;
}
else
{
lean_inc(v_a_148_);
lean_dec(v___x_147_);
v___x_150_ = lean_box(0);
v_isShared_151_ = v_isSharedCheck_156_;
goto v_resetjp_149_;
}
v_resetjp_149_:
{
lean_object* v___x_152_; lean_object* v___x_154_; 
lean_inc(v_ref_146_);
v___x_152_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_152_, 0, v_ref_146_);
lean_ctor_set(v___x_152_, 1, v_a_148_);
if (v_isShared_151_ == 0)
{
lean_ctor_set_tag(v___x_150_, 1);
lean_ctor_set(v___x_150_, 0, v___x_152_);
v___x_154_ = v___x_150_;
goto v_reusejp_153_;
}
else
{
lean_object* v_reuseFailAlloc_155_; 
v_reuseFailAlloc_155_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_155_, 0, v___x_152_);
v___x_154_ = v_reuseFailAlloc_155_;
goto v_reusejp_153_;
}
v_reusejp_153_:
{
return v___x_154_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_existsi_spec__0___redArg___boxed(lean_object* v_msg_157_, lean_object* v___y_158_, lean_object* v___y_159_, lean_object* v___y_160_, lean_object* v___y_161_, lean_object* v___y_162_){
_start:
{
lean_object* v_res_163_; 
v_res_163_ = lp_mathlib_Lean_throwError___at___00Lean_MVarId_existsi_spec__0___redArg(v_msg_157_, v___y_158_, v___y_159_, v___y_160_, v___y_161_);
lean_dec(v___y_161_);
lean_dec_ref(v___y_160_);
lean_dec(v___y_159_);
lean_dec_ref(v___y_158_);
return v_res_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__5_spec__6___redArg(lean_object* v_x_164_, lean_object* v_x_165_, lean_object* v_x_166_, lean_object* v_x_167_){
_start:
{
lean_object* v_ks_168_; lean_object* v_vs_169_; lean_object* v___x_171_; uint8_t v_isShared_172_; uint8_t v_isSharedCheck_193_; 
v_ks_168_ = lean_ctor_get(v_x_164_, 0);
v_vs_169_ = lean_ctor_get(v_x_164_, 1);
v_isSharedCheck_193_ = !lean_is_exclusive(v_x_164_);
if (v_isSharedCheck_193_ == 0)
{
v___x_171_ = v_x_164_;
v_isShared_172_ = v_isSharedCheck_193_;
goto v_resetjp_170_;
}
else
{
lean_inc(v_vs_169_);
lean_inc(v_ks_168_);
lean_dec(v_x_164_);
v___x_171_ = lean_box(0);
v_isShared_172_ = v_isSharedCheck_193_;
goto v_resetjp_170_;
}
v_resetjp_170_:
{
lean_object* v___x_173_; uint8_t v___x_174_; 
v___x_173_ = lean_array_get_size(v_ks_168_);
v___x_174_ = lean_nat_dec_lt(v_x_165_, v___x_173_);
if (v___x_174_ == 0)
{
lean_object* v___x_175_; lean_object* v___x_176_; lean_object* v___x_178_; 
lean_dec(v_x_165_);
v___x_175_ = lean_array_push(v_ks_168_, v_x_166_);
v___x_176_ = lean_array_push(v_vs_169_, v_x_167_);
if (v_isShared_172_ == 0)
{
lean_ctor_set(v___x_171_, 1, v___x_176_);
lean_ctor_set(v___x_171_, 0, v___x_175_);
v___x_178_ = v___x_171_;
goto v_reusejp_177_;
}
else
{
lean_object* v_reuseFailAlloc_179_; 
v_reuseFailAlloc_179_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_179_, 0, v___x_175_);
lean_ctor_set(v_reuseFailAlloc_179_, 1, v___x_176_);
v___x_178_ = v_reuseFailAlloc_179_;
goto v_reusejp_177_;
}
v_reusejp_177_:
{
return v___x_178_;
}
}
else
{
lean_object* v_k_x27_180_; uint8_t v___x_181_; 
v_k_x27_180_ = lean_array_fget_borrowed(v_ks_168_, v_x_165_);
v___x_181_ = l_Lean_instBEqMVarId_beq(v_x_166_, v_k_x27_180_);
if (v___x_181_ == 0)
{
lean_object* v___x_183_; 
if (v_isShared_172_ == 0)
{
v___x_183_ = v___x_171_;
goto v_reusejp_182_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v_ks_168_);
lean_ctor_set(v_reuseFailAlloc_187_, 1, v_vs_169_);
v___x_183_ = v_reuseFailAlloc_187_;
goto v_reusejp_182_;
}
v_reusejp_182_:
{
lean_object* v___x_184_; lean_object* v___x_185_; 
v___x_184_ = lean_unsigned_to_nat(1u);
v___x_185_ = lean_nat_add(v_x_165_, v___x_184_);
lean_dec(v_x_165_);
v_x_164_ = v___x_183_;
v_x_165_ = v___x_185_;
goto _start;
}
}
else
{
lean_object* v___x_188_; lean_object* v___x_189_; lean_object* v___x_191_; 
v___x_188_ = lean_array_fset(v_ks_168_, v_x_165_, v_x_166_);
v___x_189_ = lean_array_fset(v_vs_169_, v_x_165_, v_x_167_);
lean_dec(v_x_165_);
if (v_isShared_172_ == 0)
{
lean_ctor_set(v___x_171_, 1, v___x_189_);
lean_ctor_set(v___x_171_, 0, v___x_188_);
v___x_191_ = v___x_171_;
goto v_reusejp_190_;
}
else
{
lean_object* v_reuseFailAlloc_192_; 
v_reuseFailAlloc_192_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_192_, 0, v___x_188_);
lean_ctor_set(v_reuseFailAlloc_192_, 1, v___x_189_);
v___x_191_ = v_reuseFailAlloc_192_;
goto v_reusejp_190_;
}
v_reusejp_190_:
{
return v___x_191_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__5___redArg(lean_object* v_n_194_, lean_object* v_k_195_, lean_object* v_v_196_){
_start:
{
lean_object* v___x_197_; lean_object* v___x_198_; 
v___x_197_ = lean_unsigned_to_nat(0u);
v___x_198_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__5_spec__6___redArg(v_n_194_, v___x_197_, v_k_195_, v_v_196_);
return v___x_198_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_199_; 
v___x_199_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3___redArg(lean_object* v_x_200_, size_t v_x_201_, size_t v_x_202_, lean_object* v_x_203_, lean_object* v_x_204_){
_start:
{
if (lean_obj_tag(v_x_200_) == 0)
{
lean_object* v_es_205_; size_t v___x_206_; size_t v___x_207_; lean_object* v_j_208_; lean_object* v___x_209_; uint8_t v___x_210_; 
v_es_205_ = lean_ctor_get(v_x_200_, 0);
v___x_206_ = ((size_t)31ULL);
v___x_207_ = lean_usize_land(v_x_201_, v___x_206_);
v_j_208_ = lean_usize_to_nat(v___x_207_);
v___x_209_ = lean_array_get_size(v_es_205_);
v___x_210_ = lean_nat_dec_lt(v_j_208_, v___x_209_);
if (v___x_210_ == 0)
{
lean_dec(v_j_208_);
lean_dec(v_x_204_);
lean_dec(v_x_203_);
return v_x_200_;
}
else
{
lean_object* v___x_212_; uint8_t v_isShared_213_; uint8_t v_isSharedCheck_249_; 
lean_inc_ref(v_es_205_);
v_isSharedCheck_249_ = !lean_is_exclusive(v_x_200_);
if (v_isSharedCheck_249_ == 0)
{
lean_object* v_unused_250_; 
v_unused_250_ = lean_ctor_get(v_x_200_, 0);
lean_dec(v_unused_250_);
v___x_212_ = v_x_200_;
v_isShared_213_ = v_isSharedCheck_249_;
goto v_resetjp_211_;
}
else
{
lean_dec(v_x_200_);
v___x_212_ = lean_box(0);
v_isShared_213_ = v_isSharedCheck_249_;
goto v_resetjp_211_;
}
v_resetjp_211_:
{
lean_object* v_v_214_; lean_object* v___x_215_; lean_object* v_xs_x27_216_; lean_object* v___y_218_; 
v_v_214_ = lean_array_fget(v_es_205_, v_j_208_);
v___x_215_ = lean_box(0);
v_xs_x27_216_ = lean_array_fset(v_es_205_, v_j_208_, v___x_215_);
switch(lean_obj_tag(v_v_214_))
{
case 0:
{
lean_object* v_key_223_; lean_object* v_val_224_; lean_object* v___x_226_; uint8_t v_isShared_227_; uint8_t v_isSharedCheck_234_; 
v_key_223_ = lean_ctor_get(v_v_214_, 0);
v_val_224_ = lean_ctor_get(v_v_214_, 1);
v_isSharedCheck_234_ = !lean_is_exclusive(v_v_214_);
if (v_isSharedCheck_234_ == 0)
{
v___x_226_ = v_v_214_;
v_isShared_227_ = v_isSharedCheck_234_;
goto v_resetjp_225_;
}
else
{
lean_inc(v_val_224_);
lean_inc(v_key_223_);
lean_dec(v_v_214_);
v___x_226_ = lean_box(0);
v_isShared_227_ = v_isSharedCheck_234_;
goto v_resetjp_225_;
}
v_resetjp_225_:
{
uint8_t v___x_228_; 
v___x_228_ = l_Lean_instBEqMVarId_beq(v_x_203_, v_key_223_);
if (v___x_228_ == 0)
{
lean_object* v___x_229_; lean_object* v___x_230_; 
lean_del_object(v___x_226_);
v___x_229_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_223_, v_val_224_, v_x_203_, v_x_204_);
v___x_230_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_230_, 0, v___x_229_);
v___y_218_ = v___x_230_;
goto v___jp_217_;
}
else
{
lean_object* v___x_232_; 
lean_dec(v_val_224_);
lean_dec(v_key_223_);
if (v_isShared_227_ == 0)
{
lean_ctor_set(v___x_226_, 1, v_x_204_);
lean_ctor_set(v___x_226_, 0, v_x_203_);
v___x_232_ = v___x_226_;
goto v_reusejp_231_;
}
else
{
lean_object* v_reuseFailAlloc_233_; 
v_reuseFailAlloc_233_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_233_, 0, v_x_203_);
lean_ctor_set(v_reuseFailAlloc_233_, 1, v_x_204_);
v___x_232_ = v_reuseFailAlloc_233_;
goto v_reusejp_231_;
}
v_reusejp_231_:
{
v___y_218_ = v___x_232_;
goto v___jp_217_;
}
}
}
}
case 1:
{
lean_object* v_node_235_; lean_object* v___x_237_; uint8_t v_isShared_238_; uint8_t v_isSharedCheck_247_; 
v_node_235_ = lean_ctor_get(v_v_214_, 0);
v_isSharedCheck_247_ = !lean_is_exclusive(v_v_214_);
if (v_isSharedCheck_247_ == 0)
{
v___x_237_ = v_v_214_;
v_isShared_238_ = v_isSharedCheck_247_;
goto v_resetjp_236_;
}
else
{
lean_inc(v_node_235_);
lean_dec(v_v_214_);
v___x_237_ = lean_box(0);
v_isShared_238_ = v_isSharedCheck_247_;
goto v_resetjp_236_;
}
v_resetjp_236_:
{
size_t v___x_239_; size_t v___x_240_; size_t v___x_241_; size_t v___x_242_; lean_object* v___x_243_; lean_object* v___x_245_; 
v___x_239_ = ((size_t)5ULL);
v___x_240_ = lean_usize_shift_right(v_x_201_, v___x_239_);
v___x_241_ = ((size_t)1ULL);
v___x_242_ = lean_usize_add(v_x_202_, v___x_241_);
v___x_243_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3___redArg(v_node_235_, v___x_240_, v___x_242_, v_x_203_, v_x_204_);
if (v_isShared_238_ == 0)
{
lean_ctor_set(v___x_237_, 0, v___x_243_);
v___x_245_ = v___x_237_;
goto v_reusejp_244_;
}
else
{
lean_object* v_reuseFailAlloc_246_; 
v_reuseFailAlloc_246_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_246_, 0, v___x_243_);
v___x_245_ = v_reuseFailAlloc_246_;
goto v_reusejp_244_;
}
v_reusejp_244_:
{
v___y_218_ = v___x_245_;
goto v___jp_217_;
}
}
}
default: 
{
lean_object* v___x_248_; 
v___x_248_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_248_, 0, v_x_203_);
lean_ctor_set(v___x_248_, 1, v_x_204_);
v___y_218_ = v___x_248_;
goto v___jp_217_;
}
}
v___jp_217_:
{
lean_object* v___x_219_; lean_object* v___x_221_; 
v___x_219_ = lean_array_fset(v_xs_x27_216_, v_j_208_, v___y_218_);
lean_dec(v_j_208_);
if (v_isShared_213_ == 0)
{
lean_ctor_set(v___x_212_, 0, v___x_219_);
v___x_221_ = v___x_212_;
goto v_reusejp_220_;
}
else
{
lean_object* v_reuseFailAlloc_222_; 
v_reuseFailAlloc_222_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_222_, 0, v___x_219_);
v___x_221_ = v_reuseFailAlloc_222_;
goto v_reusejp_220_;
}
v_reusejp_220_:
{
return v___x_221_;
}
}
}
}
}
else
{
lean_object* v_ks_251_; lean_object* v_vs_252_; lean_object* v___x_254_; uint8_t v_isShared_255_; uint8_t v_isSharedCheck_272_; 
v_ks_251_ = lean_ctor_get(v_x_200_, 0);
v_vs_252_ = lean_ctor_get(v_x_200_, 1);
v_isSharedCheck_272_ = !lean_is_exclusive(v_x_200_);
if (v_isSharedCheck_272_ == 0)
{
v___x_254_ = v_x_200_;
v_isShared_255_ = v_isSharedCheck_272_;
goto v_resetjp_253_;
}
else
{
lean_inc(v_vs_252_);
lean_inc(v_ks_251_);
lean_dec(v_x_200_);
v___x_254_ = lean_box(0);
v_isShared_255_ = v_isSharedCheck_272_;
goto v_resetjp_253_;
}
v_resetjp_253_:
{
lean_object* v___x_257_; 
if (v_isShared_255_ == 0)
{
v___x_257_ = v___x_254_;
goto v_reusejp_256_;
}
else
{
lean_object* v_reuseFailAlloc_271_; 
v_reuseFailAlloc_271_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_271_, 0, v_ks_251_);
lean_ctor_set(v_reuseFailAlloc_271_, 1, v_vs_252_);
v___x_257_ = v_reuseFailAlloc_271_;
goto v_reusejp_256_;
}
v_reusejp_256_:
{
lean_object* v_newNode_258_; uint8_t v___y_260_; size_t v___x_266_; uint8_t v___x_267_; 
v_newNode_258_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__5___redArg(v___x_257_, v_x_203_, v_x_204_);
v___x_266_ = ((size_t)7ULL);
v___x_267_ = lean_usize_dec_le(v___x_266_, v_x_202_);
if (v___x_267_ == 0)
{
lean_object* v___x_268_; lean_object* v___x_269_; uint8_t v___x_270_; 
v___x_268_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_258_);
v___x_269_ = lean_unsigned_to_nat(4u);
v___x_270_ = lean_nat_dec_lt(v___x_268_, v___x_269_);
lean_dec(v___x_268_);
v___y_260_ = v___x_270_;
goto v___jp_259_;
}
else
{
v___y_260_ = v___x_267_;
goto v___jp_259_;
}
v___jp_259_:
{
if (v___y_260_ == 0)
{
lean_object* v_ks_261_; lean_object* v_vs_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; 
v_ks_261_ = lean_ctor_get(v_newNode_258_, 0);
lean_inc_ref(v_ks_261_);
v_vs_262_ = lean_ctor_get(v_newNode_258_, 1);
lean_inc_ref(v_vs_262_);
lean_dec_ref(v_newNode_258_);
v___x_263_ = lean_unsigned_to_nat(0u);
v___x_264_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3___redArg___closed__0);
v___x_265_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__6___redArg(v_x_202_, v_ks_261_, v_vs_262_, v___x_263_, v___x_264_);
lean_dec_ref(v_vs_262_);
lean_dec_ref(v_ks_261_);
return v___x_265_;
}
else
{
return v_newNode_258_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__6___redArg(size_t v_depth_273_, lean_object* v_keys_274_, lean_object* v_vals_275_, lean_object* v_i_276_, lean_object* v_entries_277_){
_start:
{
lean_object* v___x_278_; uint8_t v___x_279_; 
v___x_278_ = lean_array_get_size(v_keys_274_);
v___x_279_ = lean_nat_dec_lt(v_i_276_, v___x_278_);
if (v___x_279_ == 0)
{
lean_dec(v_i_276_);
return v_entries_277_;
}
else
{
lean_object* v_k_280_; lean_object* v_v_281_; uint64_t v___x_282_; size_t v_h_283_; size_t v___x_284_; lean_object* v___x_285_; size_t v___x_286_; size_t v___x_287_; size_t v___x_288_; size_t v_h_289_; lean_object* v___x_290_; lean_object* v___x_291_; 
v_k_280_ = lean_array_fget_borrowed(v_keys_274_, v_i_276_);
v_v_281_ = lean_array_fget_borrowed(v_vals_275_, v_i_276_);
v___x_282_ = l_Lean_instHashableMVarId_hash(v_k_280_);
v_h_283_ = lean_uint64_to_usize(v___x_282_);
v___x_284_ = ((size_t)5ULL);
v___x_285_ = lean_unsigned_to_nat(1u);
v___x_286_ = ((size_t)1ULL);
v___x_287_ = lean_usize_sub(v_depth_273_, v___x_286_);
v___x_288_ = lean_usize_mul(v___x_284_, v___x_287_);
v_h_289_ = lean_usize_shift_right(v_h_283_, v___x_288_);
v___x_290_ = lean_nat_add(v_i_276_, v___x_285_);
lean_dec(v_i_276_);
lean_inc(v_v_281_);
lean_inc(v_k_280_);
v___x_291_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3___redArg(v_entries_277_, v_h_289_, v_depth_273_, v_k_280_, v_v_281_);
v_i_276_ = v___x_290_;
v_entries_277_ = v___x_291_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__6___redArg___boxed(lean_object* v_depth_293_, lean_object* v_keys_294_, lean_object* v_vals_295_, lean_object* v_i_296_, lean_object* v_entries_297_){
_start:
{
size_t v_depth_boxed_298_; lean_object* v_res_299_; 
v_depth_boxed_298_ = lean_unbox_usize(v_depth_293_);
lean_dec(v_depth_293_);
v_res_299_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__6___redArg(v_depth_boxed_298_, v_keys_294_, v_vals_295_, v_i_296_, v_entries_297_);
lean_dec_ref(v_vals_295_);
lean_dec_ref(v_keys_294_);
return v_res_299_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3___redArg___boxed(lean_object* v_x_300_, lean_object* v_x_301_, lean_object* v_x_302_, lean_object* v_x_303_, lean_object* v_x_304_){
_start:
{
size_t v_x_4409__boxed_305_; size_t v_x_4410__boxed_306_; lean_object* v_res_307_; 
v_x_4409__boxed_305_ = lean_unbox_usize(v_x_301_);
lean_dec(v_x_301_);
v_x_4410__boxed_306_ = lean_unbox_usize(v_x_302_);
lean_dec(v_x_302_);
v_res_307_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3___redArg(v_x_300_, v_x_4409__boxed_305_, v_x_4410__boxed_306_, v_x_303_, v_x_304_);
return v_res_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2___redArg(lean_object* v_x_308_, lean_object* v_x_309_, lean_object* v_x_310_){
_start:
{
uint64_t v___x_311_; size_t v___x_312_; size_t v___x_313_; lean_object* v___x_314_; 
v___x_311_ = l_Lean_instHashableMVarId_hash(v_x_309_);
v___x_312_ = lean_uint64_to_usize(v___x_311_);
v___x_313_ = ((size_t)1ULL);
v___x_314_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3___redArg(v_x_308_, v___x_312_, v___x_313_, v_x_309_, v_x_310_);
return v___x_314_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1___redArg(lean_object* v_mvarId_315_, lean_object* v_val_316_, lean_object* v___y_317_){
_start:
{
lean_object* v___x_319_; lean_object* v_mctx_320_; lean_object* v_cache_321_; lean_object* v_zetaDeltaFVarIds_322_; lean_object* v_postponed_323_; lean_object* v_diag_324_; lean_object* v___x_326_; uint8_t v_isShared_327_; uint8_t v_isSharedCheck_352_; 
v___x_319_ = lean_st_ref_take(v___y_317_);
v_mctx_320_ = lean_ctor_get(v___x_319_, 0);
v_cache_321_ = lean_ctor_get(v___x_319_, 1);
v_zetaDeltaFVarIds_322_ = lean_ctor_get(v___x_319_, 2);
v_postponed_323_ = lean_ctor_get(v___x_319_, 3);
v_diag_324_ = lean_ctor_get(v___x_319_, 4);
v_isSharedCheck_352_ = !lean_is_exclusive(v___x_319_);
if (v_isSharedCheck_352_ == 0)
{
v___x_326_ = v___x_319_;
v_isShared_327_ = v_isSharedCheck_352_;
goto v_resetjp_325_;
}
else
{
lean_inc(v_diag_324_);
lean_inc(v_postponed_323_);
lean_inc(v_zetaDeltaFVarIds_322_);
lean_inc(v_cache_321_);
lean_inc(v_mctx_320_);
lean_dec(v___x_319_);
v___x_326_ = lean_box(0);
v_isShared_327_ = v_isSharedCheck_352_;
goto v_resetjp_325_;
}
v_resetjp_325_:
{
lean_object* v_depth_328_; lean_object* v_levelAssignDepth_329_; lean_object* v_lmvarCounter_330_; lean_object* v_mvarCounter_331_; lean_object* v_lDecls_332_; lean_object* v_decls_333_; lean_object* v_userNames_334_; lean_object* v_lAssignment_335_; lean_object* v_eAssignment_336_; lean_object* v_dAssignment_337_; lean_object* v___x_339_; uint8_t v_isShared_340_; uint8_t v_isSharedCheck_351_; 
v_depth_328_ = lean_ctor_get(v_mctx_320_, 0);
v_levelAssignDepth_329_ = lean_ctor_get(v_mctx_320_, 1);
v_lmvarCounter_330_ = lean_ctor_get(v_mctx_320_, 2);
v_mvarCounter_331_ = lean_ctor_get(v_mctx_320_, 3);
v_lDecls_332_ = lean_ctor_get(v_mctx_320_, 4);
v_decls_333_ = lean_ctor_get(v_mctx_320_, 5);
v_userNames_334_ = lean_ctor_get(v_mctx_320_, 6);
v_lAssignment_335_ = lean_ctor_get(v_mctx_320_, 7);
v_eAssignment_336_ = lean_ctor_get(v_mctx_320_, 8);
v_dAssignment_337_ = lean_ctor_get(v_mctx_320_, 9);
v_isSharedCheck_351_ = !lean_is_exclusive(v_mctx_320_);
if (v_isSharedCheck_351_ == 0)
{
v___x_339_ = v_mctx_320_;
v_isShared_340_ = v_isSharedCheck_351_;
goto v_resetjp_338_;
}
else
{
lean_inc(v_dAssignment_337_);
lean_inc(v_eAssignment_336_);
lean_inc(v_lAssignment_335_);
lean_inc(v_userNames_334_);
lean_inc(v_decls_333_);
lean_inc(v_lDecls_332_);
lean_inc(v_mvarCounter_331_);
lean_inc(v_lmvarCounter_330_);
lean_inc(v_levelAssignDepth_329_);
lean_inc(v_depth_328_);
lean_dec(v_mctx_320_);
v___x_339_ = lean_box(0);
v_isShared_340_ = v_isSharedCheck_351_;
goto v_resetjp_338_;
}
v_resetjp_338_:
{
lean_object* v___x_341_; lean_object* v___x_343_; 
v___x_341_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2___redArg(v_eAssignment_336_, v_mvarId_315_, v_val_316_);
if (v_isShared_340_ == 0)
{
lean_ctor_set(v___x_339_, 8, v___x_341_);
v___x_343_ = v___x_339_;
goto v_reusejp_342_;
}
else
{
lean_object* v_reuseFailAlloc_350_; 
v_reuseFailAlloc_350_ = lean_alloc_ctor(0, 10, 0);
lean_ctor_set(v_reuseFailAlloc_350_, 0, v_depth_328_);
lean_ctor_set(v_reuseFailAlloc_350_, 1, v_levelAssignDepth_329_);
lean_ctor_set(v_reuseFailAlloc_350_, 2, v_lmvarCounter_330_);
lean_ctor_set(v_reuseFailAlloc_350_, 3, v_mvarCounter_331_);
lean_ctor_set(v_reuseFailAlloc_350_, 4, v_lDecls_332_);
lean_ctor_set(v_reuseFailAlloc_350_, 5, v_decls_333_);
lean_ctor_set(v_reuseFailAlloc_350_, 6, v_userNames_334_);
lean_ctor_set(v_reuseFailAlloc_350_, 7, v_lAssignment_335_);
lean_ctor_set(v_reuseFailAlloc_350_, 8, v___x_341_);
lean_ctor_set(v_reuseFailAlloc_350_, 9, v_dAssignment_337_);
v___x_343_ = v_reuseFailAlloc_350_;
goto v_reusejp_342_;
}
v_reusejp_342_:
{
lean_object* v___x_345_; 
if (v_isShared_327_ == 0)
{
lean_ctor_set(v___x_326_, 0, v___x_343_);
v___x_345_ = v___x_326_;
goto v_reusejp_344_;
}
else
{
lean_object* v_reuseFailAlloc_349_; 
v_reuseFailAlloc_349_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_349_, 0, v___x_343_);
lean_ctor_set(v_reuseFailAlloc_349_, 1, v_cache_321_);
lean_ctor_set(v_reuseFailAlloc_349_, 2, v_zetaDeltaFVarIds_322_);
lean_ctor_set(v_reuseFailAlloc_349_, 3, v_postponed_323_);
lean_ctor_set(v_reuseFailAlloc_349_, 4, v_diag_324_);
v___x_345_ = v_reuseFailAlloc_349_;
goto v_reusejp_344_;
}
v_reusejp_344_:
{
lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; 
v___x_346_ = lean_st_ref_set(v___y_317_, v___x_345_);
v___x_347_ = lean_box(0);
v___x_348_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_348_, 0, v___x_347_);
return v___x_348_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1___redArg___boxed(lean_object* v_mvarId_353_, lean_object* v_val_354_, lean_object* v___y_355_, lean_object* v___y_356_){
_start:
{
lean_object* v_res_357_; 
v_res_357_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1___redArg(v_mvarId_353_, v_val_354_, v___y_355_);
lean_dec(v___y_355_);
return v_res_357_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__0(lean_object* v_x_358_){
_start:
{
uint8_t v___x_359_; 
v___x_359_ = 0;
return v___x_359_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__0___boxed(lean_object* v_x_360_){
_start:
{
uint8_t v_res_361_; lean_object* v_r_362_; 
v_res_361_ = lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___lam__0(v_x_360_);
lean_dec(v_x_360_);
v_r_362_ = lean_box(v_res_361_);
return v_r_362_;
}
}
static lean_object* _init_lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__1(void){
_start:
{
lean_object* v___x_364_; lean_object* v___x_365_; 
v___x_364_ = ((lean_object*)(lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__0));
v___x_365_ = l_Lean_stringToMessageData(v___x_364_);
return v___x_365_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2(lean_object* v_x_381_, lean_object* v_x_382_, lean_object* v___y_383_, lean_object* v___y_384_, lean_object* v___y_385_, lean_object* v___y_386_){
_start:
{
if (lean_obj_tag(v_x_382_) == 0)
{
lean_object* v___x_388_; 
v___x_388_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_388_, 0, v_x_381_);
return v___x_388_;
}
else
{
lean_object* v_head_389_; lean_object* v_tail_390_; lean_object* v___y_392_; lean_object* v___y_393_; lean_object* v___y_394_; lean_object* v___y_395_; lean_object* v___f_398_; lean_object* v___x_399_; lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; 
v_head_389_ = lean_ctor_get(v_x_382_, 0);
lean_inc(v_head_389_);
v_tail_390_ = lean_ctor_get(v_x_382_, 1);
lean_inc(v_tail_390_);
lean_dec_ref_known(v_x_382_, 2);
v___f_398_ = ((lean_object*)(lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__3));
v___x_399_ = lean_alloc_closure((void*)(l_Lean_Elab_Tactic_run___boxed), 9, 2);
lean_closure_set(v___x_399_, 0, v_x_381_);
lean_closure_set(v___x_399_, 1, v___f_398_);
v___x_400_ = ((lean_object*)(lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__5));
v___x_401_ = ((lean_object*)(lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__6));
v___x_402_ = l_Lean_Elab_Term_TermElabM_run___redArg(v___x_399_, v___x_400_, v___x_401_, v___y_383_, v___y_384_, v___y_385_, v___y_386_);
if (lean_obj_tag(v___x_402_) == 0)
{
lean_object* v_a_403_; lean_object* v_fst_404_; 
v_a_403_ = lean_ctor_get(v___x_402_, 0);
lean_inc(v_a_403_);
lean_dec_ref_known(v___x_402_, 1);
v_fst_404_ = lean_ctor_get(v_a_403_, 0);
lean_inc(v_fst_404_);
lean_dec(v_a_403_);
if (lean_obj_tag(v_fst_404_) == 1)
{
lean_object* v_tail_405_; 
v_tail_405_ = lean_ctor_get(v_fst_404_, 1);
lean_inc(v_tail_405_);
if (lean_obj_tag(v_tail_405_) == 1)
{
lean_object* v_tail_406_; 
v_tail_406_ = lean_ctor_get(v_tail_405_, 1);
if (lean_obj_tag(v_tail_406_) == 0)
{
lean_object* v_head_407_; lean_object* v_head_408_; lean_object* v___x_409_; 
v_head_407_ = lean_ctor_get(v_fst_404_, 0);
lean_inc(v_head_407_);
lean_dec_ref_known(v_fst_404_, 2);
v_head_408_ = lean_ctor_get(v_tail_405_, 0);
lean_inc(v_head_408_);
lean_dec_ref_known(v_tail_405_, 2);
v___x_409_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1___redArg(v_head_407_, v_head_389_, v___y_384_);
lean_dec_ref(v___x_409_);
v_x_381_ = v_head_408_;
v_x_382_ = v_tail_390_;
goto _start;
}
else
{
lean_dec_ref_known(v_tail_405_, 2);
lean_dec_ref_known(v_fst_404_, 2);
lean_dec(v_tail_390_);
lean_dec(v_head_389_);
v___y_392_ = v___y_383_;
v___y_393_ = v___y_384_;
v___y_394_ = v___y_385_;
v___y_395_ = v___y_386_;
goto v___jp_391_;
}
}
else
{
lean_dec_ref_known(v_fst_404_, 2);
lean_dec(v_tail_405_);
lean_dec(v_tail_390_);
lean_dec(v_head_389_);
v___y_392_ = v___y_383_;
v___y_393_ = v___y_384_;
v___y_394_ = v___y_385_;
v___y_395_ = v___y_386_;
goto v___jp_391_;
}
}
else
{
lean_dec(v_fst_404_);
lean_dec(v_tail_390_);
lean_dec(v_head_389_);
v___y_392_ = v___y_383_;
v___y_393_ = v___y_384_;
v___y_394_ = v___y_385_;
v___y_395_ = v___y_386_;
goto v___jp_391_;
}
}
else
{
lean_object* v_a_411_; lean_object* v___x_413_; uint8_t v_isShared_414_; uint8_t v_isSharedCheck_418_; 
lean_dec(v_tail_390_);
lean_dec(v_head_389_);
v_a_411_ = lean_ctor_get(v___x_402_, 0);
v_isSharedCheck_418_ = !lean_is_exclusive(v___x_402_);
if (v_isSharedCheck_418_ == 0)
{
v___x_413_ = v___x_402_;
v_isShared_414_ = v_isSharedCheck_418_;
goto v_resetjp_412_;
}
else
{
lean_inc(v_a_411_);
lean_dec(v___x_402_);
v___x_413_ = lean_box(0);
v_isShared_414_ = v_isSharedCheck_418_;
goto v_resetjp_412_;
}
v_resetjp_412_:
{
lean_object* v___x_416_; 
if (v_isShared_414_ == 0)
{
v___x_416_ = v___x_413_;
goto v_reusejp_415_;
}
else
{
lean_object* v_reuseFailAlloc_417_; 
v_reuseFailAlloc_417_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_417_, 0, v_a_411_);
v___x_416_ = v_reuseFailAlloc_417_;
goto v_reusejp_415_;
}
v_reusejp_415_:
{
return v___x_416_;
}
}
}
v___jp_391_:
{
lean_object* v___x_396_; lean_object* v___x_397_; 
v___x_396_ = lean_obj_once(&lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__1, &lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__1_once, _init_lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___closed__1);
v___x_397_ = lp_mathlib_Lean_throwError___at___00Lean_MVarId_existsi_spec__0___redArg(v___x_396_, v___y_392_, v___y_393_, v___y_394_, v___y_395_);
return v___x_397_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2___boxed(lean_object* v_x_419_, lean_object* v_x_420_, lean_object* v___y_421_, lean_object* v___y_422_, lean_object* v___y_423_, lean_object* v___y_424_, lean_object* v___y_425_){
_start:
{
lean_object* v_res_426_; 
v_res_426_ = lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2(v_x_419_, v_x_420_, v___y_421_, v___y_422_, v___y_423_, v___y_424_);
lean_dec(v___y_424_);
lean_dec_ref(v___y_423_);
lean_dec(v___y_422_);
lean_dec_ref(v___y_421_);
return v_res_426_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_existsi(lean_object* v_mvar_427_, lean_object* v_es_428_, lean_object* v_a_429_, lean_object* v_a_430_, lean_object* v_a_431_, lean_object* v_a_432_){
_start:
{
lean_object* v___x_434_; 
v___x_434_ = lp_mathlib_List_foldlM___at___00Lean_MVarId_existsi_spec__2(v_mvar_427_, v_es_428_, v_a_429_, v_a_430_, v_a_431_, v_a_432_);
return v___x_434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_existsi___boxed(lean_object* v_mvar_435_, lean_object* v_es_436_, lean_object* v_a_437_, lean_object* v_a_438_, lean_object* v_a_439_, lean_object* v_a_440_, lean_object* v_a_441_){
_start:
{
lean_object* v_res_442_; 
v_res_442_ = lp_mathlib_Lean_MVarId_existsi(v_mvar_435_, v_es_436_, v_a_437_, v_a_438_, v_a_439_, v_a_440_);
lean_dec(v_a_440_);
lean_dec_ref(v_a_439_);
lean_dec(v_a_438_);
lean_dec_ref(v_a_437_);
return v_res_442_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_existsi_spec__0(lean_object* v_00_u03b1_443_, lean_object* v_msg_444_, lean_object* v___y_445_, lean_object* v___y_446_, lean_object* v___y_447_, lean_object* v___y_448_){
_start:
{
lean_object* v___x_450_; 
v___x_450_ = lp_mathlib_Lean_throwError___at___00Lean_MVarId_existsi_spec__0___redArg(v_msg_444_, v___y_445_, v___y_446_, v___y_447_, v___y_448_);
return v___x_450_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Lean_MVarId_existsi_spec__0___boxed(lean_object* v_00_u03b1_451_, lean_object* v_msg_452_, lean_object* v___y_453_, lean_object* v___y_454_, lean_object* v___y_455_, lean_object* v___y_456_, lean_object* v___y_457_){
_start:
{
lean_object* v_res_458_; 
v_res_458_ = lp_mathlib_Lean_throwError___at___00Lean_MVarId_existsi_spec__0(v_00_u03b1_451_, v_msg_452_, v___y_453_, v___y_454_, v___y_455_, v___y_456_);
lean_dec(v___y_456_);
lean_dec_ref(v___y_455_);
lean_dec(v___y_454_);
lean_dec_ref(v___y_453_);
return v_res_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1(lean_object* v_mvarId_459_, lean_object* v_val_460_, lean_object* v___y_461_, lean_object* v___y_462_, lean_object* v___y_463_, lean_object* v___y_464_){
_start:
{
lean_object* v___x_466_; 
v___x_466_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1___redArg(v_mvarId_459_, v_val_460_, v___y_462_);
return v___x_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1___boxed(lean_object* v_mvarId_467_, lean_object* v_val_468_, lean_object* v___y_469_, lean_object* v___y_470_, lean_object* v___y_471_, lean_object* v___y_472_, lean_object* v___y_473_){
_start:
{
lean_object* v_res_474_; 
v_res_474_ = lp_mathlib_Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1(v_mvarId_467_, v_val_468_, v___y_469_, v___y_470_, v___y_471_, v___y_472_);
lean_dec(v___y_472_);
lean_dec_ref(v___y_471_);
lean_dec(v___y_470_);
lean_dec_ref(v___y_469_);
return v_res_474_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2(lean_object* v_00_u03b2_475_, lean_object* v_x_476_, lean_object* v_x_477_, lean_object* v_x_478_){
_start:
{
lean_object* v___x_479_; 
v___x_479_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2___redArg(v_x_476_, v_x_477_, v_x_478_);
return v___x_479_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3(lean_object* v_00_u03b2_480_, lean_object* v_x_481_, size_t v_x_482_, size_t v_x_483_, lean_object* v_x_484_, lean_object* v_x_485_){
_start:
{
lean_object* v___x_486_; 
v___x_486_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3___redArg(v_x_481_, v_x_482_, v_x_483_, v_x_484_, v_x_485_);
return v___x_486_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3___boxed(lean_object* v_00_u03b2_487_, lean_object* v_x_488_, lean_object* v_x_489_, lean_object* v_x_490_, lean_object* v_x_491_, lean_object* v_x_492_){
_start:
{
size_t v_x_4850__boxed_493_; size_t v_x_4851__boxed_494_; lean_object* v_res_495_; 
v_x_4850__boxed_493_ = lean_unbox_usize(v_x_489_);
lean_dec(v_x_489_);
v_x_4851__boxed_494_ = lean_unbox_usize(v_x_490_);
lean_dec(v_x_490_);
v_res_495_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3(v_00_u03b2_487_, v_x_488_, v_x_4850__boxed_493_, v_x_4851__boxed_494_, v_x_491_, v_x_492_);
return v_res_495_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__5(lean_object* v_00_u03b2_496_, lean_object* v_n_497_, lean_object* v_k_498_, lean_object* v_v_499_){
_start:
{
lean_object* v___x_500_; 
v___x_500_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__5___redArg(v_n_497_, v_k_498_, v_v_499_);
return v___x_500_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__6(lean_object* v_00_u03b2_501_, size_t v_depth_502_, lean_object* v_keys_503_, lean_object* v_vals_504_, lean_object* v_heq_505_, lean_object* v_i_506_, lean_object* v_entries_507_){
_start:
{
lean_object* v___x_508_; 
v___x_508_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__6___redArg(v_depth_502_, v_keys_503_, v_vals_504_, v_i_506_, v_entries_507_);
return v___x_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__6___boxed(lean_object* v_00_u03b2_509_, lean_object* v_depth_510_, lean_object* v_keys_511_, lean_object* v_vals_512_, lean_object* v_heq_513_, lean_object* v_i_514_, lean_object* v_entries_515_){
_start:
{
size_t v_depth_boxed_516_; lean_object* v_res_517_; 
v_depth_boxed_516_ = lean_unbox_usize(v_depth_510_);
lean_dec(v_depth_510_);
v_res_517_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__6(v_00_u03b2_509_, v_depth_boxed_516_, v_keys_511_, v_vals_512_, v_heq_513_, v_i_514_, v_entries_515_);
lean_dec_ref(v_vals_512_);
lean_dec_ref(v_keys_511_);
return v_res_517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__5_spec__6(lean_object* v_00_u03b2_518_, lean_object* v_x_519_, lean_object* v_x_520_, lean_object* v_x_521_, lean_object* v_x_522_){
_start:
{
lean_object* v___x_523_; 
v___x_523_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00Lean_MVarId_assign___at___00Lean_MVarId_existsi_spec__1_spec__2_spec__3_spec__5_spec__6___redArg(v_x_519_, v_x_520_, v_x_521_, v_x_522_);
return v___x_523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_MVarId_intros_x21_run(lean_object* v_mvarId_524_, lean_object* v_acc_525_, lean_object* v_g_526_, lean_object* v_a_527_, lean_object* v_a_528_, lean_object* v_a_529_, lean_object* v_a_530_){
_start:
{
lean_object* v___y_533_; uint8_t v___y_534_; lean_object* v___y_538_; lean_object* v_a_539_; uint8_t v___x_542_; lean_object* v___x_543_; 
v___x_542_ = 0;
lean_inc(v_mvarId_524_);
v___x_543_ = l_Lean_Meta_intro1Core(v_mvarId_524_, v___x_542_, v_a_527_, v_a_528_, v_a_529_, v_a_530_);
if (lean_obj_tag(v___x_543_) == 0)
{
lean_object* v_a_544_; lean_object* v_fst_545_; lean_object* v_snd_546_; lean_object* v___x_547_; lean_object* v___x_548_; 
v_a_544_ = lean_ctor_get(v___x_543_, 0);
lean_inc(v_a_544_);
lean_dec_ref_known(v___x_543_, 1);
v_fst_545_ = lean_ctor_get(v_a_544_, 0);
lean_inc(v_fst_545_);
v_snd_546_ = lean_ctor_get(v_a_544_, 1);
lean_inc(v_snd_546_);
lean_dec(v_a_544_);
lean_inc_ref(v_acc_525_);
v___x_547_ = lean_array_push(v_acc_525_, v_fst_545_);
v___x_548_ = lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_MVarId_intros_x21_run(v_mvarId_524_, v___x_547_, v_snd_546_, v_a_527_, v_a_528_, v_a_529_, v_a_530_);
if (lean_obj_tag(v___x_548_) == 0)
{
lean_dec(v_g_526_);
lean_dec_ref(v_acc_525_);
return v___x_548_;
}
else
{
lean_object* v_a_549_; 
v_a_549_ = lean_ctor_get(v___x_548_, 0);
lean_inc(v_a_549_);
v___y_538_ = v___x_548_;
v_a_539_ = v_a_549_;
goto v___jp_537_;
}
}
else
{
lean_object* v_a_550_; lean_object* v___x_552_; uint8_t v_isShared_553_; uint8_t v_isSharedCheck_557_; 
lean_dec(v_mvarId_524_);
v_a_550_ = lean_ctor_get(v___x_543_, 0);
v_isSharedCheck_557_ = !lean_is_exclusive(v___x_543_);
if (v_isSharedCheck_557_ == 0)
{
v___x_552_ = v___x_543_;
v_isShared_553_ = v_isSharedCheck_557_;
goto v_resetjp_551_;
}
else
{
lean_inc(v_a_550_);
lean_dec(v___x_543_);
v___x_552_ = lean_box(0);
v_isShared_553_ = v_isSharedCheck_557_;
goto v_resetjp_551_;
}
v_resetjp_551_:
{
lean_object* v___x_555_; 
lean_inc(v_a_550_);
if (v_isShared_553_ == 0)
{
v___x_555_ = v___x_552_;
goto v_reusejp_554_;
}
else
{
lean_object* v_reuseFailAlloc_556_; 
v_reuseFailAlloc_556_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_556_, 0, v_a_550_);
v___x_555_ = v_reuseFailAlloc_556_;
goto v_reusejp_554_;
}
v_reusejp_554_:
{
v___y_538_ = v___x_555_;
v_a_539_ = v_a_550_;
goto v___jp_537_;
}
}
}
v___jp_532_:
{
if (v___y_534_ == 0)
{
lean_object* v___x_535_; lean_object* v___x_536_; 
lean_dec_ref(v___y_533_);
v___x_535_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_535_, 0, v_acc_525_);
lean_ctor_set(v___x_535_, 1, v_g_526_);
v___x_536_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_536_, 0, v___x_535_);
return v___x_536_;
}
else
{
lean_dec(v_g_526_);
lean_dec_ref(v_acc_525_);
return v___y_533_;
}
}
v___jp_537_:
{
uint8_t v___x_540_; 
v___x_540_ = l_Lean_Exception_isInterrupt(v_a_539_);
if (v___x_540_ == 0)
{
uint8_t v___x_541_; 
v___x_541_ = l_Lean_Exception_isRuntime(v_a_539_);
v___y_533_ = v___y_538_;
v___y_534_ = v___x_541_;
goto v___jp_532_;
}
else
{
lean_dec_ref(v_a_539_);
v___y_533_ = v___y_538_;
v___y_534_ = v___x_540_;
goto v___jp_532_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_MVarId_intros_x21_run___boxed(lean_object* v_mvarId_558_, lean_object* v_acc_559_, lean_object* v_g_560_, lean_object* v_a_561_, lean_object* v_a_562_, lean_object* v_a_563_, lean_object* v_a_564_, lean_object* v_a_565_){
_start:
{
lean_object* v_res_566_; 
v_res_566_ = lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_MVarId_intros_x21_run(v_mvarId_558_, v_acc_559_, v_g_560_, v_a_561_, v_a_562_, v_a_563_, v_a_564_);
lean_dec(v_a_564_);
lean_dec_ref(v_a_563_);
lean_dec(v_a_562_);
lean_dec_ref(v_a_561_);
return v_res_566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_intros_x21(lean_object* v_mvarId_569_, lean_object* v_a_570_, lean_object* v_a_571_, lean_object* v_a_572_, lean_object* v_a_573_){
_start:
{
lean_object* v___x_575_; lean_object* v___x_576_; 
v___x_575_ = ((lean_object*)(lp_mathlib_Lean_MVarId_intros_x21___closed__0));
lean_inc(v_mvarId_569_);
v___x_576_ = lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_MVarId_intros_x21_run(v_mvarId_569_, v___x_575_, v_mvarId_569_, v_a_570_, v_a_571_, v_a_572_, v_a_573_);
return v___x_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_intros_x21___boxed(lean_object* v_mvarId_577_, lean_object* v_a_578_, lean_object* v_a_579_, lean_object* v_a_580_, lean_object* v_a_581_, lean_object* v_a_582_){
_start:
{
lean_object* v_res_583_; 
v_res_583_ = lp_mathlib_Lean_MVarId_intros_x21(v_mvarId_577_, v_a_578_, v_a_579_, v_a_580_, v_a_581_);
lean_dec(v_a_581_);
lean_dec_ref(v_a_580_);
lean_dec(v_a_579_);
lean_dec_ref(v_a_578_);
return v_res_583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_getType_x27_x27_spec__0___redArg(lean_object* v_e_584_, lean_object* v___y_585_){
_start:
{
uint8_t v___x_587_; 
v___x_587_ = l_Lean_Expr_hasMVar(v_e_584_);
if (v___x_587_ == 0)
{
lean_object* v___x_588_; 
v___x_588_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_588_, 0, v_e_584_);
return v___x_588_;
}
else
{
lean_object* v___x_589_; lean_object* v_mctx_590_; lean_object* v___x_591_; lean_object* v_fst_592_; lean_object* v_snd_593_; lean_object* v___x_594_; lean_object* v_cache_595_; lean_object* v_zetaDeltaFVarIds_596_; lean_object* v_postponed_597_; lean_object* v_diag_598_; lean_object* v___x_600_; uint8_t v_isShared_601_; uint8_t v_isSharedCheck_607_; 
v___x_589_ = lean_st_ref_get(v___y_585_);
v_mctx_590_ = lean_ctor_get(v___x_589_, 0);
lean_inc_ref(v_mctx_590_);
lean_dec(v___x_589_);
v___x_591_ = l_Lean_instantiateMVarsCore(v_mctx_590_, v_e_584_);
v_fst_592_ = lean_ctor_get(v___x_591_, 0);
lean_inc(v_fst_592_);
v_snd_593_ = lean_ctor_get(v___x_591_, 1);
lean_inc(v_snd_593_);
lean_dec_ref(v___x_591_);
v___x_594_ = lean_st_ref_take(v___y_585_);
v_cache_595_ = lean_ctor_get(v___x_594_, 1);
v_zetaDeltaFVarIds_596_ = lean_ctor_get(v___x_594_, 2);
v_postponed_597_ = lean_ctor_get(v___x_594_, 3);
v_diag_598_ = lean_ctor_get(v___x_594_, 4);
v_isSharedCheck_607_ = !lean_is_exclusive(v___x_594_);
if (v_isSharedCheck_607_ == 0)
{
lean_object* v_unused_608_; 
v_unused_608_ = lean_ctor_get(v___x_594_, 0);
lean_dec(v_unused_608_);
v___x_600_ = v___x_594_;
v_isShared_601_ = v_isSharedCheck_607_;
goto v_resetjp_599_;
}
else
{
lean_inc(v_diag_598_);
lean_inc(v_postponed_597_);
lean_inc(v_zetaDeltaFVarIds_596_);
lean_inc(v_cache_595_);
lean_dec(v___x_594_);
v___x_600_ = lean_box(0);
v_isShared_601_ = v_isSharedCheck_607_;
goto v_resetjp_599_;
}
v_resetjp_599_:
{
lean_object* v___x_603_; 
if (v_isShared_601_ == 0)
{
lean_ctor_set(v___x_600_, 0, v_snd_593_);
v___x_603_ = v___x_600_;
goto v_reusejp_602_;
}
else
{
lean_object* v_reuseFailAlloc_606_; 
v_reuseFailAlloc_606_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_606_, 0, v_snd_593_);
lean_ctor_set(v_reuseFailAlloc_606_, 1, v_cache_595_);
lean_ctor_set(v_reuseFailAlloc_606_, 2, v_zetaDeltaFVarIds_596_);
lean_ctor_set(v_reuseFailAlloc_606_, 3, v_postponed_597_);
lean_ctor_set(v_reuseFailAlloc_606_, 4, v_diag_598_);
v___x_603_ = v_reuseFailAlloc_606_;
goto v_reusejp_602_;
}
v_reusejp_602_:
{
lean_object* v___x_604_; lean_object* v___x_605_; 
v___x_604_ = lean_st_ref_set(v___y_585_, v___x_603_);
v___x_605_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_605_, 0, v_fst_592_);
return v___x_605_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_getType_x27_x27_spec__0___redArg___boxed(lean_object* v_e_609_, lean_object* v___y_610_, lean_object* v___y_611_){
_start:
{
lean_object* v_res_612_; 
v_res_612_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_getType_x27_x27_spec__0___redArg(v_e_609_, v___y_610_);
lean_dec(v___y_610_);
return v_res_612_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_getType_x27_x27_spec__0(lean_object* v_e_613_, lean_object* v___y_614_, lean_object* v___y_615_, lean_object* v___y_616_, lean_object* v___y_617_){
_start:
{
lean_object* v___x_619_; 
v___x_619_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_getType_x27_x27_spec__0___redArg(v_e_613_, v___y_615_);
return v___x_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_getType_x27_x27_spec__0___boxed(lean_object* v_e_620_, lean_object* v___y_621_, lean_object* v___y_622_, lean_object* v___y_623_, lean_object* v___y_624_, lean_object* v___y_625_){
_start:
{
lean_object* v_res_626_; 
v_res_626_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_getType_x27_x27_spec__0(v_e_620_, v___y_621_, v___y_622_, v___y_623_, v___y_624_);
lean_dec(v___y_624_);
lean_dec_ref(v___y_623_);
lean_dec(v___y_622_);
lean_dec_ref(v___y_621_);
return v_res_626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_getType_x27_x27(lean_object* v_mvarId_627_, lean_object* v_a_628_, lean_object* v_a_629_, lean_object* v_a_630_, lean_object* v_a_631_){
_start:
{
lean_object* v___x_633_; 
v___x_633_ = l_Lean_MVarId_getType(v_mvarId_627_, v_a_628_, v_a_629_, v_a_630_, v_a_631_);
if (lean_obj_tag(v___x_633_) == 0)
{
lean_object* v_a_634_; lean_object* v___x_635_; lean_object* v_a_636_; lean_object* v___x_638_; uint8_t v_isShared_639_; uint8_t v_isSharedCheck_644_; 
v_a_634_ = lean_ctor_get(v___x_633_, 0);
lean_inc(v_a_634_);
lean_dec_ref_known(v___x_633_, 1);
v___x_635_ = lp_mathlib_Lean_instantiateMVars___at___00Lean_MVarId_getType_x27_x27_spec__0___redArg(v_a_634_, v_a_629_);
v_a_636_ = lean_ctor_get(v___x_635_, 0);
v_isSharedCheck_644_ = !lean_is_exclusive(v___x_635_);
if (v_isSharedCheck_644_ == 0)
{
v___x_638_ = v___x_635_;
v_isShared_639_ = v_isSharedCheck_644_;
goto v_resetjp_637_;
}
else
{
lean_inc(v_a_636_);
lean_dec(v___x_635_);
v___x_638_ = lean_box(0);
v_isShared_639_ = v_isSharedCheck_644_;
goto v_resetjp_637_;
}
v_resetjp_637_:
{
lean_object* v___x_640_; lean_object* v___x_642_; 
v___x_640_ = l_Lean_Expr_cleanupAnnotations(v_a_636_);
if (v_isShared_639_ == 0)
{
lean_ctor_set(v___x_638_, 0, v___x_640_);
v___x_642_ = v___x_638_;
goto v_reusejp_641_;
}
else
{
lean_object* v_reuseFailAlloc_643_; 
v_reuseFailAlloc_643_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_643_, 0, v___x_640_);
v___x_642_ = v_reuseFailAlloc_643_;
goto v_reusejp_641_;
}
v_reusejp_641_:
{
return v___x_642_;
}
}
}
else
{
return v___x_633_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_getType_x27_x27___boxed(lean_object* v_mvarId_645_, lean_object* v_a_646_, lean_object* v_a_647_, lean_object* v_a_648_, lean_object* v_a_649_, lean_object* v_a_650_){
_start:
{
lean_object* v_res_651_; 
v_res_651_ = lp_mathlib_Lean_MVarId_getType_x27_x27(v_mvarId_645_, v_a_646_, v_a_647_, v_a_648_, v_a_649_);
lean_dec(v_a_649_);
lean_dec_ref(v_a_648_);
lean_dec(v_a_647_);
lean_dec_ref(v_a_646_);
return v_res_651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27___lam__0(lean_object* v_tac_652_, lean_object* v___y_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_){
_start:
{
lean_object* v___x_662_; 
v___x_662_ = l_Lean_Elab_Tactic_getMainGoal___redArg(v___y_654_, v___y_657_, v___y_658_, v___y_659_, v___y_660_);
if (lean_obj_tag(v___x_662_) == 0)
{
lean_object* v_a_663_; lean_object* v___x_664_; 
v_a_663_ = lean_ctor_get(v___x_662_, 0);
lean_inc(v_a_663_);
lean_dec_ref_known(v___x_662_, 1);
lean_inc(v___y_660_);
lean_inc_ref(v___y_659_);
lean_inc(v___y_658_);
lean_inc_ref(v___y_657_);
v___x_664_ = lean_apply_6(v_tac_652_, v_a_663_, v___y_657_, v___y_658_, v___y_659_, v___y_660_, lean_box(0));
if (lean_obj_tag(v___x_664_) == 0)
{
lean_object* v_a_665_; lean_object* v___x_666_; lean_object* v___x_667_; lean_object* v___x_668_; 
v_a_665_ = lean_ctor_get(v___x_664_, 0);
lean_inc(v_a_665_);
lean_dec_ref_known(v___x_664_, 1);
v___x_666_ = lean_box(0);
v___x_667_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_667_, 0, v_a_665_);
lean_ctor_set(v___x_667_, 1, v___x_666_);
v___x_668_ = l_Lean_Elab_Tactic_replaceMainGoal___redArg(v___x_667_, v___y_654_, v___y_657_, v___y_658_, v___y_659_, v___y_660_);
lean_dec(v___y_660_);
lean_dec_ref(v___y_659_);
lean_dec(v___y_658_);
lean_dec_ref(v___y_657_);
if (lean_obj_tag(v___x_668_) == 0)
{
lean_object* v___x_670_; uint8_t v_isShared_671_; uint8_t v_isSharedCheck_676_; 
v_isSharedCheck_676_ = !lean_is_exclusive(v___x_668_);
if (v_isSharedCheck_676_ == 0)
{
lean_object* v_unused_677_; 
v_unused_677_ = lean_ctor_get(v___x_668_, 0);
lean_dec(v_unused_677_);
v___x_670_ = v___x_668_;
v_isShared_671_ = v_isSharedCheck_676_;
goto v_resetjp_669_;
}
else
{
lean_dec(v___x_668_);
v___x_670_ = lean_box(0);
v_isShared_671_ = v_isSharedCheck_676_;
goto v_resetjp_669_;
}
v_resetjp_669_:
{
lean_object* v___x_672_; lean_object* v___x_674_; 
v___x_672_ = lean_box(0);
if (v_isShared_671_ == 0)
{
lean_ctor_set(v___x_670_, 0, v___x_672_);
v___x_674_ = v___x_670_;
goto v_reusejp_673_;
}
else
{
lean_object* v_reuseFailAlloc_675_; 
v_reuseFailAlloc_675_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_675_, 0, v___x_672_);
v___x_674_ = v_reuseFailAlloc_675_;
goto v_reusejp_673_;
}
v_reusejp_673_:
{
return v___x_674_;
}
}
}
else
{
return v___x_668_;
}
}
else
{
lean_object* v_a_678_; lean_object* v___x_680_; uint8_t v_isShared_681_; uint8_t v_isSharedCheck_685_; 
lean_dec(v___y_660_);
lean_dec_ref(v___y_659_);
lean_dec(v___y_658_);
lean_dec_ref(v___y_657_);
v_a_678_ = lean_ctor_get(v___x_664_, 0);
v_isSharedCheck_685_ = !lean_is_exclusive(v___x_664_);
if (v_isSharedCheck_685_ == 0)
{
v___x_680_ = v___x_664_;
v_isShared_681_ = v_isSharedCheck_685_;
goto v_resetjp_679_;
}
else
{
lean_inc(v_a_678_);
lean_dec(v___x_664_);
v___x_680_ = lean_box(0);
v_isShared_681_ = v_isSharedCheck_685_;
goto v_resetjp_679_;
}
v_resetjp_679_:
{
lean_object* v___x_683_; 
if (v_isShared_681_ == 0)
{
v___x_683_ = v___x_680_;
goto v_reusejp_682_;
}
else
{
lean_object* v_reuseFailAlloc_684_; 
v_reuseFailAlloc_684_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_684_, 0, v_a_678_);
v___x_683_ = v_reuseFailAlloc_684_;
goto v_reusejp_682_;
}
v_reusejp_682_:
{
return v___x_683_;
}
}
}
}
else
{
lean_object* v_a_686_; lean_object* v___x_688_; uint8_t v_isShared_689_; uint8_t v_isSharedCheck_693_; 
lean_dec(v___y_660_);
lean_dec_ref(v___y_659_);
lean_dec(v___y_658_);
lean_dec_ref(v___y_657_);
lean_dec_ref(v_tac_652_);
v_a_686_ = lean_ctor_get(v___x_662_, 0);
v_isSharedCheck_693_ = !lean_is_exclusive(v___x_662_);
if (v_isSharedCheck_693_ == 0)
{
v___x_688_ = v___x_662_;
v_isShared_689_ = v_isSharedCheck_693_;
goto v_resetjp_687_;
}
else
{
lean_inc(v_a_686_);
lean_dec(v___x_662_);
v___x_688_ = lean_box(0);
v_isShared_689_ = v_isSharedCheck_693_;
goto v_resetjp_687_;
}
v_resetjp_687_:
{
lean_object* v___x_691_; 
if (v_isShared_689_ == 0)
{
v___x_691_ = v___x_688_;
goto v_reusejp_690_;
}
else
{
lean_object* v_reuseFailAlloc_692_; 
v_reuseFailAlloc_692_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_692_, 0, v_a_686_);
v___x_691_ = v_reuseFailAlloc_692_;
goto v_reusejp_690_;
}
v_reusejp_690_:
{
return v___x_691_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27___lam__0___boxed(lean_object* v_tac_694_, lean_object* v___y_695_, lean_object* v___y_696_, lean_object* v___y_697_, lean_object* v___y_698_, lean_object* v___y_699_, lean_object* v___y_700_, lean_object* v___y_701_, lean_object* v___y_702_, lean_object* v___y_703_){
_start:
{
lean_object* v_res_704_; 
v_res_704_ = lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27___lam__0(v_tac_694_, v___y_695_, v___y_696_, v___y_697_, v___y_698_, v___y_699_, v___y_700_, v___y_701_, v___y_702_);
lean_dec(v___y_698_);
lean_dec_ref(v___y_697_);
lean_dec(v___y_696_);
lean_dec_ref(v___y_695_);
return v_res_704_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27(lean_object* v_tac_705_, lean_object* v_a_706_, lean_object* v_a_707_, lean_object* v_a_708_, lean_object* v_a_709_, lean_object* v_a_710_, lean_object* v_a_711_, lean_object* v_a_712_, lean_object* v_a_713_){
_start:
{
lean_object* v___f_715_; lean_object* v___x_716_; 
v___f_715_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27___lam__0___boxed), 10, 1);
lean_closure_set(v___f_715_, 0, v_tac_705_);
v___x_716_ = l_Lean_Elab_Tactic_withMainContext___redArg(v___f_715_, v_a_706_, v_a_707_, v_a_708_, v_a_709_, v_a_710_, v_a_711_, v_a_712_, v_a_713_);
return v___x_716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27___boxed(lean_object* v_tac_717_, lean_object* v_a_718_, lean_object* v_a_719_, lean_object* v_a_720_, lean_object* v_a_721_, lean_object* v_a_722_, lean_object* v_a_723_, lean_object* v_a_724_, lean_object* v_a_725_, lean_object* v_a_726_){
_start:
{
lean_object* v_res_727_; 
v_res_727_ = lp_mathlib_Lean_Elab_Tactic_liftMetaTactic_x27(v_tac_717_, v_a_718_, v_a_719_, v_a_720_, v_a_721_, v_a_722_, v_a_723_, v_a_724_, v_a_725_);
lean_dec(v_a_725_);
lean_dec_ref(v_a_724_);
lean_dec(v_a_723_);
lean_dec_ref(v_a_722_);
lean_dec(v_a_721_);
lean_dec_ref(v_a_720_);
lean_dec(v_a_719_);
lean_dec_ref(v_a_718_);
return v_res_727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore___redArg(lean_object* v_x_728_, lean_object* v_ctx_729_, lean_object* v_s_730_, lean_object* v_a_731_, lean_object* v_a_732_, lean_object* v_a_733_, lean_object* v_a_734_, lean_object* v_a_735_, lean_object* v_a_736_){
_start:
{
lean_object* v___x_738_; lean_object* v___x_739_; 
v___x_738_ = lean_st_mk_ref(v_s_730_);
lean_inc(v_a_736_);
lean_inc_ref(v_a_735_);
lean_inc(v_a_734_);
lean_inc_ref(v_a_733_);
lean_inc(v_a_732_);
lean_inc_ref(v_a_731_);
lean_inc(v___x_738_);
v___x_739_ = lean_apply_9(v_x_728_, v_ctx_729_, v___x_738_, v_a_731_, v_a_732_, v_a_733_, v_a_734_, v_a_735_, v_a_736_, lean_box(0));
if (lean_obj_tag(v___x_739_) == 0)
{
lean_object* v_a_740_; lean_object* v___x_742_; uint8_t v_isShared_743_; uint8_t v_isSharedCheck_749_; 
v_a_740_ = lean_ctor_get(v___x_739_, 0);
v_isSharedCheck_749_ = !lean_is_exclusive(v___x_739_);
if (v_isSharedCheck_749_ == 0)
{
v___x_742_ = v___x_739_;
v_isShared_743_ = v_isSharedCheck_749_;
goto v_resetjp_741_;
}
else
{
lean_inc(v_a_740_);
lean_dec(v___x_739_);
v___x_742_ = lean_box(0);
v_isShared_743_ = v_isSharedCheck_749_;
goto v_resetjp_741_;
}
v_resetjp_741_:
{
lean_object* v___x_744_; lean_object* v___x_745_; lean_object* v___x_747_; 
v___x_744_ = lean_st_ref_get(v___x_738_);
lean_dec(v___x_738_);
v___x_745_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_745_, 0, v_a_740_);
lean_ctor_set(v___x_745_, 1, v___x_744_);
if (v_isShared_743_ == 0)
{
lean_ctor_set(v___x_742_, 0, v___x_745_);
v___x_747_ = v___x_742_;
goto v_reusejp_746_;
}
else
{
lean_object* v_reuseFailAlloc_748_; 
v_reuseFailAlloc_748_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_748_, 0, v___x_745_);
v___x_747_ = v_reuseFailAlloc_748_;
goto v_reusejp_746_;
}
v_reusejp_746_:
{
return v___x_747_;
}
}
}
else
{
lean_object* v_a_750_; lean_object* v___x_752_; uint8_t v_isShared_753_; uint8_t v_isSharedCheck_757_; 
lean_dec(v___x_738_);
v_a_750_ = lean_ctor_get(v___x_739_, 0);
v_isSharedCheck_757_ = !lean_is_exclusive(v___x_739_);
if (v_isSharedCheck_757_ == 0)
{
v___x_752_ = v___x_739_;
v_isShared_753_ = v_isSharedCheck_757_;
goto v_resetjp_751_;
}
else
{
lean_inc(v_a_750_);
lean_dec(v___x_739_);
v___x_752_ = lean_box(0);
v_isShared_753_ = v_isSharedCheck_757_;
goto v_resetjp_751_;
}
v_resetjp_751_:
{
lean_object* v___x_755_; 
if (v_isShared_753_ == 0)
{
v___x_755_ = v___x_752_;
goto v_reusejp_754_;
}
else
{
lean_object* v_reuseFailAlloc_756_; 
v_reuseFailAlloc_756_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_756_, 0, v_a_750_);
v___x_755_ = v_reuseFailAlloc_756_;
goto v_reusejp_754_;
}
v_reusejp_754_:
{
return v___x_755_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore___redArg___boxed(lean_object* v_x_758_, lean_object* v_ctx_759_, lean_object* v_s_760_, lean_object* v_a_761_, lean_object* v_a_762_, lean_object* v_a_763_, lean_object* v_a_764_, lean_object* v_a_765_, lean_object* v_a_766_, lean_object* v_a_767_){
_start:
{
lean_object* v_res_768_; 
v_res_768_ = lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore___redArg(v_x_758_, v_ctx_759_, v_s_760_, v_a_761_, v_a_762_, v_a_763_, v_a_764_, v_a_765_, v_a_766_);
lean_dec(v_a_766_);
lean_dec_ref(v_a_765_);
lean_dec(v_a_764_);
lean_dec_ref(v_a_763_);
lean_dec(v_a_762_);
lean_dec_ref(v_a_761_);
return v_res_768_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore(lean_object* v_00_u03b1_769_, lean_object* v_x_770_, lean_object* v_ctx_771_, lean_object* v_s_772_, lean_object* v_a_773_, lean_object* v_a_774_, lean_object* v_a_775_, lean_object* v_a_776_, lean_object* v_a_777_, lean_object* v_a_778_){
_start:
{
lean_object* v___x_780_; lean_object* v___x_781_; 
v___x_780_ = lean_st_mk_ref(v_s_772_);
lean_inc(v_a_778_);
lean_inc_ref(v_a_777_);
lean_inc(v_a_776_);
lean_inc_ref(v_a_775_);
lean_inc(v_a_774_);
lean_inc_ref(v_a_773_);
lean_inc(v___x_780_);
v___x_781_ = lean_apply_9(v_x_770_, v_ctx_771_, v___x_780_, v_a_773_, v_a_774_, v_a_775_, v_a_776_, v_a_777_, v_a_778_, lean_box(0));
if (lean_obj_tag(v___x_781_) == 0)
{
lean_object* v_a_782_; lean_object* v___x_784_; uint8_t v_isShared_785_; uint8_t v_isSharedCheck_791_; 
v_a_782_ = lean_ctor_get(v___x_781_, 0);
v_isSharedCheck_791_ = !lean_is_exclusive(v___x_781_);
if (v_isSharedCheck_791_ == 0)
{
v___x_784_ = v___x_781_;
v_isShared_785_ = v_isSharedCheck_791_;
goto v_resetjp_783_;
}
else
{
lean_inc(v_a_782_);
lean_dec(v___x_781_);
v___x_784_ = lean_box(0);
v_isShared_785_ = v_isSharedCheck_791_;
goto v_resetjp_783_;
}
v_resetjp_783_:
{
lean_object* v___x_786_; lean_object* v___x_787_; lean_object* v___x_789_; 
v___x_786_ = lean_st_ref_get(v___x_780_);
lean_dec(v___x_780_);
v___x_787_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_787_, 0, v_a_782_);
lean_ctor_set(v___x_787_, 1, v___x_786_);
if (v_isShared_785_ == 0)
{
lean_ctor_set(v___x_784_, 0, v___x_787_);
v___x_789_ = v___x_784_;
goto v_reusejp_788_;
}
else
{
lean_object* v_reuseFailAlloc_790_; 
v_reuseFailAlloc_790_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_790_, 0, v___x_787_);
v___x_789_ = v_reuseFailAlloc_790_;
goto v_reusejp_788_;
}
v_reusejp_788_:
{
return v___x_789_;
}
}
}
else
{
lean_object* v_a_792_; lean_object* v___x_794_; uint8_t v_isShared_795_; uint8_t v_isSharedCheck_799_; 
lean_dec(v___x_780_);
v_a_792_ = lean_ctor_get(v___x_781_, 0);
v_isSharedCheck_799_ = !lean_is_exclusive(v___x_781_);
if (v_isSharedCheck_799_ == 0)
{
v___x_794_ = v___x_781_;
v_isShared_795_ = v_isSharedCheck_799_;
goto v_resetjp_793_;
}
else
{
lean_inc(v_a_792_);
lean_dec(v___x_781_);
v___x_794_ = lean_box(0);
v_isShared_795_ = v_isSharedCheck_799_;
goto v_resetjp_793_;
}
v_resetjp_793_:
{
lean_object* v___x_797_; 
if (v_isShared_795_ == 0)
{
v___x_797_ = v___x_794_;
goto v_reusejp_796_;
}
else
{
lean_object* v_reuseFailAlloc_798_; 
v_reuseFailAlloc_798_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_798_, 0, v_a_792_);
v___x_797_ = v_reuseFailAlloc_798_;
goto v_reusejp_796_;
}
v_reusejp_796_:
{
return v___x_797_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore___boxed(lean_object* v_00_u03b1_800_, lean_object* v_x_801_, lean_object* v_ctx_802_, lean_object* v_s_803_, lean_object* v_a_804_, lean_object* v_a_805_, lean_object* v_a_806_, lean_object* v_a_807_, lean_object* v_a_808_, lean_object* v_a_809_, lean_object* v_a_810_){
_start:
{
lean_object* v_res_811_; 
v_res_811_ = lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore(v_00_u03b1_800_, v_x_801_, v_ctx_802_, v_s_803_, v_a_804_, v_a_805_, v_a_806_, v_a_807_, v_a_808_, v_a_809_);
lean_dec(v_a_809_);
lean_dec_ref(v_a_808_);
lean_dec(v_a_807_);
lean_dec_ref(v_a_806_);
lean_dec(v_a_805_);
lean_dec_ref(v_a_804_);
return v_res_811_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore_x27___redArg(lean_object* v_x_812_, lean_object* v_ctx_813_, lean_object* v_s_814_, lean_object* v_a_815_, lean_object* v_a_816_, lean_object* v_a_817_, lean_object* v_a_818_, lean_object* v_a_819_, lean_object* v_a_820_){
_start:
{
lean_object* v___x_822_; lean_object* v___x_823_; 
v___x_822_ = lean_st_mk_ref(v_s_814_);
lean_inc(v_a_820_);
lean_inc_ref(v_a_819_);
lean_inc(v_a_818_);
lean_inc_ref(v_a_817_);
lean_inc(v_a_816_);
lean_inc_ref(v_a_815_);
lean_inc(v___x_822_);
v___x_823_ = lean_apply_9(v_x_812_, v_ctx_813_, v___x_822_, v_a_815_, v_a_816_, v_a_817_, v_a_818_, v_a_819_, v_a_820_, lean_box(0));
if (lean_obj_tag(v___x_823_) == 0)
{
lean_object* v_a_824_; lean_object* v___x_826_; uint8_t v_isShared_827_; uint8_t v_isSharedCheck_832_; 
v_a_824_ = lean_ctor_get(v___x_823_, 0);
v_isSharedCheck_832_ = !lean_is_exclusive(v___x_823_);
if (v_isSharedCheck_832_ == 0)
{
v___x_826_ = v___x_823_;
v_isShared_827_ = v_isSharedCheck_832_;
goto v_resetjp_825_;
}
else
{
lean_inc(v_a_824_);
lean_dec(v___x_823_);
v___x_826_ = lean_box(0);
v_isShared_827_ = v_isSharedCheck_832_;
goto v_resetjp_825_;
}
v_resetjp_825_:
{
lean_object* v___x_828_; lean_object* v___x_830_; 
v___x_828_ = lean_st_ref_get(v___x_822_);
lean_dec(v___x_822_);
lean_dec(v___x_828_);
if (v_isShared_827_ == 0)
{
v___x_830_ = v___x_826_;
goto v_reusejp_829_;
}
else
{
lean_object* v_reuseFailAlloc_831_; 
v_reuseFailAlloc_831_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_831_, 0, v_a_824_);
v___x_830_ = v_reuseFailAlloc_831_;
goto v_reusejp_829_;
}
v_reusejp_829_:
{
return v___x_830_;
}
}
}
else
{
lean_dec(v___x_822_);
return v___x_823_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore_x27___redArg___boxed(lean_object* v_x_833_, lean_object* v_ctx_834_, lean_object* v_s_835_, lean_object* v_a_836_, lean_object* v_a_837_, lean_object* v_a_838_, lean_object* v_a_839_, lean_object* v_a_840_, lean_object* v_a_841_, lean_object* v_a_842_){
_start:
{
lean_object* v_res_843_; 
v_res_843_ = lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore_x27___redArg(v_x_833_, v_ctx_834_, v_s_835_, v_a_836_, v_a_837_, v_a_838_, v_a_839_, v_a_840_, v_a_841_);
lean_dec(v_a_841_);
lean_dec_ref(v_a_840_);
lean_dec(v_a_839_);
lean_dec_ref(v_a_838_);
lean_dec(v_a_837_);
lean_dec_ref(v_a_836_);
return v_res_843_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore_x27(lean_object* v_00_u03b1_844_, lean_object* v_x_845_, lean_object* v_ctx_846_, lean_object* v_s_847_, lean_object* v_a_848_, lean_object* v_a_849_, lean_object* v_a_850_, lean_object* v_a_851_, lean_object* v_a_852_, lean_object* v_a_853_){
_start:
{
lean_object* v___x_855_; lean_object* v___x_856_; 
v___x_855_ = lean_st_mk_ref(v_s_847_);
lean_inc(v_a_853_);
lean_inc_ref(v_a_852_);
lean_inc(v_a_851_);
lean_inc_ref(v_a_850_);
lean_inc(v_a_849_);
lean_inc_ref(v_a_848_);
lean_inc(v___x_855_);
v___x_856_ = lean_apply_9(v_x_845_, v_ctx_846_, v___x_855_, v_a_848_, v_a_849_, v_a_850_, v_a_851_, v_a_852_, v_a_853_, lean_box(0));
if (lean_obj_tag(v___x_856_) == 0)
{
lean_object* v_a_857_; lean_object* v___x_859_; uint8_t v_isShared_860_; uint8_t v_isSharedCheck_865_; 
v_a_857_ = lean_ctor_get(v___x_856_, 0);
v_isSharedCheck_865_ = !lean_is_exclusive(v___x_856_);
if (v_isSharedCheck_865_ == 0)
{
v___x_859_ = v___x_856_;
v_isShared_860_ = v_isSharedCheck_865_;
goto v_resetjp_858_;
}
else
{
lean_inc(v_a_857_);
lean_dec(v___x_856_);
v___x_859_ = lean_box(0);
v_isShared_860_ = v_isSharedCheck_865_;
goto v_resetjp_858_;
}
v_resetjp_858_:
{
lean_object* v___x_861_; lean_object* v___x_863_; 
v___x_861_ = lean_st_ref_get(v___x_855_);
lean_dec(v___x_855_);
lean_dec(v___x_861_);
if (v_isShared_860_ == 0)
{
v___x_863_ = v___x_859_;
goto v_reusejp_862_;
}
else
{
lean_object* v_reuseFailAlloc_864_; 
v_reuseFailAlloc_864_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_864_, 0, v_a_857_);
v___x_863_ = v_reuseFailAlloc_864_;
goto v_reusejp_862_;
}
v_reusejp_862_:
{
return v___x_863_;
}
}
}
else
{
lean_dec(v___x_855_);
return v___x_856_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore_x27___boxed(lean_object* v_00_u03b1_866_, lean_object* v_x_867_, lean_object* v_ctx_868_, lean_object* v_s_869_, lean_object* v_a_870_, lean_object* v_a_871_, lean_object* v_a_872_, lean_object* v_a_873_, lean_object* v_a_874_, lean_object* v_a_875_, lean_object* v_a_876_){
_start:
{
lean_object* v_res_877_; 
v_res_877_ = lp_mathlib___private_Mathlib_Lean_Meta_0__Lean_Elab_Tactic_TacticM_runCore_x27(v_00_u03b1_866_, v_x_867_, v_ctx_868_, v_s_869_, v_a_870_, v_a_871_, v_a_872_, v_a_873_, v_a_874_, v_a_875_);
lean_dec(v_a_875_);
lean_dec_ref(v_a_874_);
lean_dec(v_a_873_);
lean_dec_ref(v_a_872_);
lean_dec(v_a_871_);
lean_dec_ref(v_a_870_);
return v_res_877_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0___redArg___lam__0(lean_object* v_x_878_, lean_object* v___y_879_, lean_object* v___y_880_, lean_object* v___y_881_, lean_object* v___y_882_, lean_object* v___y_883_, lean_object* v___y_884_){
_start:
{
lean_object* v___x_886_; 
lean_inc(v___y_880_);
lean_inc_ref(v___y_879_);
v___x_886_ = lean_apply_7(v_x_878_, v___y_879_, v___y_880_, v___y_881_, v___y_882_, v___y_883_, v___y_884_, lean_box(0));
return v___x_886_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0___redArg___lam__0___boxed(lean_object* v_x_887_, lean_object* v___y_888_, lean_object* v___y_889_, lean_object* v___y_890_, lean_object* v___y_891_, lean_object* v___y_892_, lean_object* v___y_893_, lean_object* v___y_894_){
_start:
{
lean_object* v_res_895_; 
v_res_895_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0___redArg___lam__0(v_x_887_, v___y_888_, v___y_889_, v___y_890_, v___y_891_, v___y_892_, v___y_893_);
lean_dec(v___y_889_);
lean_dec_ref(v___y_888_);
return v_res_895_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0___redArg(lean_object* v_mvarId_896_, lean_object* v_x_897_, lean_object* v___y_898_, lean_object* v___y_899_, lean_object* v___y_900_, lean_object* v___y_901_, lean_object* v___y_902_, lean_object* v___y_903_){
_start:
{
lean_object* v___f_905_; lean_object* v___x_906_; 
lean_inc(v___y_899_);
lean_inc_ref(v___y_898_);
v___f_905_ = lean_alloc_closure((void*)(lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_905_, 0, v_x_897_);
lean_closure_set(v___f_905_, 1, v___y_898_);
lean_closure_set(v___f_905_, 2, v___y_899_);
v___x_906_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withMVarContextImp(lean_box(0), v_mvarId_896_, v___f_905_, v___y_900_, v___y_901_, v___y_902_, v___y_903_);
if (lean_obj_tag(v___x_906_) == 0)
{
return v___x_906_;
}
else
{
lean_object* v_a_907_; lean_object* v___x_909_; uint8_t v_isShared_910_; uint8_t v_isSharedCheck_914_; 
v_a_907_ = lean_ctor_get(v___x_906_, 0);
v_isSharedCheck_914_ = !lean_is_exclusive(v___x_906_);
if (v_isSharedCheck_914_ == 0)
{
v___x_909_ = v___x_906_;
v_isShared_910_ = v_isSharedCheck_914_;
goto v_resetjp_908_;
}
else
{
lean_inc(v_a_907_);
lean_dec(v___x_906_);
v___x_909_ = lean_box(0);
v_isShared_910_ = v_isSharedCheck_914_;
goto v_resetjp_908_;
}
v_resetjp_908_:
{
lean_object* v___x_912_; 
if (v_isShared_910_ == 0)
{
v___x_912_ = v___x_909_;
goto v_reusejp_911_;
}
else
{
lean_object* v_reuseFailAlloc_913_; 
v_reuseFailAlloc_913_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_913_, 0, v_a_907_);
v___x_912_ = v_reuseFailAlloc_913_;
goto v_reusejp_911_;
}
v_reusejp_911_:
{
return v___x_912_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0___redArg___boxed(lean_object* v_mvarId_915_, lean_object* v_x_916_, lean_object* v___y_917_, lean_object* v___y_918_, lean_object* v___y_919_, lean_object* v___y_920_, lean_object* v___y_921_, lean_object* v___y_922_, lean_object* v___y_923_){
_start:
{
lean_object* v_res_924_; 
v_res_924_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0___redArg(v_mvarId_915_, v_x_916_, v___y_917_, v___y_918_, v___y_919_, v___y_920_, v___y_921_, v___y_922_);
lean_dec(v___y_922_);
lean_dec_ref(v___y_921_);
lean_dec(v___y_920_);
lean_dec_ref(v___y_919_);
lean_dec(v___y_918_);
lean_dec_ref(v___y_917_);
return v_res_924_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0(lean_object* v_00_u03b1_925_, lean_object* v_mvarId_926_, lean_object* v_x_927_, lean_object* v___y_928_, lean_object* v___y_929_, lean_object* v___y_930_, lean_object* v___y_931_, lean_object* v___y_932_, lean_object* v___y_933_){
_start:
{
lean_object* v___x_935_; 
v___x_935_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0___redArg(v_mvarId_926_, v_x_927_, v___y_928_, v___y_929_, v___y_930_, v___y_931_, v___y_932_, v___y_933_);
return v___x_935_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0___boxed(lean_object* v_00_u03b1_936_, lean_object* v_mvarId_937_, lean_object* v_x_938_, lean_object* v___y_939_, lean_object* v___y_940_, lean_object* v___y_941_, lean_object* v___y_942_, lean_object* v___y_943_, lean_object* v___y_944_, lean_object* v___y_945_){
_start:
{
lean_object* v_res_946_; 
v_res_946_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0(v_00_u03b1_936_, v_mvarId_937_, v_x_938_, v___y_939_, v___y_940_, v___y_941_, v___y_942_, v___y_943_, v___y_944_);
lean_dec(v___y_944_);
lean_dec_ref(v___y_943_);
lean_dec(v___y_942_);
lean_dec_ref(v___y_941_);
lean_dec(v___y_940_);
lean_dec_ref(v___y_939_);
return v_res_946_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_run__for___redArg___lam__0(lean_object* v___y_947_, lean_object* v_pendingMVars_948_, lean_object* v_a_x3f_949_){
_start:
{
lean_object* v___x_951_; lean_object* v_levelNames_952_; lean_object* v_syntheticMVars_953_; lean_object* v_mvarErrorInfos_954_; lean_object* v_levelMVarErrorInfos_955_; lean_object* v_mvarArgNames_956_; lean_object* v_letRecsToLift_957_; lean_object* v___x_959_; uint8_t v_isShared_960_; uint8_t v_isSharedCheck_967_; 
v___x_951_ = lean_st_ref_take(v___y_947_);
v_levelNames_952_ = lean_ctor_get(v___x_951_, 0);
v_syntheticMVars_953_ = lean_ctor_get(v___x_951_, 1);
v_mvarErrorInfos_954_ = lean_ctor_get(v___x_951_, 3);
v_levelMVarErrorInfos_955_ = lean_ctor_get(v___x_951_, 4);
v_mvarArgNames_956_ = lean_ctor_get(v___x_951_, 5);
v_letRecsToLift_957_ = lean_ctor_get(v___x_951_, 6);
v_isSharedCheck_967_ = !lean_is_exclusive(v___x_951_);
if (v_isSharedCheck_967_ == 0)
{
lean_object* v_unused_968_; 
v_unused_968_ = lean_ctor_get(v___x_951_, 2);
lean_dec(v_unused_968_);
v___x_959_ = v___x_951_;
v_isShared_960_ = v_isSharedCheck_967_;
goto v_resetjp_958_;
}
else
{
lean_inc(v_letRecsToLift_957_);
lean_inc(v_mvarArgNames_956_);
lean_inc(v_levelMVarErrorInfos_955_);
lean_inc(v_mvarErrorInfos_954_);
lean_inc(v_syntheticMVars_953_);
lean_inc(v_levelNames_952_);
lean_dec(v___x_951_);
v___x_959_ = lean_box(0);
v_isShared_960_ = v_isSharedCheck_967_;
goto v_resetjp_958_;
}
v_resetjp_958_:
{
lean_object* v___x_962_; 
if (v_isShared_960_ == 0)
{
lean_ctor_set(v___x_959_, 2, v_pendingMVars_948_);
v___x_962_ = v___x_959_;
goto v_reusejp_961_;
}
else
{
lean_object* v_reuseFailAlloc_966_; 
v_reuseFailAlloc_966_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_966_, 0, v_levelNames_952_);
lean_ctor_set(v_reuseFailAlloc_966_, 1, v_syntheticMVars_953_);
lean_ctor_set(v_reuseFailAlloc_966_, 2, v_pendingMVars_948_);
lean_ctor_set(v_reuseFailAlloc_966_, 3, v_mvarErrorInfos_954_);
lean_ctor_set(v_reuseFailAlloc_966_, 4, v_levelMVarErrorInfos_955_);
lean_ctor_set(v_reuseFailAlloc_966_, 5, v_mvarArgNames_956_);
lean_ctor_set(v_reuseFailAlloc_966_, 6, v_letRecsToLift_957_);
v___x_962_ = v_reuseFailAlloc_966_;
goto v_reusejp_961_;
}
v_reusejp_961_:
{
lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; 
v___x_963_ = lean_st_ref_set(v___y_947_, v___x_962_);
v___x_964_ = lean_box(0);
v___x_965_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_965_, 0, v___x_964_);
return v___x_965_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_run__for___redArg___lam__0___boxed(lean_object* v___y_969_, lean_object* v_pendingMVars_970_, lean_object* v_a_x3f_971_, lean_object* v___y_972_){
_start:
{
lean_object* v_res_973_; 
v_res_973_ = lp_mathlib_Lean_Elab_Tactic_run__for___redArg___lam__0(v___y_969_, v_pendingMVars_970_, v_a_x3f_971_);
lean_dec(v_a_x3f_971_);
lean_dec(v___y_969_);
return v_res_973_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_run__for___redArg___lam__1(lean_object* v_mvarId_977_, lean_object* v_x_978_, lean_object* v___y_979_, lean_object* v___y_980_, lean_object* v___y_981_, lean_object* v___y_982_, lean_object* v___y_983_, lean_object* v___y_984_){
_start:
{
lean_object* v___x_986_; lean_object* v___x_987_; lean_object* v_levelNames_988_; lean_object* v_syntheticMVars_989_; lean_object* v_mvarErrorInfos_990_; lean_object* v_levelMVarErrorInfos_991_; lean_object* v_mvarArgNames_992_; lean_object* v_letRecsToLift_993_; lean_object* v___x_995_; uint8_t v_isShared_996_; uint8_t v_isSharedCheck_1063_; 
v___x_986_ = lean_st_ref_get(v___y_980_);
v___x_987_ = lean_st_ref_take(v___y_980_);
v_levelNames_988_ = lean_ctor_get(v___x_987_, 0);
v_syntheticMVars_989_ = lean_ctor_get(v___x_987_, 1);
v_mvarErrorInfos_990_ = lean_ctor_get(v___x_987_, 3);
v_levelMVarErrorInfos_991_ = lean_ctor_get(v___x_987_, 4);
v_mvarArgNames_992_ = lean_ctor_get(v___x_987_, 5);
v_letRecsToLift_993_ = lean_ctor_get(v___x_987_, 6);
v_isSharedCheck_1063_ = !lean_is_exclusive(v___x_987_);
if (v_isSharedCheck_1063_ == 0)
{
lean_object* v_unused_1064_; 
v_unused_1064_ = lean_ctor_get(v___x_987_, 2);
lean_dec(v_unused_1064_);
v___x_995_ = v___x_987_;
v_isShared_996_ = v_isSharedCheck_1063_;
goto v_resetjp_994_;
}
else
{
lean_inc(v_letRecsToLift_993_);
lean_inc(v_mvarArgNames_992_);
lean_inc(v_levelMVarErrorInfos_991_);
lean_inc(v_mvarErrorInfos_990_);
lean_inc(v_syntheticMVars_989_);
lean_inc(v_levelNames_988_);
lean_dec(v___x_987_);
v___x_995_ = lean_box(0);
v_isShared_996_ = v_isSharedCheck_1063_;
goto v_resetjp_994_;
}
v_resetjp_994_:
{
lean_object* v___x_997_; lean_object* v___x_999_; 
v___x_997_ = lean_box(0);
if (v_isShared_996_ == 0)
{
lean_ctor_set(v___x_995_, 2, v___x_997_);
v___x_999_ = v___x_995_;
goto v_reusejp_998_;
}
else
{
lean_object* v_reuseFailAlloc_1062_; 
v_reuseFailAlloc_1062_ = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(v_reuseFailAlloc_1062_, 0, v_levelNames_988_);
lean_ctor_set(v_reuseFailAlloc_1062_, 1, v_syntheticMVars_989_);
lean_ctor_set(v_reuseFailAlloc_1062_, 2, v___x_997_);
lean_ctor_set(v_reuseFailAlloc_1062_, 3, v_mvarErrorInfos_990_);
lean_ctor_set(v_reuseFailAlloc_1062_, 4, v_levelMVarErrorInfos_991_);
lean_ctor_set(v_reuseFailAlloc_1062_, 5, v_mvarArgNames_992_);
lean_ctor_set(v_reuseFailAlloc_1062_, 6, v_letRecsToLift_993_);
v___x_999_ = v_reuseFailAlloc_1062_;
goto v_reusejp_998_;
}
v_reusejp_998_:
{
lean_object* v___x_1000_; lean_object* v___x_1001_; lean_object* v___x_1002_; lean_object* v_pendingMVars_1003_; lean_object* v_a_1005_; lean_object* v_a_1018_; lean_object* v___x_1029_; 
v___x_1000_ = lean_st_ref_set(v___y_980_, v___x_999_);
v___x_1001_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1001_, 0, v_mvarId_977_);
lean_ctor_set(v___x_1001_, 1, v___x_997_);
v___x_1002_ = lean_st_mk_ref(v___x_1001_);
v_pendingMVars_1003_ = lean_ctor_get(v___x_986_, 2);
lean_inc(v_pendingMVars_1003_);
lean_dec(v___x_986_);
v___x_1029_ = l_Lean_Elab_Tactic_saveState___redArg(v___x_1002_, v___y_980_, v___y_982_, v___y_984_);
if (lean_obj_tag(v___x_1029_) == 0)
{
lean_object* v_a_1030_; lean_object* v___x_1031_; lean_object* v___y_1033_; uint8_t v___y_1034_; lean_object* v_a_1044_; lean_object* v___x_1047_; 
v_a_1030_ = lean_ctor_get(v___x_1029_, 0);
lean_inc(v_a_1030_);
lean_dec_ref_known(v___x_1029_, 1);
v___x_1031_ = ((lean_object*)(lp_mathlib_Lean_Elab_Tactic_run__for___redArg___lam__1___closed__0));
lean_inc(v___y_984_);
lean_inc_ref(v___y_983_);
lean_inc(v___y_982_);
lean_inc_ref(v___y_981_);
lean_inc(v___y_980_);
lean_inc_ref(v___y_979_);
lean_inc(v___x_1002_);
v___x_1047_ = lean_apply_9(v_x_978_, v___x_1031_, v___x_1002_, v___y_979_, v___y_980_, v___y_981_, v___y_982_, v___y_983_, v___y_984_, lean_box(0));
if (lean_obj_tag(v___x_1047_) == 0)
{
lean_object* v_a_1048_; lean_object* v___x_1049_; 
v_a_1048_ = lean_ctor_get(v___x_1047_, 0);
lean_inc(v_a_1048_);
lean_dec_ref_known(v___x_1047_, 1);
v___x_1049_ = l_Lean_Elab_Tactic_getUnsolvedGoals(v___x_1031_, v___x_1002_, v___y_979_, v___y_980_, v___y_981_, v___y_982_, v___y_983_, v___y_984_);
if (lean_obj_tag(v___x_1049_) == 0)
{
lean_object* v_a_1050_; lean_object* v___x_1052_; uint8_t v_isShared_1053_; uint8_t v_isSharedCheck_1058_; 
lean_dec(v_a_1030_);
lean_dec(v___y_984_);
lean_dec_ref(v___y_983_);
lean_dec(v___y_982_);
lean_dec_ref(v___y_981_);
lean_dec_ref(v___y_979_);
v_a_1050_ = lean_ctor_get(v___x_1049_, 0);
v_isSharedCheck_1058_ = !lean_is_exclusive(v___x_1049_);
if (v_isSharedCheck_1058_ == 0)
{
v___x_1052_ = v___x_1049_;
v_isShared_1053_ = v_isSharedCheck_1058_;
goto v_resetjp_1051_;
}
else
{
lean_inc(v_a_1050_);
lean_dec(v___x_1049_);
v___x_1052_ = lean_box(0);
v_isShared_1053_ = v_isSharedCheck_1058_;
goto v_resetjp_1051_;
}
v_resetjp_1051_:
{
lean_object* v___x_1055_; 
if (v_isShared_1053_ == 0)
{
lean_ctor_set_tag(v___x_1052_, 1);
lean_ctor_set(v___x_1052_, 0, v_a_1048_);
v___x_1055_ = v___x_1052_;
goto v_reusejp_1054_;
}
else
{
lean_object* v_reuseFailAlloc_1057_; 
v_reuseFailAlloc_1057_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1057_, 0, v_a_1048_);
v___x_1055_ = v_reuseFailAlloc_1057_;
goto v_reusejp_1054_;
}
v_reusejp_1054_:
{
lean_object* v___x_1056_; 
v___x_1056_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1056_, 0, v___x_1055_);
lean_ctor_set(v___x_1056_, 1, v_a_1050_);
v_a_1005_ = v___x_1056_;
goto v___jp_1004_;
}
}
}
else
{
lean_object* v_a_1059_; 
lean_dec(v_a_1048_);
v_a_1059_ = lean_ctor_get(v___x_1049_, 0);
lean_inc(v_a_1059_);
lean_dec_ref_known(v___x_1049_, 1);
v_a_1044_ = v_a_1059_;
goto v___jp_1043_;
}
}
else
{
lean_object* v_a_1060_; 
v_a_1060_ = lean_ctor_get(v___x_1047_, 0);
lean_inc(v_a_1060_);
lean_dec_ref_known(v___x_1047_, 1);
v_a_1044_ = v_a_1060_;
goto v___jp_1043_;
}
v___jp_1032_:
{
if (v___y_1034_ == 0)
{
lean_object* v___x_1035_; 
v___x_1035_ = l_Lean_Elab_Tactic_SavedState_restore___redArg(v_a_1030_, v___y_1034_, v___x_1002_, v___y_979_, v___y_980_, v___y_981_, v___y_982_, v___y_983_, v___y_984_);
if (lean_obj_tag(v___x_1035_) == 0)
{
uint8_t v___x_1036_; 
lean_dec_ref_known(v___x_1035_, 1);
v___x_1036_ = l_Lean_Elab_isAbortTacticException(v___y_1033_);
if (v___x_1036_ == 0)
{
lean_dec(v___x_1002_);
lean_dec(v___y_984_);
lean_dec_ref(v___y_983_);
lean_dec(v___y_982_);
lean_dec_ref(v___y_981_);
lean_dec_ref(v___y_979_);
v_a_1018_ = v___y_1033_;
goto v___jp_1017_;
}
else
{
lean_object* v___x_1037_; 
lean_dec_ref(v___y_1033_);
v___x_1037_ = l_Lean_Elab_Tactic_getUnsolvedGoals(v___x_1031_, v___x_1002_, v___y_979_, v___y_980_, v___y_981_, v___y_982_, v___y_983_, v___y_984_);
lean_dec(v___y_984_);
lean_dec_ref(v___y_983_);
lean_dec(v___y_982_);
lean_dec_ref(v___y_981_);
lean_dec_ref(v___y_979_);
if (lean_obj_tag(v___x_1037_) == 0)
{
lean_object* v_a_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; 
v_a_1038_ = lean_ctor_get(v___x_1037_, 0);
lean_inc(v_a_1038_);
lean_dec_ref_known(v___x_1037_, 1);
v___x_1039_ = lean_box(0);
v___x_1040_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1040_, 0, v___x_1039_);
lean_ctor_set(v___x_1040_, 1, v_a_1038_);
v_a_1005_ = v___x_1040_;
goto v___jp_1004_;
}
else
{
lean_object* v_a_1041_; 
lean_dec(v___x_1002_);
v_a_1041_ = lean_ctor_get(v___x_1037_, 0);
lean_inc(v_a_1041_);
lean_dec_ref_known(v___x_1037_, 1);
v_a_1018_ = v_a_1041_;
goto v___jp_1017_;
}
}
}
else
{
lean_object* v_a_1042_; 
lean_dec_ref(v___y_1033_);
lean_dec(v___x_1002_);
lean_dec(v___y_984_);
lean_dec_ref(v___y_983_);
lean_dec(v___y_982_);
lean_dec_ref(v___y_981_);
lean_dec_ref(v___y_979_);
v_a_1042_ = lean_ctor_get(v___x_1035_, 0);
lean_inc(v_a_1042_);
lean_dec_ref_known(v___x_1035_, 1);
v_a_1018_ = v_a_1042_;
goto v___jp_1017_;
}
}
else
{
lean_dec(v_a_1030_);
lean_dec(v___x_1002_);
lean_dec(v___y_984_);
lean_dec_ref(v___y_983_);
lean_dec(v___y_982_);
lean_dec_ref(v___y_981_);
lean_dec_ref(v___y_979_);
v_a_1018_ = v___y_1033_;
goto v___jp_1017_;
}
}
v___jp_1043_:
{
uint8_t v___x_1045_; 
v___x_1045_ = l_Lean_Exception_isInterrupt(v_a_1044_);
if (v___x_1045_ == 0)
{
uint8_t v___x_1046_; 
lean_inc_ref(v_a_1044_);
v___x_1046_ = l_Lean_Exception_isRuntime(v_a_1044_);
v___y_1033_ = v_a_1044_;
v___y_1034_ = v___x_1046_;
goto v___jp_1032_;
}
else
{
v___y_1033_ = v_a_1044_;
v___y_1034_ = v___x_1045_;
goto v___jp_1032_;
}
}
}
else
{
lean_object* v_a_1061_; 
lean_dec(v___x_1002_);
lean_dec(v___y_984_);
lean_dec_ref(v___y_983_);
lean_dec(v___y_982_);
lean_dec_ref(v___y_981_);
lean_dec_ref(v___y_979_);
lean_dec_ref(v_x_978_);
v_a_1061_ = lean_ctor_get(v___x_1029_, 0);
lean_inc(v_a_1061_);
lean_dec_ref_known(v___x_1029_, 1);
v_a_1018_ = v_a_1061_;
goto v___jp_1017_;
}
v___jp_1004_:
{
lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1010_; uint8_t v_isShared_1011_; uint8_t v_isSharedCheck_1015_; 
v___x_1006_ = lean_st_ref_get(v___x_1002_);
lean_dec(v___x_1002_);
lean_dec(v___x_1006_);
lean_inc_ref(v_a_1005_);
v___x_1007_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1007_, 0, v_a_1005_);
v___x_1008_ = lp_mathlib_Lean_Elab_Tactic_run__for___redArg___lam__0(v___y_980_, v_pendingMVars_1003_, v___x_1007_);
lean_dec_ref_known(v___x_1007_, 1);
lean_dec(v___y_980_);
v_isSharedCheck_1015_ = !lean_is_exclusive(v___x_1008_);
if (v_isSharedCheck_1015_ == 0)
{
lean_object* v_unused_1016_; 
v_unused_1016_ = lean_ctor_get(v___x_1008_, 0);
lean_dec(v_unused_1016_);
v___x_1010_ = v___x_1008_;
v_isShared_1011_ = v_isSharedCheck_1015_;
goto v_resetjp_1009_;
}
else
{
lean_dec(v___x_1008_);
v___x_1010_ = lean_box(0);
v_isShared_1011_ = v_isSharedCheck_1015_;
goto v_resetjp_1009_;
}
v_resetjp_1009_:
{
lean_object* v___x_1013_; 
if (v_isShared_1011_ == 0)
{
lean_ctor_set(v___x_1010_, 0, v_a_1005_);
v___x_1013_ = v___x_1010_;
goto v_reusejp_1012_;
}
else
{
lean_object* v_reuseFailAlloc_1014_; 
v_reuseFailAlloc_1014_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1014_, 0, v_a_1005_);
v___x_1013_ = v_reuseFailAlloc_1014_;
goto v_reusejp_1012_;
}
v_reusejp_1012_:
{
return v___x_1013_;
}
}
}
v___jp_1017_:
{
lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1022_; uint8_t v_isShared_1023_; uint8_t v_isSharedCheck_1027_; 
v___x_1019_ = lean_box(0);
v___x_1020_ = lp_mathlib_Lean_Elab_Tactic_run__for___redArg___lam__0(v___y_980_, v_pendingMVars_1003_, v___x_1019_);
lean_dec(v___y_980_);
v_isSharedCheck_1027_ = !lean_is_exclusive(v___x_1020_);
if (v_isSharedCheck_1027_ == 0)
{
lean_object* v_unused_1028_; 
v_unused_1028_ = lean_ctor_get(v___x_1020_, 0);
lean_dec(v_unused_1028_);
v___x_1022_ = v___x_1020_;
v_isShared_1023_ = v_isSharedCheck_1027_;
goto v_resetjp_1021_;
}
else
{
lean_dec(v___x_1020_);
v___x_1022_ = lean_box(0);
v_isShared_1023_ = v_isSharedCheck_1027_;
goto v_resetjp_1021_;
}
v_resetjp_1021_:
{
lean_object* v___x_1025_; 
if (v_isShared_1023_ == 0)
{
lean_ctor_set_tag(v___x_1022_, 1);
lean_ctor_set(v___x_1022_, 0, v_a_1018_);
v___x_1025_ = v___x_1022_;
goto v_reusejp_1024_;
}
else
{
lean_object* v_reuseFailAlloc_1026_; 
v_reuseFailAlloc_1026_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1026_, 0, v_a_1018_);
v___x_1025_ = v_reuseFailAlloc_1026_;
goto v_reusejp_1024_;
}
v_reusejp_1024_:
{
return v___x_1025_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_run__for___redArg___lam__1___boxed(lean_object* v_mvarId_1065_, lean_object* v_x_1066_, lean_object* v___y_1067_, lean_object* v___y_1068_, lean_object* v___y_1069_, lean_object* v___y_1070_, lean_object* v___y_1071_, lean_object* v___y_1072_, lean_object* v___y_1073_){
_start:
{
lean_object* v_res_1074_; 
v_res_1074_ = lp_mathlib_Lean_Elab_Tactic_run__for___redArg___lam__1(v_mvarId_1065_, v_x_1066_, v___y_1067_, v___y_1068_, v___y_1069_, v___y_1070_, v___y_1071_, v___y_1072_);
return v_res_1074_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_run__for___redArg(lean_object* v_mvarId_1075_, lean_object* v_x_1076_, lean_object* v_a_1077_, lean_object* v_a_1078_, lean_object* v_a_1079_, lean_object* v_a_1080_, lean_object* v_a_1081_, lean_object* v_a_1082_){
_start:
{
lean_object* v___f_1084_; lean_object* v___x_1085_; 
lean_inc(v_mvarId_1075_);
v___f_1084_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Elab_Tactic_run__for___redArg___lam__1___boxed), 9, 2);
lean_closure_set(v___f_1084_, 0, v_mvarId_1075_);
lean_closure_set(v___f_1084_, 1, v_x_1076_);
v___x_1085_ = lp_mathlib_Lean_MVarId_withContext___at___00Lean_Elab_Tactic_run__for_spec__0___redArg(v_mvarId_1075_, v___f_1084_, v_a_1077_, v_a_1078_, v_a_1079_, v_a_1080_, v_a_1081_, v_a_1082_);
return v___x_1085_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_run__for___redArg___boxed(lean_object* v_mvarId_1086_, lean_object* v_x_1087_, lean_object* v_a_1088_, lean_object* v_a_1089_, lean_object* v_a_1090_, lean_object* v_a_1091_, lean_object* v_a_1092_, lean_object* v_a_1093_, lean_object* v_a_1094_){
_start:
{
lean_object* v_res_1095_; 
v_res_1095_ = lp_mathlib_Lean_Elab_Tactic_run__for___redArg(v_mvarId_1086_, v_x_1087_, v_a_1088_, v_a_1089_, v_a_1090_, v_a_1091_, v_a_1092_, v_a_1093_);
lean_dec(v_a_1093_);
lean_dec_ref(v_a_1092_);
lean_dec(v_a_1091_);
lean_dec_ref(v_a_1090_);
lean_dec(v_a_1089_);
lean_dec_ref(v_a_1088_);
return v_res_1095_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_run__for(lean_object* v_00_u03b1_1096_, lean_object* v_mvarId_1097_, lean_object* v_x_1098_, lean_object* v_a_1099_, lean_object* v_a_1100_, lean_object* v_a_1101_, lean_object* v_a_1102_, lean_object* v_a_1103_, lean_object* v_a_1104_){
_start:
{
lean_object* v___x_1106_; 
v___x_1106_ = lp_mathlib_Lean_Elab_Tactic_run__for___redArg(v_mvarId_1097_, v_x_1098_, v_a_1099_, v_a_1100_, v_a_1101_, v_a_1102_, v_a_1103_, v_a_1104_);
return v___x_1106_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_Tactic_run__for___boxed(lean_object* v_00_u03b1_1107_, lean_object* v_mvarId_1108_, lean_object* v_x_1109_, lean_object* v_a_1110_, lean_object* v_a_1111_, lean_object* v_a_1112_, lean_object* v_a_1113_, lean_object* v_a_1114_, lean_object* v_a_1115_, lean_object* v_a_1116_){
_start:
{
lean_object* v_res_1117_; 
v_res_1117_ = lp_mathlib_Lean_Elab_Tactic_run__for(v_00_u03b1_1107_, v_mvarId_1108_, v_x_1109_, v_a_1110_, v_a_1111_, v_a_1112_, v_a_1113_, v_a_1114_, v_a_1115_);
lean_dec(v_a_1115_);
lean_dec_ref(v_a_1114_);
lean_dec(v_a_1113_);
lean_dec_ref(v_a_1112_);
lean_dec(v_a_1111_);
lean_dec_ref(v_a_1110_);
return v_res_1117_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Term(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Assert(uint8_t builtin);
lean_object* runtime_initialize_Lean_Meta_Tactic_Clear(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Lean_Meta(uint8_t builtin) {
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
res = runtime_initialize_Lean_Elab_Term(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Assert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Meta_Tactic_Clear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Lean_Meta(uint8_t builtin) {
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
lean_object* initialize_Lean_Elab_Term(uint8_t builtin);
lean_object* initialize_Lean_Elab_Tactic_Basic(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Assert(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Clear(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_Meta(uint8_t builtin) {
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
res = initialize_Lean_Elab_Term(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Elab_Tactic_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Assert(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Clear(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Lean_Meta(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Lean_Meta(builtin);
}
#ifdef __cplusplus
}
#endif
