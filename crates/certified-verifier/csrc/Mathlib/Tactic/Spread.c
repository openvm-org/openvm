// Lean compiler output
// Module: Mathlib.Tactic.Spread
// Imports: public import Init public meta import Init public import Mathlib.Init public meta import Lean.Elab.Binders
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
size_t lean_usize_add(size_t, size_t);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* lean_array_uget_borrowed(lean_object*, size_t);
lean_object* lean_array_push(lean_object*, lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
lean_object* lean_array_uset(lean_object*, size_t, lean_object*);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
uint64_t l_Lean_instHashableFVarId_hash(lean_object*);
size_t lean_uint64_to_usize(uint64_t);
size_t lean_usize_sub(size_t, size_t);
size_t lean_usize_mul(size_t, size_t);
size_t lean_usize_shift_right(size_t, size_t);
lean_object* lean_nat_add(lean_object*, lean_object*);
size_t lean_usize_land(size_t, size_t);
lean_object* lean_usize_to_nat(size_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
lean_object* lean_array_fset(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_instBEqFVarId_beq(lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkCollisionNode___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_mkEmptyEntries(lean_object*, lean_object*);
uint8_t lean_usize_dec_le(size_t, size_t);
lean_object* l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l_Lean_Macro_throwUnsupported___redArg(lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_TSyntax_getId(lean_object*);
lean_object* l_Lean_Name_eraseMacroScopes(lean_object*);
uint8_t lean_name_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* l_Lean_Macro_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkIdent(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_SepArray_ofElems(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* l_Lean_mkSepArray(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray2___redArg(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Syntax_TSepArray_getElems___redArg(lean_object*);
uint8_t l_Lean_Syntax_isNone(lean_object*);
lean_object* l_Lean_Syntax_getArgs(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTermEnsuringType(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkLetFVars(lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
double lean_float_of_nat(lean_object*);
lean_object* l_Lean_PersistentArray_push___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_addLocalVarInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvarId_x21(lean_object*);
lean_object* lean_local_ctx_find(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_setKind(lean_object*, uint8_t);
lean_object* l_Lean_PersistentArray_set___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_append(lean_object*, lean_object*);
uint8_t l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MessageData_ofName(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_MessageData_ofExpr(lean_object*);
static const lean_string_object lp_mathlib_letImplDetailStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "letImplDetailStx"};
static const lean_object* lp_mathlib_letImplDetailStx___closed__0 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__0_value;
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_letImplDetailStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(108, 95, 82, 225, 100, 211, 248, 36)}};
static const lean_object* lp_mathlib_letImplDetailStx___closed__1 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__1_value;
static const lean_string_object lp_mathlib_letImplDetailStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_letImplDetailStx___closed__2 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__2_value;
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_letImplDetailStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_letImplDetailStx___closed__3 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__3_value;
static const lean_string_object lp_mathlib_letImplDetailStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "let_impl_detail "};
static const lean_object* lp_mathlib_letImplDetailStx___closed__4 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__4_value;
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_letImplDetailStx___closed__4_value)}};
static const lean_object* lp_mathlib_letImplDetailStx___closed__5 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__5_value;
static const lean_string_object lp_mathlib_letImplDetailStx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_letImplDetailStx___closed__6 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__6_value;
static const lean_string_object lp_mathlib_letImplDetailStx___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_letImplDetailStx___closed__7 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__7_value;
static const lean_string_object lp_mathlib_letImplDetailStx___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_letImplDetailStx___closed__8 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__8_value;
static const lean_string_object lp_mathlib_letImplDetailStx___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_letImplDetailStx___closed__9 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__9_value;
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_letImplDetailStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_letImplDetailStx___closed__10_value_aux_0),((lean_object*)&lp_mathlib_letImplDetailStx___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_letImplDetailStx___closed__10_value_aux_1),((lean_object*)&lp_mathlib_letImplDetailStx___closed__8_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_letImplDetailStx___closed__10_value_aux_2),((lean_object*)&lp_mathlib_letImplDetailStx___closed__9_value),LEAN_SCALAR_PTR_LITERAL(36, 143, 235, 174, 172, 186, 143, 206)}};
static const lean_object* lp_mathlib_letImplDetailStx___closed__10 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__10_value;
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 8}, .m_objs = {((lean_object*)&lp_mathlib_letImplDetailStx___closed__10_value)}};
static const lean_object* lp_mathlib_letImplDetailStx___closed__11 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__11_value;
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_letImplDetailStx___closed__3_value),((lean_object*)&lp_mathlib_letImplDetailStx___closed__5_value),((lean_object*)&lp_mathlib_letImplDetailStx___closed__11_value)}};
static const lean_object* lp_mathlib_letImplDetailStx___closed__12 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__12_value;
static const lean_string_object lp_mathlib_letImplDetailStx___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = " := "};
static const lean_object* lp_mathlib_letImplDetailStx___closed__13 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__13_value;
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_letImplDetailStx___closed__13_value)}};
static const lean_object* lp_mathlib_letImplDetailStx___closed__14 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__14_value;
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_letImplDetailStx___closed__3_value),((lean_object*)&lp_mathlib_letImplDetailStx___closed__12_value),((lean_object*)&lp_mathlib_letImplDetailStx___closed__14_value)}};
static const lean_object* lp_mathlib_letImplDetailStx___closed__15 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__15_value;
static const lean_string_object lp_mathlib_letImplDetailStx___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_letImplDetailStx___closed__16 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__16_value;
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_letImplDetailStx___closed__16_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_letImplDetailStx___closed__17 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__17_value;
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_letImplDetailStx___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_letImplDetailStx___closed__18 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__18_value;
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_letImplDetailStx___closed__3_value),((lean_object*)&lp_mathlib_letImplDetailStx___closed__15_value),((lean_object*)&lp_mathlib_letImplDetailStx___closed__18_value)}};
static const lean_object* lp_mathlib_letImplDetailStx___closed__19 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__19_value;
static const lean_string_object lp_mathlib_letImplDetailStx___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "; "};
static const lean_object* lp_mathlib_letImplDetailStx___closed__20 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__20_value;
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_letImplDetailStx___closed__20_value)}};
static const lean_object* lp_mathlib_letImplDetailStx___closed__21 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__21_value;
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_letImplDetailStx___closed__3_value),((lean_object*)&lp_mathlib_letImplDetailStx___closed__19_value),((lean_object*)&lp_mathlib_letImplDetailStx___closed__21_value)}};
static const lean_object* lp_mathlib_letImplDetailStx___closed__22 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__22_value;
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_letImplDetailStx___closed__3_value),((lean_object*)&lp_mathlib_letImplDetailStx___closed__22_value),((lean_object*)&lp_mathlib_letImplDetailStx___closed__18_value)}};
static const lean_object* lp_mathlib_letImplDetailStx___closed__23 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__23_value;
static const lean_ctor_object lp_mathlib_letImplDetailStx___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_letImplDetailStx___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_letImplDetailStx___closed__23_value)}};
static const lean_object* lp_mathlib_letImplDetailStx___closed__24 = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__24_value;
LEAN_EXPORT const lean_object* lp_mathlib_letImplDetailStx = (const lean_object*)&lp_mathlib_letImplDetailStx___closed__24_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00elabLetImplDetail_spec__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00elabLetImplDetail_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00elabLetImplDetail_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00elabLetImplDetail_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4___redArg(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_elabLetImplDetail___lam__0(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_elabLetImplDetail___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__5_spec__8___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__5___redArg(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__6___redArg(size_t, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__6___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_elabLetImplDetail___lam__1(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_elabLetImplDetail___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00elabLetImplDetail_spec__5_spec__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00elabLetImplDetail_spec__5_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static double lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg___closed__0;
static const lean_string_object lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg___closed__1 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg___closed__1_value;
static const lean_array_object lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg___closed__2 = (const lean_object*)&lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_elabLetImplDetail___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_elabLetImplDetail___closed__0 = (const lean_object*)&lp_mathlib_elabLetImplDetail___closed__0_value;
static const lean_string_object lp_mathlib_elabLetImplDetail___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "let"};
static const lean_object* lp_mathlib_elabLetImplDetail___closed__1 = (const lean_object*)&lp_mathlib_elabLetImplDetail___closed__1_value;
static const lean_string_object lp_mathlib_elabLetImplDetail___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "decl"};
static const lean_object* lp_mathlib_elabLetImplDetail___closed__2 = (const lean_object*)&lp_mathlib_elabLetImplDetail___closed__2_value;
static const lean_ctor_object lp_mathlib_elabLetImplDetail___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_elabLetImplDetail___closed__0_value),LEAN_SCALAR_PTR_LITERAL(13, 84, 199, 228, 250, 36, 60, 178)}};
static const lean_ctor_object lp_mathlib_elabLetImplDetail___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_elabLetImplDetail___closed__3_value_aux_0),((lean_object*)&lp_mathlib_elabLetImplDetail___closed__1_value),LEAN_SCALAR_PTR_LITERAL(221, 9, 221, 202, 9, 173, 58, 127)}};
static const lean_ctor_object lp_mathlib_elabLetImplDetail___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_elabLetImplDetail___closed__3_value_aux_1),((lean_object*)&lp_mathlib_elabLetImplDetail___closed__2_value),LEAN_SCALAR_PTR_LITERAL(132, 25, 49, 206, 109, 94, 77, 137)}};
static const lean_object* lp_mathlib_elabLetImplDetail___closed__3 = (const lean_object*)&lp_mathlib_elabLetImplDetail___closed__3_value;
static const lean_string_object lp_mathlib_elabLetImplDetail___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "trace"};
static const lean_object* lp_mathlib_elabLetImplDetail___closed__4 = (const lean_object*)&lp_mathlib_elabLetImplDetail___closed__4_value;
static const lean_ctor_object lp_mathlib_elabLetImplDetail___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_elabLetImplDetail___closed__4_value),LEAN_SCALAR_PTR_LITERAL(212, 145, 141, 177, 67, 149, 127, 197)}};
static const lean_object* lp_mathlib_elabLetImplDetail___closed__5 = (const lean_object*)&lp_mathlib_elabLetImplDetail___closed__5_value;
static lean_once_cell_t lp_mathlib_elabLetImplDetail___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_elabLetImplDetail___closed__6;
static const lean_string_object lp_mathlib_elabLetImplDetail___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " : "};
static const lean_object* lp_mathlib_elabLetImplDetail___closed__7 = (const lean_object*)&lp_mathlib_elabLetImplDetail___closed__7_value;
static lean_once_cell_t lp_mathlib_elabLetImplDetail___closed__8_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_elabLetImplDetail___closed__8;
static lean_once_cell_t lp_mathlib_elabLetImplDetail___closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_elabLetImplDetail___closed__9;
LEAN_EXPORT lean_object* lp_mathlib_elabLetImplDetail(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_elabLetImplDetail___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3(lean_object*, lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__6(lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__5_spec__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__3(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "__spread"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 127, 41, 55, 163, 164, 142, 111)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2___redArg(size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__5(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__5___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__4(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__4___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "let_impl_detail"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6___closed__0_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ":="};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ";"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "}"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "{"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__3_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__4_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__4;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "with"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__6_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__7;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ":"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0(lean_object*, lean_object*, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__7(uint8_t, lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__7___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__0(size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__0___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "structInstField"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__0 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_letImplDetailStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__1_value_aux_0),((lean_object*)&lp_mathlib_letImplDetailStx___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__1_value_aux_1),((lean_object*)&lp_mathlib_letImplDetailStx___closed__8_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__1_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(50, 77, 20, 88, 28, 210, 230, 84)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__1 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__1_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "structInstLVal"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__2 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_letImplDetailStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__3_value_aux_0),((lean_object*)&lp_mathlib_letImplDetailStx___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__3_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__3_value_aux_1),((lean_object*)&lp_mathlib_letImplDetailStx___closed__8_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__3_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(185, 133, 6, 147, 6, 183, 100, 198)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__3 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__3_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_letImplDetailStx___closed__9_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__4 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__4_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "structInstFieldDef"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__5 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__6_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_letImplDetailStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__6_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__6_value_aux_0),((lean_object*)&lp_mathlib_letImplDetailStx___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__6_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__6_value_aux_1),((lean_object*)&lp_mathlib_letImplDetailStx___closed__8_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__6_value_aux_2),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__5_value),LEAN_SCALAR_PTR_LITERAL(81, 102, 39, 227, 176, 252, 65, 103)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__6 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__6_value;
static const lean_string_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "__"};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__7 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__7_value),LEAN_SCALAR_PTR_LITERAL(223, 16, 38, 188, 153, 111, 201, 204)}};
static const lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__8 = (const lean_object*)&lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg(lean_object*, size_t, size_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "structInst"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_letImplDetailStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__1_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__1_value_aux_0),((lean_object*)&lp_mathlib_letImplDetailStx___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__1_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__1_value_aux_1),((lean_object*)&lp_mathlib_letImplDetailStx___closed__8_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__1_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(50, 43, 73, 62, 118, 124, 31, 28)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__1_value;
static const lean_array_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__2_value),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__2_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__3_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "optEllipsis"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_letImplDetailStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_letImplDetailStx___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__5_value_aux_1),((lean_object*)&lp_mathlib_letImplDetailStx___closed__8_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__5_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(13, 1, 242, 203, 207, 188, 181, 160)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 17, .m_capacity = 17, .m_length = 16, .m_data = "structInstFields"};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__6_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_letImplDetailStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_letImplDetailStx___closed__7_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_letImplDetailStx___closed__8_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(0, 82, 141, 43, 62, 171, 163, 69)}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__7_value;
static const lean_array_object lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_58_; lean_object* v___x_59_; lean_object* v___x_60_; 
v___x_58_ = lean_box(0);
v___x_59_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_60_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_60_, 0, v___x_59_);
lean_ctor_set(v___x_60_, 1, v___x_58_);
return v___x_60_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0___redArg(){
_start:
{
lean_object* v___x_62_; lean_object* v___x_63_; 
v___x_62_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0___redArg___closed__0);
v___x_63_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_63_, 0, v___x_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0___redArg___boxed(lean_object* v___y_64_){
_start:
{
lean_object* v_res_65_; 
v_res_65_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0___redArg();
return v_res_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0(lean_object* v_00_u03b1_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_, lean_object* v___y_72_){
_start:
{
lean_object* v___x_74_; 
v___x_74_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0___redArg();
return v___x_74_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0___boxed(lean_object* v_00_u03b1_75_, lean_object* v___y_76_, lean_object* v___y_77_, lean_object* v___y_78_, lean_object* v___y_79_, lean_object* v___y_80_, lean_object* v___y_81_, lean_object* v___y_82_){
_start:
{
lean_object* v_res_83_; 
v_res_83_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0(v_00_u03b1_75_, v___y_76_, v___y_77_, v___y_78_, v___y_79_, v___y_80_, v___y_81_);
lean_dec(v___y_81_);
lean_dec_ref(v___y_80_);
lean_dec(v___y_79_);
lean_dec_ref(v___y_78_);
lean_dec(v___y_77_);
lean_dec_ref(v___y_76_);
return v_res_83_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00elabLetImplDetail_spec__1___redArg(lean_object* v_e_84_, lean_object* v___y_85_){
_start:
{
uint8_t v___x_87_; 
v___x_87_ = l_Lean_Expr_hasMVar(v_e_84_);
if (v___x_87_ == 0)
{
lean_object* v___x_88_; 
v___x_88_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_88_, 0, v_e_84_);
return v___x_88_;
}
else
{
lean_object* v___x_89_; lean_object* v_mctx_90_; lean_object* v___x_91_; lean_object* v_fst_92_; lean_object* v_snd_93_; lean_object* v___x_94_; lean_object* v_cache_95_; lean_object* v_zetaDeltaFVarIds_96_; lean_object* v_postponed_97_; lean_object* v_diag_98_; lean_object* v___x_100_; uint8_t v_isShared_101_; uint8_t v_isSharedCheck_107_; 
v___x_89_ = lean_st_ref_get(v___y_85_);
v_mctx_90_ = lean_ctor_get(v___x_89_, 0);
lean_inc_ref(v_mctx_90_);
lean_dec(v___x_89_);
v___x_91_ = l_Lean_instantiateMVarsCore(v_mctx_90_, v_e_84_);
v_fst_92_ = lean_ctor_get(v___x_91_, 0);
lean_inc(v_fst_92_);
v_snd_93_ = lean_ctor_get(v___x_91_, 1);
lean_inc(v_snd_93_);
lean_dec_ref(v___x_91_);
v___x_94_ = lean_st_ref_take(v___y_85_);
v_cache_95_ = lean_ctor_get(v___x_94_, 1);
v_zetaDeltaFVarIds_96_ = lean_ctor_get(v___x_94_, 2);
v_postponed_97_ = lean_ctor_get(v___x_94_, 3);
v_diag_98_ = lean_ctor_get(v___x_94_, 4);
v_isSharedCheck_107_ = !lean_is_exclusive(v___x_94_);
if (v_isSharedCheck_107_ == 0)
{
lean_object* v_unused_108_; 
v_unused_108_ = lean_ctor_get(v___x_94_, 0);
lean_dec(v_unused_108_);
v___x_100_ = v___x_94_;
v_isShared_101_ = v_isSharedCheck_107_;
goto v_resetjp_99_;
}
else
{
lean_inc(v_diag_98_);
lean_inc(v_postponed_97_);
lean_inc(v_zetaDeltaFVarIds_96_);
lean_inc(v_cache_95_);
lean_dec(v___x_94_);
v___x_100_ = lean_box(0);
v_isShared_101_ = v_isSharedCheck_107_;
goto v_resetjp_99_;
}
v_resetjp_99_:
{
lean_object* v___x_103_; 
if (v_isShared_101_ == 0)
{
lean_ctor_set(v___x_100_, 0, v_snd_93_);
v___x_103_ = v___x_100_;
goto v_reusejp_102_;
}
else
{
lean_object* v_reuseFailAlloc_106_; 
v_reuseFailAlloc_106_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_106_, 0, v_snd_93_);
lean_ctor_set(v_reuseFailAlloc_106_, 1, v_cache_95_);
lean_ctor_set(v_reuseFailAlloc_106_, 2, v_zetaDeltaFVarIds_96_);
lean_ctor_set(v_reuseFailAlloc_106_, 3, v_postponed_97_);
lean_ctor_set(v_reuseFailAlloc_106_, 4, v_diag_98_);
v___x_103_ = v_reuseFailAlloc_106_;
goto v_reusejp_102_;
}
v_reusejp_102_:
{
lean_object* v___x_104_; lean_object* v___x_105_; 
v___x_104_ = lean_st_ref_set(v___y_85_, v___x_103_);
v___x_105_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_105_, 0, v_fst_92_);
return v___x_105_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00elabLetImplDetail_spec__1___redArg___boxed(lean_object* v_e_109_, lean_object* v___y_110_, lean_object* v___y_111_){
_start:
{
lean_object* v_res_112_; 
v_res_112_ = lp_mathlib_Lean_instantiateMVars___at___00elabLetImplDetail_spec__1___redArg(v_e_109_, v___y_110_);
lean_dec(v___y_110_);
return v_res_112_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00elabLetImplDetail_spec__1(lean_object* v_e_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_, lean_object* v___y_118_, lean_object* v___y_119_){
_start:
{
lean_object* v___x_121_; 
v___x_121_ = lp_mathlib_Lean_instantiateMVars___at___00elabLetImplDetail_spec__1___redArg(v_e_113_, v___y_117_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateMVars___at___00elabLetImplDetail_spec__1___boxed(lean_object* v_e_122_, lean_object* v___y_123_, lean_object* v___y_124_, lean_object* v___y_125_, lean_object* v___y_126_, lean_object* v___y_127_, lean_object* v___y_128_, lean_object* v___y_129_){
_start:
{
lean_object* v_res_130_; 
v_res_130_ = lp_mathlib_Lean_instantiateMVars___at___00elabLetImplDetail_spec__1(v_e_122_, v___y_123_, v___y_124_, v___y_125_, v___y_126_, v___y_127_, v___y_128_);
lean_dec(v___y_128_);
lean_dec_ref(v___y_127_);
lean_dec(v___y_126_);
lean_dec_ref(v___y_125_);
lean_dec(v___y_124_);
lean_dec_ref(v___y_123_);
return v_res_130_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2___redArg___lam__0(lean_object* v_x_131_, lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_, lean_object* v___y_135_, lean_object* v___y_136_, lean_object* v___y_137_){
_start:
{
lean_object* v___x_139_; 
lean_inc(v___y_133_);
lean_inc_ref(v___y_132_);
v___x_139_ = lean_apply_7(v_x_131_, v___y_132_, v___y_133_, v___y_134_, v___y_135_, v___y_136_, v___y_137_, lean_box(0));
return v___x_139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2___redArg___lam__0___boxed(lean_object* v_x_140_, lean_object* v___y_141_, lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v___y_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_){
_start:
{
lean_object* v_res_148_; 
v_res_148_ = lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2___redArg___lam__0(v_x_140_, v___y_141_, v___y_142_, v___y_143_, v___y_144_, v___y_145_, v___y_146_);
lean_dec(v___y_142_);
lean_dec_ref(v___y_141_);
return v_res_148_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2___redArg(lean_object* v_lctx_149_, lean_object* v_localInsts_150_, lean_object* v_x_151_, lean_object* v___y_152_, lean_object* v___y_153_, lean_object* v___y_154_, lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_){
_start:
{
lean_object* v___f_159_; lean_object* v___x_160_; 
lean_inc(v___y_153_);
lean_inc_ref(v___y_152_);
v___f_159_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2___redArg___lam__0___boxed), 8, 3);
lean_closure_set(v___f_159_, 0, v_x_151_);
lean_closure_set(v___f_159_, 1, v___y_152_);
lean_closure_set(v___f_159_, 2, v___y_153_);
v___x_160_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLocalContextImp(lean_box(0), v_lctx_149_, v_localInsts_150_, v___f_159_, v___y_154_, v___y_155_, v___y_156_, v___y_157_);
if (lean_obj_tag(v___x_160_) == 0)
{
return v___x_160_;
}
else
{
lean_object* v_a_161_; lean_object* v___x_163_; uint8_t v_isShared_164_; uint8_t v_isSharedCheck_168_; 
v_a_161_ = lean_ctor_get(v___x_160_, 0);
v_isSharedCheck_168_ = !lean_is_exclusive(v___x_160_);
if (v_isSharedCheck_168_ == 0)
{
v___x_163_ = v___x_160_;
v_isShared_164_ = v_isSharedCheck_168_;
goto v_resetjp_162_;
}
else
{
lean_inc(v_a_161_);
lean_dec(v___x_160_);
v___x_163_ = lean_box(0);
v_isShared_164_ = v_isSharedCheck_168_;
goto v_resetjp_162_;
}
v_resetjp_162_:
{
lean_object* v___x_166_; 
if (v_isShared_164_ == 0)
{
v___x_166_ = v___x_163_;
goto v_reusejp_165_;
}
else
{
lean_object* v_reuseFailAlloc_167_; 
v_reuseFailAlloc_167_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_167_, 0, v_a_161_);
v___x_166_ = v_reuseFailAlloc_167_;
goto v_reusejp_165_;
}
v_reusejp_165_:
{
return v___x_166_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2___redArg___boxed(lean_object* v_lctx_169_, lean_object* v_localInsts_170_, lean_object* v_x_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_, lean_object* v___y_176_, lean_object* v___y_177_, lean_object* v___y_178_){
_start:
{
lean_object* v_res_179_; 
v_res_179_ = lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2___redArg(v_lctx_169_, v_localInsts_170_, v_x_171_, v___y_172_, v___y_173_, v___y_174_, v___y_175_, v___y_176_, v___y_177_);
lean_dec(v___y_177_);
lean_dec_ref(v___y_176_);
lean_dec(v___y_175_);
lean_dec_ref(v___y_174_);
lean_dec(v___y_173_);
lean_dec_ref(v___y_172_);
return v_res_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2(lean_object* v_00_u03b1_180_, lean_object* v_lctx_181_, lean_object* v_localInsts_182_, lean_object* v_x_183_, lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_){
_start:
{
lean_object* v___x_191_; 
v___x_191_ = lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2___redArg(v_lctx_181_, v_localInsts_182_, v_x_183_, v___y_184_, v___y_185_, v___y_186_, v___y_187_, v___y_188_, v___y_189_);
return v___x_191_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2___boxed(lean_object* v_00_u03b1_192_, lean_object* v_lctx_193_, lean_object* v_localInsts_194_, lean_object* v_x_195_, lean_object* v___y_196_, lean_object* v___y_197_, lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_, lean_object* v___y_201_, lean_object* v___y_202_){
_start:
{
lean_object* v_res_203_; 
v_res_203_ = lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2(v_00_u03b1_192_, v_lctx_193_, v_localInsts_194_, v_x_195_, v___y_196_, v___y_197_, v___y_198_, v___y_199_, v___y_200_, v___y_201_);
lean_dec(v___y_201_);
lean_dec_ref(v___y_200_);
lean_dec(v___y_199_);
lean_dec_ref(v___y_198_);
lean_dec(v___y_197_);
lean_dec_ref(v___y_196_);
return v_res_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4___redArg___lam__0(lean_object* v_k_204_, lean_object* v___y_205_, lean_object* v___y_206_, lean_object* v_b_207_, lean_object* v___y_208_, lean_object* v___y_209_, lean_object* v___y_210_, lean_object* v___y_211_){
_start:
{
lean_object* v___x_213_; 
lean_inc(v___y_211_);
lean_inc_ref(v___y_210_);
lean_inc(v___y_209_);
lean_inc_ref(v___y_208_);
lean_inc(v___y_206_);
lean_inc_ref(v___y_205_);
v___x_213_ = lean_apply_8(v_k_204_, v_b_207_, v___y_205_, v___y_206_, v___y_208_, v___y_209_, v___y_210_, v___y_211_, lean_box(0));
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4___redArg___lam__0___boxed(lean_object* v_k_214_, lean_object* v___y_215_, lean_object* v___y_216_, lean_object* v_b_217_, lean_object* v___y_218_, lean_object* v___y_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_){
_start:
{
lean_object* v_res_223_; 
v_res_223_ = lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4___redArg___lam__0(v_k_214_, v___y_215_, v___y_216_, v_b_217_, v___y_218_, v___y_219_, v___y_220_, v___y_221_);
lean_dec(v___y_221_);
lean_dec_ref(v___y_220_);
lean_dec(v___y_219_);
lean_dec_ref(v___y_218_);
lean_dec(v___y_216_);
lean_dec_ref(v___y_215_);
return v_res_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4___redArg(lean_object* v_name_224_, lean_object* v_type_225_, lean_object* v_val_226_, lean_object* v_k_227_, uint8_t v_nondep_228_, uint8_t v_kind_229_, lean_object* v___y_230_, lean_object* v___y_231_, lean_object* v___y_232_, lean_object* v___y_233_, lean_object* v___y_234_, lean_object* v___y_235_){
_start:
{
lean_object* v___f_237_; lean_object* v___x_238_; 
lean_inc(v___y_231_);
lean_inc_ref(v___y_230_);
v___f_237_ = lean_alloc_closure((void*)(lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4___redArg___lam__0___boxed), 9, 3);
lean_closure_set(v___f_237_, 0, v_k_227_);
lean_closure_set(v___f_237_, 1, v___y_230_);
lean_closure_set(v___f_237_, 2, v___y_231_);
v___x_238_ = l___private_Lean_Meta_Basic_0__Lean_Meta_withLetDeclImp(lean_box(0), v_name_224_, v_type_225_, v_val_226_, v___f_237_, v_nondep_228_, v_kind_229_, v___y_232_, v___y_233_, v___y_234_, v___y_235_);
if (lean_obj_tag(v___x_238_) == 0)
{
return v___x_238_;
}
else
{
lean_object* v_a_239_; lean_object* v___x_241_; uint8_t v_isShared_242_; uint8_t v_isSharedCheck_246_; 
v_a_239_ = lean_ctor_get(v___x_238_, 0);
v_isSharedCheck_246_ = !lean_is_exclusive(v___x_238_);
if (v_isSharedCheck_246_ == 0)
{
v___x_241_ = v___x_238_;
v_isShared_242_ = v_isSharedCheck_246_;
goto v_resetjp_240_;
}
else
{
lean_inc(v_a_239_);
lean_dec(v___x_238_);
v___x_241_ = lean_box(0);
v_isShared_242_ = v_isSharedCheck_246_;
goto v_resetjp_240_;
}
v_resetjp_240_:
{
lean_object* v___x_244_; 
if (v_isShared_242_ == 0)
{
v___x_244_ = v___x_241_;
goto v_reusejp_243_;
}
else
{
lean_object* v_reuseFailAlloc_245_; 
v_reuseFailAlloc_245_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_245_, 0, v_a_239_);
v___x_244_ = v_reuseFailAlloc_245_;
goto v_reusejp_243_;
}
v_reusejp_243_:
{
return v___x_244_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4___redArg___boxed(lean_object* v_name_247_, lean_object* v_type_248_, lean_object* v_val_249_, lean_object* v_k_250_, lean_object* v_nondep_251_, lean_object* v_kind_252_, lean_object* v___y_253_, lean_object* v___y_254_, lean_object* v___y_255_, lean_object* v___y_256_, lean_object* v___y_257_, lean_object* v___y_258_, lean_object* v___y_259_){
_start:
{
uint8_t v_nondep_boxed_260_; uint8_t v_kind_boxed_261_; lean_object* v_res_262_; 
v_nondep_boxed_260_ = lean_unbox(v_nondep_251_);
v_kind_boxed_261_ = lean_unbox(v_kind_252_);
v_res_262_ = lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4___redArg(v_name_247_, v_type_248_, v_val_249_, v_k_250_, v_nondep_boxed_260_, v_kind_boxed_261_, v___y_253_, v___y_254_, v___y_255_, v___y_256_, v___y_257_, v___y_258_);
lean_dec(v___y_258_);
lean_dec_ref(v___y_257_);
lean_dec(v___y_256_);
lean_dec_ref(v___y_255_);
lean_dec(v___y_254_);
lean_dec_ref(v___y_253_);
return v_res_262_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4(lean_object* v_00_u03b1_263_, lean_object* v_name_264_, lean_object* v_type_265_, lean_object* v_val_266_, lean_object* v_k_267_, uint8_t v_nondep_268_, uint8_t v_kind_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_, lean_object* v___y_273_, lean_object* v___y_274_, lean_object* v___y_275_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4___redArg(v_name_264_, v_type_265_, v_val_266_, v_k_267_, v_nondep_268_, v_kind_269_, v___y_270_, v___y_271_, v___y_272_, v___y_273_, v___y_274_, v___y_275_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4___boxed(lean_object* v_00_u03b1_278_, lean_object* v_name_279_, lean_object* v_type_280_, lean_object* v_val_281_, lean_object* v_k_282_, lean_object* v_nondep_283_, lean_object* v_kind_284_, lean_object* v___y_285_, lean_object* v___y_286_, lean_object* v___y_287_, lean_object* v___y_288_, lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_){
_start:
{
uint8_t v_nondep_boxed_292_; uint8_t v_kind_boxed_293_; lean_object* v_res_294_; 
v_nondep_boxed_292_ = lean_unbox(v_nondep_283_);
v_kind_boxed_293_ = lean_unbox(v_kind_284_);
v_res_294_ = lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4(v_00_u03b1_278_, v_name_279_, v_type_280_, v_val_281_, v_k_282_, v_nondep_boxed_292_, v_kind_boxed_293_, v___y_285_, v___y_286_, v___y_287_, v___y_288_, v___y_289_, v___y_290_);
lean_dec(v___y_290_);
lean_dec_ref(v___y_289_);
lean_dec(v___y_288_);
lean_dec_ref(v___y_287_);
lean_dec(v___y_286_);
lean_dec_ref(v___y_285_);
return v_res_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_elabLetImplDetail___lam__0(lean_object* v___x_295_, lean_object* v_expectedType_x3f_296_, uint8_t v___x_297_, lean_object* v___x_298_, lean_object* v___x_299_, lean_object* v_x_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_, lean_object* v___y_306_){
_start:
{
lean_object* v___x_308_; 
v___x_308_ = l_Lean_Elab_Term_elabTermEnsuringType(v___x_295_, v_expectedType_x3f_296_, v___x_297_, v___x_297_, v___x_298_, v___y_301_, v___y_302_, v___y_303_, v___y_304_, v___y_305_, v___y_306_);
if (lean_obj_tag(v___x_308_) == 0)
{
lean_object* v_a_309_; lean_object* v___x_310_; lean_object* v_a_311_; lean_object* v___x_312_; lean_object* v___x_313_; uint8_t v___x_314_; uint8_t v___x_315_; lean_object* v___x_316_; 
v_a_309_ = lean_ctor_get(v___x_308_, 0);
lean_inc(v_a_309_);
lean_dec_ref_known(v___x_308_, 1);
v___x_310_ = lp_mathlib_Lean_instantiateMVars___at___00elabLetImplDetail_spec__1___redArg(v_a_309_, v___y_304_);
v_a_311_ = lean_ctor_get(v___x_310_, 0);
lean_inc(v_a_311_);
lean_dec_ref(v___x_310_);
v___x_312_ = lean_mk_empty_array_with_capacity(v___x_299_);
v___x_313_ = lean_array_push(v___x_312_, v_x_300_);
v___x_314_ = 0;
v___x_315_ = 1;
v___x_316_ = l_Lean_Meta_mkLetFVars(v___x_313_, v_a_311_, v___x_314_, v___x_297_, v___x_315_, v___y_303_, v___y_304_, v___y_305_, v___y_306_);
lean_dec_ref(v___x_313_);
return v___x_316_;
}
else
{
lean_dec_ref(v_x_300_);
return v___x_308_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_elabLetImplDetail___lam__0___boxed(lean_object* v___x_317_, lean_object* v_expectedType_x3f_318_, lean_object* v___x_319_, lean_object* v___x_320_, lean_object* v___x_321_, lean_object* v_x_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_, lean_object* v___y_328_, lean_object* v___y_329_){
_start:
{
uint8_t v___x_8222__boxed_330_; lean_object* v_res_331_; 
v___x_8222__boxed_330_ = lean_unbox(v___x_319_);
v_res_331_ = lp_mathlib_elabLetImplDetail___lam__0(v___x_317_, v_expectedType_x3f_318_, v___x_8222__boxed_330_, v___x_320_, v___x_321_, v_x_322_, v___y_323_, v___y_324_, v___y_325_, v___y_326_, v___y_327_, v___y_328_);
lean_dec(v___y_328_);
lean_dec_ref(v___y_327_);
lean_dec(v___y_326_);
lean_dec_ref(v___y_325_);
lean_dec(v___y_324_);
lean_dec_ref(v___y_323_);
lean_dec(v___x_321_);
return v_res_331_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__5_spec__8___redArg(lean_object* v_x_332_, lean_object* v_x_333_, lean_object* v_x_334_, lean_object* v_x_335_){
_start:
{
lean_object* v_ks_336_; lean_object* v_vs_337_; lean_object* v___x_339_; uint8_t v_isShared_340_; uint8_t v_isSharedCheck_361_; 
v_ks_336_ = lean_ctor_get(v_x_332_, 0);
v_vs_337_ = lean_ctor_get(v_x_332_, 1);
v_isSharedCheck_361_ = !lean_is_exclusive(v_x_332_);
if (v_isSharedCheck_361_ == 0)
{
v___x_339_ = v_x_332_;
v_isShared_340_ = v_isSharedCheck_361_;
goto v_resetjp_338_;
}
else
{
lean_inc(v_vs_337_);
lean_inc(v_ks_336_);
lean_dec(v_x_332_);
v___x_339_ = lean_box(0);
v_isShared_340_ = v_isSharedCheck_361_;
goto v_resetjp_338_;
}
v_resetjp_338_:
{
lean_object* v___x_341_; uint8_t v___x_342_; 
v___x_341_ = lean_array_get_size(v_ks_336_);
v___x_342_ = lean_nat_dec_lt(v_x_333_, v___x_341_);
if (v___x_342_ == 0)
{
lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_346_; 
lean_dec(v_x_333_);
v___x_343_ = lean_array_push(v_ks_336_, v_x_334_);
v___x_344_ = lean_array_push(v_vs_337_, v_x_335_);
if (v_isShared_340_ == 0)
{
lean_ctor_set(v___x_339_, 1, v___x_344_);
lean_ctor_set(v___x_339_, 0, v___x_343_);
v___x_346_ = v___x_339_;
goto v_reusejp_345_;
}
else
{
lean_object* v_reuseFailAlloc_347_; 
v_reuseFailAlloc_347_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_347_, 0, v___x_343_);
lean_ctor_set(v_reuseFailAlloc_347_, 1, v___x_344_);
v___x_346_ = v_reuseFailAlloc_347_;
goto v_reusejp_345_;
}
v_reusejp_345_:
{
return v___x_346_;
}
}
else
{
lean_object* v_k_x27_348_; uint8_t v___x_349_; 
v_k_x27_348_ = lean_array_fget_borrowed(v_ks_336_, v_x_333_);
v___x_349_ = l_Lean_instBEqFVarId_beq(v_x_334_, v_k_x27_348_);
if (v___x_349_ == 0)
{
lean_object* v___x_351_; 
if (v_isShared_340_ == 0)
{
v___x_351_ = v___x_339_;
goto v_reusejp_350_;
}
else
{
lean_object* v_reuseFailAlloc_355_; 
v_reuseFailAlloc_355_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_355_, 0, v_ks_336_);
lean_ctor_set(v_reuseFailAlloc_355_, 1, v_vs_337_);
v___x_351_ = v_reuseFailAlloc_355_;
goto v_reusejp_350_;
}
v_reusejp_350_:
{
lean_object* v___x_352_; lean_object* v___x_353_; 
v___x_352_ = lean_unsigned_to_nat(1u);
v___x_353_ = lean_nat_add(v_x_333_, v___x_352_);
lean_dec(v_x_333_);
v_x_332_ = v___x_351_;
v_x_333_ = v___x_353_;
goto _start;
}
}
else
{
lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_359_; 
v___x_356_ = lean_array_fset(v_ks_336_, v_x_333_, v_x_334_);
v___x_357_ = lean_array_fset(v_vs_337_, v_x_333_, v_x_335_);
lean_dec(v_x_333_);
if (v_isShared_340_ == 0)
{
lean_ctor_set(v___x_339_, 1, v___x_357_);
lean_ctor_set(v___x_339_, 0, v___x_356_);
v___x_359_ = v___x_339_;
goto v_reusejp_358_;
}
else
{
lean_object* v_reuseFailAlloc_360_; 
v_reuseFailAlloc_360_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_360_, 0, v___x_356_);
lean_ctor_set(v_reuseFailAlloc_360_, 1, v___x_357_);
v___x_359_ = v_reuseFailAlloc_360_;
goto v_reusejp_358_;
}
v_reusejp_358_:
{
return v___x_359_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__5___redArg(lean_object* v_n_362_, lean_object* v_k_363_, lean_object* v_v_364_){
_start:
{
lean_object* v___x_365_; lean_object* v___x_366_; 
v___x_365_ = lean_unsigned_to_nat(0u);
v___x_366_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__5_spec__8___redArg(v_n_362_, v___x_365_, v_k_363_, v_v_364_);
return v___x_366_;
}
}
static lean_object* _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3___redArg___closed__0(void){
_start:
{
lean_object* v___x_367_; 
v___x_367_ = l_Lean_PersistentHashMap_mkEmptyEntries(lean_box(0), lean_box(0));
return v___x_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3___redArg(lean_object* v_x_368_, size_t v_x_369_, size_t v_x_370_, lean_object* v_x_371_, lean_object* v_x_372_){
_start:
{
if (lean_obj_tag(v_x_368_) == 0)
{
lean_object* v_es_373_; size_t v___x_374_; size_t v___x_375_; lean_object* v_j_376_; lean_object* v___x_377_; uint8_t v___x_378_; 
v_es_373_ = lean_ctor_get(v_x_368_, 0);
v___x_374_ = ((size_t)31ULL);
v___x_375_ = lean_usize_land(v_x_369_, v___x_374_);
v_j_376_ = lean_usize_to_nat(v___x_375_);
v___x_377_ = lean_array_get_size(v_es_373_);
v___x_378_ = lean_nat_dec_lt(v_j_376_, v___x_377_);
if (v___x_378_ == 0)
{
lean_dec(v_j_376_);
lean_dec(v_x_372_);
lean_dec(v_x_371_);
return v_x_368_;
}
else
{
lean_object* v___x_380_; uint8_t v_isShared_381_; uint8_t v_isSharedCheck_417_; 
lean_inc_ref(v_es_373_);
v_isSharedCheck_417_ = !lean_is_exclusive(v_x_368_);
if (v_isSharedCheck_417_ == 0)
{
lean_object* v_unused_418_; 
v_unused_418_ = lean_ctor_get(v_x_368_, 0);
lean_dec(v_unused_418_);
v___x_380_ = v_x_368_;
v_isShared_381_ = v_isSharedCheck_417_;
goto v_resetjp_379_;
}
else
{
lean_dec(v_x_368_);
v___x_380_ = lean_box(0);
v_isShared_381_ = v_isSharedCheck_417_;
goto v_resetjp_379_;
}
v_resetjp_379_:
{
lean_object* v_v_382_; lean_object* v___x_383_; lean_object* v_xs_x27_384_; lean_object* v___y_386_; 
v_v_382_ = lean_array_fget(v_es_373_, v_j_376_);
v___x_383_ = lean_box(0);
v_xs_x27_384_ = lean_array_fset(v_es_373_, v_j_376_, v___x_383_);
switch(lean_obj_tag(v_v_382_))
{
case 0:
{
lean_object* v_key_391_; lean_object* v_val_392_; lean_object* v___x_394_; uint8_t v_isShared_395_; uint8_t v_isSharedCheck_402_; 
v_key_391_ = lean_ctor_get(v_v_382_, 0);
v_val_392_ = lean_ctor_get(v_v_382_, 1);
v_isSharedCheck_402_ = !lean_is_exclusive(v_v_382_);
if (v_isSharedCheck_402_ == 0)
{
v___x_394_ = v_v_382_;
v_isShared_395_ = v_isSharedCheck_402_;
goto v_resetjp_393_;
}
else
{
lean_inc(v_val_392_);
lean_inc(v_key_391_);
lean_dec(v_v_382_);
v___x_394_ = lean_box(0);
v_isShared_395_ = v_isSharedCheck_402_;
goto v_resetjp_393_;
}
v_resetjp_393_:
{
uint8_t v___x_396_; 
v___x_396_ = l_Lean_instBEqFVarId_beq(v_x_371_, v_key_391_);
if (v___x_396_ == 0)
{
lean_object* v___x_397_; lean_object* v___x_398_; 
lean_del_object(v___x_394_);
v___x_397_ = l_Lean_PersistentHashMap_mkCollisionNode___redArg(v_key_391_, v_val_392_, v_x_371_, v_x_372_);
v___x_398_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_398_, 0, v___x_397_);
v___y_386_ = v___x_398_;
goto v___jp_385_;
}
else
{
lean_object* v___x_400_; 
lean_dec(v_val_392_);
lean_dec(v_key_391_);
if (v_isShared_395_ == 0)
{
lean_ctor_set(v___x_394_, 1, v_x_372_);
lean_ctor_set(v___x_394_, 0, v_x_371_);
v___x_400_ = v___x_394_;
goto v_reusejp_399_;
}
else
{
lean_object* v_reuseFailAlloc_401_; 
v_reuseFailAlloc_401_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_401_, 0, v_x_371_);
lean_ctor_set(v_reuseFailAlloc_401_, 1, v_x_372_);
v___x_400_ = v_reuseFailAlloc_401_;
goto v_reusejp_399_;
}
v_reusejp_399_:
{
v___y_386_ = v___x_400_;
goto v___jp_385_;
}
}
}
}
case 1:
{
lean_object* v_node_403_; lean_object* v___x_405_; uint8_t v_isShared_406_; uint8_t v_isSharedCheck_415_; 
v_node_403_ = lean_ctor_get(v_v_382_, 0);
v_isSharedCheck_415_ = !lean_is_exclusive(v_v_382_);
if (v_isSharedCheck_415_ == 0)
{
v___x_405_ = v_v_382_;
v_isShared_406_ = v_isSharedCheck_415_;
goto v_resetjp_404_;
}
else
{
lean_inc(v_node_403_);
lean_dec(v_v_382_);
v___x_405_ = lean_box(0);
v_isShared_406_ = v_isSharedCheck_415_;
goto v_resetjp_404_;
}
v_resetjp_404_:
{
size_t v___x_407_; size_t v___x_408_; size_t v___x_409_; size_t v___x_410_; lean_object* v___x_411_; lean_object* v___x_413_; 
v___x_407_ = ((size_t)5ULL);
v___x_408_ = lean_usize_shift_right(v_x_369_, v___x_407_);
v___x_409_ = ((size_t)1ULL);
v___x_410_ = lean_usize_add(v_x_370_, v___x_409_);
v___x_411_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3___redArg(v_node_403_, v___x_408_, v___x_410_, v_x_371_, v_x_372_);
if (v_isShared_406_ == 0)
{
lean_ctor_set(v___x_405_, 0, v___x_411_);
v___x_413_ = v___x_405_;
goto v_reusejp_412_;
}
else
{
lean_object* v_reuseFailAlloc_414_; 
v_reuseFailAlloc_414_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_414_, 0, v___x_411_);
v___x_413_ = v_reuseFailAlloc_414_;
goto v_reusejp_412_;
}
v_reusejp_412_:
{
v___y_386_ = v___x_413_;
goto v___jp_385_;
}
}
}
default: 
{
lean_object* v___x_416_; 
v___x_416_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_416_, 0, v_x_371_);
lean_ctor_set(v___x_416_, 1, v_x_372_);
v___y_386_ = v___x_416_;
goto v___jp_385_;
}
}
v___jp_385_:
{
lean_object* v___x_387_; lean_object* v___x_389_; 
v___x_387_ = lean_array_fset(v_xs_x27_384_, v_j_376_, v___y_386_);
lean_dec(v_j_376_);
if (v_isShared_381_ == 0)
{
lean_ctor_set(v___x_380_, 0, v___x_387_);
v___x_389_ = v___x_380_;
goto v_reusejp_388_;
}
else
{
lean_object* v_reuseFailAlloc_390_; 
v_reuseFailAlloc_390_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_390_, 0, v___x_387_);
v___x_389_ = v_reuseFailAlloc_390_;
goto v_reusejp_388_;
}
v_reusejp_388_:
{
return v___x_389_;
}
}
}
}
}
else
{
lean_object* v_ks_419_; lean_object* v_vs_420_; lean_object* v___x_422_; uint8_t v_isShared_423_; uint8_t v_isSharedCheck_440_; 
v_ks_419_ = lean_ctor_get(v_x_368_, 0);
v_vs_420_ = lean_ctor_get(v_x_368_, 1);
v_isSharedCheck_440_ = !lean_is_exclusive(v_x_368_);
if (v_isSharedCheck_440_ == 0)
{
v___x_422_ = v_x_368_;
v_isShared_423_ = v_isSharedCheck_440_;
goto v_resetjp_421_;
}
else
{
lean_inc(v_vs_420_);
lean_inc(v_ks_419_);
lean_dec(v_x_368_);
v___x_422_ = lean_box(0);
v_isShared_423_ = v_isSharedCheck_440_;
goto v_resetjp_421_;
}
v_resetjp_421_:
{
lean_object* v___x_425_; 
if (v_isShared_423_ == 0)
{
v___x_425_ = v___x_422_;
goto v_reusejp_424_;
}
else
{
lean_object* v_reuseFailAlloc_439_; 
v_reuseFailAlloc_439_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_439_, 0, v_ks_419_);
lean_ctor_set(v_reuseFailAlloc_439_, 1, v_vs_420_);
v___x_425_ = v_reuseFailAlloc_439_;
goto v_reusejp_424_;
}
v_reusejp_424_:
{
lean_object* v_newNode_426_; uint8_t v___y_428_; size_t v___x_434_; uint8_t v___x_435_; 
v_newNode_426_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__5___redArg(v___x_425_, v_x_371_, v_x_372_);
v___x_434_ = ((size_t)7ULL);
v___x_435_ = lean_usize_dec_le(v___x_434_, v_x_370_);
if (v___x_435_ == 0)
{
lean_object* v___x_436_; lean_object* v___x_437_; uint8_t v___x_438_; 
v___x_436_ = l_Lean_PersistentHashMap_getCollisionNodeSize___redArg(v_newNode_426_);
v___x_437_ = lean_unsigned_to_nat(4u);
v___x_438_ = lean_nat_dec_lt(v___x_436_, v___x_437_);
lean_dec(v___x_436_);
v___y_428_ = v___x_438_;
goto v___jp_427_;
}
else
{
v___y_428_ = v___x_435_;
goto v___jp_427_;
}
v___jp_427_:
{
if (v___y_428_ == 0)
{
lean_object* v_ks_429_; lean_object* v_vs_430_; lean_object* v___x_431_; lean_object* v___x_432_; lean_object* v___x_433_; 
v_ks_429_ = lean_ctor_get(v_newNode_426_, 0);
lean_inc_ref(v_ks_429_);
v_vs_430_ = lean_ctor_get(v_newNode_426_, 1);
lean_inc_ref(v_vs_430_);
lean_dec_ref(v_newNode_426_);
v___x_431_ = lean_unsigned_to_nat(0u);
v___x_432_ = lean_obj_once(&lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3___redArg___closed__0, &lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3___redArg___closed__0_once, _init_lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3___redArg___closed__0);
v___x_433_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__6___redArg(v_x_370_, v_ks_429_, v_vs_430_, v___x_431_, v___x_432_);
lean_dec_ref(v_vs_430_);
lean_dec_ref(v_ks_429_);
return v___x_433_;
}
else
{
return v_newNode_426_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__6___redArg(size_t v_depth_441_, lean_object* v_keys_442_, lean_object* v_vals_443_, lean_object* v_i_444_, lean_object* v_entries_445_){
_start:
{
lean_object* v___x_446_; uint8_t v___x_447_; 
v___x_446_ = lean_array_get_size(v_keys_442_);
v___x_447_ = lean_nat_dec_lt(v_i_444_, v___x_446_);
if (v___x_447_ == 0)
{
lean_dec(v_i_444_);
return v_entries_445_;
}
else
{
lean_object* v_k_448_; lean_object* v_v_449_; uint64_t v___x_450_; size_t v_h_451_; size_t v___x_452_; lean_object* v___x_453_; size_t v___x_454_; size_t v___x_455_; size_t v___x_456_; size_t v_h_457_; lean_object* v___x_458_; lean_object* v___x_459_; 
v_k_448_ = lean_array_fget_borrowed(v_keys_442_, v_i_444_);
v_v_449_ = lean_array_fget_borrowed(v_vals_443_, v_i_444_);
v___x_450_ = l_Lean_instHashableFVarId_hash(v_k_448_);
v_h_451_ = lean_uint64_to_usize(v___x_450_);
v___x_452_ = ((size_t)5ULL);
v___x_453_ = lean_unsigned_to_nat(1u);
v___x_454_ = ((size_t)1ULL);
v___x_455_ = lean_usize_sub(v_depth_441_, v___x_454_);
v___x_456_ = lean_usize_mul(v___x_452_, v___x_455_);
v_h_457_ = lean_usize_shift_right(v_h_451_, v___x_456_);
v___x_458_ = lean_nat_add(v_i_444_, v___x_453_);
lean_dec(v_i_444_);
lean_inc(v_v_449_);
lean_inc(v_k_448_);
v___x_459_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3___redArg(v_entries_445_, v_h_457_, v_depth_441_, v_k_448_, v_v_449_);
v_i_444_ = v___x_458_;
v_entries_445_ = v___x_459_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__6___redArg___boxed(lean_object* v_depth_461_, lean_object* v_keys_462_, lean_object* v_vals_463_, lean_object* v_i_464_, lean_object* v_entries_465_){
_start:
{
size_t v_depth_boxed_466_; lean_object* v_res_467_; 
v_depth_boxed_466_ = lean_unbox_usize(v_depth_461_);
lean_dec(v_depth_461_);
v_res_467_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__6___redArg(v_depth_boxed_466_, v_keys_462_, v_vals_463_, v_i_464_, v_entries_465_);
lean_dec_ref(v_vals_463_);
lean_dec_ref(v_keys_462_);
return v_res_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3___redArg___boxed(lean_object* v_x_468_, lean_object* v_x_469_, lean_object* v_x_470_, lean_object* v_x_471_, lean_object* v_x_472_){
_start:
{
size_t v_x_8351__boxed_473_; size_t v_x_8352__boxed_474_; lean_object* v_res_475_; 
v_x_8351__boxed_473_ = lean_unbox_usize(v_x_469_);
lean_dec(v_x_469_);
v_x_8352__boxed_474_ = lean_unbox_usize(v_x_470_);
lean_dec(v_x_470_);
v_res_475_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3___redArg(v_x_468_, v_x_8351__boxed_473_, v_x_8352__boxed_474_, v_x_471_, v_x_472_);
return v_res_475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3___redArg(lean_object* v_x_476_, lean_object* v_x_477_, lean_object* v_x_478_){
_start:
{
uint64_t v___x_479_; size_t v___x_480_; size_t v___x_481_; lean_object* v___x_482_; 
v___x_479_ = l_Lean_instHashableFVarId_hash(v_x_477_);
v___x_480_ = lean_uint64_to_usize(v___x_479_);
v___x_481_ = ((size_t)1ULL);
v___x_482_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3___redArg(v_x_476_, v___x_480_, v___x_481_, v_x_477_, v_x_478_);
return v___x_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_elabLetImplDetail___lam__1(lean_object* v_id_483_, lean_object* v___x_484_, lean_object* v_expectedType_x3f_485_, uint8_t v___x_486_, lean_object* v___x_487_, lean_object* v_x_488_, lean_object* v___y_489_, lean_object* v___y_490_, lean_object* v___y_491_, lean_object* v___y_492_, lean_object* v___y_493_, lean_object* v___y_494_){
_start:
{
lean_object* v___x_496_; 
lean_inc_ref(v_x_488_);
v___x_496_ = l_Lean_Elab_Term_addLocalVarInfo(v_id_483_, v_x_488_, v___y_489_, v___y_490_, v___y_491_, v___y_492_, v___y_493_, v___y_494_);
if (lean_obj_tag(v___x_496_) == 0)
{
lean_object* v_lctx_497_; lean_object* v_localInstances_498_; lean_object* v___y_500_; lean_object* v_fvarIdToDecl_505_; lean_object* v_decls_506_; lean_object* v_auxDeclToFullName_507_; lean_object* v___x_508_; lean_object* v___x_509_; 
lean_dec_ref_known(v___x_496_, 1);
v_lctx_497_ = lean_ctor_get(v___y_491_, 2);
v_localInstances_498_ = lean_ctor_get(v___y_491_, 3);
v_fvarIdToDecl_505_ = lean_ctor_get(v_lctx_497_, 0);
v_decls_506_ = lean_ctor_get(v_lctx_497_, 1);
v_auxDeclToFullName_507_ = lean_ctor_get(v_lctx_497_, 2);
v___x_508_ = l_Lean_Expr_fvarId_x21(v_x_488_);
lean_inc_ref(v_lctx_497_);
v___x_509_ = lean_local_ctx_find(v_lctx_497_, v___x_508_);
if (lean_obj_tag(v___x_509_) == 0)
{
lean_inc_ref(v_lctx_497_);
v___y_500_ = v_lctx_497_;
goto v___jp_499_;
}
else
{
lean_object* v_val_510_; lean_object* v___x_512_; uint8_t v_isShared_513_; uint8_t v_isSharedCheck_529_; 
v_val_510_ = lean_ctor_get(v___x_509_, 0);
v_isSharedCheck_529_ = !lean_is_exclusive(v___x_509_);
if (v_isSharedCheck_529_ == 0)
{
v___x_512_ = v___x_509_;
v_isShared_513_ = v_isSharedCheck_529_;
goto v_resetjp_511_;
}
else
{
lean_inc(v_val_510_);
lean_dec(v___x_509_);
v___x_512_ = lean_box(0);
v_isShared_513_ = v_isSharedCheck_529_;
goto v_resetjp_511_;
}
v_resetjp_511_:
{
uint8_t v___x_514_; lean_object* v___x_515_; lean_object* v___y_517_; lean_object* v___y_518_; lean_object* v___y_525_; lean_object* v_fvarId_528_; 
v___x_514_ = 1;
v___x_515_ = l_Lean_LocalDecl_setKind(v_val_510_, v___x_514_);
v_fvarId_528_ = lean_ctor_get(v___x_515_, 1);
lean_inc(v_fvarId_528_);
v___y_525_ = v_fvarId_528_;
goto v___jp_524_;
v___jp_516_:
{
lean_object* v___x_520_; 
if (v_isShared_513_ == 0)
{
lean_ctor_set(v___x_512_, 0, v___x_515_);
v___x_520_ = v___x_512_;
goto v_reusejp_519_;
}
else
{
lean_object* v_reuseFailAlloc_523_; 
v_reuseFailAlloc_523_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_523_, 0, v___x_515_);
v___x_520_ = v_reuseFailAlloc_523_;
goto v_reusejp_519_;
}
v_reusejp_519_:
{
lean_object* v___x_521_; lean_object* v___x_522_; 
lean_inc_ref(v_decls_506_);
v___x_521_ = l_Lean_PersistentArray_set___redArg(v_decls_506_, v___y_518_, v___x_520_);
lean_dec(v___y_518_);
lean_inc(v_auxDeclToFullName_507_);
v___x_522_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_522_, 0, v___y_517_);
lean_ctor_set(v___x_522_, 1, v___x_521_);
lean_ctor_set(v___x_522_, 2, v_auxDeclToFullName_507_);
v___y_500_ = v___x_522_;
goto v___jp_499_;
}
}
v___jp_524_:
{
lean_object* v___x_526_; lean_object* v_index_527_; 
lean_inc_ref(v___x_515_);
lean_inc_ref(v_fvarIdToDecl_505_);
v___x_526_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3___redArg(v_fvarIdToDecl_505_, v___y_525_, v___x_515_);
v_index_527_ = lean_ctor_get(v___x_515_, 0);
lean_inc(v_index_527_);
v___y_517_ = v___x_526_;
v___y_518_ = v_index_527_;
goto v___jp_516_;
}
}
}
v___jp_499_:
{
lean_object* v___x_501_; lean_object* v___x_502_; lean_object* v___f_503_; lean_object* v___x_504_; 
v___x_501_ = lean_box(0);
v___x_502_ = lean_box(v___x_486_);
v___f_503_ = lean_alloc_closure((void*)(lp_mathlib_elabLetImplDetail___lam__0___boxed), 13, 6);
lean_closure_set(v___f_503_, 0, v___x_484_);
lean_closure_set(v___f_503_, 1, v_expectedType_x3f_485_);
lean_closure_set(v___f_503_, 2, v___x_502_);
lean_closure_set(v___f_503_, 3, v___x_501_);
lean_closure_set(v___f_503_, 4, v___x_487_);
lean_closure_set(v___f_503_, 5, v_x_488_);
lean_inc_ref(v_localInstances_498_);
v___x_504_ = lp_mathlib_Lean_Meta_withLCtx___at___00elabLetImplDetail_spec__2___redArg(v___y_500_, v_localInstances_498_, v___f_503_, v___y_489_, v___y_490_, v___y_491_, v___y_492_, v___y_493_, v___y_494_);
return v___x_504_;
}
}
else
{
lean_object* v_a_530_; lean_object* v___x_532_; uint8_t v_isShared_533_; uint8_t v_isSharedCheck_537_; 
lean_dec_ref(v_x_488_);
lean_dec(v___x_487_);
lean_dec(v_expectedType_x3f_485_);
lean_dec(v___x_484_);
v_a_530_ = lean_ctor_get(v___x_496_, 0);
v_isSharedCheck_537_ = !lean_is_exclusive(v___x_496_);
if (v_isSharedCheck_537_ == 0)
{
v___x_532_ = v___x_496_;
v_isShared_533_ = v_isSharedCheck_537_;
goto v_resetjp_531_;
}
else
{
lean_inc(v_a_530_);
lean_dec(v___x_496_);
v___x_532_ = lean_box(0);
v_isShared_533_ = v_isSharedCheck_537_;
goto v_resetjp_531_;
}
v_resetjp_531_:
{
lean_object* v___x_535_; 
if (v_isShared_533_ == 0)
{
v___x_535_ = v___x_532_;
goto v_reusejp_534_;
}
else
{
lean_object* v_reuseFailAlloc_536_; 
v_reuseFailAlloc_536_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_536_, 0, v_a_530_);
v___x_535_ = v_reuseFailAlloc_536_;
goto v_reusejp_534_;
}
v_reusejp_534_:
{
return v___x_535_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_elabLetImplDetail___lam__1___boxed(lean_object* v_id_538_, lean_object* v___x_539_, lean_object* v_expectedType_x3f_540_, lean_object* v___x_541_, lean_object* v___x_542_, lean_object* v_x_543_, lean_object* v___y_544_, lean_object* v___y_545_, lean_object* v___y_546_, lean_object* v___y_547_, lean_object* v___y_548_, lean_object* v___y_549_, lean_object* v___y_550_){
_start:
{
uint8_t v___x_8521__boxed_551_; lean_object* v_res_552_; 
v___x_8521__boxed_551_ = lean_unbox(v___x_541_);
v_res_552_ = lp_mathlib_elabLetImplDetail___lam__1(v_id_538_, v___x_539_, v_expectedType_x3f_540_, v___x_8521__boxed_551_, v___x_542_, v_x_543_, v___y_544_, v___y_545_, v___y_546_, v___y_547_, v___y_548_, v___y_549_);
lean_dec(v___y_549_);
lean_dec_ref(v___y_548_);
lean_dec(v___y_547_);
lean_dec_ref(v___y_546_);
lean_dec(v___y_545_);
lean_dec_ref(v___y_544_);
return v_res_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00elabLetImplDetail_spec__5_spec__6(lean_object* v_msgData_553_, lean_object* v___y_554_, lean_object* v___y_555_, lean_object* v___y_556_, lean_object* v___y_557_){
_start:
{
lean_object* v___x_559_; lean_object* v_env_560_; lean_object* v___x_561_; lean_object* v_mctx_562_; lean_object* v_lctx_563_; lean_object* v_options_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___x_567_; 
v___x_559_ = lean_st_ref_get(v___y_557_);
v_env_560_ = lean_ctor_get(v___x_559_, 0);
lean_inc_ref(v_env_560_);
lean_dec(v___x_559_);
v___x_561_ = lean_st_ref_get(v___y_555_);
v_mctx_562_ = lean_ctor_get(v___x_561_, 0);
lean_inc_ref(v_mctx_562_);
lean_dec(v___x_561_);
v_lctx_563_ = lean_ctor_get(v___y_554_, 2);
v_options_564_ = lean_ctor_get(v___y_556_, 2);
lean_inc_ref(v_options_564_);
lean_inc_ref(v_lctx_563_);
v___x_565_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_565_, 0, v_env_560_);
lean_ctor_set(v___x_565_, 1, v_mctx_562_);
lean_ctor_set(v___x_565_, 2, v_lctx_563_);
lean_ctor_set(v___x_565_, 3, v_options_564_);
v___x_566_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_566_, 0, v___x_565_);
lean_ctor_set(v___x_566_, 1, v_msgData_553_);
v___x_567_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_567_, 0, v___x_566_);
return v___x_567_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00elabLetImplDetail_spec__5_spec__6___boxed(lean_object* v_msgData_568_, lean_object* v___y_569_, lean_object* v___y_570_, lean_object* v___y_571_, lean_object* v___y_572_, lean_object* v___y_573_){
_start:
{
lean_object* v_res_574_; 
v_res_574_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00elabLetImplDetail_spec__5_spec__6(v_msgData_568_, v___y_569_, v___y_570_, v___y_571_, v___y_572_);
lean_dec(v___y_572_);
lean_dec_ref(v___y_571_);
lean_dec(v___y_570_);
lean_dec_ref(v___y_569_);
return v_res_574_;
}
}
static double _init_lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg___closed__0(void){
_start:
{
lean_object* v___x_575_; double v___x_576_; 
v___x_575_ = lean_unsigned_to_nat(0u);
v___x_576_ = lean_float_of_nat(v___x_575_);
return v___x_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg(lean_object* v_cls_580_, lean_object* v_msg_581_, lean_object* v___y_582_, lean_object* v___y_583_, lean_object* v___y_584_, lean_object* v___y_585_){
_start:
{
lean_object* v_ref_587_; lean_object* v___x_588_; lean_object* v_a_589_; lean_object* v___x_591_; uint8_t v_isShared_592_; uint8_t v_isSharedCheck_633_; 
v_ref_587_ = lean_ctor_get(v___y_584_, 5);
v___x_588_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_addTrace___at___00elabLetImplDetail_spec__5_spec__6(v_msg_581_, v___y_582_, v___y_583_, v___y_584_, v___y_585_);
v_a_589_ = lean_ctor_get(v___x_588_, 0);
v_isSharedCheck_633_ = !lean_is_exclusive(v___x_588_);
if (v_isSharedCheck_633_ == 0)
{
v___x_591_ = v___x_588_;
v_isShared_592_ = v_isSharedCheck_633_;
goto v_resetjp_590_;
}
else
{
lean_inc(v_a_589_);
lean_dec(v___x_588_);
v___x_591_ = lean_box(0);
v_isShared_592_ = v_isSharedCheck_633_;
goto v_resetjp_590_;
}
v_resetjp_590_:
{
lean_object* v___x_593_; lean_object* v_traceState_594_; lean_object* v_env_595_; lean_object* v_nextMacroScope_596_; lean_object* v_ngen_597_; lean_object* v_auxDeclNGen_598_; lean_object* v_cache_599_; lean_object* v_messages_600_; lean_object* v_infoState_601_; lean_object* v_snapshotTasks_602_; lean_object* v___x_604_; uint8_t v_isShared_605_; uint8_t v_isSharedCheck_632_; 
v___x_593_ = lean_st_ref_take(v___y_585_);
v_traceState_594_ = lean_ctor_get(v___x_593_, 4);
v_env_595_ = lean_ctor_get(v___x_593_, 0);
v_nextMacroScope_596_ = lean_ctor_get(v___x_593_, 1);
v_ngen_597_ = lean_ctor_get(v___x_593_, 2);
v_auxDeclNGen_598_ = lean_ctor_get(v___x_593_, 3);
v_cache_599_ = lean_ctor_get(v___x_593_, 5);
v_messages_600_ = lean_ctor_get(v___x_593_, 6);
v_infoState_601_ = lean_ctor_get(v___x_593_, 7);
v_snapshotTasks_602_ = lean_ctor_get(v___x_593_, 8);
v_isSharedCheck_632_ = !lean_is_exclusive(v___x_593_);
if (v_isSharedCheck_632_ == 0)
{
v___x_604_ = v___x_593_;
v_isShared_605_ = v_isSharedCheck_632_;
goto v_resetjp_603_;
}
else
{
lean_inc(v_snapshotTasks_602_);
lean_inc(v_infoState_601_);
lean_inc(v_messages_600_);
lean_inc(v_cache_599_);
lean_inc(v_traceState_594_);
lean_inc(v_auxDeclNGen_598_);
lean_inc(v_ngen_597_);
lean_inc(v_nextMacroScope_596_);
lean_inc(v_env_595_);
lean_dec(v___x_593_);
v___x_604_ = lean_box(0);
v_isShared_605_ = v_isSharedCheck_632_;
goto v_resetjp_603_;
}
v_resetjp_603_:
{
uint64_t v_tid_606_; lean_object* v_traces_607_; lean_object* v___x_609_; uint8_t v_isShared_610_; uint8_t v_isSharedCheck_631_; 
v_tid_606_ = lean_ctor_get_uint64(v_traceState_594_, sizeof(void*)*1);
v_traces_607_ = lean_ctor_get(v_traceState_594_, 0);
v_isSharedCheck_631_ = !lean_is_exclusive(v_traceState_594_);
if (v_isSharedCheck_631_ == 0)
{
v___x_609_ = v_traceState_594_;
v_isShared_610_ = v_isSharedCheck_631_;
goto v_resetjp_608_;
}
else
{
lean_inc(v_traces_607_);
lean_dec(v_traceState_594_);
v___x_609_ = lean_box(0);
v_isShared_610_ = v_isSharedCheck_631_;
goto v_resetjp_608_;
}
v_resetjp_608_:
{
lean_object* v___x_611_; double v___x_612_; uint8_t v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_621_; 
v___x_611_ = lean_box(0);
v___x_612_ = lean_float_once(&lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg___closed__0, &lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg___closed__0_once, _init_lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg___closed__0);
v___x_613_ = 0;
v___x_614_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg___closed__1));
v___x_615_ = lean_alloc_ctor(0, 3, 17);
lean_ctor_set(v___x_615_, 0, v_cls_580_);
lean_ctor_set(v___x_615_, 1, v___x_611_);
lean_ctor_set(v___x_615_, 2, v___x_614_);
lean_ctor_set_float(v___x_615_, sizeof(void*)*3, v___x_612_);
lean_ctor_set_float(v___x_615_, sizeof(void*)*3 + 8, v___x_612_);
lean_ctor_set_uint8(v___x_615_, sizeof(void*)*3 + 16, v___x_613_);
v___x_616_ = ((lean_object*)(lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg___closed__2));
v___x_617_ = lean_alloc_ctor(9, 3, 0);
lean_ctor_set(v___x_617_, 0, v___x_615_);
lean_ctor_set(v___x_617_, 1, v_a_589_);
lean_ctor_set(v___x_617_, 2, v___x_616_);
lean_inc(v_ref_587_);
v___x_618_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_618_, 0, v_ref_587_);
lean_ctor_set(v___x_618_, 1, v___x_617_);
v___x_619_ = l_Lean_PersistentArray_push___redArg(v_traces_607_, v___x_618_);
if (v_isShared_610_ == 0)
{
lean_ctor_set(v___x_609_, 0, v___x_619_);
v___x_621_ = v___x_609_;
goto v_reusejp_620_;
}
else
{
lean_object* v_reuseFailAlloc_630_; 
v_reuseFailAlloc_630_ = lean_alloc_ctor(0, 1, 8);
lean_ctor_set(v_reuseFailAlloc_630_, 0, v___x_619_);
lean_ctor_set_uint64(v_reuseFailAlloc_630_, sizeof(void*)*1, v_tid_606_);
v___x_621_ = v_reuseFailAlloc_630_;
goto v_reusejp_620_;
}
v_reusejp_620_:
{
lean_object* v___x_623_; 
if (v_isShared_605_ == 0)
{
lean_ctor_set(v___x_604_, 4, v___x_621_);
v___x_623_ = v___x_604_;
goto v_reusejp_622_;
}
else
{
lean_object* v_reuseFailAlloc_629_; 
v_reuseFailAlloc_629_ = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(v_reuseFailAlloc_629_, 0, v_env_595_);
lean_ctor_set(v_reuseFailAlloc_629_, 1, v_nextMacroScope_596_);
lean_ctor_set(v_reuseFailAlloc_629_, 2, v_ngen_597_);
lean_ctor_set(v_reuseFailAlloc_629_, 3, v_auxDeclNGen_598_);
lean_ctor_set(v_reuseFailAlloc_629_, 4, v___x_621_);
lean_ctor_set(v_reuseFailAlloc_629_, 5, v_cache_599_);
lean_ctor_set(v_reuseFailAlloc_629_, 6, v_messages_600_);
lean_ctor_set(v_reuseFailAlloc_629_, 7, v_infoState_601_);
lean_ctor_set(v_reuseFailAlloc_629_, 8, v_snapshotTasks_602_);
v___x_623_ = v_reuseFailAlloc_629_;
goto v_reusejp_622_;
}
v_reusejp_622_:
{
lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_627_; 
v___x_624_ = lean_st_ref_set(v___y_585_, v___x_623_);
v___x_625_ = lean_box(0);
if (v_isShared_592_ == 0)
{
lean_ctor_set(v___x_591_, 0, v___x_625_);
v___x_627_ = v___x_591_;
goto v_reusejp_626_;
}
else
{
lean_object* v_reuseFailAlloc_628_; 
v_reuseFailAlloc_628_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_628_, 0, v___x_625_);
v___x_627_ = v_reuseFailAlloc_628_;
goto v_reusejp_626_;
}
v_reusejp_626_:
{
return v___x_627_;
}
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg___boxed(lean_object* v_cls_634_, lean_object* v_msg_635_, lean_object* v___y_636_, lean_object* v___y_637_, lean_object* v___y_638_, lean_object* v___y_639_, lean_object* v___y_640_){
_start:
{
lean_object* v_res_641_; 
v_res_641_ = lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg(v_cls_634_, v_msg_635_, v___y_636_, v___y_637_, v___y_638_, v___y_639_);
lean_dec(v___y_639_);
lean_dec_ref(v___y_638_);
lean_dec(v___y_637_);
lean_dec_ref(v___y_636_);
return v_res_641_;
}
}
static lean_object* _init_lp_mathlib_elabLetImplDetail___closed__6(void){
_start:
{
lean_object* v___x_652_; lean_object* v___x_653_; lean_object* v___x_654_; 
v___x_652_ = ((lean_object*)(lp_mathlib_elabLetImplDetail___closed__3));
v___x_653_ = ((lean_object*)(lp_mathlib_elabLetImplDetail___closed__5));
v___x_654_ = l_Lean_Name_append(v___x_653_, v___x_652_);
return v___x_654_;
}
}
static lean_object* _init_lp_mathlib_elabLetImplDetail___closed__8(void){
_start:
{
lean_object* v___x_656_; lean_object* v___x_657_; 
v___x_656_ = ((lean_object*)(lp_mathlib_elabLetImplDetail___closed__7));
v___x_657_ = l_Lean_stringToMessageData(v___x_656_);
return v___x_657_;
}
}
static lean_object* _init_lp_mathlib_elabLetImplDetail___closed__9(void){
_start:
{
lean_object* v___x_658_; lean_object* v___x_659_; 
v___x_658_ = ((lean_object*)(lp_mathlib_letImplDetailStx___closed__13));
v___x_659_ = l_Lean_stringToMessageData(v___x_658_);
return v___x_659_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_elabLetImplDetail(lean_object* v_stx_660_, lean_object* v_expectedType_x3f_661_, lean_object* v_a_662_, lean_object* v_a_663_, lean_object* v_a_664_, lean_object* v_a_665_, lean_object* v_a_666_, lean_object* v_a_667_){
_start:
{
lean_object* v___x_669_; uint8_t v___x_670_; 
v___x_669_ = ((lean_object*)(lp_mathlib_letImplDetailStx___closed__1));
lean_inc(v_stx_660_);
v___x_670_ = l_Lean_Syntax_isOfKind(v_stx_660_, v___x_669_);
if (v___x_670_ == 0)
{
lean_object* v___x_671_; 
lean_dec(v_expectedType_x3f_661_);
lean_dec(v_stx_660_);
v___x_671_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00elabLetImplDetail_spec__0___redArg();
return v___x_671_;
}
else
{
lean_object* v___x_672_; lean_object* v___x_673_; lean_object* v___x_674_; lean_object* v___x_675_; 
v___x_672_ = lean_unsigned_to_nat(3u);
v___x_673_ = l_Lean_Syntax_getArg(v_stx_660_, v___x_672_);
v___x_674_ = lean_box(0);
v___x_675_ = l_Lean_Elab_Term_elabTerm(v___x_673_, v___x_674_, v___x_670_, v___x_670_, v_a_662_, v_a_663_, v_a_664_, v_a_665_, v_a_666_, v_a_667_);
if (lean_obj_tag(v___x_675_) == 0)
{
lean_object* v_a_676_; lean_object* v___x_677_; 
v_a_676_ = lean_ctor_get(v___x_675_, 0);
lean_inc_n(v_a_676_, 2);
lean_dec_ref_known(v___x_675_, 1);
lean_inc(v_a_667_);
lean_inc_ref(v_a_666_);
lean_inc(v_a_665_);
lean_inc_ref(v_a_664_);
v___x_677_ = lean_infer_type(v_a_676_, v_a_664_, v_a_665_, v_a_666_, v_a_667_);
if (lean_obj_tag(v___x_677_) == 0)
{
lean_object* v_options_678_; lean_object* v_a_679_; lean_object* v_inheritedTraceOptions_680_; uint8_t v_hasTrace_681_; lean_object* v___x_682_; lean_object* v_id_683_; lean_object* v___x_684_; lean_object* v___x_685_; lean_object* v___x_686_; lean_object* v___f_687_; lean_object* v___y_689_; lean_object* v___y_690_; lean_object* v___y_691_; lean_object* v___y_692_; lean_object* v___y_693_; lean_object* v___y_694_; 
v_options_678_ = lean_ctor_get(v_a_666_, 2);
v_a_679_ = lean_ctor_get(v___x_677_, 0);
lean_inc(v_a_679_);
lean_dec_ref_known(v___x_677_, 1);
v_inheritedTraceOptions_680_ = lean_ctor_get(v_a_666_, 13);
v_hasTrace_681_ = lean_ctor_get_uint8(v_options_678_, sizeof(void*)*1);
v___x_682_ = lean_unsigned_to_nat(1u);
v_id_683_ = l_Lean_Syntax_getArg(v_stx_660_, v___x_682_);
v___x_684_ = lean_unsigned_to_nat(5u);
v___x_685_ = l_Lean_Syntax_getArg(v_stx_660_, v___x_684_);
lean_dec(v_stx_660_);
v___x_686_ = lean_box(v___x_670_);
lean_inc(v_id_683_);
v___f_687_ = lean_alloc_closure((void*)(lp_mathlib_elabLetImplDetail___lam__1___boxed), 13, 5);
lean_closure_set(v___f_687_, 0, v_id_683_);
lean_closure_set(v___f_687_, 1, v___x_685_);
lean_closure_set(v___f_687_, 2, v_expectedType_x3f_661_);
lean_closure_set(v___f_687_, 3, v___x_686_);
lean_closure_set(v___f_687_, 4, v___x_682_);
if (v_hasTrace_681_ == 0)
{
v___y_689_ = v_a_662_;
v___y_690_ = v_a_663_;
v___y_691_ = v_a_664_;
v___y_692_ = v_a_665_;
v___y_693_ = v_a_666_;
v___y_694_ = v_a_667_;
goto v___jp_688_;
}
else
{
lean_object* v___x_699_; lean_object* v___x_700_; uint8_t v___x_701_; 
v___x_699_ = ((lean_object*)(lp_mathlib_elabLetImplDetail___closed__3));
v___x_700_ = lean_obj_once(&lp_mathlib_elabLetImplDetail___closed__6, &lp_mathlib_elabLetImplDetail___closed__6_once, _init_lp_mathlib_elabLetImplDetail___closed__6);
v___x_701_ = l___private_Lean_Util_Trace_0__Lean_checkTraceOption_go(v_inheritedTraceOptions_680_, v_options_678_, v___x_700_);
if (v___x_701_ == 0)
{
v___y_689_ = v_a_662_;
v___y_690_ = v_a_663_;
v___y_691_ = v_a_664_;
v___y_692_ = v_a_665_;
v___y_693_ = v_a_666_;
v___y_694_ = v_a_667_;
goto v___jp_688_;
}
else
{
lean_object* v___x_702_; lean_object* v___x_703_; lean_object* v___x_704_; lean_object* v___x_705_; lean_object* v___x_706_; lean_object* v___x_707_; lean_object* v___x_708_; lean_object* v___x_709_; lean_object* v___x_710_; lean_object* v___x_711_; lean_object* v___x_712_; 
v___x_702_ = l_Lean_TSyntax_getId(v_id_683_);
v___x_703_ = l_Lean_MessageData_ofName(v___x_702_);
v___x_704_ = lean_obj_once(&lp_mathlib_elabLetImplDetail___closed__8, &lp_mathlib_elabLetImplDetail___closed__8_once, _init_lp_mathlib_elabLetImplDetail___closed__8);
v___x_705_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_705_, 0, v___x_703_);
lean_ctor_set(v___x_705_, 1, v___x_704_);
lean_inc(v_a_679_);
v___x_706_ = l_Lean_MessageData_ofExpr(v_a_679_);
v___x_707_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_707_, 0, v___x_705_);
lean_ctor_set(v___x_707_, 1, v___x_706_);
v___x_708_ = lean_obj_once(&lp_mathlib_elabLetImplDetail___closed__9, &lp_mathlib_elabLetImplDetail___closed__9_once, _init_lp_mathlib_elabLetImplDetail___closed__9);
v___x_709_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_709_, 0, v___x_707_);
lean_ctor_set(v___x_709_, 1, v___x_708_);
lean_inc(v_a_676_);
v___x_710_ = l_Lean_MessageData_ofExpr(v_a_676_);
v___x_711_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_711_, 0, v___x_709_);
lean_ctor_set(v___x_711_, 1, v___x_710_);
v___x_712_ = lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg(v___x_699_, v___x_711_, v_a_664_, v_a_665_, v_a_666_, v_a_667_);
if (lean_obj_tag(v___x_712_) == 0)
{
lean_dec_ref_known(v___x_712_, 1);
v___y_689_ = v_a_662_;
v___y_690_ = v_a_663_;
v___y_691_ = v_a_664_;
v___y_692_ = v_a_665_;
v___y_693_ = v_a_666_;
v___y_694_ = v_a_667_;
goto v___jp_688_;
}
else
{
lean_object* v_a_713_; lean_object* v___x_715_; uint8_t v_isShared_716_; uint8_t v_isSharedCheck_720_; 
lean_dec_ref(v___f_687_);
lean_dec(v_id_683_);
lean_dec(v_a_679_);
lean_dec(v_a_676_);
v_a_713_ = lean_ctor_get(v___x_712_, 0);
v_isSharedCheck_720_ = !lean_is_exclusive(v___x_712_);
if (v_isSharedCheck_720_ == 0)
{
v___x_715_ = v___x_712_;
v_isShared_716_ = v_isSharedCheck_720_;
goto v_resetjp_714_;
}
else
{
lean_inc(v_a_713_);
lean_dec(v___x_712_);
v___x_715_ = lean_box(0);
v_isShared_716_ = v_isSharedCheck_720_;
goto v_resetjp_714_;
}
v_resetjp_714_:
{
lean_object* v___x_718_; 
if (v_isShared_716_ == 0)
{
v___x_718_ = v___x_715_;
goto v_reusejp_717_;
}
else
{
lean_object* v_reuseFailAlloc_719_; 
v_reuseFailAlloc_719_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_719_, 0, v_a_713_);
v___x_718_ = v_reuseFailAlloc_719_;
goto v_reusejp_717_;
}
v_reusejp_717_:
{
return v___x_718_;
}
}
}
}
}
v___jp_688_:
{
lean_object* v___x_695_; uint8_t v___x_696_; uint8_t v___x_697_; lean_object* v___x_698_; 
v___x_695_ = l_Lean_TSyntax_getId(v_id_683_);
lean_dec(v_id_683_);
v___x_696_ = 0;
v___x_697_ = 0;
v___x_698_ = lp_mathlib_Lean_Meta_withLetDecl___at___00elabLetImplDetail_spec__4___redArg(v___x_695_, v_a_679_, v_a_676_, v___f_687_, v___x_696_, v___x_697_, v___y_689_, v___y_690_, v___y_691_, v___y_692_, v___y_693_, v___y_694_);
return v___x_698_;
}
}
else
{
lean_dec(v_a_676_);
lean_dec(v_expectedType_x3f_661_);
lean_dec(v_stx_660_);
return v___x_677_;
}
}
else
{
lean_dec(v_expectedType_x3f_661_);
lean_dec(v_stx_660_);
return v___x_675_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_elabLetImplDetail___boxed(lean_object* v_stx_721_, lean_object* v_expectedType_x3f_722_, lean_object* v_a_723_, lean_object* v_a_724_, lean_object* v_a_725_, lean_object* v_a_726_, lean_object* v_a_727_, lean_object* v_a_728_, lean_object* v_a_729_){
_start:
{
lean_object* v_res_730_; 
v_res_730_ = lp_mathlib_elabLetImplDetail(v_stx_721_, v_expectedType_x3f_722_, v_a_723_, v_a_724_, v_a_725_, v_a_726_, v_a_727_, v_a_728_);
lean_dec(v_a_728_);
lean_dec_ref(v_a_727_);
lean_dec(v_a_726_);
lean_dec_ref(v_a_725_);
lean_dec(v_a_724_);
lean_dec_ref(v_a_723_);
return v_res_730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3(lean_object* v_00_u03b2_731_, lean_object* v_x_732_, lean_object* v_x_733_, lean_object* v_x_734_){
_start:
{
lean_object* v___x_735_; 
v___x_735_ = lp_mathlib_Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3___redArg(v_x_732_, v_x_733_, v_x_734_);
return v___x_735_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5(lean_object* v_cls_736_, lean_object* v_msg_737_, lean_object* v___y_738_, lean_object* v___y_739_, lean_object* v___y_740_, lean_object* v___y_741_, lean_object* v___y_742_, lean_object* v___y_743_){
_start:
{
lean_object* v___x_745_; 
v___x_745_ = lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___redArg(v_cls_736_, v_msg_737_, v___y_740_, v___y_741_, v___y_742_, v___y_743_);
return v___x_745_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5___boxed(lean_object* v_cls_746_, lean_object* v_msg_747_, lean_object* v___y_748_, lean_object* v___y_749_, lean_object* v___y_750_, lean_object* v___y_751_, lean_object* v___y_752_, lean_object* v___y_753_, lean_object* v___y_754_){
_start:
{
lean_object* v_res_755_; 
v_res_755_ = lp_mathlib_Lean_addTrace___at___00elabLetImplDetail_spec__5(v_cls_746_, v_msg_747_, v___y_748_, v___y_749_, v___y_750_, v___y_751_, v___y_752_, v___y_753_);
lean_dec(v___y_753_);
lean_dec_ref(v___y_752_);
lean_dec(v___y_751_);
lean_dec_ref(v___y_750_);
lean_dec(v___y_749_);
lean_dec_ref(v___y_748_);
return v_res_755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3(lean_object* v_00_u03b2_756_, lean_object* v_x_757_, size_t v_x_758_, size_t v_x_759_, lean_object* v_x_760_, lean_object* v_x_761_){
_start:
{
lean_object* v___x_762_; 
v___x_762_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3___redArg(v_x_757_, v_x_758_, v_x_759_, v_x_760_, v_x_761_);
return v___x_762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3___boxed(lean_object* v_00_u03b2_763_, lean_object* v_x_764_, lean_object* v_x_765_, lean_object* v_x_766_, lean_object* v_x_767_, lean_object* v_x_768_){
_start:
{
size_t v_x_8972__boxed_769_; size_t v_x_8973__boxed_770_; lean_object* v_res_771_; 
v_x_8972__boxed_769_ = lean_unbox_usize(v_x_765_);
lean_dec(v_x_765_);
v_x_8973__boxed_770_ = lean_unbox_usize(v_x_766_);
lean_dec(v_x_766_);
v_res_771_ = lp_mathlib_Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3(v_00_u03b2_763_, v_x_764_, v_x_8972__boxed_769_, v_x_8973__boxed_770_, v_x_767_, v_x_768_);
return v_res_771_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__5(lean_object* v_00_u03b2_772_, lean_object* v_n_773_, lean_object* v_k_774_, lean_object* v_v_775_){
_start:
{
lean_object* v___x_776_; 
v___x_776_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__5___redArg(v_n_773_, v_k_774_, v_v_775_);
return v___x_776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__6(lean_object* v_00_u03b2_777_, size_t v_depth_778_, lean_object* v_keys_779_, lean_object* v_vals_780_, lean_object* v_heq_781_, lean_object* v_i_782_, lean_object* v_entries_783_){
_start:
{
lean_object* v___x_784_; 
v___x_784_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__6___redArg(v_depth_778_, v_keys_779_, v_vals_780_, v_i_782_, v_entries_783_);
return v___x_784_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__6___boxed(lean_object* v_00_u03b2_785_, lean_object* v_depth_786_, lean_object* v_keys_787_, lean_object* v_vals_788_, lean_object* v_heq_789_, lean_object* v_i_790_, lean_object* v_entries_791_){
_start:
{
size_t v_depth_boxed_792_; lean_object* v_res_793_; 
v_depth_boxed_792_ = lean_unbox_usize(v_depth_786_);
lean_dec(v_depth_786_);
v_res_793_ = lp_mathlib___private_Lean_Data_PersistentHashMap_0__Lean_PersistentHashMap_insertAux_traverse___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__6(v_00_u03b2_785_, v_depth_boxed_792_, v_keys_787_, v_vals_788_, v_heq_789_, v_i_790_, v_entries_791_);
lean_dec_ref(v_vals_788_);
lean_dec_ref(v_keys_787_);
return v_res_793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__5_spec__8(lean_object* v_00_u03b2_794_, lean_object* v_x_795_, lean_object* v_x_796_, lean_object* v_x_797_, lean_object* v_x_798_){
_start:
{
lean_object* v___x_799_; 
v___x_799_ = lp_mathlib_Lean_PersistentHashMap_insertAtCollisionNodeAux___at___00Lean_PersistentHashMap_insertAtCollisionNode___at___00Lean_PersistentHashMap_insertAux___at___00Lean_PersistentHashMap_insert___at___00elabLetImplDetail_spec__3_spec__3_spec__5_spec__8___redArg(v_x_795_, v_x_796_, v_x_797_, v_x_798_);
return v___x_799_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__3(size_t v_sz_800_, size_t v_i_801_, lean_object* v_bs_802_){
_start:
{
uint8_t v___x_803_; 
v___x_803_ = lean_usize_dec_lt(v_i_801_, v_sz_800_);
if (v___x_803_ == 0)
{
return v_bs_802_;
}
else
{
lean_object* v_v_804_; lean_object* v_fst_805_; lean_object* v___x_806_; lean_object* v_bs_x27_807_; size_t v___x_808_; size_t v___x_809_; lean_object* v___x_810_; 
v_v_804_ = lean_array_uget_borrowed(v_bs_802_, v_i_801_);
v_fst_805_ = lean_ctor_get(v_v_804_, 0);
lean_inc(v_fst_805_);
v___x_806_ = lean_unsigned_to_nat(0u);
v_bs_x27_807_ = lean_array_uset(v_bs_802_, v_i_801_, v___x_806_);
v___x_808_ = ((size_t)1ULL);
v___x_809_ = lean_usize_add(v_i_801_, v___x_808_);
v___x_810_ = lean_array_uset(v_bs_x27_807_, v_i_801_, v_fst_805_);
v_i_801_ = v___x_809_;
v_bs_802_ = v___x_810_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__3___boxed(lean_object* v_sz_812_, lean_object* v_i_813_, lean_object* v_bs_814_){
_start:
{
size_t v_sz_boxed_815_; size_t v_i_boxed_816_; lean_object* v_res_817_; 
v_sz_boxed_815_ = lean_unbox_usize(v_sz_812_);
lean_dec(v_sz_812_);
v_i_boxed_816_ = lean_unbox_usize(v_i_813_);
lean_dec(v_i_813_);
v_res_817_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__3(v_sz_boxed_815_, v_i_boxed_816_, v_bs_814_);
return v_res_817_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2___redArg(size_t v_sz_821_, size_t v_i_822_, lean_object* v_bs_823_, lean_object* v___y_824_, lean_object* v___y_825_){
_start:
{
uint8_t v___x_826_; 
v___x_826_ = lean_usize_dec_lt(v_i_822_, v_sz_821_);
if (v___x_826_ == 0)
{
lean_object* v___x_827_; 
v___x_827_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_827_, 0, v_bs_823_);
lean_ctor_set(v___x_827_, 1, v___y_825_);
return v___x_827_;
}
else
{
lean_object* v_v_828_; lean_object* v___x_829_; lean_object* v_bs_x27_830_; lean_object* v___x_831_; lean_object* v___x_832_; lean_object* v___x_833_; lean_object* v___x_834_; 
v_v_828_ = lean_array_uget(v_bs_823_, v_i_822_);
v___x_829_ = lean_unsigned_to_nat(0u);
v_bs_x27_830_ = lean_array_uset(v_bs_823_, v_i_822_, v___x_829_);
v___x_831_ = lean_usize_to_nat(v_i_822_);
v___x_832_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2___redArg___closed__1));
v___x_833_ = l_Lean_Name_num___override(v___x_832_, v___x_831_);
v___x_834_ = l_Lean_Macro_addMacroScope(v___x_833_, v___y_824_, v___y_825_);
if (lean_obj_tag(v___x_834_) == 0)
{
lean_object* v_a_835_; lean_object* v_a_836_; lean_object* v___x_837_; lean_object* v___x_838_; size_t v___x_839_; size_t v___x_840_; lean_object* v___x_841_; 
v_a_835_ = lean_ctor_get(v___x_834_, 0);
lean_inc(v_a_835_);
v_a_836_ = lean_ctor_get(v___x_834_, 1);
lean_inc(v_a_836_);
lean_dec_ref_known(v___x_834_, 2);
v___x_837_ = l_Lean_mkIdent(v_a_835_);
v___x_838_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_838_, 0, v___x_837_);
lean_ctor_set(v___x_838_, 1, v_v_828_);
v___x_839_ = ((size_t)1ULL);
v___x_840_ = lean_usize_add(v_i_822_, v___x_839_);
v___x_841_ = lean_array_uset(v_bs_x27_830_, v_i_822_, v___x_838_);
v_i_822_ = v___x_840_;
v_bs_823_ = v___x_841_;
v___y_825_ = v_a_836_;
goto _start;
}
else
{
lean_object* v_a_843_; lean_object* v_a_844_; lean_object* v___x_846_; uint8_t v_isShared_847_; uint8_t v_isSharedCheck_851_; 
lean_dec_ref(v_bs_x27_830_);
lean_dec(v_v_828_);
v_a_843_ = lean_ctor_get(v___x_834_, 0);
v_a_844_ = lean_ctor_get(v___x_834_, 1);
v_isSharedCheck_851_ = !lean_is_exclusive(v___x_834_);
if (v_isSharedCheck_851_ == 0)
{
v___x_846_ = v___x_834_;
v_isShared_847_ = v_isSharedCheck_851_;
goto v_resetjp_845_;
}
else
{
lean_inc(v_a_844_);
lean_inc(v_a_843_);
lean_dec(v___x_834_);
v___x_846_ = lean_box(0);
v_isShared_847_ = v_isSharedCheck_851_;
goto v_resetjp_845_;
}
v_resetjp_845_:
{
lean_object* v___x_849_; 
if (v_isShared_847_ == 0)
{
v___x_849_ = v___x_846_;
goto v_reusejp_848_;
}
else
{
lean_object* v_reuseFailAlloc_850_; 
v_reuseFailAlloc_850_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_850_, 0, v_a_843_);
lean_ctor_set(v_reuseFailAlloc_850_, 1, v_a_844_);
v___x_849_ = v_reuseFailAlloc_850_;
goto v_reusejp_848_;
}
v_reusejp_848_:
{
return v___x_849_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2___redArg___boxed(lean_object* v_sz_852_, lean_object* v_i_853_, lean_object* v_bs_854_, lean_object* v___y_855_, lean_object* v___y_856_){
_start:
{
size_t v_sz_boxed_857_; size_t v_i_boxed_858_; lean_object* v_res_859_; 
v_sz_boxed_857_ = lean_unbox_usize(v_sz_852_);
lean_dec(v_sz_852_);
v_i_boxed_858_ = lean_unbox_usize(v_i_853_);
lean_dec(v_i_853_);
v_res_859_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2___redArg(v_sz_boxed_857_, v_i_boxed_858_, v_bs_854_, v___y_855_, v___y_856_);
lean_dec_ref(v___y_855_);
return v_res_859_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__5(size_t v_sz_860_, size_t v_i_861_, lean_object* v_bs_862_){
_start:
{
uint8_t v___x_863_; 
v___x_863_ = lean_usize_dec_lt(v_i_861_, v_sz_860_);
if (v___x_863_ == 0)
{
return v_bs_862_;
}
else
{
lean_object* v_v_864_; lean_object* v___x_865_; lean_object* v_bs_x27_866_; size_t v___x_867_; size_t v___x_868_; lean_object* v___x_869_; 
v_v_864_ = lean_array_uget(v_bs_862_, v_i_861_);
v___x_865_ = lean_unsigned_to_nat(0u);
v_bs_x27_866_ = lean_array_uset(v_bs_862_, v_i_861_, v___x_865_);
v___x_867_ = ((size_t)1ULL);
v___x_868_ = lean_usize_add(v_i_861_, v___x_867_);
v___x_869_ = lean_array_uset(v_bs_x27_866_, v_i_861_, v_v_864_);
v_i_861_ = v___x_868_;
v_bs_862_ = v___x_869_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__5___boxed(lean_object* v_sz_871_, lean_object* v_i_872_, lean_object* v_bs_873_){
_start:
{
size_t v_sz_boxed_874_; size_t v_i_boxed_875_; lean_object* v_res_876_; 
v_sz_boxed_874_ = lean_unbox_usize(v_sz_871_);
lean_dec(v_sz_871_);
v_i_boxed_875_ = lean_unbox_usize(v_i_872_);
lean_dec(v_i_872_);
v_res_876_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__5(v_sz_boxed_874_, v_i_boxed_875_, v_bs_873_);
return v_res_876_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__4(size_t v_sz_877_, size_t v_i_878_, lean_object* v_bs_879_){
_start:
{
uint8_t v___x_880_; 
v___x_880_ = lean_usize_dec_lt(v_i_878_, v_sz_877_);
if (v___x_880_ == 0)
{
return v_bs_879_;
}
else
{
lean_object* v_v_881_; lean_object* v___x_882_; lean_object* v_bs_x27_883_; size_t v___x_884_; size_t v___x_885_; lean_object* v___x_886_; 
v_v_881_ = lean_array_uget(v_bs_879_, v_i_878_);
v___x_882_ = lean_unsigned_to_nat(0u);
v_bs_x27_883_ = lean_array_uset(v_bs_879_, v_i_878_, v___x_882_);
v___x_884_ = ((size_t)1ULL);
v___x_885_ = lean_usize_add(v_i_878_, v___x_884_);
v___x_886_ = lean_array_uset(v_bs_x27_883_, v_i_878_, v_v_881_);
v_i_878_ = v___x_885_;
v_bs_879_ = v___x_886_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__4___boxed(lean_object* v_sz_888_, lean_object* v_i_889_, lean_object* v_bs_890_){
_start:
{
size_t v_sz_boxed_891_; size_t v_i_boxed_892_; lean_object* v_res_893_; 
v_sz_boxed_891_ = lean_unbox_usize(v_sz_888_);
lean_dec(v_sz_888_);
v_i_boxed_892_ = lean_unbox_usize(v_i_889_);
lean_dec(v_i_889_);
v_res_893_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__4(v_sz_boxed_891_, v_i_boxed_892_, v_bs_890_);
return v_res_893_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6(lean_object* v_as_897_, size_t v_i_898_, size_t v_stop_899_, lean_object* v_b_900_, lean_object* v___y_901_, lean_object* v___y_902_){
_start:
{
uint8_t v___x_903_; 
v___x_903_ = lean_usize_dec_eq(v_i_898_, v_stop_899_);
if (v___x_903_ == 0)
{
size_t v___x_904_; size_t v___x_905_; lean_object* v___x_906_; lean_object* v_fst_907_; lean_object* v_snd_908_; lean_object* v___x_910_; uint8_t v_isShared_911_; uint8_t v_isSharedCheck_925_; 
v___x_904_ = ((size_t)1ULL);
v___x_905_ = lean_usize_sub(v_i_898_, v___x_904_);
v___x_906_ = lean_array_uget(v_as_897_, v___x_905_);
v_fst_907_ = lean_ctor_get(v___x_906_, 0);
v_snd_908_ = lean_ctor_get(v___x_906_, 1);
v_isSharedCheck_925_ = !lean_is_exclusive(v___x_906_);
if (v_isSharedCheck_925_ == 0)
{
v___x_910_ = v___x_906_;
v_isShared_911_ = v_isSharedCheck_925_;
goto v_resetjp_909_;
}
else
{
lean_inc(v_snd_908_);
lean_inc(v_fst_907_);
lean_dec(v___x_906_);
v___x_910_ = lean_box(0);
v_isShared_911_ = v_isSharedCheck_925_;
goto v_resetjp_909_;
}
v_resetjp_909_:
{
lean_object* v_ref_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___x_917_; 
v_ref_912_ = lean_ctor_get(v___y_901_, 5);
v___x_913_ = l_Lean_SourceInfo_fromRef(v_ref_912_, v___x_903_);
v___x_914_ = ((lean_object*)(lp_mathlib_letImplDetailStx___closed__1));
v___x_915_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6___closed__0));
lean_inc(v___x_913_);
if (v_isShared_911_ == 0)
{
lean_ctor_set_tag(v___x_910_, 2);
lean_ctor_set(v___x_910_, 1, v___x_915_);
lean_ctor_set(v___x_910_, 0, v___x_913_);
v___x_917_ = v___x_910_;
goto v_reusejp_916_;
}
else
{
lean_object* v_reuseFailAlloc_924_; 
v_reuseFailAlloc_924_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v_reuseFailAlloc_924_, 0, v___x_913_);
lean_ctor_set(v_reuseFailAlloc_924_, 1, v___x_915_);
v___x_917_ = v_reuseFailAlloc_924_;
goto v_reusejp_916_;
}
v_reusejp_916_:
{
lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___x_920_; lean_object* v___x_921_; lean_object* v___x_922_; 
v___x_918_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6___closed__1));
lean_inc_n(v___x_913_, 2);
v___x_919_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_919_, 0, v___x_913_);
lean_ctor_set(v___x_919_, 1, v___x_918_);
v___x_920_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6___closed__2));
v___x_921_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_921_, 0, v___x_913_);
lean_ctor_set(v___x_921_, 1, v___x_920_);
v___x_922_ = l_Lean_Syntax_node6(v___x_913_, v___x_914_, v___x_917_, v_fst_907_, v___x_919_, v_snd_908_, v___x_921_, v_b_900_);
v_i_898_ = v___x_905_;
v_b_900_ = v___x_922_;
goto _start;
}
}
}
else
{
lean_object* v___x_926_; 
v___x_926_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_926_, 0, v_b_900_);
lean_ctor_set(v___x_926_, 1, v___y_902_);
return v___x_926_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6___boxed(lean_object* v_as_927_, lean_object* v_i_928_, lean_object* v_stop_929_, lean_object* v_b_930_, lean_object* v___y_931_, lean_object* v___y_932_){
_start:
{
size_t v_i_boxed_933_; size_t v_stop_boxed_934_; lean_object* v_res_935_; 
v_i_boxed_933_ = lean_unbox_usize(v_i_928_);
lean_dec(v_i_928_);
v_stop_boxed_934_ = lean_unbox_usize(v_stop_929_);
lean_dec(v_stop_929_);
v_res_935_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6(v_as_927_, v_i_boxed_933_, v_stop_boxed_934_, v_b_930_, v___y_931_, v___y_932_);
lean_dec_ref(v___y_931_);
lean_dec_ref(v_as_927_);
return v_res_935_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__4(void){
_start:
{
lean_object* v___x_941_; 
v___x_941_ = l_Array_mkArray0(lean_box(0));
return v___x_941_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__7(void){
_start:
{
lean_object* v___x_944_; lean_object* v___x_945_; 
v___x_944_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__5));
v___x_945_ = l_Lean_mkAtom(v___x_944_);
return v___x_945_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0(lean_object* v_fst_947_, lean_object* v___x_948_, size_t v___x_949_, lean_object* v___x_950_, lean_object* v___x_951_, lean_object* v_snd_952_, lean_object* v___x_953_, lean_object* v___x_954_, lean_object* v_ty_x3f_955_, lean_object* v_srcs_956_, lean_object* v_____r_957_, lean_object* v___y_958_, lean_object* v___y_959_){
_start:
{
lean_object* v_macroScope_960_; lean_object* v_traceMsgs_961_; lean_object* v_expandedMacroDecls_962_; lean_object* v___x_964_; uint8_t v_isShared_965_; uint8_t v_isSharedCheck_1051_; 
v_macroScope_960_ = lean_ctor_get(v___y_959_, 0);
v_traceMsgs_961_ = lean_ctor_get(v___y_959_, 1);
v_expandedMacroDecls_962_ = lean_ctor_get(v___y_959_, 2);
v_isSharedCheck_1051_ = !lean_is_exclusive(v___y_959_);
if (v_isSharedCheck_1051_ == 0)
{
v___x_964_ = v___y_959_;
v_isShared_965_ = v_isSharedCheck_1051_;
goto v_resetjp_963_;
}
else
{
lean_inc(v_expandedMacroDecls_962_);
lean_inc(v_traceMsgs_961_);
lean_inc(v_macroScope_960_);
lean_dec(v___y_959_);
v___x_964_ = lean_box(0);
v_isShared_965_ = v_isSharedCheck_1051_;
goto v_resetjp_963_;
}
v_resetjp_963_:
{
lean_object* v_methods_966_; lean_object* v_quotContext_967_; lean_object* v_currRecDepth_968_; lean_object* v_maxRecDepth_969_; lean_object* v_ref_970_; size_t v_sz_971_; lean_object* v___x_972_; lean_object* v___x_974_; 
v_methods_966_ = lean_ctor_get(v___y_958_, 0);
v_quotContext_967_ = lean_ctor_get(v___y_958_, 1);
v_currRecDepth_968_ = lean_ctor_get(v___y_958_, 3);
v_maxRecDepth_969_ = lean_ctor_get(v___y_958_, 4);
v_ref_970_ = lean_ctor_get(v___y_958_, 5);
v_sz_971_ = lean_array_size(v_fst_947_);
v___x_972_ = lean_nat_add(v_macroScope_960_, v___x_948_);
if (v_isShared_965_ == 0)
{
lean_ctor_set(v___x_964_, 0, v___x_972_);
v___x_974_ = v___x_964_;
goto v_reusejp_973_;
}
else
{
lean_object* v_reuseFailAlloc_1050_; 
v_reuseFailAlloc_1050_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v_reuseFailAlloc_1050_, 0, v___x_972_);
lean_ctor_set(v_reuseFailAlloc_1050_, 1, v_traceMsgs_961_);
lean_ctor_set(v_reuseFailAlloc_1050_, 2, v_expandedMacroDecls_962_);
v___x_974_ = v_reuseFailAlloc_1050_;
goto v_reusejp_973_;
}
v_reusejp_973_:
{
lean_object* v___x_975_; lean_object* v___x_976_; 
lean_inc(v_ref_970_);
lean_inc(v_maxRecDepth_969_);
lean_inc(v_currRecDepth_968_);
lean_inc(v_quotContext_967_);
lean_inc(v_methods_966_);
v___x_975_ = lean_alloc_ctor(0, 6, 0);
lean_ctor_set(v___x_975_, 0, v_methods_966_);
lean_ctor_set(v___x_975_, 1, v_quotContext_967_);
lean_ctor_set(v___x_975_, 2, v_macroScope_960_);
lean_ctor_set(v___x_975_, 3, v_currRecDepth_968_);
lean_ctor_set(v___x_975_, 4, v_maxRecDepth_969_);
lean_ctor_set(v___x_975_, 5, v_ref_970_);
v___x_976_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2___redArg(v_sz_971_, v___x_949_, v_fst_947_, v___x_975_, v___x_974_);
lean_dec_ref_known(v___x_975_, 6);
if (lean_obj_tag(v___x_976_) == 0)
{
lean_object* v_a_977_; lean_object* v_a_978_; lean_object* v___x_980_; uint8_t v_isShared_981_; uint8_t v_isSharedCheck_1040_; 
v_a_977_ = lean_ctor_get(v___x_976_, 0);
v_a_978_ = lean_ctor_get(v___x_976_, 1);
v_isSharedCheck_1040_ = !lean_is_exclusive(v___x_976_);
if (v_isSharedCheck_1040_ == 0)
{
v___x_980_ = v___x_976_;
v_isShared_981_ = v_isSharedCheck_1040_;
goto v_resetjp_979_;
}
else
{
lean_inc(v_a_978_);
lean_inc(v_a_977_);
lean_dec(v___x_976_);
v___x_980_ = lean_box(0);
v_isShared_981_ = v_isSharedCheck_1040_;
goto v_resetjp_979_;
}
v_resetjp_979_:
{
lean_object* v___y_983_; lean_object* v___y_984_; lean_object* v___y_985_; lean_object* v___y_986_; lean_object* v___y_987_; lean_object* v___y_988_; lean_object* v___y_989_; lean_object* v___y_990_; lean_object* v___y_1004_; 
if (lean_obj_tag(v_srcs_956_) == 0)
{
lean_object* v___x_1037_; 
v___x_1037_ = lean_mk_empty_array_with_capacity(v___x_951_);
v___y_1004_ = v___x_1037_;
goto v___jp_1003_;
}
else
{
lean_object* v_val_1038_; lean_object* v___x_1039_; 
v_val_1038_ = lean_ctor_get(v_srcs_956_, 0);
v___x_1039_ = l_Lean_Syntax_TSepArray_getElems___redArg(v_val_1038_);
v___y_1004_ = v___x_1039_;
goto v___jp_1003_;
}
v___jp_982_:
{
lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v___x_993_; lean_object* v___x_994_; lean_object* v___x_995_; lean_object* v___x_996_; uint8_t v___x_997_; 
v___x_991_ = l_Array_append___redArg(v___y_984_, v___y_990_);
lean_dec_ref(v___y_990_);
lean_inc_n(v___y_983_, 2);
v___x_992_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_992_, 0, v___y_983_);
lean_ctor_set(v___x_992_, 1, v___y_986_);
lean_ctor_set(v___x_992_, 2, v___x_991_);
v___x_993_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__0));
v___x_994_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_994_, 0, v___y_983_);
lean_ctor_set(v___x_994_, 1, v___x_993_);
v___x_995_ = l_Lean_Syntax_node6(v___y_983_, v___x_950_, v___y_987_, v___y_989_, v___y_985_, v___y_988_, v___x_992_, v___x_994_);
v___x_996_ = lean_array_get_size(v_a_977_);
v___x_997_ = lean_nat_dec_lt(v___x_951_, v___x_996_);
if (v___x_997_ == 0)
{
lean_object* v___x_999_; 
lean_dec(v_a_977_);
if (v_isShared_981_ == 0)
{
lean_ctor_set(v___x_980_, 0, v___x_995_);
v___x_999_ = v___x_980_;
goto v_reusejp_998_;
}
else
{
lean_object* v_reuseFailAlloc_1000_; 
v_reuseFailAlloc_1000_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1000_, 0, v___x_995_);
lean_ctor_set(v_reuseFailAlloc_1000_, 1, v_a_978_);
v___x_999_ = v_reuseFailAlloc_1000_;
goto v_reusejp_998_;
}
v_reusejp_998_:
{
return v___x_999_;
}
}
else
{
size_t v___x_1001_; lean_object* v___x_1002_; 
lean_del_object(v___x_980_);
v___x_1001_ = lean_usize_of_nat(v___x_996_);
v___x_1002_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldrMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__6(v_a_977_, v___x_1001_, v___x_949_, v___x_995_, v___y_958_, v_a_978_);
lean_dec(v_a_977_);
return v___x_1002_;
}
}
v___jp_1003_:
{
size_t v_sz_1005_; lean_object* v___x_1006_; size_t v_sz_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; uint8_t v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; size_t v_sz_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; 
v_sz_1005_ = lean_array_size(v_a_977_);
lean_inc(v_a_977_);
v___x_1006_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__3(v_sz_1005_, v___x_949_, v_a_977_);
v_sz_1007_ = lean_array_size(v___x_1006_);
v___x_1008_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__4(v_sz_1007_, v___x_949_, v___x_1006_);
v___x_1009_ = l_Array_append___redArg(v___y_1004_, v___x_1008_);
lean_dec_ref(v___x_1008_);
v___x_1010_ = 0;
v___x_1011_ = l_Lean_SourceInfo_fromRef(v_ref_970_, v___x_1010_);
v___x_1012_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__1));
lean_inc_n(v___x_1011_, 8);
v___x_1013_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1013_, 0, v___x_1011_);
lean_ctor_set(v___x_1013_, 1, v___x_1012_);
v___x_1014_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__3));
v___x_1015_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__4, &lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__4_once, _init_lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__4);
v___x_1016_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__5));
v___x_1017_ = l_Lean_Syntax_SepArray_ofElems(v___x_1016_, v___x_1009_);
lean_dec_ref(v___x_1009_);
v___x_1018_ = l_Array_append___redArg(v___x_1015_, v___x_1017_);
lean_dec_ref(v___x_1017_);
v___x_1019_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1019_, 0, v___x_1011_);
lean_ctor_set(v___x_1019_, 1, v___x_1014_);
lean_ctor_set(v___x_1019_, 2, v___x_1018_);
v___x_1020_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__6));
v___x_1021_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1021_, 0, v___x_1011_);
lean_ctor_set(v___x_1021_, 1, v___x_1020_);
v___x_1022_ = l_Lean_Syntax_node2(v___x_1011_, v___x_1014_, v___x_1019_, v___x_1021_);
v_sz_1023_ = lean_array_size(v_snd_952_);
v___x_1024_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__5(v_sz_1023_, v___x_949_, v_snd_952_);
v___x_1025_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__7, &lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__7_once, _init_lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__7);
v___x_1026_ = l_Lean_mkSepArray(v___x_1024_, v___x_1025_);
lean_dec_ref(v___x_1024_);
v___x_1027_ = l_Array_append___redArg(v___x_1015_, v___x_1026_);
lean_dec_ref(v___x_1026_);
v___x_1028_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1028_, 0, v___x_1011_);
lean_ctor_set(v___x_1028_, 1, v___x_1014_);
lean_ctor_set(v___x_1028_, 2, v___x_1027_);
v___x_1029_ = l_Lean_Syntax_node1(v___x_1011_, v___x_953_, v___x_1028_);
v___x_1030_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1030_, 0, v___x_1011_);
lean_ctor_set(v___x_1030_, 1, v___x_1014_);
lean_ctor_set(v___x_1030_, 2, v___x_1015_);
v___x_1031_ = l_Lean_Syntax_node1(v___x_1011_, v___x_954_, v___x_1030_);
if (lean_obj_tag(v_ty_x3f_955_) == 1)
{
lean_object* v_val_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; 
v_val_1032_ = lean_ctor_get(v_ty_x3f_955_, 0);
lean_inc(v_val_1032_);
lean_dec_ref_known(v_ty_x3f_955_, 1);
v___x_1033_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___closed__8));
lean_inc(v___x_1011_);
v___x_1034_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1034_, 0, v___x_1011_);
lean_ctor_set(v___x_1034_, 1, v___x_1033_);
v___x_1035_ = l_Array_mkArray2___redArg(v___x_1034_, v_val_1032_);
v___y_983_ = v___x_1011_;
v___y_984_ = v___x_1015_;
v___y_985_ = v___x_1029_;
v___y_986_ = v___x_1014_;
v___y_987_ = v___x_1013_;
v___y_988_ = v___x_1031_;
v___y_989_ = v___x_1022_;
v___y_990_ = v___x_1035_;
goto v___jp_982_;
}
else
{
lean_object* v___x_1036_; 
lean_dec(v_ty_x3f_955_);
v___x_1036_ = lean_mk_empty_array_with_capacity(v___x_951_);
v___y_983_ = v___x_1011_;
v___y_984_ = v___x_1015_;
v___y_985_ = v___x_1029_;
v___y_986_ = v___x_1014_;
v___y_987_ = v___x_1013_;
v___y_988_ = v___x_1031_;
v___y_989_ = v___x_1022_;
v___y_990_ = v___x_1036_;
goto v___jp_982_;
}
}
}
}
else
{
lean_object* v_a_1041_; lean_object* v_a_1042_; lean_object* v___x_1044_; uint8_t v_isShared_1045_; uint8_t v_isSharedCheck_1049_; 
lean_dec(v_ty_x3f_955_);
lean_dec(v___x_954_);
lean_dec(v___x_953_);
lean_dec(v_snd_952_);
lean_dec(v___x_950_);
v_a_1041_ = lean_ctor_get(v___x_976_, 0);
v_a_1042_ = lean_ctor_get(v___x_976_, 1);
v_isSharedCheck_1049_ = !lean_is_exclusive(v___x_976_);
if (v_isSharedCheck_1049_ == 0)
{
v___x_1044_ = v___x_976_;
v_isShared_1045_ = v_isSharedCheck_1049_;
goto v_resetjp_1043_;
}
else
{
lean_inc(v_a_1042_);
lean_inc(v_a_1041_);
lean_dec(v___x_976_);
v___x_1044_ = lean_box(0);
v_isShared_1045_ = v_isSharedCheck_1049_;
goto v_resetjp_1043_;
}
v_resetjp_1043_:
{
lean_object* v___x_1047_; 
if (v_isShared_1045_ == 0)
{
v___x_1047_ = v___x_1044_;
goto v_reusejp_1046_;
}
else
{
lean_object* v_reuseFailAlloc_1048_; 
v_reuseFailAlloc_1048_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1048_, 0, v_a_1041_);
lean_ctor_set(v_reuseFailAlloc_1048_, 1, v_a_1042_);
v___x_1047_ = v_reuseFailAlloc_1048_;
goto v_reusejp_1046_;
}
v_reusejp_1046_:
{
return v___x_1047_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0___boxed(lean_object* v_fst_1052_, lean_object* v___x_1053_, lean_object* v___x_1054_, lean_object* v___x_1055_, lean_object* v___x_1056_, lean_object* v_snd_1057_, lean_object* v___x_1058_, lean_object* v___x_1059_, lean_object* v_ty_x3f_1060_, lean_object* v_srcs_1061_, lean_object* v_____r_1062_, lean_object* v___y_1063_, lean_object* v___y_1064_){
_start:
{
size_t v___x_20630__boxed_1065_; lean_object* v_res_1066_; 
v___x_20630__boxed_1065_ = lean_unbox_usize(v___x_1054_);
lean_dec(v___x_1054_);
v_res_1066_ = lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0(v_fst_1052_, v___x_1053_, v___x_20630__boxed_1065_, v___x_1055_, v___x_1056_, v_snd_1057_, v___x_1058_, v___x_1059_, v_ty_x3f_1060_, v_srcs_1061_, v_____r_1062_, v___y_1063_, v___y_1064_);
lean_dec_ref(v___y_1063_);
lean_dec(v_srcs_1061_);
lean_dec(v___x_1056_);
lean_dec(v___x_1053_);
return v_res_1066_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__7(uint8_t v___x_1067_, lean_object* v_as_1068_, size_t v_i_1069_, size_t v_stop_1070_, lean_object* v_b_1071_){
_start:
{
lean_object* v___y_1073_; uint8_t v___x_1077_; 
v___x_1077_ = lean_usize_dec_eq(v_i_1069_, v_stop_1070_);
if (v___x_1077_ == 0)
{
lean_object* v_fst_1078_; uint8_t v___x_1079_; 
v_fst_1078_ = lean_ctor_get(v_b_1071_, 0);
v___x_1079_ = lean_unbox(v_fst_1078_);
if (v___x_1079_ == 0)
{
lean_object* v_snd_1080_; lean_object* v___x_1082_; uint8_t v_isShared_1083_; uint8_t v_isSharedCheck_1088_; 
v_snd_1080_ = lean_ctor_get(v_b_1071_, 1);
v_isSharedCheck_1088_ = !lean_is_exclusive(v_b_1071_);
if (v_isSharedCheck_1088_ == 0)
{
lean_object* v_unused_1089_; 
v_unused_1089_ = lean_ctor_get(v_b_1071_, 0);
lean_dec(v_unused_1089_);
v___x_1082_ = v_b_1071_;
v_isShared_1083_ = v_isSharedCheck_1088_;
goto v_resetjp_1081_;
}
else
{
lean_inc(v_snd_1080_);
lean_dec(v_b_1071_);
v___x_1082_ = lean_box(0);
v_isShared_1083_ = v_isSharedCheck_1088_;
goto v_resetjp_1081_;
}
v_resetjp_1081_:
{
lean_object* v___x_1084_; lean_object* v___x_1086_; 
v___x_1084_ = lean_box(v___x_1067_);
if (v_isShared_1083_ == 0)
{
lean_ctor_set(v___x_1082_, 0, v___x_1084_);
v___x_1086_ = v___x_1082_;
goto v_reusejp_1085_;
}
else
{
lean_object* v_reuseFailAlloc_1087_; 
v_reuseFailAlloc_1087_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1087_, 0, v___x_1084_);
lean_ctor_set(v_reuseFailAlloc_1087_, 1, v_snd_1080_);
v___x_1086_ = v_reuseFailAlloc_1087_;
goto v_reusejp_1085_;
}
v_reusejp_1085_:
{
v___y_1073_ = v___x_1086_;
goto v___jp_1072_;
}
}
}
else
{
lean_object* v_snd_1090_; lean_object* v___x_1092_; uint8_t v_isShared_1093_; uint8_t v_isSharedCheck_1100_; 
v_snd_1090_ = lean_ctor_get(v_b_1071_, 1);
v_isSharedCheck_1100_ = !lean_is_exclusive(v_b_1071_);
if (v_isSharedCheck_1100_ == 0)
{
lean_object* v_unused_1101_; 
v_unused_1101_ = lean_ctor_get(v_b_1071_, 0);
lean_dec(v_unused_1101_);
v___x_1092_ = v_b_1071_;
v_isShared_1093_ = v_isSharedCheck_1100_;
goto v_resetjp_1091_;
}
else
{
lean_inc(v_snd_1090_);
lean_dec(v_b_1071_);
v___x_1092_ = lean_box(0);
v_isShared_1093_ = v_isSharedCheck_1100_;
goto v_resetjp_1091_;
}
v_resetjp_1091_:
{
lean_object* v___x_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; lean_object* v___x_1098_; 
v___x_1094_ = lean_array_uget_borrowed(v_as_1068_, v_i_1069_);
lean_inc(v___x_1094_);
v___x_1095_ = lean_array_push(v_snd_1090_, v___x_1094_);
v___x_1096_ = lean_box(v___x_1077_);
if (v_isShared_1093_ == 0)
{
lean_ctor_set(v___x_1092_, 1, v___x_1095_);
lean_ctor_set(v___x_1092_, 0, v___x_1096_);
v___x_1098_ = v___x_1092_;
goto v_reusejp_1097_;
}
else
{
lean_object* v_reuseFailAlloc_1099_; 
v_reuseFailAlloc_1099_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1099_, 0, v___x_1096_);
lean_ctor_set(v_reuseFailAlloc_1099_, 1, v___x_1095_);
v___x_1098_ = v_reuseFailAlloc_1099_;
goto v_reusejp_1097_;
}
v_reusejp_1097_:
{
v___y_1073_ = v___x_1098_;
goto v___jp_1072_;
}
}
}
}
else
{
return v_b_1071_;
}
v___jp_1072_:
{
size_t v___x_1074_; size_t v___x_1075_; 
v___x_1074_ = ((size_t)1ULL);
v___x_1075_ = lean_usize_add(v_i_1069_, v___x_1074_);
v_i_1069_ = v___x_1075_;
v_b_1071_ = v___y_1073_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__7___boxed(lean_object* v___x_1102_, lean_object* v_as_1103_, lean_object* v_i_1104_, lean_object* v_stop_1105_, lean_object* v_b_1106_){
_start:
{
uint8_t v___x_20827__boxed_1107_; size_t v_i_boxed_1108_; size_t v_stop_boxed_1109_; lean_object* v_res_1110_; 
v___x_20827__boxed_1107_ = lean_unbox(v___x_1102_);
v_i_boxed_1108_ = lean_unbox_usize(v_i_1104_);
lean_dec(v_i_1104_);
v_stop_boxed_1109_ = lean_unbox_usize(v_stop_1105_);
lean_dec(v_stop_1105_);
v_res_1110_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__7(v___x_20827__boxed_1107_, v_as_1103_, v_i_boxed_1108_, v_stop_boxed_1109_, v_b_1106_);
lean_dec_ref(v_as_1103_);
return v_res_1110_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__0(size_t v_sz_1111_, size_t v_i_1112_, lean_object* v_bs_1113_){
_start:
{
uint8_t v___x_1114_; 
v___x_1114_ = lean_usize_dec_lt(v_i_1112_, v_sz_1111_);
if (v___x_1114_ == 0)
{
lean_object* v___x_1115_; 
v___x_1115_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1115_, 0, v_bs_1113_);
return v___x_1115_;
}
else
{
lean_object* v_v_1116_; lean_object* v___x_1117_; lean_object* v_bs_x27_1118_; size_t v___x_1119_; size_t v___x_1120_; lean_object* v___x_1121_; 
v_v_1116_ = lean_array_uget(v_bs_1113_, v_i_1112_);
v___x_1117_ = lean_unsigned_to_nat(0u);
v_bs_x27_1118_ = lean_array_uset(v_bs_1113_, v_i_1112_, v___x_1117_);
v___x_1119_ = ((size_t)1ULL);
v___x_1120_ = lean_usize_add(v_i_1112_, v___x_1119_);
v___x_1121_ = lean_array_uset(v_bs_x27_1118_, v_i_1112_, v_v_1116_);
v_i_1112_ = v___x_1120_;
v_bs_1113_ = v___x_1121_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__0___boxed(lean_object* v_sz_1123_, lean_object* v_i_1124_, lean_object* v_bs_1125_){
_start:
{
size_t v_sz_boxed_1126_; size_t v_i_boxed_1127_; lean_object* v_res_1128_; 
v_sz_boxed_1126_ = lean_unbox_usize(v_sz_1123_);
lean_dec(v_sz_1123_);
v_i_boxed_1127_ = lean_unbox_usize(v_i_1124_);
lean_dec(v_i_1124_);
v_res_1128_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__0(v_sz_boxed_1126_, v_i_boxed_1127_, v_bs_1125_);
return v_res_1128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg(lean_object* v_as_1152_, size_t v_sz_1153_, size_t v_i_1154_, lean_object* v_b_1155_, lean_object* v___y_1156_){
_start:
{
lean_object* v_a_1158_; lean_object* v_a_1159_; uint8_t v___x_1163_; 
v___x_1163_ = lean_usize_dec_lt(v_i_1154_, v_sz_1153_);
if (v___x_1163_ == 0)
{
lean_object* v___x_1164_; 
v___x_1164_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1164_, 0, v_b_1155_);
lean_ctor_set(v___x_1164_, 1, v___y_1156_);
return v___x_1164_;
}
else
{
lean_object* v_fst_1165_; lean_object* v_snd_1166_; lean_object* v___x_1168_; uint8_t v_isShared_1169_; uint8_t v_isSharedCheck_1335_; 
v_fst_1165_ = lean_ctor_get(v_b_1155_, 0);
v_snd_1166_ = lean_ctor_get(v_b_1155_, 1);
v_isSharedCheck_1335_ = !lean_is_exclusive(v_b_1155_);
if (v_isSharedCheck_1335_ == 0)
{
v___x_1168_ = v_b_1155_;
v_isShared_1169_ = v_isSharedCheck_1335_;
goto v_resetjp_1167_;
}
else
{
lean_inc(v_snd_1166_);
lean_inc(v_fst_1165_);
lean_dec(v_b_1155_);
v___x_1168_ = lean_box(0);
v_isShared_1169_ = v_isSharedCheck_1335_;
goto v_resetjp_1167_;
}
v_resetjp_1167_:
{
lean_object* v_a_1170_; lean_object* v___x_1171_; uint8_t v___x_1172_; 
v_a_1170_ = lean_array_uget_borrowed(v_as_1152_, v_i_1154_);
v___x_1171_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__1));
lean_inc(v_a_1170_);
v___x_1172_ = l_Lean_Syntax_isOfKind(v_a_1170_, v___x_1171_);
if (v___x_1172_ == 0)
{
lean_object* v___x_1173_; 
v___x_1173_ = l_Lean_Macro_throwUnsupported___redArg(v___y_1156_);
if (lean_obj_tag(v___x_1173_) == 0)
{
lean_object* v_a_1174_; lean_object* v___x_1176_; 
v_a_1174_ = lean_ctor_get(v___x_1173_, 1);
lean_inc(v_a_1174_);
lean_dec_ref_known(v___x_1173_, 2);
if (v_isShared_1169_ == 0)
{
v___x_1176_ = v___x_1168_;
goto v_reusejp_1175_;
}
else
{
lean_object* v_reuseFailAlloc_1177_; 
v_reuseFailAlloc_1177_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1177_, 0, v_fst_1165_);
lean_ctor_set(v_reuseFailAlloc_1177_, 1, v_snd_1166_);
v___x_1176_ = v_reuseFailAlloc_1177_;
goto v_reusejp_1175_;
}
v_reusejp_1175_:
{
v_a_1158_ = v___x_1176_;
v_a_1159_ = v_a_1174_;
goto v___jp_1157_;
}
}
else
{
lean_object* v_a_1178_; lean_object* v_a_1179_; lean_object* v___x_1181_; uint8_t v_isShared_1182_; uint8_t v_isSharedCheck_1186_; 
lean_del_object(v___x_1168_);
lean_dec(v_snd_1166_);
lean_dec(v_fst_1165_);
v_a_1178_ = lean_ctor_get(v___x_1173_, 0);
v_a_1179_ = lean_ctor_get(v___x_1173_, 1);
v_isSharedCheck_1186_ = !lean_is_exclusive(v___x_1173_);
if (v_isSharedCheck_1186_ == 0)
{
v___x_1181_ = v___x_1173_;
v_isShared_1182_ = v_isSharedCheck_1186_;
goto v_resetjp_1180_;
}
else
{
lean_inc(v_a_1179_);
lean_inc(v_a_1178_);
lean_dec(v___x_1173_);
v___x_1181_ = lean_box(0);
v_isShared_1182_ = v_isSharedCheck_1186_;
goto v_resetjp_1180_;
}
v_resetjp_1180_:
{
lean_object* v___x_1184_; 
if (v_isShared_1182_ == 0)
{
v___x_1184_ = v___x_1181_;
goto v_reusejp_1183_;
}
else
{
lean_object* v_reuseFailAlloc_1185_; 
v_reuseFailAlloc_1185_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1185_, 0, v_a_1178_);
lean_ctor_set(v_reuseFailAlloc_1185_, 1, v_a_1179_);
v___x_1184_ = v_reuseFailAlloc_1185_;
goto v_reusejp_1183_;
}
v_reusejp_1183_:
{
return v___x_1184_;
}
}
}
}
else
{
lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; uint8_t v___x_1190_; 
v___x_1187_ = lean_unsigned_to_nat(0u);
v___x_1188_ = l_Lean_Syntax_getArg(v_a_1170_, v___x_1187_);
v___x_1189_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__3));
lean_inc(v___x_1188_);
v___x_1190_ = l_Lean_Syntax_isOfKind(v___x_1188_, v___x_1189_);
if (v___x_1190_ == 0)
{
lean_object* v___x_1191_; 
lean_dec(v___x_1188_);
v___x_1191_ = l_Lean_Macro_throwUnsupported___redArg(v___y_1156_);
if (lean_obj_tag(v___x_1191_) == 0)
{
lean_object* v_a_1192_; lean_object* v___x_1194_; 
v_a_1192_ = lean_ctor_get(v___x_1191_, 1);
lean_inc(v_a_1192_);
lean_dec_ref_known(v___x_1191_, 2);
if (v_isShared_1169_ == 0)
{
v___x_1194_ = v___x_1168_;
goto v_reusejp_1193_;
}
else
{
lean_object* v_reuseFailAlloc_1195_; 
v_reuseFailAlloc_1195_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1195_, 0, v_fst_1165_);
lean_ctor_set(v_reuseFailAlloc_1195_, 1, v_snd_1166_);
v___x_1194_ = v_reuseFailAlloc_1195_;
goto v_reusejp_1193_;
}
v_reusejp_1193_:
{
v_a_1158_ = v___x_1194_;
v_a_1159_ = v_a_1192_;
goto v___jp_1157_;
}
}
else
{
lean_object* v_a_1196_; lean_object* v_a_1197_; lean_object* v___x_1199_; uint8_t v_isShared_1200_; uint8_t v_isSharedCheck_1204_; 
lean_del_object(v___x_1168_);
lean_dec(v_snd_1166_);
lean_dec(v_fst_1165_);
v_a_1196_ = lean_ctor_get(v___x_1191_, 0);
v_a_1197_ = lean_ctor_get(v___x_1191_, 1);
v_isSharedCheck_1204_ = !lean_is_exclusive(v___x_1191_);
if (v_isSharedCheck_1204_ == 0)
{
v___x_1199_ = v___x_1191_;
v_isShared_1200_ = v_isSharedCheck_1204_;
goto v_resetjp_1198_;
}
else
{
lean_inc(v_a_1197_);
lean_inc(v_a_1196_);
lean_dec(v___x_1191_);
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
v_reuseFailAlloc_1203_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1203_, 0, v_a_1196_);
lean_ctor_set(v_reuseFailAlloc_1203_, 1, v_a_1197_);
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
lean_object* v___x_1205_; lean_object* v___x_1206_; uint8_t v___x_1207_; 
v___x_1205_ = l_Lean_Syntax_getArg(v___x_1188_, v___x_1187_);
v___x_1206_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__4));
lean_inc(v___x_1205_);
v___x_1207_ = l_Lean_Syntax_isOfKind(v___x_1205_, v___x_1206_);
if (v___x_1207_ == 0)
{
lean_object* v___x_1208_; 
lean_dec(v___x_1205_);
lean_dec(v___x_1188_);
v___x_1208_ = l_Lean_Macro_throwUnsupported___redArg(v___y_1156_);
if (lean_obj_tag(v___x_1208_) == 0)
{
lean_object* v_a_1209_; lean_object* v___x_1211_; 
v_a_1209_ = lean_ctor_get(v___x_1208_, 1);
lean_inc(v_a_1209_);
lean_dec_ref_known(v___x_1208_, 2);
if (v_isShared_1169_ == 0)
{
v___x_1211_ = v___x_1168_;
goto v_reusejp_1210_;
}
else
{
lean_object* v_reuseFailAlloc_1212_; 
v_reuseFailAlloc_1212_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1212_, 0, v_fst_1165_);
lean_ctor_set(v_reuseFailAlloc_1212_, 1, v_snd_1166_);
v___x_1211_ = v_reuseFailAlloc_1212_;
goto v_reusejp_1210_;
}
v_reusejp_1210_:
{
v_a_1158_ = v___x_1211_;
v_a_1159_ = v_a_1209_;
goto v___jp_1157_;
}
}
else
{
lean_object* v_a_1213_; lean_object* v_a_1214_; lean_object* v___x_1216_; uint8_t v_isShared_1217_; uint8_t v_isSharedCheck_1221_; 
lean_del_object(v___x_1168_);
lean_dec(v_snd_1166_);
lean_dec(v_fst_1165_);
v_a_1213_ = lean_ctor_get(v___x_1208_, 0);
v_a_1214_ = lean_ctor_get(v___x_1208_, 1);
v_isSharedCheck_1221_ = !lean_is_exclusive(v___x_1208_);
if (v_isSharedCheck_1221_ == 0)
{
v___x_1216_ = v___x_1208_;
v_isShared_1217_ = v_isSharedCheck_1221_;
goto v_resetjp_1215_;
}
else
{
lean_inc(v_a_1214_);
lean_inc(v_a_1213_);
lean_dec(v___x_1208_);
v___x_1216_ = lean_box(0);
v_isShared_1217_ = v_isSharedCheck_1221_;
goto v_resetjp_1215_;
}
v_resetjp_1215_:
{
lean_object* v___x_1219_; 
if (v_isShared_1217_ == 0)
{
v___x_1219_ = v___x_1216_;
goto v_reusejp_1218_;
}
else
{
lean_object* v_reuseFailAlloc_1220_; 
v_reuseFailAlloc_1220_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1220_, 0, v_a_1213_);
lean_ctor_set(v_reuseFailAlloc_1220_, 1, v_a_1214_);
v___x_1219_ = v_reuseFailAlloc_1220_;
goto v_reusejp_1218_;
}
v_reusejp_1218_:
{
return v___x_1219_;
}
}
}
}
else
{
lean_object* v___x_1222_; lean_object* v___x_1223_; uint8_t v___x_1224_; 
v___x_1222_ = lean_unsigned_to_nat(1u);
v___x_1223_ = l_Lean_Syntax_getArg(v___x_1188_, v___x_1222_);
lean_dec(v___x_1188_);
v___x_1224_ = l_Lean_Syntax_matchesNull(v___x_1223_, v___x_1187_);
if (v___x_1224_ == 0)
{
lean_object* v___x_1225_; 
lean_dec(v___x_1205_);
v___x_1225_ = l_Lean_Macro_throwUnsupported___redArg(v___y_1156_);
if (lean_obj_tag(v___x_1225_) == 0)
{
lean_object* v_a_1226_; lean_object* v___x_1228_; 
v_a_1226_ = lean_ctor_get(v___x_1225_, 1);
lean_inc(v_a_1226_);
lean_dec_ref_known(v___x_1225_, 2);
if (v_isShared_1169_ == 0)
{
v___x_1228_ = v___x_1168_;
goto v_reusejp_1227_;
}
else
{
lean_object* v_reuseFailAlloc_1229_; 
v_reuseFailAlloc_1229_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1229_, 0, v_fst_1165_);
lean_ctor_set(v_reuseFailAlloc_1229_, 1, v_snd_1166_);
v___x_1228_ = v_reuseFailAlloc_1229_;
goto v_reusejp_1227_;
}
v_reusejp_1227_:
{
v_a_1158_ = v___x_1228_;
v_a_1159_ = v_a_1226_;
goto v___jp_1157_;
}
}
else
{
lean_object* v_a_1230_; lean_object* v_a_1231_; lean_object* v___x_1233_; uint8_t v_isShared_1234_; uint8_t v_isSharedCheck_1238_; 
lean_del_object(v___x_1168_);
lean_dec(v_snd_1166_);
lean_dec(v_fst_1165_);
v_a_1230_ = lean_ctor_get(v___x_1225_, 0);
v_a_1231_ = lean_ctor_get(v___x_1225_, 1);
v_isSharedCheck_1238_ = !lean_is_exclusive(v___x_1225_);
if (v_isSharedCheck_1238_ == 0)
{
v___x_1233_ = v___x_1225_;
v_isShared_1234_ = v_isSharedCheck_1238_;
goto v_resetjp_1232_;
}
else
{
lean_inc(v_a_1231_);
lean_inc(v_a_1230_);
lean_dec(v___x_1225_);
v___x_1233_ = lean_box(0);
v_isShared_1234_ = v_isSharedCheck_1238_;
goto v_resetjp_1232_;
}
v_resetjp_1232_:
{
lean_object* v___x_1236_; 
if (v_isShared_1234_ == 0)
{
v___x_1236_ = v___x_1233_;
goto v_reusejp_1235_;
}
else
{
lean_object* v_reuseFailAlloc_1237_; 
v_reuseFailAlloc_1237_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1237_, 0, v_a_1230_);
lean_ctor_set(v_reuseFailAlloc_1237_, 1, v_a_1231_);
v___x_1236_ = v_reuseFailAlloc_1237_;
goto v_reusejp_1235_;
}
v_reusejp_1235_:
{
return v___x_1236_;
}
}
}
}
else
{
lean_object* v___x_1239_; lean_object* v___x_1240_; uint8_t v___x_1241_; 
v___x_1239_ = lean_unsigned_to_nat(3u);
v___x_1240_ = l_Lean_Syntax_getArg(v_a_1170_, v___x_1222_);
lean_inc(v___x_1240_);
v___x_1241_ = l_Lean_Syntax_matchesNull(v___x_1240_, v___x_1239_);
if (v___x_1241_ == 0)
{
lean_object* v___x_1242_; 
lean_dec(v___x_1240_);
lean_dec(v___x_1205_);
v___x_1242_ = l_Lean_Macro_throwUnsupported___redArg(v___y_1156_);
if (lean_obj_tag(v___x_1242_) == 0)
{
lean_object* v_a_1243_; lean_object* v___x_1245_; 
v_a_1243_ = lean_ctor_get(v___x_1242_, 1);
lean_inc(v_a_1243_);
lean_dec_ref_known(v___x_1242_, 2);
if (v_isShared_1169_ == 0)
{
v___x_1245_ = v___x_1168_;
goto v_reusejp_1244_;
}
else
{
lean_object* v_reuseFailAlloc_1246_; 
v_reuseFailAlloc_1246_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1246_, 0, v_fst_1165_);
lean_ctor_set(v_reuseFailAlloc_1246_, 1, v_snd_1166_);
v___x_1245_ = v_reuseFailAlloc_1246_;
goto v_reusejp_1244_;
}
v_reusejp_1244_:
{
v_a_1158_ = v___x_1245_;
v_a_1159_ = v_a_1243_;
goto v___jp_1157_;
}
}
else
{
lean_object* v_a_1247_; lean_object* v_a_1248_; lean_object* v___x_1250_; uint8_t v_isShared_1251_; uint8_t v_isSharedCheck_1255_; 
lean_del_object(v___x_1168_);
lean_dec(v_snd_1166_);
lean_dec(v_fst_1165_);
v_a_1247_ = lean_ctor_get(v___x_1242_, 0);
v_a_1248_ = lean_ctor_get(v___x_1242_, 1);
v_isSharedCheck_1255_ = !lean_is_exclusive(v___x_1242_);
if (v_isSharedCheck_1255_ == 0)
{
v___x_1250_ = v___x_1242_;
v_isShared_1251_ = v_isSharedCheck_1255_;
goto v_resetjp_1249_;
}
else
{
lean_inc(v_a_1248_);
lean_inc(v_a_1247_);
lean_dec(v___x_1242_);
v___x_1250_ = lean_box(0);
v_isShared_1251_ = v_isSharedCheck_1255_;
goto v_resetjp_1249_;
}
v_resetjp_1249_:
{
lean_object* v___x_1253_; 
if (v_isShared_1251_ == 0)
{
v___x_1253_ = v___x_1250_;
goto v_reusejp_1252_;
}
else
{
lean_object* v_reuseFailAlloc_1254_; 
v_reuseFailAlloc_1254_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1254_, 0, v_a_1247_);
lean_ctor_set(v_reuseFailAlloc_1254_, 1, v_a_1248_);
v___x_1253_ = v_reuseFailAlloc_1254_;
goto v_reusejp_1252_;
}
v_reusejp_1252_:
{
return v___x_1253_;
}
}
}
}
else
{
lean_object* v___x_1256_; uint8_t v___x_1257_; 
v___x_1256_ = l_Lean_Syntax_getArg(v___x_1240_, v___x_1187_);
v___x_1257_ = l_Lean_Syntax_matchesNull(v___x_1256_, v___x_1187_);
if (v___x_1257_ == 0)
{
lean_object* v___x_1258_; 
lean_dec(v___x_1240_);
lean_dec(v___x_1205_);
v___x_1258_ = l_Lean_Macro_throwUnsupported___redArg(v___y_1156_);
if (lean_obj_tag(v___x_1258_) == 0)
{
lean_object* v_a_1259_; lean_object* v___x_1261_; 
v_a_1259_ = lean_ctor_get(v___x_1258_, 1);
lean_inc(v_a_1259_);
lean_dec_ref_known(v___x_1258_, 2);
if (v_isShared_1169_ == 0)
{
v___x_1261_ = v___x_1168_;
goto v_reusejp_1260_;
}
else
{
lean_object* v_reuseFailAlloc_1262_; 
v_reuseFailAlloc_1262_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1262_, 0, v_fst_1165_);
lean_ctor_set(v_reuseFailAlloc_1262_, 1, v_snd_1166_);
v___x_1261_ = v_reuseFailAlloc_1262_;
goto v_reusejp_1260_;
}
v_reusejp_1260_:
{
v_a_1158_ = v___x_1261_;
v_a_1159_ = v_a_1259_;
goto v___jp_1157_;
}
}
else
{
lean_object* v_a_1263_; lean_object* v_a_1264_; lean_object* v___x_1266_; uint8_t v_isShared_1267_; uint8_t v_isSharedCheck_1271_; 
lean_del_object(v___x_1168_);
lean_dec(v_snd_1166_);
lean_dec(v_fst_1165_);
v_a_1263_ = lean_ctor_get(v___x_1258_, 0);
v_a_1264_ = lean_ctor_get(v___x_1258_, 1);
v_isSharedCheck_1271_ = !lean_is_exclusive(v___x_1258_);
if (v_isSharedCheck_1271_ == 0)
{
v___x_1266_ = v___x_1258_;
v_isShared_1267_ = v_isSharedCheck_1271_;
goto v_resetjp_1265_;
}
else
{
lean_inc(v_a_1264_);
lean_inc(v_a_1263_);
lean_dec(v___x_1258_);
v___x_1266_ = lean_box(0);
v_isShared_1267_ = v_isSharedCheck_1271_;
goto v_resetjp_1265_;
}
v_resetjp_1265_:
{
lean_object* v___x_1269_; 
if (v_isShared_1267_ == 0)
{
v___x_1269_ = v___x_1266_;
goto v_reusejp_1268_;
}
else
{
lean_object* v_reuseFailAlloc_1270_; 
v_reuseFailAlloc_1270_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1270_, 0, v_a_1263_);
lean_ctor_set(v_reuseFailAlloc_1270_, 1, v_a_1264_);
v___x_1269_ = v_reuseFailAlloc_1270_;
goto v_reusejp_1268_;
}
v_reusejp_1268_:
{
return v___x_1269_;
}
}
}
}
else
{
lean_object* v___x_1272_; uint8_t v___x_1273_; 
v___x_1272_ = l_Lean_Syntax_getArg(v___x_1240_, v___x_1222_);
v___x_1273_ = l_Lean_Syntax_matchesNull(v___x_1272_, v___x_1187_);
if (v___x_1273_ == 0)
{
lean_object* v___x_1274_; 
lean_dec(v___x_1240_);
lean_dec(v___x_1205_);
v___x_1274_ = l_Lean_Macro_throwUnsupported___redArg(v___y_1156_);
if (lean_obj_tag(v___x_1274_) == 0)
{
lean_object* v_a_1275_; lean_object* v___x_1277_; 
v_a_1275_ = lean_ctor_get(v___x_1274_, 1);
lean_inc(v_a_1275_);
lean_dec_ref_known(v___x_1274_, 2);
if (v_isShared_1169_ == 0)
{
v___x_1277_ = v___x_1168_;
goto v_reusejp_1276_;
}
else
{
lean_object* v_reuseFailAlloc_1278_; 
v_reuseFailAlloc_1278_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1278_, 0, v_fst_1165_);
lean_ctor_set(v_reuseFailAlloc_1278_, 1, v_snd_1166_);
v___x_1277_ = v_reuseFailAlloc_1278_;
goto v_reusejp_1276_;
}
v_reusejp_1276_:
{
v_a_1158_ = v___x_1277_;
v_a_1159_ = v_a_1275_;
goto v___jp_1157_;
}
}
else
{
lean_object* v_a_1279_; lean_object* v_a_1280_; lean_object* v___x_1282_; uint8_t v_isShared_1283_; uint8_t v_isSharedCheck_1287_; 
lean_del_object(v___x_1168_);
lean_dec(v_snd_1166_);
lean_dec(v_fst_1165_);
v_a_1279_ = lean_ctor_get(v___x_1274_, 0);
v_a_1280_ = lean_ctor_get(v___x_1274_, 1);
v_isSharedCheck_1287_ = !lean_is_exclusive(v___x_1274_);
if (v_isSharedCheck_1287_ == 0)
{
v___x_1282_ = v___x_1274_;
v_isShared_1283_ = v_isSharedCheck_1287_;
goto v_resetjp_1281_;
}
else
{
lean_inc(v_a_1280_);
lean_inc(v_a_1279_);
lean_dec(v___x_1274_);
v___x_1282_ = lean_box(0);
v_isShared_1283_ = v_isSharedCheck_1287_;
goto v_resetjp_1281_;
}
v_resetjp_1281_:
{
lean_object* v___x_1285_; 
if (v_isShared_1283_ == 0)
{
v___x_1285_ = v___x_1282_;
goto v_reusejp_1284_;
}
else
{
lean_object* v_reuseFailAlloc_1286_; 
v_reuseFailAlloc_1286_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1286_, 0, v_a_1279_);
lean_ctor_set(v_reuseFailAlloc_1286_, 1, v_a_1280_);
v___x_1285_ = v_reuseFailAlloc_1286_;
goto v_reusejp_1284_;
}
v_reusejp_1284_:
{
return v___x_1285_;
}
}
}
}
else
{
lean_object* v___x_1288_; lean_object* v___x_1289_; lean_object* v___x_1290_; uint8_t v___x_1291_; 
v___x_1288_ = lean_unsigned_to_nat(2u);
v___x_1289_ = l_Lean_Syntax_getArg(v___x_1240_, v___x_1288_);
lean_dec(v___x_1240_);
v___x_1290_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__6));
lean_inc(v___x_1289_);
v___x_1291_ = l_Lean_Syntax_isOfKind(v___x_1289_, v___x_1290_);
if (v___x_1291_ == 0)
{
lean_object* v___x_1292_; 
lean_dec(v___x_1289_);
lean_dec(v___x_1205_);
v___x_1292_ = l_Lean_Macro_throwUnsupported___redArg(v___y_1156_);
if (lean_obj_tag(v___x_1292_) == 0)
{
lean_object* v_a_1293_; lean_object* v___x_1295_; 
v_a_1293_ = lean_ctor_get(v___x_1292_, 1);
lean_inc(v_a_1293_);
lean_dec_ref_known(v___x_1292_, 2);
if (v_isShared_1169_ == 0)
{
v___x_1295_ = v___x_1168_;
goto v_reusejp_1294_;
}
else
{
lean_object* v_reuseFailAlloc_1296_; 
v_reuseFailAlloc_1296_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1296_, 0, v_fst_1165_);
lean_ctor_set(v_reuseFailAlloc_1296_, 1, v_snd_1166_);
v___x_1295_ = v_reuseFailAlloc_1296_;
goto v_reusejp_1294_;
}
v_reusejp_1294_:
{
v_a_1158_ = v___x_1295_;
v_a_1159_ = v_a_1293_;
goto v___jp_1157_;
}
}
else
{
lean_object* v_a_1297_; lean_object* v_a_1298_; lean_object* v___x_1300_; uint8_t v_isShared_1301_; uint8_t v_isSharedCheck_1305_; 
lean_del_object(v___x_1168_);
lean_dec(v_snd_1166_);
lean_dec(v_fst_1165_);
v_a_1297_ = lean_ctor_get(v___x_1292_, 0);
v_a_1298_ = lean_ctor_get(v___x_1292_, 1);
v_isSharedCheck_1305_ = !lean_is_exclusive(v___x_1292_);
if (v_isSharedCheck_1305_ == 0)
{
v___x_1300_ = v___x_1292_;
v_isShared_1301_ = v_isSharedCheck_1305_;
goto v_resetjp_1299_;
}
else
{
lean_inc(v_a_1298_);
lean_inc(v_a_1297_);
lean_dec(v___x_1292_);
v___x_1300_ = lean_box(0);
v_isShared_1301_ = v_isSharedCheck_1305_;
goto v_resetjp_1299_;
}
v_resetjp_1299_:
{
lean_object* v___x_1303_; 
if (v_isShared_1301_ == 0)
{
v___x_1303_ = v___x_1300_;
goto v_reusejp_1302_;
}
else
{
lean_object* v_reuseFailAlloc_1304_; 
v_reuseFailAlloc_1304_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1304_, 0, v_a_1297_);
lean_ctor_set(v_reuseFailAlloc_1304_, 1, v_a_1298_);
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
lean_object* v___x_1306_; uint8_t v___x_1307_; 
v___x_1306_ = l_Lean_Syntax_getArg(v___x_1289_, v___x_1222_);
v___x_1307_ = l_Lean_Syntax_matchesNull(v___x_1306_, v___x_1187_);
if (v___x_1307_ == 0)
{
lean_object* v___x_1308_; 
lean_dec(v___x_1289_);
lean_dec(v___x_1205_);
v___x_1308_ = l_Lean_Macro_throwUnsupported___redArg(v___y_1156_);
if (lean_obj_tag(v___x_1308_) == 0)
{
lean_object* v_a_1309_; lean_object* v___x_1311_; 
v_a_1309_ = lean_ctor_get(v___x_1308_, 1);
lean_inc(v_a_1309_);
lean_dec_ref_known(v___x_1308_, 2);
if (v_isShared_1169_ == 0)
{
v___x_1311_ = v___x_1168_;
goto v_reusejp_1310_;
}
else
{
lean_object* v_reuseFailAlloc_1312_; 
v_reuseFailAlloc_1312_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1312_, 0, v_fst_1165_);
lean_ctor_set(v_reuseFailAlloc_1312_, 1, v_snd_1166_);
v___x_1311_ = v_reuseFailAlloc_1312_;
goto v_reusejp_1310_;
}
v_reusejp_1310_:
{
v_a_1158_ = v___x_1311_;
v_a_1159_ = v_a_1309_;
goto v___jp_1157_;
}
}
else
{
lean_object* v_a_1313_; lean_object* v_a_1314_; lean_object* v___x_1316_; uint8_t v_isShared_1317_; uint8_t v_isSharedCheck_1321_; 
lean_del_object(v___x_1168_);
lean_dec(v_snd_1166_);
lean_dec(v_fst_1165_);
v_a_1313_ = lean_ctor_get(v___x_1308_, 0);
v_a_1314_ = lean_ctor_get(v___x_1308_, 1);
v_isSharedCheck_1321_ = !lean_is_exclusive(v___x_1308_);
if (v_isSharedCheck_1321_ == 0)
{
v___x_1316_ = v___x_1308_;
v_isShared_1317_ = v_isSharedCheck_1321_;
goto v_resetjp_1315_;
}
else
{
lean_inc(v_a_1314_);
lean_inc(v_a_1313_);
lean_dec(v___x_1308_);
v___x_1316_ = lean_box(0);
v_isShared_1317_ = v_isSharedCheck_1321_;
goto v_resetjp_1315_;
}
v_resetjp_1315_:
{
lean_object* v___x_1319_; 
if (v_isShared_1317_ == 0)
{
v___x_1319_ = v___x_1316_;
goto v_reusejp_1318_;
}
else
{
lean_object* v_reuseFailAlloc_1320_; 
v_reuseFailAlloc_1320_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1320_, 0, v_a_1313_);
lean_ctor_set(v_reuseFailAlloc_1320_, 1, v_a_1314_);
v___x_1319_ = v_reuseFailAlloc_1320_;
goto v_reusejp_1318_;
}
v_reusejp_1318_:
{
return v___x_1319_;
}
}
}
}
else
{
lean_object* v___x_1322_; lean_object* v___x_1323_; lean_object* v___x_1324_; uint8_t v___x_1325_; 
v___x_1322_ = l_Lean_TSyntax_getId(v___x_1205_);
lean_dec(v___x_1205_);
v___x_1323_ = l_Lean_Name_eraseMacroScopes(v___x_1322_);
lean_dec(v___x_1322_);
v___x_1324_ = ((lean_object*)(lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___closed__8));
v___x_1325_ = lean_name_eq(v___x_1323_, v___x_1324_);
lean_dec(v___x_1323_);
if (v___x_1325_ == 0)
{
lean_object* v___x_1326_; lean_object* v___x_1328_; 
lean_dec(v___x_1289_);
lean_inc(v_a_1170_);
v___x_1326_ = lean_array_push(v_snd_1166_, v_a_1170_);
if (v_isShared_1169_ == 0)
{
lean_ctor_set(v___x_1168_, 1, v___x_1326_);
v___x_1328_ = v___x_1168_;
goto v_reusejp_1327_;
}
else
{
lean_object* v_reuseFailAlloc_1329_; 
v_reuseFailAlloc_1329_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1329_, 0, v_fst_1165_);
lean_ctor_set(v_reuseFailAlloc_1329_, 1, v___x_1326_);
v___x_1328_ = v_reuseFailAlloc_1329_;
goto v_reusejp_1327_;
}
v_reusejp_1327_:
{
v_a_1158_ = v___x_1328_;
v_a_1159_ = v___y_1156_;
goto v___jp_1157_;
}
}
else
{
lean_object* v___x_1330_; lean_object* v___x_1331_; lean_object* v___x_1333_; 
v___x_1330_ = l_Lean_Syntax_getArg(v___x_1289_, v___x_1288_);
lean_dec(v___x_1289_);
v___x_1331_ = lean_array_push(v_fst_1165_, v___x_1330_);
if (v_isShared_1169_ == 0)
{
lean_ctor_set(v___x_1168_, 0, v___x_1331_);
v___x_1333_ = v___x_1168_;
goto v_reusejp_1332_;
}
else
{
lean_object* v_reuseFailAlloc_1334_; 
v_reuseFailAlloc_1334_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1334_, 0, v___x_1331_);
lean_ctor_set(v_reuseFailAlloc_1334_, 1, v_snd_1166_);
v___x_1333_ = v_reuseFailAlloc_1334_;
goto v_reusejp_1332_;
}
v_reusejp_1332_:
{
v_a_1158_ = v___x_1333_;
v_a_1159_ = v___y_1156_;
goto v___jp_1157_;
}
}
}
}
}
}
}
}
}
}
}
}
}
v___jp_1157_:
{
size_t v___x_1160_; size_t v___x_1161_; 
v___x_1160_ = ((size_t)1ULL);
v___x_1161_ = lean_usize_add(v_i_1154_, v___x_1160_);
v_i_1154_ = v___x_1161_;
v_b_1155_ = v_a_1158_;
v___y_1156_ = v_a_1159_;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg___boxed(lean_object* v_as_1336_, lean_object* v_sz_1337_, lean_object* v_i_1338_, lean_object* v_b_1339_, lean_object* v___y_1340_){
_start:
{
size_t v_sz_boxed_1341_; size_t v_i_boxed_1342_; lean_object* v_res_1343_; 
v_sz_boxed_1341_ = lean_unbox_usize(v_sz_1337_);
lean_dec(v_sz_1337_);
v_i_boxed_1342_ = lean_unbox_usize(v_i_1338_);
lean_dec(v_i_1338_);
v_res_1343_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg(v_as_1336_, v_sz_boxed_1341_, v_i_boxed_1342_, v_b_1339_, v___y_1340_);
lean_dec_ref(v_as_1336_);
return v_res_1343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1(lean_object* v_x_1368_, lean_object* v_a_1369_, lean_object* v_a_1370_){
_start:
{
lean_object* v___y_1372_; lean_object* v___x_1391_; uint8_t v___x_1392_; 
v___x_1391_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__1));
lean_inc(v_x_1368_);
v___x_1392_ = l_Lean_Syntax_isOfKind(v_x_1368_, v___x_1391_);
if (v___x_1392_ == 0)
{
lean_object* v___x_1393_; lean_object* v___x_1394_; 
lean_dec(v_x_1368_);
v___x_1393_ = lean_box(1);
v___x_1394_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1394_, 0, v___x_1393_);
lean_ctor_set(v___x_1394_, 1, v_a_1370_);
return v___x_1394_;
}
else
{
lean_object* v___x_1395_; lean_object* v___x_1396_; lean_object* v___y_1398_; size_t v___y_1399_; lean_object* v___y_1400_; lean_object* v___y_1401_; size_t v___y_1402_; lean_object* v___y_1403_; lean_object* v_ty_x3f_1404_; lean_object* v___y_1405_; lean_object* v___y_1406_; lean_object* v___y_1441_; lean_object* v___y_1442_; lean_object* v___y_1443_; lean_object* v___y_1444_; lean_object* v___y_1445_; lean_object* v___y_1446_; lean_object* v_srcs_1479_; lean_object* v___y_1480_; lean_object* v___y_1481_; lean_object* v___x_1504_; uint8_t v___x_1505_; 
v___x_1395_ = lean_unsigned_to_nat(0u);
v___x_1396_ = lean_unsigned_to_nat(1u);
v___x_1504_ = l_Lean_Syntax_getArg(v_x_1368_, v___x_1396_);
v___x_1505_ = l_Lean_Syntax_isNone(v___x_1504_);
if (v___x_1505_ == 0)
{
lean_object* v___x_1506_; uint8_t v___x_1507_; 
v___x_1506_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_1504_);
v___x_1507_ = l_Lean_Syntax_matchesNull(v___x_1504_, v___x_1506_);
if (v___x_1507_ == 0)
{
lean_object* v___x_1508_; lean_object* v___x_1509_; 
lean_dec(v___x_1504_);
lean_dec(v_x_1368_);
v___x_1508_ = lean_box(1);
v___x_1509_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1509_, 0, v___x_1508_);
lean_ctor_set(v___x_1509_, 1, v_a_1370_);
return v___x_1509_;
}
else
{
lean_object* v___x_1510_; lean_object* v_srcs_1511_; lean_object* v___x_1512_; 
v___x_1510_ = l_Lean_Syntax_getArg(v___x_1504_, v___x_1395_);
lean_dec(v___x_1504_);
v_srcs_1511_ = l_Lean_Syntax_getArgs(v___x_1510_);
lean_dec(v___x_1510_);
v___x_1512_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1512_, 0, v_srcs_1511_);
v_srcs_1479_ = v___x_1512_;
v___y_1480_ = v_a_1369_;
v___y_1481_ = v_a_1370_;
goto v___jp_1478_;
}
}
else
{
lean_object* v___x_1513_; 
lean_dec(v___x_1504_);
v___x_1513_ = lean_box(0);
v_srcs_1479_ = v___x_1513_;
v___y_1480_ = v_a_1369_;
v___y_1481_ = v_a_1370_;
goto v___jp_1478_;
}
v___jp_1397_:
{
lean_object* v___x_1407_; size_t v_sz_1408_; lean_object* v___x_1409_; 
v___x_1407_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__3));
v_sz_1408_ = lean_array_size(v___y_1403_);
v___x_1409_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg(v___y_1403_, v_sz_1408_, v___y_1402_, v___x_1407_, v___y_1406_);
lean_dec_ref(v___y_1403_);
if (lean_obj_tag(v___x_1409_) == 0)
{
lean_object* v_a_1410_; lean_object* v_a_1411_; lean_object* v_fst_1412_; lean_object* v_snd_1413_; lean_object* v___x_1414_; uint8_t v___x_1415_; 
v_a_1410_ = lean_ctor_get(v___x_1409_, 0);
lean_inc(v_a_1410_);
v_a_1411_ = lean_ctor_get(v___x_1409_, 1);
lean_inc(v_a_1411_);
lean_dec_ref_known(v___x_1409_, 2);
v_fst_1412_ = lean_ctor_get(v_a_1410_, 0);
lean_inc(v_fst_1412_);
v_snd_1413_ = lean_ctor_get(v_a_1410_, 1);
lean_inc(v_snd_1413_);
lean_dec(v_a_1410_);
v___x_1414_ = lean_array_get_size(v_fst_1412_);
v___x_1415_ = lean_nat_dec_eq(v___x_1414_, v___x_1395_);
if (v___x_1415_ == 0)
{
lean_object* v___x_1416_; lean_object* v___x_1417_; 
v___x_1416_ = lean_box(0);
lean_inc(v___y_1400_);
lean_inc(v___y_1398_);
v___x_1417_ = lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0(v_fst_1412_, v___x_1396_, v___y_1399_, v___x_1391_, v___x_1395_, v_snd_1413_, v___y_1398_, v___y_1400_, v_ty_x3f_1404_, v___y_1401_, v___x_1416_, v___y_1405_, v_a_1411_);
lean_dec(v___y_1401_);
v___y_1372_ = v___x_1417_;
goto v___jp_1371_;
}
else
{
lean_object* v___x_1418_; 
v___x_1418_ = l_Lean_Macro_throwUnsupported___redArg(v_a_1411_);
if (lean_obj_tag(v___x_1418_) == 0)
{
lean_object* v_a_1419_; lean_object* v_a_1420_; lean_object* v___x_1421_; 
v_a_1419_ = lean_ctor_get(v___x_1418_, 0);
lean_inc(v_a_1419_);
v_a_1420_ = lean_ctor_get(v___x_1418_, 1);
lean_inc(v_a_1420_);
lean_dec_ref_known(v___x_1418_, 2);
lean_inc(v___y_1400_);
lean_inc(v___y_1398_);
v___x_1421_ = lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___lam__0(v_fst_1412_, v___x_1396_, v___y_1399_, v___x_1391_, v___x_1395_, v_snd_1413_, v___y_1398_, v___y_1400_, v_ty_x3f_1404_, v___y_1401_, v_a_1419_, v___y_1405_, v_a_1420_);
lean_dec(v___y_1401_);
v___y_1372_ = v___x_1421_;
goto v___jp_1371_;
}
else
{
lean_object* v_a_1422_; lean_object* v_a_1423_; lean_object* v___x_1425_; uint8_t v_isShared_1426_; uint8_t v_isSharedCheck_1430_; 
lean_dec(v_snd_1413_);
lean_dec(v_fst_1412_);
lean_dec(v_ty_x3f_1404_);
lean_dec(v___y_1401_);
v_a_1422_ = lean_ctor_get(v___x_1418_, 0);
v_a_1423_ = lean_ctor_get(v___x_1418_, 1);
v_isSharedCheck_1430_ = !lean_is_exclusive(v___x_1418_);
if (v_isSharedCheck_1430_ == 0)
{
v___x_1425_ = v___x_1418_;
v_isShared_1426_ = v_isSharedCheck_1430_;
goto v_resetjp_1424_;
}
else
{
lean_inc(v_a_1423_);
lean_inc(v_a_1422_);
lean_dec(v___x_1418_);
v___x_1425_ = lean_box(0);
v_isShared_1426_ = v_isSharedCheck_1430_;
goto v_resetjp_1424_;
}
v_resetjp_1424_:
{
lean_object* v___x_1428_; 
if (v_isShared_1426_ == 0)
{
v___x_1428_ = v___x_1425_;
goto v_reusejp_1427_;
}
else
{
lean_object* v_reuseFailAlloc_1429_; 
v_reuseFailAlloc_1429_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1429_, 0, v_a_1422_);
lean_ctor_set(v_reuseFailAlloc_1429_, 1, v_a_1423_);
v___x_1428_ = v_reuseFailAlloc_1429_;
goto v_reusejp_1427_;
}
v_reusejp_1427_:
{
return v___x_1428_;
}
}
}
}
}
else
{
lean_object* v_a_1431_; lean_object* v_a_1432_; lean_object* v___x_1434_; uint8_t v_isShared_1435_; uint8_t v_isSharedCheck_1439_; 
lean_dec(v_ty_x3f_1404_);
lean_dec(v___y_1401_);
v_a_1431_ = lean_ctor_get(v___x_1409_, 0);
v_a_1432_ = lean_ctor_get(v___x_1409_, 1);
v_isSharedCheck_1439_ = !lean_is_exclusive(v___x_1409_);
if (v_isSharedCheck_1439_ == 0)
{
v___x_1434_ = v___x_1409_;
v_isShared_1435_ = v_isSharedCheck_1439_;
goto v_resetjp_1433_;
}
else
{
lean_inc(v_a_1432_);
lean_inc(v_a_1431_);
lean_dec(v___x_1409_);
v___x_1434_ = lean_box(0);
v_isShared_1435_ = v_isSharedCheck_1439_;
goto v_resetjp_1433_;
}
v_resetjp_1433_:
{
lean_object* v___x_1437_; 
if (v_isShared_1435_ == 0)
{
v___x_1437_ = v___x_1434_;
goto v_reusejp_1436_;
}
else
{
lean_object* v_reuseFailAlloc_1438_; 
v_reuseFailAlloc_1438_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1438_, 0, v_a_1431_);
lean_ctor_set(v_reuseFailAlloc_1438_, 1, v_a_1432_);
v___x_1437_ = v_reuseFailAlloc_1438_;
goto v_reusejp_1436_;
}
v_reusejp_1436_:
{
return v___x_1437_;
}
}
}
}
v___jp_1440_:
{
size_t v_sz_1447_; size_t v___x_1448_; lean_object* v___x_1449_; 
v_sz_1447_ = lean_array_size(v___y_1446_);
v___x_1448_ = ((size_t)0ULL);
v___x_1449_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__0(v_sz_1447_, v___x_1448_, v___y_1446_);
if (lean_obj_tag(v___x_1449_) == 0)
{
lean_object* v___x_1450_; lean_object* v___x_1451_; 
lean_dec(v___y_1442_);
lean_dec(v_x_1368_);
v___x_1450_ = lean_box(1);
v___x_1451_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1451_, 0, v___x_1450_);
lean_ctor_set(v___x_1451_, 1, v___y_1445_);
return v___x_1451_;
}
else
{
lean_object* v_val_1452_; lean_object* v___x_1454_; uint8_t v_isShared_1455_; uint8_t v_isSharedCheck_1477_; 
v_val_1452_ = lean_ctor_get(v___x_1449_, 0);
v_isSharedCheck_1477_ = !lean_is_exclusive(v___x_1449_);
if (v_isSharedCheck_1477_ == 0)
{
v___x_1454_ = v___x_1449_;
v_isShared_1455_ = v_isSharedCheck_1477_;
goto v_resetjp_1453_;
}
else
{
lean_inc(v_val_1452_);
lean_dec(v___x_1449_);
v___x_1454_ = lean_box(0);
v_isShared_1455_ = v_isSharedCheck_1477_;
goto v_resetjp_1453_;
}
v_resetjp_1453_:
{
lean_object* v___x_1456_; lean_object* v___x_1457_; lean_object* v___x_1458_; uint8_t v___x_1459_; 
v___x_1456_ = lean_unsigned_to_nat(3u);
v___x_1457_ = l_Lean_Syntax_getArg(v_x_1368_, v___x_1456_);
v___x_1458_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__5));
lean_inc(v___x_1457_);
v___x_1459_ = l_Lean_Syntax_isOfKind(v___x_1457_, v___x_1458_);
if (v___x_1459_ == 0)
{
lean_object* v___x_1460_; lean_object* v___x_1461_; 
lean_dec(v___x_1457_);
lean_del_object(v___x_1454_);
lean_dec(v_val_1452_);
lean_dec(v___y_1442_);
lean_dec(v_x_1368_);
v___x_1460_ = lean_box(1);
v___x_1461_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1461_, 0, v___x_1460_);
lean_ctor_set(v___x_1461_, 1, v___y_1445_);
return v___x_1461_;
}
else
{
lean_object* v___x_1462_; uint8_t v___x_1463_; 
v___x_1462_ = l_Lean_Syntax_getArg(v___x_1457_, v___x_1395_);
lean_dec(v___x_1457_);
v___x_1463_ = l_Lean_Syntax_matchesNull(v___x_1462_, v___x_1395_);
if (v___x_1463_ == 0)
{
lean_object* v___x_1464_; lean_object* v___x_1465_; 
lean_del_object(v___x_1454_);
lean_dec(v_val_1452_);
lean_dec(v___y_1442_);
lean_dec(v_x_1368_);
v___x_1464_ = lean_box(1);
v___x_1465_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1465_, 0, v___x_1464_);
lean_ctor_set(v___x_1465_, 1, v___y_1445_);
return v___x_1465_;
}
else
{
lean_object* v___x_1466_; lean_object* v___x_1467_; uint8_t v___x_1468_; 
v___x_1466_ = lean_unsigned_to_nat(4u);
v___x_1467_ = l_Lean_Syntax_getArg(v_x_1368_, v___x_1466_);
lean_dec(v_x_1368_);
v___x_1468_ = l_Lean_Syntax_isNone(v___x_1467_);
if (v___x_1468_ == 0)
{
uint8_t v___x_1469_; 
lean_inc(v___x_1467_);
v___x_1469_ = l_Lean_Syntax_matchesNull(v___x_1467_, v___y_1444_);
if (v___x_1469_ == 0)
{
lean_object* v___x_1470_; lean_object* v___x_1471_; 
lean_dec(v___x_1467_);
lean_del_object(v___x_1454_);
lean_dec(v_val_1452_);
lean_dec(v___y_1442_);
v___x_1470_ = lean_box(1);
v___x_1471_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1471_, 0, v___x_1470_);
lean_ctor_set(v___x_1471_, 1, v___y_1445_);
return v___x_1471_;
}
else
{
lean_object* v_ty_x3f_1472_; lean_object* v___x_1474_; 
v_ty_x3f_1472_ = l_Lean_Syntax_getArg(v___x_1467_, v___x_1396_);
lean_dec(v___x_1467_);
if (v_isShared_1455_ == 0)
{
lean_ctor_set(v___x_1454_, 0, v_ty_x3f_1472_);
v___x_1474_ = v___x_1454_;
goto v_reusejp_1473_;
}
else
{
lean_object* v_reuseFailAlloc_1475_; 
v_reuseFailAlloc_1475_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1475_, 0, v_ty_x3f_1472_);
v___x_1474_ = v_reuseFailAlloc_1475_;
goto v_reusejp_1473_;
}
v_reusejp_1473_:
{
v___y_1398_ = v___y_1441_;
v___y_1399_ = v___x_1448_;
v___y_1400_ = v___x_1458_;
v___y_1401_ = v___y_1442_;
v___y_1402_ = v___x_1448_;
v___y_1403_ = v_val_1452_;
v_ty_x3f_1404_ = v___x_1474_;
v___y_1405_ = v___y_1443_;
v___y_1406_ = v___y_1445_;
goto v___jp_1397_;
}
}
}
else
{
lean_object* v___x_1476_; 
lean_dec(v___x_1467_);
lean_del_object(v___x_1454_);
v___x_1476_ = lean_box(0);
v___y_1398_ = v___y_1441_;
v___y_1399_ = v___x_1448_;
v___y_1400_ = v___x_1458_;
v___y_1401_ = v___y_1442_;
v___y_1402_ = v___x_1448_;
v___y_1403_ = v_val_1452_;
v_ty_x3f_1404_ = v___x_1476_;
v___y_1405_ = v___y_1443_;
v___y_1406_ = v___y_1445_;
goto v___jp_1397_;
}
}
}
}
}
}
v___jp_1478_:
{
lean_object* v___x_1482_; lean_object* v___x_1483_; lean_object* v___x_1484_; uint8_t v___x_1485_; 
v___x_1482_ = lean_unsigned_to_nat(2u);
v___x_1483_ = l_Lean_Syntax_getArg(v_x_1368_, v___x_1482_);
v___x_1484_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__7));
lean_inc(v___x_1483_);
v___x_1485_ = l_Lean_Syntax_isOfKind(v___x_1483_, v___x_1484_);
if (v___x_1485_ == 0)
{
lean_object* v___x_1486_; lean_object* v___x_1487_; 
lean_dec(v___x_1483_);
lean_dec(v_srcs_1479_);
lean_dec(v_x_1368_);
v___x_1486_ = lean_box(1);
v___x_1487_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1487_, 0, v___x_1486_);
lean_ctor_set(v___x_1487_, 1, v___y_1481_);
return v___x_1487_;
}
else
{
lean_object* v___x_1488_; lean_object* v___x_1489_; lean_object* v___x_1490_; lean_object* v___x_1491_; uint8_t v___x_1492_; 
v___x_1488_ = l_Lean_Syntax_getArg(v___x_1483_, v___x_1395_);
lean_dec(v___x_1483_);
v___x_1489_ = l_Lean_Syntax_getArgs(v___x_1488_);
lean_dec(v___x_1488_);
v___x_1490_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___closed__8));
v___x_1491_ = lean_array_get_size(v___x_1489_);
v___x_1492_ = lean_nat_dec_lt(v___x_1395_, v___x_1491_);
if (v___x_1492_ == 0)
{
lean_dec_ref(v___x_1489_);
v___y_1441_ = v___x_1484_;
v___y_1442_ = v_srcs_1479_;
v___y_1443_ = v___y_1480_;
v___y_1444_ = v___x_1482_;
v___y_1445_ = v___y_1481_;
v___y_1446_ = v___x_1490_;
goto v___jp_1440_;
}
else
{
lean_object* v___x_1493_; lean_object* v___x_1494_; uint8_t v___x_1495_; 
v___x_1493_ = lean_box(v___x_1485_);
v___x_1494_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1494_, 0, v___x_1493_);
lean_ctor_set(v___x_1494_, 1, v___x_1490_);
v___x_1495_ = lean_nat_dec_le(v___x_1491_, v___x_1491_);
if (v___x_1495_ == 0)
{
if (v___x_1492_ == 0)
{
lean_dec_ref_known(v___x_1494_, 2);
lean_dec_ref(v___x_1489_);
v___y_1441_ = v___x_1484_;
v___y_1442_ = v_srcs_1479_;
v___y_1443_ = v___y_1480_;
v___y_1444_ = v___x_1482_;
v___y_1445_ = v___y_1481_;
v___y_1446_ = v___x_1490_;
goto v___jp_1440_;
}
else
{
size_t v___x_1496_; size_t v___x_1497_; lean_object* v___x_1498_; lean_object* v_snd_1499_; 
v___x_1496_ = ((size_t)0ULL);
v___x_1497_ = lean_usize_of_nat(v___x_1491_);
v___x_1498_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__7(v___x_1485_, v___x_1489_, v___x_1496_, v___x_1497_, v___x_1494_);
lean_dec_ref(v___x_1489_);
v_snd_1499_ = lean_ctor_get(v___x_1498_, 1);
lean_inc(v_snd_1499_);
lean_dec_ref(v___x_1498_);
v___y_1441_ = v___x_1484_;
v___y_1442_ = v_srcs_1479_;
v___y_1443_ = v___y_1480_;
v___y_1444_ = v___x_1482_;
v___y_1445_ = v___y_1481_;
v___y_1446_ = v_snd_1499_;
goto v___jp_1440_;
}
}
else
{
size_t v___x_1500_; size_t v___x_1501_; lean_object* v___x_1502_; lean_object* v_snd_1503_; 
v___x_1500_ = ((size_t)0ULL);
v___x_1501_ = lean_usize_of_nat(v___x_1491_);
v___x_1502_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__7(v___x_1485_, v___x_1489_, v___x_1500_, v___x_1501_, v___x_1494_);
lean_dec_ref(v___x_1489_);
v_snd_1503_ = lean_ctor_get(v___x_1502_, 1);
lean_inc(v_snd_1503_);
lean_dec_ref(v___x_1502_);
v___y_1441_ = v___x_1484_;
v___y_1442_ = v_srcs_1479_;
v___y_1443_ = v___y_1480_;
v___y_1444_ = v___x_1482_;
v___y_1445_ = v___y_1481_;
v___y_1446_ = v_snd_1503_;
goto v___jp_1440_;
}
}
}
}
}
v___jp_1371_:
{
if (lean_obj_tag(v___y_1372_) == 0)
{
lean_object* v_a_1373_; lean_object* v_a_1374_; lean_object* v___x_1376_; uint8_t v_isShared_1377_; uint8_t v_isSharedCheck_1381_; 
v_a_1373_ = lean_ctor_get(v___y_1372_, 0);
v_a_1374_ = lean_ctor_get(v___y_1372_, 1);
v_isSharedCheck_1381_ = !lean_is_exclusive(v___y_1372_);
if (v_isSharedCheck_1381_ == 0)
{
v___x_1376_ = v___y_1372_;
v_isShared_1377_ = v_isSharedCheck_1381_;
goto v_resetjp_1375_;
}
else
{
lean_inc(v_a_1374_);
lean_inc(v_a_1373_);
lean_dec(v___y_1372_);
v___x_1376_ = lean_box(0);
v_isShared_1377_ = v_isSharedCheck_1381_;
goto v_resetjp_1375_;
}
v_resetjp_1375_:
{
lean_object* v___x_1379_; 
if (v_isShared_1377_ == 0)
{
v___x_1379_ = v___x_1376_;
goto v_reusejp_1378_;
}
else
{
lean_object* v_reuseFailAlloc_1380_; 
v_reuseFailAlloc_1380_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1380_, 0, v_a_1373_);
lean_ctor_set(v_reuseFailAlloc_1380_, 1, v_a_1374_);
v___x_1379_ = v_reuseFailAlloc_1380_;
goto v_reusejp_1378_;
}
v_reusejp_1378_:
{
return v___x_1379_;
}
}
}
else
{
lean_object* v_a_1382_; lean_object* v_a_1383_; lean_object* v___x_1385_; uint8_t v_isShared_1386_; uint8_t v_isSharedCheck_1390_; 
v_a_1382_ = lean_ctor_get(v___y_1372_, 0);
v_a_1383_ = lean_ctor_get(v___y_1372_, 1);
v_isSharedCheck_1390_ = !lean_is_exclusive(v___y_1372_);
if (v_isSharedCheck_1390_ == 0)
{
v___x_1385_ = v___y_1372_;
v_isShared_1386_ = v_isSharedCheck_1390_;
goto v_resetjp_1384_;
}
else
{
lean_inc(v_a_1383_);
lean_inc(v_a_1382_);
lean_dec(v___y_1372_);
v___x_1385_ = lean_box(0);
v_isShared_1386_ = v_isSharedCheck_1390_;
goto v_resetjp_1384_;
}
v_resetjp_1384_:
{
lean_object* v___x_1388_; 
if (v_isShared_1386_ == 0)
{
v___x_1388_ = v___x_1385_;
goto v_reusejp_1387_;
}
else
{
lean_object* v_reuseFailAlloc_1389_; 
v_reuseFailAlloc_1389_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1389_, 0, v_a_1382_);
lean_ctor_set(v_reuseFailAlloc_1389_, 1, v_a_1383_);
v___x_1388_ = v_reuseFailAlloc_1389_;
goto v_reusejp_1387_;
}
v_reusejp_1387_:
{
return v___x_1388_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1___boxed(lean_object* v_x_1514_, lean_object* v_a_1515_, lean_object* v_a_1516_){
_start:
{
lean_object* v_res_1517_; 
v_res_1517_ = lp_mathlib___aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1(v_x_1514_, v_a_1515_, v_a_1516_);
lean_dec_ref(v_a_1515_);
return v_res_1517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1(lean_object* v_as_1518_, size_t v_sz_1519_, size_t v_i_1520_, lean_object* v_b_1521_, lean_object* v___y_1522_, lean_object* v___y_1523_){
_start:
{
lean_object* v___x_1524_; 
v___x_1524_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___redArg(v_as_1518_, v_sz_1519_, v_i_1520_, v_b_1521_, v___y_1523_);
return v___x_1524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1___boxed(lean_object* v_as_1525_, lean_object* v_sz_1526_, lean_object* v_i_1527_, lean_object* v_b_1528_, lean_object* v___y_1529_, lean_object* v___y_1530_){
_start:
{
size_t v_sz_boxed_1531_; size_t v_i_boxed_1532_; lean_object* v_res_1533_; 
v_sz_boxed_1531_ = lean_unbox_usize(v_sz_1526_);
lean_dec(v_sz_1526_);
v_i_boxed_1532_ = lean_unbox_usize(v_i_1527_);
lean_dec(v_i_1527_);
v_res_1533_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__1(v_as_1525_, v_sz_boxed_1531_, v_i_boxed_1532_, v_b_1528_, v___y_1529_, v___y_1530_);
lean_dec_ref(v___y_1529_);
lean_dec_ref(v_as_1525_);
return v_res_1533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2(lean_object* v_as_1534_, size_t v_sz_1535_, size_t v_i_1536_, lean_object* v_bs_1537_, lean_object* v___y_1538_, lean_object* v___y_1539_){
_start:
{
lean_object* v___x_1540_; 
v___x_1540_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2___redArg(v_sz_1535_, v_i_1536_, v_bs_1537_, v___y_1538_, v___y_1539_);
return v___x_1540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2___boxed(lean_object* v_as_1541_, lean_object* v_sz_1542_, lean_object* v_i_1543_, lean_object* v_bs_1544_, lean_object* v___y_1545_, lean_object* v___y_1546_){
_start:
{
size_t v_sz_boxed_1547_; size_t v_i_boxed_1548_; lean_object* v_res_1549_; 
v_sz_boxed_1547_ = lean_unbox_usize(v_sz_1542_);
lean_dec(v_sz_1542_);
v_i_boxed_1548_ = lean_unbox_usize(v_i_1543_);
lean_dec(v_i_1543_);
v_res_1549_ = lp_mathlib___private_Init_Data_Array_Basic_0__Array_mapFinIdxMUnsafe_map___at___00__aux__Mathlib__Tactic__Spread______macroRules__Lean__Parser__Term__structInst__1_spec__2(v_as_1541_, v_sz_boxed_1547_, v_i_boxed_1548_, v_bs_1544_, v___y_1545_, v___y_1546_);
lean_dec_ref(v___y_1545_);
lean_dec_ref(v_as_1541_);
return v_res_1549_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin) {
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
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Elab_Binders(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Elab_Binders(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Elab_Binders(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin) {
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
res = initialize_Lean_Elab_Binders(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Tactic_Spread(builtin);
}
#ifdef __cplusplus
}
#endif
