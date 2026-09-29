// Lean compiler output
// Module: Mathlib.Data.Fintype.Sets
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.BooleanAlgebra public import Mathlib.Data.Finset.SymmDiff public import Mathlib.Data.Fintype.OfMap
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
lean_object* lp_mathlib_Multiset_attach___redArg(lean_object*);
lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_subtype___redArg(lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableMem___aux__1___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Meta_mkFreshLevelMVar(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_mkSort(lean_object*);
lean_object* l_Lean_Meta_mkFreshExprMVar(lean_object*, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_whnfR(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabAppArgs(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_dedup___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_Quotation_precheck(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Set_toFinset___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_subtype___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Set_toFinset___redArg___closed__0 = (const lean_object*)&lp_mathlib_Set_toFinset___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Set_toFinset___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_toFinset(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemOfFintype___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemOfFintype___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemOfFintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemOfFintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_fintypeCoeSort___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_fintypeCoeSort(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_List_Subtype_fintype___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Subtype_fintype___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Subtype_fintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_List_Subtype_fintype(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Subtype_fintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Subtype_fintype(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Subtype_fintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Subtype_fintype(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinsetCoe_fintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinsetCoe_fintype(lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_Prop_fintype___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Prop_fintype___closed__0 = (const lean_object*)&lp_mathlib_Prop_fintype___closed__0_value;
static const lean_ctor_object lp_mathlib_Prop_fintype___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Prop_fintype___closed__0_value)}};
static const lean_object* lp_mathlib_Prop_fintype___closed__1 = (const lean_object*)&lp_mathlib_Prop_fintype___closed__1_value;
LEAN_EXPORT const lean_object* lp_mathlib_Prop_fintype = (const lean_object*)&lp_mathlib_Prop_fintype___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Subtype_fintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_fintype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_setFintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_setFintype(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_finsetStx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "finsetStx"};
static const lean_object* lp_mathlib_finsetStx___closed__0 = (const lean_object*)&lp_mathlib_finsetStx___closed__0_value;
static const lean_ctor_object lp_mathlib_finsetStx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_finsetStx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(244, 212, 197, 80, 197, 7, 34, 44)}};
static const lean_object* lp_mathlib_finsetStx___closed__1 = (const lean_object*)&lp_mathlib_finsetStx___closed__1_value;
static const lean_string_object lp_mathlib_finsetStx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_finsetStx___closed__2 = (const lean_object*)&lp_mathlib_finsetStx___closed__2_value;
static const lean_ctor_object lp_mathlib_finsetStx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_finsetStx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_finsetStx___closed__3 = (const lean_object*)&lp_mathlib_finsetStx___closed__3_value;
static const lean_string_object lp_mathlib_finsetStx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "finset% "};
static const lean_object* lp_mathlib_finsetStx___closed__4 = (const lean_object*)&lp_mathlib_finsetStx___closed__4_value;
static const lean_ctor_object lp_mathlib_finsetStx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_finsetStx___closed__4_value)}};
static const lean_object* lp_mathlib_finsetStx___closed__5 = (const lean_object*)&lp_mathlib_finsetStx___closed__5_value;
static const lean_string_object lp_mathlib_finsetStx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_finsetStx___closed__6 = (const lean_object*)&lp_mathlib_finsetStx___closed__6_value;
static const lean_ctor_object lp_mathlib_finsetStx___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_finsetStx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_finsetStx___closed__7 = (const lean_object*)&lp_mathlib_finsetStx___closed__7_value;
static const lean_ctor_object lp_mathlib_finsetStx___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_finsetStx___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_finsetStx___closed__8 = (const lean_object*)&lp_mathlib_finsetStx___closed__8_value;
static const lean_ctor_object lp_mathlib_finsetStx___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_finsetStx___closed__3_value),((lean_object*)&lp_mathlib_finsetStx___closed__5_value),((lean_object*)&lp_mathlib_finsetStx___closed__8_value)}};
static const lean_object* lp_mathlib_finsetStx___closed__9 = (const lean_object*)&lp_mathlib_finsetStx___closed__9_value;
static const lean_ctor_object lp_mathlib_finsetStx___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_finsetStx___closed__1_value),((lean_object*)(((size_t)(1022) << 1) | 1)),((lean_object*)&lp_mathlib_finsetStx___closed__9_value)}};
static const lean_object* lp_mathlib_finsetStx___closed__10 = (const lean_object*)&lp_mathlib_finsetStx___closed__10_value;
LEAN_EXPORT const lean_object* lp_mathlib_finsetStx = (const lean_object*)&lp_mathlib_finsetStx___closed__10_value;
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Finset"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Set"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__3_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "toFinset"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 214, 213, 227, 101, 196, 147, 255)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__5_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__4_value),LEAN_SCALAR_PTR_LITERAL(198, 172, 142, 183, 27, 231, 58, 108)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__5_value;
static const lean_array_object lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00precheckFinsetStx_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00precheckFinsetStx_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00precheckFinsetStx_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00precheckFinsetStx_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_precheckFinsetStx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_precheckFinsetStx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_toFinset___redArg(lean_object* v_inst_2_){
_start:
{
lean_object* v___f_3_; lean_object* v___x_4_; 
v___f_3_ = ((lean_object*)(lp_mathlib_Set_toFinset___redArg___closed__0));
v___x_4_ = lp_mathlib_Finset_map___redArg(v___f_3_, v_inst_2_);
return v___x_4_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_toFinset(lean_object* v_00_u03b1_5_, lean_object* v_s_6_, lean_object* v_inst_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_Set_toFinset___redArg(v_inst_7_);
return v___x_8_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemOfFintype___redArg(lean_object* v_inst_9_, lean_object* v_inst_10_, lean_object* v_a_11_){
_start:
{
lean_object* v___x_12_; uint8_t v___x_13_; 
v___x_12_ = lp_mathlib_Set_toFinset___redArg(v_inst_10_);
v___x_13_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v_inst_9_, v_a_11_, v___x_12_);
return v___x_13_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemOfFintype___redArg___boxed(lean_object* v_inst_14_, lean_object* v_inst_15_, lean_object* v_a_16_){
_start:
{
uint8_t v_res_17_; lean_object* v_r_18_; 
v_res_17_ = lp_mathlib_Set_decidableMemOfFintype___redArg(v_inst_14_, v_inst_15_, v_a_16_);
v_r_18_ = lean_box(v_res_17_);
return v_r_18_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Set_decidableMemOfFintype(lean_object* v_00_u03b1_19_, lean_object* v_inst_20_, lean_object* v_s_21_, lean_object* v_inst_22_, lean_object* v_a_23_){
_start:
{
uint8_t v___x_24_; 
v___x_24_ = lp_mathlib_Set_decidableMemOfFintype___redArg(v_inst_20_, v_inst_22_, v_a_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_decidableMemOfFintype___boxed(lean_object* v_00_u03b1_25_, lean_object* v_inst_26_, lean_object* v_s_27_, lean_object* v_inst_28_, lean_object* v_a_29_){
_start:
{
uint8_t v_res_30_; lean_object* v_r_31_; 
v_res_30_ = lp_mathlib_Set_decidableMemOfFintype(v_00_u03b1_25_, v_inst_26_, v_s_27_, v_inst_28_, v_a_29_);
v_r_31_ = lean_box(v_res_30_);
return v_r_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_fintypeCoeSort___redArg(lean_object* v_s_32_){
_start:
{
lean_object* v___x_33_; 
v___x_33_ = lp_mathlib_Multiset_attach___redArg(v_s_32_);
return v___x_33_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_fintypeCoeSort(lean_object* v_00_u03b1_34_, lean_object* v_s_35_){
_start:
{
lean_object* v___x_36_; 
v___x_36_ = lp_mathlib_Multiset_attach___redArg(v_s_35_);
return v___x_36_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_List_Subtype_fintype___redArg___lam__0(lean_object* v_inst_37_, lean_object* v_a_38_, lean_object* v_b_39_){
_start:
{
lean_object* v___x_40_; uint8_t v___x_41_; 
v___x_40_ = lean_apply_2(v_inst_37_, v_a_38_, v_b_39_);
v___x_41_ = lean_unbox(v___x_40_);
return v___x_41_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Subtype_fintype___redArg___lam__0___boxed(lean_object* v_inst_42_, lean_object* v_a_43_, lean_object* v_b_44_){
_start:
{
uint8_t v_res_45_; lean_object* v_r_46_; 
v_res_45_ = lp_mathlib_List_Subtype_fintype___redArg___lam__0(v_inst_42_, v_a_43_, v_b_44_);
v_r_46_ = lean_box(v_res_45_);
return v_r_46_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Subtype_fintype___redArg(lean_object* v_inst_47_, lean_object* v_l_48_){
_start:
{
lean_object* v___f_49_; lean_object* v___x_50_; 
v___f_49_ = lean_alloc_closure((void*)(lp_mathlib_List_Subtype_fintype___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_49_, 0, v_inst_47_);
v___x_50_ = lp_mathlib_List_dedup___redArg(v___f_49_, v_l_48_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_List_Subtype_fintype(lean_object* v_00_u03b1_51_, lean_object* v_inst_52_, lean_object* v_l_53_){
_start:
{
lean_object* v___x_54_; 
v___x_54_ = lp_mathlib_List_Subtype_fintype___redArg(v_inst_52_, v_l_53_);
return v___x_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Subtype_fintype___redArg(lean_object* v_inst_55_, lean_object* v_s_56_){
_start:
{
lean_object* v___f_57_; lean_object* v___x_58_; lean_object* v___x_59_; 
v___f_57_ = lean_alloc_closure((void*)(lp_mathlib_List_Subtype_fintype___redArg___lam__0___boxed), 3, 1);
lean_closure_set(v___f_57_, 0, v_inst_55_);
v___x_58_ = lp_mathlib_Multiset_attach___redArg(v_s_56_);
v___x_59_ = lp_mathlib_List_dedup___redArg(v___f_57_, v___x_58_);
return v___x_59_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_Subtype_fintype(lean_object* v_00_u03b1_60_, lean_object* v_inst_61_, lean_object* v_s_62_){
_start:
{
lean_object* v___x_63_; 
v___x_63_ = lp_mathlib_Multiset_Subtype_fintype___redArg(v_inst_61_, v_s_62_);
return v___x_63_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Subtype_fintype___redArg(lean_object* v_s_64_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lp_mathlib_Multiset_attach___redArg(v_s_64_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Subtype_fintype(lean_object* v_00_u03b1_66_, lean_object* v_s_67_){
_start:
{
lean_object* v___x_68_; 
v___x_68_ = lp_mathlib_Multiset_attach___redArg(v_s_67_);
return v___x_68_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinsetCoe_fintype___redArg(lean_object* v_s_69_){
_start:
{
lean_object* v___x_70_; 
v___x_70_ = lp_mathlib_Multiset_attach___redArg(v_s_69_);
return v___x_70_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinsetCoe_fintype(lean_object* v_00_u03b1_71_, lean_object* v_s_72_){
_start:
{
lean_object* v___x_73_; 
v___x_73_ = lp_mathlib_Multiset_attach___redArg(v_s_72_);
return v___x_73_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_fintype___redArg(lean_object* v_inst_79_, lean_object* v_inst_80_){
_start:
{
lean_object* v___x_81_; lean_object* v___x_82_; 
v___x_81_ = lp_mathlib_Multiset_filter___redArg(v_inst_79_, v_inst_80_);
v___x_82_ = lp_mathlib_Fintype_subtype___redArg(v___x_81_);
return v___x_82_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_fintype(lean_object* v_00_u03b1_83_, lean_object* v_p_84_, lean_object* v_inst_85_, lean_object* v_inst_86_){
_start:
{
lean_object* v___x_87_; 
v___x_87_ = lp_mathlib_Subtype_fintype___redArg(v_inst_85_, v_inst_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_setFintype___redArg(lean_object* v_inst_88_, lean_object* v_inst_89_){
_start:
{
lean_object* v___x_90_; 
v___x_90_ = lp_mathlib_Subtype_fintype___redArg(v_inst_89_, v_inst_88_);
return v___x_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_setFintype(lean_object* v_00_u03b1_91_, lean_object* v_inst_92_, lean_object* v_s_93_, lean_object* v_inst_94_){
_start:
{
lean_object* v___x_95_; 
v___x_95_ = lp_mathlib_Subtype_fintype___redArg(v_inst_94_, v_inst_92_);
return v___x_95_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; 
v___x_120_ = lean_box(0);
v___x_121_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_122_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_122_, 0, v___x_121_);
lean_ctor_set(v___x_122_, 1, v___x_120_);
return v___x_122_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___redArg(){
_start:
{
lean_object* v___x_124_; lean_object* v___x_125_; 
v___x_124_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___redArg___closed__0);
v___x_125_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_125_, 0, v___x_124_);
return v___x_125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___redArg___boxed(lean_object* v___y_126_){
_start:
{
lean_object* v_res_127_; 
v_res_127_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___redArg();
return v_res_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0(lean_object* v_00_u03b1_128_, lean_object* v___y_129_, lean_object* v___y_130_, lean_object* v___y_131_, lean_object* v___y_132_, lean_object* v___y_133_, lean_object* v___y_134_){
_start:
{
lean_object* v___x_136_; 
v___x_136_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___redArg();
return v___x_136_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___boxed(lean_object* v_00_u03b1_137_, lean_object* v___y_138_, lean_object* v___y_139_, lean_object* v___y_140_, lean_object* v___y_141_, lean_object* v___y_142_, lean_object* v___y_143_, lean_object* v___y_144_){
_start:
{
lean_object* v_res_145_; 
v_res_145_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0(v_00_u03b1_137_, v___y_138_, v___y_139_, v___y_140_, v___y_141_, v___y_142_, v___y_143_);
lean_dec(v___y_143_);
lean_dec_ref(v___y_142_);
lean_dec(v___y_141_);
lean_dec_ref(v___y_140_);
lean_dec(v___y_139_);
lean_dec_ref(v___y_138_);
return v_res_145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg(lean_object* v_stx_158_, lean_object* v_a_159_, lean_object* v_a_160_, lean_object* v_a_161_, lean_object* v_a_162_, lean_object* v_a_163_, lean_object* v_a_164_){
_start:
{
lean_object* v___x_166_; uint8_t v___x_167_; 
v___x_166_ = ((lean_object*)(lp_mathlib_finsetStx___closed__1));
lean_inc(v_stx_158_);
v___x_167_ = l_Lean_Syntax_isOfKind(v_stx_158_, v___x_166_);
if (v___x_167_ == 0)
{
lean_object* v___x_168_; 
lean_dec(v_stx_158_);
v___x_168_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___redArg();
return v___x_168_;
}
else
{
lean_object* v___x_169_; 
v___x_169_ = l_Lean_Meta_mkFreshLevelMVar(v_a_161_, v_a_162_, v_a_163_, v_a_164_);
if (lean_obj_tag(v___x_169_) == 0)
{
lean_object* v_a_170_; lean_object* v___x_171_; lean_object* v___x_172_; lean_object* v___x_173_; uint8_t v___x_174_; lean_object* v___x_175_; lean_object* v___x_176_; 
v_a_170_ = lean_ctor_get(v___x_169_, 0);
lean_inc_n(v_a_170_, 2);
lean_dec_ref_known(v___x_169_, 1);
v___x_171_ = l_Lean_Level_succ___override(v_a_170_);
v___x_172_ = l_Lean_mkSort(v___x_171_);
v___x_173_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_173_, 0, v___x_172_);
v___x_174_ = 0;
v___x_175_ = lean_box(0);
v___x_176_ = l_Lean_Meta_mkFreshExprMVar(v___x_173_, v___x_174_, v___x_175_, v_a_161_, v_a_162_, v_a_163_, v_a_164_);
if (lean_obj_tag(v___x_176_) == 0)
{
lean_object* v_a_177_; lean_object* v___x_179_; uint8_t v_isShared_180_; uint8_t v_isSharedCheck_221_; 
v_a_177_ = lean_ctor_get(v___x_176_, 0);
v_isSharedCheck_221_ = !lean_is_exclusive(v___x_176_);
if (v_isSharedCheck_221_ == 0)
{
v___x_179_ = v___x_176_;
v_isShared_180_ = v_isSharedCheck_221_;
goto v_resetjp_178_;
}
else
{
lean_inc(v_a_177_);
lean_dec(v___x_176_);
v___x_179_ = lean_box(0);
v_isShared_180_ = v_isSharedCheck_221_;
goto v_resetjp_178_;
}
v_resetjp_178_:
{
lean_object* v___x_181_; lean_object* v___x_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_185_; lean_object* v___x_186_; lean_object* v___x_187_; lean_object* v___x_189_; 
v___x_181_ = lean_unsigned_to_nat(1u);
v___x_182_ = l_Lean_Syntax_getArg(v_stx_158_, v___x_181_);
lean_dec(v_stx_158_);
v___x_183_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__1));
v___x_184_ = lean_box(0);
v___x_185_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_185_, 0, v_a_170_);
lean_ctor_set(v___x_185_, 1, v___x_184_);
lean_inc_ref(v___x_185_);
v___x_186_ = l_Lean_Expr_const___override(v___x_183_, v___x_185_);
v___x_187_ = l_Lean_Expr_app___override(v___x_186_, v_a_177_);
if (v_isShared_180_ == 0)
{
lean_ctor_set_tag(v___x_179_, 1);
lean_ctor_set(v___x_179_, 0, v___x_187_);
v___x_189_ = v___x_179_;
goto v_reusejp_188_;
}
else
{
lean_object* v_reuseFailAlloc_220_; 
v_reuseFailAlloc_220_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_220_, 0, v___x_187_);
v___x_189_ = v_reuseFailAlloc_220_;
goto v_reusejp_188_;
}
v_reusejp_188_:
{
lean_object* v___x_190_; 
v___x_190_ = l_Lean_Elab_Term_elabTerm(v___x_182_, v___x_189_, v___x_167_, v___x_167_, v_a_159_, v_a_160_, v_a_161_, v_a_162_, v_a_163_, v_a_164_);
if (lean_obj_tag(v___x_190_) == 0)
{
lean_object* v_a_191_; lean_object* v___x_192_; 
v_a_191_ = lean_ctor_get(v___x_190_, 0);
lean_inc_n(v_a_191_, 2);
lean_dec_ref_known(v___x_190_, 1);
lean_inc(v_a_164_);
lean_inc_ref(v_a_163_);
lean_inc(v_a_162_);
lean_inc_ref(v_a_161_);
v___x_192_ = lean_infer_type(v_a_191_, v_a_161_, v_a_162_, v_a_163_, v_a_164_);
if (lean_obj_tag(v___x_192_) == 0)
{
lean_object* v_a_193_; lean_object* v___x_195_; uint8_t v_isShared_196_; uint8_t v_isSharedCheck_219_; 
v_a_193_ = lean_ctor_get(v___x_192_, 0);
v_isSharedCheck_219_ = !lean_is_exclusive(v___x_192_);
if (v_isSharedCheck_219_ == 0)
{
v___x_195_ = v___x_192_;
v_isShared_196_ = v_isSharedCheck_219_;
goto v_resetjp_194_;
}
else
{
lean_inc(v_a_193_);
lean_dec(v___x_192_);
v___x_195_ = lean_box(0);
v_isShared_196_ = v_isSharedCheck_219_;
goto v_resetjp_194_;
}
v_resetjp_194_:
{
lean_object* v___x_197_; 
v___x_197_ = l_Lean_Meta_whnfR(v_a_193_, v_a_161_, v_a_162_, v_a_163_, v_a_164_);
if (lean_obj_tag(v___x_197_) == 0)
{
lean_object* v_a_198_; lean_object* v___x_200_; uint8_t v_isShared_201_; uint8_t v_isSharedCheck_218_; 
v_a_198_ = lean_ctor_get(v___x_197_, 0);
v_isSharedCheck_218_ = !lean_is_exclusive(v___x_197_);
if (v_isSharedCheck_218_ == 0)
{
v___x_200_ = v___x_197_;
v_isShared_201_ = v_isSharedCheck_218_;
goto v_resetjp_199_;
}
else
{
lean_inc(v_a_198_);
lean_dec(v___x_197_);
v___x_200_ = lean_box(0);
v_isShared_201_ = v_isSharedCheck_218_;
goto v_resetjp_199_;
}
v_resetjp_199_:
{
lean_object* v___x_202_; uint8_t v___x_203_; 
v___x_202_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__3));
v___x_203_ = l_Lean_Expr_isAppOfArity(v_a_198_, v___x_202_, v___x_181_);
lean_dec(v_a_198_);
if (v___x_203_ == 0)
{
lean_object* v___x_205_; 
lean_del_object(v___x_195_);
lean_dec_ref_known(v___x_185_, 2);
if (v_isShared_201_ == 0)
{
lean_ctor_set(v___x_200_, 0, v_a_191_);
v___x_205_ = v___x_200_;
goto v_reusejp_204_;
}
else
{
lean_object* v_reuseFailAlloc_206_; 
v_reuseFailAlloc_206_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_206_, 0, v_a_191_);
v___x_205_ = v_reuseFailAlloc_206_;
goto v_reusejp_204_;
}
v_reusejp_204_:
{
return v___x_205_;
}
}
else
{
lean_object* v___x_207_; lean_object* v___x_208_; lean_object* v___x_209_; lean_object* v___x_211_; 
lean_del_object(v___x_200_);
v___x_207_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__5));
v___x_208_ = l_Lean_Expr_const___override(v___x_207_, v___x_185_);
v___x_209_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___closed__6));
if (v_isShared_196_ == 0)
{
lean_ctor_set_tag(v___x_195_, 1);
lean_ctor_set(v___x_195_, 0, v_a_191_);
v___x_211_ = v___x_195_;
goto v_reusejp_210_;
}
else
{
lean_object* v_reuseFailAlloc_217_; 
v_reuseFailAlloc_217_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_217_, 0, v_a_191_);
v___x_211_ = v_reuseFailAlloc_217_;
goto v_reusejp_210_;
}
v_reusejp_210_:
{
lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; uint8_t v___x_215_; lean_object* v___x_216_; 
v___x_212_ = lean_mk_empty_array_with_capacity(v___x_181_);
v___x_213_ = lean_array_push(v___x_212_, v___x_211_);
v___x_214_ = lean_box(0);
v___x_215_ = 0;
v___x_216_ = l_Lean_Elab_Term_elabAppArgs(v___x_208_, v___x_209_, v___x_213_, v___x_214_, v___x_215_, v___x_215_, v___x_167_, v_a_159_, v_a_160_, v_a_161_, v_a_162_, v_a_163_, v_a_164_);
return v___x_216_;
}
}
}
}
else
{
lean_del_object(v___x_195_);
lean_dec(v_a_191_);
lean_dec_ref_known(v___x_185_, 2);
return v___x_197_;
}
}
}
else
{
lean_dec(v_a_191_);
lean_dec_ref_known(v___x_185_, 2);
return v___x_192_;
}
}
else
{
lean_dec_ref_known(v___x_185_, 2);
return v___x_190_;
}
}
}
}
else
{
lean_dec(v_a_170_);
lean_dec(v_stx_158_);
return v___x_176_;
}
}
else
{
lean_object* v_a_222_; lean_object* v___x_224_; uint8_t v_isShared_225_; uint8_t v_isSharedCheck_229_; 
lean_dec(v_stx_158_);
v_a_222_ = lean_ctor_get(v___x_169_, 0);
v_isSharedCheck_229_ = !lean_is_exclusive(v___x_169_);
if (v_isSharedCheck_229_ == 0)
{
v___x_224_ = v___x_169_;
v_isShared_225_ = v_isSharedCheck_229_;
goto v_resetjp_223_;
}
else
{
lean_inc(v_a_222_);
lean_dec(v___x_169_);
v___x_224_ = lean_box(0);
v_isShared_225_ = v_isSharedCheck_229_;
goto v_resetjp_223_;
}
v_resetjp_223_:
{
lean_object* v___x_227_; 
if (v_isShared_225_ == 0)
{
v___x_227_ = v___x_224_;
goto v_reusejp_226_;
}
else
{
lean_object* v_reuseFailAlloc_228_; 
v_reuseFailAlloc_228_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_228_, 0, v_a_222_);
v___x_227_ = v_reuseFailAlloc_228_;
goto v_reusejp_226_;
}
v_reusejp_226_:
{
return v___x_227_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg___boxed(lean_object* v_stx_230_, lean_object* v_a_231_, lean_object* v_a_232_, lean_object* v_a_233_, lean_object* v_a_234_, lean_object* v_a_235_, lean_object* v_a_236_, lean_object* v_a_237_){
_start:
{
lean_object* v_res_238_; 
v_res_238_ = lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg(v_stx_230_, v_a_231_, v_a_232_, v_a_233_, v_a_234_, v_a_235_, v_a_236_);
lean_dec(v_a_236_);
lean_dec_ref(v_a_235_);
lean_dec(v_a_234_);
lean_dec_ref(v_a_233_);
lean_dec(v_a_232_);
lean_dec_ref(v_a_231_);
return v_res_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1(lean_object* v_stx_239_, lean_object* v_x_240_, lean_object* v_a_241_, lean_object* v_a_242_, lean_object* v_a_243_, lean_object* v_a_244_, lean_object* v_a_245_, lean_object* v_a_246_){
_start:
{
lean_object* v___x_248_; 
v___x_248_ = lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___redArg(v_stx_239_, v_a_241_, v_a_242_, v_a_243_, v_a_244_, v_a_245_, v_a_246_);
return v___x_248_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1___boxed(lean_object* v_stx_249_, lean_object* v_x_250_, lean_object* v_a_251_, lean_object* v_a_252_, lean_object* v_a_253_, lean_object* v_a_254_, lean_object* v_a_255_, lean_object* v_a_256_, lean_object* v_a_257_){
_start:
{
lean_object* v_res_258_; 
v_res_258_ = lp_mathlib___aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1(v_stx_249_, v_x_250_, v_a_251_, v_a_252_, v_a_253_, v_a_254_, v_a_255_, v_a_256_);
lean_dec(v_a_256_);
lean_dec_ref(v_a_255_);
lean_dec(v_a_254_);
lean_dec_ref(v_a_253_);
lean_dec(v_a_252_);
lean_dec_ref(v_a_251_);
lean_dec(v_x_250_);
return v_res_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00precheckFinsetStx_spec__0___redArg(){
_start:
{
lean_object* v___x_260_; lean_object* v___x_261_; 
v___x_260_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00__aux__Mathlib__Data__Fintype__Sets______elabRules__finsetStx__1_spec__0___redArg___closed__0);
v___x_261_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_261_, 0, v___x_260_);
return v___x_261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00precheckFinsetStx_spec__0___redArg___boxed(lean_object* v___y_262_){
_start:
{
lean_object* v_res_263_; 
v_res_263_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00precheckFinsetStx_spec__0___redArg();
return v_res_263_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00precheckFinsetStx_spec__0(lean_object* v_00_u03b1_264_, lean_object* v___y_265_, lean_object* v___y_266_, lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_){
_start:
{
lean_object* v___x_273_; 
v___x_273_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00precheckFinsetStx_spec__0___redArg();
return v___x_273_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00precheckFinsetStx_spec__0___boxed(lean_object* v_00_u03b1_274_, lean_object* v___y_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_, lean_object* v___y_282_){
_start:
{
lean_object* v_res_283_; 
v_res_283_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00precheckFinsetStx_spec__0(v_00_u03b1_274_, v___y_275_, v___y_276_, v___y_277_, v___y_278_, v___y_279_, v___y_280_, v___y_281_);
lean_dec(v___y_281_);
lean_dec_ref(v___y_280_);
lean_dec(v___y_279_);
lean_dec_ref(v___y_278_);
lean_dec(v___y_277_);
lean_dec_ref(v___y_276_);
lean_dec(v___y_275_);
return v_res_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_precheckFinsetStx(lean_object* v_x_284_, lean_object* v_a_285_, lean_object* v_a_286_, lean_object* v_a_287_, lean_object* v_a_288_, lean_object* v_a_289_, lean_object* v_a_290_, lean_object* v_a_291_){
_start:
{
lean_object* v___x_293_; uint8_t v___x_294_; 
v___x_293_ = ((lean_object*)(lp_mathlib_finsetStx___closed__1));
lean_inc(v_x_284_);
v___x_294_ = l_Lean_Syntax_isOfKind(v_x_284_, v___x_293_);
if (v___x_294_ == 0)
{
lean_object* v___x_295_; 
lean_dec(v_x_284_);
v___x_295_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00precheckFinsetStx_spec__0___redArg();
return v___x_295_;
}
else
{
lean_object* v___x_296_; lean_object* v___x_297_; lean_object* v___x_298_; 
v___x_296_ = lean_unsigned_to_nat(1u);
v___x_297_ = l_Lean_Syntax_getArg(v_x_284_, v___x_296_);
lean_dec(v_x_284_);
v___x_298_ = l_Lean_Elab_Term_Quotation_precheck(v___x_297_, v_a_285_, v_a_286_, v_a_287_, v_a_288_, v_a_289_, v_a_290_, v_a_291_);
return v___x_298_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_precheckFinsetStx___boxed(lean_object* v_x_299_, lean_object* v_a_300_, lean_object* v_a_301_, lean_object* v_a_302_, lean_object* v_a_303_, lean_object* v_a_304_, lean_object* v_a_305_, lean_object* v_a_306_, lean_object* v_a_307_){
_start:
{
lean_object* v_res_308_; 
v_res_308_ = lp_mathlib_precheckFinsetStx(v_x_299_, v_a_300_, v_a_301_, v_a_302_, v_a_303_, v_a_304_, v_a_305_, v_a_306_);
lean_dec(v_a_306_);
lean_dec_ref(v_a_305_);
lean_dec(v_a_304_);
lean_dec_ref(v_a_303_);
lean_dec(v_a_302_);
lean_dec_ref(v_a_301_);
lean_dec(v_a_300_);
return v_res_308_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_SymmDiff(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_OfMap(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Fintype_Sets(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_SymmDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_OfMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Fintype_Sets(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_SymmDiff(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_OfMap(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Sets(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_BooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_SymmDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_OfMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Fintype_Sets(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Fintype_Sets(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Fintype_Sets(builtin);
}
#ifdef __cplusplus
}
#endif
