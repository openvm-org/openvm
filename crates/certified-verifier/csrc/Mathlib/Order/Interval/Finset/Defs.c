// Lean compiler output
// Module: Mathlib.Order.Interval.Finset.Defs
// Imports: public import Init public meta import Init public import Mathlib.Data.Finset.Preimage public import Mathlib.Data.Finset.Prod public import Mathlib.Order.Hom.WithTopBot public import Mathlib.Order.Interval.Set.UnorderedInterval
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
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
extern lean_object* l_Lean_Elab_unsupportedSyntaxExceptionId;
lean_object* lp_mathlib_Fintype_subtype___redArg(lean_object*);
lean_object* lp_mathlib_Finset_subtype___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_coeWithTop___lam__0(lean_object*);
lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg___lam__0(lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* lp_mathlib_Multiset_product___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_swap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_fintype___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Set_toFinset___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Term_elabTerm(lean_object*, lean_object*, uint8_t, uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_mk_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_mk_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_mk_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder___lam__0___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_IsEmpty_toLocallyFiniteOrder___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_IsEmpty_toLocallyFiniteOrder___lam__0___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder___closed__0 = (const lean_object*)&lp_mathlib_IsEmpty_toLocallyFiniteOrder___closed__0_value;
static const lean_ctor_object lp_mathlib_IsEmpty_toLocallyFiniteOrder___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_IsEmpty_toLocallyFiniteOrder___closed__0_value),((lean_object*)&lp_mathlib_IsEmpty_toLocallyFiniteOrder___closed__0_value),((lean_object*)&lp_mathlib_IsEmpty_toLocallyFiniteOrder___closed__0_value),((lean_object*)&lp_mathlib_IsEmpty_toLocallyFiniteOrder___closed__0_value)}};
static const lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder___closed__1 = (const lean_object*)&lp_mathlib_IsEmpty_toLocallyFiniteOrder___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___lam__0___boxed(lean_object*);
static const lean_closure_object lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___closed__0 = (const lean_object*)&lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___closed__0_value;
static const lean_ctor_object lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___closed__0_value),((lean_object*)&lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___closed__0_value)}};
static const lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___closed__1 = (const lean_object*)&lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___closed__0_value),((lean_object*)&lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___closed__0_value)}};
static const lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___closed__0 = (const lean_object*)&lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderBot(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Icc___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Icc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Icc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ico___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ico(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ico___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioc___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioo___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ici___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ici(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ici___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iic___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iic(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iic___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioi___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioi(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioi___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iio___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iio(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iio___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_uIcc___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_uIcc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "FinsetInterval"};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__0 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__0_value;
static const lean_string_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "term[[_,_]]"};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__1 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__1_value;
static const lean_ctor_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__0_value),LEAN_SCALAR_PTR_LITERAL(120, 229, 140, 158, 53, 86, 112, 175)}};
static const lean_ctor_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__2_value_aux_0),((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__1_value),LEAN_SCALAR_PTR_LITERAL(4, 5, 183, 103, 187, 255, 97, 87)}};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__2 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__2_value;
static const lean_string_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__3 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__3_value;
static const lean_ctor_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__4 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__4_value;
static const lean_string_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "[["};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__5 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__5_value;
static const lean_ctor_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__5_value)}};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__6 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__6_value;
static const lean_string_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__7 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__7_value;
static const lean_ctor_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__8 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__8_value;
static const lean_ctor_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__9 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__9_value;
static const lean_ctor_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__4_value),((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__6_value),((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__9_value)}};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__10 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__10_value;
static const lean_string_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__11 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__11_value;
static const lean_ctor_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__11_value)}};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__12 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__12_value;
static const lean_ctor_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__4_value),((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__10_value),((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__12_value)}};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__13 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__13_value;
static const lean_ctor_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__4_value),((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__13_value),((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__9_value)}};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__14 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__14_value;
static const lean_string_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "]]"};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__15 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__15_value;
static const lean_ctor_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__15_value)}};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__16 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__16_value;
static const lean_ctor_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__4_value),((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__14_value),((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__16_value)}};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__17 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__17_value;
static const lean_ctor_object lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__2_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__17_value)}};
static const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__18 = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__18_value;
LEAN_EXPORT const lean_object* lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d = (const lean_object*)&lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__18_value;
static const lean_string_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0 = (const lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value;
static const lean_string_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1 = (const lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1_value;
static const lean_string_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2 = (const lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2_value;
static const lean_string_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__3 = (const lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__3_value;
static const lean_ctor_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4 = (const lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4_value;
static const lean_string_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "Finset.uIcc"};
static const lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__5 = (const lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__5_value;
static lean_once_cell_t lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6;
static const lean_string_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Finset"};
static const lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7 = (const lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7_value;
static const lean_string_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "uIcc"};
static const lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__8 = (const lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__8_value;
static const lean_ctor_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9_value_aux_0),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(36, 231, 39, 86, 39, 124, 235, 245)}};
static const lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9 = (const lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9_value;
static const lean_ctor_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__10 = (const lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__10_value;
static const lean_ctor_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__11 = (const lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__11_value;
static const lean_string_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__12 = (const lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__12_value;
static const lean_ctor_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13 = (const lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13_value;
LEAN_EXPORT lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__0 = (const lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__0_value;
static const lean_ctor_object lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__1 = (const lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___boxed(lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__0 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__0_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__1 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__1_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "setBuilder"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__2 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__2_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__3_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__1_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__3_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__2_value),LEAN_SCALAR_PTR_LITERAL(55, 252, 174, 2, 80, 49, 173, 214)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__3 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__3_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__4 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__4_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "ExtendedBinder"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__5 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__5_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "extBinder"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__6 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__6_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__5_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__6_value),LEAN_SCALAR_PTR_LITERAL(140, 4, 199, 115, 152, 1, 62, 3)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__7 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__7_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "binderIdent"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__8 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__8_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__9_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__9_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__8_value),LEAN_SCALAR_PTR_LITERAL(37, 194, 68, 106, 254, 181, 31, 191)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__9 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__9_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred≤_"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__10 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__10_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__10_value),LEAN_SCALAR_PTR_LITERAL(118, 254, 52, 209, 56, 53, 218, 188)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__11 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__11_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 12, .m_data = "binderPred≥_"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__12 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__12_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__13_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__13_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__12_value),LEAN_SCALAR_PTR_LITERAL(212, 139, 67, 46, 49, 133, 157, 246)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__13 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__13_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "binderPred<_"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__14 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__14_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__15_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__14_value),LEAN_SCALAR_PTR_LITERAL(87, 122, 58, 108, 39, 32, 195, 29)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__15 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__15_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 12, .m_data = "binderPred>_"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__16 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__16_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__17_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__16_value),LEAN_SCALAR_PTR_LITERAL(140, 244, 246, 184, 111, 78, 213, 47)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__17 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__17_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "Finset.filter"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__18 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__18_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "filter"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__20 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__20_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__20_value),LEAN_SCALAR_PTR_LITERAL(88, 243, 224, 152, 142, 113, 169, 220)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__22 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__22_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__22_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__23 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__23_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "paren"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__24 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__24_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25_value_aux_0),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25_value_aux_1),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__24_value),LEAN_SCALAR_PTR_LITERAL(124, 9, 161, 194, 227, 100, 20, 110)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__26 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__26_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27_value_aux_0),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27_value_aux_1),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__26_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__28 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__28_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__29 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__29_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__29_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__30 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__30_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__31 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__31_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__33_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__33_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__1_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__33 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__33_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__33_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__34 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__34_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__35_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__4_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__35_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__5_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__35 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__35_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__35_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__36 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__36_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__37_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__37_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__1_value),LEAN_SCALAR_PTR_LITERAL(194, 50, 106, 158, 41, 60, 103, 214)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__37 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__37_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__37_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__38 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__38_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Elab"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__39 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__39_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__40_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__40_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__40_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__39_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__40_value_aux_1),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(252, 225, 247, 249, 114, 131, 135, 109)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__40 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__40_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__40_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__41 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__41_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__42_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__42_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__39_value),LEAN_SCALAR_PTR_LITERAL(52, 247, 248, 201, 92, 23, 188, 159)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__42 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__42_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__42_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__43 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__43_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__44_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__44_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__39_value),LEAN_SCALAR_PTR_LITERAL(228, 185, 96, 51, 222, 54, 124, 240)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__44 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__44_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__44_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__45 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__45_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__46 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__46_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__46_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__47 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__47_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "Function"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__48 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__48_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__48_value),LEAN_SCALAR_PTR_LITERAL(225, 8, 186, 189, 152, 89, 197, 12)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__49 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__49_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__49_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__50 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__50_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__51 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__51_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__51_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__52 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__52_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__52_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__53 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__53_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__50_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__53_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__54 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__54_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__47_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__54_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__55 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__55_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__45_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__55_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__56 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__56_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__43_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__56_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__57 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__57_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__41_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__57_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__58 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__58_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__34_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__58_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__59 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__59_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__38_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__59_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__60 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__60_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__36_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__60_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__61 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__61_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__34_value),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__61_value)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__62 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__62_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "fun"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__63 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__63_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64_value_aux_0),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64_value_aux_1),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__63_value),LEAN_SCALAR_PTR_LITERAL(249, 155, 133, 242, 71, 132, 191, 97)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "basicFun"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__65 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__65_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66_value_aux_0),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66_value_aux_1),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66_value_aux_2),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__65_value),LEAN_SCALAR_PTR_LITERAL(209, 134, 40, 160, 122, 195, 31, 223)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "↦"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__68 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__68_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__69 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__69_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Finset.Ioi"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__70 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__70_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__71_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__71;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Ioi"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__72 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__72_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__73_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__73_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__72_value),LEAN_SCALAR_PTR_LITERAL(81, 37, 134, 251, 22, 233, 98, 30)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__73 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__73_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__73_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__74 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__74_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__74_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__75 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__75_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Finset.Iio"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__76 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__76_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__77_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__77;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iio"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__78 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__78_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__79_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__79_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__78_value),LEAN_SCALAR_PTR_LITERAL(143, 117, 206, 115, 76, 88, 226, 57)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__79 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__79_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__80_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__79_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__80 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__80_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__80_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__81 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__81_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__82_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Finset.Ici"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__82 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__82_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__83_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__83;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__84_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Ici"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__84 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__84_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__85_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__85_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__85_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__84_value),LEAN_SCALAR_PTR_LITERAL(230, 63, 145, 248, 144, 220, 203, 85)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__85 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__85_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__86_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__85_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__86 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__86_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__87_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__86_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__87 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__87_value;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__88_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "Finset.Iic"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__88 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__88_value;
static lean_once_cell_t lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__89_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__89;
static const lean_string_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__90_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "Iic"};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__90 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__90_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__91_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(87, 75, 221, 45, 221, 79, 84, 42)}};
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__91_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__91_value_aux_0),((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__90_value),LEAN_SCALAR_PTR_LITERAL(47, 251, 203, 130, 143, 174, 27, 182)}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__91 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__91_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__92_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__91_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__92 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__92_value;
static const lean_ctor_object lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__93_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__92_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__93 = (const lean_object*)&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__93_value;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIcc___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIcc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIcc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIco___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIco(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIco___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoc___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoc___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoo___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIci___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIci(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIci___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIic___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIic(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIic___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoi___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoi(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoi___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIio___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIio(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIio___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeUIcc___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeUIcc(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___closed__0 = (const lean_object*)&lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderTop___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderTop___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderBot___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderBot___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_WithTop_insertTop___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Function_Embedding_coeWithTop___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithTop_insertTop___lam__0___closed__0 = (const lean_object*)&lp_mathlib_WithTop_insertTop___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithTop_insertTop___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_WithTop_insertTop___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithTop_insertTop___lam__0, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_WithTop_insertTop___closed__0 = (const lean_object*)&lp_mathlib_WithTop_insertTop___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithTop_insertTop(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_insertBot___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_WithBot_insertBot___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_WithBot_insertBot___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_WithBot_insertBot___closed__0 = (const lean_object*)&lp_mathlib_WithBot_insertBot___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_WithBot_insertBot(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLocallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLocallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLocallyFiniteOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLocallyFiniteOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLocallyFiniteOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLocallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder_match__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder_match__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder_match__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder_match__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_mk_x27___redArg(lean_object* v_finsetIcc_1_, lean_object* v_finsetIco_2_, lean_object* v_finsetIoc_3_, lean_object* v_finsetIoo_4_){
_start:
{
lean_object* v___x_5_; lean_object* v___x_6_; lean_object* v___x_7_; lean_object* v___x_8_; lean_object* v___x_9_; 
v___x_5_ = lean_alloc_closure((void*)(lp_mathlib_Function_swap), 6, 4);
lean_closure_set(v___x_5_, 0, lean_box(0));
lean_closure_set(v___x_5_, 1, lean_box(0));
lean_closure_set(v___x_5_, 2, lean_box(0));
lean_closure_set(v___x_5_, 3, v_finsetIcc_1_);
v___x_6_ = lean_alloc_closure((void*)(lp_mathlib_Function_swap), 6, 4);
lean_closure_set(v___x_6_, 0, lean_box(0));
lean_closure_set(v___x_6_, 1, lean_box(0));
lean_closure_set(v___x_6_, 2, lean_box(0));
lean_closure_set(v___x_6_, 3, v_finsetIoc_3_);
v___x_7_ = lean_alloc_closure((void*)(lp_mathlib_Function_swap), 6, 4);
lean_closure_set(v___x_7_, 0, lean_box(0));
lean_closure_set(v___x_7_, 1, lean_box(0));
lean_closure_set(v___x_7_, 2, lean_box(0));
lean_closure_set(v___x_7_, 3, v_finsetIco_2_);
v___x_8_ = lean_alloc_closure((void*)(lp_mathlib_Function_swap), 6, 4);
lean_closure_set(v___x_8_, 0, lean_box(0));
lean_closure_set(v___x_8_, 1, lean_box(0));
lean_closure_set(v___x_8_, 2, lean_box(0));
lean_closure_set(v___x_8_, 3, v_finsetIoo_4_);
v___x_9_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_9_, 0, v___x_5_);
lean_ctor_set(v___x_9_, 1, v___x_6_);
lean_ctor_set(v___x_9_, 2, v___x_7_);
lean_ctor_set(v___x_9_, 3, v___x_8_);
return v___x_9_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_mk_x27(lean_object* v_00_u03b1_10_, lean_object* v_inst_11_, lean_object* v_finsetIcc_12_, lean_object* v_finsetIco_13_, lean_object* v_finsetIoc_14_, lean_object* v_finsetIoo_15_, lean_object* v_finset__mem__Icc_16_, lean_object* v_finset__mem__Ico_17_, lean_object* v_finset__mem__Ioc_18_, lean_object* v_finset__mem__Ioo_19_){
_start:
{
lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; lean_object* v___x_23_; lean_object* v___x_24_; 
v___x_20_ = lean_alloc_closure((void*)(lp_mathlib_Function_swap), 6, 4);
lean_closure_set(v___x_20_, 0, lean_box(0));
lean_closure_set(v___x_20_, 1, lean_box(0));
lean_closure_set(v___x_20_, 2, lean_box(0));
lean_closure_set(v___x_20_, 3, v_finsetIcc_12_);
v___x_21_ = lean_alloc_closure((void*)(lp_mathlib_Function_swap), 6, 4);
lean_closure_set(v___x_21_, 0, lean_box(0));
lean_closure_set(v___x_21_, 1, lean_box(0));
lean_closure_set(v___x_21_, 2, lean_box(0));
lean_closure_set(v___x_21_, 3, v_finsetIoc_14_);
v___x_22_ = lean_alloc_closure((void*)(lp_mathlib_Function_swap), 6, 4);
lean_closure_set(v___x_22_, 0, lean_box(0));
lean_closure_set(v___x_22_, 1, lean_box(0));
lean_closure_set(v___x_22_, 2, lean_box(0));
lean_closure_set(v___x_22_, 3, v_finsetIco_13_);
v___x_23_ = lean_alloc_closure((void*)(lp_mathlib_Function_swap), 6, 4);
lean_closure_set(v___x_23_, 0, lean_box(0));
lean_closure_set(v___x_23_, 1, lean_box(0));
lean_closure_set(v___x_23_, 2, lean_box(0));
lean_closure_set(v___x_23_, 3, v_finsetIoo_15_);
v___x_24_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_24_, 0, v___x_20_);
lean_ctor_set(v___x_24_, 1, v___x_21_);
lean_ctor_set(v___x_24_, 2, v___x_22_);
lean_ctor_set(v___x_24_, 3, v___x_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_mk_x27___boxed(lean_object* v_00_u03b1_25_, lean_object* v_inst_26_, lean_object* v_finsetIcc_27_, lean_object* v_finsetIco_28_, lean_object* v_finsetIoc_29_, lean_object* v_finsetIoo_30_, lean_object* v_finset__mem__Icc_31_, lean_object* v_finset__mem__Ico_32_, lean_object* v_finset__mem__Ioc_33_, lean_object* v_finset__mem__Ioo_34_){
_start:
{
lean_object* v_res_35_; 
v_res_35_ = lp_mathlib_LocallyFiniteOrder_mk_x27(v_00_u03b1_25_, v_inst_26_, v_finsetIcc_27_, v_finsetIco_28_, v_finsetIoc_29_, v_finsetIoo_30_, v_finset__mem__Icc_31_, v_finset__mem__Ico_32_, v_finset__mem__Ioc_33_, v_finset__mem__Ioo_34_);
lean_dec_ref(v_inst_26_);
return v_res_35_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__0(lean_object* v_inst_36_, lean_object* v_b_37_, lean_object* v_a_38_){
_start:
{
lean_object* v___x_39_; uint8_t v___x_40_; 
v___x_39_ = lean_apply_2(v_inst_36_, v_b_37_, v_a_38_);
v___x_40_ = lean_unbox(v___x_39_);
if (v___x_40_ == 0)
{
uint8_t v___x_41_; 
v___x_41_ = 1;
return v___x_41_;
}
else
{
uint8_t v___x_42_; 
v___x_42_ = 0;
return v___x_42_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__0___boxed(lean_object* v_inst_43_, lean_object* v_b_44_, lean_object* v_a_45_){
_start:
{
uint8_t v_res_46_; lean_object* v_r_47_; 
v_res_46_ = lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__0(v_inst_43_, v_b_44_, v_a_45_);
v_r_47_ = lean_box(v_res_46_);
return v_r_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__1(lean_object* v_inst_48_, lean_object* v_finsetIcc_49_, lean_object* v_a_50_, lean_object* v_b_51_){
_start:
{
lean_object* v___f_52_; lean_object* v___x_53_; lean_object* v___x_54_; 
lean_inc(v_b_51_);
v___f_52_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_52_, 0, v_inst_48_);
lean_closure_set(v___f_52_, 1, v_b_51_);
v___x_53_ = lean_apply_2(v_finsetIcc_49_, v_a_50_, v_b_51_);
v___x_54_ = lp_mathlib_Multiset_filter___redArg(v___f_52_, v___x_53_);
return v___x_54_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__2(lean_object* v_inst_55_, lean_object* v_a_56_, lean_object* v_a_57_){
_start:
{
lean_object* v___x_58_; uint8_t v___x_59_; 
v___x_58_ = lean_apply_2(v_inst_55_, v_a_57_, v_a_56_);
v___x_59_ = lean_unbox(v___x_58_);
if (v___x_59_ == 0)
{
uint8_t v___x_60_; 
v___x_60_ = 1;
return v___x_60_;
}
else
{
uint8_t v___x_61_; 
v___x_61_ = 0;
return v___x_61_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__2___boxed(lean_object* v_inst_62_, lean_object* v_a_63_, lean_object* v_a_64_){
_start:
{
uint8_t v_res_65_; lean_object* v_r_66_; 
v_res_65_ = lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__2(v_inst_62_, v_a_63_, v_a_64_);
v_r_66_ = lean_box(v_res_65_);
return v_r_66_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__3(lean_object* v_inst_67_, lean_object* v_finsetIcc_68_, lean_object* v_a_69_, lean_object* v_b_70_){
_start:
{
lean_object* v___f_71_; lean_object* v___x_72_; lean_object* v___x_73_; 
lean_inc(v_a_69_);
v___f_71_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__2___boxed), 3, 2);
lean_closure_set(v___f_71_, 0, v_inst_67_);
lean_closure_set(v___f_71_, 1, v_a_69_);
v___x_72_ = lean_apply_2(v_finsetIcc_68_, v_a_69_, v_b_70_);
v___x_73_ = lp_mathlib_Multiset_filter___redArg(v___f_71_, v___x_72_);
return v___x_73_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__4(lean_object* v_inst_74_, lean_object* v_b_75_, lean_object* v_a_76_, lean_object* v_a_77_){
_start:
{
lean_object* v___x_78_; lean_object* v___x_79_; uint8_t v___x_80_; 
lean_inc_ref(v_inst_74_);
lean_inc(v_a_77_);
v___x_78_ = lean_apply_2(v_inst_74_, v_b_75_, v_a_77_);
v___x_79_ = lean_apply_2(v_inst_74_, v_a_77_, v_a_76_);
v___x_80_ = lean_unbox(v___x_79_);
if (v___x_80_ == 0)
{
uint8_t v___x_81_; 
v___x_81_ = lean_unbox(v___x_78_);
if (v___x_81_ == 0)
{
uint8_t v___x_82_; 
v___x_82_ = 1;
return v___x_82_;
}
else
{
uint8_t v___x_83_; 
v___x_83_ = lean_unbox(v___x_79_);
return v___x_83_;
}
}
else
{
uint8_t v___x_84_; 
v___x_84_ = 0;
return v___x_84_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__4___boxed(lean_object* v_inst_85_, lean_object* v_b_86_, lean_object* v_a_87_, lean_object* v_a_88_){
_start:
{
uint8_t v_res_89_; lean_object* v_r_90_; 
v_res_89_ = lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__4(v_inst_85_, v_b_86_, v_a_87_, v_a_88_);
v_r_90_ = lean_box(v_res_89_);
return v_r_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__5(lean_object* v_inst_91_, lean_object* v_finsetIcc_92_, lean_object* v_a_93_, lean_object* v_b_94_){
_start:
{
lean_object* v___f_95_; lean_object* v___x_96_; lean_object* v___x_97_; 
lean_inc(v_a_93_);
lean_inc(v_b_94_);
v___f_95_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__4___boxed), 4, 3);
lean_closure_set(v___f_95_, 0, v_inst_91_);
lean_closure_set(v___f_95_, 1, v_b_94_);
lean_closure_set(v___f_95_, 2, v_a_93_);
v___x_96_ = lean_apply_2(v_finsetIcc_92_, v_a_93_, v_b_94_);
v___x_97_ = lp_mathlib_Multiset_filter___redArg(v___f_95_, v___x_96_);
return v___x_97_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg(lean_object* v_inst_98_, lean_object* v_finsetIcc_99_){
_start:
{
lean_object* v___f_100_; lean_object* v___f_101_; lean_object* v___f_102_; lean_object* v___x_103_; 
lean_inc_n(v_finsetIcc_99_, 3);
lean_inc_ref_n(v_inst_98_, 2);
v___f_100_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__1), 4, 2);
lean_closure_set(v___f_100_, 0, v_inst_98_);
lean_closure_set(v___f_100_, 1, v_finsetIcc_99_);
v___f_101_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__3), 4, 2);
lean_closure_set(v___f_101_, 0, v_inst_98_);
lean_closure_set(v___f_101_, 1, v_finsetIcc_99_);
v___f_102_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__5), 4, 2);
lean_closure_set(v___f_102_, 0, v_inst_98_);
lean_closure_set(v___f_102_, 1, v_finsetIcc_99_);
v___x_103_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_103_, 0, v_finsetIcc_99_);
lean_ctor_set(v___x_103_, 1, v___f_100_);
lean_ctor_set(v___x_103_, 2, v___f_101_);
lean_ctor_set(v___x_103_, 3, v___f_102_);
return v___x_103_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27(lean_object* v_00_u03b1_104_, lean_object* v_inst_105_, lean_object* v_inst_106_, lean_object* v_finsetIcc_107_, lean_object* v_mem__Icc_108_){
_start:
{
lean_object* v___x_109_; 
v___x_109_ = lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg(v_inst_106_, v_finsetIcc_107_);
return v___x_109_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc_x27___boxed(lean_object* v_00_u03b1_110_, lean_object* v_inst_111_, lean_object* v_inst_112_, lean_object* v_finsetIcc_113_, lean_object* v_mem__Icc_114_){
_start:
{
lean_object* v_res_115_; 
v_res_115_ = lp_mathlib_LocallyFiniteOrder_ofIcc_x27(v_00_u03b1_110_, v_inst_111_, v_inst_112_, v_finsetIcc_113_, v_mem__Icc_114_);
lean_dec_ref(v_inst_111_);
return v_res_115_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__0(lean_object* v_inst_116_, lean_object* v_b_117_, lean_object* v_a_118_){
_start:
{
lean_object* v___x_119_; uint8_t v___x_120_; 
v___x_119_ = lean_apply_2(v_inst_116_, v_a_118_, v_b_117_);
v___x_120_ = lean_unbox(v___x_119_);
if (v___x_120_ == 0)
{
uint8_t v___x_121_; 
v___x_121_ = 1;
return v___x_121_;
}
else
{
uint8_t v___x_122_; 
v___x_122_ = 0;
return v___x_122_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__0___boxed(lean_object* v_inst_123_, lean_object* v_b_124_, lean_object* v_a_125_){
_start:
{
uint8_t v_res_126_; lean_object* v_r_127_; 
v_res_126_ = lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__0(v_inst_123_, v_b_124_, v_a_125_);
v_r_127_ = lean_box(v_res_126_);
return v_r_127_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__1(lean_object* v_inst_128_, lean_object* v_finsetIcc_129_, lean_object* v_a_130_, lean_object* v_b_131_){
_start:
{
lean_object* v___f_132_; lean_object* v___x_133_; lean_object* v___x_134_; 
lean_inc(v_b_131_);
v___f_132_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_132_, 0, v_inst_128_);
lean_closure_set(v___f_132_, 1, v_b_131_);
v___x_133_ = lean_apply_2(v_finsetIcc_129_, v_a_130_, v_b_131_);
v___x_134_ = lp_mathlib_Multiset_filter___redArg(v___f_132_, v___x_133_);
return v___x_134_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__2(lean_object* v_inst_135_, lean_object* v_a_136_, lean_object* v_a_137_){
_start:
{
lean_object* v___x_138_; uint8_t v___x_139_; 
v___x_138_ = lean_apply_2(v_inst_135_, v_a_136_, v_a_137_);
v___x_139_ = lean_unbox(v___x_138_);
if (v___x_139_ == 0)
{
uint8_t v___x_140_; 
v___x_140_ = 1;
return v___x_140_;
}
else
{
uint8_t v___x_141_; 
v___x_141_ = 0;
return v___x_141_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__2___boxed(lean_object* v_inst_142_, lean_object* v_a_143_, lean_object* v_a_144_){
_start:
{
uint8_t v_res_145_; lean_object* v_r_146_; 
v_res_145_ = lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__2(v_inst_142_, v_a_143_, v_a_144_);
v_r_146_ = lean_box(v_res_145_);
return v_r_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__3(lean_object* v_inst_147_, lean_object* v_finsetIcc_148_, lean_object* v_a_149_, lean_object* v_b_150_){
_start:
{
lean_object* v___f_151_; lean_object* v___x_152_; lean_object* v___x_153_; 
lean_inc(v_a_149_);
v___f_151_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__2___boxed), 3, 2);
lean_closure_set(v___f_151_, 0, v_inst_147_);
lean_closure_set(v___f_151_, 1, v_a_149_);
v___x_152_ = lean_apply_2(v_finsetIcc_148_, v_a_149_, v_b_150_);
v___x_153_ = lp_mathlib_Multiset_filter___redArg(v___f_151_, v___x_152_);
return v___x_153_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__4(lean_object* v_inst_154_, lean_object* v_b_155_, lean_object* v_a_156_, lean_object* v_a_157_){
_start:
{
lean_object* v___x_158_; lean_object* v___x_159_; uint8_t v___x_160_; 
lean_inc_ref(v_inst_154_);
lean_inc(v_a_157_);
v___x_158_ = lean_apply_2(v_inst_154_, v_a_157_, v_b_155_);
v___x_159_ = lean_apply_2(v_inst_154_, v_a_156_, v_a_157_);
v___x_160_ = lean_unbox(v___x_159_);
if (v___x_160_ == 0)
{
uint8_t v___x_161_; 
v___x_161_ = lean_unbox(v___x_158_);
if (v___x_161_ == 0)
{
uint8_t v___x_162_; 
v___x_162_ = 1;
return v___x_162_;
}
else
{
uint8_t v___x_163_; 
v___x_163_ = lean_unbox(v___x_159_);
return v___x_163_;
}
}
else
{
uint8_t v___x_164_; 
v___x_164_ = 0;
return v___x_164_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__4___boxed(lean_object* v_inst_165_, lean_object* v_b_166_, lean_object* v_a_167_, lean_object* v_a_168_){
_start:
{
uint8_t v_res_169_; lean_object* v_r_170_; 
v_res_169_ = lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__4(v_inst_165_, v_b_166_, v_a_167_, v_a_168_);
v_r_170_ = lean_box(v_res_169_);
return v_r_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__5(lean_object* v_inst_171_, lean_object* v_finsetIcc_172_, lean_object* v_a_173_, lean_object* v_b_174_){
_start:
{
lean_object* v___f_175_; lean_object* v___x_176_; lean_object* v___x_177_; 
lean_inc(v_a_173_);
lean_inc(v_b_174_);
v___f_175_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__4___boxed), 4, 3);
lean_closure_set(v___f_175_, 0, v_inst_171_);
lean_closure_set(v___f_175_, 1, v_b_174_);
lean_closure_set(v___f_175_, 2, v_a_173_);
v___x_176_ = lean_apply_2(v_finsetIcc_172_, v_a_173_, v_b_174_);
v___x_177_ = lp_mathlib_Multiset_filter___redArg(v___f_175_, v___x_176_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___redArg(lean_object* v_inst_178_, lean_object* v_finsetIcc_179_){
_start:
{
lean_object* v___f_180_; lean_object* v___f_181_; lean_object* v___f_182_; lean_object* v___x_183_; 
lean_inc_n(v_finsetIcc_179_, 3);
lean_inc_ref_n(v_inst_178_, 2);
v___f_180_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__1), 4, 2);
lean_closure_set(v___f_180_, 0, v_inst_178_);
lean_closure_set(v___f_180_, 1, v_finsetIcc_179_);
v___f_181_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__3), 4, 2);
lean_closure_set(v___f_181_, 0, v_inst_178_);
lean_closure_set(v___f_181_, 1, v_finsetIcc_179_);
v___f_182_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__5), 4, 2);
lean_closure_set(v___f_182_, 0, v_inst_178_);
lean_closure_set(v___f_182_, 1, v_finsetIcc_179_);
v___x_183_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_183_, 0, v_finsetIcc_179_);
lean_ctor_set(v___x_183_, 1, v___f_180_);
lean_ctor_set(v___x_183_, 2, v___f_181_);
lean_ctor_set(v___x_183_, 3, v___f_182_);
return v___x_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc(lean_object* v_00_u03b1_184_, lean_object* v_inst_185_, lean_object* v_inst_186_, lean_object* v_finsetIcc_187_, lean_object* v_mem__Icc_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_mathlib_LocallyFiniteOrder_ofIcc___redArg(v_inst_186_, v_finsetIcc_187_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofIcc___boxed(lean_object* v_00_u03b1_190_, lean_object* v_inst_191_, lean_object* v_inst_192_, lean_object* v_finsetIcc_193_, lean_object* v_mem__Icc_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_mathlib_LocallyFiniteOrder_ofIcc(v_00_u03b1_190_, v_inst_191_, v_inst_192_, v_finsetIcc_193_, v_mem__Icc_194_);
lean_dec_ref(v_inst_191_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___redArg___lam__1(lean_object* v_inst_196_, lean_object* v_finsetIci_197_, lean_object* v_a_198_){
_start:
{
lean_object* v___f_199_; lean_object* v___x_200_; lean_object* v___x_201_; 
lean_inc(v_a_198_);
v___f_199_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg___lam__2___boxed), 3, 2);
lean_closure_set(v___f_199_, 0, v_inst_196_);
lean_closure_set(v___f_199_, 1, v_a_198_);
v___x_200_ = lean_apply_1(v_finsetIci_197_, v_a_198_);
v___x_201_ = lp_mathlib_Multiset_filter___redArg(v___f_199_, v___x_200_);
return v___x_201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___redArg(lean_object* v_inst_202_, lean_object* v_finsetIci_203_){
_start:
{
lean_object* v___f_204_; lean_object* v___x_205_; 
lean_inc(v_finsetIci_203_);
v___f_204_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___redArg___lam__1), 3, 2);
lean_closure_set(v___f_204_, 0, v_inst_202_);
lean_closure_set(v___f_204_, 1, v_finsetIci_203_);
v___x_205_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_205_, 0, v___f_204_);
lean_ctor_set(v___x_205_, 1, v_finsetIci_203_);
return v___x_205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci_x27(lean_object* v_00_u03b1_206_, lean_object* v_inst_207_, lean_object* v_inst_208_, lean_object* v_finsetIci_209_, lean_object* v_mem__Ici_210_){
_start:
{
lean_object* v___x_211_; 
v___x_211_ = lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___redArg(v_inst_208_, v_finsetIci_209_);
return v___x_211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___boxed(lean_object* v_00_u03b1_212_, lean_object* v_inst_213_, lean_object* v_inst_214_, lean_object* v_finsetIci_215_, lean_object* v_mem__Ici_216_){
_start:
{
lean_object* v_res_217_; 
v_res_217_ = lp_mathlib_LocallyFiniteOrderTop_ofIci_x27(v_00_u03b1_212_, v_inst_213_, v_inst_214_, v_finsetIci_215_, v_mem__Ici_216_);
lean_dec_ref(v_inst_213_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___redArg___lam__1(lean_object* v_inst_218_, lean_object* v_finsetIci_219_, lean_object* v_a_220_){
_start:
{
lean_object* v___f_221_; lean_object* v___x_222_; lean_object* v___x_223_; 
lean_inc(v_a_220_);
v___f_221_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofIcc___redArg___lam__2___boxed), 3, 2);
lean_closure_set(v___f_221_, 0, v_inst_218_);
lean_closure_set(v___f_221_, 1, v_a_220_);
v___x_222_ = lean_apply_1(v_finsetIci_219_, v_a_220_);
v___x_223_ = lp_mathlib_Multiset_filter___redArg(v___f_221_, v___x_222_);
return v___x_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___redArg(lean_object* v_inst_224_, lean_object* v_finsetIci_225_){
_start:
{
lean_object* v___f_226_; lean_object* v___x_227_; 
lean_inc(v_finsetIci_225_);
v___f_226_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___redArg___lam__1), 3, 2);
lean_closure_set(v___f_226_, 0, v_inst_224_);
lean_closure_set(v___f_226_, 1, v_finsetIci_225_);
v___x_227_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_227_, 0, v___f_226_);
lean_ctor_set(v___x_227_, 1, v_finsetIci_225_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic_x27(lean_object* v_00_u03b1_228_, lean_object* v_inst_229_, lean_object* v_inst_230_, lean_object* v_finsetIci_231_, lean_object* v_mem__Ici_232_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___redArg(v_inst_230_, v_finsetIci_231_);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___boxed(lean_object* v_00_u03b1_234_, lean_object* v_inst_235_, lean_object* v_inst_236_, lean_object* v_finsetIci_237_, lean_object* v_mem__Ici_238_){
_start:
{
lean_object* v_res_239_; 
v_res_239_ = lp_mathlib_LocallyFiniteOrderBot_ofIic_x27(v_00_u03b1_234_, v_inst_235_, v_inst_236_, v_finsetIci_237_, v_mem__Ici_238_);
lean_dec_ref(v_inst_235_);
return v_res_239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci___redArg(lean_object* v_inst_240_, lean_object* v_finsetIci_241_){
_start:
{
lean_object* v___f_242_; lean_object* v___x_243_; 
lean_inc(v_finsetIci_241_);
v___f_242_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___redArg___lam__1), 3, 2);
lean_closure_set(v___f_242_, 0, v_inst_240_);
lean_closure_set(v___f_242_, 1, v_finsetIci_241_);
v___x_243_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_243_, 0, v___f_242_);
lean_ctor_set(v___x_243_, 1, v_finsetIci_241_);
return v___x_243_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci(lean_object* v_00_u03b1_244_, lean_object* v_inst_245_, lean_object* v_inst_246_, lean_object* v_finsetIci_247_, lean_object* v_mem__Ici_248_){
_start:
{
lean_object* v___x_249_; 
v___x_249_ = lp_mathlib_LocallyFiniteOrderTop_ofIci___redArg(v_inst_246_, v_finsetIci_247_);
return v___x_249_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderTop_ofIci___boxed(lean_object* v_00_u03b1_250_, lean_object* v_inst_251_, lean_object* v_inst_252_, lean_object* v_finsetIci_253_, lean_object* v_mem__Ici_254_){
_start:
{
lean_object* v_res_255_; 
v_res_255_ = lp_mathlib_LocallyFiniteOrderTop_ofIci(v_00_u03b1_250_, v_inst_251_, v_inst_252_, v_finsetIci_253_, v_mem__Ici_254_);
lean_dec_ref(v_inst_251_);
return v_res_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic___redArg(lean_object* v_inst_256_, lean_object* v_finsetIci_257_){
_start:
{
lean_object* v___f_258_; lean_object* v___x_259_; 
lean_inc(v_finsetIci_257_);
v___f_258_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___redArg___lam__1), 3, 2);
lean_closure_set(v___f_258_, 0, v_inst_256_);
lean_closure_set(v___f_258_, 1, v_finsetIci_257_);
v___x_259_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_259_, 0, v___f_258_);
lean_ctor_set(v___x_259_, 1, v_finsetIci_257_);
return v___x_259_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic(lean_object* v_00_u03b1_260_, lean_object* v_inst_261_, lean_object* v_inst_262_, lean_object* v_finsetIci_263_, lean_object* v_mem__Ici_264_){
_start:
{
lean_object* v___x_265_; 
v___x_265_ = lp_mathlib_LocallyFiniteOrderBot_ofIic___redArg(v_inst_262_, v_finsetIci_263_);
return v___x_265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrderBot_ofIic___boxed(lean_object* v_00_u03b1_266_, lean_object* v_inst_267_, lean_object* v_inst_268_, lean_object* v_finsetIci_269_, lean_object* v_mem__Ici_270_){
_start:
{
lean_object* v_res_271_; 
v_res_271_ = lp_mathlib_LocallyFiniteOrderBot_ofIic(v_00_u03b1_266_, v_inst_267_, v_inst_268_, v_finsetIci_269_, v_mem__Ici_270_);
lean_dec_ref(v_inst_267_);
return v_res_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder___lam__0(lean_object* v_a_272_, lean_object* v___y_273_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder___lam__0___boxed(lean_object* v_a_274_, lean_object* v___y_275_){
_start:
{
lean_object* v_res_276_; 
v_res_276_ = lp_mathlib_IsEmpty_toLocallyFiniteOrder___lam__0(v_a_274_, v___y_275_);
lean_dec(v___y_275_);
lean_dec(v_a_274_);
return v_res_276_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder(lean_object* v_00_u03b1_280_, lean_object* v_inst_281_, lean_object* v_inst_282_){
_start:
{
lean_object* v___x_283_; 
v___x_283_ = ((lean_object*)(lp_mathlib_IsEmpty_toLocallyFiniteOrder___closed__1));
return v___x_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrder___boxed(lean_object* v_00_u03b1_284_, lean_object* v_inst_285_, lean_object* v_inst_286_){
_start:
{
lean_object* v_res_287_; 
v_res_287_ = lp_mathlib_IsEmpty_toLocallyFiniteOrder(v_00_u03b1_284_, v_inst_285_, v_inst_286_);
lean_dec_ref(v_inst_285_);
return v_res_287_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___lam__0(lean_object* v_a_288_){
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___lam__0___boxed(lean_object* v_a_289_){
_start:
{
lean_object* v_res_290_; 
v_res_290_ = lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___lam__0(v_a_289_);
lean_dec(v_a_289_);
return v_res_290_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop(lean_object* v_00_u03b1_294_, lean_object* v_inst_295_, lean_object* v_inst_296_){
_start:
{
lean_object* v___x_297_; 
v___x_297_ = ((lean_object*)(lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___closed__1));
return v___x_297_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderTop___boxed(lean_object* v_00_u03b1_298_, lean_object* v_inst_299_, lean_object* v_inst_300_){
_start:
{
lean_object* v_res_301_; 
v_res_301_ = lp_mathlib_IsEmpty_toLocallyFiniteOrderTop(v_00_u03b1_298_, v_inst_299_, v_inst_300_);
lean_dec_ref(v_inst_299_);
return v_res_301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderBot(lean_object* v_00_u03b1_304_, lean_object* v_inst_305_, lean_object* v_inst_306_){
_start:
{
lean_object* v___x_307_; 
v___x_307_ = ((lean_object*)(lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___closed__0));
return v___x_307_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsEmpty_toLocallyFiniteOrderBot___boxed(lean_object* v_00_u03b1_308_, lean_object* v_inst_309_, lean_object* v_inst_310_){
_start:
{
lean_object* v_res_311_; 
v_res_311_ = lp_mathlib_IsEmpty_toLocallyFiniteOrderBot(v_00_u03b1_308_, v_inst_309_, v_inst_310_);
lean_dec_ref(v_inst_309_);
return v_res_311_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Icc___redArg(lean_object* v_inst_312_, lean_object* v_a_313_, lean_object* v_b_314_){
_start:
{
lean_object* v_finsetIcc_315_; lean_object* v___x_316_; 
v_finsetIcc_315_ = lean_ctor_get(v_inst_312_, 0);
lean_inc(v_finsetIcc_315_);
lean_dec_ref(v_inst_312_);
v___x_316_ = lean_apply_2(v_finsetIcc_315_, v_a_313_, v_b_314_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Icc(lean_object* v_00_u03b1_317_, lean_object* v_inst_318_, lean_object* v_inst_319_, lean_object* v_a_320_, lean_object* v_b_321_){
_start:
{
lean_object* v___x_322_; 
v___x_322_ = lp_mathlib_Finset_Icc___redArg(v_inst_319_, v_a_320_, v_b_321_);
return v___x_322_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Icc___boxed(lean_object* v_00_u03b1_323_, lean_object* v_inst_324_, lean_object* v_inst_325_, lean_object* v_a_326_, lean_object* v_b_327_){
_start:
{
lean_object* v_res_328_; 
v_res_328_ = lp_mathlib_Finset_Icc(v_00_u03b1_323_, v_inst_324_, v_inst_325_, v_a_326_, v_b_327_);
lean_dec_ref(v_inst_324_);
return v_res_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ico___redArg(lean_object* v_inst_329_, lean_object* v_a_330_, lean_object* v_b_331_){
_start:
{
lean_object* v_finsetIco_332_; lean_object* v___x_333_; 
v_finsetIco_332_ = lean_ctor_get(v_inst_329_, 1);
lean_inc(v_finsetIco_332_);
lean_dec_ref(v_inst_329_);
v___x_333_ = lean_apply_2(v_finsetIco_332_, v_a_330_, v_b_331_);
return v___x_333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ico(lean_object* v_00_u03b1_334_, lean_object* v_inst_335_, lean_object* v_inst_336_, lean_object* v_a_337_, lean_object* v_b_338_){
_start:
{
lean_object* v___x_339_; 
v___x_339_ = lp_mathlib_Finset_Ico___redArg(v_inst_336_, v_a_337_, v_b_338_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ico___boxed(lean_object* v_00_u03b1_340_, lean_object* v_inst_341_, lean_object* v_inst_342_, lean_object* v_a_343_, lean_object* v_b_344_){
_start:
{
lean_object* v_res_345_; 
v_res_345_ = lp_mathlib_Finset_Ico(v_00_u03b1_340_, v_inst_341_, v_inst_342_, v_a_343_, v_b_344_);
lean_dec_ref(v_inst_341_);
return v_res_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioc___redArg(lean_object* v_inst_346_, lean_object* v_a_347_, lean_object* v_b_348_){
_start:
{
lean_object* v_finsetIoc_349_; lean_object* v___x_350_; 
v_finsetIoc_349_ = lean_ctor_get(v_inst_346_, 2);
lean_inc(v_finsetIoc_349_);
lean_dec_ref(v_inst_346_);
v___x_350_ = lean_apply_2(v_finsetIoc_349_, v_a_347_, v_b_348_);
return v___x_350_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioc(lean_object* v_00_u03b1_351_, lean_object* v_inst_352_, lean_object* v_inst_353_, lean_object* v_a_354_, lean_object* v_b_355_){
_start:
{
lean_object* v___x_356_; 
v___x_356_ = lp_mathlib_Finset_Ioc___redArg(v_inst_353_, v_a_354_, v_b_355_);
return v___x_356_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioc___boxed(lean_object* v_00_u03b1_357_, lean_object* v_inst_358_, lean_object* v_inst_359_, lean_object* v_a_360_, lean_object* v_b_361_){
_start:
{
lean_object* v_res_362_; 
v_res_362_ = lp_mathlib_Finset_Ioc(v_00_u03b1_357_, v_inst_358_, v_inst_359_, v_a_360_, v_b_361_);
lean_dec_ref(v_inst_358_);
return v_res_362_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioo___redArg(lean_object* v_inst_363_, lean_object* v_a_364_, lean_object* v_b_365_){
_start:
{
lean_object* v_finsetIoo_366_; lean_object* v___x_367_; 
v_finsetIoo_366_ = lean_ctor_get(v_inst_363_, 3);
lean_inc(v_finsetIoo_366_);
lean_dec_ref(v_inst_363_);
v___x_367_ = lean_apply_2(v_finsetIoo_366_, v_a_364_, v_b_365_);
return v___x_367_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioo(lean_object* v_00_u03b1_368_, lean_object* v_inst_369_, lean_object* v_inst_370_, lean_object* v_a_371_, lean_object* v_b_372_){
_start:
{
lean_object* v___x_373_; 
v___x_373_ = lp_mathlib_Finset_Ioo___redArg(v_inst_370_, v_a_371_, v_b_372_);
return v___x_373_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioo___boxed(lean_object* v_00_u03b1_374_, lean_object* v_inst_375_, lean_object* v_inst_376_, lean_object* v_a_377_, lean_object* v_b_378_){
_start:
{
lean_object* v_res_379_; 
v_res_379_ = lp_mathlib_Finset_Ioo(v_00_u03b1_374_, v_inst_375_, v_inst_376_, v_a_377_, v_b_378_);
lean_dec_ref(v_inst_375_);
return v_res_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ici___redArg(lean_object* v_inst_380_, lean_object* v_a_381_){
_start:
{
lean_object* v_finsetIci_382_; lean_object* v___x_383_; 
v_finsetIci_382_ = lean_ctor_get(v_inst_380_, 1);
lean_inc(v_finsetIci_382_);
lean_dec_ref(v_inst_380_);
v___x_383_ = lean_apply_1(v_finsetIci_382_, v_a_381_);
return v___x_383_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ici(lean_object* v_00_u03b1_384_, lean_object* v_inst_385_, lean_object* v_inst_386_, lean_object* v_a_387_){
_start:
{
lean_object* v___x_388_; 
v___x_388_ = lp_mathlib_Finset_Ici___redArg(v_inst_386_, v_a_387_);
return v___x_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ici___boxed(lean_object* v_00_u03b1_389_, lean_object* v_inst_390_, lean_object* v_inst_391_, lean_object* v_a_392_){
_start:
{
lean_object* v_res_393_; 
v_res_393_ = lp_mathlib_Finset_Ici(v_00_u03b1_389_, v_inst_390_, v_inst_391_, v_a_392_);
lean_dec_ref(v_inst_390_);
return v_res_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iic___redArg(lean_object* v_inst_394_, lean_object* v_a_395_){
_start:
{
lean_object* v_finsetIic_396_; lean_object* v___x_397_; 
v_finsetIic_396_ = lean_ctor_get(v_inst_394_, 1);
lean_inc(v_finsetIic_396_);
lean_dec_ref(v_inst_394_);
v___x_397_ = lean_apply_1(v_finsetIic_396_, v_a_395_);
return v___x_397_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iic(lean_object* v_00_u03b1_398_, lean_object* v_inst_399_, lean_object* v_inst_400_, lean_object* v_a_401_){
_start:
{
lean_object* v___x_402_; 
v___x_402_ = lp_mathlib_Finset_Iic___redArg(v_inst_400_, v_a_401_);
return v___x_402_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iic___boxed(lean_object* v_00_u03b1_403_, lean_object* v_inst_404_, lean_object* v_inst_405_, lean_object* v_a_406_){
_start:
{
lean_object* v_res_407_; 
v_res_407_ = lp_mathlib_Finset_Iic(v_00_u03b1_403_, v_inst_404_, v_inst_405_, v_a_406_);
lean_dec_ref(v_inst_404_);
return v_res_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioi___redArg(lean_object* v_inst_408_, lean_object* v_a_409_){
_start:
{
lean_object* v_finsetIoi_410_; lean_object* v___x_411_; 
v_finsetIoi_410_ = lean_ctor_get(v_inst_408_, 0);
lean_inc(v_finsetIoi_410_);
lean_dec_ref(v_inst_408_);
v___x_411_ = lean_apply_1(v_finsetIoi_410_, v_a_409_);
return v___x_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioi(lean_object* v_00_u03b1_412_, lean_object* v_inst_413_, lean_object* v_inst_414_, lean_object* v_a_415_){
_start:
{
lean_object* v___x_416_; 
v___x_416_ = lp_mathlib_Finset_Ioi___redArg(v_inst_414_, v_a_415_);
return v___x_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Ioi___boxed(lean_object* v_00_u03b1_417_, lean_object* v_inst_418_, lean_object* v_inst_419_, lean_object* v_a_420_){
_start:
{
lean_object* v_res_421_; 
v_res_421_ = lp_mathlib_Finset_Ioi(v_00_u03b1_417_, v_inst_418_, v_inst_419_, v_a_420_);
lean_dec_ref(v_inst_418_);
return v_res_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iio___redArg(lean_object* v_inst_422_, lean_object* v_a_423_){
_start:
{
lean_object* v_finsetIio_424_; lean_object* v___x_425_; 
v_finsetIio_424_ = lean_ctor_get(v_inst_422_, 0);
lean_inc(v_finsetIio_424_);
lean_dec_ref(v_inst_422_);
v___x_425_ = lean_apply_1(v_finsetIio_424_, v_a_423_);
return v___x_425_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iio(lean_object* v_00_u03b1_426_, lean_object* v_inst_427_, lean_object* v_inst_428_, lean_object* v_a_429_){
_start:
{
lean_object* v___x_430_; 
v___x_430_ = lp_mathlib_Finset_Iio___redArg(v_inst_428_, v_a_429_);
return v___x_430_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_Iio___boxed(lean_object* v_00_u03b1_431_, lean_object* v_inst_432_, lean_object* v_inst_433_, lean_object* v_a_434_){
_start:
{
lean_object* v_res_435_; 
v_res_435_ = lp_mathlib_Finset_Iio(v_00_u03b1_431_, v_inst_432_, v_inst_433_, v_a_434_);
lean_dec_ref(v_inst_432_);
return v_res_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg___lam__0(lean_object* v_inst_436_, lean_object* v_inst_437_, lean_object* v_b_438_){
_start:
{
lean_object* v___x_439_; 
v___x_439_ = lp_mathlib_Finset_Icc___redArg(v_inst_436_, v_b_438_, v_inst_437_);
return v___x_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg___lam__1(lean_object* v_inst_440_, lean_object* v_inst_441_, lean_object* v_b_442_){
_start:
{
lean_object* v___x_443_; 
v___x_443_ = lp_mathlib_Finset_Ioc___redArg(v_inst_440_, v_b_442_, v_inst_441_);
return v___x_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg(lean_object* v_inst_444_, lean_object* v_inst_445_){
_start:
{
lean_object* v___f_446_; lean_object* v___f_447_; lean_object* v___x_448_; 
lean_inc(v_inst_445_);
lean_inc_ref(v_inst_444_);
v___f_446_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg___lam__0), 3, 2);
lean_closure_set(v___f_446_, 0, v_inst_444_);
lean_closure_set(v___f_446_, 1, v_inst_445_);
v___f_447_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg___lam__1), 3, 2);
lean_closure_set(v___f_447_, 0, v_inst_444_);
lean_closure_set(v___f_447_, 1, v_inst_445_);
v___x_448_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_448_, 0, v___f_447_);
lean_ctor_set(v___x_448_, 1, v___f_446_);
return v___x_448_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop(lean_object* v_00_u03b1_449_, lean_object* v_inst_450_, lean_object* v_inst_451_, lean_object* v_inst_452_){
_start:
{
lean_object* v___x_453_; 
v___x_453_ = lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg(v_inst_451_, v_inst_452_);
return v___x_453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___boxed(lean_object* v_00_u03b1_454_, lean_object* v_inst_455_, lean_object* v_inst_456_, lean_object* v_inst_457_){
_start:
{
lean_object* v_res_458_; 
v_res_458_ = lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop(v_00_u03b1_454_, v_inst_455_, v_inst_456_, v_inst_457_);
lean_dec_ref(v_inst_455_);
return v_res_458_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg___lam__0(lean_object* v_inst_459_, lean_object* v_inst_460_, lean_object* v_b_461_){
_start:
{
lean_object* v___x_462_; 
v___x_462_ = lp_mathlib_Finset_Icc___redArg(v_inst_459_, v_inst_460_, v_b_461_);
return v___x_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg___lam__1(lean_object* v_inst_463_, lean_object* v_inst_464_, lean_object* v_b_465_){
_start:
{
lean_object* v___x_466_; 
v___x_466_ = lp_mathlib_Finset_Ico___redArg(v_inst_463_, v_inst_464_, v_b_465_);
return v___x_466_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg(lean_object* v_inst_467_, lean_object* v_inst_468_){
_start:
{
lean_object* v___f_469_; lean_object* v___f_470_; lean_object* v___x_471_; 
lean_inc(v_inst_468_);
lean_inc_ref(v_inst_467_);
v___f_469_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg___lam__0), 3, 2);
lean_closure_set(v___f_469_, 0, v_inst_467_);
lean_closure_set(v___f_469_, 1, v_inst_468_);
v___f_470_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg___lam__1), 3, 2);
lean_closure_set(v___f_470_, 0, v_inst_467_);
lean_closure_set(v___f_470_, 1, v_inst_468_);
v___x_471_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_471_, 0, v___f_470_);
lean_ctor_set(v___x_471_, 1, v___f_469_);
return v___x_471_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot(lean_object* v_00_u03b1_472_, lean_object* v_inst_473_, lean_object* v_inst_474_, lean_object* v_inst_475_){
_start:
{
lean_object* v___x_476_; 
v___x_476_ = lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg(v_inst_474_, v_inst_475_);
return v___x_476_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___boxed(lean_object* v_00_u03b1_477_, lean_object* v_inst_478_, lean_object* v_inst_479_, lean_object* v_inst_480_){
_start:
{
lean_object* v_res_481_; 
v_res_481_ = lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot(v_00_u03b1_477_, v_inst_478_, v_inst_479_, v_inst_480_);
lean_dec_ref(v_inst_478_);
return v_res_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_uIcc___redArg(lean_object* v_inst_482_, lean_object* v_inst_483_, lean_object* v_a_484_, lean_object* v_b_485_){
_start:
{
lean_object* v_toSemilatticeSup_486_; lean_object* v_inf_487_; lean_object* v_sup_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; 
v_toSemilatticeSup_486_ = lean_ctor_get(v_inst_482_, 0);
lean_inc_ref(v_toSemilatticeSup_486_);
v_inf_487_ = lean_ctor_get(v_inst_482_, 1);
lean_inc(v_inf_487_);
lean_dec_ref(v_inst_482_);
v_sup_488_ = lean_ctor_get(v_toSemilatticeSup_486_, 1);
lean_inc(v_sup_488_);
lean_dec_ref(v_toSemilatticeSup_486_);
lean_inc(v_b_485_);
lean_inc(v_a_484_);
v___x_489_ = lean_apply_2(v_inf_487_, v_a_484_, v_b_485_);
v___x_490_ = lean_apply_2(v_sup_488_, v_a_484_, v_b_485_);
v___x_491_ = lp_mathlib_Finset_Icc___redArg(v_inst_483_, v___x_489_, v___x_490_);
return v___x_491_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_uIcc(lean_object* v_00_u03b1_492_, lean_object* v_inst_493_, lean_object* v_inst_494_, lean_object* v_a_495_, lean_object* v_b_496_){
_start:
{
lean_object* v___x_497_; 
v___x_497_ = lp_mathlib_Finset_uIcc___redArg(v_inst_493_, v_inst_494_, v_a_495_, v_b_496_);
return v___x_497_;
}
}
static lean_object* _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6(void){
_start:
{
lean_object* v___x_552_; lean_object* v___x_553_; 
v___x_552_ = ((lean_object*)(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__5));
v___x_553_ = l_String_toRawSubstring_x27(v___x_552_);
return v___x_553_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1(lean_object* v_x_568_, lean_object* v_a_569_, lean_object* v_a_570_){
_start:
{
lean_object* v___x_571_; uint8_t v___x_572_; 
v___x_571_ = ((lean_object*)(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__2));
lean_inc(v_x_568_);
v___x_572_ = l_Lean_Syntax_isOfKind(v_x_568_, v___x_571_);
if (v___x_572_ == 0)
{
lean_object* v___x_573_; lean_object* v___x_574_; 
lean_dec(v_x_568_);
v___x_573_ = lean_box(1);
v___x_574_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_574_, 0, v___x_573_);
lean_ctor_set(v___x_574_, 1, v_a_570_);
return v___x_574_;
}
else
{
lean_object* v_quotContext_575_; lean_object* v_currMacroScope_576_; lean_object* v_ref_577_; lean_object* v___x_578_; lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v___x_581_; uint8_t v___x_582_; lean_object* v___x_583_; lean_object* v___x_584_; lean_object* v___x_585_; lean_object* v___x_586_; lean_object* v___x_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___x_590_; lean_object* v___x_591_; lean_object* v___x_592_; lean_object* v___x_593_; 
v_quotContext_575_ = lean_ctor_get(v_a_569_, 1);
v_currMacroScope_576_ = lean_ctor_get(v_a_569_, 2);
v_ref_577_ = lean_ctor_get(v_a_569_, 5);
v___x_578_ = lean_unsigned_to_nat(1u);
v___x_579_ = l_Lean_Syntax_getArg(v_x_568_, v___x_578_);
v___x_580_ = lean_unsigned_to_nat(3u);
v___x_581_ = l_Lean_Syntax_getArg(v_x_568_, v___x_580_);
lean_dec(v_x_568_);
v___x_582_ = 0;
v___x_583_ = l_Lean_SourceInfo_fromRef(v_ref_577_, v___x_582_);
v___x_584_ = ((lean_object*)(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4));
v___x_585_ = lean_obj_once(&lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6, &lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6_once, _init_lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__6);
v___x_586_ = ((lean_object*)(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__9));
lean_inc(v_currMacroScope_576_);
lean_inc(v_quotContext_575_);
v___x_587_ = l_Lean_addMacroScope(v_quotContext_575_, v___x_586_, v_currMacroScope_576_);
v___x_588_ = ((lean_object*)(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__11));
lean_inc_n(v___x_583_, 2);
v___x_589_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_589_, 0, v___x_583_);
lean_ctor_set(v___x_589_, 1, v___x_585_);
lean_ctor_set(v___x_589_, 2, v___x_587_);
lean_ctor_set(v___x_589_, 3, v___x_588_);
v___x_590_ = ((lean_object*)(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13));
v___x_591_ = l_Lean_Syntax_node2(v___x_583_, v___x_590_, v___x_579_, v___x_581_);
v___x_592_ = l_Lean_Syntax_node2(v___x_583_, v___x_584_, v___x_589_, v___x_591_);
v___x_593_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_593_, 0, v___x_592_);
lean_ctor_set(v___x_593_, 1, v_a_570_);
return v___x_593_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___boxed(lean_object* v_x_594_, lean_object* v_a_595_, lean_object* v_a_596_){
_start:
{
lean_object* v_res_597_; 
v_res_597_ = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1(v_x_594_, v_a_595_, v_a_596_);
lean_dec_ref(v_a_595_);
return v_res_597_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1(lean_object* v_x_601_, lean_object* v_a_602_, lean_object* v_a_603_){
_start:
{
lean_object* v___x_604_; uint8_t v___x_605_; 
v___x_604_ = ((lean_object*)(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4));
lean_inc(v_x_601_);
v___x_605_ = l_Lean_Syntax_isOfKind(v_x_601_, v___x_604_);
if (v___x_605_ == 0)
{
lean_object* v___x_606_; lean_object* v___x_607_; 
lean_dec(v_x_601_);
v___x_606_ = lean_box(0);
v___x_607_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_607_, 0, v___x_606_);
lean_ctor_set(v___x_607_, 1, v_a_603_);
return v___x_607_;
}
else
{
lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; uint8_t v___x_611_; 
v___x_608_ = lean_unsigned_to_nat(0u);
v___x_609_ = l_Lean_Syntax_getArg(v_x_601_, v___x_608_);
v___x_610_ = ((lean_object*)(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__1));
lean_inc(v___x_609_);
v___x_611_ = l_Lean_Syntax_isOfKind(v___x_609_, v___x_610_);
if (v___x_611_ == 0)
{
lean_object* v___x_612_; lean_object* v___x_613_; 
lean_dec(v___x_609_);
lean_dec(v_x_601_);
v___x_612_ = lean_box(0);
v___x_613_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_613_, 0, v___x_612_);
lean_ctor_set(v___x_613_, 1, v_a_603_);
return v___x_613_;
}
else
{
lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; uint8_t v___x_617_; 
v___x_614_ = lean_unsigned_to_nat(1u);
v___x_615_ = l_Lean_Syntax_getArg(v_x_601_, v___x_614_);
lean_dec(v_x_601_);
v___x_616_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_615_);
v___x_617_ = l_Lean_Syntax_matchesNull(v___x_615_, v___x_616_);
if (v___x_617_ == 0)
{
lean_object* v___x_618_; lean_object* v___x_619_; 
lean_dec(v___x_615_);
lean_dec(v___x_609_);
v___x_618_ = lean_box(0);
v___x_619_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_619_, 0, v___x_618_);
lean_ctor_set(v___x_619_, 1, v_a_603_);
return v___x_619_;
}
else
{
lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v_ref_622_; uint8_t v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; 
v___x_620_ = l_Lean_Syntax_getArg(v___x_615_, v___x_608_);
v___x_621_ = l_Lean_Syntax_getArg(v___x_615_, v___x_614_);
lean_dec(v___x_615_);
v_ref_622_ = l_Lean_replaceRef(v___x_609_, v_a_602_);
lean_dec(v___x_609_);
v___x_623_ = 0;
v___x_624_ = l_Lean_SourceInfo_fromRef(v_ref_622_, v___x_623_);
lean_dec(v_ref_622_);
v___x_625_ = ((lean_object*)(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__2));
v___x_626_ = ((lean_object*)(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__5));
lean_inc_n(v___x_624_, 3);
v___x_627_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_627_, 0, v___x_624_);
lean_ctor_set(v___x_627_, 1, v___x_626_);
v___x_628_ = ((lean_object*)(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__11));
v___x_629_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_629_, 0, v___x_624_);
lean_ctor_set(v___x_629_, 1, v___x_628_);
v___x_630_ = ((lean_object*)(lp_mathlib_FinsetInterval_term_x5b_x5b___x2c___x5d_x5d___closed__15));
v___x_631_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_631_, 0, v___x_624_);
lean_ctor_set(v___x_631_, 1, v___x_630_);
v___x_632_ = l_Lean_Syntax_node5(v___x_624_, v___x_625_, v___x_627_, v___x_620_, v___x_629_, v___x_621_, v___x_631_);
v___x_633_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_633_, 0, v___x_632_);
lean_ctor_set(v___x_633_, 1, v_a_603_);
return v___x_633_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___boxed(lean_object* v_x_634_, lean_object* v_a_635_, lean_object* v_a_636_){
_start:
{
lean_object* v_res_637_; 
v_res_637_ = lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1(v_x_634_, v_a_635_, v_a_636_);
lean_dec(v_a_635_);
return v_res_637_;
}
}
static lean_object* _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__0(void){
_start:
{
lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; 
v___x_638_ = lean_box(0);
v___x_639_ = l_Lean_Elab_unsupportedSyntaxExceptionId;
v___x_640_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_640_, 0, v___x_639_);
lean_ctor_set(v___x_640_, 1, v___x_638_);
return v___x_640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg(){
_start:
{
lean_object* v___x_642_; lean_object* v___x_643_; 
v___x_642_ = lean_obj_once(&lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__0, &lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__0_once, _init_lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___closed__0);
v___x_643_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_643_, 0, v___x_642_);
return v___x_643_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg___boxed(lean_object* v___y_644_){
_start:
{
lean_object* v_res_645_; 
v_res_645_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
return v_res_645_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0(lean_object* v_00_u03b1_646_, lean_object* v___y_647_, lean_object* v___y_648_, lean_object* v___y_649_, lean_object* v___y_650_, lean_object* v___y_651_, lean_object* v___y_652_){
_start:
{
lean_object* v___x_654_; 
v___x_654_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
return v___x_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___boxed(lean_object* v_00_u03b1_655_, lean_object* v___y_656_, lean_object* v___y_657_, lean_object* v___y_658_, lean_object* v___y_659_, lean_object* v___y_660_, lean_object* v___y_661_, lean_object* v___y_662_){
_start:
{
lean_object* v_res_663_; 
v_res_663_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0(v_00_u03b1_655_, v___y_656_, v___y_657_, v___y_658_, v___y_659_, v___y_660_, v___y_661_);
lean_dec(v___y_661_);
lean_dec_ref(v___y_660_);
lean_dec(v___y_659_);
lean_dec_ref(v___y_658_);
lean_dec(v___y_657_);
lean_dec_ref(v___y_656_);
return v_res_663_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19(void){
_start:
{
lean_object* v___x_699_; lean_object* v___x_700_; 
v___x_699_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__18));
v___x_700_ = l_String_toRawSubstring_x27(v___x_699_);
return v___x_700_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32(void){
_start:
{
lean_object* v___x_728_; lean_object* v___x_729_; 
v___x_728_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__31));
v___x_729_ = l_String_toRawSubstring_x27(v___x_728_);
return v___x_729_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67(void){
_start:
{
lean_object* v___x_817_; 
v___x_817_ = l_Array_mkArray0(lean_box(0));
return v___x_817_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__71(void){
_start:
{
lean_object* v___x_821_; lean_object* v___x_822_; 
v___x_821_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__70));
v___x_822_ = l_String_toRawSubstring_x27(v___x_821_);
return v___x_822_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__77(void){
_start:
{
lean_object* v___x_834_; lean_object* v___x_835_; 
v___x_834_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__76));
v___x_835_ = l_String_toRawSubstring_x27(v___x_834_);
return v___x_835_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__83(void){
_start:
{
lean_object* v___x_847_; lean_object* v___x_848_; 
v___x_847_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__82));
v___x_848_ = l_String_toRawSubstring_x27(v___x_847_);
return v___x_848_;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__89(void){
_start:
{
lean_object* v___x_860_; lean_object* v___x_861_; 
v___x_860_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__88));
v___x_861_ = l_String_toRawSubstring_x27(v___x_860_);
return v___x_861_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx(lean_object* v_x_872_, lean_object* v_x_873_, lean_object* v_a_874_, lean_object* v_a_875_, lean_object* v_a_876_, lean_object* v_a_877_, lean_object* v_a_878_, lean_object* v_a_879_){
_start:
{
lean_object* v___x_881_; uint8_t v___x_882_; 
v___x_881_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__3));
lean_inc(v_x_872_);
v___x_882_ = l_Lean_Syntax_isOfKind(v_x_872_, v___x_881_);
if (v___x_882_ == 0)
{
lean_object* v___x_883_; 
lean_dec(v_x_873_);
lean_dec(v_x_872_);
v___x_883_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
return v___x_883_;
}
else
{
lean_object* v___x_884_; lean_object* v___x_885_; lean_object* v___x_886_; uint8_t v___x_887_; 
v___x_884_ = lean_unsigned_to_nat(1u);
v___x_885_ = l_Lean_Syntax_getArg(v_x_872_, v___x_884_);
v___x_886_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__7));
lean_inc(v___x_885_);
v___x_887_ = l_Lean_Syntax_isOfKind(v___x_885_, v___x_886_);
if (v___x_887_ == 0)
{
lean_object* v___x_888_; 
lean_dec(v___x_885_);
lean_dec(v_x_873_);
lean_dec(v_x_872_);
v___x_888_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
return v___x_888_;
}
else
{
lean_object* v___x_889_; lean_object* v___x_890_; lean_object* v___x_891_; uint8_t v___x_892_; 
v___x_889_ = lean_unsigned_to_nat(0u);
v___x_890_ = l_Lean_Syntax_getArg(v___x_885_, v___x_889_);
v___x_891_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__9));
lean_inc(v___x_890_);
v___x_892_ = l_Lean_Syntax_isOfKind(v___x_890_, v___x_891_);
if (v___x_892_ == 0)
{
lean_object* v___x_893_; 
lean_dec(v___x_890_);
lean_dec(v___x_885_);
lean_dec(v_x_873_);
lean_dec(v_x_872_);
v___x_893_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
return v___x_893_;
}
else
{
lean_object* v___x_894_; lean_object* v___x_895_; uint8_t v___x_896_; 
v___x_894_ = l_Lean_Syntax_getArg(v___x_890_, v___x_889_);
lean_dec(v___x_890_);
v___x_895_ = ((lean_object*)(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______unexpand__Finset__uIcc__1___closed__1));
lean_inc(v___x_894_);
v___x_896_ = l_Lean_Syntax_isOfKind(v___x_894_, v___x_895_);
if (v___x_896_ == 0)
{
lean_object* v___x_897_; 
lean_dec(v___x_894_);
lean_dec(v___x_885_);
lean_dec(v_x_873_);
lean_dec(v_x_872_);
v___x_897_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
return v___x_897_;
}
else
{
lean_object* v___x_898_; uint8_t v___x_899_; 
v___x_898_ = l_Lean_Syntax_getArg(v___x_885_, v___x_884_);
lean_dec(v___x_885_);
lean_inc(v___x_898_);
v___x_899_ = l_Lean_Syntax_matchesNull(v___x_898_, v___x_884_);
if (v___x_899_ == 0)
{
lean_object* v___x_900_; 
lean_dec(v___x_898_);
lean_dec(v___x_894_);
lean_dec(v_x_873_);
lean_dec(v_x_872_);
v___x_900_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
return v___x_900_;
}
else
{
lean_object* v___x_901_; lean_object* v___x_902_; uint8_t v___x_903_; 
v___x_901_ = l_Lean_Syntax_getArg(v___x_898_, v___x_889_);
lean_dec(v___x_898_);
v___x_902_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__11));
lean_inc(v___x_901_);
v___x_903_ = l_Lean_Syntax_isOfKind(v___x_901_, v___x_902_);
if (v___x_903_ == 0)
{
lean_object* v___x_904_; uint8_t v___x_905_; 
v___x_904_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__13));
lean_inc(v___x_901_);
v___x_905_ = l_Lean_Syntax_isOfKind(v___x_901_, v___x_904_);
if (v___x_905_ == 0)
{
lean_object* v___x_906_; uint8_t v___x_907_; 
v___x_906_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__15));
lean_inc(v___x_901_);
v___x_907_ = l_Lean_Syntax_isOfKind(v___x_901_, v___x_906_);
if (v___x_907_ == 0)
{
lean_object* v___x_908_; uint8_t v___x_909_; 
v___x_908_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__17));
lean_inc(v___x_901_);
v___x_909_ = l_Lean_Syntax_isOfKind(v___x_901_, v___x_908_);
if (v___x_909_ == 0)
{
lean_object* v___x_910_; 
lean_dec(v___x_901_);
lean_dec(v___x_894_);
lean_dec(v_x_873_);
lean_dec(v_x_872_);
v___x_910_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
return v___x_910_;
}
else
{
lean_object* v___x_911_; 
lean_inc(v_x_873_);
v___x_911_ = lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(v_x_873_, v_a_874_, v_a_875_, v_a_876_, v_a_877_, v_a_878_, v_a_879_);
if (lean_obj_tag(v___x_911_) == 0)
{
lean_object* v_a_912_; lean_object* v___x_913_; lean_object* v___x_914_; lean_object* v___x_915_; lean_object* v___y_917_; lean_object* v___y_918_; lean_object* v___y_919_; lean_object* v___y_920_; lean_object* v___y_921_; lean_object* v___y_922_; uint8_t v___x_971_; 
v_a_912_ = lean_ctor_get(v___x_911_, 0);
lean_inc(v_a_912_);
lean_dec_ref_known(v___x_911_, 1);
v___x_913_ = l_Lean_Syntax_getArg(v___x_901_, v___x_884_);
lean_dec(v___x_901_);
v___x_914_ = lean_unsigned_to_nat(3u);
v___x_915_ = l_Lean_Syntax_getArg(v_x_872_, v___x_914_);
lean_dec(v_x_872_);
v___x_971_ = lean_unbox(v_a_912_);
lean_dec(v_a_912_);
if (v___x_971_ == 0)
{
lean_object* v___x_972_; lean_object* v_a_973_; lean_object* v___x_975_; uint8_t v_isShared_976_; uint8_t v_isSharedCheck_980_; 
lean_dec(v___x_915_);
lean_dec(v___x_913_);
lean_dec(v___x_894_);
lean_dec(v_x_873_);
v___x_972_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
v_a_973_ = lean_ctor_get(v___x_972_, 0);
v_isSharedCheck_980_ = !lean_is_exclusive(v___x_972_);
if (v_isSharedCheck_980_ == 0)
{
v___x_975_ = v___x_972_;
v_isShared_976_ = v_isSharedCheck_980_;
goto v_resetjp_974_;
}
else
{
lean_inc(v_a_973_);
lean_dec(v___x_972_);
v___x_975_ = lean_box(0);
v_isShared_976_ = v_isSharedCheck_980_;
goto v_resetjp_974_;
}
v_resetjp_974_:
{
lean_object* v___x_978_; 
if (v_isShared_976_ == 0)
{
v___x_978_ = v___x_975_;
goto v_reusejp_977_;
}
else
{
lean_object* v_reuseFailAlloc_979_; 
v_reuseFailAlloc_979_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_979_, 0, v_a_973_);
v___x_978_ = v_reuseFailAlloc_979_;
goto v_reusejp_977_;
}
v_reusejp_977_:
{
return v___x_978_;
}
}
}
else
{
v___y_917_ = v_a_874_;
v___y_918_ = v_a_875_;
v___y_919_ = v_a_876_;
v___y_920_ = v_a_877_;
v___y_921_ = v_a_878_;
v___y_922_ = v_a_879_;
goto v___jp_916_;
}
v___jp_916_:
{
lean_object* v_ref_923_; lean_object* v_quotContext_924_; lean_object* v_currMacroScope_925_; lean_object* v___x_926_; lean_object* v___x_927_; lean_object* v___x_928_; lean_object* v___x_929_; lean_object* v___x_930_; lean_object* v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; lean_object* v___x_937_; lean_object* v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; lean_object* v___x_941_; lean_object* v___x_942_; lean_object* v___x_943_; lean_object* v___x_944_; lean_object* v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; lean_object* v___x_948_; lean_object* v___x_949_; lean_object* v___x_950_; lean_object* v___x_951_; lean_object* v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; lean_object* v___x_955_; lean_object* v___x_956_; lean_object* v___x_957_; lean_object* v___x_958_; lean_object* v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; lean_object* v___x_969_; lean_object* v___x_970_; 
v_ref_923_ = lean_ctor_get(v___y_921_, 5);
v_quotContext_924_ = lean_ctor_get(v___y_921_, 10);
v_currMacroScope_925_ = lean_ctor_get(v___y_921_, 11);
v___x_926_ = l_Lean_SourceInfo_fromRef(v_ref_923_, v___x_907_);
v___x_927_ = ((lean_object*)(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4));
v___x_928_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19);
v___x_929_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21));
lean_inc_n(v_currMacroScope_925_, 3);
lean_inc_n(v_quotContext_924_, 3);
v___x_930_ = l_Lean_addMacroScope(v_quotContext_924_, v___x_929_, v_currMacroScope_925_);
v___x_931_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__23));
lean_inc_n(v___x_926_, 18);
v___x_932_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_932_, 0, v___x_926_);
lean_ctor_set(v___x_932_, 1, v___x_928_);
lean_ctor_set(v___x_932_, 2, v___x_930_);
lean_ctor_set(v___x_932_, 3, v___x_931_);
v___x_933_ = ((lean_object*)(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13));
v___x_934_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25));
v___x_935_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27));
v___x_936_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__28));
v___x_937_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_937_, 0, v___x_926_);
lean_ctor_set(v___x_937_, 1, v___x_936_);
v___x_938_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__30));
v___x_939_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32);
v___x_940_ = lean_box(0);
v___x_941_ = l_Lean_addMacroScope(v_quotContext_924_, v___x_940_, v_currMacroScope_925_);
v___x_942_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__62));
v___x_943_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_943_, 0, v___x_926_);
lean_ctor_set(v___x_943_, 1, v___x_939_);
lean_ctor_set(v___x_943_, 2, v___x_941_);
lean_ctor_set(v___x_943_, 3, v___x_942_);
v___x_944_ = l_Lean_Syntax_node1(v___x_926_, v___x_938_, v___x_943_);
v___x_945_ = l_Lean_Syntax_node2(v___x_926_, v___x_935_, v___x_937_, v___x_944_);
v___x_946_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__63));
v___x_947_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64));
v___x_948_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_948_, 0, v___x_926_);
lean_ctor_set(v___x_948_, 1, v___x_946_);
v___x_949_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66));
v___x_950_ = l_Lean_Syntax_node1(v___x_926_, v___x_933_, v___x_894_);
v___x_951_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67);
v___x_952_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_952_, 0, v___x_926_);
lean_ctor_set(v___x_952_, 1, v___x_933_);
lean_ctor_set(v___x_952_, 2, v___x_951_);
v___x_953_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__68));
v___x_954_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_954_, 0, v___x_926_);
lean_ctor_set(v___x_954_, 1, v___x_953_);
v___x_955_ = l_Lean_Syntax_node4(v___x_926_, v___x_949_, v___x_950_, v___x_952_, v___x_954_, v___x_915_);
v___x_956_ = l_Lean_Syntax_node2(v___x_926_, v___x_947_, v___x_948_, v___x_955_);
v___x_957_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__69));
v___x_958_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_958_, 0, v___x_926_);
lean_ctor_set(v___x_958_, 1, v___x_957_);
lean_inc_ref(v___x_958_);
lean_inc(v___x_945_);
v___x_959_ = l_Lean_Syntax_node3(v___x_926_, v___x_934_, v___x_945_, v___x_956_, v___x_958_);
v___x_960_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__71, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__71_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__71);
v___x_961_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__73));
v___x_962_ = l_Lean_addMacroScope(v_quotContext_924_, v___x_961_, v_currMacroScope_925_);
v___x_963_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__75));
v___x_964_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_964_, 0, v___x_926_);
lean_ctor_set(v___x_964_, 1, v___x_960_);
lean_ctor_set(v___x_964_, 2, v___x_962_);
lean_ctor_set(v___x_964_, 3, v___x_963_);
v___x_965_ = l_Lean_Syntax_node1(v___x_926_, v___x_933_, v___x_913_);
v___x_966_ = l_Lean_Syntax_node2(v___x_926_, v___x_927_, v___x_964_, v___x_965_);
v___x_967_ = l_Lean_Syntax_node3(v___x_926_, v___x_934_, v___x_945_, v___x_966_, v___x_958_);
v___x_968_ = l_Lean_Syntax_node2(v___x_926_, v___x_933_, v___x_959_, v___x_967_);
v___x_969_ = l_Lean_Syntax_node2(v___x_926_, v___x_927_, v___x_932_, v___x_968_);
v___x_970_ = l_Lean_Elab_Term_elabTerm(v___x_969_, v_x_873_, v___x_899_, v___x_899_, v___y_917_, v___y_918_, v___y_919_, v___y_920_, v___y_921_, v___y_922_);
return v___x_970_;
}
}
else
{
lean_object* v_a_981_; lean_object* v___x_983_; uint8_t v_isShared_984_; uint8_t v_isSharedCheck_988_; 
lean_dec(v___x_901_);
lean_dec(v___x_894_);
lean_dec(v_x_873_);
lean_dec(v_x_872_);
v_a_981_ = lean_ctor_get(v___x_911_, 0);
v_isSharedCheck_988_ = !lean_is_exclusive(v___x_911_);
if (v_isSharedCheck_988_ == 0)
{
v___x_983_ = v___x_911_;
v_isShared_984_ = v_isSharedCheck_988_;
goto v_resetjp_982_;
}
else
{
lean_inc(v_a_981_);
lean_dec(v___x_911_);
v___x_983_ = lean_box(0);
v_isShared_984_ = v_isSharedCheck_988_;
goto v_resetjp_982_;
}
v_resetjp_982_:
{
lean_object* v___x_986_; 
if (v_isShared_984_ == 0)
{
v___x_986_ = v___x_983_;
goto v_reusejp_985_;
}
else
{
lean_object* v_reuseFailAlloc_987_; 
v_reuseFailAlloc_987_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_987_, 0, v_a_981_);
v___x_986_ = v_reuseFailAlloc_987_;
goto v_reusejp_985_;
}
v_reusejp_985_:
{
return v___x_986_;
}
}
}
}
}
else
{
lean_object* v___x_989_; 
lean_inc(v_x_873_);
v___x_989_ = lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(v_x_873_, v_a_874_, v_a_875_, v_a_876_, v_a_877_, v_a_878_, v_a_879_);
if (lean_obj_tag(v___x_989_) == 0)
{
lean_object* v_a_990_; lean_object* v___x_991_; lean_object* v___x_992_; lean_object* v___x_993_; lean_object* v___y_995_; lean_object* v___y_996_; lean_object* v___y_997_; lean_object* v___y_998_; lean_object* v___y_999_; lean_object* v___y_1000_; uint8_t v___x_1049_; 
v_a_990_ = lean_ctor_get(v___x_989_, 0);
lean_inc(v_a_990_);
lean_dec_ref_known(v___x_989_, 1);
v___x_991_ = l_Lean_Syntax_getArg(v___x_901_, v___x_884_);
lean_dec(v___x_901_);
v___x_992_ = lean_unsigned_to_nat(3u);
v___x_993_ = l_Lean_Syntax_getArg(v_x_872_, v___x_992_);
lean_dec(v_x_872_);
v___x_1049_ = lean_unbox(v_a_990_);
lean_dec(v_a_990_);
if (v___x_1049_ == 0)
{
lean_object* v___x_1050_; lean_object* v_a_1051_; lean_object* v___x_1053_; uint8_t v_isShared_1054_; uint8_t v_isSharedCheck_1058_; 
lean_dec(v___x_993_);
lean_dec(v___x_991_);
lean_dec(v___x_894_);
lean_dec(v_x_873_);
v___x_1050_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
v_a_1051_ = lean_ctor_get(v___x_1050_, 0);
v_isSharedCheck_1058_ = !lean_is_exclusive(v___x_1050_);
if (v_isSharedCheck_1058_ == 0)
{
v___x_1053_ = v___x_1050_;
v_isShared_1054_ = v_isSharedCheck_1058_;
goto v_resetjp_1052_;
}
else
{
lean_inc(v_a_1051_);
lean_dec(v___x_1050_);
v___x_1053_ = lean_box(0);
v_isShared_1054_ = v_isSharedCheck_1058_;
goto v_resetjp_1052_;
}
v_resetjp_1052_:
{
lean_object* v___x_1056_; 
if (v_isShared_1054_ == 0)
{
v___x_1056_ = v___x_1053_;
goto v_reusejp_1055_;
}
else
{
lean_object* v_reuseFailAlloc_1057_; 
v_reuseFailAlloc_1057_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1057_, 0, v_a_1051_);
v___x_1056_ = v_reuseFailAlloc_1057_;
goto v_reusejp_1055_;
}
v_reusejp_1055_:
{
return v___x_1056_;
}
}
}
else
{
v___y_995_ = v_a_874_;
v___y_996_ = v_a_875_;
v___y_997_ = v_a_876_;
v___y_998_ = v_a_877_;
v___y_999_ = v_a_878_;
v___y_1000_ = v_a_879_;
goto v___jp_994_;
}
v___jp_994_:
{
lean_object* v_ref_1001_; lean_object* v_quotContext_1002_; lean_object* v_currMacroScope_1003_; lean_object* v___x_1004_; lean_object* v___x_1005_; lean_object* v___x_1006_; lean_object* v___x_1007_; lean_object* v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; lean_object* v___x_1011_; lean_object* v___x_1012_; lean_object* v___x_1013_; lean_object* v___x_1014_; lean_object* v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; lean_object* v___x_1018_; lean_object* v___x_1019_; lean_object* v___x_1020_; lean_object* v___x_1021_; lean_object* v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; lean_object* v___x_1025_; lean_object* v___x_1026_; lean_object* v___x_1027_; lean_object* v___x_1028_; lean_object* v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; lean_object* v___x_1032_; lean_object* v___x_1033_; lean_object* v___x_1034_; lean_object* v___x_1035_; lean_object* v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; lean_object* v___x_1039_; lean_object* v___x_1040_; lean_object* v___x_1041_; lean_object* v___x_1042_; lean_object* v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; lean_object* v___x_1046_; lean_object* v___x_1047_; lean_object* v___x_1048_; 
v_ref_1001_ = lean_ctor_get(v___y_999_, 5);
v_quotContext_1002_ = lean_ctor_get(v___y_999_, 10);
v_currMacroScope_1003_ = lean_ctor_get(v___y_999_, 11);
v___x_1004_ = l_Lean_SourceInfo_fromRef(v_ref_1001_, v___x_905_);
v___x_1005_ = ((lean_object*)(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4));
v___x_1006_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19);
v___x_1007_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21));
lean_inc_n(v_currMacroScope_1003_, 3);
lean_inc_n(v_quotContext_1002_, 3);
v___x_1008_ = l_Lean_addMacroScope(v_quotContext_1002_, v___x_1007_, v_currMacroScope_1003_);
v___x_1009_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__23));
lean_inc_n(v___x_1004_, 18);
v___x_1010_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1010_, 0, v___x_1004_);
lean_ctor_set(v___x_1010_, 1, v___x_1006_);
lean_ctor_set(v___x_1010_, 2, v___x_1008_);
lean_ctor_set(v___x_1010_, 3, v___x_1009_);
v___x_1011_ = ((lean_object*)(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13));
v___x_1012_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25));
v___x_1013_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27));
v___x_1014_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__28));
v___x_1015_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1015_, 0, v___x_1004_);
lean_ctor_set(v___x_1015_, 1, v___x_1014_);
v___x_1016_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__30));
v___x_1017_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32);
v___x_1018_ = lean_box(0);
v___x_1019_ = l_Lean_addMacroScope(v_quotContext_1002_, v___x_1018_, v_currMacroScope_1003_);
v___x_1020_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__62));
v___x_1021_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1021_, 0, v___x_1004_);
lean_ctor_set(v___x_1021_, 1, v___x_1017_);
lean_ctor_set(v___x_1021_, 2, v___x_1019_);
lean_ctor_set(v___x_1021_, 3, v___x_1020_);
v___x_1022_ = l_Lean_Syntax_node1(v___x_1004_, v___x_1016_, v___x_1021_);
v___x_1023_ = l_Lean_Syntax_node2(v___x_1004_, v___x_1013_, v___x_1015_, v___x_1022_);
v___x_1024_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__63));
v___x_1025_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64));
v___x_1026_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1026_, 0, v___x_1004_);
lean_ctor_set(v___x_1026_, 1, v___x_1024_);
v___x_1027_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66));
v___x_1028_ = l_Lean_Syntax_node1(v___x_1004_, v___x_1011_, v___x_894_);
v___x_1029_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67);
v___x_1030_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1030_, 0, v___x_1004_);
lean_ctor_set(v___x_1030_, 1, v___x_1011_);
lean_ctor_set(v___x_1030_, 2, v___x_1029_);
v___x_1031_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__68));
v___x_1032_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1032_, 0, v___x_1004_);
lean_ctor_set(v___x_1032_, 1, v___x_1031_);
v___x_1033_ = l_Lean_Syntax_node4(v___x_1004_, v___x_1027_, v___x_1028_, v___x_1030_, v___x_1032_, v___x_993_);
v___x_1034_ = l_Lean_Syntax_node2(v___x_1004_, v___x_1025_, v___x_1026_, v___x_1033_);
v___x_1035_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__69));
v___x_1036_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1036_, 0, v___x_1004_);
lean_ctor_set(v___x_1036_, 1, v___x_1035_);
lean_inc_ref(v___x_1036_);
lean_inc(v___x_1023_);
v___x_1037_ = l_Lean_Syntax_node3(v___x_1004_, v___x_1012_, v___x_1023_, v___x_1034_, v___x_1036_);
v___x_1038_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__77, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__77_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__77);
v___x_1039_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__79));
v___x_1040_ = l_Lean_addMacroScope(v_quotContext_1002_, v___x_1039_, v_currMacroScope_1003_);
v___x_1041_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__81));
v___x_1042_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1042_, 0, v___x_1004_);
lean_ctor_set(v___x_1042_, 1, v___x_1038_);
lean_ctor_set(v___x_1042_, 2, v___x_1040_);
lean_ctor_set(v___x_1042_, 3, v___x_1041_);
v___x_1043_ = l_Lean_Syntax_node1(v___x_1004_, v___x_1011_, v___x_991_);
v___x_1044_ = l_Lean_Syntax_node2(v___x_1004_, v___x_1005_, v___x_1042_, v___x_1043_);
v___x_1045_ = l_Lean_Syntax_node3(v___x_1004_, v___x_1012_, v___x_1023_, v___x_1044_, v___x_1036_);
v___x_1046_ = l_Lean_Syntax_node2(v___x_1004_, v___x_1011_, v___x_1037_, v___x_1045_);
v___x_1047_ = l_Lean_Syntax_node2(v___x_1004_, v___x_1005_, v___x_1010_, v___x_1046_);
v___x_1048_ = l_Lean_Elab_Term_elabTerm(v___x_1047_, v_x_873_, v___x_899_, v___x_899_, v___y_995_, v___y_996_, v___y_997_, v___y_998_, v___y_999_, v___y_1000_);
return v___x_1048_;
}
}
else
{
lean_object* v_a_1059_; lean_object* v___x_1061_; uint8_t v_isShared_1062_; uint8_t v_isSharedCheck_1066_; 
lean_dec(v___x_901_);
lean_dec(v___x_894_);
lean_dec(v_x_873_);
lean_dec(v_x_872_);
v_a_1059_ = lean_ctor_get(v___x_989_, 0);
v_isSharedCheck_1066_ = !lean_is_exclusive(v___x_989_);
if (v_isSharedCheck_1066_ == 0)
{
v___x_1061_ = v___x_989_;
v_isShared_1062_ = v_isSharedCheck_1066_;
goto v_resetjp_1060_;
}
else
{
lean_inc(v_a_1059_);
lean_dec(v___x_989_);
v___x_1061_ = lean_box(0);
v_isShared_1062_ = v_isSharedCheck_1066_;
goto v_resetjp_1060_;
}
v_resetjp_1060_:
{
lean_object* v___x_1064_; 
if (v_isShared_1062_ == 0)
{
v___x_1064_ = v___x_1061_;
goto v_reusejp_1063_;
}
else
{
lean_object* v_reuseFailAlloc_1065_; 
v_reuseFailAlloc_1065_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1065_, 0, v_a_1059_);
v___x_1064_ = v_reuseFailAlloc_1065_;
goto v_reusejp_1063_;
}
v_reusejp_1063_:
{
return v___x_1064_;
}
}
}
}
}
else
{
lean_object* v___x_1067_; 
lean_inc(v_x_873_);
v___x_1067_ = lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(v_x_873_, v_a_874_, v_a_875_, v_a_876_, v_a_877_, v_a_878_, v_a_879_);
if (lean_obj_tag(v___x_1067_) == 0)
{
lean_object* v_a_1068_; lean_object* v___x_1069_; lean_object* v___x_1070_; lean_object* v___x_1071_; lean_object* v___y_1073_; lean_object* v___y_1074_; lean_object* v___y_1075_; lean_object* v___y_1076_; lean_object* v___y_1077_; lean_object* v___y_1078_; uint8_t v___x_1127_; 
v_a_1068_ = lean_ctor_get(v___x_1067_, 0);
lean_inc(v_a_1068_);
lean_dec_ref_known(v___x_1067_, 1);
v___x_1069_ = l_Lean_Syntax_getArg(v___x_901_, v___x_884_);
lean_dec(v___x_901_);
v___x_1070_ = lean_unsigned_to_nat(3u);
v___x_1071_ = l_Lean_Syntax_getArg(v_x_872_, v___x_1070_);
lean_dec(v_x_872_);
v___x_1127_ = lean_unbox(v_a_1068_);
lean_dec(v_a_1068_);
if (v___x_1127_ == 0)
{
lean_object* v___x_1128_; lean_object* v_a_1129_; lean_object* v___x_1131_; uint8_t v_isShared_1132_; uint8_t v_isSharedCheck_1136_; 
lean_dec(v___x_1071_);
lean_dec(v___x_1069_);
lean_dec(v___x_894_);
lean_dec(v_x_873_);
v___x_1128_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
v_a_1129_ = lean_ctor_get(v___x_1128_, 0);
v_isSharedCheck_1136_ = !lean_is_exclusive(v___x_1128_);
if (v_isSharedCheck_1136_ == 0)
{
v___x_1131_ = v___x_1128_;
v_isShared_1132_ = v_isSharedCheck_1136_;
goto v_resetjp_1130_;
}
else
{
lean_inc(v_a_1129_);
lean_dec(v___x_1128_);
v___x_1131_ = lean_box(0);
v_isShared_1132_ = v_isSharedCheck_1136_;
goto v_resetjp_1130_;
}
v_resetjp_1130_:
{
lean_object* v___x_1134_; 
if (v_isShared_1132_ == 0)
{
v___x_1134_ = v___x_1131_;
goto v_reusejp_1133_;
}
else
{
lean_object* v_reuseFailAlloc_1135_; 
v_reuseFailAlloc_1135_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1135_, 0, v_a_1129_);
v___x_1134_ = v_reuseFailAlloc_1135_;
goto v_reusejp_1133_;
}
v_reusejp_1133_:
{
return v___x_1134_;
}
}
}
else
{
v___y_1073_ = v_a_874_;
v___y_1074_ = v_a_875_;
v___y_1075_ = v_a_876_;
v___y_1076_ = v_a_877_;
v___y_1077_ = v_a_878_;
v___y_1078_ = v_a_879_;
goto v___jp_1072_;
}
v___jp_1072_:
{
lean_object* v_ref_1079_; lean_object* v_quotContext_1080_; lean_object* v_currMacroScope_1081_; lean_object* v___x_1082_; lean_object* v___x_1083_; lean_object* v___x_1084_; lean_object* v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; lean_object* v___x_1088_; lean_object* v___x_1089_; lean_object* v___x_1090_; lean_object* v___x_1091_; lean_object* v___x_1092_; lean_object* v___x_1093_; lean_object* v___x_1094_; lean_object* v___x_1095_; lean_object* v___x_1096_; lean_object* v___x_1097_; lean_object* v___x_1098_; lean_object* v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; lean_object* v___x_1102_; lean_object* v___x_1103_; lean_object* v___x_1104_; lean_object* v___x_1105_; lean_object* v___x_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; lean_object* v___x_1109_; lean_object* v___x_1110_; lean_object* v___x_1111_; lean_object* v___x_1112_; lean_object* v___x_1113_; lean_object* v___x_1114_; lean_object* v___x_1115_; lean_object* v___x_1116_; lean_object* v___x_1117_; lean_object* v___x_1118_; lean_object* v___x_1119_; lean_object* v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; lean_object* v___x_1123_; lean_object* v___x_1124_; lean_object* v___x_1125_; lean_object* v___x_1126_; 
v_ref_1079_ = lean_ctor_get(v___y_1077_, 5);
v_quotContext_1080_ = lean_ctor_get(v___y_1077_, 10);
v_currMacroScope_1081_ = lean_ctor_get(v___y_1077_, 11);
v___x_1082_ = l_Lean_SourceInfo_fromRef(v_ref_1079_, v___x_903_);
v___x_1083_ = ((lean_object*)(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4));
v___x_1084_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19);
v___x_1085_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21));
lean_inc_n(v_currMacroScope_1081_, 3);
lean_inc_n(v_quotContext_1080_, 3);
v___x_1086_ = l_Lean_addMacroScope(v_quotContext_1080_, v___x_1085_, v_currMacroScope_1081_);
v___x_1087_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__23));
lean_inc_n(v___x_1082_, 18);
v___x_1088_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1088_, 0, v___x_1082_);
lean_ctor_set(v___x_1088_, 1, v___x_1084_);
lean_ctor_set(v___x_1088_, 2, v___x_1086_);
lean_ctor_set(v___x_1088_, 3, v___x_1087_);
v___x_1089_ = ((lean_object*)(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13));
v___x_1090_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25));
v___x_1091_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27));
v___x_1092_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__28));
v___x_1093_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1093_, 0, v___x_1082_);
lean_ctor_set(v___x_1093_, 1, v___x_1092_);
v___x_1094_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__30));
v___x_1095_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32);
v___x_1096_ = lean_box(0);
v___x_1097_ = l_Lean_addMacroScope(v_quotContext_1080_, v___x_1096_, v_currMacroScope_1081_);
v___x_1098_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__62));
v___x_1099_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1099_, 0, v___x_1082_);
lean_ctor_set(v___x_1099_, 1, v___x_1095_);
lean_ctor_set(v___x_1099_, 2, v___x_1097_);
lean_ctor_set(v___x_1099_, 3, v___x_1098_);
v___x_1100_ = l_Lean_Syntax_node1(v___x_1082_, v___x_1094_, v___x_1099_);
v___x_1101_ = l_Lean_Syntax_node2(v___x_1082_, v___x_1091_, v___x_1093_, v___x_1100_);
v___x_1102_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__63));
v___x_1103_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64));
v___x_1104_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1104_, 0, v___x_1082_);
lean_ctor_set(v___x_1104_, 1, v___x_1102_);
v___x_1105_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66));
v___x_1106_ = l_Lean_Syntax_node1(v___x_1082_, v___x_1089_, v___x_894_);
v___x_1107_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67);
v___x_1108_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1108_, 0, v___x_1082_);
lean_ctor_set(v___x_1108_, 1, v___x_1089_);
lean_ctor_set(v___x_1108_, 2, v___x_1107_);
v___x_1109_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__68));
v___x_1110_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1110_, 0, v___x_1082_);
lean_ctor_set(v___x_1110_, 1, v___x_1109_);
v___x_1111_ = l_Lean_Syntax_node4(v___x_1082_, v___x_1105_, v___x_1106_, v___x_1108_, v___x_1110_, v___x_1071_);
v___x_1112_ = l_Lean_Syntax_node2(v___x_1082_, v___x_1103_, v___x_1104_, v___x_1111_);
v___x_1113_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__69));
v___x_1114_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1114_, 0, v___x_1082_);
lean_ctor_set(v___x_1114_, 1, v___x_1113_);
lean_inc_ref(v___x_1114_);
lean_inc(v___x_1101_);
v___x_1115_ = l_Lean_Syntax_node3(v___x_1082_, v___x_1090_, v___x_1101_, v___x_1112_, v___x_1114_);
v___x_1116_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__83, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__83_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__83);
v___x_1117_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__85));
v___x_1118_ = l_Lean_addMacroScope(v_quotContext_1080_, v___x_1117_, v_currMacroScope_1081_);
v___x_1119_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__87));
v___x_1120_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1120_, 0, v___x_1082_);
lean_ctor_set(v___x_1120_, 1, v___x_1116_);
lean_ctor_set(v___x_1120_, 2, v___x_1118_);
lean_ctor_set(v___x_1120_, 3, v___x_1119_);
v___x_1121_ = l_Lean_Syntax_node1(v___x_1082_, v___x_1089_, v___x_1069_);
v___x_1122_ = l_Lean_Syntax_node2(v___x_1082_, v___x_1083_, v___x_1120_, v___x_1121_);
v___x_1123_ = l_Lean_Syntax_node3(v___x_1082_, v___x_1090_, v___x_1101_, v___x_1122_, v___x_1114_);
v___x_1124_ = l_Lean_Syntax_node2(v___x_1082_, v___x_1089_, v___x_1115_, v___x_1123_);
v___x_1125_ = l_Lean_Syntax_node2(v___x_1082_, v___x_1083_, v___x_1088_, v___x_1124_);
v___x_1126_ = l_Lean_Elab_Term_elabTerm(v___x_1125_, v_x_873_, v___x_899_, v___x_899_, v___y_1073_, v___y_1074_, v___y_1075_, v___y_1076_, v___y_1077_, v___y_1078_);
return v___x_1126_;
}
}
else
{
lean_object* v_a_1137_; lean_object* v___x_1139_; uint8_t v_isShared_1140_; uint8_t v_isSharedCheck_1144_; 
lean_dec(v___x_901_);
lean_dec(v___x_894_);
lean_dec(v_x_873_);
lean_dec(v_x_872_);
v_a_1137_ = lean_ctor_get(v___x_1067_, 0);
v_isSharedCheck_1144_ = !lean_is_exclusive(v___x_1067_);
if (v_isSharedCheck_1144_ == 0)
{
v___x_1139_ = v___x_1067_;
v_isShared_1140_ = v_isSharedCheck_1144_;
goto v_resetjp_1138_;
}
else
{
lean_inc(v_a_1137_);
lean_dec(v___x_1067_);
v___x_1139_ = lean_box(0);
v_isShared_1140_ = v_isSharedCheck_1144_;
goto v_resetjp_1138_;
}
v_resetjp_1138_:
{
lean_object* v___x_1142_; 
if (v_isShared_1140_ == 0)
{
v___x_1142_ = v___x_1139_;
goto v_reusejp_1141_;
}
else
{
lean_object* v_reuseFailAlloc_1143_; 
v_reuseFailAlloc_1143_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1143_, 0, v_a_1137_);
v___x_1142_ = v_reuseFailAlloc_1143_;
goto v_reusejp_1141_;
}
v_reusejp_1141_:
{
return v___x_1142_;
}
}
}
}
}
else
{
lean_object* v___x_1145_; 
lean_inc(v_x_873_);
v___x_1145_ = lp_mathlib_Mathlib_Meta_knownToBeFinsetNotSet(v_x_873_, v_a_874_, v_a_875_, v_a_876_, v_a_877_, v_a_878_, v_a_879_);
if (lean_obj_tag(v___x_1145_) == 0)
{
lean_object* v_a_1146_; lean_object* v___x_1147_; lean_object* v___x_1148_; lean_object* v___x_1149_; lean_object* v___y_1151_; lean_object* v___y_1152_; lean_object* v___y_1153_; lean_object* v___y_1154_; lean_object* v___y_1155_; lean_object* v___y_1156_; uint8_t v___x_1206_; 
v_a_1146_ = lean_ctor_get(v___x_1145_, 0);
lean_inc(v_a_1146_);
lean_dec_ref_known(v___x_1145_, 1);
v___x_1147_ = l_Lean_Syntax_getArg(v___x_901_, v___x_884_);
lean_dec(v___x_901_);
v___x_1148_ = lean_unsigned_to_nat(3u);
v___x_1149_ = l_Lean_Syntax_getArg(v_x_872_, v___x_1148_);
lean_dec(v_x_872_);
v___x_1206_ = lean_unbox(v_a_1146_);
lean_dec(v_a_1146_);
if (v___x_1206_ == 0)
{
lean_object* v___x_1207_; lean_object* v_a_1208_; lean_object* v___x_1210_; uint8_t v_isShared_1211_; uint8_t v_isSharedCheck_1215_; 
lean_dec(v___x_1149_);
lean_dec(v___x_1147_);
lean_dec(v___x_894_);
lean_dec(v_x_873_);
v___x_1207_ = lp_mathlib_Lean_Elab_throwUnsupportedSyntax___at___00Mathlib_Meta_elabFinsetBuilderIxx_spec__0___redArg();
v_a_1208_ = lean_ctor_get(v___x_1207_, 0);
v_isSharedCheck_1215_ = !lean_is_exclusive(v___x_1207_);
if (v_isSharedCheck_1215_ == 0)
{
v___x_1210_ = v___x_1207_;
v_isShared_1211_ = v_isSharedCheck_1215_;
goto v_resetjp_1209_;
}
else
{
lean_inc(v_a_1208_);
lean_dec(v___x_1207_);
v___x_1210_ = lean_box(0);
v_isShared_1211_ = v_isSharedCheck_1215_;
goto v_resetjp_1209_;
}
v_resetjp_1209_:
{
lean_object* v___x_1213_; 
if (v_isShared_1211_ == 0)
{
v___x_1213_ = v___x_1210_;
goto v_reusejp_1212_;
}
else
{
lean_object* v_reuseFailAlloc_1214_; 
v_reuseFailAlloc_1214_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1214_, 0, v_a_1208_);
v___x_1213_ = v_reuseFailAlloc_1214_;
goto v_reusejp_1212_;
}
v_reusejp_1212_:
{
return v___x_1213_;
}
}
}
else
{
v___y_1151_ = v_a_874_;
v___y_1152_ = v_a_875_;
v___y_1153_ = v_a_876_;
v___y_1154_ = v_a_877_;
v___y_1155_ = v_a_878_;
v___y_1156_ = v_a_879_;
goto v___jp_1150_;
}
v___jp_1150_:
{
lean_object* v_ref_1157_; lean_object* v_quotContext_1158_; lean_object* v_currMacroScope_1159_; uint8_t v___x_1160_; lean_object* v___x_1161_; lean_object* v___x_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; lean_object* v___x_1166_; lean_object* v___x_1167_; lean_object* v___x_1168_; lean_object* v___x_1169_; lean_object* v___x_1170_; lean_object* v___x_1171_; lean_object* v___x_1172_; lean_object* v___x_1173_; lean_object* v___x_1174_; lean_object* v___x_1175_; lean_object* v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; lean_object* v___x_1179_; lean_object* v___x_1180_; lean_object* v___x_1181_; lean_object* v___x_1182_; lean_object* v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1185_; lean_object* v___x_1186_; lean_object* v___x_1187_; lean_object* v___x_1188_; lean_object* v___x_1189_; lean_object* v___x_1190_; lean_object* v___x_1191_; lean_object* v___x_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1196_; lean_object* v___x_1197_; lean_object* v___x_1198_; lean_object* v___x_1199_; lean_object* v___x_1200_; lean_object* v___x_1201_; lean_object* v___x_1202_; lean_object* v___x_1203_; lean_object* v___x_1204_; lean_object* v___x_1205_; 
v_ref_1157_ = lean_ctor_get(v___y_1155_, 5);
v_quotContext_1158_ = lean_ctor_get(v___y_1155_, 10);
v_currMacroScope_1159_ = lean_ctor_get(v___y_1155_, 11);
v___x_1160_ = 0;
v___x_1161_ = l_Lean_SourceInfo_fromRef(v_ref_1157_, v___x_1160_);
v___x_1162_ = ((lean_object*)(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__4));
v___x_1163_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__19);
v___x_1164_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__21));
lean_inc_n(v_currMacroScope_1159_, 3);
lean_inc_n(v_quotContext_1158_, 3);
v___x_1165_ = l_Lean_addMacroScope(v_quotContext_1158_, v___x_1164_, v_currMacroScope_1159_);
v___x_1166_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__23));
lean_inc_n(v___x_1161_, 18);
v___x_1167_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1167_, 0, v___x_1161_);
lean_ctor_set(v___x_1167_, 1, v___x_1163_);
lean_ctor_set(v___x_1167_, 2, v___x_1165_);
lean_ctor_set(v___x_1167_, 3, v___x_1166_);
v___x_1168_ = ((lean_object*)(lp_mathlib_FinsetInterval___aux__Mathlib__Order__Interval__Finset__Defs______macroRules__FinsetInterval__term_x5b_x5b___x2c___x5d_x5d__1___closed__13));
v___x_1169_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__25));
v___x_1170_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__27));
v___x_1171_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__28));
v___x_1172_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1172_, 0, v___x_1161_);
lean_ctor_set(v___x_1172_, 1, v___x_1171_);
v___x_1173_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__30));
v___x_1174_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__32);
v___x_1175_ = lean_box(0);
v___x_1176_ = l_Lean_addMacroScope(v_quotContext_1158_, v___x_1175_, v_currMacroScope_1159_);
v___x_1177_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__62));
v___x_1178_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1178_, 0, v___x_1161_);
lean_ctor_set(v___x_1178_, 1, v___x_1174_);
lean_ctor_set(v___x_1178_, 2, v___x_1176_);
lean_ctor_set(v___x_1178_, 3, v___x_1177_);
v___x_1179_ = l_Lean_Syntax_node1(v___x_1161_, v___x_1173_, v___x_1178_);
v___x_1180_ = l_Lean_Syntax_node2(v___x_1161_, v___x_1170_, v___x_1172_, v___x_1179_);
v___x_1181_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__63));
v___x_1182_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__64));
v___x_1183_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1183_, 0, v___x_1161_);
lean_ctor_set(v___x_1183_, 1, v___x_1181_);
v___x_1184_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__66));
v___x_1185_ = l_Lean_Syntax_node1(v___x_1161_, v___x_1168_, v___x_894_);
v___x_1186_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__67);
v___x_1187_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_1187_, 0, v___x_1161_);
lean_ctor_set(v___x_1187_, 1, v___x_1168_);
lean_ctor_set(v___x_1187_, 2, v___x_1186_);
v___x_1188_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__68));
v___x_1189_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1189_, 0, v___x_1161_);
lean_ctor_set(v___x_1189_, 1, v___x_1188_);
v___x_1190_ = l_Lean_Syntax_node4(v___x_1161_, v___x_1184_, v___x_1185_, v___x_1187_, v___x_1189_, v___x_1149_);
v___x_1191_ = l_Lean_Syntax_node2(v___x_1161_, v___x_1182_, v___x_1183_, v___x_1190_);
v___x_1192_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__69));
v___x_1193_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_1193_, 0, v___x_1161_);
lean_ctor_set(v___x_1193_, 1, v___x_1192_);
lean_inc_ref(v___x_1193_);
lean_inc(v___x_1180_);
v___x_1194_ = l_Lean_Syntax_node3(v___x_1161_, v___x_1169_, v___x_1180_, v___x_1191_, v___x_1193_);
v___x_1195_ = lean_obj_once(&lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__89, &lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__89_once, _init_lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__89);
v___x_1196_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__91));
v___x_1197_ = l_Lean_addMacroScope(v_quotContext_1158_, v___x_1196_, v_currMacroScope_1159_);
v___x_1198_ = ((lean_object*)(lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___closed__93));
v___x_1199_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_1199_, 0, v___x_1161_);
lean_ctor_set(v___x_1199_, 1, v___x_1195_);
lean_ctor_set(v___x_1199_, 2, v___x_1197_);
lean_ctor_set(v___x_1199_, 3, v___x_1198_);
v___x_1200_ = l_Lean_Syntax_node1(v___x_1161_, v___x_1168_, v___x_1147_);
v___x_1201_ = l_Lean_Syntax_node2(v___x_1161_, v___x_1162_, v___x_1199_, v___x_1200_);
v___x_1202_ = l_Lean_Syntax_node3(v___x_1161_, v___x_1169_, v___x_1180_, v___x_1201_, v___x_1193_);
v___x_1203_ = l_Lean_Syntax_node2(v___x_1161_, v___x_1168_, v___x_1194_, v___x_1202_);
v___x_1204_ = l_Lean_Syntax_node2(v___x_1161_, v___x_1162_, v___x_1167_, v___x_1203_);
v___x_1205_ = l_Lean_Elab_Term_elabTerm(v___x_1204_, v_x_873_, v___x_899_, v___x_899_, v___y_1151_, v___y_1152_, v___y_1153_, v___y_1154_, v___y_1155_, v___y_1156_);
return v___x_1205_;
}
}
else
{
lean_object* v_a_1216_; lean_object* v___x_1218_; uint8_t v_isShared_1219_; uint8_t v_isSharedCheck_1223_; 
lean_dec(v___x_901_);
lean_dec(v___x_894_);
lean_dec(v_x_873_);
lean_dec(v_x_872_);
v_a_1216_ = lean_ctor_get(v___x_1145_, 0);
v_isSharedCheck_1223_ = !lean_is_exclusive(v___x_1145_);
if (v_isSharedCheck_1223_ == 0)
{
v___x_1218_ = v___x_1145_;
v_isShared_1219_ = v_isSharedCheck_1223_;
goto v_resetjp_1217_;
}
else
{
lean_inc(v_a_1216_);
lean_dec(v___x_1145_);
v___x_1218_ = lean_box(0);
v_isShared_1219_ = v_isSharedCheck_1223_;
goto v_resetjp_1217_;
}
v_resetjp_1217_:
{
lean_object* v___x_1221_; 
if (v_isShared_1219_ == 0)
{
v___x_1221_ = v___x_1218_;
goto v_reusejp_1220_;
}
else
{
lean_object* v_reuseFailAlloc_1222_; 
v_reuseFailAlloc_1222_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_1222_, 0, v_a_1216_);
v___x_1221_ = v_reuseFailAlloc_1222_;
goto v_reusejp_1220_;
}
v_reusejp_1220_:
{
return v___x_1221_;
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
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx___boxed(lean_object* v_x_1224_, lean_object* v_x_1225_, lean_object* v_a_1226_, lean_object* v_a_1227_, lean_object* v_a_1228_, lean_object* v_a_1229_, lean_object* v_a_1230_, lean_object* v_a_1231_, lean_object* v_a_1232_){
_start:
{
lean_object* v_res_1233_; 
v_res_1233_ = lp_mathlib_Mathlib_Meta_elabFinsetBuilderIxx(v_x_1224_, v_x_1225_, v_a_1226_, v_a_1227_, v_a_1228_, v_a_1229_, v_a_1230_, v_a_1231_);
lean_dec(v_a_1231_);
lean_dec_ref(v_a_1230_);
lean_dec(v_a_1229_);
lean_dec_ref(v_a_1228_);
lean_dec(v_a_1227_);
lean_dec_ref(v_a_1226_);
return v_res_1233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIcc___redArg(lean_object* v_inst_1234_, lean_object* v_a_1235_, lean_object* v_b_1236_){
_start:
{
lean_object* v___x_1237_; lean_object* v___x_1238_; 
v___x_1237_ = lp_mathlib_Finset_Icc___redArg(v_inst_1234_, v_a_1235_, v_b_1236_);
v___x_1238_ = lp_mathlib_Fintype_subtype___redArg(v___x_1237_);
return v___x_1238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIcc(lean_object* v_00_u03b1_1239_, lean_object* v_inst_1240_, lean_object* v_inst_1241_, lean_object* v_a_1242_, lean_object* v_b_1243_){
_start:
{
lean_object* v___x_1244_; 
v___x_1244_ = lp_mathlib_Set_instFintypeIcc___redArg(v_inst_1241_, v_a_1242_, v_b_1243_);
return v___x_1244_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIcc___boxed(lean_object* v_00_u03b1_1245_, lean_object* v_inst_1246_, lean_object* v_inst_1247_, lean_object* v_a_1248_, lean_object* v_b_1249_){
_start:
{
lean_object* v_res_1250_; 
v_res_1250_ = lp_mathlib_Set_instFintypeIcc(v_00_u03b1_1245_, v_inst_1246_, v_inst_1247_, v_a_1248_, v_b_1249_);
lean_dec_ref(v_inst_1246_);
return v_res_1250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIco___redArg(lean_object* v_inst_1251_, lean_object* v_a_1252_, lean_object* v_b_1253_){
_start:
{
lean_object* v___x_1254_; lean_object* v___x_1255_; 
v___x_1254_ = lp_mathlib_Finset_Ico___redArg(v_inst_1251_, v_a_1252_, v_b_1253_);
v___x_1255_ = lp_mathlib_Fintype_subtype___redArg(v___x_1254_);
return v___x_1255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIco(lean_object* v_00_u03b1_1256_, lean_object* v_inst_1257_, lean_object* v_inst_1258_, lean_object* v_a_1259_, lean_object* v_b_1260_){
_start:
{
lean_object* v___x_1261_; 
v___x_1261_ = lp_mathlib_Set_instFintypeIco___redArg(v_inst_1258_, v_a_1259_, v_b_1260_);
return v___x_1261_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIco___boxed(lean_object* v_00_u03b1_1262_, lean_object* v_inst_1263_, lean_object* v_inst_1264_, lean_object* v_a_1265_, lean_object* v_b_1266_){
_start:
{
lean_object* v_res_1267_; 
v_res_1267_ = lp_mathlib_Set_instFintypeIco(v_00_u03b1_1262_, v_inst_1263_, v_inst_1264_, v_a_1265_, v_b_1266_);
lean_dec_ref(v_inst_1263_);
return v_res_1267_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoc___redArg(lean_object* v_inst_1268_, lean_object* v_b_1269_, lean_object* v_a_1270_){
_start:
{
lean_object* v___x_1271_; lean_object* v___x_1272_; 
v___x_1271_ = lp_mathlib_Finset_Ioc___redArg(v_inst_1268_, v_b_1269_, v_a_1270_);
v___x_1272_ = lp_mathlib_Fintype_subtype___redArg(v___x_1271_);
return v___x_1272_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoc(lean_object* v_00_u03b1_1273_, lean_object* v_inst_1274_, lean_object* v_inst_1275_, lean_object* v_b_1276_, lean_object* v_a_1277_){
_start:
{
lean_object* v___x_1278_; 
v___x_1278_ = lp_mathlib_Set_instFintypeIoc___redArg(v_inst_1275_, v_b_1276_, v_a_1277_);
return v___x_1278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoc___boxed(lean_object* v_00_u03b1_1279_, lean_object* v_inst_1280_, lean_object* v_inst_1281_, lean_object* v_b_1282_, lean_object* v_a_1283_){
_start:
{
lean_object* v_res_1284_; 
v_res_1284_ = lp_mathlib_Set_instFintypeIoc(v_00_u03b1_1279_, v_inst_1280_, v_inst_1281_, v_b_1282_, v_a_1283_);
lean_dec_ref(v_inst_1280_);
return v_res_1284_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoo___redArg(lean_object* v_inst_1285_, lean_object* v_a_1286_, lean_object* v_b_1287_){
_start:
{
lean_object* v___x_1288_; lean_object* v___x_1289_; 
v___x_1288_ = lp_mathlib_Finset_Ioo___redArg(v_inst_1285_, v_a_1286_, v_b_1287_);
v___x_1289_ = lp_mathlib_Fintype_subtype___redArg(v___x_1288_);
return v___x_1289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoo(lean_object* v_00_u03b1_1290_, lean_object* v_inst_1291_, lean_object* v_inst_1292_, lean_object* v_a_1293_, lean_object* v_b_1294_){
_start:
{
lean_object* v___x_1295_; 
v___x_1295_ = lp_mathlib_Set_instFintypeIoo___redArg(v_inst_1292_, v_a_1293_, v_b_1294_);
return v___x_1295_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoo___boxed(lean_object* v_00_u03b1_1296_, lean_object* v_inst_1297_, lean_object* v_inst_1298_, lean_object* v_a_1299_, lean_object* v_b_1300_){
_start:
{
lean_object* v_res_1301_; 
v_res_1301_ = lp_mathlib_Set_instFintypeIoo(v_00_u03b1_1296_, v_inst_1297_, v_inst_1298_, v_a_1299_, v_b_1300_);
lean_dec_ref(v_inst_1297_);
return v_res_1301_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIci___redArg(lean_object* v_inst_1302_, lean_object* v_a_1303_){
_start:
{
lean_object* v___x_1304_; lean_object* v___x_1305_; 
v___x_1304_ = lp_mathlib_Finset_Ici___redArg(v_inst_1302_, v_a_1303_);
v___x_1305_ = lp_mathlib_Fintype_subtype___redArg(v___x_1304_);
return v___x_1305_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIci(lean_object* v_00_u03b1_1306_, lean_object* v_inst_1307_, lean_object* v_inst_1308_, lean_object* v_a_1309_){
_start:
{
lean_object* v___x_1310_; 
v___x_1310_ = lp_mathlib_Set_instFintypeIci___redArg(v_inst_1308_, v_a_1309_);
return v___x_1310_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIci___boxed(lean_object* v_00_u03b1_1311_, lean_object* v_inst_1312_, lean_object* v_inst_1313_, lean_object* v_a_1314_){
_start:
{
lean_object* v_res_1315_; 
v_res_1315_ = lp_mathlib_Set_instFintypeIci(v_00_u03b1_1311_, v_inst_1312_, v_inst_1313_, v_a_1314_);
lean_dec_ref(v_inst_1312_);
return v_res_1315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIic___redArg(lean_object* v_inst_1316_, lean_object* v_a_1317_){
_start:
{
lean_object* v___x_1318_; lean_object* v___x_1319_; 
v___x_1318_ = lp_mathlib_Finset_Iic___redArg(v_inst_1316_, v_a_1317_);
v___x_1319_ = lp_mathlib_Fintype_subtype___redArg(v___x_1318_);
return v___x_1319_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIic(lean_object* v_00_u03b1_1320_, lean_object* v_inst_1321_, lean_object* v_inst_1322_, lean_object* v_a_1323_){
_start:
{
lean_object* v___x_1324_; 
v___x_1324_ = lp_mathlib_Set_instFintypeIic___redArg(v_inst_1322_, v_a_1323_);
return v___x_1324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIic___boxed(lean_object* v_00_u03b1_1325_, lean_object* v_inst_1326_, lean_object* v_inst_1327_, lean_object* v_a_1328_){
_start:
{
lean_object* v_res_1329_; 
v_res_1329_ = lp_mathlib_Set_instFintypeIic(v_00_u03b1_1325_, v_inst_1326_, v_inst_1327_, v_a_1328_);
lean_dec_ref(v_inst_1326_);
return v_res_1329_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoi___redArg(lean_object* v_inst_1330_, lean_object* v_a_1331_){
_start:
{
lean_object* v___x_1332_; lean_object* v___x_1333_; 
v___x_1332_ = lp_mathlib_Finset_Ioi___redArg(v_inst_1330_, v_a_1331_);
v___x_1333_ = lp_mathlib_Fintype_subtype___redArg(v___x_1332_);
return v___x_1333_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoi(lean_object* v_00_u03b1_1334_, lean_object* v_inst_1335_, lean_object* v_inst_1336_, lean_object* v_a_1337_){
_start:
{
lean_object* v___x_1338_; 
v___x_1338_ = lp_mathlib_Set_instFintypeIoi___redArg(v_inst_1336_, v_a_1337_);
return v___x_1338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIoi___boxed(lean_object* v_00_u03b1_1339_, lean_object* v_inst_1340_, lean_object* v_inst_1341_, lean_object* v_a_1342_){
_start:
{
lean_object* v_res_1343_; 
v_res_1343_ = lp_mathlib_Set_instFintypeIoi(v_00_u03b1_1339_, v_inst_1340_, v_inst_1341_, v_a_1342_);
lean_dec_ref(v_inst_1340_);
return v_res_1343_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIio___redArg(lean_object* v_inst_1344_, lean_object* v_a_1345_){
_start:
{
lean_object* v___x_1346_; lean_object* v___x_1347_; 
v___x_1346_ = lp_mathlib_Finset_Iio___redArg(v_inst_1344_, v_a_1345_);
v___x_1347_ = lp_mathlib_Fintype_subtype___redArg(v___x_1346_);
return v___x_1347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIio(lean_object* v_00_u03b1_1348_, lean_object* v_inst_1349_, lean_object* v_inst_1350_, lean_object* v_a_1351_){
_start:
{
lean_object* v___x_1352_; 
v___x_1352_ = lp_mathlib_Set_instFintypeIio___redArg(v_inst_1350_, v_a_1351_);
return v___x_1352_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instFintypeIio___boxed(lean_object* v_00_u03b1_1353_, lean_object* v_inst_1354_, lean_object* v_inst_1355_, lean_object* v_a_1356_){
_start:
{
lean_object* v_res_1357_; 
v_res_1357_ = lp_mathlib_Set_instFintypeIio(v_00_u03b1_1353_, v_inst_1354_, v_inst_1355_, v_a_1356_);
lean_dec_ref(v_inst_1354_);
return v_res_1357_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeUIcc___redArg(lean_object* v_inst_1358_, lean_object* v_inst_1359_, lean_object* v_a_1360_, lean_object* v_b_1361_){
_start:
{
lean_object* v___x_1362_; lean_object* v___x_1363_; 
v___x_1362_ = lp_mathlib_Finset_uIcc___redArg(v_inst_1358_, v_inst_1359_, v_a_1360_, v_b_1361_);
v___x_1363_ = lp_mathlib_Fintype_subtype___redArg(v___x_1362_);
return v___x_1363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_fintypeUIcc(lean_object* v_00_u03b1_1364_, lean_object* v_inst_1365_, lean_object* v_inst_1366_, lean_object* v_a_1367_, lean_object* v_b_1368_){
_start:
{
lean_object* v___x_1369_; 
v___x_1369_ = lp_mathlib_Set_fintypeUIcc___redArg(v_inst_1365_, v_inst_1366_, v_a_1367_, v_b_1368_);
return v___x_1369_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__0(lean_object* v_inst_1370_, lean_object* v_a_1371_, lean_object* v_b_1372_, lean_object* v_a_1373_){
_start:
{
lean_object* v___x_1374_; uint8_t v___x_1375_; 
lean_inc_ref(v_inst_1370_);
lean_inc(v_a_1373_);
v___x_1374_ = lean_apply_2(v_inst_1370_, v_a_1371_, v_a_1373_);
v___x_1375_ = lean_unbox(v___x_1374_);
if (v___x_1375_ == 0)
{
uint8_t v___x_1376_; 
lean_dec(v_a_1373_);
lean_dec(v_b_1372_);
lean_dec_ref(v_inst_1370_);
v___x_1376_ = lean_unbox(v___x_1374_);
return v___x_1376_;
}
else
{
lean_object* v___x_1377_; uint8_t v___x_1378_; 
v___x_1377_ = lean_apply_2(v_inst_1370_, v_a_1373_, v_b_1372_);
v___x_1378_ = lean_unbox(v___x_1377_);
return v___x_1378_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__0___boxed(lean_object* v_inst_1379_, lean_object* v_a_1380_, lean_object* v_b_1381_, lean_object* v_a_1382_){
_start:
{
uint8_t v_res_1383_; lean_object* v_r_1384_; 
v_res_1383_ = lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__0(v_inst_1379_, v_a_1380_, v_b_1381_, v_a_1382_);
v_r_1384_ = lean_box(v_res_1383_);
return v_r_1384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__1(lean_object* v_inst_1385_, lean_object* v_inst_1386_, lean_object* v_a_1387_, lean_object* v_b_1388_){
_start:
{
lean_object* v___f_1389_; lean_object* v___x_1390_; lean_object* v___x_1391_; 
v___f_1389_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__0___boxed), 4, 3);
lean_closure_set(v___f_1389_, 0, v_inst_1385_);
lean_closure_set(v___f_1389_, 1, v_a_1387_);
lean_closure_set(v___f_1389_, 2, v_b_1388_);
v___x_1390_ = lp_mathlib_Subtype_fintype___redArg(v___f_1389_, v_inst_1386_);
v___x_1391_ = lp_mathlib_Set_toFinset___redArg(v___x_1390_);
return v___x_1391_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__2(lean_object* v_inst_1392_, lean_object* v_a_1393_, lean_object* v_inst_1394_, lean_object* v_b_1395_, lean_object* v_a_1396_){
_start:
{
lean_object* v___x_1397_; uint8_t v___x_1398_; 
lean_inc(v_a_1396_);
v___x_1397_ = lean_apply_2(v_inst_1392_, v_a_1393_, v_a_1396_);
v___x_1398_ = lean_unbox(v___x_1397_);
if (v___x_1398_ == 0)
{
uint8_t v___x_1399_; 
lean_dec(v_a_1396_);
lean_dec(v_b_1395_);
lean_dec_ref(v_inst_1394_);
v___x_1399_ = lean_unbox(v___x_1397_);
return v___x_1399_;
}
else
{
lean_object* v___x_1400_; uint8_t v___x_1401_; 
v___x_1400_ = lean_apply_2(v_inst_1394_, v_a_1396_, v_b_1395_);
v___x_1401_ = lean_unbox(v___x_1400_);
return v___x_1401_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__2___boxed(lean_object* v_inst_1402_, lean_object* v_a_1403_, lean_object* v_inst_1404_, lean_object* v_b_1405_, lean_object* v_a_1406_){
_start:
{
uint8_t v_res_1407_; lean_object* v_r_1408_; 
v_res_1407_ = lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__2(v_inst_1402_, v_a_1403_, v_inst_1404_, v_b_1405_, v_a_1406_);
v_r_1408_ = lean_box(v_res_1407_);
return v_r_1408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__3(lean_object* v_inst_1409_, lean_object* v_inst_1410_, lean_object* v_inst_1411_, lean_object* v_a_1412_, lean_object* v_b_1413_){
_start:
{
lean_object* v___f_1414_; lean_object* v___x_1415_; lean_object* v___x_1416_; 
v___f_1414_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__2___boxed), 5, 4);
lean_closure_set(v___f_1414_, 0, v_inst_1409_);
lean_closure_set(v___f_1414_, 1, v_a_1412_);
lean_closure_set(v___f_1414_, 2, v_inst_1410_);
lean_closure_set(v___f_1414_, 3, v_b_1413_);
v___x_1415_ = lp_mathlib_Subtype_fintype___redArg(v___f_1414_, v_inst_1411_);
v___x_1416_ = lp_mathlib_Set_toFinset___redArg(v___x_1415_);
return v___x_1416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___redArg(lean_object* v_inst_1417_, lean_object* v_inst_1418_, lean_object* v_inst_1419_){
_start:
{
lean_object* v___f_1420_; lean_object* v___f_1421_; lean_object* v___f_1422_; lean_object* v___f_1423_; lean_object* v___x_1424_; 
lean_inc_n(v_inst_1417_, 3);
lean_inc_ref_n(v_inst_1419_, 2);
v___f_1420_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(v___f_1420_, 0, v_inst_1419_);
lean_closure_set(v___f_1420_, 1, v_inst_1417_);
lean_inc_ref_n(v_inst_1418_, 2);
v___f_1421_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__3), 5, 3);
lean_closure_set(v___f_1421_, 0, v_inst_1419_);
lean_closure_set(v___f_1421_, 1, v_inst_1418_);
lean_closure_set(v___f_1421_, 2, v_inst_1417_);
v___f_1422_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__3), 5, 3);
lean_closure_set(v___f_1422_, 0, v_inst_1418_);
lean_closure_set(v___f_1422_, 1, v_inst_1419_);
lean_closure_set(v___f_1422_, 2, v_inst_1417_);
v___f_1423_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(v___f_1423_, 0, v_inst_1418_);
lean_closure_set(v___f_1423_, 1, v_inst_1417_);
v___x_1424_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1424_, 0, v___f_1420_);
lean_ctor_set(v___x_1424_, 1, v___f_1421_);
lean_ctor_set(v___x_1424_, 2, v___f_1422_);
lean_ctor_set(v___x_1424_, 3, v___f_1423_);
return v___x_1424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder(lean_object* v_00_u03b1_1425_, lean_object* v_inst_1426_, lean_object* v_inst_1427_, lean_object* v_inst_1428_, lean_object* v_inst_1429_){
_start:
{
lean_object* v___f_1430_; lean_object* v___f_1431_; lean_object* v___f_1432_; lean_object* v___f_1433_; lean_object* v___x_1434_; 
lean_inc_n(v_inst_1427_, 3);
lean_inc_ref_n(v_inst_1429_, 2);
v___f_1430_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(v___f_1430_, 0, v_inst_1429_);
lean_closure_set(v___f_1430_, 1, v_inst_1427_);
lean_inc_ref_n(v_inst_1428_, 2);
v___f_1431_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__3), 5, 3);
lean_closure_set(v___f_1431_, 0, v_inst_1429_);
lean_closure_set(v___f_1431_, 1, v_inst_1428_);
lean_closure_set(v___f_1431_, 2, v_inst_1427_);
v___f_1432_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__3), 5, 3);
lean_closure_set(v___f_1432_, 0, v_inst_1428_);
lean_closure_set(v___f_1432_, 1, v_inst_1429_);
lean_closure_set(v___f_1432_, 2, v_inst_1427_);
v___f_1433_ = lean_alloc_closure((void*)(lp_mathlib_Fintype_toLocallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(v___f_1433_, 0, v_inst_1428_);
lean_closure_set(v___f_1433_, 1, v_inst_1427_);
v___x_1434_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1434_, 0, v___f_1430_);
lean_ctor_set(v___x_1434_, 1, v___f_1431_);
lean_ctor_set(v___x_1434_, 2, v___f_1432_);
lean_ctor_set(v___x_1434_, 3, v___f_1433_);
return v___x_1434_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_toLocallyFiniteOrder___boxed(lean_object* v_00_u03b1_1435_, lean_object* v_inst_1436_, lean_object* v_inst_1437_, lean_object* v_inst_1438_, lean_object* v_inst_1439_){
_start:
{
lean_object* v_res_1440_; 
v_res_1440_ = lp_mathlib_Fintype_toLocallyFiniteOrder(v_00_u03b1_1435_, v_inst_1436_, v_inst_1437_, v_inst_1438_, v_inst_1439_);
lean_dec_ref(v_inst_1436_);
return v_res_1440_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__0(lean_object* v_self_1441_, lean_object* v___y_1442_){
_start:
{
lean_object* v_toFun_1443_; lean_object* v___x_1444_; 
v_toFun_1443_ = lean_ctor_get(v_self_1441_, 0);
lean_inc(v_toFun_1443_);
lean_dec_ref(v_self_1441_);
v___x_1444_ = lean_apply_1(v_toFun_1443_, v___y_1442_);
return v___x_1444_;
}
}
static lean_object* _init_lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0(void){
_start:
{
lean_object* v___x_1445_; 
v___x_1445_ = lp_mathlib_Equiv_refl(lean_box(0));
return v___x_1445_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1(lean_object* v___f_1446_, lean_object* v_inst_1447_, lean_object* v_a_1448_, lean_object* v_b_1449_){
_start:
{
lean_object* v___x_1450_; lean_object* v___x_1451_; lean_object* v___x_1452_; lean_object* v___x_1453_; 
v___x_1450_ = lean_obj_once(&lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0, &lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0_once, _init_lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0);
lean_inc(v___f_1446_);
v___x_1451_ = lean_apply_2(v___f_1446_, v___x_1450_, v_b_1449_);
v___x_1452_ = lean_apply_2(v___f_1446_, v___x_1450_, v_a_1448_);
v___x_1453_ = lp_mathlib_Finset_Icc___redArg(v_inst_1447_, v___x_1451_, v___x_1452_);
return v___x_1453_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__3(lean_object* v___f_1454_, lean_object* v_inst_1455_, lean_object* v_a_1456_, lean_object* v_b_1457_){
_start:
{
lean_object* v___x_1458_; lean_object* v___x_1459_; lean_object* v___x_1460_; lean_object* v___x_1461_; 
v___x_1458_ = lean_obj_once(&lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0, &lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0_once, _init_lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0);
lean_inc(v___f_1454_);
v___x_1459_ = lean_apply_2(v___f_1454_, v___x_1458_, v_b_1457_);
v___x_1460_ = lean_apply_2(v___f_1454_, v___x_1458_, v_a_1456_);
v___x_1461_ = lp_mathlib_Finset_Ioc___redArg(v_inst_1455_, v___x_1459_, v___x_1460_);
return v___x_1461_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__4(lean_object* v___f_1462_, lean_object* v_inst_1463_, lean_object* v_a_1464_, lean_object* v_b_1465_){
_start:
{
lean_object* v___x_1466_; lean_object* v___x_1467_; lean_object* v___x_1468_; lean_object* v___x_1469_; 
v___x_1466_ = lean_obj_once(&lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0, &lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0_once, _init_lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0);
lean_inc(v___f_1462_);
v___x_1467_ = lean_apply_2(v___f_1462_, v___x_1466_, v_b_1465_);
v___x_1468_ = lean_apply_2(v___f_1462_, v___x_1466_, v_a_1464_);
v___x_1469_ = lp_mathlib_Finset_Ico___redArg(v_inst_1463_, v___x_1467_, v___x_1468_);
return v___x_1469_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__5(lean_object* v___f_1470_, lean_object* v_inst_1471_, lean_object* v_a_1472_, lean_object* v_b_1473_){
_start:
{
lean_object* v___x_1474_; lean_object* v___x_1475_; lean_object* v___x_1476_; lean_object* v___x_1477_; 
v___x_1474_ = lean_obj_once(&lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0, &lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0_once, _init_lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0);
lean_inc(v___f_1470_);
v___x_1475_ = lean_apply_2(v___f_1470_, v___x_1474_, v_b_1473_);
v___x_1476_ = lean_apply_2(v___f_1470_, v___x_1474_, v_a_1472_);
v___x_1477_ = lp_mathlib_Finset_Ioo___redArg(v_inst_1471_, v___x_1475_, v___x_1476_);
return v___x_1477_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg(lean_object* v_inst_1479_){
_start:
{
lean_object* v___f_1480_; lean_object* v___f_1481_; lean_object* v___f_1482_; lean_object* v___f_1483_; lean_object* v___f_1484_; lean_object* v___x_1485_; 
v___f_1480_ = ((lean_object*)(lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___closed__0));
lean_inc_ref_n(v_inst_1479_, 3);
v___f_1481_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(v___f_1481_, 0, v___f_1480_);
lean_closure_set(v___f_1481_, 1, v_inst_1479_);
v___f_1482_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__3), 4, 2);
lean_closure_set(v___f_1482_, 0, v___f_1480_);
lean_closure_set(v___f_1482_, 1, v_inst_1479_);
v___f_1483_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__4), 4, 2);
lean_closure_set(v___f_1483_, 0, v___f_1480_);
lean_closure_set(v___f_1483_, 1, v_inst_1479_);
v___f_1484_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__5), 4, 2);
lean_closure_set(v___f_1484_, 0, v___f_1480_);
lean_closure_set(v___f_1484_, 1, v_inst_1479_);
v___x_1485_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1485_, 0, v___f_1481_);
lean_ctor_set(v___x_1485_, 1, v___f_1482_);
lean_ctor_set(v___x_1485_, 2, v___f_1483_);
lean_ctor_set(v___x_1485_, 3, v___f_1484_);
return v___x_1485_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder(lean_object* v_00_u03b1_1486_, lean_object* v_inst_1487_, lean_object* v_inst_1488_){
_start:
{
lean_object* v___x_1489_; 
v___x_1489_ = lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg(v_inst_1488_);
return v___x_1489_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrder___boxed(lean_object* v_00_u03b1_1490_, lean_object* v_inst_1491_, lean_object* v_inst_1492_){
_start:
{
lean_object* v_res_1493_; 
v_res_1493_ = lp_mathlib_OrderDual_instLocallyFiniteOrder(v_00_u03b1_1490_, v_inst_1491_, v_inst_1492_);
lean_dec_ref(v_inst_1491_);
return v_res_1493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg___lam__0(lean_object* v_inst_1494_, lean_object* v_a_1495_){
_start:
{
lean_object* v___x_1496_; lean_object* v_toFun_1497_; lean_object* v___x_1498_; lean_object* v___x_1499_; 
v___x_1496_ = lean_obj_once(&lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0, &lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0_once, _init_lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0);
v_toFun_1497_ = lean_ctor_get(v___x_1496_, 0);
lean_inc(v_toFun_1497_);
v___x_1498_ = lean_apply_1(v_toFun_1497_, v_a_1495_);
v___x_1499_ = lp_mathlib_Finset_Ioi___redArg(v_inst_1494_, v___x_1498_);
return v___x_1499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg___lam__1(lean_object* v_inst_1500_, lean_object* v_a_1501_){
_start:
{
lean_object* v___x_1502_; lean_object* v_toFun_1503_; lean_object* v___x_1504_; lean_object* v___x_1505_; 
v___x_1502_ = lean_obj_once(&lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0, &lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0_once, _init_lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0);
v_toFun_1503_ = lean_ctor_get(v___x_1502_, 0);
lean_inc(v_toFun_1503_);
v___x_1504_ = lean_apply_1(v_toFun_1503_, v_a_1501_);
v___x_1505_ = lp_mathlib_Finset_Ici___redArg(v_inst_1500_, v___x_1504_);
return v___x_1505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg(lean_object* v_inst_1506_){
_start:
{
lean_object* v___f_1507_; lean_object* v___f_1508_; lean_object* v___x_1509_; 
lean_inc_ref(v_inst_1506_);
v___f_1507_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1507_, 0, v_inst_1506_);
v___f_1508_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1508_, 0, v_inst_1506_);
v___x_1509_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1509_, 0, v___f_1507_);
lean_ctor_set(v___x_1509_, 1, v___f_1508_);
return v___x_1509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot(lean_object* v_00_u03b1_1510_, lean_object* v_inst_1511_, lean_object* v_inst_1512_){
_start:
{
lean_object* v___x_1513_; 
v___x_1513_ = lp_mathlib_OrderDual_instLocallyFiniteOrderBot___redArg(v_inst_1512_);
return v___x_1513_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderBot___boxed(lean_object* v_00_u03b1_1514_, lean_object* v_inst_1515_, lean_object* v_inst_1516_){
_start:
{
lean_object* v_res_1517_; 
v_res_1517_ = lp_mathlib_OrderDual_instLocallyFiniteOrderBot(v_00_u03b1_1514_, v_inst_1515_, v_inst_1516_);
lean_dec_ref(v_inst_1515_);
return v_res_1517_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg___lam__0(lean_object* v_inst_1518_, lean_object* v_a_1519_){
_start:
{
lean_object* v___x_1520_; lean_object* v_toFun_1521_; lean_object* v___x_1522_; lean_object* v___x_1523_; 
v___x_1520_ = lean_obj_once(&lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0, &lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0_once, _init_lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0);
v_toFun_1521_ = lean_ctor_get(v___x_1520_, 0);
lean_inc(v_toFun_1521_);
v___x_1522_ = lean_apply_1(v_toFun_1521_, v_a_1519_);
v___x_1523_ = lp_mathlib_Finset_Iio___redArg(v_inst_1518_, v___x_1522_);
return v___x_1523_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg___lam__1(lean_object* v_inst_1524_, lean_object* v_a_1525_){
_start:
{
lean_object* v___x_1526_; lean_object* v_toFun_1527_; lean_object* v___x_1528_; lean_object* v___x_1529_; 
v___x_1526_ = lean_obj_once(&lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0, &lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0_once, _init_lp_mathlib_OrderDual_instLocallyFiniteOrder___redArg___lam__1___closed__0);
v_toFun_1527_ = lean_ctor_get(v___x_1526_, 0);
lean_inc(v_toFun_1527_);
v___x_1528_ = lean_apply_1(v_toFun_1527_, v_a_1525_);
v___x_1529_ = lp_mathlib_Finset_Iic___redArg(v_inst_1524_, v___x_1528_);
return v___x_1529_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg(lean_object* v_inst_1530_){
_start:
{
lean_object* v___f_1531_; lean_object* v___f_1532_; lean_object* v___x_1533_; 
lean_inc_ref(v_inst_1530_);
v___f_1531_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1531_, 0, v_inst_1530_);
v___f_1532_ = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg___lam__1), 2, 1);
lean_closure_set(v___f_1532_, 0, v_inst_1530_);
v___x_1533_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1533_, 0, v___f_1531_);
lean_ctor_set(v___x_1533_, 1, v___f_1532_);
return v___x_1533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop(lean_object* v_00_u03b1_1534_, lean_object* v_inst_1535_, lean_object* v_inst_1536_){
_start:
{
lean_object* v___x_1537_; 
v___x_1537_ = lp_mathlib_OrderDual_instLocallyFiniteOrderTop___redArg(v_inst_1536_);
return v___x_1537_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instLocallyFiniteOrderTop___boxed(lean_object* v_00_u03b1_1538_, lean_object* v_inst_1539_, lean_object* v_inst_1540_){
_start:
{
lean_object* v_res_1541_; 
v_res_1541_ = lp_mathlib_OrderDual_instLocallyFiniteOrderTop(v_00_u03b1_1538_, v_inst_1539_, v_inst_1540_);
lean_dec_ref(v_inst_1539_);
return v_res_1541_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrder___redArg___lam__0(lean_object* v_inst_1542_, lean_object* v_inst_1543_, lean_object* v_x_1544_, lean_object* v_y_1545_){
_start:
{
lean_object* v_fst_1546_; lean_object* v_snd_1547_; lean_object* v_fst_1548_; lean_object* v_snd_1549_; lean_object* v___x_1550_; lean_object* v___x_1551_; lean_object* v___x_1552_; 
v_fst_1546_ = lean_ctor_get(v_x_1544_, 0);
lean_inc(v_fst_1546_);
v_snd_1547_ = lean_ctor_get(v_x_1544_, 1);
lean_inc(v_snd_1547_);
lean_dec_ref(v_x_1544_);
v_fst_1548_ = lean_ctor_get(v_y_1545_, 0);
lean_inc(v_fst_1548_);
v_snd_1549_ = lean_ctor_get(v_y_1545_, 1);
lean_inc(v_snd_1549_);
lean_dec_ref(v_y_1545_);
v___x_1550_ = lp_mathlib_Finset_Icc___redArg(v_inst_1542_, v_fst_1546_, v_fst_1548_);
v___x_1551_ = lp_mathlib_Finset_Icc___redArg(v_inst_1543_, v_snd_1547_, v_snd_1549_);
v___x_1552_ = lp_mathlib_Multiset_product___redArg(v___x_1550_, v___x_1551_);
return v___x_1552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrder___redArg(lean_object* v_inst_1553_, lean_object* v_inst_1554_, lean_object* v_inst_1555_){
_start:
{
lean_object* v___f_1556_; lean_object* v___x_1557_; 
v___f_1556_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instLocallyFiniteOrder___redArg___lam__0), 4, 2);
lean_closure_set(v___f_1556_, 0, v_inst_1553_);
lean_closure_set(v___f_1556_, 1, v_inst_1554_);
v___x_1557_ = lp_mathlib_LocallyFiniteOrder_ofIcc_x27___redArg(v_inst_1555_, v___f_1556_);
return v___x_1557_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrder(lean_object* v_00_u03b1_1558_, lean_object* v_00_u03b2_1559_, lean_object* v_inst_1560_, lean_object* v_inst_1561_, lean_object* v_inst_1562_, lean_object* v_inst_1563_, lean_object* v_inst_1564_){
_start:
{
lean_object* v___x_1565_; 
v___x_1565_ = lp_mathlib_Prod_instLocallyFiniteOrder___redArg(v_inst_1562_, v_inst_1563_, v_inst_1564_);
return v___x_1565_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrder___boxed(lean_object* v_00_u03b1_1566_, lean_object* v_00_u03b2_1567_, lean_object* v_inst_1568_, lean_object* v_inst_1569_, lean_object* v_inst_1570_, lean_object* v_inst_1571_, lean_object* v_inst_1572_){
_start:
{
lean_object* v_res_1573_; 
v_res_1573_ = lp_mathlib_Prod_instLocallyFiniteOrder(v_00_u03b1_1566_, v_00_u03b2_1567_, v_inst_1568_, v_inst_1569_, v_inst_1570_, v_inst_1571_, v_inst_1572_);
lean_dec_ref(v_inst_1569_);
lean_dec_ref(v_inst_1568_);
return v_res_1573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderTop___redArg___lam__0(lean_object* v_inst_1574_, lean_object* v_inst_1575_, lean_object* v_x_1576_){
_start:
{
lean_object* v_fst_1577_; lean_object* v_snd_1578_; lean_object* v___x_1579_; lean_object* v___x_1580_; lean_object* v___x_1581_; 
v_fst_1577_ = lean_ctor_get(v_x_1576_, 0);
lean_inc(v_fst_1577_);
v_snd_1578_ = lean_ctor_get(v_x_1576_, 1);
lean_inc(v_snd_1578_);
lean_dec_ref(v_x_1576_);
v___x_1579_ = lp_mathlib_Finset_Ici___redArg(v_inst_1574_, v_fst_1577_);
v___x_1580_ = lp_mathlib_Finset_Ici___redArg(v_inst_1575_, v_snd_1578_);
v___x_1581_ = lp_mathlib_Multiset_product___redArg(v___x_1579_, v___x_1580_);
return v___x_1581_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderTop___redArg(lean_object* v_inst_1582_, lean_object* v_inst_1583_, lean_object* v_inst_1584_){
_start:
{
lean_object* v___f_1585_; lean_object* v___x_1586_; 
v___f_1585_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instLocallyFiniteOrderTop___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1585_, 0, v_inst_1582_);
lean_closure_set(v___f_1585_, 1, v_inst_1583_);
v___x_1586_ = lp_mathlib_LocallyFiniteOrderTop_ofIci_x27___redArg(v_inst_1584_, v___f_1585_);
return v___x_1586_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderTop(lean_object* v_00_u03b1_1587_, lean_object* v_00_u03b2_1588_, lean_object* v_inst_1589_, lean_object* v_inst_1590_, lean_object* v_inst_1591_, lean_object* v_inst_1592_, lean_object* v_inst_1593_){
_start:
{
lean_object* v___x_1594_; 
v___x_1594_ = lp_mathlib_Prod_instLocallyFiniteOrderTop___redArg(v_inst_1591_, v_inst_1592_, v_inst_1593_);
return v___x_1594_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderTop___boxed(lean_object* v_00_u03b1_1595_, lean_object* v_00_u03b2_1596_, lean_object* v_inst_1597_, lean_object* v_inst_1598_, lean_object* v_inst_1599_, lean_object* v_inst_1600_, lean_object* v_inst_1601_){
_start:
{
lean_object* v_res_1602_; 
v_res_1602_ = lp_mathlib_Prod_instLocallyFiniteOrderTop(v_00_u03b1_1595_, v_00_u03b2_1596_, v_inst_1597_, v_inst_1598_, v_inst_1599_, v_inst_1600_, v_inst_1601_);
lean_dec_ref(v_inst_1598_);
lean_dec_ref(v_inst_1597_);
return v_res_1602_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderBot___redArg___lam__0(lean_object* v_inst_1603_, lean_object* v_inst_1604_, lean_object* v_x_1605_){
_start:
{
lean_object* v_fst_1606_; lean_object* v_snd_1607_; lean_object* v___x_1608_; lean_object* v___x_1609_; lean_object* v___x_1610_; 
v_fst_1606_ = lean_ctor_get(v_x_1605_, 0);
lean_inc(v_fst_1606_);
v_snd_1607_ = lean_ctor_get(v_x_1605_, 1);
lean_inc(v_snd_1607_);
lean_dec_ref(v_x_1605_);
v___x_1608_ = lp_mathlib_Finset_Iic___redArg(v_inst_1603_, v_fst_1606_);
v___x_1609_ = lp_mathlib_Finset_Iic___redArg(v_inst_1604_, v_snd_1607_);
v___x_1610_ = lp_mathlib_Multiset_product___redArg(v___x_1608_, v___x_1609_);
return v___x_1610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderBot___redArg(lean_object* v_inst_1611_, lean_object* v_inst_1612_, lean_object* v_inst_1613_){
_start:
{
lean_object* v___f_1614_; lean_object* v___x_1615_; 
v___f_1614_ = lean_alloc_closure((void*)(lp_mathlib_Prod_instLocallyFiniteOrderBot___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1614_, 0, v_inst_1611_);
lean_closure_set(v___f_1614_, 1, v_inst_1612_);
v___x_1615_ = lp_mathlib_LocallyFiniteOrderBot_ofIic_x27___redArg(v_inst_1613_, v___f_1614_);
return v___x_1615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderBot(lean_object* v_00_u03b1_1616_, lean_object* v_00_u03b2_1617_, lean_object* v_inst_1618_, lean_object* v_inst_1619_, lean_object* v_inst_1620_, lean_object* v_inst_1621_, lean_object* v_inst_1622_){
_start:
{
lean_object* v___x_1623_; 
v___x_1623_ = lp_mathlib_Prod_instLocallyFiniteOrderBot___redArg(v_inst_1620_, v_inst_1621_, v_inst_1622_);
return v___x_1623_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instLocallyFiniteOrderBot___boxed(lean_object* v_00_u03b1_1624_, lean_object* v_00_u03b2_1625_, lean_object* v_inst_1626_, lean_object* v_inst_1627_, lean_object* v_inst_1628_, lean_object* v_inst_1629_, lean_object* v_inst_1630_){
_start:
{
lean_object* v_res_1631_; 
v_res_1631_ = lp_mathlib_Prod_instLocallyFiniteOrderBot(v_00_u03b1_1624_, v_00_u03b2_1625_, v_inst_1626_, v_inst_1627_, v_inst_1628_, v_inst_1629_, v_inst_1630_);
lean_dec_ref(v_inst_1627_);
lean_dec_ref(v_inst_1626_);
return v_res_1631_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_insertTop___lam__0(lean_object* v___x_1633_, lean_object* v_s_1634_){
_start:
{
lean_object* v___f_1635_; lean_object* v___x_1636_; lean_object* v___x_1637_; 
v___f_1635_ = ((lean_object*)(lp_mathlib_WithTop_insertTop___lam__0___closed__0));
v___x_1636_ = lp_mathlib_Finset_map___redArg(v___f_1635_, v_s_1634_);
v___x_1637_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1637_, 0, v___x_1633_);
lean_ctor_set(v___x_1637_, 1, v___x_1636_);
return v___x_1637_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_insertTop(lean_object* v_00_u03b1_1640_){
_start:
{
lean_object* v___f_1641_; 
v___f_1641_ = ((lean_object*)(lp_mathlib_WithTop_insertTop___closed__0));
return v___f_1641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_insertBot___lam__0(lean_object* v_s_1642_){
_start:
{
lean_object* v___x_1643_; lean_object* v___f_1644_; lean_object* v___x_1645_; lean_object* v___x_1646_; 
v___x_1643_ = lean_box(0);
v___f_1644_ = ((lean_object*)(lp_mathlib_WithTop_insertTop___lam__0___closed__0));
v___x_1645_ = lp_mathlib_Finset_map___redArg(v___f_1644_, v_s_1642_);
v___x_1646_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1646_, 0, v___x_1643_);
lean_ctor_set(v___x_1646_, 1, v___x_1645_);
return v___x_1646_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_insertBot(lean_object* v_00_u03b1_1648_){
_start:
{
lean_object* v___f_1649_; 
v___f_1649_ = ((lean_object*)(lp_mathlib_WithBot_insertBot___closed__0));
return v___f_1649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLocallyFiniteOrder___redArg___lam__0(lean_object* v___x_1650_, lean_object* v___x_1651_, lean_object* v_inst_1652_, lean_object* v_a_1653_, lean_object* v_b_1654_){
_start:
{
if (lean_obj_tag(v_a_1653_) == 0)
{
lean_dec_ref(v_inst_1652_);
lean_dec_ref(v___x_1651_);
if (lean_obj_tag(v_b_1654_) == 0)
{
lean_object* v___x_1655_; lean_object* v___x_1656_; 
v___x_1655_ = lean_box(0);
v___x_1656_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1656_, 0, v___x_1650_);
lean_ctor_set(v___x_1656_, 1, v___x_1655_);
return v___x_1656_;
}
else
{
lean_object* v___x_1657_; 
lean_dec_ref_known(v_b_1654_, 1);
lean_dec(v___x_1650_);
v___x_1657_ = lean_box(0);
return v___x_1657_;
}
}
else
{
lean_dec(v___x_1650_);
if (lean_obj_tag(v_b_1654_) == 0)
{
lean_object* v_val_1658_; lean_object* v___x_1659_; lean_object* v___x_227__overap_1660_; lean_object* v___x_1661_; 
lean_dec_ref(v_inst_1652_);
v_val_1658_ = lean_ctor_get(v_a_1653_, 0);
lean_inc(v_val_1658_);
lean_dec_ref_known(v_a_1653_, 1);
v___x_1659_ = lp_mathlib_Finset_Ici___redArg(v___x_1651_, v_val_1658_);
v___x_227__overap_1660_ = lp_mathlib_WithTop_insertTop(lean_box(0));
v___x_1661_ = lean_apply_1(v___x_227__overap_1660_, v___x_1659_);
return v___x_1661_;
}
else
{
lean_object* v_val_1662_; lean_object* v_val_1663_; lean_object* v___f_1664_; lean_object* v___x_1665_; lean_object* v___x_1666_; 
lean_dec_ref(v___x_1651_);
v_val_1662_ = lean_ctor_get(v_a_1653_, 0);
lean_inc(v_val_1662_);
lean_dec_ref_known(v_a_1653_, 1);
v_val_1663_ = lean_ctor_get(v_b_1654_, 0);
lean_inc(v_val_1663_);
lean_dec_ref_known(v_b_1654_, 1);
v___f_1664_ = ((lean_object*)(lp_mathlib_WithTop_insertTop___lam__0___closed__0));
v___x_1665_ = lp_mathlib_Finset_Icc___redArg(v_inst_1652_, v_val_1662_, v_val_1663_);
v___x_1666_ = lp_mathlib_Finset_map___redArg(v___f_1664_, v___x_1665_);
return v___x_1666_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLocallyFiniteOrder___redArg___lam__1(lean_object* v___x_1667_, lean_object* v_inst_1668_, lean_object* v_a_1669_, lean_object* v_b_1670_){
_start:
{
if (lean_obj_tag(v_a_1669_) == 0)
{
lean_object* v___x_1671_; 
lean_dec(v_b_1670_);
lean_dec_ref(v_inst_1668_);
lean_dec_ref(v___x_1667_);
v___x_1671_ = lean_box(0);
return v___x_1671_;
}
else
{
if (lean_obj_tag(v_b_1670_) == 0)
{
lean_object* v_val_1672_; lean_object* v___f_1673_; lean_object* v___x_1674_; lean_object* v___x_1675_; 
lean_dec_ref(v_inst_1668_);
v_val_1672_ = lean_ctor_get(v_a_1669_, 0);
lean_inc(v_val_1672_);
lean_dec_ref_known(v_a_1669_, 1);
v___f_1673_ = ((lean_object*)(lp_mathlib_WithTop_insertTop___lam__0___closed__0));
v___x_1674_ = lp_mathlib_Finset_Ici___redArg(v___x_1667_, v_val_1672_);
v___x_1675_ = lp_mathlib_Finset_map___redArg(v___f_1673_, v___x_1674_);
return v___x_1675_;
}
else
{
lean_object* v_val_1676_; lean_object* v_val_1677_; lean_object* v___f_1678_; lean_object* v___x_1679_; lean_object* v___x_1680_; 
lean_dec_ref(v___x_1667_);
v_val_1676_ = lean_ctor_get(v_a_1669_, 0);
lean_inc(v_val_1676_);
lean_dec_ref_known(v_a_1669_, 1);
v_val_1677_ = lean_ctor_get(v_b_1670_, 0);
lean_inc(v_val_1677_);
lean_dec_ref_known(v_b_1670_, 1);
v___f_1678_ = ((lean_object*)(lp_mathlib_WithTop_insertTop___lam__0___closed__0));
v___x_1679_ = lp_mathlib_Finset_Ico___redArg(v_inst_1668_, v_val_1676_, v_val_1677_);
v___x_1680_ = lp_mathlib_Finset_map___redArg(v___f_1678_, v___x_1679_);
return v___x_1680_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLocallyFiniteOrder___redArg___lam__2(lean_object* v___x_1681_, lean_object* v_inst_1682_, lean_object* v_a_1683_, lean_object* v_b_1684_){
_start:
{
if (lean_obj_tag(v_a_1683_) == 0)
{
lean_object* v___x_1685_; 
lean_dec(v_b_1684_);
lean_dec_ref(v_inst_1682_);
lean_dec_ref(v___x_1681_);
v___x_1685_ = lean_box(0);
return v___x_1685_;
}
else
{
if (lean_obj_tag(v_b_1684_) == 0)
{
lean_object* v_val_1686_; lean_object* v___x_1687_; lean_object* v___x_248__overap_1688_; lean_object* v___x_1689_; 
lean_dec_ref(v_inst_1682_);
v_val_1686_ = lean_ctor_get(v_a_1683_, 0);
lean_inc(v_val_1686_);
lean_dec_ref_known(v_a_1683_, 1);
v___x_1687_ = lp_mathlib_Finset_Ioi___redArg(v___x_1681_, v_val_1686_);
v___x_248__overap_1688_ = lp_mathlib_WithTop_insertTop(lean_box(0));
v___x_1689_ = lean_apply_1(v___x_248__overap_1688_, v___x_1687_);
return v___x_1689_;
}
else
{
lean_object* v_val_1690_; lean_object* v_val_1691_; lean_object* v___f_1692_; lean_object* v___x_1693_; lean_object* v___x_1694_; 
lean_dec_ref(v___x_1681_);
v_val_1690_ = lean_ctor_get(v_a_1683_, 0);
lean_inc(v_val_1690_);
lean_dec_ref_known(v_a_1683_, 1);
v_val_1691_ = lean_ctor_get(v_b_1684_, 0);
lean_inc(v_val_1691_);
lean_dec_ref_known(v_b_1684_, 1);
v___f_1692_ = ((lean_object*)(lp_mathlib_WithTop_insertTop___lam__0___closed__0));
v___x_1693_ = lp_mathlib_Finset_Ioc___redArg(v_inst_1682_, v_val_1690_, v_val_1691_);
v___x_1694_ = lp_mathlib_Finset_map___redArg(v___f_1692_, v___x_1693_);
return v___x_1694_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLocallyFiniteOrder___redArg___lam__3(lean_object* v___x_1695_, lean_object* v_inst_1696_, lean_object* v_a_1697_, lean_object* v_b_1698_){
_start:
{
if (lean_obj_tag(v_a_1697_) == 0)
{
lean_object* v___x_1699_; 
lean_dec(v_b_1698_);
lean_dec_ref(v_inst_1696_);
lean_dec_ref(v___x_1695_);
v___x_1699_ = lean_box(0);
return v___x_1699_;
}
else
{
if (lean_obj_tag(v_b_1698_) == 0)
{
lean_object* v_val_1700_; lean_object* v___f_1701_; lean_object* v___x_1702_; lean_object* v___x_1703_; 
lean_dec_ref(v_inst_1696_);
v_val_1700_ = lean_ctor_get(v_a_1697_, 0);
lean_inc(v_val_1700_);
lean_dec_ref_known(v_a_1697_, 1);
v___f_1701_ = ((lean_object*)(lp_mathlib_WithTop_insertTop___lam__0___closed__0));
v___x_1702_ = lp_mathlib_Finset_Ioi___redArg(v___x_1695_, v_val_1700_);
v___x_1703_ = lp_mathlib_Finset_map___redArg(v___f_1701_, v___x_1702_);
return v___x_1703_;
}
else
{
lean_object* v_val_1704_; lean_object* v_val_1705_; lean_object* v___f_1706_; lean_object* v___x_1707_; lean_object* v___x_1708_; 
lean_dec_ref(v___x_1695_);
v_val_1704_ = lean_ctor_get(v_a_1697_, 0);
lean_inc(v_val_1704_);
lean_dec_ref_known(v_a_1697_, 1);
v_val_1705_ = lean_ctor_get(v_b_1698_, 0);
lean_inc(v_val_1705_);
lean_dec_ref_known(v_b_1698_, 1);
v___f_1706_ = ((lean_object*)(lp_mathlib_WithTop_insertTop___lam__0___closed__0));
v___x_1707_ = lp_mathlib_Finset_Ioo___redArg(v_inst_1696_, v_val_1704_, v_val_1705_);
v___x_1708_ = lp_mathlib_Finset_map___redArg(v___f_1706_, v___x_1707_);
return v___x_1708_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLocallyFiniteOrder___redArg(lean_object* v_inst_1709_, lean_object* v_inst_1710_){
_start:
{
lean_object* v___x_1711_; lean_object* v___x_1712_; lean_object* v___f_1713_; lean_object* v___f_1714_; lean_object* v___f_1715_; lean_object* v___f_1716_; lean_object* v___x_1717_; 
v___x_1711_ = lean_box(0);
lean_inc_ref_n(v_inst_1710_, 4);
v___x_1712_ = lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderTop___redArg(v_inst_1710_, v_inst_1709_);
lean_inc_ref_n(v___x_1712_, 3);
v___f_1713_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_instLocallyFiniteOrder___redArg___lam__0), 5, 3);
lean_closure_set(v___f_1713_, 0, v___x_1711_);
lean_closure_set(v___f_1713_, 1, v___x_1712_);
lean_closure_set(v___f_1713_, 2, v_inst_1710_);
v___f_1714_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_instLocallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(v___f_1714_, 0, v___x_1712_);
lean_closure_set(v___f_1714_, 1, v_inst_1710_);
v___f_1715_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_instLocallyFiniteOrder___redArg___lam__2), 4, 2);
lean_closure_set(v___f_1715_, 0, v___x_1712_);
lean_closure_set(v___f_1715_, 1, v_inst_1710_);
v___f_1716_ = lean_alloc_closure((void*)(lp_mathlib_WithTop_instLocallyFiniteOrder___redArg___lam__3), 4, 2);
lean_closure_set(v___f_1716_, 0, v___x_1712_);
lean_closure_set(v___f_1716_, 1, v_inst_1710_);
v___x_1717_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1717_, 0, v___f_1713_);
lean_ctor_set(v___x_1717_, 1, v___f_1714_);
lean_ctor_set(v___x_1717_, 2, v___f_1715_);
lean_ctor_set(v___x_1717_, 3, v___f_1716_);
return v___x_1717_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLocallyFiniteOrder(lean_object* v_00_u03b1_1718_, lean_object* v_inst_1719_, lean_object* v_inst_1720_, lean_object* v_inst_1721_){
_start:
{
lean_object* v___x_1722_; 
v___x_1722_ = lp_mathlib_WithTop_instLocallyFiniteOrder___redArg(v_inst_1720_, v_inst_1721_);
return v___x_1722_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithTop_instLocallyFiniteOrder___boxed(lean_object* v_00_u03b1_1723_, lean_object* v_inst_1724_, lean_object* v_inst_1725_, lean_object* v_inst_1726_){
_start:
{
lean_object* v_res_1727_; 
v_res_1727_ = lp_mathlib_WithTop_instLocallyFiniteOrder(v_00_u03b1_1723_, v_inst_1724_, v_inst_1725_, v_inst_1726_);
lean_dec_ref(v_inst_1724_);
return v_res_1727_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder_match__1___redArg(lean_object* v_a_1728_, lean_object* v_b_1729_, lean_object* v_h__1_1730_, lean_object* v_h__2_1731_, lean_object* v_h__3_1732_, lean_object* v_h__4_1733_){
_start:
{
if (lean_obj_tag(v_a_1728_) == 0)
{
lean_dec(v_h__4_1733_);
lean_dec(v_h__3_1732_);
if (lean_obj_tag(v_b_1729_) == 0)
{
lean_object* v___x_1734_; lean_object* v___x_1735_; 
lean_dec(v_h__2_1731_);
v___x_1734_ = lean_box(0);
v___x_1735_ = lean_apply_1(v_h__1_1730_, v___x_1734_);
return v___x_1735_;
}
else
{
lean_object* v_val_1736_; lean_object* v___x_1737_; 
lean_dec(v_h__1_1730_);
v_val_1736_ = lean_ctor_get(v_b_1729_, 0);
lean_inc(v_val_1736_);
lean_dec_ref_known(v_b_1729_, 1);
v___x_1737_ = lean_apply_1(v_h__2_1731_, v_val_1736_);
return v___x_1737_;
}
}
else
{
lean_dec(v_h__2_1731_);
lean_dec(v_h__1_1730_);
if (lean_obj_tag(v_b_1729_) == 0)
{
lean_object* v_val_1738_; lean_object* v___x_1739_; 
lean_dec(v_h__4_1733_);
v_val_1738_ = lean_ctor_get(v_a_1728_, 0);
lean_inc(v_val_1738_);
lean_dec_ref_known(v_a_1728_, 1);
v___x_1739_ = lean_apply_1(v_h__3_1732_, v_val_1738_);
return v___x_1739_;
}
else
{
lean_object* v_val_1740_; lean_object* v_val_1741_; lean_object* v___x_1742_; 
lean_dec(v_h__3_1732_);
v_val_1740_ = lean_ctor_get(v_a_1728_, 0);
lean_inc(v_val_1740_);
lean_dec_ref_known(v_a_1728_, 1);
v_val_1741_ = lean_ctor_get(v_b_1729_, 0);
lean_inc(v_val_1741_);
lean_dec_ref_known(v_b_1729_, 1);
v___x_1742_ = lean_apply_2(v_h__4_1733_, v_val_1740_, v_val_1741_);
return v___x_1742_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder_match__1(lean_object* v_00_u03b1_1743_, lean_object* v_motive_1744_, lean_object* v_a_1745_, lean_object* v_b_1746_, lean_object* v_h__1_1747_, lean_object* v_h__2_1748_, lean_object* v_h__3_1749_, lean_object* v_h__4_1750_){
_start:
{
lean_object* v___x_1751_; 
v___x_1751_ = lp_mathlib_WithBot_instLocallyFiniteOrder_match__1___redArg(v_a_1745_, v_b_1746_, v_h__1_1747_, v_h__2_1748_, v_h__3_1749_, v_h__4_1750_);
return v___x_1751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder_match__3___redArg(lean_object* v_a_1752_, lean_object* v_b_1753_, lean_object* v_h__1_1754_, lean_object* v_h__2_1755_, lean_object* v_h__3_1756_){
_start:
{
if (lean_obj_tag(v_a_1752_) == 0)
{
lean_object* v___x_1757_; 
lean_dec(v_h__3_1756_);
lean_dec(v_h__2_1755_);
v___x_1757_ = lean_apply_1(v_h__1_1754_, v_b_1753_);
return v___x_1757_;
}
else
{
lean_dec(v_h__1_1754_);
if (lean_obj_tag(v_b_1753_) == 0)
{
lean_object* v_val_1758_; lean_object* v___x_1759_; 
lean_dec(v_h__3_1756_);
v_val_1758_ = lean_ctor_get(v_a_1752_, 0);
lean_inc(v_val_1758_);
lean_dec_ref_known(v_a_1752_, 1);
v___x_1759_ = lean_apply_1(v_h__2_1755_, v_val_1758_);
return v___x_1759_;
}
else
{
lean_object* v_val_1760_; lean_object* v_val_1761_; lean_object* v___x_1762_; 
lean_dec(v_h__2_1755_);
v_val_1760_ = lean_ctor_get(v_a_1752_, 0);
lean_inc(v_val_1760_);
lean_dec_ref_known(v_a_1752_, 1);
v_val_1761_ = lean_ctor_get(v_b_1753_, 0);
lean_inc(v_val_1761_);
lean_dec_ref_known(v_b_1753_, 1);
v___x_1762_ = lean_apply_2(v_h__3_1756_, v_val_1760_, v_val_1761_);
return v___x_1762_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder_match__3(lean_object* v_00_u03b1_1763_, lean_object* v_motive_1764_, lean_object* v_a_1765_, lean_object* v_b_1766_, lean_object* v_h__1_1767_, lean_object* v_h__2_1768_, lean_object* v_h__3_1769_){
_start:
{
lean_object* v___x_1770_; 
v___x_1770_ = lp_mathlib_WithBot_instLocallyFiniteOrder_match__3___redArg(v_a_1765_, v_b_1766_, v_h__1_1767_, v_h__2_1768_, v_h__3_1769_);
return v___x_1770_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder___redArg___lam__0(lean_object* v___x_1771_, lean_object* v_inst_1772_, lean_object* v_a_1773_, lean_object* v_b_1774_){
_start:
{
if (lean_obj_tag(v_a_1773_) == 0)
{
lean_dec_ref(v_inst_1772_);
lean_dec_ref(v___x_1771_);
if (lean_obj_tag(v_b_1774_) == 0)
{
lean_object* v___x_1775_; lean_object* v___x_1776_; 
v___x_1775_ = lean_box(0);
v___x_1776_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1776_, 0, v_b_1774_);
lean_ctor_set(v___x_1776_, 1, v___x_1775_);
return v___x_1776_;
}
else
{
lean_object* v___x_1777_; 
lean_dec_ref_known(v_b_1774_, 1);
v___x_1777_ = lean_box(0);
return v___x_1777_;
}
}
else
{
if (lean_obj_tag(v_b_1774_) == 0)
{
lean_object* v_val_1778_; lean_object* v___x_1779_; lean_object* v___x_1780_; 
lean_dec_ref(v_inst_1772_);
v_val_1778_ = lean_ctor_get(v_a_1773_, 0);
lean_inc(v_val_1778_);
lean_dec_ref_known(v_a_1773_, 1);
v___x_1779_ = lp_mathlib_Finset_Iic___redArg(v___x_1771_, v_val_1778_);
v___x_1780_ = lp_mathlib_WithBot_insertBot___lam__0(v___x_1779_);
return v___x_1780_;
}
else
{
lean_object* v_val_1781_; lean_object* v_val_1782_; lean_object* v___f_1783_; lean_object* v___x_1784_; lean_object* v___x_1785_; 
lean_dec_ref(v___x_1771_);
v_val_1781_ = lean_ctor_get(v_a_1773_, 0);
lean_inc(v_val_1781_);
lean_dec_ref_known(v_a_1773_, 1);
v_val_1782_ = lean_ctor_get(v_b_1774_, 0);
lean_inc(v_val_1782_);
lean_dec_ref_known(v_b_1774_, 1);
v___f_1783_ = ((lean_object*)(lp_mathlib_WithTop_insertTop___lam__0___closed__0));
v___x_1784_ = lp_mathlib_Finset_Icc___redArg(v_inst_1772_, v_val_1782_, v_val_1781_);
v___x_1785_ = lp_mathlib_Finset_map___redArg(v___f_1783_, v___x_1784_);
return v___x_1785_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder___redArg___lam__1(lean_object* v___x_1786_, lean_object* v_inst_1787_, lean_object* v_a_1788_, lean_object* v_b_1789_){
_start:
{
if (lean_obj_tag(v_a_1788_) == 0)
{
lean_object* v___x_1790_; 
lean_dec(v_b_1789_);
lean_dec_ref(v_inst_1787_);
lean_dec_ref(v___x_1786_);
v___x_1790_ = lean_box(0);
return v___x_1790_;
}
else
{
if (lean_obj_tag(v_b_1789_) == 0)
{
lean_object* v_val_1791_; lean_object* v___f_1792_; lean_object* v___x_1793_; lean_object* v___x_1794_; 
lean_dec_ref(v_inst_1787_);
v_val_1791_ = lean_ctor_get(v_a_1788_, 0);
lean_inc(v_val_1791_);
lean_dec_ref_known(v_a_1788_, 1);
v___f_1792_ = ((lean_object*)(lp_mathlib_WithTop_insertTop___lam__0___closed__0));
v___x_1793_ = lp_mathlib_Finset_Iic___redArg(v___x_1786_, v_val_1791_);
v___x_1794_ = lp_mathlib_Finset_map___redArg(v___f_1792_, v___x_1793_);
return v___x_1794_;
}
else
{
lean_object* v_val_1795_; lean_object* v_val_1796_; lean_object* v___f_1797_; lean_object* v___x_1798_; lean_object* v___x_1799_; 
lean_dec_ref(v___x_1786_);
v_val_1795_ = lean_ctor_get(v_a_1788_, 0);
lean_inc(v_val_1795_);
lean_dec_ref_known(v_a_1788_, 1);
v_val_1796_ = lean_ctor_get(v_b_1789_, 0);
lean_inc(v_val_1796_);
lean_dec_ref_known(v_b_1789_, 1);
v___f_1797_ = ((lean_object*)(lp_mathlib_WithTop_insertTop___lam__0___closed__0));
v___x_1798_ = lp_mathlib_Finset_Ioc___redArg(v_inst_1787_, v_val_1796_, v_val_1795_);
v___x_1799_ = lp_mathlib_Finset_map___redArg(v___f_1797_, v___x_1798_);
return v___x_1799_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder___redArg___lam__2(lean_object* v___x_1800_, lean_object* v_inst_1801_, lean_object* v_a_1802_, lean_object* v_b_1803_){
_start:
{
if (lean_obj_tag(v_a_1802_) == 0)
{
lean_object* v___x_1804_; 
lean_dec(v_b_1803_);
lean_dec_ref(v_inst_1801_);
lean_dec_ref(v___x_1800_);
v___x_1804_ = lean_box(0);
return v___x_1804_;
}
else
{
if (lean_obj_tag(v_b_1803_) == 0)
{
lean_object* v_val_1805_; lean_object* v___x_1806_; lean_object* v___x_1807_; 
lean_dec_ref(v_inst_1801_);
v_val_1805_ = lean_ctor_get(v_a_1802_, 0);
lean_inc(v_val_1805_);
lean_dec_ref_known(v_a_1802_, 1);
v___x_1806_ = lp_mathlib_Finset_Iio___redArg(v___x_1800_, v_val_1805_);
v___x_1807_ = lp_mathlib_WithBot_insertBot___lam__0(v___x_1806_);
return v___x_1807_;
}
else
{
lean_object* v_val_1808_; lean_object* v_val_1809_; lean_object* v___f_1810_; lean_object* v___x_1811_; lean_object* v___x_1812_; 
lean_dec_ref(v___x_1800_);
v_val_1808_ = lean_ctor_get(v_a_1802_, 0);
lean_inc(v_val_1808_);
lean_dec_ref_known(v_a_1802_, 1);
v_val_1809_ = lean_ctor_get(v_b_1803_, 0);
lean_inc(v_val_1809_);
lean_dec_ref_known(v_b_1803_, 1);
v___f_1810_ = ((lean_object*)(lp_mathlib_WithTop_insertTop___lam__0___closed__0));
v___x_1811_ = lp_mathlib_Finset_Ico___redArg(v_inst_1801_, v_val_1809_, v_val_1808_);
v___x_1812_ = lp_mathlib_Finset_map___redArg(v___f_1810_, v___x_1811_);
return v___x_1812_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder___redArg___lam__3(lean_object* v___x_1813_, lean_object* v_inst_1814_, lean_object* v_a_1815_, lean_object* v_b_1816_){
_start:
{
if (lean_obj_tag(v_a_1815_) == 0)
{
lean_object* v___x_1817_; 
lean_dec(v_b_1816_);
lean_dec_ref(v_inst_1814_);
lean_dec_ref(v___x_1813_);
v___x_1817_ = lean_box(0);
return v___x_1817_;
}
else
{
if (lean_obj_tag(v_b_1816_) == 0)
{
lean_object* v_val_1818_; lean_object* v___f_1819_; lean_object* v___x_1820_; lean_object* v___x_1821_; 
lean_dec_ref(v_inst_1814_);
v_val_1818_ = lean_ctor_get(v_a_1815_, 0);
lean_inc(v_val_1818_);
lean_dec_ref_known(v_a_1815_, 1);
v___f_1819_ = ((lean_object*)(lp_mathlib_WithTop_insertTop___lam__0___closed__0));
v___x_1820_ = lp_mathlib_Finset_Iio___redArg(v___x_1813_, v_val_1818_);
v___x_1821_ = lp_mathlib_Finset_map___redArg(v___f_1819_, v___x_1820_);
return v___x_1821_;
}
else
{
lean_object* v_val_1822_; lean_object* v_val_1823_; lean_object* v___f_1824_; lean_object* v___x_1825_; lean_object* v___x_1826_; 
lean_dec_ref(v___x_1813_);
v_val_1822_ = lean_ctor_get(v_a_1815_, 0);
lean_inc(v_val_1822_);
lean_dec_ref_known(v_a_1815_, 1);
v_val_1823_ = lean_ctor_get(v_b_1816_, 0);
lean_inc(v_val_1823_);
lean_dec_ref_known(v_b_1816_, 1);
v___f_1824_ = ((lean_object*)(lp_mathlib_WithTop_insertTop___lam__0___closed__0));
v___x_1825_ = lp_mathlib_Finset_Ioo___redArg(v_inst_1814_, v_val_1823_, v_val_1822_);
v___x_1826_ = lp_mathlib_Finset_map___redArg(v___f_1824_, v___x_1825_);
return v___x_1826_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder___redArg(lean_object* v_inst_1827_, lean_object* v_inst_1828_){
_start:
{
lean_object* v___x_1829_; lean_object* v___f_1830_; lean_object* v___f_1831_; lean_object* v___f_1832_; lean_object* v___f_1833_; lean_object* v___x_1834_; lean_object* v___x_1835_; lean_object* v___x_1836_; lean_object* v___x_1837_; lean_object* v___x_1838_; 
lean_inc_ref_n(v_inst_1828_, 4);
v___x_1829_ = lp_mathlib_LocallyFiniteOrder_toLocallyFiniteOrderBot___redArg(v_inst_1828_, v_inst_1827_);
lean_inc_ref_n(v___x_1829_, 3);
v___f_1830_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_instLocallyFiniteOrder___redArg___lam__0), 4, 2);
lean_closure_set(v___f_1830_, 0, v___x_1829_);
lean_closure_set(v___f_1830_, 1, v_inst_1828_);
v___f_1831_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_instLocallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(v___f_1831_, 0, v___x_1829_);
lean_closure_set(v___f_1831_, 1, v_inst_1828_);
v___f_1832_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_instLocallyFiniteOrder___redArg___lam__2), 4, 2);
lean_closure_set(v___f_1832_, 0, v___x_1829_);
lean_closure_set(v___f_1832_, 1, v_inst_1828_);
v___f_1833_ = lean_alloc_closure((void*)(lp_mathlib_WithBot_instLocallyFiniteOrder___redArg___lam__3), 4, 2);
lean_closure_set(v___f_1833_, 0, v___x_1829_);
lean_closure_set(v___f_1833_, 1, v_inst_1828_);
v___x_1834_ = lean_alloc_closure((void*)(lp_mathlib_Function_swap), 6, 4);
lean_closure_set(v___x_1834_, 0, lean_box(0));
lean_closure_set(v___x_1834_, 1, lean_box(0));
lean_closure_set(v___x_1834_, 2, lean_box(0));
lean_closure_set(v___x_1834_, 3, v___f_1830_);
v___x_1835_ = lean_alloc_closure((void*)(lp_mathlib_Function_swap), 6, 4);
lean_closure_set(v___x_1835_, 0, lean_box(0));
lean_closure_set(v___x_1835_, 1, lean_box(0));
lean_closure_set(v___x_1835_, 2, lean_box(0));
lean_closure_set(v___x_1835_, 3, v___f_1832_);
v___x_1836_ = lean_alloc_closure((void*)(lp_mathlib_Function_swap), 6, 4);
lean_closure_set(v___x_1836_, 0, lean_box(0));
lean_closure_set(v___x_1836_, 1, lean_box(0));
lean_closure_set(v___x_1836_, 2, lean_box(0));
lean_closure_set(v___x_1836_, 3, v___f_1831_);
v___x_1837_ = lean_alloc_closure((void*)(lp_mathlib_Function_swap), 6, 4);
lean_closure_set(v___x_1837_, 0, lean_box(0));
lean_closure_set(v___x_1837_, 1, lean_box(0));
lean_closure_set(v___x_1837_, 2, lean_box(0));
lean_closure_set(v___x_1837_, 3, v___f_1833_);
v___x_1838_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1838_, 0, v___x_1834_);
lean_ctor_set(v___x_1838_, 1, v___x_1835_);
lean_ctor_set(v___x_1838_, 2, v___x_1836_);
lean_ctor_set(v___x_1838_, 3, v___x_1837_);
return v___x_1838_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder(lean_object* v_00_u03b1_1839_, lean_object* v_inst_1840_, lean_object* v_inst_1841_, lean_object* v_inst_1842_){
_start:
{
lean_object* v___x_1843_; 
v___x_1843_ = lp_mathlib_WithBot_instLocallyFiniteOrder___redArg(v_inst_1841_, v_inst_1842_);
return v___x_1843_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_WithBot_instLocallyFiniteOrder___boxed(lean_object* v_00_u03b1_1844_, lean_object* v_inst_1845_, lean_object* v_inst_1846_, lean_object* v_inst_1847_){
_start:
{
lean_object* v_res_1848_; 
v_res_1848_ = lp_mathlib_WithBot_instLocallyFiniteOrder(v_00_u03b1_1844_, v_inst_1845_, v_inst_1846_, v_inst_1847_);
lean_dec_ref(v_inst_1845_);
return v_res_1848_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__0(lean_object* v_f_1849_, lean_object* v_inst_1850_, lean_object* v_a_1851_, lean_object* v_b_1852_){
_start:
{
lean_object* v___x_1853_; lean_object* v___f_1854_; lean_object* v___x_1855_; lean_object* v___x_1856_; lean_object* v___x_1857_; lean_object* v___x_1858_; 
lean_inc_ref_n(v_f_1849_, 2);
v___x_1853_ = lp_mathlib_Equiv_symm___redArg(v_f_1849_);
v___f_1854_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1854_, 0, v___x_1853_);
v___x_1855_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_f_1849_, v_a_1851_);
v___x_1856_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_f_1849_, v_b_1852_);
v___x_1857_ = lp_mathlib_Finset_Icc___redArg(v_inst_1850_, v___x_1855_, v___x_1856_);
v___x_1858_ = lp_mathlib_Finset_map___redArg(v___f_1854_, v___x_1857_);
return v___x_1858_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__1(lean_object* v_f_1859_, lean_object* v_inst_1860_, lean_object* v_a_1861_, lean_object* v_b_1862_){
_start:
{
lean_object* v___x_1863_; lean_object* v___f_1864_; lean_object* v___x_1865_; lean_object* v___x_1866_; lean_object* v___x_1867_; lean_object* v___x_1868_; 
lean_inc_ref_n(v_f_1859_, 2);
v___x_1863_ = lp_mathlib_Equiv_symm___redArg(v_f_1859_);
v___f_1864_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1864_, 0, v___x_1863_);
v___x_1865_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_f_1859_, v_a_1861_);
v___x_1866_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_f_1859_, v_b_1862_);
v___x_1867_ = lp_mathlib_Finset_Ico___redArg(v_inst_1860_, v___x_1865_, v___x_1866_);
v___x_1868_ = lp_mathlib_Finset_map___redArg(v___f_1864_, v___x_1867_);
return v___x_1868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__2(lean_object* v_f_1869_, lean_object* v_inst_1870_, lean_object* v_a_1871_, lean_object* v_b_1872_){
_start:
{
lean_object* v___x_1873_; lean_object* v___f_1874_; lean_object* v___x_1875_; lean_object* v___x_1876_; lean_object* v___x_1877_; lean_object* v___x_1878_; 
lean_inc_ref_n(v_f_1869_, 2);
v___x_1873_ = lp_mathlib_Equiv_symm___redArg(v_f_1869_);
v___f_1874_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1874_, 0, v___x_1873_);
v___x_1875_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_f_1869_, v_a_1871_);
v___x_1876_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_f_1869_, v_b_1872_);
v___x_1877_ = lp_mathlib_Finset_Ioc___redArg(v_inst_1870_, v___x_1875_, v___x_1876_);
v___x_1878_ = lp_mathlib_Finset_map___redArg(v___f_1874_, v___x_1877_);
return v___x_1878_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__3(lean_object* v_f_1879_, lean_object* v_inst_1880_, lean_object* v_a_1881_, lean_object* v_b_1882_){
_start:
{
lean_object* v___x_1883_; lean_object* v___f_1884_; lean_object* v___x_1885_; lean_object* v___x_1886_; lean_object* v___x_1887_; lean_object* v___x_1888_; 
lean_inc_ref_n(v_f_1879_, 2);
v___x_1883_ = lp_mathlib_Equiv_symm___redArg(v_f_1879_);
v___f_1884_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1884_, 0, v___x_1883_);
v___x_1885_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_f_1879_, v_a_1881_);
v___x_1886_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_f_1879_, v_b_1882_);
v___x_1887_ = lp_mathlib_Finset_Ioo___redArg(v_inst_1880_, v___x_1885_, v___x_1886_);
v___x_1888_ = lp_mathlib_Finset_map___redArg(v___f_1884_, v___x_1887_);
return v___x_1888_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___redArg(lean_object* v_inst_1889_, lean_object* v_f_1890_){
_start:
{
lean_object* v___f_1891_; lean_object* v___f_1892_; lean_object* v___f_1893_; lean_object* v___f_1894_; lean_object* v___x_1895_; 
lean_inc_ref_n(v_inst_1889_, 3);
lean_inc_ref_n(v_f_1890_, 3);
v___f_1891_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__0), 4, 2);
lean_closure_set(v___f_1891_, 0, v_f_1890_);
lean_closure_set(v___f_1891_, 1, v_inst_1889_);
v___f_1892_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(v___f_1892_, 0, v_f_1890_);
lean_closure_set(v___f_1892_, 1, v_inst_1889_);
v___f_1893_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__2), 4, 2);
lean_closure_set(v___f_1893_, 0, v_f_1890_);
lean_closure_set(v___f_1893_, 1, v_inst_1889_);
v___f_1894_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__3), 4, 2);
lean_closure_set(v___f_1894_, 0, v_f_1890_);
lean_closure_set(v___f_1894_, 1, v_inst_1889_);
v___x_1895_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1895_, 0, v___f_1891_);
lean_ctor_set(v___x_1895_, 1, v___f_1892_);
lean_ctor_set(v___x_1895_, 2, v___f_1893_);
lean_ctor_set(v___x_1895_, 3, v___f_1894_);
return v___x_1895_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder(lean_object* v_00_u03b1_1896_, lean_object* v_00_u03b2_1897_, lean_object* v_inst_1898_, lean_object* v_inst_1899_, lean_object* v_inst_1900_, lean_object* v_f_1901_){
_start:
{
lean_object* v___f_1902_; lean_object* v___f_1903_; lean_object* v___f_1904_; lean_object* v___f_1905_; lean_object* v___x_1906_; 
lean_inc_ref_n(v_inst_1900_, 3);
lean_inc_ref_n(v_f_1901_, 3);
v___f_1902_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__0), 4, 2);
lean_closure_set(v___f_1902_, 0, v_f_1901_);
lean_closure_set(v___f_1902_, 1, v_inst_1900_);
v___f_1903_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(v___f_1903_, 0, v_f_1901_);
lean_closure_set(v___f_1903_, 1, v_inst_1900_);
v___f_1904_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__2), 4, 2);
lean_closure_set(v___f_1904_, 0, v_f_1901_);
lean_closure_set(v___f_1904_, 1, v_inst_1900_);
v___f_1905_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrder___redArg___lam__3), 4, 2);
lean_closure_set(v___f_1905_, 0, v_f_1901_);
lean_closure_set(v___f_1905_, 1, v_inst_1900_);
v___x_1906_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_1906_, 0, v___f_1902_);
lean_ctor_set(v___x_1906_, 1, v___f_1903_);
lean_ctor_set(v___x_1906_, 2, v___f_1904_);
lean_ctor_set(v___x_1906_, 3, v___f_1905_);
return v___x_1906_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrder___boxed(lean_object* v_00_u03b1_1907_, lean_object* v_00_u03b2_1908_, lean_object* v_inst_1909_, lean_object* v_inst_1910_, lean_object* v_inst_1911_, lean_object* v_f_1912_){
_start:
{
lean_object* v_res_1913_; 
v_res_1913_ = lp_mathlib_OrderIso_locallyFiniteOrder(v_00_u03b1_1907_, v_00_u03b2_1908_, v_inst_1909_, v_inst_1910_, v_inst_1911_, v_f_1912_);
lean_dec_ref(v_inst_1910_);
lean_dec_ref(v_inst_1909_);
return v_res_1913_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg___lam__0(lean_object* v_f_1914_, lean_object* v_inst_1915_, lean_object* v_a_1916_){
_start:
{
lean_object* v___x_1917_; lean_object* v___f_1918_; lean_object* v___x_1919_; lean_object* v___x_1920_; lean_object* v___x_1921_; 
lean_inc_ref(v_f_1914_);
v___x_1917_ = lp_mathlib_Equiv_symm___redArg(v_f_1914_);
v___f_1918_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1918_, 0, v___x_1917_);
v___x_1919_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_f_1914_, v_a_1916_);
v___x_1920_ = lp_mathlib_Finset_Ioi___redArg(v_inst_1915_, v___x_1919_);
v___x_1921_ = lp_mathlib_Finset_map___redArg(v___f_1918_, v___x_1920_);
return v___x_1921_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg___lam__1(lean_object* v_f_1922_, lean_object* v_inst_1923_, lean_object* v_a_1924_){
_start:
{
lean_object* v___x_1925_; lean_object* v___f_1926_; lean_object* v___x_1927_; lean_object* v___x_1928_; lean_object* v___x_1929_; 
lean_inc_ref(v_f_1922_);
v___x_1925_ = lp_mathlib_Equiv_symm___redArg(v_f_1922_);
v___f_1926_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1926_, 0, v___x_1925_);
v___x_1927_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_f_1922_, v_a_1924_);
v___x_1928_ = lp_mathlib_Finset_Ici___redArg(v_inst_1923_, v___x_1927_);
v___x_1929_ = lp_mathlib_Finset_map___redArg(v___f_1926_, v___x_1928_);
return v___x_1929_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg(lean_object* v_inst_1930_, lean_object* v_f_1931_){
_start:
{
lean_object* v___f_1932_; lean_object* v___f_1933_; lean_object* v___x_1934_; 
lean_inc_ref(v_inst_1930_);
lean_inc_ref(v_f_1931_);
v___f_1932_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1932_, 0, v_f_1931_);
lean_closure_set(v___f_1932_, 1, v_inst_1930_);
v___f_1933_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg___lam__1), 3, 2);
lean_closure_set(v___f_1933_, 0, v_f_1931_);
lean_closure_set(v___f_1933_, 1, v_inst_1930_);
v___x_1934_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1934_, 0, v___f_1932_);
lean_ctor_set(v___x_1934_, 1, v___f_1933_);
return v___x_1934_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop(lean_object* v_00_u03b1_1935_, lean_object* v_00_u03b2_1936_, lean_object* v_inst_1937_, lean_object* v_inst_1938_, lean_object* v_inst_1939_, lean_object* v_f_1940_){
_start:
{
lean_object* v___f_1941_; lean_object* v___f_1942_; lean_object* v___x_1943_; 
lean_inc_ref(v_inst_1939_);
lean_inc_ref(v_f_1940_);
v___f_1941_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1941_, 0, v_f_1940_);
lean_closure_set(v___f_1941_, 1, v_inst_1939_);
v___f_1942_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrderTop___redArg___lam__1), 3, 2);
lean_closure_set(v___f_1942_, 0, v_f_1940_);
lean_closure_set(v___f_1942_, 1, v_inst_1939_);
v___x_1943_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1943_, 0, v___f_1941_);
lean_ctor_set(v___x_1943_, 1, v___f_1942_);
return v___x_1943_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderTop___boxed(lean_object* v_00_u03b1_1944_, lean_object* v_00_u03b2_1945_, lean_object* v_inst_1946_, lean_object* v_inst_1947_, lean_object* v_inst_1948_, lean_object* v_f_1949_){
_start:
{
lean_object* v_res_1950_; 
v_res_1950_ = lp_mathlib_OrderIso_locallyFiniteOrderTop(v_00_u03b1_1944_, v_00_u03b2_1945_, v_inst_1946_, v_inst_1947_, v_inst_1948_, v_f_1949_);
lean_dec_ref(v_inst_1947_);
lean_dec_ref(v_inst_1946_);
return v_res_1950_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg___lam__0(lean_object* v_f_1951_, lean_object* v_inst_1952_, lean_object* v_a_1953_){
_start:
{
lean_object* v___x_1954_; lean_object* v___f_1955_; lean_object* v___x_1956_; lean_object* v___x_1957_; lean_object* v___x_1958_; 
lean_inc_ref(v_f_1951_);
v___x_1954_ = lp_mathlib_Equiv_symm___redArg(v_f_1951_);
v___f_1955_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1955_, 0, v___x_1954_);
v___x_1956_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_f_1951_, v_a_1953_);
v___x_1957_ = lp_mathlib_Finset_Iio___redArg(v_inst_1952_, v___x_1956_);
v___x_1958_ = lp_mathlib_Finset_map___redArg(v___f_1955_, v___x_1957_);
return v___x_1958_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg___lam__1(lean_object* v_f_1959_, lean_object* v_inst_1960_, lean_object* v_a_1961_){
_start:
{
lean_object* v___x_1962_; lean_object* v___f_1963_; lean_object* v___x_1964_; lean_object* v___x_1965_; lean_object* v___x_1966_; 
lean_inc_ref(v_f_1959_);
v___x_1962_ = lp_mathlib_Equiv_symm___redArg(v_f_1959_);
v___f_1963_ = lean_alloc_closure((void*)(lp_mathlib_Equiv_toEmbedding___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1963_, 0, v___x_1962_);
v___x_1964_ = lp_mathlib_Equiv_toEmbedding___redArg___lam__0(v_f_1959_, v_a_1961_);
v___x_1965_ = lp_mathlib_Finset_Iic___redArg(v_inst_1960_, v___x_1964_);
v___x_1966_ = lp_mathlib_Finset_map___redArg(v___f_1963_, v___x_1965_);
return v___x_1966_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg(lean_object* v_inst_1967_, lean_object* v_f_1968_){
_start:
{
lean_object* v___f_1969_; lean_object* v___f_1970_; lean_object* v___x_1971_; 
lean_inc_ref(v_inst_1967_);
lean_inc_ref(v_f_1968_);
v___f_1969_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1969_, 0, v_f_1968_);
lean_closure_set(v___f_1969_, 1, v_inst_1967_);
v___f_1970_ = lean_alloc_closure((void*)(lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg___lam__1), 3, 2);
lean_closure_set(v___f_1970_, 0, v_f_1968_);
lean_closure_set(v___f_1970_, 1, v_inst_1967_);
v___x_1971_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1971_, 0, v___f_1969_);
lean_ctor_set(v___x_1971_, 1, v___f_1970_);
return v___x_1971_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot(lean_object* v_00_u03b1_1972_, lean_object* v_00_u03b2_1973_, lean_object* v_inst_1974_, lean_object* v_inst_1975_, lean_object* v_inst_1976_, lean_object* v_f_1977_){
_start:
{
lean_object* v___x_1978_; 
v___x_1978_ = lp_mathlib_OrderIso_locallyFiniteOrderBot___redArg(v_inst_1976_, v_f_1977_);
return v___x_1978_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderIso_locallyFiniteOrderBot___boxed(lean_object* v_00_u03b1_1979_, lean_object* v_00_u03b2_1980_, lean_object* v_inst_1981_, lean_object* v_inst_1982_, lean_object* v_inst_1983_, lean_object* v_f_1984_){
_start:
{
lean_object* v_res_1985_; 
v_res_1985_ = lp_mathlib_OrderIso_locallyFiniteOrderBot(v_00_u03b1_1979_, v_00_u03b2_1980_, v_inst_1981_, v_inst_1982_, v_inst_1983_, v_f_1984_);
lean_dec_ref(v_inst_1982_);
lean_dec_ref(v_inst_1981_);
return v_res_1985_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__0(lean_object* v_inst_1986_, lean_object* v_inst_1987_, lean_object* v_a_1988_, lean_object* v_b_1989_){
_start:
{
lean_object* v___x_1990_; lean_object* v___x_1991_; 
v___x_1990_ = lp_mathlib_Finset_Icc___redArg(v_inst_1986_, v_a_1988_, v_b_1989_);
v___x_1991_ = lp_mathlib_Finset_subtype___redArg(v_inst_1987_, v___x_1990_);
return v___x_1991_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__1(lean_object* v_inst_1992_, lean_object* v_inst_1993_, lean_object* v_a_1994_, lean_object* v_b_1995_){
_start:
{
lean_object* v___x_1996_; lean_object* v___x_1997_; 
v___x_1996_ = lp_mathlib_Finset_Ico___redArg(v_inst_1992_, v_a_1994_, v_b_1995_);
v___x_1997_ = lp_mathlib_Finset_subtype___redArg(v_inst_1993_, v___x_1996_);
return v___x_1997_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__2(lean_object* v_inst_1998_, lean_object* v_inst_1999_, lean_object* v_a_2000_, lean_object* v_b_2001_){
_start:
{
lean_object* v___x_2002_; lean_object* v___x_2003_; 
v___x_2002_ = lp_mathlib_Finset_Ioc___redArg(v_inst_1998_, v_a_2000_, v_b_2001_);
v___x_2003_ = lp_mathlib_Finset_subtype___redArg(v_inst_1999_, v___x_2002_);
return v___x_2003_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__3(lean_object* v_inst_2004_, lean_object* v_inst_2005_, lean_object* v_a_2006_, lean_object* v_b_2007_){
_start:
{
lean_object* v___x_2008_; lean_object* v___x_2009_; 
v___x_2008_ = lp_mathlib_Finset_Ioo___redArg(v_inst_2004_, v_a_2006_, v_b_2007_);
v___x_2009_ = lp_mathlib_Finset_subtype___redArg(v_inst_2005_, v___x_2008_);
return v___x_2009_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___redArg(lean_object* v_inst_2010_, lean_object* v_inst_2011_){
_start:
{
lean_object* v___f_2012_; lean_object* v___f_2013_; lean_object* v___f_2014_; lean_object* v___f_2015_; lean_object* v___x_2016_; 
lean_inc_ref_n(v_inst_2010_, 3);
lean_inc_ref_n(v_inst_2011_, 3);
v___f_2012_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__0), 4, 2);
lean_closure_set(v___f_2012_, 0, v_inst_2011_);
lean_closure_set(v___f_2012_, 1, v_inst_2010_);
v___f_2013_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__1), 4, 2);
lean_closure_set(v___f_2013_, 0, v_inst_2011_);
lean_closure_set(v___f_2013_, 1, v_inst_2010_);
v___f_2014_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__2), 4, 2);
lean_closure_set(v___f_2014_, 0, v_inst_2011_);
lean_closure_set(v___f_2014_, 1, v_inst_2010_);
v___f_2015_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLocallyFiniteOrder___redArg___lam__3), 4, 2);
lean_closure_set(v___f_2015_, 0, v_inst_2011_);
lean_closure_set(v___f_2015_, 1, v_inst_2010_);
v___x_2016_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2016_, 0, v___f_2012_);
lean_ctor_set(v___x_2016_, 1, v___f_2013_);
lean_ctor_set(v___x_2016_, 2, v___f_2014_);
lean_ctor_set(v___x_2016_, 3, v___f_2015_);
return v___x_2016_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder(lean_object* v_00_u03b1_2017_, lean_object* v_inst_2018_, lean_object* v_p_2019_, lean_object* v_inst_2020_, lean_object* v_inst_2021_){
_start:
{
lean_object* v___x_2022_; 
v___x_2022_ = lp_mathlib_Subtype_instLocallyFiniteOrder___redArg(v_inst_2020_, v_inst_2021_);
return v___x_2022_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrder___boxed(lean_object* v_00_u03b1_2023_, lean_object* v_inst_2024_, lean_object* v_p_2025_, lean_object* v_inst_2026_, lean_object* v_inst_2027_){
_start:
{
lean_object* v_res_2028_; 
v_res_2028_ = lp_mathlib_Subtype_instLocallyFiniteOrder(v_00_u03b1_2023_, v_inst_2024_, v_p_2025_, v_inst_2026_, v_inst_2027_);
lean_dec_ref(v_inst_2024_);
return v_res_2028_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg___lam__0(lean_object* v_inst_2029_, lean_object* v_inst_2030_, lean_object* v_a_2031_){
_start:
{
lean_object* v___x_2032_; lean_object* v___x_2033_; 
v___x_2032_ = lp_mathlib_Finset_Ioi___redArg(v_inst_2029_, v_a_2031_);
v___x_2033_ = lp_mathlib_Finset_subtype___redArg(v_inst_2030_, v___x_2032_);
return v___x_2033_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg___lam__1(lean_object* v_inst_2034_, lean_object* v_inst_2035_, lean_object* v_a_2036_){
_start:
{
lean_object* v___x_2037_; lean_object* v___x_2038_; 
v___x_2037_ = lp_mathlib_Finset_Ici___redArg(v_inst_2034_, v_a_2036_);
v___x_2038_ = lp_mathlib_Finset_subtype___redArg(v_inst_2035_, v___x_2037_);
return v___x_2038_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg(lean_object* v_inst_2039_, lean_object* v_inst_2040_){
_start:
{
lean_object* v___f_2041_; lean_object* v___f_2042_; lean_object* v___x_2043_; 
lean_inc_ref(v_inst_2039_);
lean_inc_ref(v_inst_2040_);
v___f_2041_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2041_, 0, v_inst_2040_);
lean_closure_set(v___f_2041_, 1, v_inst_2039_);
v___f_2042_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg___lam__1), 3, 2);
lean_closure_set(v___f_2042_, 0, v_inst_2040_);
lean_closure_set(v___f_2042_, 1, v_inst_2039_);
v___x_2043_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2043_, 0, v___f_2041_);
lean_ctor_set(v___x_2043_, 1, v___f_2042_);
return v___x_2043_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop(lean_object* v_00_u03b1_2044_, lean_object* v_inst_2045_, lean_object* v_p_2046_, lean_object* v_inst_2047_, lean_object* v_inst_2048_){
_start:
{
lean_object* v___x_2049_; 
v___x_2049_ = lp_mathlib_Subtype_instLocallyFiniteOrderTop___redArg(v_inst_2047_, v_inst_2048_);
return v___x_2049_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderTop___boxed(lean_object* v_00_u03b1_2050_, lean_object* v_inst_2051_, lean_object* v_p_2052_, lean_object* v_inst_2053_, lean_object* v_inst_2054_){
_start:
{
lean_object* v_res_2055_; 
v_res_2055_ = lp_mathlib_Subtype_instLocallyFiniteOrderTop(v_00_u03b1_2050_, v_inst_2051_, v_p_2052_, v_inst_2053_, v_inst_2054_);
lean_dec_ref(v_inst_2051_);
return v_res_2055_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg___lam__0(lean_object* v_inst_2056_, lean_object* v_inst_2057_, lean_object* v_a_2058_){
_start:
{
lean_object* v___x_2059_; lean_object* v___x_2060_; 
v___x_2059_ = lp_mathlib_Finset_Iio___redArg(v_inst_2056_, v_a_2058_);
v___x_2060_ = lp_mathlib_Finset_subtype___redArg(v_inst_2057_, v___x_2059_);
return v___x_2060_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg___lam__1(lean_object* v_inst_2061_, lean_object* v_inst_2062_, lean_object* v_a_2063_){
_start:
{
lean_object* v___x_2064_; lean_object* v___x_2065_; 
v___x_2064_ = lp_mathlib_Finset_Iic___redArg(v_inst_2061_, v_a_2063_);
v___x_2065_ = lp_mathlib_Finset_subtype___redArg(v_inst_2062_, v___x_2064_);
return v___x_2065_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg(lean_object* v_inst_2066_, lean_object* v_inst_2067_){
_start:
{
lean_object* v___f_2068_; lean_object* v___f_2069_; lean_object* v___x_2070_; 
lean_inc_ref(v_inst_2066_);
lean_inc_ref(v_inst_2067_);
v___f_2068_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg___lam__0), 3, 2);
lean_closure_set(v___f_2068_, 0, v_inst_2067_);
lean_closure_set(v___f_2068_, 1, v_inst_2066_);
v___f_2069_ = lean_alloc_closure((void*)(lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg___lam__1), 3, 2);
lean_closure_set(v___f_2069_, 0, v_inst_2067_);
lean_closure_set(v___f_2069_, 1, v_inst_2066_);
v___x_2070_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2070_, 0, v___f_2068_);
lean_ctor_set(v___x_2070_, 1, v___f_2069_);
return v___x_2070_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot(lean_object* v_00_u03b1_2071_, lean_object* v_inst_2072_, lean_object* v_p_2073_, lean_object* v_inst_2074_, lean_object* v_inst_2075_){
_start:
{
lean_object* v___x_2076_; 
v___x_2076_ = lp_mathlib_Subtype_instLocallyFiniteOrderBot___redArg(v_inst_2074_, v_inst_2075_);
return v___x_2076_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subtype_instLocallyFiniteOrderBot___boxed(lean_object* v_00_u03b1_2077_, lean_object* v_inst_2078_, lean_object* v_p_2079_, lean_object* v_inst_2080_, lean_object* v_inst_2081_){
_start:
{
lean_object* v_res_2082_; 
v_res_2082_ = lp_mathlib_Subtype_instLocallyFiniteOrderBot(v_00_u03b1_2077_, v_inst_2078_, v_p_2079_, v_inst_2080_, v_inst_2081_);
lean_dec_ref(v_inst_2078_);
return v_res_2082_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0(lean_object* v_inst_2083_, lean_object* v_y_2084_, lean_object* v_a_2085_){
_start:
{
lean_object* v___x_2086_; uint8_t v___x_2087_; 
v___x_2086_ = lean_apply_2(v_inst_2083_, v_a_2085_, v_y_2084_);
v___x_2087_ = lean_unbox(v___x_2086_);
return v___x_2087_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0___boxed(lean_object* v_inst_2088_, lean_object* v_y_2089_, lean_object* v_a_2090_){
_start:
{
uint8_t v_res_2091_; lean_object* v_r_2092_; 
v_res_2091_ = lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0(v_inst_2088_, v_y_2089_, v_a_2090_);
v_r_2092_ = lean_box(v_res_2091_);
return v_r_2092_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__2(lean_object* v___f_2093_, lean_object* v_inst_2094_, lean_object* v_y_2095_, lean_object* v_a_2096_){
_start:
{
lean_object* v___x_2097_; lean_object* v___x_2098_; 
v___x_2097_ = lp_mathlib_Subtype_instLocallyFiniteOrder___redArg(v___f_2093_, v_inst_2094_);
v___x_2098_ = lp_mathlib_Finset_Icc___redArg(v___x_2097_, v_a_2096_, v_y_2095_);
return v___x_2098_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__1(lean_object* v___f_2099_, lean_object* v_inst_2100_, lean_object* v_y_2101_, lean_object* v_a_2102_){
_start:
{
lean_object* v___x_2103_; lean_object* v___x_2104_; 
v___x_2103_ = lp_mathlib_Subtype_instLocallyFiniteOrder___redArg(v___f_2099_, v_inst_2100_);
v___x_2104_ = lp_mathlib_Finset_Ioc___redArg(v___x_2103_, v_a_2102_, v_y_2101_);
return v___x_2104_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg(lean_object* v_y_2105_, lean_object* v_inst_2106_, lean_object* v_inst_2107_){
_start:
{
lean_object* v___f_2108_; lean_object* v___f_2109_; lean_object* v___f_2110_; lean_object* v___x_2111_; 
lean_inc_n(v_y_2105_, 2);
v___f_2108_ = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_2108_, 0, v_inst_2106_);
lean_closure_set(v___f_2108_, 1, v_y_2105_);
lean_inc_ref(v_inst_2107_);
lean_inc_ref(v___f_2108_);
v___f_2109_ = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__2), 4, 3);
lean_closure_set(v___f_2109_, 0, v___f_2108_);
lean_closure_set(v___f_2109_, 1, v_inst_2107_);
lean_closure_set(v___f_2109_, 2, v_y_2105_);
v___f_2110_ = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__1), 4, 3);
lean_closure_set(v___f_2110_, 0, v___f_2108_);
lean_closure_set(v___f_2110_, 1, v_inst_2107_);
lean_closure_set(v___f_2110_, 2, v_y_2105_);
v___x_2111_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2111_, 0, v___f_2110_);
lean_ctor_set(v___x_2111_, 1, v___f_2109_);
return v___x_2111_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder(lean_object* v_00_u03b1_2112_, lean_object* v_inst_2113_, lean_object* v_y_2114_, lean_object* v_inst_2115_, lean_object* v_inst_2116_){
_start:
{
lean_object* v___x_2117_; 
v___x_2117_ = lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg(v_y_2114_, v_inst_2115_, v_inst_2116_);
return v___x_2117_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___boxed(lean_object* v_00_u03b1_2118_, lean_object* v_inst_2119_, lean_object* v_y_2120_, lean_object* v_inst_2121_, lean_object* v_inst_2122_){
_start:
{
lean_object* v_res_2123_; 
v_res_2123_ = lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder(v_00_u03b1_2118_, v_inst_2119_, v_y_2120_, v_inst_2121_, v_inst_2122_);
lean_dec_ref(v_inst_2119_);
return v_res_2123_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0(lean_object* v_inst_2124_, lean_object* v_y_2125_, lean_object* v_a_2126_){
_start:
{
lean_object* v___x_2127_; uint8_t v___x_2128_; 
v___x_2127_ = lean_apply_2(v_inst_2124_, v_y_2125_, v_a_2126_);
v___x_2128_ = lean_unbox(v___x_2127_);
return v___x_2128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0___boxed(lean_object* v_inst_2129_, lean_object* v_y_2130_, lean_object* v_a_2131_){
_start:
{
uint8_t v_res_2132_; lean_object* v_r_2133_; 
v_res_2132_ = lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0(v_inst_2129_, v_y_2130_, v_a_2131_);
v_r_2133_ = lean_box(v_res_2132_);
return v_r_2133_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__2(lean_object* v___f_2134_, lean_object* v_inst_2135_, lean_object* v_y_2136_, lean_object* v_a_2137_){
_start:
{
lean_object* v___x_2138_; lean_object* v___x_2139_; 
v___x_2138_ = lp_mathlib_Subtype_instLocallyFiniteOrder___redArg(v___f_2134_, v_inst_2135_);
v___x_2139_ = lp_mathlib_Finset_Icc___redArg(v___x_2138_, v_y_2136_, v_a_2137_);
return v___x_2139_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__1(lean_object* v___f_2140_, lean_object* v_inst_2141_, lean_object* v_y_2142_, lean_object* v_a_2143_){
_start:
{
lean_object* v___x_2144_; lean_object* v___x_2145_; 
v___x_2144_ = lp_mathlib_Subtype_instLocallyFiniteOrder___redArg(v___f_2140_, v_inst_2141_);
v___x_2145_ = lp_mathlib_Finset_Ico___redArg(v___x_2144_, v_y_2142_, v_a_2143_);
return v___x_2145_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg(lean_object* v_y_2146_, lean_object* v_inst_2147_, lean_object* v_inst_2148_){
_start:
{
lean_object* v___f_2149_; lean_object* v___f_2150_; lean_object* v___f_2151_; lean_object* v___x_2152_; 
lean_inc_n(v_y_2146_, 2);
v___f_2149_ = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_2149_, 0, v_inst_2147_);
lean_closure_set(v___f_2149_, 1, v_y_2146_);
lean_inc_ref(v_inst_2148_);
lean_inc_ref(v___f_2149_);
v___f_2150_ = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__2), 4, 3);
lean_closure_set(v___f_2150_, 0, v___f_2149_);
lean_closure_set(v___f_2150_, 1, v_inst_2148_);
lean_closure_set(v___f_2150_, 2, v_y_2146_);
v___f_2151_ = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__1), 4, 3);
lean_closure_set(v___f_2151_, 0, v___f_2149_);
lean_closure_set(v___f_2151_, 1, v_inst_2148_);
lean_closure_set(v___f_2151_, 2, v_y_2146_);
v___x_2152_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2152_, 0, v___f_2151_);
lean_ctor_set(v___x_2152_, 1, v___f_2150_);
return v___x_2152_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder(lean_object* v_00_u03b1_2153_, lean_object* v_inst_2154_, lean_object* v_y_2155_, lean_object* v_inst_2156_, lean_object* v_inst_2157_){
_start:
{
lean_object* v___x_2158_; 
v___x_2158_ = lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg(v_y_2155_, v_inst_2156_, v_inst_2157_);
return v___x_2158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___boxed(lean_object* v_00_u03b1_2159_, lean_object* v_inst_2160_, lean_object* v_y_2161_, lean_object* v_inst_2162_, lean_object* v_inst_2163_){
_start:
{
lean_object* v_res_2164_; 
v_res_2164_ = lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder(v_00_u03b1_2159_, v_inst_2160_, v_y_2161_, v_inst_2162_, v_inst_2163_);
lean_dec_ref(v_inst_2160_);
return v_res_2164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__1(lean_object* v_inst_2165_, lean_object* v_y_2166_, lean_object* v___f_2167_, lean_object* v_a_2168_){
_start:
{
lean_object* v___x_2169_; lean_object* v___x_2170_; 
v___x_2169_ = lp_mathlib_Finset_Ioo___redArg(v_inst_2165_, v_a_2168_, v_y_2166_);
v___x_2170_ = lp_mathlib_Finset_subtype___redArg(v___f_2167_, v___x_2169_);
return v___x_2170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__2(lean_object* v_inst_2171_, lean_object* v_y_2172_, lean_object* v___f_2173_, lean_object* v_a_2174_){
_start:
{
lean_object* v___x_2175_; lean_object* v___x_2176_; 
v___x_2175_ = lp_mathlib_Finset_Ico___redArg(v_inst_2171_, v_a_2174_, v_y_2172_);
v___x_2176_ = lp_mathlib_Finset_subtype___redArg(v___f_2173_, v___x_2175_);
return v___x_2176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg(lean_object* v_y_2177_, lean_object* v_inst_2178_, lean_object* v_inst_2179_){
_start:
{
lean_object* v___f_2180_; lean_object* v___f_2181_; lean_object* v___f_2182_; lean_object* v___x_2183_; 
lean_inc_n(v_y_2177_, 2);
v___f_2180_ = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderTopSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_2180_, 0, v_inst_2178_);
lean_closure_set(v___f_2180_, 1, v_y_2177_);
lean_inc_ref(v___f_2180_);
lean_inc_ref(v_inst_2179_);
v___f_2181_ = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__1), 4, 3);
lean_closure_set(v___f_2181_, 0, v_inst_2179_);
lean_closure_set(v___f_2181_, 1, v_y_2177_);
lean_closure_set(v___f_2181_, 2, v___f_2180_);
v___f_2182_ = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__2), 4, 3);
lean_closure_set(v___f_2182_, 0, v_inst_2179_);
lean_closure_set(v___f_2182_, 1, v_y_2177_);
lean_closure_set(v___f_2182_, 2, v___f_2180_);
v___x_2183_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2183_, 0, v___f_2181_);
lean_ctor_set(v___x_2183_, 1, v___f_2182_);
return v___x_2183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder(lean_object* v_00_u03b1_2184_, lean_object* v_inst_2185_, lean_object* v_y_2186_, lean_object* v_inst_2187_, lean_object* v_inst_2188_){
_start:
{
lean_object* v___x_2189_; 
v___x_2189_ = lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg(v_y_2186_, v_inst_2187_, v_inst_2188_);
return v___x_2189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder___boxed(lean_object* v_00_u03b1_2190_, lean_object* v_inst_2191_, lean_object* v_y_2192_, lean_object* v_inst_2193_, lean_object* v_inst_2194_){
_start:
{
lean_object* v_res_2195_; 
v_res_2195_ = lp_mathlib_instLocallyFiniteOrderTopSubtypeLtOfDecidableLTOfLocallyFiniteOrder(v_00_u03b1_2190_, v_inst_2191_, v_y_2192_, v_inst_2193_, v_inst_2194_);
lean_dec_ref(v_inst_2191_);
return v_res_2195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__1(lean_object* v_inst_2196_, lean_object* v_y_2197_, lean_object* v___f_2198_, lean_object* v_a_2199_){
_start:
{
lean_object* v___x_2200_; lean_object* v___x_2201_; 
v___x_2200_ = lp_mathlib_Finset_Ioo___redArg(v_inst_2196_, v_y_2197_, v_a_2199_);
v___x_2201_ = lp_mathlib_Finset_subtype___redArg(v___f_2198_, v___x_2200_);
return v___x_2201_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__2(lean_object* v_inst_2202_, lean_object* v_y_2203_, lean_object* v___f_2204_, lean_object* v_a_2205_){
_start:
{
lean_object* v___x_2206_; lean_object* v___x_2207_; 
v___x_2206_ = lp_mathlib_Finset_Ioc___redArg(v_inst_2202_, v_y_2203_, v_a_2205_);
v___x_2207_ = lp_mathlib_Finset_subtype___redArg(v___f_2204_, v___x_2206_);
return v___x_2207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg(lean_object* v_y_2208_, lean_object* v_inst_2209_, lean_object* v_inst_2210_){
_start:
{
lean_object* v___f_2211_; lean_object* v___f_2212_; lean_object* v___f_2213_; lean_object* v___x_2214_; 
lean_inc_n(v_y_2208_, 2);
v___f_2211_ = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderBotSubtypeLeOfDecidableLEOfLocallyFiniteOrder___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_2211_, 0, v_inst_2209_);
lean_closure_set(v___f_2211_, 1, v_y_2208_);
lean_inc_ref(v___f_2211_);
lean_inc_ref(v_inst_2210_);
v___f_2212_ = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__1), 4, 3);
lean_closure_set(v___f_2212_, 0, v_inst_2210_);
lean_closure_set(v___f_2212_, 1, v_y_2208_);
lean_closure_set(v___f_2212_, 2, v___f_2211_);
v___f_2213_ = lean_alloc_closure((void*)(lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg___lam__2), 4, 3);
lean_closure_set(v___f_2213_, 0, v_inst_2210_);
lean_closure_set(v___f_2213_, 1, v_y_2208_);
lean_closure_set(v___f_2213_, 2, v___f_2211_);
v___x_2214_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_2214_, 0, v___f_2212_);
lean_ctor_set(v___x_2214_, 1, v___f_2213_);
return v___x_2214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder(lean_object* v_00_u03b1_2215_, lean_object* v_inst_2216_, lean_object* v_y_2217_, lean_object* v_inst_2218_, lean_object* v_inst_2219_){
_start:
{
lean_object* v___x_2220_; 
v___x_2220_ = lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___redArg(v_y_2217_, v_inst_2218_, v_inst_2219_);
return v___x_2220_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder___boxed(lean_object* v_00_u03b1_2221_, lean_object* v_inst_2222_, lean_object* v_y_2223_, lean_object* v_inst_2224_, lean_object* v_inst_2225_){
_start:
{
lean_object* v_res_2226_; 
v_res_2226_ = lp_mathlib_instLocallyFiniteOrderBotSubtypeLtOfDecidableLTOfLocallyFiniteOrder(v_00_u03b1_2221_, v_inst_2222_, v_y_2223_, v_inst_2224_, v_inst_2225_);
lean_dec_ref(v_inst_2222_);
return v_res_2226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__0(lean_object* v_inst_2227_, lean_object* v_inst_2228_, lean_object* v_f_2229_, lean_object* v_x_2230_, lean_object* v_y_2231_){
_start:
{
lean_object* v_coe_2232_; lean_object* v_inv_2233_; lean_object* v_finsetIcc_2234_; lean_object* v___x_2235_; lean_object* v___x_2236_; lean_object* v___x_2237_; lean_object* v___x_2238_; lean_object* v___x_2239_; 
v_coe_2232_ = lean_ctor_get(v_inst_2227_, 0);
lean_inc_n(v_coe_2232_, 2);
v_inv_2233_ = lean_ctor_get(v_inst_2227_, 1);
lean_inc(v_inv_2233_);
lean_dec_ref(v_inst_2227_);
v_finsetIcc_2234_ = lean_ctor_get(v_inst_2228_, 0);
lean_inc(v_finsetIcc_2234_);
lean_dec_ref(v_inst_2228_);
lean_inc_n(v_f_2229_, 2);
v___x_2235_ = lean_apply_1(v_inv_2233_, v_f_2229_);
v___x_2236_ = lean_apply_2(v_coe_2232_, v_f_2229_, v_x_2230_);
v___x_2237_ = lean_apply_2(v_coe_2232_, v_f_2229_, v_y_2231_);
v___x_2238_ = lean_apply_2(v_finsetIcc_2234_, v___x_2236_, v___x_2237_);
v___x_2239_ = lp_mathlib_Finset_map___redArg(v___x_2235_, v___x_2238_);
return v___x_2239_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__1(lean_object* v_inst_2240_, lean_object* v_inst_2241_, lean_object* v_f_2242_, lean_object* v_x_2243_, lean_object* v_y_2244_){
_start:
{
lean_object* v_coe_2245_; lean_object* v_inv_2246_; lean_object* v_finsetIco_2247_; lean_object* v___x_2248_; lean_object* v___x_2249_; lean_object* v___x_2250_; lean_object* v___x_2251_; lean_object* v___x_2252_; 
v_coe_2245_ = lean_ctor_get(v_inst_2240_, 0);
lean_inc_n(v_coe_2245_, 2);
v_inv_2246_ = lean_ctor_get(v_inst_2240_, 1);
lean_inc(v_inv_2246_);
lean_dec_ref(v_inst_2240_);
v_finsetIco_2247_ = lean_ctor_get(v_inst_2241_, 1);
lean_inc(v_finsetIco_2247_);
lean_dec_ref(v_inst_2241_);
lean_inc_n(v_f_2242_, 2);
v___x_2248_ = lean_apply_1(v_inv_2246_, v_f_2242_);
v___x_2249_ = lean_apply_2(v_coe_2245_, v_f_2242_, v_x_2243_);
v___x_2250_ = lean_apply_2(v_coe_2245_, v_f_2242_, v_y_2244_);
v___x_2251_ = lean_apply_2(v_finsetIco_2247_, v___x_2249_, v___x_2250_);
v___x_2252_ = lp_mathlib_Finset_map___redArg(v___x_2248_, v___x_2251_);
return v___x_2252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__2(lean_object* v_inst_2253_, lean_object* v_inst_2254_, lean_object* v_f_2255_, lean_object* v_x_2256_, lean_object* v_y_2257_){
_start:
{
lean_object* v_coe_2258_; lean_object* v_inv_2259_; lean_object* v_finsetIoc_2260_; lean_object* v___x_2261_; lean_object* v___x_2262_; lean_object* v___x_2263_; lean_object* v___x_2264_; lean_object* v___x_2265_; 
v_coe_2258_ = lean_ctor_get(v_inst_2253_, 0);
lean_inc_n(v_coe_2258_, 2);
v_inv_2259_ = lean_ctor_get(v_inst_2253_, 1);
lean_inc(v_inv_2259_);
lean_dec_ref(v_inst_2253_);
v_finsetIoc_2260_ = lean_ctor_get(v_inst_2254_, 2);
lean_inc(v_finsetIoc_2260_);
lean_dec_ref(v_inst_2254_);
lean_inc_n(v_f_2255_, 2);
v___x_2261_ = lean_apply_1(v_inv_2259_, v_f_2255_);
v___x_2262_ = lean_apply_2(v_coe_2258_, v_f_2255_, v_x_2256_);
v___x_2263_ = lean_apply_2(v_coe_2258_, v_f_2255_, v_y_2257_);
v___x_2264_ = lean_apply_2(v_finsetIoc_2260_, v___x_2262_, v___x_2263_);
v___x_2265_ = lp_mathlib_Finset_map___redArg(v___x_2261_, v___x_2264_);
return v___x_2265_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__3(lean_object* v_inst_2266_, lean_object* v_inst_2267_, lean_object* v_f_2268_, lean_object* v_x_2269_, lean_object* v_y_2270_){
_start:
{
lean_object* v_coe_2271_; lean_object* v_inv_2272_; lean_object* v_finsetIoo_2273_; lean_object* v___x_2274_; lean_object* v___x_2275_; lean_object* v___x_2276_; lean_object* v___x_2277_; lean_object* v___x_2278_; 
v_coe_2271_ = lean_ctor_get(v_inst_2266_, 0);
lean_inc_n(v_coe_2271_, 2);
v_inv_2272_ = lean_ctor_get(v_inst_2266_, 1);
lean_inc(v_inv_2272_);
lean_dec_ref(v_inst_2266_);
v_finsetIoo_2273_ = lean_ctor_get(v_inst_2267_, 3);
lean_inc(v_finsetIoo_2273_);
lean_dec_ref(v_inst_2267_);
lean_inc_n(v_f_2268_, 2);
v___x_2274_ = lean_apply_1(v_inv_2272_, v_f_2268_);
v___x_2275_ = lean_apply_2(v_coe_2271_, v_f_2268_, v_x_2269_);
v___x_2276_ = lean_apply_2(v_coe_2271_, v_f_2268_, v_y_2270_);
v___x_2277_ = lean_apply_2(v_finsetIoo_2273_, v___x_2275_, v___x_2276_);
v___x_2278_ = lp_mathlib_Finset_map___redArg(v___x_2274_, v___x_2277_);
return v___x_2278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg(lean_object* v_inst_2279_, lean_object* v_f_2280_, lean_object* v_inst_2281_){
_start:
{
lean_object* v___f_2282_; lean_object* v___f_2283_; lean_object* v___f_2284_; lean_object* v___f_2285_; lean_object* v___x_2286_; 
lean_inc_n(v_f_2280_, 3);
lean_inc_ref_n(v_inst_2281_, 3);
lean_inc_ref_n(v_inst_2279_, 3);
v___f_2282_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__0), 5, 3);
lean_closure_set(v___f_2282_, 0, v_inst_2279_);
lean_closure_set(v___f_2282_, 1, v_inst_2281_);
lean_closure_set(v___f_2282_, 2, v_f_2280_);
v___f_2283_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__1), 5, 3);
lean_closure_set(v___f_2283_, 0, v_inst_2279_);
lean_closure_set(v___f_2283_, 1, v_inst_2281_);
lean_closure_set(v___f_2283_, 2, v_f_2280_);
v___f_2284_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__2), 5, 3);
lean_closure_set(v___f_2284_, 0, v_inst_2279_);
lean_closure_set(v___f_2284_, 1, v_inst_2281_);
lean_closure_set(v___f_2284_, 2, v_f_2280_);
v___f_2285_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__3), 5, 3);
lean_closure_set(v___f_2285_, 0, v_inst_2279_);
lean_closure_set(v___f_2285_, 1, v_inst_2281_);
lean_closure_set(v___f_2285_, 2, v_f_2280_);
v___x_2286_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2286_, 0, v___f_2282_);
lean_ctor_set(v___x_2286_, 1, v___f_2283_);
lean_ctor_set(v___x_2286_, 2, v___f_2284_);
lean_ctor_set(v___x_2286_, 3, v___f_2285_);
return v___x_2286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass(lean_object* v_F_2287_, lean_object* v_M_2288_, lean_object* v_N_2289_, lean_object* v_inst_2290_, lean_object* v_inst_2291_, lean_object* v_inst_2292_, lean_object* v_inst_2293_, lean_object* v_f_2294_, lean_object* v_inst_2295_){
_start:
{
lean_object* v___f_2296_; lean_object* v___f_2297_; lean_object* v___f_2298_; lean_object* v___f_2299_; lean_object* v___x_2300_; 
lean_inc_n(v_f_2294_, 3);
lean_inc_ref_n(v_inst_2295_, 3);
lean_inc_ref_n(v_inst_2292_, 3);
v___f_2296_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__0), 5, 3);
lean_closure_set(v___f_2296_, 0, v_inst_2292_);
lean_closure_set(v___f_2296_, 1, v_inst_2295_);
lean_closure_set(v___f_2296_, 2, v_f_2294_);
v___f_2297_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__1), 5, 3);
lean_closure_set(v___f_2297_, 0, v_inst_2292_);
lean_closure_set(v___f_2297_, 1, v_inst_2295_);
lean_closure_set(v___f_2297_, 2, v_f_2294_);
v___f_2298_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__2), 5, 3);
lean_closure_set(v___f_2298_, 0, v_inst_2292_);
lean_closure_set(v___f_2298_, 1, v_inst_2295_);
lean_closure_set(v___f_2298_, 2, v_f_2294_);
v___f_2299_ = lean_alloc_closure((void*)(lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___redArg___lam__3), 5, 3);
lean_closure_set(v___f_2299_, 0, v_inst_2292_);
lean_closure_set(v___f_2299_, 1, v_inst_2295_);
lean_closure_set(v___f_2299_, 2, v_f_2294_);
v___x_2300_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_2300_, 0, v___f_2296_);
lean_ctor_set(v___x_2300_, 1, v___f_2297_);
lean_ctor_set(v___x_2300_, 2, v___f_2298_);
lean_ctor_set(v___x_2300_, 3, v___f_2299_);
return v___x_2300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass___boxed(lean_object* v_F_2301_, lean_object* v_M_2302_, lean_object* v_N_2303_, lean_object* v_inst_2304_, lean_object* v_inst_2305_, lean_object* v_inst_2306_, lean_object* v_inst_2307_, lean_object* v_f_2308_, lean_object* v_inst_2309_){
_start:
{
lean_object* v_res_2310_; 
v_res_2310_ = lp_mathlib_LocallyFiniteOrder_ofOrderIsoClass(v_F_2301_, v_M_2302_, v_N_2303_, v_inst_2304_, v_inst_2305_, v_inst_2306_, v_inst_2307_, v_f_2308_, v_inst_2309_);
lean_dec_ref(v_inst_2305_);
lean_dec_ref(v_inst_2304_);
return v_res_2310_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Preimage(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Finset_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Hom_WithTopBot(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Set_UnorderedInterval(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Preimage(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Finset_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Hom_WithTopBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Set_UnorderedInterval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_Interval_Finset_Defs(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Finset_Preimage(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_WithTopBot(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_UnorderedInterval(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Interval_Finset_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Preimage(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_WithTopBot(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_UnorderedInterval(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Interval_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_Interval_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_Interval_Finset_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
