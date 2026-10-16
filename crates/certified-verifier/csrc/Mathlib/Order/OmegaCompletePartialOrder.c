// Lean compiler output
// Module: Mathlib.Order.OmegaCompletePartialOrder
// Imports: public import Init public meta import Init public import Mathlib.Control.Monad.Basic public import Mathlib.Order.Iterate public import Mathlib.Order.Part public import Mathlib.Order.Preorder.Chain public import Mathlib.Order.ScottContinuity public import Mathlib.Dynamics.FixedPoints.Defs
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
lean_object* lp_mathlib_OrderHom_instPartialOrder(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OrderHom_apply___redArg(lean_object*);
lean_object* lp_mathlib_OrderHom_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_eval(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OrderHom_prod___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instPreorder(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OrderHom_fst___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_OrderHom_snd___lam__0___boxed(lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_partialOrder___redArg(lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_matchesNull(lean_object*, lean_object*);
lean_object* l_Lean_replaceRef(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OrderHom_const___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Nat_iterate___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OrderHom_Subtype_val___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Subtype_preorder(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instInhabited___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instInhabited___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instInhabited___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instMembership(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instMembership___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instLE(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instLE___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_zip___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_zip(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_zip___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_OmegaCompletePartialOrder_0__OmegaCompletePartialOrder_Chain_pair_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_OmegaCompletePartialOrder_0__OmegaCompletePartialOrder_Chain_pair_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_pair___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_pair___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_pair___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_pair(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_pair___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_lift___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_lift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_lift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_subtype___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OmegaCompletePartialOrder_subtype___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderHom_Subtype_val___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_subtype___redArg___closed__0 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_subtype___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_subtype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_subtype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOmegaCompletePartialOrderForall___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOmegaCompletePartialOrderForall___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOmegaCompletePartialOrderForall___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instOmegaCompletePartialOrderForall(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Prod_00_u03c9SupImpl___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderHom_fst___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Prod_00_u03c9SupImpl___redArg___closed__0 = (const lean_object*)&lp_mathlib_Prod_00_u03c9SupImpl___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_Prod_00_u03c9SupImpl___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OrderHom_snd___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Prod_00_u03c9SupImpl___redArg___closed__1 = (const lean_object*)&lp_mathlib_Prod_00_u03c9SupImpl___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_Prod_00_u03c9SupImpl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_00_u03c9SupImpl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instOmegaCompletePartialOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instOmegaCompletePartialOrder(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_OrderHom_00_u03c9Sup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_OrderHom_00_u03c9Sup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_OrderHom_00_u03c9Sup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_OrderHom_00_u03c9Sup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_OrderHom_omegaCompletePartialOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_OrderHom_omegaCompletePartialOrder(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 26, .m_capacity = 26, .m_length = 25, .m_data = "OmegaCompletePartialOrder"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__0 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__0_value;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 8, .m_data = "term_→𝒄_"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__1 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__1_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(88, 188, 144, 148, 94, 210, 186, 124)}};
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(214, 212, 240, 235, 47, 165, 95, 205)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__2 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__2_value;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__3 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__3_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__4 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__4_value;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 4, .m_data = " →𝒄 "};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__5 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__5_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__5_value)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__6 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__6_value;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__7 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__7_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__8 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__8_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__8_value),((lean_object*)(((size_t)(25) << 1) | 1))}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__9 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__9_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__4_value),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__6_value),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__9_value)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__10 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__10_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__2_value),((lean_object*)(((size_t)(25) << 1) | 1)),((lean_object*)(((size_t)(26) << 1) | 1)),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__10_value)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__11 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484__ = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__11_value;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__0 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__0_value;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__1 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__1_value;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__2 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__2_value;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__3 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__3_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__4 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__4_value;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "ContinuousHom"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__5 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__5_value;
static lean_once_cell_t lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__6;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(80, 81, 127, 53, 128, 148, 234, 16)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__7 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__7_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(88, 188, 144, 148, 94, 210, 186, 124)}};
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__8_value_aux_0),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__5_value),LEAN_SCALAR_PTR_LITERAL(111, 194, 38, 148, 210, 185, 199, 116)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__8 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__8_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__8_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__9 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__9_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__8_value)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__10 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__10_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__10_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__11 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__11_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__10_value),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__11_value)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__12 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__12_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__9_value),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__12_value)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__13 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__13_value;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__14 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__14_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__15 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__15_value;
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______unexpand__OmegaCompletePartialOrder__ContinuousHom__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "ident"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______unexpand__OmegaCompletePartialOrder__ContinuousHom__1___closed__0 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______unexpand__OmegaCompletePartialOrder__ContinuousHom__1___closed__0_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______unexpand__OmegaCompletePartialOrder__ContinuousHom__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______unexpand__OmegaCompletePartialOrder__ContinuousHom__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(52, 159, 208, 51, 14, 60, 6, 71)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______unexpand__OmegaCompletePartialOrder__ContinuousHom__1___closed__1 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______unexpand__OmegaCompletePartialOrder__ContinuousHom__1___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______unexpand__OmegaCompletePartialOrder__ContinuousHom__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______unexpand__OmegaCompletePartialOrder__ContinuousHom__1___boxed(lean_object*, lean_object*, lean_object*);
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_instPartialOrderContinuousHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_instPartialOrderContinuousHom___closed__0 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_instPartialOrderContinuousHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_instPartialOrderContinuousHom(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_instPartialOrderContinuousHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Simps_apply___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Simps_apply(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Simps_apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__0 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__0_value;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__1 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__1_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__2_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__2_value_aux_0),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__2_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__2_value_aux_1),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__2_value_aux_2),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__2 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__2_value;
static const lean_array_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__3 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__3_value;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__4 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__4_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__5_value_aux_0),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__5_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__5_value_aux_1),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__5_value_aux_2),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__4_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__5 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__5_value;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__6 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__6_value;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Meta"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__7 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__7_value;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "FunProp"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__8 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__8_value;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "funPropTacStx"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__9 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__9_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__10_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__10_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__10_value_aux_0),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__7_value),LEAN_SCALAR_PTR_LITERAL(210, 10, 180, 159, 248, 97, 218, 144)}};
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__10_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__10_value_aux_1),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(215, 108, 118, 4, 116, 28, 199, 219)}};
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__10_value_aux_2),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__9_value),LEAN_SCALAR_PTR_LITERAL(146, 162, 46, 139, 110, 137, 217, 136)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__10 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__10_value;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "fun_prop"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__11 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__11_value;
static lean_once_cell_t lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__12;
static lean_once_cell_t lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__13;
static const lean_string_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "optConfig"};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__14 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__14_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__15_value_aux_0),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__15_value_aux_1),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__15_value_aux_2),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(137, 208, 10, 74, 108, 50, 106, 48)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__15 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__15_value;
static const lean_ctor_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__15_value),((lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__3_value)}};
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__16 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__16_value;
static lean_once_cell_t lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__17_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__17;
static lean_once_cell_t lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__18;
static lean_once_cell_t lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__19;
static lean_once_cell_t lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__20_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__20;
static lean_once_cell_t lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__21;
static lean_once_cell_t lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__22;
static lean_once_cell_t lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__23_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__23;
static lean_once_cell_t lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__24;
static lean_once_cell_t lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__25;
static lean_once_cell_t lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__26;
static lean_once_cell_t lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__27;
static lean_once_cell_t lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__28_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__28;
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_copy___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_copy___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_copy(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_copy___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_id___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_id___closed__0 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_id___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_id(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_const___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_const(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_toMono___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_toMono___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_toMono___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_toMono___closed__0 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_toMono___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_toMono(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_toMono___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_00_u03c9Sup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_00_u03c9Sup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_00_u03c9Sup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_inst___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_inst(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Prod_apply___lam__0(lean_object*);
static const lean_closure_object lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Prod_apply___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Prod_apply___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Prod_apply___closed__0 = (const lean_object*)&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Prod_apply___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Prod_apply(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Prod_apply___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_flip___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_flip___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_flip(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_flip___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_fixedPoints_iterateChain___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_fixedPoints_iterateChain___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_fixedPoints_iterateChain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_fixedPoints_iterateChain___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instInhabited___redArg___lam__0(lean_object* v_inst_1_, lean_object* v_x_2_){
_start:
{
lean_inc(v_inst_1_);
return v_inst_1_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instInhabited___redArg___lam__0___boxed(lean_object* v_inst_3_, lean_object* v_x_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_mathlib_OmegaCompletePartialOrder_Chain_instInhabited___redArg___lam__0(v_inst_3_, v_x_4_);
lean_dec(v_x_4_);
lean_dec(v_inst_3_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instInhabited___redArg(lean_object* v_inst_6_){
_start:
{
lean_object* v___f_7_; 
v___f_7_ = lean_alloc_closure((void*)(lp_mathlib_OmegaCompletePartialOrder_Chain_instInhabited___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_7_, 0, v_inst_6_);
return v___f_7_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instInhabited(lean_object* v_00_u03b1_8_, lean_object* v_inst_9_, lean_object* v_inst_10_){
_start:
{
lean_object* v___f_11_; 
v___f_11_ = lean_alloc_closure((void*)(lp_mathlib_OmegaCompletePartialOrder_Chain_instInhabited___redArg___lam__0___boxed), 2, 1);
lean_closure_set(v___f_11_, 0, v_inst_10_);
return v___f_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instInhabited___boxed(lean_object* v_00_u03b1_12_, lean_object* v_inst_13_, lean_object* v_inst_14_){
_start:
{
lean_object* v_res_15_; 
v_res_15_ = lp_mathlib_OmegaCompletePartialOrder_Chain_instInhabited(v_00_u03b1_12_, v_inst_13_, v_inst_14_);
lean_dec_ref(v_inst_13_);
return v_res_15_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instMembership(lean_object* v_00_u03b1_16_, lean_object* v_inst_17_){
_start:
{
lean_object* v___x_18_; 
v___x_18_ = lean_box(0);
return v___x_18_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instMembership___boxed(lean_object* v_00_u03b1_19_, lean_object* v_inst_20_){
_start:
{
lean_object* v_res_21_; 
v_res_21_ = lp_mathlib_OmegaCompletePartialOrder_Chain_instMembership(v_00_u03b1_19_, v_inst_20_);
lean_dec_ref(v_inst_20_);
return v_res_21_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instLE(lean_object* v_00_u03b1_22_, lean_object* v_inst_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lean_box(0);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_instLE___boxed(lean_object* v_00_u03b1_25_, lean_object* v_inst_26_){
_start:
{
lean_object* v_res_27_; 
v_res_27_ = lp_mathlib_OmegaCompletePartialOrder_Chain_instLE(v_00_u03b1_25_, v_inst_26_);
lean_dec_ref(v_inst_26_);
return v_res_27_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_map___redArg(lean_object* v_c_28_, lean_object* v_f_29_){
_start:
{
lean_object* v___x_30_; 
v___x_30_ = lp_mathlib_OrderHom_comp___redArg(v_f_29_, v_c_28_);
return v___x_30_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_map(lean_object* v_00_u03b1_31_, lean_object* v_00_u03b2_32_, lean_object* v_inst_33_, lean_object* v_inst_34_, lean_object* v_c_35_, lean_object* v_f_36_){
_start:
{
lean_object* v___x_37_; 
v___x_37_ = lp_mathlib_OrderHom_comp___redArg(v_f_36_, v_c_35_);
return v___x_37_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_map___boxed(lean_object* v_00_u03b1_38_, lean_object* v_00_u03b2_39_, lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_c_42_, lean_object* v_f_43_){
_start:
{
lean_object* v_res_44_; 
v_res_44_ = lp_mathlib_OmegaCompletePartialOrder_Chain_map(v_00_u03b1_38_, v_00_u03b2_39_, v_inst_40_, v_inst_41_, v_c_42_, v_f_43_);
lean_dec_ref(v_inst_41_);
lean_dec_ref(v_inst_40_);
return v_res_44_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_zip___redArg(lean_object* v_c_u2080_45_, lean_object* v_c_u2081_46_){
_start:
{
lean_object* v___f_47_; 
v___f_47_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_47_, 0, v_c_u2080_45_);
lean_closure_set(v___f_47_, 1, v_c_u2081_46_);
return v___f_47_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_zip(lean_object* v_00_u03b1_48_, lean_object* v_00_u03b2_49_, lean_object* v_inst_50_, lean_object* v_inst_51_, lean_object* v_c_u2080_52_, lean_object* v_c_u2081_53_){
_start:
{
lean_object* v___f_54_; 
v___f_54_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_prod___redArg___lam__0), 3, 2);
lean_closure_set(v___f_54_, 0, v_c_u2080_52_);
lean_closure_set(v___f_54_, 1, v_c_u2081_53_);
return v___f_54_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_zip___boxed(lean_object* v_00_u03b1_55_, lean_object* v_00_u03b2_56_, lean_object* v_inst_57_, lean_object* v_inst_58_, lean_object* v_c_u2080_59_, lean_object* v_c_u2081_60_){
_start:
{
lean_object* v_res_61_; 
v_res_61_ = lp_mathlib_OmegaCompletePartialOrder_Chain_zip(v_00_u03b1_55_, v_00_u03b2_56_, v_inst_57_, v_inst_58_, v_c_u2080_59_, v_c_u2081_60_);
lean_dec_ref(v_inst_58_);
lean_dec_ref(v_inst_57_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_OmegaCompletePartialOrder_0__OmegaCompletePartialOrder_Chain_pair_match__1_splitter___redArg(lean_object* v_x_62_, lean_object* v_h__1_63_, lean_object* v_h__2_64_){
_start:
{
lean_object* v___x_65_; uint8_t v___x_66_; 
v___x_65_ = lean_unsigned_to_nat(0u);
v___x_66_ = lean_nat_dec_eq(v_x_62_, v___x_65_);
if (v___x_66_ == 0)
{
lean_object* v___x_67_; 
lean_dec(v_h__1_63_);
v___x_67_ = lean_apply_2(v_h__2_64_, v_x_62_, lean_box(0));
return v___x_67_;
}
else
{
lean_object* v___x_68_; lean_object* v___x_69_; 
lean_dec(v_h__2_64_);
lean_dec(v_x_62_);
v___x_68_ = lean_box(0);
v___x_69_ = lean_apply_1(v_h__1_63_, v___x_68_);
return v___x_69_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Order_OmegaCompletePartialOrder_0__OmegaCompletePartialOrder_Chain_pair_match__1_splitter(lean_object* v_motive_70_, lean_object* v_x_71_, lean_object* v_h__1_72_, lean_object* v_h__2_73_){
_start:
{
lean_object* v___x_74_; uint8_t v___x_75_; 
v___x_74_ = lean_unsigned_to_nat(0u);
v___x_75_ = lean_nat_dec_eq(v_x_71_, v___x_74_);
if (v___x_75_ == 0)
{
lean_object* v___x_76_; 
lean_dec(v_h__1_72_);
v___x_76_ = lean_apply_2(v_h__2_73_, v_x_71_, lean_box(0));
return v___x_76_;
}
else
{
lean_object* v___x_77_; lean_object* v___x_78_; 
lean_dec(v_h__2_73_);
lean_dec(v_x_71_);
v___x_77_ = lean_box(0);
v___x_78_ = lean_apply_1(v_h__1_72_, v___x_77_);
return v___x_78_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_pair___redArg___lam__0(lean_object* v_b_79_, lean_object* v_a_80_, lean_object* v_x_81_){
_start:
{
lean_object* v___x_82_; uint8_t v___x_83_; 
v___x_82_ = lean_unsigned_to_nat(0u);
v___x_83_ = lean_nat_dec_eq(v_x_81_, v___x_82_);
if (v___x_83_ == 0)
{
lean_inc(v_b_79_);
return v_b_79_;
}
else
{
lean_inc(v_a_80_);
return v_a_80_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_pair___redArg___lam__0___boxed(lean_object* v_b_84_, lean_object* v_a_85_, lean_object* v_x_86_){
_start:
{
lean_object* v_res_87_; 
v_res_87_ = lp_mathlib_OmegaCompletePartialOrder_Chain_pair___redArg___lam__0(v_b_84_, v_a_85_, v_x_86_);
lean_dec(v_x_86_);
lean_dec(v_a_85_);
lean_dec(v_b_84_);
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_pair___redArg(lean_object* v_a_88_, lean_object* v_b_89_){
_start:
{
lean_object* v___f_90_; 
v___f_90_ = lean_alloc_closure((void*)(lp_mathlib_OmegaCompletePartialOrder_Chain_pair___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_90_, 0, v_b_89_);
lean_closure_set(v___f_90_, 1, v_a_88_);
return v___f_90_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_pair(lean_object* v_00_u03b1_91_, lean_object* v_inst_92_, lean_object* v_a_93_, lean_object* v_b_94_, lean_object* v_hab_95_){
_start:
{
lean_object* v___f_96_; 
v___f_96_ = lean_alloc_closure((void*)(lp_mathlib_OmegaCompletePartialOrder_Chain_pair___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_96_, 0, v_b_94_);
lean_closure_set(v___f_96_, 1, v_a_93_);
return v___f_96_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_Chain_pair___boxed(lean_object* v_00_u03b1_97_, lean_object* v_inst_98_, lean_object* v_a_99_, lean_object* v_b_100_, lean_object* v_hab_101_){
_start:
{
lean_object* v_res_102_; 
v_res_102_ = lp_mathlib_OmegaCompletePartialOrder_Chain_pair(v_00_u03b1_97_, v_inst_98_, v_a_99_, v_b_100_, v_hab_101_);
lean_dec_ref(v_inst_98_);
return v_res_102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_lift___redArg(lean_object* v_inst_103_, lean_object* v_00_u03c9Sup_u2080_104_){
_start:
{
lean_object* v___x_105_; 
v___x_105_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_105_, 0, v_inst_103_);
lean_ctor_set(v___x_105_, 1, v_00_u03c9Sup_u2080_104_);
return v___x_105_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_lift(lean_object* v_00_u03b1_106_, lean_object* v_00_u03b2_107_, lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_f_110_, lean_object* v_00_u03c9Sup_u2080_111_, lean_object* v_h_112_, lean_object* v_h_x27_113_){
_start:
{
lean_object* v___x_114_; 
v___x_114_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_114_, 0, v_inst_109_);
lean_ctor_set(v___x_114_, 1, v_00_u03c9Sup_u2080_111_);
return v___x_114_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_lift___boxed(lean_object* v_00_u03b1_115_, lean_object* v_00_u03b2_116_, lean_object* v_inst_117_, lean_object* v_inst_118_, lean_object* v_f_119_, lean_object* v_00_u03c9Sup_u2080_120_, lean_object* v_h_121_, lean_object* v_h_x27_122_){
_start:
{
lean_object* v_res_123_; 
v_res_123_ = lp_mathlib_OmegaCompletePartialOrder_lift(v_00_u03b1_115_, v_00_u03b2_116_, v_inst_117_, v_inst_118_, v_f_119_, v_00_u03c9Sup_u2080_120_, v_h_121_, v_h_x27_122_);
lean_dec(v_f_119_);
lean_dec_ref(v_inst_117_);
return v_res_123_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_subtype___redArg___lam__0(lean_object* v___f_124_, lean_object* v_00_u03c9Sup_125_, lean_object* v_c_126_){
_start:
{
lean_object* v___x_127_; lean_object* v___x_128_; 
v___x_127_ = lp_mathlib_OrderHom_comp___redArg(v___f_124_, v_c_126_);
v___x_128_ = lean_apply_1(v_00_u03c9Sup_125_, v___x_127_);
return v___x_128_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_subtype___redArg(lean_object* v_inst_130_){
_start:
{
lean_object* v_toPartialOrder_131_; lean_object* v_00_u03c9Sup_132_; lean_object* v___x_134_; uint8_t v_isShared_135_; uint8_t v_isSharedCheck_142_; 
v_toPartialOrder_131_ = lean_ctor_get(v_inst_130_, 0);
v_00_u03c9Sup_132_ = lean_ctor_get(v_inst_130_, 1);
v_isSharedCheck_142_ = !lean_is_exclusive(v_inst_130_);
if (v_isSharedCheck_142_ == 0)
{
v___x_134_ = v_inst_130_;
v_isShared_135_ = v_isSharedCheck_142_;
goto v_resetjp_133_;
}
else
{
lean_inc(v_00_u03c9Sup_132_);
lean_inc(v_toPartialOrder_131_);
lean_dec(v_inst_130_);
v___x_134_ = lean_box(0);
v_isShared_135_ = v_isSharedCheck_142_;
goto v_resetjp_133_;
}
v_resetjp_133_:
{
lean_object* v___x_136_; lean_object* v___f_137_; lean_object* v___f_138_; lean_object* v___x_140_; 
v___x_136_ = lp_mathlib_Subtype_preorder(lean_box(0), v_toPartialOrder_131_, lean_box(0));
lean_dec_ref(v_toPartialOrder_131_);
v___f_137_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_subtype___redArg___closed__0));
v___f_138_ = lean_alloc_closure((void*)(lp_mathlib_OmegaCompletePartialOrder_subtype___redArg___lam__0), 3, 2);
lean_closure_set(v___f_138_, 0, v___f_137_);
lean_closure_set(v___f_138_, 1, v_00_u03c9Sup_132_);
if (v_isShared_135_ == 0)
{
lean_ctor_set(v___x_134_, 1, v___f_138_);
lean_ctor_set(v___x_134_, 0, v___x_136_);
v___x_140_ = v___x_134_;
goto v_reusejp_139_;
}
else
{
lean_object* v_reuseFailAlloc_141_; 
v_reuseFailAlloc_141_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_141_, 0, v___x_136_);
lean_ctor_set(v_reuseFailAlloc_141_, 1, v___f_138_);
v___x_140_ = v_reuseFailAlloc_141_;
goto v_reusejp_139_;
}
v_reusejp_139_:
{
return v___x_140_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_subtype(lean_object* v_00_u03b1_143_, lean_object* v_inst_144_, lean_object* v_p_145_, lean_object* v_hp_146_){
_start:
{
lean_object* v___x_147_; 
v___x_147_ = lp_mathlib_OmegaCompletePartialOrder_subtype___redArg(v_inst_144_);
return v___x_147_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOmegaCompletePartialOrderForall___redArg___lam__0(lean_object* v_inst_148_, lean_object* v_i_149_){
_start:
{
lean_object* v___x_150_; lean_object* v_toPartialOrder_151_; 
v___x_150_ = lean_apply_1(v_inst_148_, v_i_149_);
v_toPartialOrder_151_ = lean_ctor_get(v___x_150_, 0);
lean_inc_ref(v_toPartialOrder_151_);
lean_dec_ref(v___x_150_);
return v_toPartialOrder_151_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOmegaCompletePartialOrderForall___redArg___lam__1(lean_object* v_inst_152_, lean_object* v_c_153_, lean_object* v_a_154_){
_start:
{
lean_object* v___x_155_; lean_object* v_00_u03c9Sup_156_; lean_object* v___x_157_; lean_object* v___x_158_; lean_object* v___x_159_; 
lean_inc(v_a_154_);
v___x_155_ = lean_apply_1(v_inst_152_, v_a_154_);
v_00_u03c9Sup_156_ = lean_ctor_get(v___x_155_, 1);
lean_inc(v_00_u03c9Sup_156_);
lean_dec_ref(v___x_155_);
v___x_157_ = lean_alloc_closure((void*)(lp_mathlib_Function_eval), 4, 3);
lean_closure_set(v___x_157_, 0, lean_box(0));
lean_closure_set(v___x_157_, 1, lean_box(0));
lean_closure_set(v___x_157_, 2, v_a_154_);
v___x_158_ = lp_mathlib_OrderHom_comp___redArg(v___x_157_, v_c_153_);
v___x_159_ = lean_apply_1(v_00_u03c9Sup_156_, v___x_158_);
return v___x_159_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOmegaCompletePartialOrderForall___redArg(lean_object* v_inst_160_){
_start:
{
lean_object* v___f_161_; lean_object* v___f_162_; lean_object* v___x_163_; lean_object* v___x_164_; 
lean_inc_ref(v_inst_160_);
v___f_161_ = lean_alloc_closure((void*)(lp_mathlib_instOmegaCompletePartialOrderForall___redArg___lam__0), 2, 1);
lean_closure_set(v___f_161_, 0, v_inst_160_);
v___f_162_ = lean_alloc_closure((void*)(lp_mathlib_instOmegaCompletePartialOrderForall___redArg___lam__1), 3, 1);
lean_closure_set(v___f_162_, 0, v_inst_160_);
v___x_163_ = lp_mathlib_Pi_partialOrder___redArg(v___f_161_);
v___x_164_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_164_, 0, v___x_163_);
lean_ctor_set(v___x_164_, 1, v___f_162_);
return v___x_164_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instOmegaCompletePartialOrderForall(lean_object* v_00_u03b1_165_, lean_object* v_00_u03b2_166_, lean_object* v_inst_167_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lp_mathlib_instOmegaCompletePartialOrderForall___redArg(v_inst_167_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_00_u03c9SupImpl___redArg(lean_object* v_inst_171_, lean_object* v_inst_172_, lean_object* v_c_173_){
_start:
{
lean_object* v_00_u03c9Sup_174_; lean_object* v_00_u03c9Sup_175_; lean_object* v___x_177_; uint8_t v_isShared_178_; uint8_t v_isSharedCheck_188_; 
v_00_u03c9Sup_174_ = lean_ctor_get(v_inst_171_, 1);
lean_inc(v_00_u03c9Sup_174_);
lean_dec_ref(v_inst_171_);
v_00_u03c9Sup_175_ = lean_ctor_get(v_inst_172_, 1);
v_isSharedCheck_188_ = !lean_is_exclusive(v_inst_172_);
if (v_isSharedCheck_188_ == 0)
{
lean_object* v_unused_189_; 
v_unused_189_ = lean_ctor_get(v_inst_172_, 0);
lean_dec(v_unused_189_);
v___x_177_ = v_inst_172_;
v_isShared_178_ = v_isSharedCheck_188_;
goto v_resetjp_176_;
}
else
{
lean_inc(v_00_u03c9Sup_175_);
lean_dec(v_inst_172_);
v___x_177_ = lean_box(0);
v_isShared_178_ = v_isSharedCheck_188_;
goto v_resetjp_176_;
}
v_resetjp_176_:
{
lean_object* v___f_179_; lean_object* v___x_180_; lean_object* v___x_181_; lean_object* v___f_182_; lean_object* v___x_183_; lean_object* v___x_184_; lean_object* v___x_186_; 
v___f_179_ = ((lean_object*)(lp_mathlib_Prod_00_u03c9SupImpl___redArg___closed__0));
lean_inc_ref(v_c_173_);
v___x_180_ = lp_mathlib_OrderHom_comp___redArg(v___f_179_, v_c_173_);
v___x_181_ = lean_apply_1(v_00_u03c9Sup_174_, v___x_180_);
v___f_182_ = ((lean_object*)(lp_mathlib_Prod_00_u03c9SupImpl___redArg___closed__1));
v___x_183_ = lp_mathlib_OrderHom_comp___redArg(v___f_182_, v_c_173_);
v___x_184_ = lean_apply_1(v_00_u03c9Sup_175_, v___x_183_);
if (v_isShared_178_ == 0)
{
lean_ctor_set(v___x_177_, 1, v___x_184_);
lean_ctor_set(v___x_177_, 0, v___x_181_);
v___x_186_ = v___x_177_;
goto v_reusejp_185_;
}
else
{
lean_object* v_reuseFailAlloc_187_; 
v_reuseFailAlloc_187_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_187_, 0, v___x_181_);
lean_ctor_set(v_reuseFailAlloc_187_, 1, v___x_184_);
v___x_186_ = v_reuseFailAlloc_187_;
goto v_reusejp_185_;
}
v_reusejp_185_:
{
return v___x_186_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_00_u03c9SupImpl(lean_object* v_00_u03b1_190_, lean_object* v_00_u03b2_191_, lean_object* v_inst_192_, lean_object* v_inst_193_, lean_object* v_c_194_){
_start:
{
lean_object* v___x_195_; 
v___x_195_ = lp_mathlib_Prod_00_u03c9SupImpl___redArg(v_inst_192_, v_inst_193_, v_c_194_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instOmegaCompletePartialOrder___redArg(lean_object* v_inst_196_, lean_object* v_inst_197_){
_start:
{
lean_object* v_toPartialOrder_198_; lean_object* v_toPartialOrder_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; 
v_toPartialOrder_198_ = lean_ctor_get(v_inst_196_, 0);
v_toPartialOrder_199_ = lean_ctor_get(v_inst_197_, 0);
v___x_200_ = lp_mathlib_Prod_instPreorder(lean_box(0), lean_box(0), v_toPartialOrder_198_, v_toPartialOrder_199_);
v___x_201_ = lean_alloc_closure((void*)(lp_mathlib_Prod_00_u03c9SupImpl), 5, 4);
lean_closure_set(v___x_201_, 0, lean_box(0));
lean_closure_set(v___x_201_, 1, lean_box(0));
lean_closure_set(v___x_201_, 2, v_inst_196_);
lean_closure_set(v___x_201_, 3, v_inst_197_);
v___x_202_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_202_, 0, v___x_200_);
lean_ctor_set(v___x_202_, 1, v___x_201_);
return v___x_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instOmegaCompletePartialOrder(lean_object* v_00_u03b1_203_, lean_object* v_00_u03b2_204_, lean_object* v_inst_205_, lean_object* v_inst_206_){
_start:
{
lean_object* v___x_207_; 
v___x_207_ = lp_mathlib_Prod_instOmegaCompletePartialOrder___redArg(v_inst_205_, v_inst_206_);
return v___x_207_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_OrderHom_00_u03c9Sup___redArg___lam__0(lean_object* v_inst_208_, lean_object* v_c_209_, lean_object* v_a_210_){
_start:
{
lean_object* v_00_u03c9Sup_211_; lean_object* v___x_212_; lean_object* v___x_213_; lean_object* v___x_214_; 
v_00_u03c9Sup_211_ = lean_ctor_get(v_inst_208_, 1);
lean_inc(v_00_u03c9Sup_211_);
lean_dec_ref(v_inst_208_);
v___x_212_ = lp_mathlib_OrderHom_apply___redArg(v_a_210_);
v___x_213_ = lp_mathlib_OrderHom_comp___redArg(v___x_212_, v_c_209_);
v___x_214_ = lean_apply_1(v_00_u03c9Sup_211_, v___x_213_);
return v___x_214_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_OrderHom_00_u03c9Sup___redArg(lean_object* v_inst_215_, lean_object* v_c_216_){
_start:
{
lean_object* v___f_217_; 
v___f_217_ = lean_alloc_closure((void*)(lp_mathlib_OmegaCompletePartialOrder_OrderHom_00_u03c9Sup___redArg___lam__0), 3, 2);
lean_closure_set(v___f_217_, 0, v_inst_215_);
lean_closure_set(v___f_217_, 1, v_c_216_);
return v___f_217_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_OrderHom_00_u03c9Sup(lean_object* v_00_u03b1_218_, lean_object* v_00_u03b2_219_, lean_object* v_inst_220_, lean_object* v_inst_221_, lean_object* v_c_222_){
_start:
{
lean_object* v___f_223_; 
v___f_223_ = lean_alloc_closure((void*)(lp_mathlib_OmegaCompletePartialOrder_OrderHom_00_u03c9Sup___redArg___lam__0), 3, 2);
lean_closure_set(v___f_223_, 0, v_inst_221_);
lean_closure_set(v___f_223_, 1, v_c_222_);
return v___f_223_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_OrderHom_00_u03c9Sup___boxed(lean_object* v_00_u03b1_224_, lean_object* v_00_u03b2_225_, lean_object* v_inst_226_, lean_object* v_inst_227_, lean_object* v_c_228_){
_start:
{
lean_object* v_res_229_; 
v_res_229_ = lp_mathlib_OmegaCompletePartialOrder_OrderHom_00_u03c9Sup(v_00_u03b1_224_, v_00_u03b2_225_, v_inst_226_, v_inst_227_, v_c_228_);
lean_dec_ref(v_inst_226_);
return v_res_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_OrderHom_omegaCompletePartialOrder___redArg(lean_object* v_inst_230_, lean_object* v_inst_231_){
_start:
{
lean_object* v_toPartialOrder_232_; lean_object* v_toPartialOrder_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; 
v_toPartialOrder_232_ = lean_ctor_get(v_inst_230_, 0);
v_toPartialOrder_233_ = lean_ctor_get(v_inst_231_, 0);
v___x_234_ = lp_mathlib_OrderHom_instPartialOrder(lean_box(0), v_toPartialOrder_232_, lean_box(0), v_toPartialOrder_233_);
v___x_235_ = lean_alloc_closure((void*)(lp_mathlib_OmegaCompletePartialOrder_OrderHom_00_u03c9Sup___boxed), 5, 4);
lean_closure_set(v___x_235_, 0, lean_box(0));
lean_closure_set(v___x_235_, 1, lean_box(0));
lean_closure_set(v___x_235_, 2, v_inst_230_);
lean_closure_set(v___x_235_, 3, v_inst_231_);
v___x_236_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_236_, 0, v___x_234_);
lean_ctor_set(v___x_236_, 1, v___x_235_);
return v___x_236_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_OrderHom_omegaCompletePartialOrder(lean_object* v_00_u03b1_237_, lean_object* v_00_u03b2_238_, lean_object* v_inst_239_, lean_object* v_inst_240_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lp_mathlib_OmegaCompletePartialOrder_OrderHom_omegaCompletePartialOrder___redArg(v_inst_239_, v_inst_240_);
return v___x_241_;
}
}
static lean_object* _init_lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__6(void){
_start:
{
lean_object* v___x_279_; lean_object* v___x_280_; 
v___x_279_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__5));
v___x_280_ = l_String_toRawSubstring_x27(v___x_279_);
return v___x_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1(lean_object* v_x_303_, lean_object* v_a_304_, lean_object* v_a_305_){
_start:
{
lean_object* v___x_306_; uint8_t v___x_307_; 
v___x_306_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__2));
lean_inc(v_x_303_);
v___x_307_ = l_Lean_Syntax_isOfKind(v_x_303_, v___x_306_);
if (v___x_307_ == 0)
{
lean_object* v___x_308_; lean_object* v___x_309_; 
lean_dec(v_x_303_);
v___x_308_ = lean_box(1);
v___x_309_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_309_, 0, v___x_308_);
lean_ctor_set(v___x_309_, 1, v_a_305_);
return v___x_309_;
}
else
{
lean_object* v_quotContext_310_; lean_object* v_currMacroScope_311_; lean_object* v_ref_312_; lean_object* v___x_313_; lean_object* v___x_314_; lean_object* v___x_315_; lean_object* v___x_316_; uint8_t v___x_317_; lean_object* v___x_318_; lean_object* v___x_319_; lean_object* v___x_320_; lean_object* v___x_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; lean_object* v___x_327_; lean_object* v___x_328_; 
v_quotContext_310_ = lean_ctor_get(v_a_304_, 1);
v_currMacroScope_311_ = lean_ctor_get(v_a_304_, 2);
v_ref_312_ = lean_ctor_get(v_a_304_, 5);
v___x_313_ = lean_unsigned_to_nat(0u);
v___x_314_ = l_Lean_Syntax_getArg(v_x_303_, v___x_313_);
v___x_315_ = lean_unsigned_to_nat(2u);
v___x_316_ = l_Lean_Syntax_getArg(v_x_303_, v___x_315_);
lean_dec(v_x_303_);
v___x_317_ = 0;
v___x_318_ = l_Lean_SourceInfo_fromRef(v_ref_312_, v___x_317_);
v___x_319_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__4));
v___x_320_ = lean_obj_once(&lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__6, &lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__6_once, _init_lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__6);
v___x_321_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__7));
lean_inc(v_currMacroScope_311_);
lean_inc(v_quotContext_310_);
v___x_322_ = l_Lean_addMacroScope(v_quotContext_310_, v___x_321_, v_currMacroScope_311_);
v___x_323_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__13));
lean_inc_n(v___x_318_, 2);
v___x_324_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_324_, 0, v___x_318_);
lean_ctor_set(v___x_324_, 1, v___x_320_);
lean_ctor_set(v___x_324_, 2, v___x_322_);
lean_ctor_set(v___x_324_, 3, v___x_323_);
v___x_325_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__15));
v___x_326_ = l_Lean_Syntax_node2(v___x_318_, v___x_325_, v___x_314_, v___x_316_);
v___x_327_ = l_Lean_Syntax_node2(v___x_318_, v___x_319_, v___x_324_, v___x_326_);
v___x_328_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_328_, 0, v___x_327_);
lean_ctor_set(v___x_328_, 1, v_a_305_);
return v___x_328_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___boxed(lean_object* v_x_329_, lean_object* v_a_330_, lean_object* v_a_331_){
_start:
{
lean_object* v_res_332_; 
v_res_332_ = lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1(v_x_329_, v_a_330_, v_a_331_);
lean_dec_ref(v_a_330_);
return v_res_332_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______unexpand__OmegaCompletePartialOrder__ContinuousHom__1(lean_object* v_x_336_, lean_object* v_a_337_, lean_object* v_a_338_){
_start:
{
lean_object* v___x_339_; uint8_t v___x_340_; 
v___x_339_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__4));
lean_inc(v_x_336_);
v___x_340_ = l_Lean_Syntax_isOfKind(v_x_336_, v___x_339_);
if (v___x_340_ == 0)
{
lean_object* v___x_341_; lean_object* v___x_342_; 
lean_dec(v_x_336_);
v___x_341_ = lean_box(0);
v___x_342_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_342_, 0, v___x_341_);
lean_ctor_set(v___x_342_, 1, v_a_338_);
return v___x_342_;
}
else
{
lean_object* v___x_343_; lean_object* v___x_344_; lean_object* v___x_345_; uint8_t v___x_346_; 
v___x_343_ = lean_unsigned_to_nat(0u);
v___x_344_ = l_Lean_Syntax_getArg(v_x_336_, v___x_343_);
v___x_345_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______unexpand__OmegaCompletePartialOrder__ContinuousHom__1___closed__1));
lean_inc(v___x_344_);
v___x_346_ = l_Lean_Syntax_isOfKind(v___x_344_, v___x_345_);
if (v___x_346_ == 0)
{
lean_object* v___x_347_; lean_object* v___x_348_; 
lean_dec(v___x_344_);
lean_dec(v_x_336_);
v___x_347_ = lean_box(0);
v___x_348_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_348_, 0, v___x_347_);
lean_ctor_set(v___x_348_, 1, v_a_338_);
return v___x_348_;
}
else
{
lean_object* v___x_349_; lean_object* v___x_350_; lean_object* v___x_351_; uint8_t v___x_352_; 
v___x_349_ = lean_unsigned_to_nat(1u);
v___x_350_ = l_Lean_Syntax_getArg(v_x_336_, v___x_349_);
lean_dec(v_x_336_);
v___x_351_ = lean_unsigned_to_nat(2u);
lean_inc(v___x_350_);
v___x_352_ = l_Lean_Syntax_matchesNull(v___x_350_, v___x_351_);
if (v___x_352_ == 0)
{
lean_object* v___x_353_; lean_object* v___x_354_; 
lean_dec(v___x_350_);
lean_dec(v___x_344_);
v___x_353_ = lean_box(0);
v___x_354_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_354_, 0, v___x_353_);
lean_ctor_set(v___x_354_, 1, v_a_338_);
return v___x_354_;
}
else
{
lean_object* v___x_355_; lean_object* v___x_356_; lean_object* v_ref_357_; uint8_t v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_361_; lean_object* v___x_362_; lean_object* v___x_363_; lean_object* v___x_364_; 
v___x_355_ = l_Lean_Syntax_getArg(v___x_350_, v___x_343_);
v___x_356_ = l_Lean_Syntax_getArg(v___x_350_, v___x_349_);
lean_dec(v___x_350_);
v_ref_357_ = l_Lean_replaceRef(v___x_344_, v_a_337_);
lean_dec(v___x_344_);
v___x_358_ = 0;
v___x_359_ = l_Lean_SourceInfo_fromRef(v_ref_357_, v___x_358_);
lean_dec(v_ref_357_);
v___x_360_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__2));
v___x_361_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_term___u2192_U0001d484___00__closed__5));
lean_inc(v___x_359_);
v___x_362_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_362_, 0, v___x_359_);
lean_ctor_set(v___x_362_, 1, v___x_361_);
v___x_363_ = l_Lean_Syntax_node3(v___x_359_, v___x_360_, v___x_355_, v___x_362_, v___x_356_);
v___x_364_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_364_, 0, v___x_363_);
lean_ctor_set(v___x_364_, 1, v_a_338_);
return v___x_364_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______unexpand__OmegaCompletePartialOrder__ContinuousHom__1___boxed(lean_object* v_x_365_, lean_object* v_a_366_, lean_object* v_a_367_){
_start:
{
lean_object* v_res_368_; 
v_res_368_ = lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______unexpand__OmegaCompletePartialOrder__ContinuousHom__1(v_x_365_, v_a_366_, v_a_367_);
lean_dec(v_a_366_);
return v_res_368_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_instPartialOrderContinuousHom(lean_object* v_00_u03b1_372_, lean_object* v_00_u03b2_373_, lean_object* v_inst_374_, lean_object* v_inst_375_){
_start:
{
lean_object* v___x_376_; 
v___x_376_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_instPartialOrderContinuousHom___closed__0));
return v___x_376_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_instPartialOrderContinuousHom___boxed(lean_object* v_00_u03b1_377_, lean_object* v_00_u03b2_378_, lean_object* v_inst_379_, lean_object* v_inst_380_){
_start:
{
lean_object* v_res_381_; 
v_res_381_ = lp_mathlib_OmegaCompletePartialOrder_instPartialOrderContinuousHom(v_00_u03b1_377_, v_00_u03b2_378_, v_inst_379_, v_inst_380_);
lean_dec_ref(v_inst_380_);
lean_dec_ref(v_inst_379_);
return v_res_381_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Simps_apply___redArg(lean_object* v_h_382_, lean_object* v_a_383_){
_start:
{
lean_object* v___x_384_; 
v___x_384_ = lean_apply_1(v_h_382_, v_a_383_);
return v___x_384_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Simps_apply(lean_object* v_00_u03b1_385_, lean_object* v_00_u03b2_386_, lean_object* v_inst_387_, lean_object* v_inst_388_, lean_object* v_h_389_, lean_object* v_a_390_){
_start:
{
lean_object* v___x_391_; 
v___x_391_ = lean_apply_1(v_h_389_, v_a_390_);
return v___x_391_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Simps_apply___boxed(lean_object* v_00_u03b1_392_, lean_object* v_00_u03b2_393_, lean_object* v_inst_394_, lean_object* v_inst_395_, lean_object* v_h_396_, lean_object* v_a_397_){
_start:
{
lean_object* v_res_398_; 
v_res_398_ = lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Simps_apply(v_00_u03b1_392_, v_00_u03b2_393_, v_inst_394_, v_inst_395_, v_h_396_, v_a_397_);
lean_dec_ref(v_inst_395_);
lean_dec_ref(v_inst_394_);
return v_res_398_;
}
}
static lean_object* _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__12(void){
_start:
{
lean_object* v___x_424_; lean_object* v___x_425_; 
v___x_424_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__11));
v___x_425_ = l_Lean_mkAtom(v___x_424_);
return v___x_425_;
}
}
static lean_object* _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__13(void){
_start:
{
lean_object* v___x_426_; lean_object* v___x_427_; lean_object* v___x_428_; 
v___x_426_ = lean_obj_once(&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__12, &lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__12_once, _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__12);
v___x_427_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__3));
v___x_428_ = lean_array_push(v___x_427_, v___x_426_);
return v___x_428_;
}
}
static lean_object* _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__17(void){
_start:
{
lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; 
v___x_439_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__16));
v___x_440_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__3));
v___x_441_ = lean_array_push(v___x_440_, v___x_439_);
return v___x_441_;
}
}
static lean_object* _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__18(void){
_start:
{
lean_object* v___x_442_; lean_object* v___x_443_; lean_object* v___x_444_; lean_object* v___x_445_; 
v___x_442_ = lean_obj_once(&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__17, &lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__17_once, _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__17);
v___x_443_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__15));
v___x_444_ = lean_box(2);
v___x_445_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_445_, 0, v___x_444_);
lean_ctor_set(v___x_445_, 1, v___x_443_);
lean_ctor_set(v___x_445_, 2, v___x_442_);
return v___x_445_;
}
}
static lean_object* _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__19(void){
_start:
{
lean_object* v___x_446_; lean_object* v___x_447_; lean_object* v___x_448_; 
v___x_446_ = lean_obj_once(&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__18, &lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__18_once, _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__18);
v___x_447_ = lean_obj_once(&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__13, &lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__13_once, _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__13);
v___x_448_ = lean_array_push(v___x_447_, v___x_446_);
return v___x_448_;
}
}
static lean_object* _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__20(void){
_start:
{
lean_object* v___x_449_; lean_object* v___x_450_; lean_object* v___x_451_; 
v___x_449_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__16));
v___x_450_ = lean_obj_once(&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__19, &lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__19_once, _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__19);
v___x_451_ = lean_array_push(v___x_450_, v___x_449_);
return v___x_451_;
}
}
static lean_object* _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__21(void){
_start:
{
lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; 
v___x_452_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__16));
v___x_453_ = lean_obj_once(&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__20, &lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__20_once, _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__20);
v___x_454_ = lean_array_push(v___x_453_, v___x_452_);
return v___x_454_;
}
}
static lean_object* _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__22(void){
_start:
{
lean_object* v___x_455_; lean_object* v___x_456_; lean_object* v___x_457_; lean_object* v___x_458_; 
v___x_455_ = lean_obj_once(&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__21, &lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__21_once, _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__21);
v___x_456_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__10));
v___x_457_ = lean_box(2);
v___x_458_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_458_, 0, v___x_457_);
lean_ctor_set(v___x_458_, 1, v___x_456_);
lean_ctor_set(v___x_458_, 2, v___x_455_);
return v___x_458_;
}
}
static lean_object* _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__23(void){
_start:
{
lean_object* v___x_459_; lean_object* v___x_460_; lean_object* v___x_461_; 
v___x_459_ = lean_obj_once(&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__22, &lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__22_once, _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__22);
v___x_460_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__3));
v___x_461_ = lean_array_push(v___x_460_, v___x_459_);
return v___x_461_;
}
}
static lean_object* _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__24(void){
_start:
{
lean_object* v___x_462_; lean_object* v___x_463_; lean_object* v___x_464_; lean_object* v___x_465_; 
v___x_462_ = lean_obj_once(&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__23, &lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__23_once, _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__23);
v___x_463_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder___aux__Mathlib__Order__OmegaCompletePartialOrder______macroRules__OmegaCompletePartialOrder__term___u2192_U0001d484____1___closed__15));
v___x_464_ = lean_box(2);
v___x_465_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_465_, 0, v___x_464_);
lean_ctor_set(v___x_465_, 1, v___x_463_);
lean_ctor_set(v___x_465_, 2, v___x_462_);
return v___x_465_;
}
}
static lean_object* _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__25(void){
_start:
{
lean_object* v___x_466_; lean_object* v___x_467_; lean_object* v___x_468_; 
v___x_466_ = lean_obj_once(&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__24, &lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__24_once, _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__24);
v___x_467_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__3));
v___x_468_ = lean_array_push(v___x_467_, v___x_466_);
return v___x_468_;
}
}
static lean_object* _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__26(void){
_start:
{
lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; 
v___x_469_ = lean_obj_once(&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__25, &lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__25_once, _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__25);
v___x_470_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__5));
v___x_471_ = lean_box(2);
v___x_472_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_472_, 0, v___x_471_);
lean_ctor_set(v___x_472_, 1, v___x_470_);
lean_ctor_set(v___x_472_, 2, v___x_469_);
return v___x_472_;
}
}
static lean_object* _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__27(void){
_start:
{
lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; 
v___x_473_ = lean_obj_once(&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__26, &lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__26_once, _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__26);
v___x_474_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__3));
v___x_475_ = lean_array_push(v___x_474_, v___x_473_);
return v___x_475_;
}
}
static lean_object* _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__28(void){
_start:
{
lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; 
v___x_476_ = lean_obj_once(&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__27, &lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__27_once, _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__27);
v___x_477_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__2));
v___x_478_ = lean_box(2);
v___x_479_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_479_, 0, v___x_478_);
lean_ctor_set(v___x_479_, 1, v___x_477_);
lean_ctor_set(v___x_479_, 2, v___x_476_);
return v___x_479_;
}
}
static lean_object* _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1(void){
_start:
{
lean_object* v___x_480_; 
v___x_480_ = lean_obj_once(&lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__28, &lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__28_once, _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1___closed__28);
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___redArg(lean_object* v_f_481_){
_start:
{
lean_inc(v_f_481_);
return v_f_481_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___redArg___boxed(lean_object* v_f_482_){
_start:
{
lean_object* v_res_483_; 
v_res_483_ = lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___redArg(v_f_482_);
lean_dec(v_f_482_);
return v_res_483_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun(lean_object* v_00_u03b1_484_, lean_object* v_00_u03b2_485_, lean_object* v_inst_486_, lean_object* v_inst_487_, lean_object* v_f_488_, lean_object* v_hf_489_){
_start:
{
lean_inc(v_f_488_);
return v_f_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___boxed(lean_object* v_00_u03b1_490_, lean_object* v_00_u03b2_491_, lean_object* v_inst_492_, lean_object* v_inst_493_, lean_object* v_f_494_, lean_object* v_hf_495_){
_start:
{
lean_object* v_res_496_; 
v_res_496_ = lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun(v_00_u03b1_490_, v_00_u03b2_491_, v_inst_492_, v_inst_493_, v_f_494_, v_hf_495_);
lean_dec(v_f_494_);
lean_dec_ref(v_inst_493_);
lean_dec_ref(v_inst_492_);
return v_res_496_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_copy___redArg(lean_object* v_f_497_){
_start:
{
lean_inc(v_f_497_);
return v_f_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_copy___redArg___boxed(lean_object* v_f_498_){
_start:
{
lean_object* v_res_499_; 
v_res_499_ = lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_copy___redArg(v_f_498_);
lean_dec(v_f_498_);
return v_res_499_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_copy(lean_object* v_00_u03b1_500_, lean_object* v_00_u03b2_501_, lean_object* v_inst_502_, lean_object* v_inst_503_, lean_object* v_f_504_, lean_object* v_g_505_, lean_object* v_h_506_){
_start:
{
lean_inc(v_f_504_);
return v_f_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_copy___boxed(lean_object* v_00_u03b1_507_, lean_object* v_00_u03b2_508_, lean_object* v_inst_509_, lean_object* v_inst_510_, lean_object* v_f_511_, lean_object* v_g_512_, lean_object* v_h_513_){
_start:
{
lean_object* v_res_514_; 
v_res_514_ = lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_copy(v_00_u03b1_507_, v_00_u03b2_508_, v_inst_509_, v_inst_510_, v_f_511_, v_g_512_, v_h_513_);
lean_dec(v_g_512_);
lean_dec(v_f_511_);
lean_dec_ref(v_inst_510_);
lean_dec_ref(v_inst_509_);
return v_res_514_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_id(lean_object* v_00_u03b1_516_, lean_object* v_inst_517_){
_start:
{
lean_object* v___x_518_; 
v___x_518_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_id___closed__0));
return v___x_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_id___boxed(lean_object* v_00_u03b1_519_, lean_object* v_inst_520_){
_start:
{
lean_object* v_res_521_; 
v_res_521_ = lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_id(v_00_u03b1_519_, v_inst_520_);
lean_dec_ref(v_inst_520_);
return v_res_521_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_comp___redArg(lean_object* v_f_522_, lean_object* v_g_523_){
_start:
{
lean_object* v___x_524_; 
v___x_524_ = lp_mathlib_OrderHom_comp___redArg(v_f_522_, v_g_523_);
return v___x_524_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_comp(lean_object* v_00_u03b1_525_, lean_object* v_00_u03b2_526_, lean_object* v_00_u03b3_527_, lean_object* v_inst_528_, lean_object* v_inst_529_, lean_object* v_inst_530_, lean_object* v_f_531_, lean_object* v_g_532_){
_start:
{
lean_object* v___x_533_; 
v___x_533_ = lp_mathlib_OrderHom_comp___redArg(v_f_531_, v_g_532_);
return v___x_533_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_comp___boxed(lean_object* v_00_u03b1_534_, lean_object* v_00_u03b2_535_, lean_object* v_00_u03b3_536_, lean_object* v_inst_537_, lean_object* v_inst_538_, lean_object* v_inst_539_, lean_object* v_f_540_, lean_object* v_g_541_){
_start:
{
lean_object* v_res_542_; 
v_res_542_ = lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_comp(v_00_u03b1_534_, v_00_u03b2_535_, v_00_u03b3_536_, v_inst_537_, v_inst_538_, v_inst_539_, v_f_540_, v_g_541_);
lean_dec_ref(v_inst_539_);
lean_dec_ref(v_inst_538_);
lean_dec_ref(v_inst_537_);
return v_res_542_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_const___redArg(lean_object* v_x_543_){
_start:
{
lean_object* v___x_544_; 
v___x_544_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_const___lam__0___boxed), 2, 1);
lean_closure_set(v___x_544_, 0, v_x_543_);
return v___x_544_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_const(lean_object* v_00_u03b1_545_, lean_object* v_00_u03b2_546_, lean_object* v_inst_547_, lean_object* v_inst_548_, lean_object* v_x_549_){
_start:
{
lean_object* v___x_550_; 
v___x_550_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_const___lam__0___boxed), 2, 1);
lean_closure_set(v___x_550_, 0, v_x_549_);
return v___x_550_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_const___boxed(lean_object* v_00_u03b1_551_, lean_object* v_00_u03b2_552_, lean_object* v_inst_553_, lean_object* v_inst_554_, lean_object* v_x_555_){
_start:
{
lean_object* v_res_556_; 
v_res_556_ = lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_const(v_00_u03b1_551_, v_00_u03b2_552_, v_inst_553_, v_inst_554_, v_x_555_);
lean_dec_ref(v_inst_554_);
lean_dec_ref(v_inst_553_);
return v_res_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_instInhabited___redArg(lean_object* v_inst_557_){
_start:
{
lean_object* v___x_558_; 
v___x_558_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_const___lam__0___boxed), 2, 1);
lean_closure_set(v___x_558_, 0, v_inst_557_);
return v___x_558_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_instInhabited(lean_object* v_00_u03b1_559_, lean_object* v_00_u03b2_560_, lean_object* v_inst_561_, lean_object* v_inst_562_, lean_object* v_inst_563_){
_start:
{
lean_object* v___x_564_; 
v___x_564_ = lean_alloc_closure((void*)(lp_mathlib_OrderHom_const___lam__0___boxed), 2, 1);
lean_closure_set(v___x_564_, 0, v_inst_563_);
return v___x_564_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_instInhabited___boxed(lean_object* v_00_u03b1_565_, lean_object* v_00_u03b2_566_, lean_object* v_inst_567_, lean_object* v_inst_568_, lean_object* v_inst_569_){
_start:
{
lean_object* v_res_570_; 
v_res_570_ = lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_instInhabited(v_00_u03b1_565_, v_00_u03b2_566_, v_inst_567_, v_inst_568_, v_inst_569_);
lean_dec_ref(v_inst_568_);
lean_dec_ref(v_inst_567_);
return v_res_570_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_toMono___lam__0(lean_object* v_f_571_, lean_object* v___y_572_){
_start:
{
lean_object* v___x_573_; 
v___x_573_ = lean_apply_1(v_f_571_, v___y_572_);
return v___x_573_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_toMono(lean_object* v_00_u03b1_575_, lean_object* v_00_u03b2_576_, lean_object* v_inst_577_, lean_object* v_inst_578_){
_start:
{
lean_object* v___f_579_; 
v___f_579_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_toMono___closed__0));
return v___f_579_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_toMono___boxed(lean_object* v_00_u03b1_580_, lean_object* v_00_u03b2_581_, lean_object* v_inst_582_, lean_object* v_inst_583_){
_start:
{
lean_object* v_res_584_; 
v_res_584_ = lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_toMono(v_00_u03b1_580_, v_00_u03b2_581_, v_inst_582_, v_inst_583_);
lean_dec_ref(v_inst_583_);
lean_dec_ref(v_inst_582_);
return v_res_584_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_00_u03c9Sup___redArg(lean_object* v_inst_585_, lean_object* v_c_586_){
_start:
{
lean_object* v___f_587_; lean_object* v___x_588_; lean_object* v___f_589_; 
v___f_587_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_toMono___closed__0));
v___x_588_ = lp_mathlib_OrderHom_comp___redArg(v___f_587_, v_c_586_);
v___f_589_ = lean_alloc_closure((void*)(lp_mathlib_OmegaCompletePartialOrder_OrderHom_00_u03c9Sup___redArg___lam__0), 3, 2);
lean_closure_set(v___f_589_, 0, v_inst_585_);
lean_closure_set(v___f_589_, 1, v___x_588_);
return v___f_589_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_00_u03c9Sup(lean_object* v_00_u03b1_590_, lean_object* v_00_u03b2_591_, lean_object* v_inst_592_, lean_object* v_inst_593_, lean_object* v_c_594_){
_start:
{
lean_object* v___x_595_; 
v___x_595_ = lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_00_u03c9Sup___redArg(v_inst_593_, v_c_594_);
return v___x_595_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_00_u03c9Sup___boxed(lean_object* v_00_u03b1_596_, lean_object* v_00_u03b2_597_, lean_object* v_inst_598_, lean_object* v_inst_599_, lean_object* v_c_600_){
_start:
{
lean_object* v_res_601_; 
v_res_601_ = lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_00_u03c9Sup(v_00_u03b1_596_, v_00_u03b2_597_, v_inst_598_, v_inst_599_, v_c_600_);
lean_dec_ref(v_inst_598_);
return v_res_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_inst___redArg(lean_object* v_inst_602_, lean_object* v_inst_603_){
_start:
{
lean_object* v___x_604_; lean_object* v___x_605_; lean_object* v___x_606_; 
v___x_604_ = lp_mathlib_OmegaCompletePartialOrder_instPartialOrderContinuousHom(lean_box(0), lean_box(0), v_inst_602_, v_inst_603_);
v___x_605_ = lean_alloc_closure((void*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_00_u03c9Sup___boxed), 5, 4);
lean_closure_set(v___x_605_, 0, lean_box(0));
lean_closure_set(v___x_605_, 1, lean_box(0));
lean_closure_set(v___x_605_, 2, v_inst_602_);
lean_closure_set(v___x_605_, 3, v_inst_603_);
v___x_606_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_606_, 0, v___x_604_);
lean_ctor_set(v___x_606_, 1, v___x_605_);
return v___x_606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_inst(lean_object* v_00_u03b1_607_, lean_object* v_00_u03b2_608_, lean_object* v_inst_609_, lean_object* v_inst_610_){
_start:
{
lean_object* v___x_611_; 
v___x_611_ = lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_inst___redArg(v_inst_609_, v_inst_610_);
return v___x_611_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Prod_apply___lam__0(lean_object* v_f_612_){
_start:
{
lean_object* v_fst_613_; lean_object* v_snd_614_; lean_object* v___x_615_; 
v_fst_613_ = lean_ctor_get(v_f_612_, 0);
lean_inc(v_fst_613_);
v_snd_614_ = lean_ctor_get(v_f_612_, 1);
lean_inc(v_snd_614_);
lean_dec_ref(v_f_612_);
v___x_615_ = lean_apply_1(v_fst_613_, v_snd_614_);
return v___x_615_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Prod_apply(lean_object* v_00_u03b1_617_, lean_object* v_00_u03b2_618_, lean_object* v_inst_619_, lean_object* v_inst_620_){
_start:
{
lean_object* v___f_621_; 
v___f_621_ = ((lean_object*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Prod_apply___closed__0));
return v___f_621_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Prod_apply___boxed(lean_object* v_00_u03b1_622_, lean_object* v_00_u03b2_623_, lean_object* v_inst_624_, lean_object* v_inst_625_){
_start:
{
lean_object* v_res_626_; 
v_res_626_ = lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_Prod_apply(v_00_u03b1_622_, v_00_u03b2_623_, v_inst_624_, v_inst_625_);
lean_dec_ref(v_inst_625_);
lean_dec_ref(v_inst_624_);
return v_res_626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_flip___redArg___lam__0(lean_object* v_f_627_, lean_object* v_x_628_, lean_object* v_y_629_){
_start:
{
lean_object* v___x_630_; 
v___x_630_ = lean_apply_2(v_f_627_, v_y_629_, v_x_628_);
return v___x_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_flip___redArg(lean_object* v_f_631_){
_start:
{
lean_object* v___f_632_; 
v___f_632_ = lean_alloc_closure((void*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_flip___redArg___lam__0), 3, 1);
lean_closure_set(v___f_632_, 0, v_f_631_);
return v___f_632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_flip(lean_object* v_00_u03b2_633_, lean_object* v_00_u03b3_634_, lean_object* v_inst_635_, lean_object* v_inst_636_, lean_object* v_00_u03b1_637_, lean_object* v_f_638_){
_start:
{
lean_object* v___f_639_; 
v___f_639_ = lean_alloc_closure((void*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_flip___redArg___lam__0), 3, 1);
lean_closure_set(v___f_639_, 0, v_f_638_);
return v___f_639_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_flip___boxed(lean_object* v_00_u03b2_640_, lean_object* v_00_u03b3_641_, lean_object* v_inst_642_, lean_object* v_inst_643_, lean_object* v_00_u03b1_644_, lean_object* v_f_645_){
_start:
{
lean_object* v_res_646_; 
v_res_646_ = lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_flip(v_00_u03b2_640_, v_00_u03b3_641_, v_inst_642_, v_inst_643_, v_00_u03b1_644_, v_f_645_);
lean_dec_ref(v_inst_643_);
lean_dec_ref(v_inst_642_);
return v_res_646_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_fixedPoints_iterateChain___redArg___lam__1(lean_object* v___f_647_, lean_object* v_x_648_, lean_object* v_n_649_){
_start:
{
lean_object* v___x_650_; 
v___x_650_ = lp_mathlib_Nat_iterate___redArg(v___f_647_, v_n_649_, v_x_648_);
return v___x_650_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_fixedPoints_iterateChain___redArg(lean_object* v_f_651_, lean_object* v_x_652_){
_start:
{
lean_object* v___f_653_; lean_object* v___f_654_; 
v___f_653_ = lean_alloc_closure((void*)(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_toMono___lam__0), 2, 1);
lean_closure_set(v___f_653_, 0, v_f_651_);
v___f_654_ = lean_alloc_closure((void*)(lp_mathlib_OmegaCompletePartialOrder_fixedPoints_iterateChain___redArg___lam__1), 3, 2);
lean_closure_set(v___f_654_, 0, v___f_653_);
lean_closure_set(v___f_654_, 1, v_x_652_);
return v___f_654_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_fixedPoints_iterateChain(lean_object* v_00_u03b1_655_, lean_object* v_inst_656_, lean_object* v_f_657_, lean_object* v_x_658_, lean_object* v_h_659_){
_start:
{
lean_object* v___x_660_; 
v___x_660_ = lp_mathlib_OmegaCompletePartialOrder_fixedPoints_iterateChain___redArg(v_f_657_, v_x_658_);
return v___x_660_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OmegaCompletePartialOrder_fixedPoints_iterateChain___boxed(lean_object* v_00_u03b1_661_, lean_object* v_inst_662_, lean_object* v_f_663_, lean_object* v_x_664_, lean_object* v_h_665_){
_start:
{
lean_object* v_res_666_; 
v_res_666_ = lp_mathlib_OmegaCompletePartialOrder_fixedPoints_iterateChain(v_00_u03b1_661_, v_inst_662_, v_f_663_, v_x_664_, v_h_665_);
lean_dec_ref(v_inst_662_);
return v_res_666_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Control_Monad_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Iterate(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Part(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_Preorder_Chain(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Order_ScottContinuity(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Dynamics_FixedPoints_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Order_OmegaCompletePartialOrder(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Control_Monad_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Part(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_Preorder_Chain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_ScottContinuity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Dynamics_FixedPoints_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Order_OmegaCompletePartialOrder(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1 = _init_lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1();
lean_mark_persistent(lp_mathlib_OmegaCompletePartialOrder_ContinuousHom_ofFun___auto__1);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Monad_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Iterate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Part(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Preorder_Chain(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_ScottContinuity(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Dynamics_FixedPoints_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_OmegaCompletePartialOrder(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Monad_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Part(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Preorder_Chain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ScottContinuity(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Dynamics_FixedPoints_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Order_OmegaCompletePartialOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Order_OmegaCompletePartialOrder(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Order_OmegaCompletePartialOrder(builtin);
}
#ifdef __cplusplus
}
#endif
