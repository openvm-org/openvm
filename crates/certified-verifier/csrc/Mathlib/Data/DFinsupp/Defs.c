// Lean compiler output
// Module: Mathlib.Data.DFinsupp.Defs
// Imports: public import Init public meta import Init public import Mathlib.Data.Set.Finite.Basic public import Mathlib.Algebra.Group.InjSurj public import Mathlib.Algebra.Group.Equiv.Defs public import Mathlib.Algebra.Group.Pi.Basic public import Mathlib.Algebra.Notation.Prod public import Mathlib.Algebra.Group.Basic
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
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
extern lean_object* lp_batteries_Batteries_ExtendedBinder_extBinders;
uint8_t lp_mathlib_Multiset_decidableMem___aux__1___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_dedup___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Syntax_node4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
lean_object* lp_mathlib_ZSMul_toSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddZeroClass_toAddZero___redArg(lean_object*);
lean_object* lp_mathlib_Function_Injective_addMonoid___redArg(lean_object*, lean_object*, lean_object*);
uint8_t l_List_decidablePerm___redArg(lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableDforallMultiset___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Function_update___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_MatchState_delabVar(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_MatchState_getBinders(lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Mathlib_Notation3_MatchState_empty;
lean_object* lp_mathlib_Mathlib_Notation3_matchVar___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchVar___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Mathlib_Notation3_matchScoped(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_cast(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_attach___redArg(lean_object*);
lean_object* lp_mathlib_Pi_single___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPNotation___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_sigmaCongrRight___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_term_u03a0_u2080___x2c___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 13, .m_capacity = 13, .m_length = 9, .m_data = "termΠ₀_,_"};
static const lean_object* lp_mathlib_term_u03a0_u2080___x2c___00__closed__0 = (const lean_object*)&lp_mathlib_term_u03a0_u2080___x2c___00__closed__0_value;
static const lean_ctor_object lp_mathlib_term_u03a0_u2080___x2c___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u03a0_u2080___x2c___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(178, 14, 213, 132, 77, 54, 249, 63)}};
static const lean_object* lp_mathlib_term_u03a0_u2080___x2c___00__closed__1 = (const lean_object*)&lp_mathlib_term_u03a0_u2080___x2c___00__closed__1_value;
static const lean_string_object lp_mathlib_term_u03a0_u2080___x2c___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_term_u03a0_u2080___x2c___00__closed__2 = (const lean_object*)&lp_mathlib_term_u03a0_u2080___x2c___00__closed__2_value;
static const lean_ctor_object lp_mathlib_term_u03a0_u2080___x2c___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u03a0_u2080___x2c___00__closed__2_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_term_u03a0_u2080___x2c___00__closed__3 = (const lean_object*)&lp_mathlib_term_u03a0_u2080___x2c___00__closed__3_value;
static const lean_string_object lp_mathlib_term_u03a0_u2080___x2c___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 2, .m_data = "Π₀"};
static const lean_object* lp_mathlib_term_u03a0_u2080___x2c___00__closed__4 = (const lean_object*)&lp_mathlib_term_u03a0_u2080___x2c___00__closed__4_value;
static const lean_ctor_object lp_mathlib_term_u03a0_u2080___x2c___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u03a0_u2080___x2c___00__closed__4_value)}};
static const lean_object* lp_mathlib_term_u03a0_u2080___x2c___00__closed__5 = (const lean_object*)&lp_mathlib_term_u03a0_u2080___x2c___00__closed__5_value;
static lean_once_cell_t lp_mathlib_term_u03a0_u2080___x2c___00__closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u03a0_u2080___x2c___00__closed__6;
static const lean_string_object lp_mathlib_term_u03a0_u2080___x2c___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = ", "};
static const lean_object* lp_mathlib_term_u03a0_u2080___x2c___00__closed__7 = (const lean_object*)&lp_mathlib_term_u03a0_u2080___x2c___00__closed__7_value;
static const lean_ctor_object lp_mathlib_term_u03a0_u2080___x2c___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_term_u03a0_u2080___x2c___00__closed__7_value)}};
static const lean_object* lp_mathlib_term_u03a0_u2080___x2c___00__closed__8 = (const lean_object*)&lp_mathlib_term_u03a0_u2080___x2c___00__closed__8_value;
static lean_once_cell_t lp_mathlib_term_u03a0_u2080___x2c___00__closed__9_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u03a0_u2080___x2c___00__closed__9;
static const lean_string_object lp_mathlib_term_u03a0_u2080___x2c___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_term_u03a0_u2080___x2c___00__closed__10 = (const lean_object*)&lp_mathlib_term_u03a0_u2080___x2c___00__closed__10_value;
static const lean_ctor_object lp_mathlib_term_u03a0_u2080___x2c___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_term_u03a0_u2080___x2c___00__closed__10_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_term_u03a0_u2080___x2c___00__closed__11 = (const lean_object*)&lp_mathlib_term_u03a0_u2080___x2c___00__closed__11_value;
static const lean_ctor_object lp_mathlib_term_u03a0_u2080___x2c___00__closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_term_u03a0_u2080___x2c___00__closed__11_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_term_u03a0_u2080___x2c___00__closed__12 = (const lean_object*)&lp_mathlib_term_u03a0_u2080___x2c___00__closed__12_value;
static lean_once_cell_t lp_mathlib_term_u03a0_u2080___x2c___00__closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u03a0_u2080___x2c___00__closed__13;
static lean_once_cell_t lp_mathlib_term_u03a0_u2080___x2c___00__closed__14_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_term_u03a0_u2080___x2c___00__closed__14;
LEAN_EXPORT lean_object* lp_mathlib_term_u03a0_u2080___x2c__;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__0_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Notation3"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 29, .m_capacity = 29, .m_length = 28, .m_data = "termExpand_binders%(_=>_)_,_"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__2_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__3_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(118, 213, 161, 2, 73, 184, 31, 228)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__3_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__3_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(187, 176, 22, 214, 10, 13, 147, 22)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__3_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(120, 7, 237, 26, 3, 243, 131, 214)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__3_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "expand_binders%"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__4_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__5_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "f"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__6_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__7_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__7;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(29, 68, 183, 24, 128, 148, 178, 23)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__8_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "=>"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__9_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__10 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__10_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__11 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__11_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__12 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__12_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__13 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__13_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__14_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__14_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__14_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__11_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__14_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__14_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__14_value_aux_2),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__14 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__14_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "DFinsupp"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__15 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__15_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__16_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__16;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(88, 134, 61, 53, 96, 200, 64, 187)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__17 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__17_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__18 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__18_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__17_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__19 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__19_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__19_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__20 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__20_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__18_value),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__20_value)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__21 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__21_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__22 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__22_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__22_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__23 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__23_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__24 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__24_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ","};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__25 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__25_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__2___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 3, .m_data = "Π₀ "};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__2___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__2___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__2(uint8_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "r"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__0_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__0_value),LEAN_SCALAR_PTR_LITERAL(201, 206, 29, 183, 206, 15, 98, 41)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__1_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Batteries"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__2_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "ExtendedBinder"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__3_value;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "extBinders"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__4_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__5_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__5_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__5_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__4_value),LEAN_SCALAR_PTR_LITERAL(142, 202, 111, 171, 129, 134, 17, 161)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__5_value;
static lean_once_cell_t lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__6_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__6;
static const lean_string_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 20, .m_capacity = 20, .m_length = 19, .m_data = "extBinderCollection"};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__7 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__7_value;
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__8_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__2_value),LEAN_SCALAR_PTR_LITERAL(70, 222, 136, 192, 226, 112, 165, 223)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__8_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__8_value_aux_0),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__3_value),LEAN_SCALAR_PTR_LITERAL(56, 78, 248, 154, 49, 0, 91, 17)}};
static const lean_ctor_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__8_value_aux_1),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__7_value),LEAN_SCALAR_PTR_LITERAL(144, 58, 22, 199, 215, 82, 42, 232)}};
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__8 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__8_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Mathlib_Notation3_matchVar___boxed, .m_arity = 9, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__8_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__9 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__9_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__0 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__0_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__1___boxed, .m_arity = 8, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__1 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__1_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__0_value),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__1_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__2 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__2_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__3 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__3_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__4 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__4_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__2_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__5 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__5_value;
static const lean_closure_object lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__4_value),((lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__5_value)} };
static const lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__6 = (const lean_object*)&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instZero___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instZero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instInhabited___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_zipWith___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_zipWith___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_zipWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_zipWith___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_piecewise___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_piecewise___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_piecewise___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_piecewise(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_piecewise___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAdd___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAdd___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAdd(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_addZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_addZeroClass(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_hasNatScalar___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_hasNatScalar___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_hasNatScalar___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_hasNatScalar(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAddMonoid___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAddMonoid___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAddMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAddMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coeFnAddMonoidHom___lam__0(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DFinsupp_coeFnAddMonoidHom___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DFinsupp_coeFnAddMonoidHom___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DFinsupp_coeFnAddMonoidHom___closed__0 = (const lean_object*)&lp_mathlib_DFinsupp_coeFnAddMonoidHom___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coeFnAddMonoidHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coeFnAddMonoidHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_addCommMonoid___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_addCommMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_addCommMonoid(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instNeg___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instNeg___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instNeg___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instNeg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSub___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSub___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSub___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSub(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_hasIntScalar___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_hasIntScalar___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_hasIntScalar___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_hasIntScalar(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAddGroup___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAddGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAddGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_addCommGroup___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_addCommGroup(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filter___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filterAddMonoidHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filterAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomain___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomain___redArg___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomain___redArg___lam__1___boxed(lean_object*);
static const lean_closure_object lp_mathlib_DFinsupp_subtypeDomain___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DFinsupp_subtypeDomain___redArg___lam__1___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DFinsupp_subtypeDomain___redArg___closed__0 = (const lean_object*)&lp_mathlib_DFinsupp_subtypeDomain___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomain___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomain(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomain___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomainAddMonoidHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomainAddMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mk___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mk___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mk(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_unique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_unique(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_uniqueOfIsEmpty___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_uniqueOfIsEmpty(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivFunOnFintype___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivFunOnFintype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivFunOnFintype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivFunOnFintype___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_single___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_single(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_erase___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_erase___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_erase(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_update___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_update___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_update(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_update___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_singleAddHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_singleAddHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_eraseAddHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_eraseAddHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mkAddGroupHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mkAddGroupHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_support___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_support___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_support___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_support(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_support___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeSupportEqEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeSupportEqEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeSupportEqEquiv___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DFinsupp_subtypeSupportEqEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DFinsupp_subtypeSupportEqEquiv___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DFinsupp_subtypeSupportEqEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_DFinsupp_subtypeSupportEqEquiv___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeSupportEqEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeSupportEqEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeSupportEqEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaFinsetFunEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaFinsetFunEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableZero___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableZero___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableZero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableZero___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableZero(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableZero___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__1(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_instDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_comapDomain_x27___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_comapDomain_x27___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_comapDomain_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_comapDomain_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__1___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__1___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__2(lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_DFinsupp_equivCongrLeft___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__1___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DFinsupp_equivCongrLeft___redArg___closed__0 = (const lean_object*)&lp_mathlib_DFinsupp_equivCongrLeft___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivCongrLeft___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivCongrLeft(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_extendWith___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_extendWith___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_extendWith___redArg___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_DFinsupp_extendWith___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_DFinsupp_extendWith___redArg___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_DFinsupp_extendWith___redArg___closed__0 = (const lean_object*)&lp_mathlib_DFinsupp_extendWith___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_extendWith___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_extendWith(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_extendWith___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_addMonoidHom___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_addMonoidHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_addMonoidHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_addEquiv___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_addEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_addEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_addEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_term_u03a0_u2080___x2c___00__closed__6(void){
_start:
{
lean_object* v___x_10_; lean_object* v___x_11_; lean_object* v___x_12_; lean_object* v___x_13_; 
v___x_10_ = lp_batteries_Batteries_ExtendedBinder_extBinders;
v___x_11_ = ((lean_object*)(lp_mathlib_term_u03a0_u2080___x2c___00__closed__5));
v___x_12_ = ((lean_object*)(lp_mathlib_term_u03a0_u2080___x2c___00__closed__3));
v___x_13_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_13_, 0, v___x_12_);
lean_ctor_set(v___x_13_, 1, v___x_11_);
lean_ctor_set(v___x_13_, 2, v___x_10_);
return v___x_13_;
}
}
static lean_object* _init_lp_mathlib_term_u03a0_u2080___x2c___00__closed__9(void){
_start:
{
lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; 
v___x_17_ = ((lean_object*)(lp_mathlib_term_u03a0_u2080___x2c___00__closed__8));
v___x_18_ = lean_obj_once(&lp_mathlib_term_u03a0_u2080___x2c___00__closed__6, &lp_mathlib_term_u03a0_u2080___x2c___00__closed__6_once, _init_lp_mathlib_term_u03a0_u2080___x2c___00__closed__6);
v___x_19_ = ((lean_object*)(lp_mathlib_term_u03a0_u2080___x2c___00__closed__3));
v___x_20_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_20_, 0, v___x_19_);
lean_ctor_set(v___x_20_, 1, v___x_18_);
lean_ctor_set(v___x_20_, 2, v___x_17_);
return v___x_20_;
}
}
static lean_object* _init_lp_mathlib_term_u03a0_u2080___x2c___00__closed__13(void){
_start:
{
lean_object* v___x_27_; lean_object* v___x_28_; lean_object* v___x_29_; lean_object* v___x_30_; 
v___x_27_ = ((lean_object*)(lp_mathlib_term_u03a0_u2080___x2c___00__closed__12));
v___x_28_ = lean_obj_once(&lp_mathlib_term_u03a0_u2080___x2c___00__closed__9, &lp_mathlib_term_u03a0_u2080___x2c___00__closed__9_once, _init_lp_mathlib_term_u03a0_u2080___x2c___00__closed__9);
v___x_29_ = ((lean_object*)(lp_mathlib_term_u03a0_u2080___x2c___00__closed__3));
v___x_30_ = lean_alloc_ctor(2, 3, 0);
lean_ctor_set(v___x_30_, 0, v___x_29_);
lean_ctor_set(v___x_30_, 1, v___x_28_);
lean_ctor_set(v___x_30_, 2, v___x_27_);
return v___x_30_;
}
}
static lean_object* _init_lp_mathlib_term_u03a0_u2080___x2c___00__closed__14(void){
_start:
{
lean_object* v___x_31_; lean_object* v___x_32_; lean_object* v___x_33_; lean_object* v___x_34_; 
v___x_31_ = lean_obj_once(&lp_mathlib_term_u03a0_u2080___x2c___00__closed__13, &lp_mathlib_term_u03a0_u2080___x2c___00__closed__13_once, _init_lp_mathlib_term_u03a0_u2080___x2c___00__closed__13);
v___x_32_ = lean_unsigned_to_nat(1022u);
v___x_33_ = ((lean_object*)(lp_mathlib_term_u03a0_u2080___x2c___00__closed__1));
v___x_34_ = lean_alloc_ctor(3, 3, 0);
lean_ctor_set(v___x_34_, 0, v___x_33_);
lean_ctor_set(v___x_34_, 1, v___x_32_);
lean_ctor_set(v___x_34_, 2, v___x_31_);
return v___x_34_;
}
}
static lean_object* _init_lp_mathlib_term_u03a0_u2080___x2c__(void){
_start:
{
lean_object* v___x_35_; 
v___x_35_ = lean_obj_once(&lp_mathlib_term_u03a0_u2080___x2c___00__closed__14, &lp_mathlib_term_u03a0_u2080___x2c___00__closed__14_once, _init_lp_mathlib_term_u03a0_u2080___x2c___00__closed__14);
return v___x_35_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__7(void){
_start:
{
lean_object* v___x_46_; lean_object* v___x_47_; 
v___x_46_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__6));
v___x_47_ = l_String_toRawSubstring_x27(v___x_46_);
return v___x_47_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__16(void){
_start:
{
lean_object* v___x_61_; lean_object* v___x_62_; 
v___x_61_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__15));
v___x_62_ = l_String_toRawSubstring_x27(v___x_61_);
return v___x_62_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1(lean_object* v_x_81_, lean_object* v_a_82_, lean_object* v_a_83_){
_start:
{
lean_object* v___x_84_; uint8_t v___x_85_; 
v___x_84_ = ((lean_object*)(lp_mathlib_term_u03a0_u2080___x2c___00__closed__1));
lean_inc(v_x_81_);
v___x_85_ = l_Lean_Syntax_isOfKind(v_x_81_, v___x_84_);
if (v___x_85_ == 0)
{
lean_object* v___x_86_; lean_object* v___x_87_; 
lean_dec(v_x_81_);
v___x_86_ = lean_box(1);
v___x_87_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_87_, 0, v___x_86_);
lean_ctor_set(v___x_87_, 1, v_a_83_);
return v___x_87_;
}
else
{
lean_object* v_quotContext_88_; lean_object* v_currMacroScope_89_; lean_object* v_ref_90_; lean_object* v___x_91_; lean_object* v___x_92_; lean_object* v___x_93_; lean_object* v___x_94_; uint8_t v___x_95_; lean_object* v___x_96_; lean_object* v___x_97_; lean_object* v___x_98_; lean_object* v___x_99_; lean_object* v___x_100_; lean_object* v___x_101_; lean_object* v___x_102_; lean_object* v___x_103_; lean_object* v___x_104_; lean_object* v___x_105_; lean_object* v___x_106_; lean_object* v___x_107_; lean_object* v___x_108_; lean_object* v___x_109_; lean_object* v___x_110_; lean_object* v___x_111_; lean_object* v___x_112_; lean_object* v___x_113_; lean_object* v___x_114_; lean_object* v___x_115_; lean_object* v___x_116_; lean_object* v___x_117_; lean_object* v___x_118_; lean_object* v___x_119_; lean_object* v___x_120_; lean_object* v___x_121_; lean_object* v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; lean_object* v___x_127_; lean_object* v___x_128_; lean_object* v___x_129_; lean_object* v___x_130_; lean_object* v___x_131_; lean_object* v___x_132_; lean_object* v___x_133_; lean_object* v___x_134_; 
v_quotContext_88_ = lean_ctor_get(v_a_82_, 1);
v_currMacroScope_89_ = lean_ctor_get(v_a_82_, 2);
v_ref_90_ = lean_ctor_get(v_a_82_, 5);
v___x_91_ = lean_unsigned_to_nat(1u);
v___x_92_ = l_Lean_Syntax_getArg(v_x_81_, v___x_91_);
v___x_93_ = lean_unsigned_to_nat(3u);
v___x_94_ = l_Lean_Syntax_getArg(v_x_81_, v___x_93_);
lean_dec(v_x_81_);
v___x_95_ = 0;
v___x_96_ = l_Lean_SourceInfo_fromRef(v_ref_90_, v___x_95_);
v___x_97_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__3));
v___x_98_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__4));
lean_inc_n(v___x_96_, 9);
v___x_99_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_99_, 0, v___x_96_);
lean_ctor_set(v___x_99_, 1, v___x_98_);
v___x_100_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__5));
v___x_101_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_101_, 0, v___x_96_);
lean_ctor_set(v___x_101_, 1, v___x_100_);
v___x_102_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__7, &lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__7_once, _init_lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__7);
v___x_103_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__8));
lean_inc_n(v_currMacroScope_89_, 2);
lean_inc_n(v_quotContext_88_, 2);
v___x_104_ = l_Lean_addMacroScope(v_quotContext_88_, v___x_103_, v_currMacroScope_89_);
v___x_105_ = lean_box(0);
v___x_106_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_106_, 0, v___x_96_);
lean_ctor_set(v___x_106_, 1, v___x_102_);
lean_ctor_set(v___x_106_, 2, v___x_104_);
lean_ctor_set(v___x_106_, 3, v___x_105_);
v___x_107_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__9));
v___x_108_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_108_, 0, v___x_96_);
lean_ctor_set(v___x_108_, 1, v___x_107_);
v___x_109_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__14));
v___x_110_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__16, &lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__16_once, _init_lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__16);
v___x_111_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__17));
v___x_112_ = l_Lean_addMacroScope(v_quotContext_88_, v___x_111_, v_currMacroScope_89_);
v___x_113_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__21));
v___x_114_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_114_, 0, v___x_96_);
lean_ctor_set(v___x_114_, 1, v___x_110_);
lean_ctor_set(v___x_114_, 2, v___x_112_);
lean_ctor_set(v___x_114_, 3, v___x_113_);
v___x_115_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__23));
lean_inc_ref(v___x_106_);
v___x_116_ = l_Lean_Syntax_node1(v___x_96_, v___x_115_, v___x_106_);
v___x_117_ = l_Lean_Syntax_node2(v___x_96_, v___x_109_, v___x_114_, v___x_116_);
v___x_118_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__24));
v___x_119_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_119_, 0, v___x_96_);
lean_ctor_set(v___x_119_, 1, v___x_118_);
v___x_120_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__25));
v___x_121_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_121_, 0, v___x_96_);
lean_ctor_set(v___x_121_, 1, v___x_120_);
v___x_122_ = lean_unsigned_to_nat(9u);
v___x_123_ = lean_mk_empty_array_with_capacity(v___x_122_);
v___x_124_ = lean_array_push(v___x_123_, v___x_99_);
v___x_125_ = lean_array_push(v___x_124_, v___x_101_);
v___x_126_ = lean_array_push(v___x_125_, v___x_106_);
v___x_127_ = lean_array_push(v___x_126_, v___x_108_);
v___x_128_ = lean_array_push(v___x_127_, v___x_117_);
v___x_129_ = lean_array_push(v___x_128_, v___x_119_);
v___x_130_ = lean_array_push(v___x_129_, v___x_92_);
v___x_131_ = lean_array_push(v___x_130_, v___x_121_);
v___x_132_ = lean_array_push(v___x_131_, v___x_94_);
v___x_133_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_133_, 0, v___x_96_);
lean_ctor_set(v___x_133_, 1, v___x_97_);
lean_ctor_set(v___x_133_, 2, v___x_132_);
v___x_134_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_134_, 0, v___x_133_);
lean_ctor_set(v___x_134_, 1, v_a_83_);
return v___x_134_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___boxed(lean_object* v_x_135_, lean_object* v_a_136_, lean_object* v_a_137_){
_start:
{
lean_object* v_res_138_; 
v_res_138_ = lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1(v_x_135_, v_a_136_, v_a_137_);
lean_dec_ref(v_a_136_);
return v_res_138_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1_spec__0___redArg(lean_object* v___y_139_){
_start:
{
lean_object* v_subExpr_141_; lean_object* v_expr_142_; lean_object* v___x_143_; 
v_subExpr_141_ = lean_ctor_get(v___y_139_, 3);
v_expr_142_ = lean_ctor_get(v_subExpr_141_, 0);
lean_inc_ref(v_expr_142_);
v___x_143_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_143_, 0, v_expr_142_);
return v___x_143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1_spec__0___redArg___boxed(lean_object* v___y_144_, lean_object* v___y_145_){
_start:
{
lean_object* v_res_146_; 
v_res_146_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1_spec__0___redArg(v___y_144_);
lean_dec_ref(v___y_144_);
return v_res_146_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1_spec__0(lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_, lean_object* v___y_151_, lean_object* v___y_152_){
_start:
{
lean_object* v___x_154_; 
v___x_154_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1_spec__0___redArg(v___y_147_);
return v___x_154_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1_spec__0___boxed(lean_object* v___y_155_, lean_object* v___y_156_, lean_object* v___y_157_, lean_object* v___y_158_, lean_object* v___y_159_, lean_object* v___y_160_, lean_object* v___y_161_){
_start:
{
lean_object* v_res_162_; 
v_res_162_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1_spec__0(v___y_155_, v___y_156_, v___y_157_, v___y_158_, v___y_159_, v___y_160_);
lean_dec(v___y_160_);
lean_dec_ref(v___y_159_);
lean_dec(v___y_158_);
lean_dec_ref(v___y_157_);
lean_dec(v___y_156_);
lean_dec_ref(v___y_155_);
return v_res_162_;
}
}
LEAN_EXPORT uint8_t lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__0(lean_object* v_x_163_){
_start:
{
lean_object* v___x_164_; uint8_t v___x_165_; 
v___x_164_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__17));
v___x_165_ = l_Lean_Expr_isConstOf(v_x_163_, v___x_164_);
return v___x_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__0___boxed(lean_object* v_x_166_){
_start:
{
uint8_t v_res_167_; lean_object* v_r_168_; 
v_res_167_ = lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__0(v_x_166_);
lean_dec_ref(v_x_166_);
v_r_168_ = lean_box(v_res_167_);
return v_r_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__1(lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_){
_start:
{
lean_object* v___x_177_; 
v___x_177_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_177_, 0, v___y_169_);
return v___x_177_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__1___boxed(lean_object* v___y_178_, lean_object* v___y_179_, lean_object* v___y_180_, lean_object* v___y_181_, lean_object* v___y_182_, lean_object* v___y_183_, lean_object* v___y_184_, lean_object* v___y_185_){
_start:
{
lean_object* v_res_186_; 
v_res_186_ = lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__1(v___y_178_, v___y_179_, v___y_180_, v___y_181_, v___y_182_, v___y_183_, v___y_184_);
lean_dec(v___y_184_);
lean_dec_ref(v___y_183_);
lean_dec(v___y_182_);
lean_dec_ref(v___y_181_);
lean_dec(v___y_180_);
lean_dec_ref(v___y_179_);
return v_res_186_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__2(uint8_t v___x_188_, lean_object* v___x_189_, lean_object* v_a_190_, lean_object* v___y_191_, lean_object* v___y_192_, lean_object* v___y_193_, lean_object* v___y_194_, lean_object* v___y_195_, lean_object* v___y_196_){
_start:
{
lean_object* v_ref_198_; lean_object* v___x_199_; lean_object* v___x_200_; lean_object* v___x_201_; lean_object* v___x_202_; lean_object* v___x_203_; lean_object* v___x_204_; lean_object* v___x_205_; lean_object* v___x_206_; 
v_ref_198_ = lean_ctor_get(v___y_195_, 5);
v___x_199_ = l_Lean_SourceInfo_fromRef(v_ref_198_, v___x_188_);
v___x_200_ = ((lean_object*)(lp_mathlib_term_u03a0_u2080___x2c___00__closed__1));
v___x_201_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__2___closed__0));
lean_inc_n(v___x_199_, 2);
v___x_202_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_202_, 0, v___x_199_);
lean_ctor_set(v___x_202_, 1, v___x_201_);
v___x_203_ = ((lean_object*)(lp_mathlib_term_u03a0_u2080___x2c___00__closed__7));
v___x_204_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_204_, 0, v___x_199_);
lean_ctor_set(v___x_204_, 1, v___x_203_);
v___x_205_ = l_Lean_Syntax_node4(v___x_199_, v___x_200_, v___x_202_, v___x_189_, v___x_204_, v_a_190_);
v___x_206_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_206_, 0, v___x_205_);
return v___x_206_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__2___boxed(lean_object* v___x_207_, lean_object* v___x_208_, lean_object* v_a_209_, lean_object* v___y_210_, lean_object* v___y_211_, lean_object* v___y_212_, lean_object* v___y_213_, lean_object* v___y_214_, lean_object* v___y_215_, lean_object* v___y_216_){
_start:
{
uint8_t v___x_6995__boxed_217_; lean_object* v_res_218_; 
v___x_6995__boxed_217_ = lean_unbox(v___x_207_);
v_res_218_ = lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__2(v___x_6995__boxed_217_, v___x_208_, v_a_209_, v___y_210_, v___y_211_, v___y_212_, v___y_213_, v___y_214_, v___y_215_);
lean_dec(v___y_215_);
lean_dec_ref(v___y_214_);
lean_dec(v___y_213_);
lean_dec_ref(v___y_212_);
lean_dec(v___y_211_);
lean_dec_ref(v___y_210_);
return v_res_218_;
}
}
static lean_object* _init_lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__6(void){
_start:
{
lean_object* v___x_229_; 
v___x_229_ = l_Array_mkArray0(lean_box(0));
return v___x_229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3(lean_object* v___f_237_, lean_object* v___f_238_, lean_object* v___y_239_, lean_object* v___y_240_, lean_object* v___y_241_, lean_object* v___y_242_, lean_object* v___y_243_, lean_object* v___y_244_){
_start:
{
lean_object* v___x_246_; lean_object* v_a_247_; lean_object* v___x_249_; uint8_t v_isShared_250_; uint8_t v_isSharedCheck_293_; 
v___x_246_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1_spec__0___redArg(v___y_239_);
v_a_247_ = lean_ctor_get(v___x_246_, 0);
v_isSharedCheck_293_ = !lean_is_exclusive(v___x_246_);
if (v_isSharedCheck_293_ == 0)
{
v___x_249_ = v___x_246_;
v_isShared_250_ = v_isSharedCheck_293_;
goto v_resetjp_248_;
}
else
{
lean_inc(v_a_247_);
lean_dec(v___x_246_);
v___x_249_ = lean_box(0);
v_isShared_250_ = v_isSharedCheck_293_;
goto v_resetjp_248_;
}
v_resetjp_248_:
{
lean_object* v___x_251_; lean_object* v___y_253_; lean_object* v___x_283_; lean_object* v___x_284_; 
v___x_251_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__1));
v___x_283_ = lp_mathlib_Mathlib_Notation3_MatchState_empty;
v___x_284_ = lp_mathlib_Mathlib_Notation3_matchVar___redArg(v___x_251_, v___x_283_, v___y_239_, v___y_241_);
if (lean_obj_tag(v___x_284_) == 0)
{
lean_object* v_a_285_; lean_object* v___x_286_; lean_object* v___x_287_; lean_object* v___x_288_; lean_object* v___x_289_; lean_object* v___x_290_; lean_object* v___x_291_; lean_object* v___x_292_; 
v_a_285_ = lean_ctor_get(v___x_284_, 0);
lean_inc(v_a_285_);
lean_dec_ref_known(v___x_284_, 1);
v___x_286_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchExpr___boxed), 9, 1);
lean_closure_set(v___x_286_, 0, v___f_237_);
lean_inc_ref(v___f_238_);
v___x_287_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_287_, 0, v___x_286_);
lean_closure_set(v___x_287_, 1, v___f_238_);
v___x_288_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__8));
v___x_289_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__9));
v___x_290_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_290_, 0, v___x_287_);
lean_closure_set(v___x_290_, 1, v___x_289_);
v___x_291_ = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Notation3_matchApp___boxed), 10, 2);
lean_closure_set(v___x_291_, 0, v___x_290_);
lean_closure_set(v___x_291_, 1, v___f_238_);
v___x_292_ = lp_mathlib_Mathlib_Notation3_matchScoped(v___x_251_, v___x_288_, v___x_291_, v_a_285_, v___y_239_, v___y_240_, v___y_241_, v___y_242_, v___y_243_, v___y_244_);
v___y_253_ = v___x_292_;
goto v___jp_252_;
}
else
{
lean_dec_ref(v___f_238_);
lean_dec_ref(v___f_237_);
v___y_253_ = v___x_284_;
goto v___jp_252_;
}
v___jp_252_:
{
if (lean_obj_tag(v___y_253_) == 0)
{
lean_object* v_a_254_; lean_object* v_ref_255_; lean_object* v___x_257_; 
v_a_254_ = lean_ctor_get(v___y_253_, 0);
lean_inc(v_a_254_);
lean_dec_ref_known(v___y_253_, 1);
v_ref_255_ = lean_ctor_get(v___y_243_, 5);
if (v_isShared_250_ == 0)
{
lean_ctor_set_tag(v___x_249_, 1);
v___x_257_ = v___x_249_;
goto v_reusejp_256_;
}
else
{
lean_object* v_reuseFailAlloc_274_; 
v_reuseFailAlloc_274_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_274_, 0, v_a_247_);
v___x_257_ = v_reuseFailAlloc_274_;
goto v_reusejp_256_;
}
v_reusejp_256_:
{
lean_object* v___x_258_; 
v___x_258_ = lp_mathlib_Mathlib_Notation3_MatchState_delabVar(v_a_254_, v___x_251_, v___x_257_, v___y_239_, v___y_240_, v___y_241_, v___y_242_, v___y_243_, v___y_244_);
if (lean_obj_tag(v___x_258_) == 0)
{
lean_object* v_a_259_; uint8_t v___x_260_; lean_object* v___x_261_; lean_object* v___x_262_; lean_object* v___x_263_; lean_object* v___x_264_; lean_object* v___x_265_; lean_object* v___x_266_; lean_object* v___x_267_; lean_object* v___x_268_; lean_object* v___x_269_; lean_object* v___x_270_; lean_object* v___x_271_; lean_object* v___f_272_; lean_object* v___x_273_; 
v_a_259_ = lean_ctor_get(v___x_258_, 0);
lean_inc(v_a_259_);
lean_dec_ref_known(v___x_258_, 1);
v___x_260_ = 0;
v___x_261_ = l_Lean_SourceInfo_fromRef(v_ref_255_, v___x_260_);
v___x_262_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__5));
v___x_263_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______macroRules__term_u03a0_u2080___x2c____1___closed__23));
v___x_264_ = lean_obj_once(&lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__6, &lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__6_once, _init_lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__6);
v___x_265_ = lp_mathlib_Mathlib_Notation3_MatchState_getBinders(v_a_254_);
lean_dec(v_a_254_);
v___x_266_ = l_Array_append___redArg(v___x_264_, v___x_265_);
lean_dec_ref(v___x_265_);
lean_inc_n(v___x_261_, 2);
v___x_267_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_267_, 0, v___x_261_);
lean_ctor_set(v___x_267_, 1, v___x_263_);
lean_ctor_set(v___x_267_, 2, v___x_266_);
v___x_268_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___closed__8));
v___x_269_ = l_Lean_Syntax_node1(v___x_261_, v___x_268_, v___x_267_);
v___x_270_ = l_Lean_Syntax_node1(v___x_261_, v___x_262_, v___x_269_);
v___x_271_ = lean_box(v___x_260_);
v___f_272_ = lean_alloc_closure((void*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__2___boxed), 10, 3);
lean_closure_set(v___f_272_, 0, v___x_271_);
lean_closure_set(v___f_272_, 1, v___x_270_);
lean_closure_set(v___f_272_, 2, v_a_259_);
v___x_273_ = lp_mathlib_Mathlib_Notation3_withHeadRefIfTagAppFns(v___f_272_, v___y_239_, v___y_240_, v___y_241_, v___y_242_, v___y_243_, v___y_244_);
return v___x_273_;
}
else
{
lean_dec(v_a_254_);
return v___x_258_;
}
}
}
else
{
lean_object* v_a_275_; lean_object* v___x_277_; uint8_t v_isShared_278_; uint8_t v_isSharedCheck_282_; 
lean_del_object(v___x_249_);
lean_dec(v_a_247_);
v_a_275_ = lean_ctor_get(v___y_253_, 0);
v_isSharedCheck_282_ = !lean_is_exclusive(v___y_253_);
if (v_isSharedCheck_282_ == 0)
{
v___x_277_ = v___y_253_;
v_isShared_278_ = v_isSharedCheck_282_;
goto v_resetjp_276_;
}
else
{
lean_inc(v_a_275_);
lean_dec(v___y_253_);
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
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3___boxed(lean_object* v___f_294_, lean_object* v___f_295_, lean_object* v___y_296_, lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_){
_start:
{
lean_object* v_res_303_; 
v_res_303_ = lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___lam__3(v___f_294_, v___f_295_, v___y_296_, v___y_297_, v___y_298_, v___y_299_, v___y_300_, v___y_301_);
lean_dec(v___y_301_);
lean_dec_ref(v___y_300_);
lean_dec(v___y_299_);
lean_dec_ref(v___y_298_);
lean_dec(v___y_297_);
lean_dec_ref(v___y_296_);
return v_res_303_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1(lean_object* v_a_317_, lean_object* v_a_318_, lean_object* v_a_319_, lean_object* v_a_320_, lean_object* v_a_321_, lean_object* v_a_322_){
_start:
{
lean_object* v___x_324_; lean_object* v___x_325_; lean_object* v___x_326_; 
v___x_324_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__3));
v___x_325_ = ((lean_object*)(lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___closed__6));
v___x_326_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_324_, v___x_325_, v_a_317_, v_a_318_, v_a_319_, v_a_320_, v_a_321_, v_a_322_);
return v___x_326_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1___boxed(lean_object* v_a_327_, lean_object* v_a_328_, lean_object* v_a_329_, lean_object* v_a_330_, lean_object* v_a_331_, lean_object* v_a_332_, lean_object* v_a_333_){
_start:
{
lean_object* v_res_334_; 
v_res_334_ = lp_mathlib___aux__Mathlib__Data__DFinsupp__Defs______delab__app__term_u03a0_u2080___x2c____1(v_a_327_, v_a_328_, v_a_329_, v_a_330_, v_a_331_, v_a_332_);
lean_dec(v_a_332_);
lean_dec_ref(v_a_331_);
lean_dec(v_a_330_);
lean_dec_ref(v_a_329_);
lean_dec(v_a_328_);
lean_dec_ref(v_a_327_);
return v_res_334_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instZero___redArg___lam__0(lean_object* v_inst_335_, lean_object* v_x_336_){
_start:
{
lean_object* v___x_337_; 
v___x_337_ = lean_apply_1(v_inst_335_, v_x_336_);
return v___x_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instZero___redArg(lean_object* v_inst_338_){
_start:
{
lean_object* v___f_339_; lean_object* v___x_340_; lean_object* v___x_341_; 
v___f_339_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instZero___redArg___lam__0), 2, 1);
lean_closure_set(v___f_339_, 0, v_inst_338_);
v___x_340_ = lean_box(0);
v___x_341_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_341_, 0, v___f_339_);
lean_ctor_set(v___x_341_, 1, v___x_340_);
return v___x_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instZero(lean_object* v_00_u03b9_342_, lean_object* v_00_u03b2_343_, lean_object* v_inst_344_){
_start:
{
lean_object* v___x_345_; 
v___x_345_ = lp_mathlib_DFinsupp_instZero___redArg(v_inst_344_);
return v___x_345_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instInhabited___redArg(lean_object* v_inst_346_){
_start:
{
lean_object* v___f_347_; lean_object* v___x_348_; lean_object* v___x_349_; 
v___f_347_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instZero___redArg___lam__0), 2, 1);
lean_closure_set(v___f_347_, 0, v_inst_346_);
v___x_348_ = lean_box(0);
v___x_349_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_349_, 0, v___f_347_);
lean_ctor_set(v___x_349_, 1, v___x_348_);
return v___x_349_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instInhabited(lean_object* v_00_u03b9_350_, lean_object* v_00_u03b2_351_, lean_object* v_inst_352_){
_start:
{
lean_object* v___x_353_; 
v___x_353_ = lp_mathlib_DFinsupp_instInhabited___redArg(v_inst_352_);
return v___x_353_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange___redArg___lam__0(lean_object* v_toFun_354_, lean_object* v_f_355_, lean_object* v_i_356_){
_start:
{
lean_object* v___x_357_; lean_object* v___x_358_; 
lean_inc(v_i_356_);
v___x_357_ = lean_apply_1(v_toFun_354_, v_i_356_);
v___x_358_ = lean_apply_2(v_f_355_, v_i_356_, v___x_357_);
return v___x_358_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange___redArg(lean_object* v_f_359_, lean_object* v_x_360_){
_start:
{
lean_object* v_toFun_361_; lean_object* v_support_x27_362_; lean_object* v___x_364_; uint8_t v_isShared_365_; uint8_t v_isSharedCheck_370_; 
v_toFun_361_ = lean_ctor_get(v_x_360_, 0);
v_support_x27_362_ = lean_ctor_get(v_x_360_, 1);
v_isSharedCheck_370_ = !lean_is_exclusive(v_x_360_);
if (v_isSharedCheck_370_ == 0)
{
v___x_364_ = v_x_360_;
v_isShared_365_ = v_isSharedCheck_370_;
goto v_resetjp_363_;
}
else
{
lean_inc(v_support_x27_362_);
lean_inc(v_toFun_361_);
lean_dec(v_x_360_);
v___x_364_ = lean_box(0);
v_isShared_365_ = v_isSharedCheck_370_;
goto v_resetjp_363_;
}
v_resetjp_363_:
{
lean_object* v___f_366_; lean_object* v___x_368_; 
v___f_366_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_mapRange___redArg___lam__0), 3, 2);
lean_closure_set(v___f_366_, 0, v_toFun_361_);
lean_closure_set(v___f_366_, 1, v_f_359_);
if (v_isShared_365_ == 0)
{
lean_ctor_set(v___x_364_, 0, v___f_366_);
v___x_368_ = v___x_364_;
goto v_reusejp_367_;
}
else
{
lean_object* v_reuseFailAlloc_369_; 
v_reuseFailAlloc_369_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_369_, 0, v___f_366_);
lean_ctor_set(v_reuseFailAlloc_369_, 1, v_support_x27_362_);
v___x_368_ = v_reuseFailAlloc_369_;
goto v_reusejp_367_;
}
v_reusejp_367_:
{
return v___x_368_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange(lean_object* v_00_u03b9_371_, lean_object* v_00_u03b2_u2081_372_, lean_object* v_00_u03b2_u2082_373_, lean_object* v_inst_374_, lean_object* v_inst_375_, lean_object* v_f_376_, lean_object* v_hf_377_, lean_object* v_x_378_){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lp_mathlib_DFinsupp_mapRange___redArg(v_f_376_, v_x_378_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange___boxed(lean_object* v_00_u03b9_380_, lean_object* v_00_u03b2_u2081_381_, lean_object* v_00_u03b2_u2082_382_, lean_object* v_inst_383_, lean_object* v_inst_384_, lean_object* v_f_385_, lean_object* v_hf_386_, lean_object* v_x_387_){
_start:
{
lean_object* v_res_388_; 
v_res_388_ = lp_mathlib_DFinsupp_mapRange(v_00_u03b9_380_, v_00_u03b2_u2081_381_, v_00_u03b2_u2082_382_, v_inst_383_, v_inst_384_, v_f_385_, v_hf_386_, v_x_387_);
lean_dec(v_inst_384_);
lean_dec(v_inst_383_);
return v_res_388_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_zipWith___redArg___lam__0(lean_object* v_toFun_389_, lean_object* v_toFun_390_, lean_object* v_f_391_, lean_object* v_i_392_){
_start:
{
lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; 
lean_inc_n(v_i_392_, 2);
v___x_393_ = lean_apply_1(v_toFun_389_, v_i_392_);
v___x_394_ = lean_apply_1(v_toFun_390_, v_i_392_);
v___x_395_ = lean_apply_3(v_f_391_, v_i_392_, v___x_393_, v___x_394_);
return v___x_395_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_zipWith___redArg(lean_object* v_f_396_, lean_object* v_x_397_, lean_object* v_y_398_){
_start:
{
lean_object* v_toFun_399_; lean_object* v_support_x27_400_; lean_object* v_toFun_401_; lean_object* v_support_x27_402_; lean_object* v___x_404_; uint8_t v_isShared_405_; uint8_t v_isSharedCheck_411_; 
v_toFun_399_ = lean_ctor_get(v_x_397_, 0);
lean_inc(v_toFun_399_);
v_support_x27_400_ = lean_ctor_get(v_x_397_, 1);
lean_inc(v_support_x27_400_);
lean_dec_ref(v_x_397_);
v_toFun_401_ = lean_ctor_get(v_y_398_, 0);
v_support_x27_402_ = lean_ctor_get(v_y_398_, 1);
v_isSharedCheck_411_ = !lean_is_exclusive(v_y_398_);
if (v_isSharedCheck_411_ == 0)
{
v___x_404_ = v_y_398_;
v_isShared_405_ = v_isSharedCheck_411_;
goto v_resetjp_403_;
}
else
{
lean_inc(v_support_x27_402_);
lean_inc(v_toFun_401_);
lean_dec(v_y_398_);
v___x_404_ = lean_box(0);
v_isShared_405_ = v_isSharedCheck_411_;
goto v_resetjp_403_;
}
v_resetjp_403_:
{
lean_object* v___f_406_; lean_object* v___x_407_; lean_object* v___x_409_; 
v___f_406_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_zipWith___redArg___lam__0), 4, 3);
lean_closure_set(v___f_406_, 0, v_toFun_399_);
lean_closure_set(v___f_406_, 1, v_toFun_401_);
lean_closure_set(v___f_406_, 2, v_f_396_);
v___x_407_ = l_List_appendTR___redArg(v_support_x27_400_, v_support_x27_402_);
if (v_isShared_405_ == 0)
{
lean_ctor_set(v___x_404_, 1, v___x_407_);
lean_ctor_set(v___x_404_, 0, v___f_406_);
v___x_409_ = v___x_404_;
goto v_reusejp_408_;
}
else
{
lean_object* v_reuseFailAlloc_410_; 
v_reuseFailAlloc_410_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_410_, 0, v___f_406_);
lean_ctor_set(v_reuseFailAlloc_410_, 1, v___x_407_);
v___x_409_ = v_reuseFailAlloc_410_;
goto v_reusejp_408_;
}
v_reusejp_408_:
{
return v___x_409_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_zipWith(lean_object* v_00_u03b9_412_, lean_object* v_00_u03b2_413_, lean_object* v_00_u03b2_u2081_414_, lean_object* v_00_u03b2_u2082_415_, lean_object* v_inst_416_, lean_object* v_inst_417_, lean_object* v_inst_418_, lean_object* v_f_419_, lean_object* v_hf_420_, lean_object* v_x_421_, lean_object* v_y_422_){
_start:
{
lean_object* v___x_423_; 
v___x_423_ = lp_mathlib_DFinsupp_zipWith___redArg(v_f_419_, v_x_421_, v_y_422_);
return v___x_423_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_zipWith___boxed(lean_object* v_00_u03b9_424_, lean_object* v_00_u03b2_425_, lean_object* v_00_u03b2_u2081_426_, lean_object* v_00_u03b2_u2082_427_, lean_object* v_inst_428_, lean_object* v_inst_429_, lean_object* v_inst_430_, lean_object* v_f_431_, lean_object* v_hf_432_, lean_object* v_x_433_, lean_object* v_y_434_){
_start:
{
lean_object* v_res_435_; 
v_res_435_ = lp_mathlib_DFinsupp_zipWith(v_00_u03b9_424_, v_00_u03b2_425_, v_00_u03b2_u2081_426_, v_00_u03b2_u2082_427_, v_inst_428_, v_inst_429_, v_inst_430_, v_f_431_, v_hf_432_, v_x_433_, v_y_434_);
lean_dec(v_inst_430_);
lean_dec(v_inst_429_);
lean_dec(v_inst_428_);
return v_res_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_piecewise___redArg___lam__0(lean_object* v_inst_436_, lean_object* v_i_437_, lean_object* v_x_438_, lean_object* v_y_439_){
_start:
{
lean_object* v___x_440_; uint8_t v___x_441_; 
v___x_440_ = lean_apply_1(v_inst_436_, v_i_437_);
v___x_441_ = lean_unbox(v___x_440_);
if (v___x_441_ == 0)
{
lean_inc(v_y_439_);
return v_y_439_;
}
else
{
lean_inc(v_x_438_);
return v_x_438_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_piecewise___redArg___lam__0___boxed(lean_object* v_inst_442_, lean_object* v_i_443_, lean_object* v_x_444_, lean_object* v_y_445_){
_start:
{
lean_object* v_res_446_; 
v_res_446_ = lp_mathlib_DFinsupp_piecewise___redArg___lam__0(v_inst_442_, v_i_443_, v_x_444_, v_y_445_);
lean_dec(v_y_445_);
lean_dec(v_x_444_);
return v_res_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_piecewise___redArg(lean_object* v_x_447_, lean_object* v_y_448_, lean_object* v_inst_449_){
_start:
{
lean_object* v___f_450_; lean_object* v___x_451_; 
v___f_450_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_piecewise___redArg___lam__0___boxed), 4, 1);
lean_closure_set(v___f_450_, 0, v_inst_449_);
v___x_451_ = lp_mathlib_DFinsupp_zipWith___redArg(v___f_450_, v_x_447_, v_y_448_);
return v___x_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_piecewise(lean_object* v_00_u03b9_452_, lean_object* v_00_u03b2_453_, lean_object* v_inst_454_, lean_object* v_x_455_, lean_object* v_y_456_, lean_object* v_s_457_, lean_object* v_inst_458_){
_start:
{
lean_object* v___x_459_; 
v___x_459_ = lp_mathlib_DFinsupp_piecewise___redArg(v_x_455_, v_y_456_, v_inst_458_);
return v___x_459_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_piecewise___boxed(lean_object* v_00_u03b9_460_, lean_object* v_00_u03b2_461_, lean_object* v_inst_462_, lean_object* v_x_463_, lean_object* v_y_464_, lean_object* v_s_465_, lean_object* v_inst_466_){
_start:
{
lean_object* v_res_467_; 
v_res_467_ = lp_mathlib_DFinsupp_piecewise(v_00_u03b9_460_, v_00_u03b2_461_, v_inst_462_, v_x_463_, v_y_464_, v_s_465_, v_inst_466_);
lean_dec(v_inst_462_);
return v_res_467_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAdd___redArg___lam__0(lean_object* v_inst_468_, lean_object* v_i_469_){
_start:
{
lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v_toZero_472_; 
v___x_470_ = lean_apply_1(v_inst_468_, v_i_469_);
v___x_471_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_470_);
v_toZero_472_ = lean_ctor_get(v___x_471_, 0);
lean_inc(v_toZero_472_);
lean_dec_ref(v___x_471_);
return v_toZero_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAdd___redArg___lam__1(lean_object* v_inst_473_, lean_object* v_x_474_, lean_object* v_x1_475_, lean_object* v_x2_476_){
_start:
{
lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v_toAdd_479_; lean_object* v___x_480_; 
v___x_477_ = lean_apply_1(v_inst_473_, v_x_474_);
v___x_478_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_477_);
v_toAdd_479_ = lean_ctor_get(v___x_478_, 1);
lean_inc(v_toAdd_479_);
lean_dec_ref(v___x_478_);
v___x_480_ = lean_apply_2(v_toAdd_479_, v_x1_475_, v_x2_476_);
return v___x_480_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAdd___redArg(lean_object* v_inst_481_){
_start:
{
lean_object* v___f_482_; lean_object* v___f_483_; lean_object* v___x_484_; 
lean_inc_ref(v_inst_481_);
v___f_482_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instAdd___redArg___lam__0), 2, 1);
lean_closure_set(v___f_482_, 0, v_inst_481_);
v___f_483_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instAdd___redArg___lam__1), 4, 1);
lean_closure_set(v___f_483_, 0, v_inst_481_);
lean_inc_ref_n(v___f_482_, 2);
v___x_484_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_zipWith___boxed), 11, 9);
lean_closure_set(v___x_484_, 0, lean_box(0));
lean_closure_set(v___x_484_, 1, lean_box(0));
lean_closure_set(v___x_484_, 2, lean_box(0));
lean_closure_set(v___x_484_, 3, lean_box(0));
lean_closure_set(v___x_484_, 4, v___f_482_);
lean_closure_set(v___x_484_, 5, v___f_482_);
lean_closure_set(v___x_484_, 6, v___f_482_);
lean_closure_set(v___x_484_, 7, v___f_483_);
lean_closure_set(v___x_484_, 8, lean_box(0));
return v___x_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAdd(lean_object* v_00_u03b9_485_, lean_object* v_00_u03b2_486_, lean_object* v_inst_487_){
_start:
{
lean_object* v___x_488_; 
v___x_488_ = lp_mathlib_DFinsupp_instAdd___redArg(v_inst_487_);
return v___x_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_addZeroClass___redArg(lean_object* v_inst_489_){
_start:
{
lean_object* v___f_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v___x_493_; 
lean_inc_ref(v_inst_489_);
v___f_490_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instAdd___redArg___lam__0), 2, 1);
lean_closure_set(v___f_490_, 0, v_inst_489_);
v___x_491_ = lp_mathlib_DFinsupp_instAdd___redArg(v_inst_489_);
v___x_492_ = lp_mathlib_DFinsupp_instZero___redArg(v___f_490_);
v___x_493_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_493_, 0, v___x_492_);
lean_ctor_set(v___x_493_, 1, v___x_491_);
return v___x_493_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_addZeroClass(lean_object* v_00_u03b9_494_, lean_object* v_00_u03b2_495_, lean_object* v_inst_496_){
_start:
{
lean_object* v___x_497_; 
v___x_497_ = lp_mathlib_DFinsupp_addZeroClass___redArg(v_inst_496_);
return v___x_497_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_hasNatScalar___redArg___lam__0(lean_object* v_inst_498_, lean_object* v_c_499_, lean_object* v_x_500_, lean_object* v_x_501_){
_start:
{
lean_object* v___x_502_; lean_object* v_toNSMul_503_; lean_object* v___x_504_; 
v___x_502_ = lean_apply_1(v_inst_498_, v_x_500_);
v_toNSMul_503_ = lean_ctor_get(v___x_502_, 2);
lean_inc(v_toNSMul_503_);
lean_dec_ref(v___x_502_);
v___x_504_ = lean_apply_2(v_toNSMul_503_, v_c_499_, v_x_501_);
return v___x_504_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_hasNatScalar___redArg___lam__1(lean_object* v_inst_505_, lean_object* v_c_506_, lean_object* v_v_507_){
_start:
{
lean_object* v___f_508_; lean_object* v___x_509_; 
v___f_508_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_hasNatScalar___redArg___lam__0), 4, 2);
lean_closure_set(v___f_508_, 0, v_inst_505_);
lean_closure_set(v___f_508_, 1, v_c_506_);
v___x_509_ = lp_mathlib_DFinsupp_mapRange___redArg(v___f_508_, v_v_507_);
return v___x_509_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_hasNatScalar___redArg(lean_object* v_inst_510_){
_start:
{
lean_object* v___f_511_; 
v___f_511_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_hasNatScalar___redArg___lam__1), 3, 1);
lean_closure_set(v___f_511_, 0, v_inst_510_);
return v___f_511_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_hasNatScalar(lean_object* v_00_u03b9_512_, lean_object* v_00_u03b2_513_, lean_object* v_inst_514_){
_start:
{
lean_object* v___f_515_; 
v___f_515_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_hasNatScalar___redArg___lam__1), 3, 1);
lean_closure_set(v___f_515_, 0, v_inst_514_);
return v___f_515_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAddMonoid___redArg___lam__0(lean_object* v_inst_516_, lean_object* v_i_517_){
_start:
{
lean_object* v___x_518_; lean_object* v___x_519_; 
v___x_518_ = lean_apply_1(v_inst_516_, v_i_517_);
v___x_519_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_518_);
lean_dec_ref(v___x_518_);
return v___x_519_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAddMonoid___redArg___lam__1(lean_object* v_inst_520_, lean_object* v_i_521_){
_start:
{
lean_object* v___x_522_; lean_object* v___x_523_; lean_object* v___x_524_; lean_object* v_toZero_525_; 
v___x_522_ = lean_apply_1(v_inst_520_, v_i_521_);
v___x_523_ = lp_mathlib_AddMonoid_toAddZeroClass___redArg(v___x_522_);
lean_dec_ref(v___x_522_);
v___x_524_ = lp_mathlib_AddZeroClass_toAddZero___redArg(v___x_523_);
v_toZero_525_ = lean_ctor_get(v___x_524_, 0);
lean_inc(v_toZero_525_);
lean_dec_ref(v___x_524_);
return v_toZero_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAddMonoid___redArg(lean_object* v_inst_526_){
_start:
{
lean_object* v___f_527_; lean_object* v___f_528_; lean_object* v___f_529_; lean_object* v___x_530_; lean_object* v___x_531_; lean_object* v___x_532_; 
lean_inc_ref_n(v_inst_526_, 2);
v___f_527_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instAddMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_527_, 0, v_inst_526_);
v___f_528_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instAddMonoid___redArg___lam__1), 2, 1);
lean_closure_set(v___f_528_, 0, v_inst_526_);
v___f_529_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_hasNatScalar___redArg___lam__1), 3, 1);
lean_closure_set(v___f_529_, 0, v_inst_526_);
v___x_530_ = lp_mathlib_DFinsupp_instAdd___redArg(v___f_527_);
v___x_531_ = lp_mathlib_DFinsupp_instZero___redArg(v___f_528_);
v___x_532_ = lp_mathlib_Function_Injective_addMonoid___redArg(v___x_530_, v___x_531_, v___f_529_);
return v___x_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAddMonoid(lean_object* v_00_u03b9_533_, lean_object* v_00_u03b2_534_, lean_object* v_inst_535_){
_start:
{
lean_object* v___x_536_; 
v___x_536_ = lp_mathlib_DFinsupp_instAddMonoid___redArg(v_inst_535_);
return v___x_536_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coeFnAddMonoidHom___lam__0(lean_object* v_f_537_, lean_object* v___y_538_){
_start:
{
lean_object* v_toFun_539_; lean_object* v___x_540_; 
v_toFun_539_ = lean_ctor_get(v_f_537_, 0);
lean_inc(v_toFun_539_);
lean_dec_ref(v_f_537_);
v___x_540_ = lean_apply_1(v_toFun_539_, v___y_538_);
return v___x_540_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coeFnAddMonoidHom(lean_object* v_00_u03b9_542_, lean_object* v_00_u03b2_543_, lean_object* v_inst_544_){
_start:
{
lean_object* v___f_545_; 
v___f_545_ = ((lean_object*)(lp_mathlib_DFinsupp_coeFnAddMonoidHom___closed__0));
return v___f_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_coeFnAddMonoidHom___boxed(lean_object* v_00_u03b9_546_, lean_object* v_00_u03b2_547_, lean_object* v_inst_548_){
_start:
{
lean_object* v_res_549_; 
v_res_549_ = lp_mathlib_DFinsupp_coeFnAddMonoidHom(v_00_u03b9_546_, v_00_u03b2_547_, v_inst_548_);
lean_dec_ref(v_inst_548_);
return v_res_549_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_addCommMonoid___redArg___lam__0(lean_object* v_inst_550_, lean_object* v_i_551_){
_start:
{
lean_object* v___x_552_; 
v___x_552_ = lean_apply_1(v_inst_550_, v_i_551_);
return v___x_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_addCommMonoid___redArg(lean_object* v_inst_553_){
_start:
{
lean_object* v___f_554_; lean_object* v___x_555_; 
v___f_554_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_addCommMonoid___redArg___lam__0), 2, 1);
lean_closure_set(v___f_554_, 0, v_inst_553_);
v___x_555_ = lp_mathlib_DFinsupp_instAddMonoid___redArg(v___f_554_);
return v___x_555_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_addCommMonoid(lean_object* v_00_u03b9_556_, lean_object* v_00_u03b2_557_, lean_object* v_inst_558_){
_start:
{
lean_object* v___x_559_; 
v___x_559_ = lp_mathlib_DFinsupp_addCommMonoid___redArg(v_inst_558_);
return v___x_559_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instNeg___redArg___lam__0(lean_object* v_inst_560_, lean_object* v_x_561_, lean_object* v___y_562_){
_start:
{
lean_object* v___x_563_; lean_object* v___x_564_; lean_object* v_toNeg_565_; lean_object* v___x_566_; 
v___x_563_ = lean_apply_1(v_inst_560_, v_x_561_);
v___x_564_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v___x_563_);
lean_dec_ref(v___x_563_);
v_toNeg_565_ = lean_ctor_get(v___x_564_, 1);
lean_inc(v_toNeg_565_);
lean_dec_ref(v___x_564_);
v___x_566_ = lean_apply_1(v_toNeg_565_, v___y_562_);
return v___x_566_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instNeg___redArg___lam__1(lean_object* v___f_567_, lean_object* v_f_568_){
_start:
{
lean_object* v___x_569_; 
v___x_569_ = lp_mathlib_DFinsupp_mapRange___redArg(v___f_567_, v_f_568_);
return v___x_569_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instNeg___redArg(lean_object* v_inst_570_){
_start:
{
lean_object* v___f_571_; lean_object* v___f_572_; 
v___f_571_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instNeg___redArg___lam__0), 3, 1);
lean_closure_set(v___f_571_, 0, v_inst_570_);
v___f_572_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instNeg___redArg___lam__1), 2, 1);
lean_closure_set(v___f_572_, 0, v___f_571_);
return v___f_572_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instNeg(lean_object* v_00_u03b9_573_, lean_object* v_00_u03b2_574_, lean_object* v_inst_575_){
_start:
{
lean_object* v___x_576_; 
v___x_576_ = lp_mathlib_DFinsupp_instNeg___redArg(v_inst_575_);
return v___x_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSub___redArg___lam__0(lean_object* v_inst_577_, lean_object* v_i_578_){
_start:
{
lean_object* v___x_579_; lean_object* v___x_580_; lean_object* v_toZero_581_; 
v___x_579_ = lean_apply_1(v_inst_577_, v_i_578_);
v___x_580_ = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(v___x_579_);
lean_dec_ref(v___x_579_);
v_toZero_581_ = lean_ctor_get(v___x_580_, 0);
lean_inc(v_toZero_581_);
lean_dec_ref(v___x_580_);
return v_toZero_581_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSub___redArg___lam__1(lean_object* v_inst_582_, lean_object* v_x_583_, lean_object* v___y_584_, lean_object* v___y_585_){
_start:
{
lean_object* v___x_586_; lean_object* v_toSub_587_; lean_object* v___x_588_; 
v___x_586_ = lean_apply_1(v_inst_582_, v_x_583_);
v_toSub_587_ = lean_ctor_get(v___x_586_, 2);
lean_inc(v_toSub_587_);
lean_dec_ref(v___x_586_);
v___x_588_ = lean_apply_2(v_toSub_587_, v___y_584_, v___y_585_);
return v___x_588_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSub___redArg(lean_object* v_inst_589_){
_start:
{
lean_object* v___f_590_; lean_object* v___f_591_; lean_object* v___x_592_; 
lean_inc_ref(v_inst_589_);
v___f_590_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instSub___redArg___lam__0), 2, 1);
lean_closure_set(v___f_590_, 0, v_inst_589_);
v___f_591_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instSub___redArg___lam__1), 4, 1);
lean_closure_set(v___f_591_, 0, v_inst_589_);
lean_inc_ref_n(v___f_590_, 2);
v___x_592_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_zipWith___boxed), 11, 9);
lean_closure_set(v___x_592_, 0, lean_box(0));
lean_closure_set(v___x_592_, 1, lean_box(0));
lean_closure_set(v___x_592_, 2, lean_box(0));
lean_closure_set(v___x_592_, 3, lean_box(0));
lean_closure_set(v___x_592_, 4, v___f_590_);
lean_closure_set(v___x_592_, 5, v___f_590_);
lean_closure_set(v___x_592_, 6, v___f_590_);
lean_closure_set(v___x_592_, 7, v___f_591_);
lean_closure_set(v___x_592_, 8, lean_box(0));
return v___x_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instSub(lean_object* v_00_u03b9_593_, lean_object* v_00_u03b2_594_, lean_object* v_inst_595_){
_start:
{
lean_object* v___x_596_; 
v___x_596_ = lp_mathlib_DFinsupp_instSub___redArg(v_inst_595_);
return v___x_596_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_hasIntScalar___redArg___lam__0(lean_object* v_inst_597_, lean_object* v_c_598_, lean_object* v_x_599_, lean_object* v_x_600_){
_start:
{
lean_object* v___x_601_; lean_object* v_toZSMul_602_; lean_object* v___x_603_; 
v___x_601_ = lean_apply_1(v_inst_597_, v_x_599_);
v_toZSMul_602_ = lean_ctor_get(v___x_601_, 3);
lean_inc(v_toZSMul_602_);
lean_dec_ref(v___x_601_);
v___x_603_ = lean_apply_2(v_toZSMul_602_, v_c_598_, v_x_600_);
return v___x_603_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_hasIntScalar___redArg___lam__1(lean_object* v_inst_604_, lean_object* v_c_605_, lean_object* v_v_606_){
_start:
{
lean_object* v___f_607_; lean_object* v___x_608_; 
v___f_607_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_hasIntScalar___redArg___lam__0), 4, 2);
lean_closure_set(v___f_607_, 0, v_inst_604_);
lean_closure_set(v___f_607_, 1, v_c_605_);
v___x_608_ = lp_mathlib_DFinsupp_mapRange___redArg(v___f_607_, v_v_606_);
return v___x_608_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_hasIntScalar___redArg(lean_object* v_inst_609_){
_start:
{
lean_object* v___f_610_; 
v___f_610_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_hasIntScalar___redArg___lam__1), 3, 1);
lean_closure_set(v___f_610_, 0, v_inst_609_);
return v___f_610_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_hasIntScalar(lean_object* v_00_u03b9_611_, lean_object* v_00_u03b2_612_, lean_object* v_inst_613_){
_start:
{
lean_object* v___f_614_; 
v___f_614_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_hasIntScalar___redArg___lam__1), 3, 1);
lean_closure_set(v___f_614_, 0, v_inst_613_);
return v___f_614_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAddGroup___redArg___lam__0(lean_object* v_inst_615_, lean_object* v_i_616_){
_start:
{
lean_object* v___x_617_; lean_object* v_toAddMonoid_618_; 
v___x_617_ = lean_apply_1(v_inst_615_, v_i_616_);
v_toAddMonoid_618_ = lean_ctor_get(v___x_617_, 0);
lean_inc_ref(v_toAddMonoid_618_);
lean_dec_ref(v___x_617_);
return v_toAddMonoid_618_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAddGroup___redArg(lean_object* v_inst_619_){
_start:
{
lean_object* v___f_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___f_623_; lean_object* v___f_624_; lean_object* v___x_625_; lean_object* v___x_626_; 
lean_inc_ref_n(v_inst_619_, 3);
v___f_620_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instAddGroup___redArg___lam__0), 2, 1);
lean_closure_set(v___f_620_, 0, v_inst_619_);
v___x_621_ = lp_mathlib_DFinsupp_instNeg___redArg(v_inst_619_);
v___x_622_ = lp_mathlib_DFinsupp_instSub___redArg(v_inst_619_);
v___f_623_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_hasIntScalar___redArg___lam__1), 3, 1);
lean_closure_set(v___f_623_, 0, v_inst_619_);
v___f_624_ = lean_alloc_closure((void*)(lp_mathlib_ZSMul_toSMul___redArg___lam__0), 3, 1);
lean_closure_set(v___f_624_, 0, v___f_623_);
v___x_625_ = lp_mathlib_DFinsupp_instAddMonoid___redArg(v___f_620_);
v___x_626_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_626_, 0, v___x_625_);
lean_ctor_set(v___x_626_, 1, v___x_621_);
lean_ctor_set(v___x_626_, 2, v___x_622_);
lean_ctor_set(v___x_626_, 3, v___f_624_);
return v___x_626_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instAddGroup(lean_object* v_00_u03b9_627_, lean_object* v_00_u03b2_628_, lean_object* v_inst_629_){
_start:
{
lean_object* v___x_630_; 
v___x_630_ = lp_mathlib_DFinsupp_instAddGroup___redArg(v_inst_629_);
return v___x_630_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_addCommGroup___redArg___lam__0(lean_object* v_inst_631_, lean_object* v_i_632_){
_start:
{
lean_object* v___x_633_; 
v___x_633_ = lean_apply_1(v_inst_631_, v_i_632_);
return v___x_633_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_addCommGroup___redArg(lean_object* v_inst_634_){
_start:
{
lean_object* v___f_635_; lean_object* v___x_636_; 
v___f_635_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_addCommGroup___redArg___lam__0), 2, 1);
lean_closure_set(v___f_635_, 0, v_inst_634_);
v___x_636_ = lp_mathlib_DFinsupp_instAddGroup___redArg(v___f_635_);
return v___x_636_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_addCommGroup(lean_object* v_00_u03b9_637_, lean_object* v_00_u03b2_638_, lean_object* v_inst_639_){
_start:
{
lean_object* v___x_640_; 
v___x_640_ = lp_mathlib_DFinsupp_addCommGroup___redArg(v_inst_639_);
return v___x_640_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filter___redArg___lam__0(lean_object* v_inst_641_, lean_object* v_inst_642_, lean_object* v_toFun_643_, lean_object* v_i_644_){
_start:
{
lean_object* v___x_645_; uint8_t v___x_646_; 
lean_inc(v_i_644_);
v___x_645_ = lean_apply_1(v_inst_641_, v_i_644_);
v___x_646_ = lean_unbox(v___x_645_);
if (v___x_646_ == 0)
{
lean_object* v___x_647_; 
lean_dec(v_toFun_643_);
v___x_647_ = lean_apply_1(v_inst_642_, v_i_644_);
return v___x_647_;
}
else
{
lean_object* v___x_648_; 
lean_dec(v_inst_642_);
v___x_648_ = lean_apply_1(v_toFun_643_, v_i_644_);
return v___x_648_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filter___redArg(lean_object* v_inst_649_, lean_object* v_inst_650_, lean_object* v_x_651_){
_start:
{
lean_object* v_toFun_652_; lean_object* v_support_x27_653_; lean_object* v___x_655_; uint8_t v_isShared_656_; uint8_t v_isSharedCheck_661_; 
v_toFun_652_ = lean_ctor_get(v_x_651_, 0);
v_support_x27_653_ = lean_ctor_get(v_x_651_, 1);
v_isSharedCheck_661_ = !lean_is_exclusive(v_x_651_);
if (v_isSharedCheck_661_ == 0)
{
v___x_655_ = v_x_651_;
v_isShared_656_ = v_isSharedCheck_661_;
goto v_resetjp_654_;
}
else
{
lean_inc(v_support_x27_653_);
lean_inc(v_toFun_652_);
lean_dec(v_x_651_);
v___x_655_ = lean_box(0);
v_isShared_656_ = v_isSharedCheck_661_;
goto v_resetjp_654_;
}
v_resetjp_654_:
{
lean_object* v___f_657_; lean_object* v___x_659_; 
v___f_657_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_filter___redArg___lam__0), 4, 3);
lean_closure_set(v___f_657_, 0, v_inst_650_);
lean_closure_set(v___f_657_, 1, v_inst_649_);
lean_closure_set(v___f_657_, 2, v_toFun_652_);
if (v_isShared_656_ == 0)
{
lean_ctor_set(v___x_655_, 0, v___f_657_);
v___x_659_ = v___x_655_;
goto v_reusejp_658_;
}
else
{
lean_object* v_reuseFailAlloc_660_; 
v_reuseFailAlloc_660_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_660_, 0, v___f_657_);
lean_ctor_set(v_reuseFailAlloc_660_, 1, v_support_x27_653_);
v___x_659_ = v_reuseFailAlloc_660_;
goto v_reusejp_658_;
}
v_reusejp_658_:
{
return v___x_659_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filter(lean_object* v_00_u03b9_662_, lean_object* v_00_u03b2_663_, lean_object* v_inst_664_, lean_object* v_p_665_, lean_object* v_inst_666_, lean_object* v_x_667_){
_start:
{
lean_object* v___x_668_; 
v___x_668_ = lp_mathlib_DFinsupp_filter___redArg(v_inst_664_, v_inst_666_, v_x_667_);
return v___x_668_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filterAddMonoidHom___redArg(lean_object* v_inst_669_, lean_object* v_inst_670_){
_start:
{
lean_object* v___f_671_; lean_object* v___x_672_; 
v___f_671_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instAdd___redArg___lam__0), 2, 1);
lean_closure_set(v___f_671_, 0, v_inst_669_);
v___x_672_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_filter), 6, 5);
lean_closure_set(v___x_672_, 0, lean_box(0));
lean_closure_set(v___x_672_, 1, lean_box(0));
lean_closure_set(v___x_672_, 2, v___f_671_);
lean_closure_set(v___x_672_, 3, lean_box(0));
lean_closure_set(v___x_672_, 4, v_inst_670_);
return v___x_672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_filterAddMonoidHom(lean_object* v_00_u03b9_673_, lean_object* v_00_u03b2_674_, lean_object* v_inst_675_, lean_object* v_p_676_, lean_object* v_inst_677_){
_start:
{
lean_object* v___x_678_; 
v___x_678_ = lp_mathlib_DFinsupp_filterAddMonoidHom___redArg(v_inst_675_, v_inst_677_);
return v___x_678_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomain___redArg___lam__0(lean_object* v_toFun_679_, lean_object* v_i_680_){
_start:
{
lean_object* v___x_681_; 
v___x_681_ = lean_apply_1(v_toFun_679_, v_i_680_);
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomain___redArg___lam__1(lean_object* v_j_682_){
_start:
{
lean_inc(v_j_682_);
return v_j_682_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomain___redArg___lam__1___boxed(lean_object* v_j_683_){
_start:
{
lean_object* v_res_684_; 
v_res_684_ = lp_mathlib_DFinsupp_subtypeDomain___redArg___lam__1(v_j_683_);
lean_dec(v_j_683_);
return v_res_684_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomain___redArg(lean_object* v_inst_686_, lean_object* v_x_687_){
_start:
{
lean_object* v_toFun_688_; lean_object* v_support_x27_689_; lean_object* v___x_691_; uint8_t v_isShared_692_; uint8_t v_isSharedCheck_701_; 
v_toFun_688_ = lean_ctor_get(v_x_687_, 0);
v_support_x27_689_ = lean_ctor_get(v_x_687_, 1);
v_isSharedCheck_701_ = !lean_is_exclusive(v_x_687_);
if (v_isSharedCheck_701_ == 0)
{
v___x_691_ = v_x_687_;
v_isShared_692_ = v_isSharedCheck_701_;
goto v_resetjp_690_;
}
else
{
lean_inc(v_support_x27_689_);
lean_inc(v_toFun_688_);
lean_dec(v_x_687_);
v___x_691_ = lean_box(0);
v_isShared_692_ = v_isSharedCheck_701_;
goto v_resetjp_690_;
}
v_resetjp_690_:
{
lean_object* v___f_693_; lean_object* v___f_694_; lean_object* v___x_695_; lean_object* v___x_696_; lean_object* v___x_697_; lean_object* v___x_699_; 
v___f_693_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_subtypeDomain___redArg___lam__0), 2, 1);
lean_closure_set(v___f_693_, 0, v_toFun_688_);
v___f_694_ = ((lean_object*)(lp_mathlib_DFinsupp_subtypeDomain___redArg___closed__0));
v___x_695_ = lp_mathlib_Multiset_filter___redArg(v_inst_686_, v_support_x27_689_);
v___x_696_ = lp_mathlib_Multiset_attach___redArg(v___x_695_);
v___x_697_ = lp_mathlib_Multiset_map___redArg(v___f_694_, v___x_696_);
if (v_isShared_692_ == 0)
{
lean_ctor_set(v___x_691_, 1, v___x_697_);
lean_ctor_set(v___x_691_, 0, v___f_693_);
v___x_699_ = v___x_691_;
goto v_reusejp_698_;
}
else
{
lean_object* v_reuseFailAlloc_700_; 
v_reuseFailAlloc_700_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_700_, 0, v___f_693_);
lean_ctor_set(v_reuseFailAlloc_700_, 1, v___x_697_);
v___x_699_ = v_reuseFailAlloc_700_;
goto v_reusejp_698_;
}
v_reusejp_698_:
{
return v___x_699_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomain(lean_object* v_00_u03b9_702_, lean_object* v_00_u03b2_703_, lean_object* v_inst_704_, lean_object* v_p_705_, lean_object* v_inst_706_, lean_object* v_x_707_){
_start:
{
lean_object* v___x_708_; 
v___x_708_ = lp_mathlib_DFinsupp_subtypeDomain___redArg(v_inst_706_, v_x_707_);
return v___x_708_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomain___boxed(lean_object* v_00_u03b9_709_, lean_object* v_00_u03b2_710_, lean_object* v_inst_711_, lean_object* v_p_712_, lean_object* v_inst_713_, lean_object* v_x_714_){
_start:
{
lean_object* v_res_715_; 
v_res_715_ = lp_mathlib_DFinsupp_subtypeDomain(v_00_u03b9_709_, v_00_u03b2_710_, v_inst_711_, v_p_712_, v_inst_713_, v_x_714_);
lean_dec(v_inst_711_);
return v_res_715_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomainAddMonoidHom___redArg(lean_object* v_inst_716_, lean_object* v_inst_717_){
_start:
{
lean_object* v___f_718_; lean_object* v___x_719_; 
v___f_718_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instAdd___redArg___lam__0), 2, 1);
lean_closure_set(v___f_718_, 0, v_inst_716_);
v___x_719_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_subtypeDomain___boxed), 6, 5);
lean_closure_set(v___x_719_, 0, lean_box(0));
lean_closure_set(v___x_719_, 1, lean_box(0));
lean_closure_set(v___x_719_, 2, v___f_718_);
lean_closure_set(v___x_719_, 3, lean_box(0));
lean_closure_set(v___x_719_, 4, v_inst_717_);
return v___x_719_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeDomainAddMonoidHom(lean_object* v_00_u03b9_720_, lean_object* v_00_u03b2_721_, lean_object* v_inst_722_, lean_object* v_p_723_, lean_object* v_inst_724_){
_start:
{
lean_object* v___x_725_; 
v___x_725_ = lp_mathlib_DFinsupp_subtypeDomainAddMonoidHom___redArg(v_inst_722_, v_inst_724_);
return v___x_725_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mk___redArg___lam__0(lean_object* v_inst_726_, lean_object* v_s_727_, lean_object* v_inst_728_, lean_object* v_x_729_, lean_object* v_i_730_){
_start:
{
uint8_t v___x_731_; 
lean_inc(v_i_730_);
v___x_731_ = lp_mathlib_Multiset_decidableMem___aux__1___redArg(v_inst_726_, v_i_730_, v_s_727_);
if (v___x_731_ == 0)
{
lean_object* v___x_732_; 
lean_dec(v_x_729_);
v___x_732_ = lean_apply_1(v_inst_728_, v_i_730_);
return v___x_732_;
}
else
{
lean_object* v___x_733_; 
lean_dec(v_inst_728_);
v___x_733_ = lean_apply_1(v_x_729_, v_i_730_);
return v___x_733_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mk___redArg(lean_object* v_inst_734_, lean_object* v_inst_735_, lean_object* v_s_736_, lean_object* v_x_737_){
_start:
{
lean_object* v___f_738_; lean_object* v___x_739_; 
lean_inc(v_s_736_);
v___f_738_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_mk___redArg___lam__0), 5, 4);
lean_closure_set(v___f_738_, 0, v_inst_735_);
lean_closure_set(v___f_738_, 1, v_s_736_);
lean_closure_set(v___f_738_, 2, v_inst_734_);
lean_closure_set(v___f_738_, 3, v_x_737_);
v___x_739_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_739_, 0, v___f_738_);
lean_ctor_set(v___x_739_, 1, v_s_736_);
return v___x_739_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mk(lean_object* v_00_u03b9_740_, lean_object* v_00_u03b2_741_, lean_object* v_inst_742_, lean_object* v_inst_743_, lean_object* v_s_744_, lean_object* v_x_745_){
_start:
{
lean_object* v___x_746_; 
v___x_746_ = lp_mathlib_DFinsupp_mk___redArg(v_inst_742_, v_inst_743_, v_s_744_, v_x_745_);
return v___x_746_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_unique___redArg(lean_object* v_inst_747_){
_start:
{
lean_object* v___x_748_; 
v___x_748_ = lp_mathlib_DFinsupp_instInhabited___redArg(v_inst_747_);
return v___x_748_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_unique(lean_object* v_00_u03b9_749_, lean_object* v_00_u03b2_750_, lean_object* v_inst_751_, lean_object* v_inst_752_){
_start:
{
lean_object* v___x_753_; 
v___x_753_ = lp_mathlib_DFinsupp_instInhabited___redArg(v_inst_751_);
return v___x_753_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_uniqueOfIsEmpty___redArg(lean_object* v_inst_754_){
_start:
{
lean_object* v___x_755_; 
v___x_755_ = lp_mathlib_DFinsupp_instInhabited___redArg(v_inst_754_);
return v___x_755_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_uniqueOfIsEmpty(lean_object* v_00_u03b9_756_, lean_object* v_00_u03b2_757_, lean_object* v_inst_758_, lean_object* v_inst_759_){
_start:
{
lean_object* v___x_760_; 
v___x_760_ = lp_mathlib_DFinsupp_instInhabited___redArg(v_inst_758_);
return v___x_760_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivFunOnFintype___redArg___lam__1(lean_object* v_inst_761_, lean_object* v_f_762_){
_start:
{
lean_object* v___x_763_; 
v___x_763_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_763_, 0, v_f_762_);
lean_ctor_set(v___x_763_, 1, v_inst_761_);
return v___x_763_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivFunOnFintype___redArg(lean_object* v_inst_764_){
_start:
{
lean_object* v___f_765_; lean_object* v___f_766_; lean_object* v___x_767_; 
v___f_765_ = ((lean_object*)(lp_mathlib_DFinsupp_coeFnAddMonoidHom___closed__0));
v___f_766_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_equivFunOnFintype___redArg___lam__1), 2, 1);
lean_closure_set(v___f_766_, 0, v_inst_764_);
v___x_767_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_767_, 0, v___f_765_);
lean_ctor_set(v___x_767_, 1, v___f_766_);
return v___x_767_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivFunOnFintype(lean_object* v_00_u03b9_768_, lean_object* v_00_u03b2_769_, lean_object* v_inst_770_, lean_object* v_inst_771_){
_start:
{
lean_object* v___x_772_; 
v___x_772_ = lp_mathlib_DFinsupp_equivFunOnFintype___redArg(v_inst_771_);
return v___x_772_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivFunOnFintype___boxed(lean_object* v_00_u03b9_773_, lean_object* v_00_u03b2_774_, lean_object* v_inst_775_, lean_object* v_inst_776_){
_start:
{
lean_object* v_res_777_; 
v_res_777_ = lp_mathlib_DFinsupp_equivFunOnFintype(v_00_u03b9_773_, v_00_u03b2_774_, v_inst_775_, v_inst_776_);
lean_dec(v_inst_775_);
return v_res_777_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_single___redArg(lean_object* v_inst_778_, lean_object* v_inst_779_, lean_object* v_i_780_, lean_object* v_b_781_){
_start:
{
lean_object* v___x_782_; lean_object* v___x_783_; lean_object* v___x_784_; lean_object* v___x_785_; 
lean_inc(v_i_780_);
v___x_782_ = lean_alloc_closure((void*)(lp_mathlib_Pi_single___boxed), 7, 6);
lean_closure_set(v___x_782_, 0, lean_box(0));
lean_closure_set(v___x_782_, 1, lean_box(0));
lean_closure_set(v___x_782_, 2, v_inst_778_);
lean_closure_set(v___x_782_, 3, v_inst_779_);
lean_closure_set(v___x_782_, 4, v_i_780_);
lean_closure_set(v___x_782_, 5, v_b_781_);
v___x_783_ = lean_box(0);
v___x_784_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_784_, 0, v_i_780_);
lean_ctor_set(v___x_784_, 1, v___x_783_);
v___x_785_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_785_, 0, v___x_782_);
lean_ctor_set(v___x_785_, 1, v___x_784_);
return v___x_785_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_single(lean_object* v_00_u03b9_786_, lean_object* v_00_u03b2_787_, lean_object* v_inst_788_, lean_object* v_inst_789_, lean_object* v_i_790_, lean_object* v_b_791_){
_start:
{
lean_object* v___x_792_; 
v___x_792_ = lp_mathlib_DFinsupp_single___redArg(v_inst_788_, v_inst_789_, v_i_790_, v_b_791_);
return v___x_792_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_erase___redArg___lam__0(lean_object* v_inst_793_, lean_object* v_i_794_, lean_object* v_toFun_795_, lean_object* v_inst_796_, lean_object* v_j_797_){
_start:
{
lean_object* v___x_798_; uint8_t v___x_799_; 
lean_inc(v_j_797_);
v___x_798_ = lean_apply_2(v_inst_793_, v_j_797_, v_i_794_);
v___x_799_ = lean_unbox(v___x_798_);
if (v___x_799_ == 0)
{
lean_object* v___x_800_; 
lean_dec(v_inst_796_);
v___x_800_ = lean_apply_1(v_toFun_795_, v_j_797_);
return v___x_800_;
}
else
{
lean_object* v___x_801_; 
lean_dec(v_toFun_795_);
v___x_801_ = lean_apply_1(v_inst_796_, v_j_797_);
return v___x_801_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_erase___redArg(lean_object* v_inst_802_, lean_object* v_inst_803_, lean_object* v_i_804_, lean_object* v_x_805_){
_start:
{
lean_object* v_toFun_806_; lean_object* v_support_x27_807_; lean_object* v___x_809_; uint8_t v_isShared_810_; uint8_t v_isSharedCheck_815_; 
v_toFun_806_ = lean_ctor_get(v_x_805_, 0);
v_support_x27_807_ = lean_ctor_get(v_x_805_, 1);
v_isSharedCheck_815_ = !lean_is_exclusive(v_x_805_);
if (v_isSharedCheck_815_ == 0)
{
v___x_809_ = v_x_805_;
v_isShared_810_ = v_isSharedCheck_815_;
goto v_resetjp_808_;
}
else
{
lean_inc(v_support_x27_807_);
lean_inc(v_toFun_806_);
lean_dec(v_x_805_);
v___x_809_ = lean_box(0);
v_isShared_810_ = v_isSharedCheck_815_;
goto v_resetjp_808_;
}
v_resetjp_808_:
{
lean_object* v___f_811_; lean_object* v___x_813_; 
v___f_811_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_erase___redArg___lam__0), 5, 4);
lean_closure_set(v___f_811_, 0, v_inst_803_);
lean_closure_set(v___f_811_, 1, v_i_804_);
lean_closure_set(v___f_811_, 2, v_toFun_806_);
lean_closure_set(v___f_811_, 3, v_inst_802_);
if (v_isShared_810_ == 0)
{
lean_ctor_set(v___x_809_, 0, v___f_811_);
v___x_813_ = v___x_809_;
goto v_reusejp_812_;
}
else
{
lean_object* v_reuseFailAlloc_814_; 
v_reuseFailAlloc_814_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_814_, 0, v___f_811_);
lean_ctor_set(v_reuseFailAlloc_814_, 1, v_support_x27_807_);
v___x_813_ = v_reuseFailAlloc_814_;
goto v_reusejp_812_;
}
v_reusejp_812_:
{
return v___x_813_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_erase(lean_object* v_00_u03b9_816_, lean_object* v_00_u03b2_817_, lean_object* v_inst_818_, lean_object* v_inst_819_, lean_object* v_i_820_, lean_object* v_x_821_){
_start:
{
lean_object* v___x_822_; 
v___x_822_ = lp_mathlib_DFinsupp_erase___redArg(v_inst_818_, v_inst_819_, v_i_820_, v_x_821_);
return v___x_822_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_update___redArg___lam__0(lean_object* v_toFun_823_, lean_object* v___y_824_){
_start:
{
lean_object* v___x_825_; 
v___x_825_ = lean_apply_1(v_toFun_823_, v___y_824_);
return v___x_825_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_update___redArg(lean_object* v_inst_826_, lean_object* v_f_827_, lean_object* v_i_828_, lean_object* v_b_829_){
_start:
{
lean_object* v_toFun_830_; lean_object* v_support_x27_831_; lean_object* v___x_833_; uint8_t v_isShared_834_; uint8_t v_isSharedCheck_841_; 
v_toFun_830_ = lean_ctor_get(v_f_827_, 0);
v_support_x27_831_ = lean_ctor_get(v_f_827_, 1);
v_isSharedCheck_841_ = !lean_is_exclusive(v_f_827_);
if (v_isSharedCheck_841_ == 0)
{
v___x_833_ = v_f_827_;
v_isShared_834_ = v_isSharedCheck_841_;
goto v_resetjp_832_;
}
else
{
lean_inc(v_support_x27_831_);
lean_inc(v_toFun_830_);
lean_dec(v_f_827_);
v___x_833_ = lean_box(0);
v_isShared_834_ = v_isSharedCheck_841_;
goto v_resetjp_832_;
}
v_resetjp_832_:
{
lean_object* v___f_835_; lean_object* v___x_836_; lean_object* v___x_837_; lean_object* v___x_839_; 
v___f_835_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_update___redArg___lam__0), 2, 1);
lean_closure_set(v___f_835_, 0, v_toFun_830_);
lean_inc(v_i_828_);
v___x_836_ = lean_alloc_closure((void*)(lp_mathlib_Function_update___boxed), 7, 6);
lean_closure_set(v___x_836_, 0, lean_box(0));
lean_closure_set(v___x_836_, 1, lean_box(0));
lean_closure_set(v___x_836_, 2, v_inst_826_);
lean_closure_set(v___x_836_, 3, v___f_835_);
lean_closure_set(v___x_836_, 4, v_i_828_);
lean_closure_set(v___x_836_, 5, v_b_829_);
v___x_837_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_837_, 0, v_i_828_);
lean_ctor_set(v___x_837_, 1, v_support_x27_831_);
if (v_isShared_834_ == 0)
{
lean_ctor_set(v___x_833_, 1, v___x_837_);
lean_ctor_set(v___x_833_, 0, v___x_836_);
v___x_839_ = v___x_833_;
goto v_reusejp_838_;
}
else
{
lean_object* v_reuseFailAlloc_840_; 
v_reuseFailAlloc_840_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_840_, 0, v___x_836_);
lean_ctor_set(v_reuseFailAlloc_840_, 1, v___x_837_);
v___x_839_ = v_reuseFailAlloc_840_;
goto v_reusejp_838_;
}
v_reusejp_838_:
{
return v___x_839_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_update(lean_object* v_00_u03b9_842_, lean_object* v_00_u03b2_843_, lean_object* v_inst_844_, lean_object* v_inst_845_, lean_object* v_f_846_, lean_object* v_i_847_, lean_object* v_b_848_){
_start:
{
lean_object* v___x_849_; 
v___x_849_ = lp_mathlib_DFinsupp_update___redArg(v_inst_845_, v_f_846_, v_i_847_, v_b_848_);
return v___x_849_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_update___boxed(lean_object* v_00_u03b9_850_, lean_object* v_00_u03b2_851_, lean_object* v_inst_852_, lean_object* v_inst_853_, lean_object* v_f_854_, lean_object* v_i_855_, lean_object* v_b_856_){
_start:
{
lean_object* v_res_857_; 
v_res_857_ = lp_mathlib_DFinsupp_update(v_00_u03b9_850_, v_00_u03b2_851_, v_inst_852_, v_inst_853_, v_f_854_, v_i_855_, v_b_856_);
lean_dec(v_inst_852_);
return v_res_857_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_singleAddHom___redArg(lean_object* v_inst_858_, lean_object* v_inst_859_, lean_object* v_i_860_){
_start:
{
lean_object* v___f_861_; lean_object* v___x_862_; 
v___f_861_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instAdd___redArg___lam__0), 2, 1);
lean_closure_set(v___f_861_, 0, v_inst_859_);
v___x_862_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_single), 6, 5);
lean_closure_set(v___x_862_, 0, lean_box(0));
lean_closure_set(v___x_862_, 1, lean_box(0));
lean_closure_set(v___x_862_, 2, v___f_861_);
lean_closure_set(v___x_862_, 3, v_inst_858_);
lean_closure_set(v___x_862_, 4, v_i_860_);
return v___x_862_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_singleAddHom(lean_object* v_00_u03b9_863_, lean_object* v_00_u03b2_864_, lean_object* v_inst_865_, lean_object* v_inst_866_, lean_object* v_i_867_){
_start:
{
lean_object* v___x_868_; 
v___x_868_ = lp_mathlib_DFinsupp_singleAddHom___redArg(v_inst_865_, v_inst_866_, v_i_867_);
return v___x_868_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_eraseAddHom___redArg(lean_object* v_inst_869_, lean_object* v_inst_870_, lean_object* v_i_871_){
_start:
{
lean_object* v___f_872_; lean_object* v___x_873_; 
v___f_872_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instAdd___redArg___lam__0), 2, 1);
lean_closure_set(v___f_872_, 0, v_inst_870_);
v___x_873_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_erase), 6, 5);
lean_closure_set(v___x_873_, 0, lean_box(0));
lean_closure_set(v___x_873_, 1, lean_box(0));
lean_closure_set(v___x_873_, 2, v___f_872_);
lean_closure_set(v___x_873_, 3, v_inst_869_);
lean_closure_set(v___x_873_, 4, v_i_871_);
return v___x_873_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_eraseAddHom(lean_object* v_00_u03b9_874_, lean_object* v_00_u03b2_875_, lean_object* v_inst_876_, lean_object* v_inst_877_, lean_object* v_i_878_){
_start:
{
lean_object* v___x_879_; 
v___x_879_ = lp_mathlib_DFinsupp_eraseAddHom___redArg(v_inst_876_, v_inst_877_, v_i_878_);
return v___x_879_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mkAddGroupHom___redArg(lean_object* v_inst_880_, lean_object* v_inst_881_, lean_object* v_s_882_){
_start:
{
lean_object* v___f_883_; lean_object* v___x_884_; 
v___f_883_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instSub___redArg___lam__0), 2, 1);
lean_closure_set(v___f_883_, 0, v_inst_881_);
v___x_884_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_mk), 6, 5);
lean_closure_set(v___x_884_, 0, lean_box(0));
lean_closure_set(v___x_884_, 1, lean_box(0));
lean_closure_set(v___x_884_, 2, v___f_883_);
lean_closure_set(v___x_884_, 3, v_inst_880_);
lean_closure_set(v___x_884_, 4, v_s_882_);
return v___x_884_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mkAddGroupHom(lean_object* v_00_u03b9_885_, lean_object* v_00_u03b2_886_, lean_object* v_inst_887_, lean_object* v_inst_888_, lean_object* v_s_889_){
_start:
{
lean_object* v___x_890_; 
v___x_890_ = lp_mathlib_DFinsupp_mkAddGroupHom___redArg(v_inst_887_, v_inst_888_, v_s_889_);
return v___x_890_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_support___redArg___lam__0(lean_object* v_toFun_891_, lean_object* v_inst_892_, lean_object* v_a_893_){
_start:
{
lean_object* v___x_894_; lean_object* v___x_895_; uint8_t v___x_896_; 
lean_inc(v_a_893_);
v___x_894_ = lean_apply_1(v_toFun_891_, v_a_893_);
v___x_895_ = lean_apply_2(v_inst_892_, v_a_893_, v___x_894_);
v___x_896_ = lean_unbox(v___x_895_);
return v___x_896_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_support___redArg___lam__0___boxed(lean_object* v_toFun_897_, lean_object* v_inst_898_, lean_object* v_a_899_){
_start:
{
uint8_t v_res_900_; lean_object* v_r_901_; 
v_res_900_ = lp_mathlib_DFinsupp_support___redArg___lam__0(v_toFun_897_, v_inst_898_, v_a_899_);
v_r_901_ = lean_box(v_res_900_);
return v_r_901_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_support___redArg(lean_object* v_inst_902_, lean_object* v_inst_903_, lean_object* v_f_904_){
_start:
{
lean_object* v_toFun_905_; lean_object* v_support_x27_906_; lean_object* v___f_907_; lean_object* v___x_908_; lean_object* v___x_909_; 
v_toFun_905_ = lean_ctor_get(v_f_904_, 0);
lean_inc(v_toFun_905_);
v_support_x27_906_ = lean_ctor_get(v_f_904_, 1);
lean_inc(v_support_x27_906_);
lean_dec_ref(v_f_904_);
v___f_907_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_support___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_907_, 0, v_toFun_905_);
lean_closure_set(v___f_907_, 1, v_inst_903_);
v___x_908_ = lp_mathlib_List_dedup___redArg(v_inst_902_, v_support_x27_906_);
v___x_909_ = lp_mathlib_Multiset_filter___redArg(v___f_907_, v___x_908_);
return v___x_909_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_support(lean_object* v_00_u03b9_910_, lean_object* v_00_u03b2_911_, lean_object* v_inst_912_, lean_object* v_inst_913_, lean_object* v_inst_914_, lean_object* v_f_915_){
_start:
{
lean_object* v___x_916_; 
v___x_916_ = lp_mathlib_DFinsupp_support___redArg(v_inst_912_, v_inst_914_, v_f_915_);
return v___x_916_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_support___boxed(lean_object* v_00_u03b9_917_, lean_object* v_00_u03b2_918_, lean_object* v_inst_919_, lean_object* v_inst_920_, lean_object* v_inst_921_, lean_object* v_f_922_){
_start:
{
lean_object* v_res_923_; 
v_res_923_ = lp_mathlib_DFinsupp_support(v_00_u03b9_917_, v_00_u03b2_918_, v_inst_919_, v_inst_920_, v_inst_921_, v_f_922_);
lean_dec(v_inst_920_);
return v_res_923_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeSupportEqEquiv___redArg___lam__0(lean_object* v_x_924_, lean_object* v___y_925_){
_start:
{
lean_object* v_toFun_926_; lean_object* v___x_927_; 
v_toFun_926_ = lean_ctor_get(v_x_924_, 0);
lean_inc(v_toFun_926_);
lean_dec_ref(v_x_924_);
v___x_927_ = lean_apply_1(v_toFun_926_, v___y_925_);
return v___x_927_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeSupportEqEquiv___redArg___lam__1(lean_object* v_f_928_, lean_object* v_i_929_){
_start:
{
lean_object* v___x_930_; 
v___x_930_ = lean_apply_1(v_f_928_, v_i_929_);
return v___x_930_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeSupportEqEquiv___redArg___lam__2(lean_object* v_inst_931_, lean_object* v_inst_932_, lean_object* v_s_933_, lean_object* v_f_934_){
_start:
{
lean_object* v___f_935_; lean_object* v___x_936_; 
v___f_935_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_subtypeSupportEqEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_935_, 0, v_f_934_);
v___x_936_ = lp_mathlib_DFinsupp_mk___redArg(v_inst_931_, v_inst_932_, v_s_933_, v___f_935_);
return v___x_936_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeSupportEqEquiv___redArg(lean_object* v_inst_938_, lean_object* v_inst_939_, lean_object* v_s_940_){
_start:
{
lean_object* v___f_941_; lean_object* v___f_942_; lean_object* v___x_943_; 
v___f_941_ = ((lean_object*)(lp_mathlib_DFinsupp_subtypeSupportEqEquiv___redArg___closed__0));
v___f_942_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_subtypeSupportEqEquiv___redArg___lam__2), 4, 3);
lean_closure_set(v___f_942_, 0, v_inst_939_);
lean_closure_set(v___f_942_, 1, v_inst_938_);
lean_closure_set(v___f_942_, 2, v_s_940_);
v___x_943_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_943_, 0, v___f_941_);
lean_ctor_set(v___x_943_, 1, v___f_942_);
return v___x_943_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeSupportEqEquiv(lean_object* v_00_u03b9_944_, lean_object* v_00_u03b2_945_, lean_object* v_inst_946_, lean_object* v_inst_947_, lean_object* v_inst_948_, lean_object* v_s_949_){
_start:
{
lean_object* v___x_950_; 
v___x_950_ = lp_mathlib_DFinsupp_subtypeSupportEqEquiv___redArg(v_inst_946_, v_inst_947_, v_s_949_);
return v___x_950_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_subtypeSupportEqEquiv___boxed(lean_object* v_00_u03b9_951_, lean_object* v_00_u03b2_952_, lean_object* v_inst_953_, lean_object* v_inst_954_, lean_object* v_inst_955_, lean_object* v_s_956_){
_start:
{
lean_object* v_res_957_; 
v_res_957_ = lp_mathlib_DFinsupp_subtypeSupportEqEquiv(v_00_u03b9_951_, v_00_u03b2_952_, v_inst_953_, v_inst_954_, v_inst_955_, v_s_956_);
lean_dec_ref(v_inst_955_);
return v_res_957_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaFinsetFunEquiv___redArg(lean_object* v_inst_958_, lean_object* v_inst_959_, lean_object* v_inst_960_){
_start:
{
lean_object* v___x_961_; lean_object* v___x_962_; lean_object* v___x_963_; lean_object* v___x_964_; lean_object* v___x_965_; lean_object* v___x_966_; 
lean_inc_ref(v_inst_960_);
lean_inc(v_inst_959_);
lean_inc_ref(v_inst_958_);
v___x_961_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_support___boxed), 6, 5);
lean_closure_set(v___x_961_, 0, lean_box(0));
lean_closure_set(v___x_961_, 1, lean_box(0));
lean_closure_set(v___x_961_, 2, v_inst_958_);
lean_closure_set(v___x_961_, 3, v_inst_959_);
lean_closure_set(v___x_961_, 4, v_inst_960_);
v___x_962_ = lp_mathlib_Equiv_sigmaFiberEquiv___redArg(v___x_961_);
v___x_963_ = lp_mathlib_Equiv_symm___redArg(v___x_962_);
v___x_964_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_subtypeSupportEqEquiv___boxed), 6, 5);
lean_closure_set(v___x_964_, 0, lean_box(0));
lean_closure_set(v___x_964_, 1, lean_box(0));
lean_closure_set(v___x_964_, 2, v_inst_958_);
lean_closure_set(v___x_964_, 3, v_inst_959_);
lean_closure_set(v___x_964_, 4, v_inst_960_);
v___x_965_ = lp_mathlib_Equiv_sigmaCongrRight___redArg(v___x_964_);
v___x_966_ = lp_mathlib_Equiv_trans___redArg(v___x_963_, v___x_965_);
return v___x_966_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_sigmaFinsetFunEquiv(lean_object* v_00_u03b9_967_, lean_object* v_00_u03b2_968_, lean_object* v_inst_969_, lean_object* v_inst_970_, lean_object* v_inst_971_){
_start:
{
lean_object* v___x_972_; 
v___x_972_ = lp_mathlib_DFinsupp_sigmaFinsetFunEquiv___redArg(v_inst_969_, v_inst_970_, v_inst_971_);
return v___x_972_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableZero___redArg___lam__0(lean_object* v_toFun_973_, lean_object* v_inst_974_, lean_object* v_a_975_, lean_object* v_h_976_){
_start:
{
lean_object* v___x_977_; lean_object* v___x_978_; uint8_t v___x_979_; 
lean_inc(v_a_975_);
v___x_977_ = lean_apply_1(v_toFun_973_, v_a_975_);
v___x_978_ = lean_apply_2(v_inst_974_, v_a_975_, v___x_977_);
v___x_979_ = lean_unbox(v___x_978_);
return v___x_979_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableZero___redArg___lam__0___boxed(lean_object* v_toFun_980_, lean_object* v_inst_981_, lean_object* v_a_982_, lean_object* v_h_983_){
_start:
{
uint8_t v_res_984_; lean_object* v_r_985_; 
v_res_984_ = lp_mathlib_DFinsupp_decidableZero___redArg___lam__0(v_toFun_980_, v_inst_981_, v_a_982_, v_h_983_);
v_r_985_ = lean_box(v_res_984_);
return v_r_985_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableZero___redArg(lean_object* v_inst_986_, lean_object* v_f_987_){
_start:
{
lean_object* v_toFun_988_; lean_object* v_support_x27_989_; lean_object* v___f_990_; uint8_t v___x_991_; 
v_toFun_988_ = lean_ctor_get(v_f_987_, 0);
lean_inc(v_toFun_988_);
v_support_x27_989_ = lean_ctor_get(v_f_987_, 1);
lean_inc(v_support_x27_989_);
lean_dec_ref(v_f_987_);
v___f_990_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_decidableZero___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_990_, 0, v_toFun_988_);
lean_closure_set(v___f_990_, 1, v_inst_986_);
v___x_991_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v_support_x27_989_, v___f_990_);
return v___x_991_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableZero___redArg___boxed(lean_object* v_inst_992_, lean_object* v_f_993_){
_start:
{
uint8_t v_res_994_; lean_object* v_r_995_; 
v_res_994_ = lp_mathlib_DFinsupp_decidableZero___redArg(v_inst_992_, v_f_993_);
v_r_995_ = lean_box(v_res_994_);
return v_r_995_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_decidableZero(lean_object* v_00_u03b9_996_, lean_object* v_00_u03b2_997_, lean_object* v_inst_998_, lean_object* v_inst_999_, lean_object* v_f_1000_){
_start:
{
uint8_t v___x_1001_; 
v___x_1001_ = lp_mathlib_DFinsupp_decidableZero___redArg(v_inst_999_, v_f_1000_);
return v___x_1001_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_decidableZero___boxed(lean_object* v_00_u03b9_1002_, lean_object* v_00_u03b2_1003_, lean_object* v_inst_1004_, lean_object* v_inst_1005_, lean_object* v_f_1006_){
_start:
{
uint8_t v_res_1007_; lean_object* v_r_1008_; 
v_res_1007_ = lp_mathlib_DFinsupp_decidableZero(v_00_u03b9_1002_, v_00_u03b2_1003_, v_inst_1004_, v_inst_1005_, v_f_1006_);
lean_dec(v_inst_1004_);
v_r_1008_ = lean_box(v_res_1007_);
return v_r_1008_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__0(lean_object* v_inst_1009_, lean_object* v_inst_1010_, lean_object* v_i_1011_, lean_object* v_x_1012_){
_start:
{
lean_object* v___x_1013_; lean_object* v___x_1014_; uint8_t v___x_1015_; 
lean_inc(v_i_1011_);
v___x_1013_ = lean_apply_1(v_inst_1009_, v_i_1011_);
v___x_1014_ = lean_apply_3(v_inst_1010_, v_i_1011_, v_x_1012_, v___x_1013_);
v___x_1015_ = lean_unbox(v___x_1014_);
if (v___x_1015_ == 0)
{
uint8_t v___x_1016_; 
v___x_1016_ = 1;
return v___x_1016_;
}
else
{
uint8_t v___x_1017_; 
v___x_1017_ = 0;
return v___x_1017_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__0___boxed(lean_object* v_inst_1018_, lean_object* v_inst_1019_, lean_object* v_i_1020_, lean_object* v_x_1021_){
_start:
{
uint8_t v_res_1022_; lean_object* v_r_1023_; 
v_res_1022_ = lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__0(v_inst_1018_, v_inst_1019_, v_i_1020_, v_x_1021_);
v_r_1023_ = lean_box(v_res_1022_);
return v_r_1023_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__1(lean_object* v_inst_1024_, lean_object* v_inst_1025_, uint8_t v___x_1026_, lean_object* v_i_1027_, lean_object* v_x_1028_){
_start:
{
lean_object* v___x_1029_; lean_object* v___x_1030_; uint8_t v___x_1031_; 
lean_inc(v_i_1027_);
v___x_1029_ = lean_apply_1(v_inst_1024_, v_i_1027_);
v___x_1030_ = lean_apply_3(v_inst_1025_, v_i_1027_, v_x_1028_, v___x_1029_);
v___x_1031_ = lean_unbox(v___x_1030_);
if (v___x_1031_ == 0)
{
return v___x_1026_;
}
else
{
uint8_t v___x_1032_; 
v___x_1032_ = 0;
return v___x_1032_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__1___boxed(lean_object* v_inst_1033_, lean_object* v_inst_1034_, lean_object* v___x_1035_, lean_object* v_i_1036_, lean_object* v_x_1037_){
_start:
{
uint8_t v___x_164__boxed_1038_; uint8_t v_res_1039_; lean_object* v_r_1040_; 
v___x_164__boxed_1038_ = lean_unbox(v___x_1035_);
v_res_1039_ = lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__1(v_inst_1033_, v_inst_1034_, v___x_164__boxed_1038_, v_i_1036_, v_x_1037_);
v_r_1040_ = lean_box(v_res_1039_);
return v_r_1040_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__3(lean_object* v___f_1041_, lean_object* v_f_1042_, lean_object* v_g_1043_, lean_object* v_inst_1044_, lean_object* v_a_1045_, lean_object* v_h_1046_){
_start:
{
lean_object* v___x_1047_; lean_object* v___x_1048_; lean_object* v___x_1049_; uint8_t v___x_1050_; 
lean_inc(v___f_1041_);
lean_inc_n(v_a_1045_, 2);
v___x_1047_ = lean_apply_2(v___f_1041_, v_f_1042_, v_a_1045_);
v___x_1048_ = lean_apply_2(v___f_1041_, v_g_1043_, v_a_1045_);
v___x_1049_ = lean_apply_3(v_inst_1044_, v_a_1045_, v___x_1047_, v___x_1048_);
v___x_1050_ = lean_unbox(v___x_1049_);
return v___x_1050_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__3___boxed(lean_object* v___f_1051_, lean_object* v_f_1052_, lean_object* v_g_1053_, lean_object* v_inst_1054_, lean_object* v_a_1055_, lean_object* v_h_1056_){
_start:
{
uint8_t v_res_1057_; lean_object* v_r_1058_; 
v_res_1057_ = lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__3(v___f_1051_, v_f_1052_, v_g_1053_, v_inst_1054_, v_a_1055_, v_h_1056_);
v_r_1058_ = lean_box(v_res_1057_);
return v_r_1058_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_instDecidableEq___redArg(lean_object* v_inst_1059_, lean_object* v_inst_1060_, lean_object* v_inst_1061_, lean_object* v_f_1062_, lean_object* v_g_1063_){
_start:
{
lean_object* v___f_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; uint8_t v___x_1067_; 
lean_inc_ref(v_inst_1061_);
lean_inc(v_inst_1060_);
v___f_1064_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__0___boxed), 4, 2);
lean_closure_set(v___f_1064_, 0, v_inst_1060_);
lean_closure_set(v___f_1064_, 1, v_inst_1061_);
lean_inc_ref(v_f_1062_);
lean_inc_ref(v___f_1064_);
lean_inc_ref_n(v_inst_1059_, 3);
v___x_1065_ = lp_mathlib_DFinsupp_support___redArg(v_inst_1059_, v___f_1064_, v_f_1062_);
lean_inc_ref(v_g_1063_);
v___x_1066_ = lp_mathlib_DFinsupp_support___redArg(v_inst_1059_, v___f_1064_, v_g_1063_);
v___x_1067_ = l_List_decidablePerm___redArg(v_inst_1059_, v___x_1065_, v___x_1066_);
if (v___x_1067_ == 0)
{
lean_dec_ref(v_g_1063_);
lean_dec_ref(v_f_1062_);
lean_dec_ref(v_inst_1061_);
lean_dec(v_inst_1060_);
lean_dec_ref(v_inst_1059_);
return v___x_1067_;
}
else
{
lean_object* v___x_1068_; lean_object* v___f_1069_; lean_object* v___f_1070_; lean_object* v___f_1071_; lean_object* v___x_1072_; uint8_t v___x_1073_; 
v___x_1068_ = lean_box(v___x_1067_);
lean_inc_ref(v_inst_1061_);
v___f_1069_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__1___boxed), 5, 3);
lean_closure_set(v___f_1069_, 0, v_inst_1060_);
lean_closure_set(v___f_1069_, 1, v_inst_1061_);
lean_closure_set(v___f_1069_, 2, v___x_1068_);
v___f_1070_ = ((lean_object*)(lp_mathlib_DFinsupp_coeFnAddMonoidHom___closed__0));
lean_inc_ref(v_f_1062_);
v___f_1071_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instDecidableEq___redArg___lam__3___boxed), 6, 4);
lean_closure_set(v___f_1071_, 0, v___f_1070_);
lean_closure_set(v___f_1071_, 1, v_f_1062_);
lean_closure_set(v___f_1071_, 2, v_g_1063_);
lean_closure_set(v___f_1071_, 3, v_inst_1061_);
v___x_1072_ = lp_mathlib_DFinsupp_support___redArg(v_inst_1059_, v___f_1069_, v_f_1062_);
v___x_1073_ = lp_mathlib_Multiset_decidableDforallMultiset___redArg(v___x_1072_, v___f_1071_);
return v___x_1073_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instDecidableEq___redArg___boxed(lean_object* v_inst_1074_, lean_object* v_inst_1075_, lean_object* v_inst_1076_, lean_object* v_f_1077_, lean_object* v_g_1078_){
_start:
{
uint8_t v_res_1079_; lean_object* v_r_1080_; 
v_res_1079_ = lp_mathlib_DFinsupp_instDecidableEq___redArg(v_inst_1074_, v_inst_1075_, v_inst_1076_, v_f_1077_, v_g_1078_);
v_r_1080_ = lean_box(v_res_1079_);
return v_r_1080_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_DFinsupp_instDecidableEq(lean_object* v_00_u03b9_1081_, lean_object* v_00_u03b2_1082_, lean_object* v_inst_1083_, lean_object* v_inst_1084_, lean_object* v_inst_1085_, lean_object* v_f_1086_, lean_object* v_g_1087_){
_start:
{
uint8_t v___x_1088_; 
v___x_1088_ = lp_mathlib_DFinsupp_instDecidableEq___redArg(v_inst_1083_, v_inst_1084_, v_inst_1085_, v_f_1086_, v_g_1087_);
return v___x_1088_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_instDecidableEq___boxed(lean_object* v_00_u03b9_1089_, lean_object* v_00_u03b2_1090_, lean_object* v_inst_1091_, lean_object* v_inst_1092_, lean_object* v_inst_1093_, lean_object* v_f_1094_, lean_object* v_g_1095_){
_start:
{
uint8_t v_res_1096_; lean_object* v_r_1097_; 
v_res_1096_ = lp_mathlib_DFinsupp_instDecidableEq(v_00_u03b9_1089_, v_00_u03b2_1090_, v_inst_1091_, v_inst_1092_, v_inst_1093_, v_f_1094_, v_g_1095_);
v_r_1097_ = lean_box(v_res_1096_);
return v_r_1097_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_comapDomain_x27___redArg___lam__0(lean_object* v_h_1098_, lean_object* v_toFun_1099_, lean_object* v_x_1100_){
_start:
{
lean_object* v___x_1101_; lean_object* v___x_1102_; 
v___x_1101_ = lean_apply_1(v_h_1098_, v_x_1100_);
v___x_1102_ = lean_apply_1(v_toFun_1099_, v___x_1101_);
return v___x_1102_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_comapDomain_x27___redArg(lean_object* v_h_1103_, lean_object* v_h_x27_1104_, lean_object* v_f_1105_){
_start:
{
lean_object* v_toFun_1106_; lean_object* v_support_x27_1107_; lean_object* v___x_1109_; uint8_t v_isShared_1110_; uint8_t v_isSharedCheck_1116_; 
v_toFun_1106_ = lean_ctor_get(v_f_1105_, 0);
v_support_x27_1107_ = lean_ctor_get(v_f_1105_, 1);
v_isSharedCheck_1116_ = !lean_is_exclusive(v_f_1105_);
if (v_isSharedCheck_1116_ == 0)
{
v___x_1109_ = v_f_1105_;
v_isShared_1110_ = v_isSharedCheck_1116_;
goto v_resetjp_1108_;
}
else
{
lean_inc(v_support_x27_1107_);
lean_inc(v_toFun_1106_);
lean_dec(v_f_1105_);
v___x_1109_ = lean_box(0);
v_isShared_1110_ = v_isSharedCheck_1116_;
goto v_resetjp_1108_;
}
v_resetjp_1108_:
{
lean_object* v___f_1111_; lean_object* v___x_1112_; lean_object* v___x_1114_; 
v___f_1111_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_comapDomain_x27___redArg___lam__0), 3, 2);
lean_closure_set(v___f_1111_, 0, v_h_1103_);
lean_closure_set(v___f_1111_, 1, v_toFun_1106_);
v___x_1112_ = lp_mathlib_Multiset_map___redArg(v_h_x27_1104_, v_support_x27_1107_);
if (v_isShared_1110_ == 0)
{
lean_ctor_set(v___x_1109_, 1, v___x_1112_);
lean_ctor_set(v___x_1109_, 0, v___f_1111_);
v___x_1114_ = v___x_1109_;
goto v_reusejp_1113_;
}
else
{
lean_object* v_reuseFailAlloc_1115_; 
v_reuseFailAlloc_1115_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1115_, 0, v___f_1111_);
lean_ctor_set(v_reuseFailAlloc_1115_, 1, v___x_1112_);
v___x_1114_ = v_reuseFailAlloc_1115_;
goto v_reusejp_1113_;
}
v_reusejp_1113_:
{
return v___x_1114_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_comapDomain_x27(lean_object* v_00_u03b9_1117_, lean_object* v_00_u03b2_1118_, lean_object* v_00_u03ba_1119_, lean_object* v_inst_1120_, lean_object* v_h_1121_, lean_object* v_h_x27_1122_, lean_object* v_hh_x27_1123_, lean_object* v_f_1124_){
_start:
{
lean_object* v___x_1125_; 
v___x_1125_ = lp_mathlib_DFinsupp_comapDomain_x27___redArg(v_h_1121_, v_h_x27_1122_, v_f_1124_);
return v___x_1125_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_comapDomain_x27___boxed(lean_object* v_00_u03b9_1126_, lean_object* v_00_u03b2_1127_, lean_object* v_00_u03ba_1128_, lean_object* v_inst_1129_, lean_object* v_h_1130_, lean_object* v_h_x27_1131_, lean_object* v_hh_x27_1132_, lean_object* v_f_1133_){
_start:
{
lean_object* v_res_1134_; 
v_res_1134_ = lp_mathlib_DFinsupp_comapDomain_x27(v_00_u03b9_1126_, v_00_u03b2_1127_, v_00_u03ba_1128_, v_inst_1129_, v_h_1130_, v_h_x27_1131_, v_hh_x27_1132_, v_f_1133_);
lean_dec(v_inst_1129_);
return v_res_1134_;
}
}
static lean_object* _init_lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__1___closed__0(void){
_start:
{
lean_object* v___x_1135_; 
v___x_1135_ = lp_mathlib_Equiv_cast(lean_box(0), lean_box(0), lean_box(0));
return v___x_1135_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__1(lean_object* v_i_1136_, lean_object* v___y_1137_){
_start:
{
lean_object* v___x_1138_; lean_object* v_toFun_1139_; lean_object* v___x_1140_; 
v___x_1138_ = lean_obj_once(&lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__1___closed__0, &lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__1___closed__0_once, _init_lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__1___closed__0);
v_toFun_1139_ = lean_ctor_get(v___x_1138_, 0);
lean_inc(v_toFun_1139_);
v___x_1140_ = lean_apply_1(v_toFun_1139_, v___y_1137_);
return v___x_1140_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__1___boxed(lean_object* v_i_1141_, lean_object* v___y_1142_){
_start:
{
lean_object* v_res_1143_; 
v_res_1143_ = lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__1(v_i_1141_, v___y_1142_);
lean_dec(v_i_1141_);
return v_res_1143_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__0(lean_object* v___f_1144_, lean_object* v_invFun_1145_, lean_object* v___f_1146_, lean_object* v_f_1147_){
_start:
{
lean_object* v___x_1148_; lean_object* v___x_1149_; 
v___x_1148_ = lp_mathlib_DFinsupp_comapDomain_x27___redArg(v___f_1144_, v_invFun_1145_, v_f_1147_);
v___x_1149_ = lp_mathlib_DFinsupp_mapRange___redArg(v___f_1146_, v___x_1148_);
return v___x_1149_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__2(lean_object* v___x_1150_, lean_object* v___y_1151_){
_start:
{
lean_object* v_toFun_1152_; lean_object* v___x_1153_; 
v_toFun_1152_ = lean_ctor_get(v___x_1150_, 0);
lean_inc(v_toFun_1152_);
lean_dec_ref(v___x_1150_);
v___x_1153_ = lean_apply_1(v_toFun_1152_, v___y_1151_);
return v___x_1153_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivCongrLeft___redArg(lean_object* v_inst_1155_, lean_object* v_h_1156_){
_start:
{
lean_object* v_toFun_1157_; lean_object* v_invFun_1158_; lean_object* v___f_1159_; lean_object* v___f_1160_; lean_object* v___f_1161_; lean_object* v___x_1162_; lean_object* v___f_1163_; lean_object* v___x_1164_; lean_object* v___x_1165_; 
v_toFun_1157_ = lean_ctor_get(v_h_1156_, 0);
lean_inc_n(v_toFun_1157_, 2);
v_invFun_1158_ = lean_ctor_get(v_h_1156_, 1);
v___f_1159_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_update___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1159_, 0, v_toFun_1157_);
v___f_1160_ = ((lean_object*)(lp_mathlib_DFinsupp_equivCongrLeft___redArg___closed__0));
lean_inc(v_invFun_1158_);
v___f_1161_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__0), 4, 3);
lean_closure_set(v___f_1161_, 0, v___f_1159_);
lean_closure_set(v___f_1161_, 1, v_invFun_1158_);
lean_closure_set(v___f_1161_, 2, v___f_1160_);
v___x_1162_ = lp_mathlib_Equiv_symm___redArg(v_h_1156_);
v___f_1163_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_equivCongrLeft___redArg___lam__2), 2, 1);
lean_closure_set(v___f_1163_, 0, v___x_1162_);
v___x_1164_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_comapDomain_x27___boxed), 8, 7);
lean_closure_set(v___x_1164_, 0, lean_box(0));
lean_closure_set(v___x_1164_, 1, lean_box(0));
lean_closure_set(v___x_1164_, 2, lean_box(0));
lean_closure_set(v___x_1164_, 3, v_inst_1155_);
lean_closure_set(v___x_1164_, 4, v___f_1163_);
lean_closure_set(v___x_1164_, 5, v_toFun_1157_);
lean_closure_set(v___x_1164_, 6, lean_box(0));
v___x_1165_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1165_, 0, v___x_1164_);
lean_ctor_set(v___x_1165_, 1, v___f_1161_);
return v___x_1165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_equivCongrLeft(lean_object* v_00_u03b9_1166_, lean_object* v_00_u03b2_1167_, lean_object* v_00_u03ba_1168_, lean_object* v_inst_1169_, lean_object* v_h_1170_){
_start:
{
lean_object* v___x_1171_; 
v___x_1171_ = lp_mathlib_DFinsupp_equivCongrLeft___redArg(v_inst_1169_, v_h_1170_);
return v___x_1171_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_extendWith___redArg___lam__0(lean_object* v_a_1172_, lean_object* v_toFun_1173_, lean_object* v_i_1174_){
_start:
{
if (lean_obj_tag(v_i_1174_) == 0)
{
lean_dec(v_toFun_1173_);
lean_inc(v_a_1172_);
return v_a_1172_;
}
else
{
lean_object* v_val_1175_; lean_object* v___x_1176_; 
v_val_1175_ = lean_ctor_get(v_i_1174_, 0);
lean_inc(v_val_1175_);
lean_dec_ref_known(v_i_1174_, 1);
v___x_1176_ = lean_apply_1(v_toFun_1173_, v_val_1175_);
return v___x_1176_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_extendWith___redArg___lam__0___boxed(lean_object* v_a_1177_, lean_object* v_toFun_1178_, lean_object* v_i_1179_){
_start:
{
lean_object* v_res_1180_; 
v_res_1180_ = lp_mathlib_DFinsupp_extendWith___redArg___lam__0(v_a_1177_, v_toFun_1178_, v_i_1179_);
lean_dec(v_a_1177_);
return v_res_1180_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_extendWith___redArg___lam__1(lean_object* v_val_1181_){
_start:
{
lean_object* v___x_1182_; 
v___x_1182_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_1182_, 0, v_val_1181_);
return v___x_1182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_extendWith___redArg(lean_object* v_a_1184_, lean_object* v_f_1185_){
_start:
{
lean_object* v_toFun_1186_; lean_object* v_support_x27_1187_; lean_object* v___x_1189_; uint8_t v_isShared_1190_; uint8_t v_isSharedCheck_1199_; 
v_toFun_1186_ = lean_ctor_get(v_f_1185_, 0);
v_support_x27_1187_ = lean_ctor_get(v_f_1185_, 1);
v_isSharedCheck_1199_ = !lean_is_exclusive(v_f_1185_);
if (v_isSharedCheck_1199_ == 0)
{
v___x_1189_ = v_f_1185_;
v_isShared_1190_ = v_isSharedCheck_1199_;
goto v_resetjp_1188_;
}
else
{
lean_inc(v_support_x27_1187_);
lean_inc(v_toFun_1186_);
lean_dec(v_f_1185_);
v___x_1189_ = lean_box(0);
v_isShared_1190_ = v_isSharedCheck_1199_;
goto v_resetjp_1188_;
}
v_resetjp_1188_:
{
lean_object* v___f_1191_; lean_object* v___f_1192_; lean_object* v___x_1193_; lean_object* v___x_1194_; lean_object* v___x_1195_; lean_object* v___x_1197_; 
v___f_1191_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_extendWith___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_1191_, 0, v_a_1184_);
lean_closure_set(v___f_1191_, 1, v_toFun_1186_);
v___f_1192_ = ((lean_object*)(lp_mathlib_DFinsupp_extendWith___redArg___closed__0));
v___x_1193_ = lean_box(0);
v___x_1194_ = lp_mathlib_Multiset_map___redArg(v___f_1192_, v_support_x27_1187_);
v___x_1195_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_1195_, 0, v___x_1193_);
lean_ctor_set(v___x_1195_, 1, v___x_1194_);
if (v_isShared_1190_ == 0)
{
lean_ctor_set(v___x_1189_, 1, v___x_1195_);
lean_ctor_set(v___x_1189_, 0, v___f_1191_);
v___x_1197_ = v___x_1189_;
goto v_reusejp_1196_;
}
else
{
lean_object* v_reuseFailAlloc_1198_; 
v_reuseFailAlloc_1198_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_1198_, 0, v___f_1191_);
lean_ctor_set(v_reuseFailAlloc_1198_, 1, v___x_1195_);
v___x_1197_ = v_reuseFailAlloc_1198_;
goto v_reusejp_1196_;
}
v_reusejp_1196_:
{
return v___x_1197_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_extendWith(lean_object* v_00_u03b9_1200_, lean_object* v_00_u03b1_1201_, lean_object* v_inst_1202_, lean_object* v_a_1203_, lean_object* v_f_1204_){
_start:
{
lean_object* v___x_1205_; 
v___x_1205_ = lp_mathlib_DFinsupp_extendWith___redArg(v_a_1203_, v_f_1204_);
return v___x_1205_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_extendWith___boxed(lean_object* v_00_u03b9_1206_, lean_object* v_00_u03b1_1207_, lean_object* v_inst_1208_, lean_object* v_a_1209_, lean_object* v_f_1210_){
_start:
{
lean_object* v_res_1211_; 
v_res_1211_ = lp_mathlib_DFinsupp_extendWith(v_00_u03b9_1206_, v_00_u03b1_1207_, v_inst_1208_, v_a_1209_, v_f_1210_);
lean_dec(v_inst_1208_);
return v_res_1211_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_addMonoidHom___redArg___lam__2(lean_object* v_f_1212_, lean_object* v_i_1213_, lean_object* v_x_1214_){
_start:
{
lean_object* v___x_1215_; 
v___x_1215_ = lean_apply_2(v_f_1212_, v_i_1213_, v_x_1214_);
return v___x_1215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_addMonoidHom___redArg(lean_object* v_inst_1216_, lean_object* v_inst_1217_, lean_object* v_f_1218_){
_start:
{
lean_object* v___f_1219_; lean_object* v___f_1220_; lean_object* v___f_1221_; lean_object* v___x_1222_; 
v___f_1219_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instAdd___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1219_, 0, v_inst_1216_);
v___f_1220_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instAdd___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1220_, 0, v_inst_1217_);
v___f_1221_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_mapRange_addMonoidHom___redArg___lam__2), 3, 1);
lean_closure_set(v___f_1221_, 0, v_f_1218_);
v___x_1222_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_mapRange___boxed), 8, 7);
lean_closure_set(v___x_1222_, 0, lean_box(0));
lean_closure_set(v___x_1222_, 1, lean_box(0));
lean_closure_set(v___x_1222_, 2, lean_box(0));
lean_closure_set(v___x_1222_, 3, v___f_1219_);
lean_closure_set(v___x_1222_, 4, v___f_1220_);
lean_closure_set(v___x_1222_, 5, v___f_1221_);
lean_closure_set(v___x_1222_, 6, lean_box(0));
return v___x_1222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_addMonoidHom(lean_object* v_00_u03b9_1223_, lean_object* v_00_u03b2_u2081_1224_, lean_object* v_00_u03b2_u2082_1225_, lean_object* v_inst_1226_, lean_object* v_inst_1227_, lean_object* v_f_1228_){
_start:
{
lean_object* v___x_1229_; 
v___x_1229_ = lp_mathlib_DFinsupp_mapRange_addMonoidHom___redArg(v_inst_1226_, v_inst_1227_, v_f_1228_);
return v___x_1229_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_addEquiv___redArg___lam__2(lean_object* v_e_1230_, lean_object* v_i_1231_, lean_object* v_x_1232_){
_start:
{
lean_object* v___x_1233_; lean_object* v_toFun_1234_; lean_object* v___x_1235_; 
v___x_1233_ = lean_apply_1(v_e_1230_, v_i_1231_);
v_toFun_1234_ = lean_ctor_get(v___x_1233_, 0);
lean_inc(v_toFun_1234_);
lean_dec_ref(v___x_1233_);
v___x_1235_ = lean_apply_1(v_toFun_1234_, v_x_1232_);
return v___x_1235_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_addEquiv___redArg___lam__0(lean_object* v_e_1236_, lean_object* v_i_1237_, lean_object* v_x_1238_){
_start:
{
lean_object* v___x_1239_; lean_object* v___x_1240_; lean_object* v_toFun_1241_; lean_object* v___x_1242_; 
v___x_1239_ = lean_apply_1(v_e_1236_, v_i_1237_);
v___x_1240_ = lp_mathlib_Equiv_symm___redArg(v___x_1239_);
v_toFun_1241_ = lean_ctor_get(v___x_1240_, 0);
lean_inc(v_toFun_1241_);
lean_dec_ref(v___x_1240_);
v___x_1242_ = lean_apply_1(v_toFun_1241_, v_x_1238_);
return v___x_1242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_addEquiv___redArg(lean_object* v_inst_1243_, lean_object* v_inst_1244_, lean_object* v_e_1245_){
_start:
{
lean_object* v___f_1246_; lean_object* v___f_1247_; lean_object* v___f_1248_; lean_object* v___f_1249_; lean_object* v___x_1250_; lean_object* v___x_1251_; lean_object* v___x_1252_; 
v___f_1246_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instAdd___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1246_, 0, v_inst_1243_);
v___f_1247_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_instAdd___redArg___lam__0), 2, 1);
lean_closure_set(v___f_1247_, 0, v_inst_1244_);
lean_inc_ref(v_e_1245_);
v___f_1248_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_mapRange_addEquiv___redArg___lam__2), 3, 1);
lean_closure_set(v___f_1248_, 0, v_e_1245_);
v___f_1249_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_mapRange_addEquiv___redArg___lam__0), 3, 1);
lean_closure_set(v___f_1249_, 0, v_e_1245_);
lean_inc_ref(v___f_1247_);
lean_inc_ref(v___f_1246_);
v___x_1250_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_mapRange___boxed), 8, 7);
lean_closure_set(v___x_1250_, 0, lean_box(0));
lean_closure_set(v___x_1250_, 1, lean_box(0));
lean_closure_set(v___x_1250_, 2, lean_box(0));
lean_closure_set(v___x_1250_, 3, v___f_1246_);
lean_closure_set(v___x_1250_, 4, v___f_1247_);
lean_closure_set(v___x_1250_, 5, v___f_1248_);
lean_closure_set(v___x_1250_, 6, lean_box(0));
v___x_1251_ = lean_alloc_closure((void*)(lp_mathlib_DFinsupp_mapRange___boxed), 8, 7);
lean_closure_set(v___x_1251_, 0, lean_box(0));
lean_closure_set(v___x_1251_, 1, lean_box(0));
lean_closure_set(v___x_1251_, 2, lean_box(0));
lean_closure_set(v___x_1251_, 3, v___f_1247_);
lean_closure_set(v___x_1251_, 4, v___f_1246_);
lean_closure_set(v___x_1251_, 5, v___f_1249_);
lean_closure_set(v___x_1251_, 6, lean_box(0));
v___x_1252_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_1252_, 0, v___x_1250_);
lean_ctor_set(v___x_1252_, 1, v___x_1251_);
return v___x_1252_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_DFinsupp_mapRange_addEquiv(lean_object* v_00_u03b9_1253_, lean_object* v_00_u03b2_u2081_1254_, lean_object* v_00_u03b2_u2082_1255_, lean_object* v_inst_1256_, lean_object* v_inst_1257_, lean_object* v_e_1258_){
_start:
{
lean_object* v___x_1259_; 
v___x_1259_ = lp_mathlib_DFinsupp_mapRange_addEquiv___redArg(v_inst_1256_, v_inst_1257_, v_e_1258_);
return v___x_1259_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_InjSurj(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Notation_Prod(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Defs(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Notation_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_DFinsupp_Defs(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_term_u03a0_u2080___x2c__ = _init_lp_mathlib_term_u03a0_u2080___x2c__();
lean_mark_persistent(lp_mathlib_term_u03a0_u2080___x2c__);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Defs(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Equiv_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Pi_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_DFinsupp_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_DFinsupp_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_DFinsupp_Defs(builtin);
}
#ifdef __cplusplus
}
#endif
