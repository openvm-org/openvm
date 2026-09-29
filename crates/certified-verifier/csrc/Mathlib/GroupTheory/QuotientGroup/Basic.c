// Lean compiler output
// Module: Mathlib.GroupTheory.QuotientGroup.Basic
// Imports: public import Init public meta import Init public import Mathlib.Algebra.Group.Subgroup.Pointwise public import Mathlib.Data.Int.Cast.Lemmas public import Mathlib.GroupTheory.Coset.Basic public import Mathlib.GroupTheory.QuotientGroup.Defs public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
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
lean_object* lp_mathlib_MonoidHom_codRestrict___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Con_lift___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_QuotientAddGroup_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_OneHom_id___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_QuotientAddGroup_map___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddMonoidHom_toAddEquiv___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Name_str___override(lean_object*, lean_object*);
lean_object* l_Lean_Name_num___override(lean_object*, lean_object*);
lean_object* lp_mathlib_QuotientGroup_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_MonoidHom_toMulEquiv___redArg(lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_QuotientGroup_map___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddSubgroupClass_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_SubgroupClass_inclusion___lam__0___boxed(lean_object*);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* lp_mathlib_Subgroup_quotientEquivOfEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubgroupClass_toGroup___redArg(lean_object*);
lean_object* lp_mathlib_QuotientGroup_Quotient_group___redArg(lean_object*, lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_AddSubgroup_quotientEquivOfEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "_private"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__0_value),LEAN_SCALAR_PTR_LITERAL(103, 214, 75, 80, 34, 198, 193, 153)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "Mathlib"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__2_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__1_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__2_value),LEAN_SCALAR_PTR_LITERAL(234, 232, 174, 134, 127, 136, 69, 92)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__3 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__3_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "GroupTheory"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__3_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__4_value),LEAN_SCALAR_PTR_LITERAL(21, 126, 254, 74, 51, 201, 216, 222)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "QuotientGroup"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__6_value),LEAN_SCALAR_PTR_LITERAL(42, 76, 215, 158, 185, 28, 159, 105)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__7_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "Basic"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__8_value),LEAN_SCALAR_PTR_LITERAL(137, 167, 2, 52, 255, 222, 67, 219)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(76, 242, 239, 32, 63, 228, 190, 130)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__6_value),LEAN_SCALAR_PTR_LITERAL(247, 84, 93, 146, 168, 236, 0, 193)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__11_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "termQ"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__12_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__11_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__12_value),LEAN_SCALAR_PTR_LITERAL(143, 23, 196, 6, 133, 112, 229, 174)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__13_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = " Q "};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__14_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__14_value)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__13_value),((lean_object*)(((size_t)(1024) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__15_value)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__16_value;
LEAN_EXPORT const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__16_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_⧸_"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__0 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__0_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(193, 111, 223, 60, 234, 196, 87, 111)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__1 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__1_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "G"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__2 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__2_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__3;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(101, 55, 191, 37, 243, 21, 34, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__4 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__4_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(101, 55, 191, 37, 243, 21, 34, 158)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__5 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__5_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "_@"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__6 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__6_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__5_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(216, 211, 51, 65, 145, 150, 96, 194)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__7 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__7_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__7_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__2_value),LEAN_SCALAR_PTR_LITERAL(193, 52, 230, 80, 73, 178, 33, 196)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__8 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__8_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__8_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__4_value),LEAN_SCALAR_PTR_LITERAL(242, 242, 112, 36, 24, 65, 120, 180)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__9 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__9_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__9_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__6_value),LEAN_SCALAR_PTR_LITERAL(73, 63, 16, 228, 41, 21, 136, 155)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__10 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__10_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__10_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__8_value),LEAN_SCALAR_PTR_LITERAL(190, 232, 122, 223, 43, 247, 148, 36)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__11 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__11_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__11_value),((lean_object*)(((size_t)(21039524) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(164, 194, 186, 171, 107, 156, 199, 142)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__12 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__12_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "_hygCtx"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__13 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__13_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__12_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(11, 101, 105, 200, 243, 147, 215, 167)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__14 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__14_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "_hyg"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__15 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__15_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__14_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(75, 98, 86, 162, 240, 104, 151, 195)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__16 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__16_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__16_value),((lean_object*)(((size_t)(18) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(63, 47, 75, 168, 90, 235, 96, 165)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__17 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__17_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__17_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__18 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__18_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__18_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__19 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__19_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 1, .m_data = "⧸"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__20 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__20_value;
static const lean_string_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "N"};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__21 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__21_value;
static lean_once_cell_t lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__22;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(144, 2, 116, 232, 240, 236, 195, 248)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__23 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__23_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__21_value),LEAN_SCALAR_PTR_LITERAL(144, 2, 116, 232, 240, 236, 195, 248)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__24 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__24_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__24_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(17, 78, 60, 226, 45, 52, 159, 77)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__25 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__25_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__25_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__2_value),LEAN_SCALAR_PTR_LITERAL(196, 74, 253, 98, 91, 226, 17, 81)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__26 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__26_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__26_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__4_value),LEAN_SCALAR_PTR_LITERAL(219, 44, 246, 180, 28, 105, 98, 5)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__27 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__27_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__27_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__6_value),LEAN_SCALAR_PTR_LITERAL(188, 53, 246, 59, 197, 214, 80, 121)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__28 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__28_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__28_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__8_value),LEAN_SCALAR_PTR_LITERAL(47, 219, 222, 184, 50, 38, 244, 211)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__29 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__29_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__29_value),((lean_object*)(((size_t)(21039524) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(105, 46, 102, 200, 48, 170, 144, 246)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__30 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__30_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__30_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__13_value),LEAN_SCALAR_PTR_LITERAL(58, 220, 232, 212, 151, 39, 10, 194)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__31 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__31_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__31_value),((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(6, 127, 138, 188, 132, 138, 162, 12)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__32 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__32_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__32_value),((lean_object*)(((size_t)(20) << 1) | 1)),LEAN_SCALAR_PTR_LITERAL(17, 199, 134, 170, 28, 43, 131, 78)}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__33 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__33_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__33_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__34 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__34_value;
static const lean_ctor_object lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__34_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__35 = (const lean_object*)&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__35_value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_prodEquiv___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_prodEquiv___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_QuotientGroup_prodEquiv___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_QuotientGroup_prodEquiv___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_QuotientGroup_prodEquiv___closed__0 = (const lean_object*)&lp_mathlib_QuotientGroup_prodEquiv___closed__0_value;
static const lean_closure_object lp_mathlib_QuotientGroup_prodEquiv___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_QuotientGroup_prodEquiv___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_QuotientGroup_prodEquiv___closed__1 = (const lean_object*)&lp_mathlib_QuotientGroup_prodEquiv___closed__1_value;
static const lean_ctor_object lp_mathlib_QuotientGroup_prodEquiv___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_QuotientGroup_prodEquiv___closed__0_value),((lean_object*)&lp_mathlib_QuotientGroup_prodEquiv___closed__1_value)}};
static const lean_object* lp_mathlib_QuotientGroup_prodEquiv___closed__2 = (const lean_object*)&lp_mathlib_QuotientGroup_prodEquiv___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_prodEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_prodEquiv_match__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_prodEquiv_match__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_prodEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_prodEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_prodMulEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_prodMulEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_prodMulEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_prodMulEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_prodAddEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_prodAddEquiv___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_prodAddEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_prodAddEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_kerLift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_kerLift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_kerLift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_kerLift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_kerLift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_kerLift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_rangeKerLift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_rangeKerLift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_rangeKerLift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_rangeKerLift___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_rangeKerLift(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_rangeKerLift___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientKerEquivOfRightInverse___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientKerEquivOfRightInverse___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientKerEquivOfRightInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientKerEquivOfRightInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientKerEquivOfRightInverse___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientKerEquivOfRightInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientKerEquivOfRightInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_QuotientGroup_quotientBot___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_OneHom_id___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_QuotientGroup_quotientBot___redArg___closed__0 = (const lean_object*)&lp_mathlib_QuotientGroup_quotientBot___redArg___closed__0_value;
static const lean_closure_object lp_mathlib_QuotientGroup_quotientBot___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_id___boxed, .m_arity = 2, .m_num_fixed = 1, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))} };
static const lean_object* lp_mathlib_QuotientGroup_quotientBot___redArg___closed__1 = (const lean_object*)&lp_mathlib_QuotientGroup_quotientBot___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientMulEquivOfEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientMulEquivOfEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientMulEquivOfEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientMulEquivOfEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientAddEquivOfEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientAddEquivOfEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientAddEquivOfEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientAddEquivOfEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_QuotientGroup_quotientMapSubgroupOfOfLe___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_SubgroupClass_inclusion___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_QuotientGroup_quotientMapSubgroupOfOfLe___redArg___closed__0 = (const lean_object*)&lp_mathlib_QuotientGroup_quotientMapSubgroupOfOfLe___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientMapSubgroupOfOfLe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientMapSubgroupOfOfLe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientMapAddSubgroupOfOfLe___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientMapAddSubgroupOfOfLe(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_equivQuotientSubgroupOfOfEq___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_equivQuotientSubgroupOfOfEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_equivQuotientAddSubgroupOfOfEq___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_equivQuotientAddSubgroupOfOfEq(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_homQuotientZPowOfHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_homQuotientZPowOfHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_homQuotientZPowOfHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_homQuotientZSMulOfHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_homQuotientZSMulOfHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_homQuotientZSMulOfHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_equivQuotientZPowOfEquiv___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_equivQuotientZPowOfEquiv___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_equivQuotientZPowOfEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_equivQuotientZPowOfEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_equivQuotientZPowOfEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_equivQuotientZSMulOfEquiv___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_equivQuotientZSMulOfEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_equivQuotientZSMulOfEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientQuotientEquivQuotientAux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientQuotientEquivQuotientAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientQuotientEquivQuotientAux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientQuotientEquivQuotientAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientQuotientEquivQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientQuotientEquivQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientQuotientEquivQuotient___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientQuotientEquivQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_comapMk_x27OrderIso___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_comapMk_x27OrderIso___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_QuotientGroup_comapMk_x27OrderIso___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_QuotientGroup_comapMk_x27OrderIso___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_QuotientGroup_comapMk_x27OrderIso___closed__0 = (const lean_object*)&lp_mathlib_QuotientGroup_comapMk_x27OrderIso___closed__0_value;
static const lean_closure_object lp_mathlib_QuotientGroup_comapMk_x27OrderIso___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_QuotientGroup_comapMk_x27OrderIso___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_QuotientGroup_comapMk_x27OrderIso___closed__1 = (const lean_object*)&lp_mathlib_QuotientGroup_comapMk_x27OrderIso___closed__1_value;
static const lean_ctor_object lp_mathlib_QuotientGroup_comapMk_x27OrderIso___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_QuotientGroup_comapMk_x27OrderIso___closed__0_value),((lean_object*)&lp_mathlib_QuotientGroup_comapMk_x27OrderIso___closed__1_value)}};
static const lean_object* lp_mathlib_QuotientGroup_comapMk_x27OrderIso___closed__2 = (const lean_object*)&lp_mathlib_QuotientGroup_comapMk_x27OrderIso___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_comapMk_x27OrderIso(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_comapMk_x27OrderIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___lam__1(lean_object*);
static const lean_closure_object lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___closed__0 = (const lean_object*)&lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___closed__0_value;
static const lean_closure_object lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___lam__1, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___closed__1 = (const lean_object*)&lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___closed__1_value;
static const lean_ctor_object lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___closed__0_value),((lean_object*)&lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___closed__1_value)}};
static const lean_object* lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___closed__2 = (const lean_object*)&lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___closed__2_value;
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrictHomKerEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_MonoidHom_domRestrictHomKerEquiv___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Con_lift___redArg___lam__0, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_MonoidHom_domRestrictHomKerEquiv___redArg___closed__0 = (const lean_object*)&lp_mathlib_MonoidHom_domRestrictHomKerEquiv___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrictHomKerEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrictHomKerEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrictHomKerEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv_match__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv_match__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv_match__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_restrictHomKerEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_restrictHomKerEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_restrictHomKerEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_restrictHomKerEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_restrictHomKerEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_restrictHomKerEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__3(void){
_start:
{
lean_object* v___x_43_; lean_object* v___x_44_; 
v___x_43_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__2));
v___x_44_ = l_String_toRawSubstring_x27(v___x_43_);
return v___x_44_;
}
}
static lean_object* _init_lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__22(void){
_start:
{
lean_object* v___x_88_; lean_object* v___x_89_; 
v___x_88_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__21));
v___x_89_ = l_String_toRawSubstring_x27(v___x_88_);
return v___x_89_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1(lean_object* v_x_128_, lean_object* v_a_129_, lean_object* v_a_130_){
_start:
{
lean_object* v___x_131_; uint8_t v___x_132_; 
v___x_131_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup_termQ___closed__13));
v___x_132_ = l_Lean_Syntax_isOfKind(v_x_128_, v___x_131_);
if (v___x_132_ == 0)
{
lean_object* v___x_133_; lean_object* v___x_134_; 
v___x_133_ = lean_box(1);
v___x_134_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_134_, 0, v___x_133_);
lean_ctor_set(v___x_134_, 1, v_a_130_);
return v___x_134_;
}
else
{
lean_object* v_quotContext_135_; lean_object* v_currMacroScope_136_; lean_object* v_ref_137_; uint8_t v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; lean_object* v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; lean_object* v___x_149_; lean_object* v___x_150_; lean_object* v___x_151_; lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; 
v_quotContext_135_ = lean_ctor_get(v_a_129_, 1);
v_currMacroScope_136_ = lean_ctor_get(v_a_129_, 2);
v_ref_137_ = lean_ctor_get(v_a_129_, 5);
v___x_138_ = 0;
v___x_139_ = l_Lean_SourceInfo_fromRef(v_ref_137_, v___x_138_);
v___x_140_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__1));
v___x_141_ = lean_obj_once(&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__3, &lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__3_once, _init_lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__3);
v___x_142_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__4));
lean_inc_n(v_currMacroScope_136_, 2);
lean_inc_n(v_quotContext_135_, 2);
v___x_143_ = l_Lean_addMacroScope(v_quotContext_135_, v___x_142_, v_currMacroScope_136_);
v___x_144_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__19));
lean_inc_n(v___x_139_, 3);
v___x_145_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_145_, 0, v___x_139_);
lean_ctor_set(v___x_145_, 1, v___x_141_);
lean_ctor_set(v___x_145_, 2, v___x_143_);
lean_ctor_set(v___x_145_, 3, v___x_144_);
v___x_146_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__20));
v___x_147_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_147_, 0, v___x_139_);
lean_ctor_set(v___x_147_, 1, v___x_146_);
v___x_148_ = lean_obj_once(&lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__22, &lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__22_once, _init_lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__22);
v___x_149_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__23));
v___x_150_ = l_Lean_addMacroScope(v_quotContext_135_, v___x_149_, v_currMacroScope_136_);
v___x_151_ = ((lean_object*)(lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___closed__35));
v___x_152_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_152_, 0, v___x_139_);
lean_ctor_set(v___x_152_, 1, v___x_148_);
lean_ctor_set(v___x_152_, 2, v___x_150_);
lean_ctor_set(v___x_152_, 3, v___x_151_);
v___x_153_ = l_Lean_Syntax_node3(v___x_139_, v___x_140_, v___x_145_, v___x_147_, v___x_152_);
v___x_154_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_154_, 0, v___x_153_);
lean_ctor_set(v___x_154_, 1, v_a_130_);
return v___x_154_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1___boxed(lean_object* v_x_155_, lean_object* v_a_156_, lean_object* v_a_157_){
_start:
{
lean_object* v_res_158_; 
v_res_158_ = lp_mathlib___private_Mathlib_GroupTheory_QuotientGroup_Basic_0__QuotientGroup___aux__Mathlib__GroupTheory__QuotientGroup__Basic______macroRules____private__Mathlib__GroupTheory__QuotientGroup__Basic__0__QuotientGroup__termQ__1(v_x_155_, v_a_156_, v_a_157_);
lean_dec_ref(v_a_156_);
return v_res_158_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_prodEquiv___lam__0(lean_object* v_q_159_){
_start:
{
lean_object* v_fst_160_; lean_object* v_snd_161_; lean_object* v___x_163_; uint8_t v_isShared_164_; uint8_t v_isSharedCheck_168_; 
v_fst_160_ = lean_ctor_get(v_q_159_, 0);
v_snd_161_ = lean_ctor_get(v_q_159_, 1);
v_isSharedCheck_168_ = !lean_is_exclusive(v_q_159_);
if (v_isSharedCheck_168_ == 0)
{
v___x_163_ = v_q_159_;
v_isShared_164_ = v_isSharedCheck_168_;
goto v_resetjp_162_;
}
else
{
lean_inc(v_snd_161_);
lean_inc(v_fst_160_);
lean_dec(v_q_159_);
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
v_reuseFailAlloc_167_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_167_, 0, v_fst_160_);
lean_ctor_set(v_reuseFailAlloc_167_, 1, v_snd_161_);
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
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_prodEquiv___lam__1(lean_object* v_q_169_){
_start:
{
lean_object* v_fst_170_; lean_object* v_snd_171_; lean_object* v___x_173_; uint8_t v_isShared_174_; uint8_t v_isSharedCheck_178_; 
v_fst_170_ = lean_ctor_get(v_q_169_, 0);
v_snd_171_ = lean_ctor_get(v_q_169_, 1);
v_isSharedCheck_178_ = !lean_is_exclusive(v_q_169_);
if (v_isSharedCheck_178_ == 0)
{
v___x_173_ = v_q_169_;
v_isShared_174_ = v_isSharedCheck_178_;
goto v_resetjp_172_;
}
else
{
lean_inc(v_snd_171_);
lean_inc(v_fst_170_);
lean_dec(v_q_169_);
v___x_173_ = lean_box(0);
v_isShared_174_ = v_isSharedCheck_178_;
goto v_resetjp_172_;
}
v_resetjp_172_:
{
lean_object* v___x_176_; 
if (v_isShared_174_ == 0)
{
v___x_176_ = v___x_173_;
goto v_reusejp_175_;
}
else
{
lean_object* v_reuseFailAlloc_177_; 
v_reuseFailAlloc_177_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_177_, 0, v_fst_170_);
lean_ctor_set(v_reuseFailAlloc_177_, 1, v_snd_171_);
v___x_176_ = v_reuseFailAlloc_177_;
goto v_reusejp_175_;
}
v_reusejp_175_:
{
return v___x_176_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_prodEquiv(lean_object* v_G_184_, lean_object* v_inst_185_, lean_object* v_H_186_, lean_object* v_inst_187_, lean_object* v_A_188_, lean_object* v_B_189_){
_start:
{
lean_object* v___x_190_; 
v___x_190_ = ((lean_object*)(lp_mathlib_QuotientGroup_prodEquiv___closed__2));
return v___x_190_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_prodEquiv___boxed(lean_object* v_G_191_, lean_object* v_inst_192_, lean_object* v_H_193_, lean_object* v_inst_194_, lean_object* v_A_195_, lean_object* v_B_196_){
_start:
{
lean_object* v_res_197_; 
v_res_197_ = lp_mathlib_QuotientGroup_prodEquiv(v_G_191_, v_inst_192_, v_H_193_, v_inst_194_, v_A_195_, v_B_196_);
lean_dec_ref(v_inst_194_);
lean_dec_ref(v_inst_192_);
return v_res_197_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_prodEquiv_match__1___redArg(lean_object* v_x_198_, lean_object* v_h__1_199_){
_start:
{
lean_object* v_fst_200_; lean_object* v_snd_201_; lean_object* v___x_202_; 
v_fst_200_ = lean_ctor_get(v_x_198_, 0);
lean_inc(v_fst_200_);
v_snd_201_ = lean_ctor_get(v_x_198_, 1);
lean_inc(v_snd_201_);
lean_dec_ref(v_x_198_);
v___x_202_ = lean_apply_2(v_h__1_199_, v_fst_200_, v_snd_201_);
return v___x_202_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_prodEquiv_match__1(lean_object* v_G_203_, lean_object* v_H_204_, lean_object* v_motive_205_, lean_object* v_x_206_, lean_object* v_h__1_207_){
_start:
{
lean_object* v___x_208_; 
v___x_208_ = lp_mathlib_QuotientAddGroup_prodEquiv_match__1___redArg(v_x_206_, v_h__1_207_);
return v___x_208_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_prodEquiv(lean_object* v_G_209_, lean_object* v_inst_210_, lean_object* v_H_211_, lean_object* v_inst_212_, lean_object* v_A_213_, lean_object* v_B_214_){
_start:
{
lean_object* v___x_215_; 
v___x_215_ = ((lean_object*)(lp_mathlib_QuotientGroup_prodEquiv___closed__2));
return v___x_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_prodEquiv___boxed(lean_object* v_G_216_, lean_object* v_inst_217_, lean_object* v_H_218_, lean_object* v_inst_219_, lean_object* v_A_220_, lean_object* v_B_221_){
_start:
{
lean_object* v_res_222_; 
v_res_222_ = lp_mathlib_QuotientAddGroup_prodEquiv(v_G_216_, v_inst_217_, v_H_218_, v_inst_219_, v_A_220_, v_B_221_);
lean_dec_ref(v_inst_219_);
lean_dec_ref(v_inst_217_);
return v_res_222_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_prodMulEquiv___redArg(lean_object* v_inst_223_, lean_object* v_inst_224_, lean_object* v_A_225_, lean_object* v_B_226_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lp_mathlib_QuotientGroup_prodEquiv(lean_box(0), v_inst_223_, lean_box(0), v_inst_224_, v_A_225_, v_B_226_);
return v___x_227_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_prodMulEquiv___redArg___boxed(lean_object* v_inst_228_, lean_object* v_inst_229_, lean_object* v_A_230_, lean_object* v_B_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_mathlib_QuotientGroup_prodMulEquiv___redArg(v_inst_228_, v_inst_229_, v_A_230_, v_B_231_);
lean_dec_ref(v_inst_229_);
lean_dec_ref(v_inst_228_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_prodMulEquiv(lean_object* v_G_233_, lean_object* v_inst_234_, lean_object* v_H_235_, lean_object* v_inst_236_, lean_object* v_A_237_, lean_object* v_B_238_, lean_object* v_inst_239_, lean_object* v_inst_240_){
_start:
{
lean_object* v___x_241_; 
v___x_241_ = lp_mathlib_QuotientGroup_prodEquiv(lean_box(0), v_inst_234_, lean_box(0), v_inst_236_, v_A_237_, v_B_238_);
return v___x_241_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_prodMulEquiv___boxed(lean_object* v_G_242_, lean_object* v_inst_243_, lean_object* v_H_244_, lean_object* v_inst_245_, lean_object* v_A_246_, lean_object* v_B_247_, lean_object* v_inst_248_, lean_object* v_inst_249_){
_start:
{
lean_object* v_res_250_; 
v_res_250_ = lp_mathlib_QuotientGroup_prodMulEquiv(v_G_242_, v_inst_243_, v_H_244_, v_inst_245_, v_A_246_, v_B_247_, v_inst_248_, v_inst_249_);
lean_dec_ref(v_inst_245_);
lean_dec_ref(v_inst_243_);
return v_res_250_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_prodAddEquiv___redArg(lean_object* v_inst_251_, lean_object* v_inst_252_, lean_object* v_A_253_, lean_object* v_B_254_){
_start:
{
lean_object* v___x_255_; 
v___x_255_ = lp_mathlib_QuotientAddGroup_prodEquiv(lean_box(0), v_inst_251_, lean_box(0), v_inst_252_, v_A_253_, v_B_254_);
return v___x_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_prodAddEquiv___redArg___boxed(lean_object* v_inst_256_, lean_object* v_inst_257_, lean_object* v_A_258_, lean_object* v_B_259_){
_start:
{
lean_object* v_res_260_; 
v_res_260_ = lp_mathlib_QuotientAddGroup_prodAddEquiv___redArg(v_inst_256_, v_inst_257_, v_A_258_, v_B_259_);
lean_dec_ref(v_inst_257_);
lean_dec_ref(v_inst_256_);
return v_res_260_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_prodAddEquiv(lean_object* v_G_261_, lean_object* v_inst_262_, lean_object* v_H_263_, lean_object* v_inst_264_, lean_object* v_A_265_, lean_object* v_B_266_, lean_object* v_inst_267_, lean_object* v_inst_268_){
_start:
{
lean_object* v___x_269_; 
v___x_269_ = lp_mathlib_QuotientAddGroup_prodEquiv(lean_box(0), v_inst_262_, lean_box(0), v_inst_264_, v_A_265_, v_B_266_);
return v___x_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_prodAddEquiv___boxed(lean_object* v_G_270_, lean_object* v_inst_271_, lean_object* v_H_272_, lean_object* v_inst_273_, lean_object* v_A_274_, lean_object* v_B_275_, lean_object* v_inst_276_, lean_object* v_inst_277_){
_start:
{
lean_object* v_res_278_; 
v_res_278_ = lp_mathlib_QuotientAddGroup_prodAddEquiv(v_G_270_, v_inst_271_, v_H_272_, v_inst_273_, v_A_274_, v_B_275_, v_inst_276_, v_inst_277_);
lean_dec_ref(v_inst_273_);
lean_dec_ref(v_inst_271_);
return v_res_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_kerLift___redArg(lean_object* v_00_u03c6_279_){
_start:
{
lean_object* v___f_280_; 
v___f_280_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_280_, 0, v_00_u03c6_279_);
return v___f_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_kerLift(lean_object* v_G_281_, lean_object* v_inst_282_, lean_object* v_H_283_, lean_object* v_inst_284_, lean_object* v_00_u03c6_285_){
_start:
{
lean_object* v___f_286_; 
v___f_286_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_286_, 0, v_00_u03c6_285_);
return v___f_286_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_kerLift___boxed(lean_object* v_G_287_, lean_object* v_inst_288_, lean_object* v_H_289_, lean_object* v_inst_290_, lean_object* v_00_u03c6_291_){
_start:
{
lean_object* v_res_292_; 
v_res_292_ = lp_mathlib_QuotientGroup_kerLift(v_G_287_, v_inst_288_, v_H_289_, v_inst_290_, v_00_u03c6_291_);
lean_dec_ref(v_inst_290_);
lean_dec_ref(v_inst_288_);
return v_res_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_kerLift___redArg(lean_object* v_00_u03c6_293_){
_start:
{
lean_object* v___f_294_; 
v___f_294_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_294_, 0, v_00_u03c6_293_);
return v___f_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_kerLift(lean_object* v_G_295_, lean_object* v_inst_296_, lean_object* v_H_297_, lean_object* v_inst_298_, lean_object* v_00_u03c6_299_){
_start:
{
lean_object* v___f_300_; 
v___f_300_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_300_, 0, v_00_u03c6_299_);
return v___f_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_kerLift___boxed(lean_object* v_G_301_, lean_object* v_inst_302_, lean_object* v_H_303_, lean_object* v_inst_304_, lean_object* v_00_u03c6_305_){
_start:
{
lean_object* v_res_306_; 
v_res_306_ = lp_mathlib_QuotientAddGroup_kerLift(v_G_301_, v_inst_302_, v_H_303_, v_inst_304_, v_00_u03c6_305_);
lean_dec_ref(v_inst_304_);
lean_dec_ref(v_inst_302_);
return v_res_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_rangeKerLift___redArg(lean_object* v_00_u03c6_307_){
_start:
{
lean_object* v___f_308_; lean_object* v___f_309_; 
v___f_308_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_308_, 0, v_00_u03c6_307_);
v___f_309_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_309_, 0, v___f_308_);
return v___f_309_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_rangeKerLift(lean_object* v_G_310_, lean_object* v_inst_311_, lean_object* v_H_312_, lean_object* v_inst_313_, lean_object* v_00_u03c6_314_){
_start:
{
lean_object* v___x_315_; 
v___x_315_ = lp_mathlib_QuotientGroup_rangeKerLift___redArg(v_00_u03c6_314_);
return v___x_315_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_rangeKerLift___boxed(lean_object* v_G_316_, lean_object* v_inst_317_, lean_object* v_H_318_, lean_object* v_inst_319_, lean_object* v_00_u03c6_320_){
_start:
{
lean_object* v_res_321_; 
v_res_321_ = lp_mathlib_QuotientGroup_rangeKerLift(v_G_316_, v_inst_317_, v_H_318_, v_inst_319_, v_00_u03c6_320_);
lean_dec_ref(v_inst_319_);
lean_dec_ref(v_inst_317_);
return v_res_321_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_rangeKerLift___redArg(lean_object* v_00_u03c6_322_){
_start:
{
lean_object* v___f_323_; lean_object* v___f_324_; 
v___f_323_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_codRestrict___redArg___lam__0), 2, 1);
lean_closure_set(v___f_323_, 0, v_00_u03c6_322_);
v___f_324_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_324_, 0, v___f_323_);
return v___f_324_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_rangeKerLift(lean_object* v_G_325_, lean_object* v_inst_326_, lean_object* v_H_327_, lean_object* v_inst_328_, lean_object* v_00_u03c6_329_){
_start:
{
lean_object* v___x_330_; 
v___x_330_ = lp_mathlib_QuotientAddGroup_rangeKerLift___redArg(v_00_u03c6_329_);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_rangeKerLift___boxed(lean_object* v_G_331_, lean_object* v_inst_332_, lean_object* v_H_333_, lean_object* v_inst_334_, lean_object* v_00_u03c6_335_){
_start:
{
lean_object* v_res_336_; 
v_res_336_ = lp_mathlib_QuotientAddGroup_rangeKerLift(v_G_331_, v_inst_332_, v_H_333_, v_inst_334_, v_00_u03c6_335_);
lean_dec_ref(v_inst_334_);
lean_dec_ref(v_inst_332_);
return v_res_336_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientKerEquivOfRightInverse___redArg___lam__0(lean_object* v_00_u03c6_337_, lean_object* v___y_338_){
_start:
{
lean_object* v___x_339_; 
v___x_339_ = lp_mathlib_Con_lift___redArg___lam__0(v_00_u03c6_337_, v___y_338_);
return v___x_339_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientKerEquivOfRightInverse___redArg(lean_object* v_inst_340_, lean_object* v_00_u03c6_341_, lean_object* v_00_u03c8_342_){
_start:
{
lean_object* v___f_343_; lean_object* v___x_344_; lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; 
v___f_343_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_quotientKerEquivOfRightInverse___redArg___lam__0), 2, 1);
lean_closure_set(v___f_343_, 0, v_00_u03c6_341_);
v___x_344_ = lean_box(0);
v___x_345_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_345_, 0, lean_box(0));
lean_closure_set(v___x_345_, 1, v_inst_340_);
lean_closure_set(v___x_345_, 2, v___x_344_);
v___x_346_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_346_, 0, lean_box(0));
lean_closure_set(v___x_346_, 1, lean_box(0));
lean_closure_set(v___x_346_, 2, lean_box(0));
lean_closure_set(v___x_346_, 3, v___x_345_);
lean_closure_set(v___x_346_, 4, v_00_u03c8_342_);
v___x_347_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_347_, 0, v___f_343_);
lean_ctor_set(v___x_347_, 1, v___x_346_);
return v___x_347_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientKerEquivOfRightInverse(lean_object* v_G_348_, lean_object* v_inst_349_, lean_object* v_H_350_, lean_object* v_inst_351_, lean_object* v_00_u03c6_352_, lean_object* v_00_u03c8_353_, lean_object* v_h_u03c6_354_){
_start:
{
lean_object* v___x_355_; 
v___x_355_ = lp_mathlib_QuotientGroup_quotientKerEquivOfRightInverse___redArg(v_inst_349_, v_00_u03c6_352_, v_00_u03c8_353_);
return v___x_355_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientKerEquivOfRightInverse___boxed(lean_object* v_G_356_, lean_object* v_inst_357_, lean_object* v_H_358_, lean_object* v_inst_359_, lean_object* v_00_u03c6_360_, lean_object* v_00_u03c8_361_, lean_object* v_h_u03c6_362_){
_start:
{
lean_object* v_res_363_; 
v_res_363_ = lp_mathlib_QuotientGroup_quotientKerEquivOfRightInverse(v_G_356_, v_inst_357_, v_H_358_, v_inst_359_, v_00_u03c6_360_, v_00_u03c8_361_, v_h_u03c6_362_);
lean_dec_ref(v_inst_359_);
return v_res_363_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientKerEquivOfRightInverse___redArg(lean_object* v_inst_364_, lean_object* v_00_u03c6_365_, lean_object* v_00_u03c8_366_){
_start:
{
lean_object* v___f_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; 
v___f_367_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_quotientKerEquivOfRightInverse___redArg___lam__0), 2, 1);
lean_closure_set(v___f_367_, 0, v_00_u03c6_365_);
v___x_368_ = lean_box(0);
v___x_369_ = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_369_, 0, lean_box(0));
lean_closure_set(v___x_369_, 1, v_inst_364_);
lean_closure_set(v___x_369_, 2, v___x_368_);
v___x_370_ = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(v___x_370_, 0, lean_box(0));
lean_closure_set(v___x_370_, 1, lean_box(0));
lean_closure_set(v___x_370_, 2, lean_box(0));
lean_closure_set(v___x_370_, 3, v___x_369_);
lean_closure_set(v___x_370_, 4, v_00_u03c8_366_);
v___x_371_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_371_, 0, v___f_367_);
lean_ctor_set(v___x_371_, 1, v___x_370_);
return v___x_371_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientKerEquivOfRightInverse(lean_object* v_G_372_, lean_object* v_inst_373_, lean_object* v_H_374_, lean_object* v_inst_375_, lean_object* v_00_u03c6_376_, lean_object* v_00_u03c8_377_, lean_object* v_h_u03c6_378_){
_start:
{
lean_object* v___x_379_; 
v___x_379_ = lp_mathlib_QuotientAddGroup_quotientKerEquivOfRightInverse___redArg(v_inst_373_, v_00_u03c6_376_, v_00_u03c8_377_);
return v___x_379_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientKerEquivOfRightInverse___boxed(lean_object* v_G_380_, lean_object* v_inst_381_, lean_object* v_H_382_, lean_object* v_inst_383_, lean_object* v_00_u03c6_384_, lean_object* v_00_u03c8_385_, lean_object* v_h_u03c6_386_){
_start:
{
lean_object* v_res_387_; 
v_res_387_ = lp_mathlib_QuotientAddGroup_quotientKerEquivOfRightInverse(v_G_380_, v_inst_381_, v_H_382_, v_inst_383_, v_00_u03c6_384_, v_00_u03c8_385_, v_h_u03c6_386_);
lean_dec_ref(v_inst_383_);
return v_res_387_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientBot___redArg(lean_object* v_inst_390_){
_start:
{
lean_object* v___f_391_; lean_object* v___x_392_; lean_object* v___x_393_; 
v___f_391_ = ((lean_object*)(lp_mathlib_QuotientGroup_quotientBot___redArg___closed__0));
v___x_392_ = ((lean_object*)(lp_mathlib_QuotientGroup_quotientBot___redArg___closed__1));
v___x_393_ = lp_mathlib_QuotientGroup_quotientKerEquivOfRightInverse___redArg(v_inst_390_, v___f_391_, v___x_392_);
return v___x_393_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientBot(lean_object* v_G_394_, lean_object* v_inst_395_){
_start:
{
lean_object* v___x_396_; 
v___x_396_ = lp_mathlib_QuotientGroup_quotientBot___redArg(v_inst_395_);
return v___x_396_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientBot___redArg(lean_object* v_inst_397_){
_start:
{
lean_object* v___f_398_; lean_object* v___x_399_; lean_object* v___x_400_; 
v___f_398_ = ((lean_object*)(lp_mathlib_QuotientGroup_quotientBot___redArg___closed__0));
v___x_399_ = ((lean_object*)(lp_mathlib_QuotientGroup_quotientBot___redArg___closed__1));
v___x_400_ = lp_mathlib_QuotientAddGroup_quotientKerEquivOfRightInverse___redArg(v_inst_397_, v___f_398_, v___x_399_);
return v___x_400_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientBot(lean_object* v_G_401_, lean_object* v_inst_402_){
_start:
{
lean_object* v___x_403_; 
v___x_403_ = lp_mathlib_QuotientAddGroup_quotientBot___redArg(v_inst_402_);
return v___x_403_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientMulEquivOfEq___redArg(lean_object* v_inst_404_, lean_object* v_M_405_, lean_object* v_N_406_){
_start:
{
lean_object* v___x_407_; 
v___x_407_ = lp_mathlib_Subgroup_quotientEquivOfEq(lean_box(0), v_inst_404_, v_M_405_, v_N_406_, lean_box(0));
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientMulEquivOfEq___redArg___boxed(lean_object* v_inst_408_, lean_object* v_M_409_, lean_object* v_N_410_){
_start:
{
lean_object* v_res_411_; 
v_res_411_ = lp_mathlib_QuotientGroup_quotientMulEquivOfEq___redArg(v_inst_408_, v_M_409_, v_N_410_);
lean_dec_ref(v_inst_408_);
return v_res_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientMulEquivOfEq(lean_object* v_G_412_, lean_object* v_inst_413_, lean_object* v_M_414_, lean_object* v_N_415_, lean_object* v_inst_416_, lean_object* v_inst_417_, lean_object* v_h_418_){
_start:
{
lean_object* v___x_419_; 
v___x_419_ = lp_mathlib_Subgroup_quotientEquivOfEq(lean_box(0), v_inst_413_, v_M_414_, v_N_415_, lean_box(0));
return v___x_419_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientMulEquivOfEq___boxed(lean_object* v_G_420_, lean_object* v_inst_421_, lean_object* v_M_422_, lean_object* v_N_423_, lean_object* v_inst_424_, lean_object* v_inst_425_, lean_object* v_h_426_){
_start:
{
lean_object* v_res_427_; 
v_res_427_ = lp_mathlib_QuotientGroup_quotientMulEquivOfEq(v_G_420_, v_inst_421_, v_M_422_, v_N_423_, v_inst_424_, v_inst_425_, v_h_426_);
lean_dec_ref(v_inst_421_);
return v_res_427_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientAddEquivOfEq___redArg(lean_object* v_inst_428_, lean_object* v_M_429_, lean_object* v_N_430_){
_start:
{
lean_object* v___x_431_; 
v___x_431_ = lp_mathlib_AddSubgroup_quotientEquivOfEq(lean_box(0), v_inst_428_, v_M_429_, v_N_430_, lean_box(0));
return v___x_431_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientAddEquivOfEq___redArg___boxed(lean_object* v_inst_432_, lean_object* v_M_433_, lean_object* v_N_434_){
_start:
{
lean_object* v_res_435_; 
v_res_435_ = lp_mathlib_QuotientAddGroup_quotientAddEquivOfEq___redArg(v_inst_432_, v_M_433_, v_N_434_);
lean_dec_ref(v_inst_432_);
return v_res_435_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientAddEquivOfEq(lean_object* v_G_436_, lean_object* v_inst_437_, lean_object* v_M_438_, lean_object* v_N_439_, lean_object* v_inst_440_, lean_object* v_inst_441_, lean_object* v_h_442_){
_start:
{
lean_object* v___x_443_; 
v___x_443_ = lp_mathlib_AddSubgroup_quotientEquivOfEq(lean_box(0), v_inst_437_, v_M_438_, v_N_439_, lean_box(0));
return v___x_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientAddEquivOfEq___boxed(lean_object* v_G_444_, lean_object* v_inst_445_, lean_object* v_M_446_, lean_object* v_N_447_, lean_object* v_inst_448_, lean_object* v_inst_449_, lean_object* v_h_450_){
_start:
{
lean_object* v_res_451_; 
v_res_451_ = lp_mathlib_QuotientAddGroup_quotientAddEquivOfEq(v_G_444_, v_inst_445_, v_M_446_, v_N_447_, v_inst_448_, v_inst_449_, v_h_450_);
lean_dec_ref(v_inst_445_);
return v_res_451_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientMapSubgroupOfOfLe___redArg(lean_object* v_inst_453_){
_start:
{
lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___f_456_; lean_object* v___x_457_; 
v___x_454_ = lp_mathlib_SubgroupClass_toGroup___redArg(v_inst_453_);
v___x_455_ = lean_box(0);
v___f_456_ = ((lean_object*)(lp_mathlib_QuotientGroup_quotientMapSubgroupOfOfLe___redArg___closed__0));
v___x_457_ = lp_mathlib_QuotientGroup_map___redArg(v___x_454_, v___x_455_, v___f_456_);
return v___x_457_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientMapSubgroupOfOfLe(lean_object* v_G_458_, lean_object* v_inst_459_, lean_object* v_A_x27_460_, lean_object* v_A_461_, lean_object* v_B_x27_462_, lean_object* v_B_463_, lean_object* v___hAN_464_, lean_object* v___hBN_465_, lean_object* v_h_x27_466_, lean_object* v_h_467_){
_start:
{
lean_object* v___x_468_; 
v___x_468_ = lp_mathlib_QuotientGroup_quotientMapSubgroupOfOfLe___redArg(v_inst_459_);
return v___x_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientMapAddSubgroupOfOfLe___redArg(lean_object* v_inst_469_){
_start:
{
lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___f_472_; lean_object* v___x_473_; 
v___x_470_ = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(v_inst_469_);
v___x_471_ = lean_box(0);
v___f_472_ = ((lean_object*)(lp_mathlib_QuotientGroup_quotientMapSubgroupOfOfLe___redArg___closed__0));
v___x_473_ = lp_mathlib_QuotientAddGroup_map___redArg(v___x_470_, v___x_471_, v___f_472_);
return v___x_473_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientMapAddSubgroupOfOfLe(lean_object* v_G_474_, lean_object* v_inst_475_, lean_object* v_A_x27_476_, lean_object* v_A_477_, lean_object* v_B_x27_478_, lean_object* v_B_479_, lean_object* v___hAN_480_, lean_object* v___hBN_481_, lean_object* v_h_x27_482_, lean_object* v_h_483_){
_start:
{
lean_object* v___x_484_; 
v___x_484_ = lp_mathlib_QuotientAddGroup_quotientMapAddSubgroupOfOfLe___redArg(v_inst_475_);
return v___x_484_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_equivQuotientSubgroupOfOfEq___redArg(lean_object* v_inst_485_){
_start:
{
lean_object* v___x_486_; lean_object* v___x_487_; 
v___x_486_ = lp_mathlib_QuotientGroup_quotientMapSubgroupOfOfLe___redArg(v_inst_485_);
lean_inc(v___x_486_);
v___x_487_ = lp_mathlib_MonoidHom_toMulEquiv___redArg(v___x_486_, v___x_486_);
return v___x_487_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_equivQuotientSubgroupOfOfEq(lean_object* v_G_488_, lean_object* v_inst_489_, lean_object* v_A_x27_490_, lean_object* v_A_491_, lean_object* v_B_x27_492_, lean_object* v_B_493_, lean_object* v_hAN_494_, lean_object* v_hBN_495_, lean_object* v_h_x27_496_, lean_object* v_h_497_){
_start:
{
lean_object* v___x_498_; 
v___x_498_ = lp_mathlib_QuotientGroup_equivQuotientSubgroupOfOfEq___redArg(v_inst_489_);
return v___x_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_equivQuotientAddSubgroupOfOfEq___redArg(lean_object* v_inst_499_){
_start:
{
lean_object* v___x_500_; lean_object* v___x_501_; 
v___x_500_ = lp_mathlib_QuotientAddGroup_quotientMapAddSubgroupOfOfLe___redArg(v_inst_499_);
lean_inc(v___x_500_);
v___x_501_ = lp_mathlib_AddMonoidHom_toAddEquiv___redArg(v___x_500_, v___x_500_);
return v___x_501_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_equivQuotientAddSubgroupOfOfEq(lean_object* v_G_502_, lean_object* v_inst_503_, lean_object* v_A_x27_504_, lean_object* v_A_505_, lean_object* v_B_x27_506_, lean_object* v_B_507_, lean_object* v_hAN_508_, lean_object* v_hBN_509_, lean_object* v_h_x27_510_, lean_object* v_h_511_){
_start:
{
lean_object* v___x_512_; 
v___x_512_ = lp_mathlib_QuotientAddGroup_equivQuotientAddSubgroupOfOfEq___redArg(v_inst_503_);
return v___x_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_homQuotientZPowOfHom___redArg(lean_object* v_inst_513_, lean_object* v_f_514_){
_start:
{
lean_object* v___x_515_; lean_object* v___x_516_; lean_object* v___f_517_; lean_object* v___f_518_; 
v___x_515_ = lean_box(0);
v___x_516_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_516_, 0, lean_box(0));
lean_closure_set(v___x_516_, 1, v_inst_513_);
lean_closure_set(v___x_516_, 2, v___x_515_);
v___f_517_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_517_, 0, v_f_514_);
lean_closure_set(v___f_517_, 1, v___x_516_);
v___f_518_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_518_, 0, v___f_517_);
return v___f_518_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_homQuotientZPowOfHom(lean_object* v_A_519_, lean_object* v_B_520_, lean_object* v_inst_521_, lean_object* v_inst_522_, lean_object* v_f_523_, lean_object* v_n_524_){
_start:
{
lean_object* v___x_525_; 
v___x_525_ = lp_mathlib_QuotientGroup_homQuotientZPowOfHom___redArg(v_inst_522_, v_f_523_);
return v___x_525_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_homQuotientZPowOfHom___boxed(lean_object* v_A_526_, lean_object* v_B_527_, lean_object* v_inst_528_, lean_object* v_inst_529_, lean_object* v_f_530_, lean_object* v_n_531_){
_start:
{
lean_object* v_res_532_; 
v_res_532_ = lp_mathlib_QuotientGroup_homQuotientZPowOfHom(v_A_526_, v_B_527_, v_inst_528_, v_inst_529_, v_f_530_, v_n_531_);
lean_dec(v_n_531_);
lean_dec_ref(v_inst_528_);
return v_res_532_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_homQuotientZSMulOfHom___redArg(lean_object* v_inst_533_, lean_object* v_f_534_){
_start:
{
lean_object* v___x_535_; lean_object* v___x_536_; lean_object* v___f_537_; lean_object* v___f_538_; 
v___x_535_ = lean_box(0);
v___x_536_ = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_536_, 0, lean_box(0));
lean_closure_set(v___x_536_, 1, v_inst_533_);
lean_closure_set(v___x_536_, 2, v___x_535_);
v___f_537_ = lean_alloc_closure((void*)(lp_mathlib_OneHom_comp___redArg___lam__0), 3, 2);
lean_closure_set(v___f_537_, 0, v_f_534_);
lean_closure_set(v___f_537_, 1, v___x_536_);
v___f_538_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_538_, 0, v___f_537_);
return v___f_538_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_homQuotientZSMulOfHom(lean_object* v_A_539_, lean_object* v_B_540_, lean_object* v_inst_541_, lean_object* v_inst_542_, lean_object* v_f_543_, lean_object* v_n_544_){
_start:
{
lean_object* v___x_545_; 
v___x_545_ = lp_mathlib_QuotientAddGroup_homQuotientZSMulOfHom___redArg(v_inst_542_, v_f_543_);
return v___x_545_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_homQuotientZSMulOfHom___boxed(lean_object* v_A_546_, lean_object* v_B_547_, lean_object* v_inst_548_, lean_object* v_inst_549_, lean_object* v_f_550_, lean_object* v_n_551_){
_start:
{
lean_object* v_res_552_; 
v_res_552_ = lp_mathlib_QuotientAddGroup_homQuotientZSMulOfHom(v_A_546_, v_B_547_, v_inst_548_, v_inst_549_, v_f_550_, v_n_551_);
lean_dec(v_n_551_);
lean_dec_ref(v_inst_548_);
return v_res_552_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_equivQuotientZPowOfEquiv___redArg___lam__0(lean_object* v_e_553_, lean_object* v___y_554_){
_start:
{
lean_object* v_toFun_555_; lean_object* v___x_556_; 
v_toFun_555_ = lean_ctor_get(v_e_553_, 0);
lean_inc(v_toFun_555_);
lean_dec_ref(v_e_553_);
v___x_556_ = lean_apply_1(v_toFun_555_, v___y_554_);
return v___x_556_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_equivQuotientZPowOfEquiv___redArg___lam__1(lean_object* v___x_557_, lean_object* v___y_558_){
_start:
{
lean_object* v_toFun_559_; lean_object* v___x_560_; 
v_toFun_559_ = lean_ctor_get(v___x_557_, 0);
lean_inc(v_toFun_559_);
lean_dec_ref(v___x_557_);
v___x_560_ = lean_apply_1(v_toFun_559_, v___y_558_);
return v___x_560_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_equivQuotientZPowOfEquiv___redArg(lean_object* v_inst_561_, lean_object* v_inst_562_, lean_object* v_e_563_){
_start:
{
lean_object* v___f_564_; lean_object* v___x_565_; lean_object* v___x_566_; lean_object* v___f_567_; lean_object* v___x_568_; lean_object* v___x_569_; 
lean_inc_ref(v_e_563_);
v___f_564_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_equivQuotientZPowOfEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_564_, 0, v_e_563_);
v___x_565_ = lp_mathlib_QuotientGroup_homQuotientZPowOfHom___redArg(v_inst_562_, v___f_564_);
v___x_566_ = lp_mathlib_Equiv_symm___redArg(v_e_563_);
v___f_567_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_equivQuotientZPowOfEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_567_, 0, v___x_566_);
v___x_568_ = lp_mathlib_QuotientGroup_homQuotientZPowOfHom___redArg(v_inst_561_, v___f_567_);
v___x_569_ = lp_mathlib_MonoidHom_toMulEquiv___redArg(v___x_565_, v___x_568_);
return v___x_569_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_equivQuotientZPowOfEquiv(lean_object* v_A_570_, lean_object* v_B_571_, lean_object* v_inst_572_, lean_object* v_inst_573_, lean_object* v_e_574_, lean_object* v_n_575_){
_start:
{
lean_object* v___x_576_; 
v___x_576_ = lp_mathlib_QuotientGroup_equivQuotientZPowOfEquiv___redArg(v_inst_572_, v_inst_573_, v_e_574_);
return v___x_576_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_equivQuotientZPowOfEquiv___boxed(lean_object* v_A_577_, lean_object* v_B_578_, lean_object* v_inst_579_, lean_object* v_inst_580_, lean_object* v_e_581_, lean_object* v_n_582_){
_start:
{
lean_object* v_res_583_; 
v_res_583_ = lp_mathlib_QuotientGroup_equivQuotientZPowOfEquiv(v_A_577_, v_B_578_, v_inst_579_, v_inst_580_, v_e_581_, v_n_582_);
lean_dec(v_n_582_);
return v_res_583_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_equivQuotientZSMulOfEquiv___redArg(lean_object* v_inst_584_, lean_object* v_inst_585_, lean_object* v_e_586_){
_start:
{
lean_object* v___f_587_; lean_object* v___x_588_; lean_object* v___x_589_; lean_object* v___f_590_; lean_object* v___x_591_; lean_object* v___x_592_; 
lean_inc_ref(v_e_586_);
v___f_587_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_equivQuotientZPowOfEquiv___redArg___lam__0), 2, 1);
lean_closure_set(v___f_587_, 0, v_e_586_);
v___x_588_ = lp_mathlib_QuotientAddGroup_homQuotientZSMulOfHom___redArg(v_inst_585_, v___f_587_);
v___x_589_ = lp_mathlib_Equiv_symm___redArg(v_e_586_);
v___f_590_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_equivQuotientZPowOfEquiv___redArg___lam__1), 2, 1);
lean_closure_set(v___f_590_, 0, v___x_589_);
v___x_591_ = lp_mathlib_QuotientAddGroup_homQuotientZSMulOfHom___redArg(v_inst_584_, v___f_590_);
v___x_592_ = lp_mathlib_AddMonoidHom_toAddEquiv___redArg(v___x_588_, v___x_591_);
return v___x_592_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_equivQuotientZSMulOfEquiv(lean_object* v_A_593_, lean_object* v_B_594_, lean_object* v_inst_595_, lean_object* v_inst_596_, lean_object* v_e_597_, lean_object* v_n_598_){
_start:
{
lean_object* v___x_599_; 
v___x_599_ = lp_mathlib_QuotientAddGroup_equivQuotientZSMulOfEquiv___redArg(v_inst_595_, v_inst_596_, v_e_597_);
return v___x_599_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_equivQuotientZSMulOfEquiv___boxed(lean_object* v_A_600_, lean_object* v_B_601_, lean_object* v_inst_602_, lean_object* v_inst_603_, lean_object* v_e_604_, lean_object* v_n_605_){
_start:
{
lean_object* v_res_606_; 
v_res_606_ = lp_mathlib_QuotientAddGroup_equivQuotientZSMulOfEquiv(v_A_600_, v_B_601_, v_inst_602_, v_inst_603_, v_e_604_, v_n_605_);
lean_dec(v_n_605_);
return v_res_606_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientQuotientEquivQuotientAux___redArg(lean_object* v_inst_607_, lean_object* v_M_608_){
_start:
{
lean_object* v___f_609_; lean_object* v___x_610_; lean_object* v___f_611_; 
v___f_609_ = ((lean_object*)(lp_mathlib_QuotientGroup_quotientBot___redArg___closed__0));
v___x_610_ = lp_mathlib_QuotientGroup_map___redArg(v_inst_607_, v_M_608_, v___f_609_);
v___f_611_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_611_, 0, v___x_610_);
return v___f_611_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientQuotientEquivQuotientAux(lean_object* v_G_612_, lean_object* v_inst_613_, lean_object* v_N_614_, lean_object* v_nN_615_, lean_object* v_M_616_, lean_object* v_nM_617_, lean_object* v_h_618_){
_start:
{
lean_object* v___x_619_; 
v___x_619_ = lp_mathlib_QuotientGroup_quotientQuotientEquivQuotientAux___redArg(v_inst_613_, v_M_616_);
return v___x_619_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientQuotientEquivQuotientAux___redArg(lean_object* v_inst_620_, lean_object* v_M_621_){
_start:
{
lean_object* v___f_622_; lean_object* v___x_623_; lean_object* v___f_624_; 
v___f_622_ = ((lean_object*)(lp_mathlib_QuotientGroup_quotientBot___redArg___closed__0));
v___x_623_ = lp_mathlib_QuotientAddGroup_map___redArg(v_inst_620_, v_M_621_, v___f_622_);
v___f_624_ = lean_alloc_closure((void*)(lp_mathlib_Con_lift___redArg___lam__0), 2, 1);
lean_closure_set(v___f_624_, 0, v___x_623_);
return v___f_624_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientQuotientEquivQuotientAux(lean_object* v_G_625_, lean_object* v_inst_626_, lean_object* v_N_627_, lean_object* v_nN_628_, lean_object* v_M_629_, lean_object* v_nM_630_, lean_object* v_h_631_){
_start:
{
lean_object* v___x_632_; 
v___x_632_ = lp_mathlib_QuotientAddGroup_quotientQuotientEquivQuotientAux___redArg(v_inst_626_, v_M_629_);
return v___x_632_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientQuotientEquivQuotient___redArg(lean_object* v_inst_633_, lean_object* v_N_634_, lean_object* v_M_635_){
_start:
{
lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; 
lean_inc_ref_n(v_inst_633_, 2);
v___x_636_ = lp_mathlib_QuotientGroup_Quotient_group___redArg(v_inst_633_, v_N_634_);
v___x_637_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_637_, 0, lean_box(0));
lean_closure_set(v___x_637_, 1, v_inst_633_);
lean_closure_set(v___x_637_, 2, v_N_634_);
v___x_638_ = lean_box(0);
v___x_639_ = lp_mathlib_QuotientGroup_quotientQuotientEquivQuotientAux___redArg(v_inst_633_, v_M_635_);
v___x_640_ = lp_mathlib_QuotientGroup_map___redArg(v___x_636_, v___x_638_, v___x_637_);
v___x_641_ = lp_mathlib_MonoidHom_toMulEquiv___redArg(v___x_639_, v___x_640_);
return v___x_641_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_quotientQuotientEquivQuotient(lean_object* v_G_642_, lean_object* v_inst_643_, lean_object* v_N_644_, lean_object* v_nN_645_, lean_object* v_M_646_, lean_object* v_nM_647_, lean_object* v_h_648_){
_start:
{
lean_object* v___x_649_; 
v___x_649_ = lp_mathlib_QuotientGroup_quotientQuotientEquivQuotient___redArg(v_inst_643_, v_N_644_, v_M_646_);
return v___x_649_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientQuotientEquivQuotient___redArg(lean_object* v_inst_650_, lean_object* v_N_651_, lean_object* v_M_652_){
_start:
{
lean_object* v___x_653_; lean_object* v___x_654_; lean_object* v___x_655_; lean_object* v___x_656_; lean_object* v___x_657_; lean_object* v___x_658_; 
lean_inc_ref_n(v_inst_650_, 2);
v___x_653_ = lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg(v_inst_650_, v_N_651_);
v___x_654_ = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_654_, 0, lean_box(0));
lean_closure_set(v___x_654_, 1, v_inst_650_);
lean_closure_set(v___x_654_, 2, v_N_651_);
v___x_655_ = lean_box(0);
v___x_656_ = lp_mathlib_QuotientAddGroup_quotientQuotientEquivQuotientAux___redArg(v_inst_650_, v_M_652_);
v___x_657_ = lp_mathlib_QuotientAddGroup_map___redArg(v___x_653_, v___x_655_, v___x_654_);
v___x_658_ = lp_mathlib_AddMonoidHom_toAddEquiv___redArg(v___x_656_, v___x_657_);
return v___x_658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_quotientQuotientEquivQuotient(lean_object* v_G_659_, lean_object* v_inst_660_, lean_object* v_N_661_, lean_object* v_nN_662_, lean_object* v_M_663_, lean_object* v_nM_664_, lean_object* v_h_665_){
_start:
{
lean_object* v___x_666_; 
v___x_666_ = lp_mathlib_QuotientAddGroup_quotientQuotientEquivQuotient___redArg(v_inst_660_, v_N_661_, v_M_663_);
return v___x_666_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_comapMk_x27OrderIso___lam__0(lean_object* v_H_x27_667_){
_start:
{
lean_object* v___x_668_; 
v___x_668_ = lean_box(0);
return v___x_668_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_comapMk_x27OrderIso___lam__1(lean_object* v_H_669_){
_start:
{
lean_object* v___x_670_; 
v___x_670_ = lean_box(0);
return v___x_670_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_comapMk_x27OrderIso(lean_object* v_G_676_, lean_object* v_inst_677_, lean_object* v_N_678_, lean_object* v_hn_679_){
_start:
{
lean_object* v___x_680_; 
v___x_680_ = ((lean_object*)(lp_mathlib_QuotientGroup_comapMk_x27OrderIso___closed__2));
return v___x_680_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientGroup_comapMk_x27OrderIso___boxed(lean_object* v_G_681_, lean_object* v_inst_682_, lean_object* v_N_683_, lean_object* v_hn_684_){
_start:
{
lean_object* v_res_685_; 
v_res_685_ = lp_mathlib_QuotientGroup_comapMk_x27OrderIso(v_G_681_, v_inst_682_, v_N_683_, v_hn_684_);
lean_dec_ref(v_inst_682_);
return v_res_685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___lam__0(lean_object* v_H_x27_686_){
_start:
{
lean_object* v___x_687_; 
v___x_687_ = lean_box(0);
return v___x_687_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___lam__1(lean_object* v_H_688_){
_start:
{
lean_object* v___x_689_; 
v___x_689_ = lean_box(0);
return v___x_689_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso(lean_object* v_G_695_, lean_object* v_inst_696_, lean_object* v_N_697_, lean_object* v_hn_698_){
_start:
{
lean_object* v___x_699_; 
v___x_699_ = ((lean_object*)(lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___closed__2));
return v___x_699_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso___boxed(lean_object* v_G_700_, lean_object* v_inst_701_, lean_object* v_N_702_, lean_object* v_hn_703_){
_start:
{
lean_object* v_res_704_; 
v_res_704_ = lp_mathlib_QuotientAddGroup_comapMk_x27OrderIso(v_G_700_, v_inst_701_, v_N_702_, v_hn_703_);
lean_dec_ref(v_inst_701_);
return v_res_704_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrictHomKerEquiv___redArg___lam__0(lean_object* v_inst_705_, lean_object* v_H_706_, lean_object* v_f_707_, lean_object* v___y_708_){
_start:
{
lean_object* v___x_709_; lean_object* v___x_710_; 
v___x_709_ = lean_alloc_closure((void*)(lp_mathlib_QuotientGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_709_, 0, lean_box(0));
lean_closure_set(v___x_709_, 1, v_inst_705_);
lean_closure_set(v___x_709_, 2, v_H_706_);
v___x_710_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___x_709_, v_f_707_, v___y_708_);
return v___x_710_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrictHomKerEquiv___redArg(lean_object* v_inst_712_, lean_object* v_H_713_){
_start:
{
lean_object* v___f_714_; lean_object* v___f_715_; lean_object* v___x_716_; 
v___f_714_ = ((lean_object*)(lp_mathlib_MonoidHom_domRestrictHomKerEquiv___redArg___closed__0));
v___f_715_ = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_domRestrictHomKerEquiv___redArg___lam__0), 4, 2);
lean_closure_set(v___f_715_, 0, v_inst_712_);
lean_closure_set(v___f_715_, 1, v_H_713_);
v___x_716_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_716_, 0, v___f_714_);
lean_ctor_set(v___x_716_, 1, v___f_715_);
return v___x_716_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrictHomKerEquiv(lean_object* v_G_717_, lean_object* v_inst_718_, lean_object* v_A_719_, lean_object* v_inst_720_, lean_object* v_H_721_, lean_object* v_inst_722_){
_start:
{
lean_object* v___x_723_; 
v___x_723_ = lp_mathlib_MonoidHom_domRestrictHomKerEquiv___redArg(v_inst_718_, v_H_721_);
return v___x_723_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_domRestrictHomKerEquiv___boxed(lean_object* v_G_724_, lean_object* v_inst_725_, lean_object* v_A_726_, lean_object* v_inst_727_, lean_object* v_H_728_, lean_object* v_inst_729_){
_start:
{
lean_object* v_res_730_; 
v_res_730_ = lp_mathlib_MonoidHom_domRestrictHomKerEquiv(v_G_724_, v_inst_725_, v_A_726_, v_inst_727_, v_H_728_, v_inst_729_);
lean_dec_ref(v_inst_727_);
return v_res_730_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv_match__1___redArg(lean_object* v_x_731_, lean_object* v_h__1_732_){
_start:
{
lean_object* v___x_733_; 
v___x_733_ = lean_apply_2(v_h__1_732_, v_x_731_, lean_box(0));
return v___x_733_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv_match__1(lean_object* v_G_734_, lean_object* v_inst_735_, lean_object* v_A_736_, lean_object* v_inst_737_, lean_object* v_H_738_, lean_object* v_motive_739_, lean_object* v_x_740_, lean_object* v_h__1_741_){
_start:
{
lean_object* v___x_742_; 
v___x_742_ = lean_apply_2(v_h__1_741_, v_x_740_, lean_box(0));
return v___x_742_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv_match__1___boxed(lean_object* v_G_743_, lean_object* v_inst_744_, lean_object* v_A_745_, lean_object* v_inst_746_, lean_object* v_H_747_, lean_object* v_motive_748_, lean_object* v_x_749_, lean_object* v_h__1_750_){
_start:
{
lean_object* v_res_751_; 
v_res_751_ = lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv_match__1(v_G_743_, v_inst_744_, v_A_745_, v_inst_746_, v_H_747_, v_motive_748_, v_x_749_, v_h__1_750_);
lean_dec_ref(v_inst_746_);
lean_dec_ref(v_inst_744_);
return v_res_751_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv___redArg___lam__0(lean_object* v_inst_752_, lean_object* v_H_753_, lean_object* v_f_754_, lean_object* v___y_755_){
_start:
{
lean_object* v___x_756_; lean_object* v___x_757_; 
v___x_756_ = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_mk___boxed), 4, 3);
lean_closure_set(v___x_756_, 0, lean_box(0));
lean_closure_set(v___x_756_, 1, v_inst_752_);
lean_closure_set(v___x_756_, 2, v_H_753_);
v___x_757_ = lp_mathlib_OneHom_comp___redArg___lam__0(v___x_756_, v_f_754_, v___y_755_);
return v___x_757_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv___redArg(lean_object* v_inst_758_, lean_object* v_H_759_){
_start:
{
lean_object* v___f_760_; lean_object* v___f_761_; lean_object* v___x_762_; 
v___f_760_ = ((lean_object*)(lp_mathlib_MonoidHom_domRestrictHomKerEquiv___redArg___closed__0));
v___f_761_ = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv___redArg___lam__0), 4, 2);
lean_closure_set(v___f_761_, 0, v_inst_758_);
lean_closure_set(v___f_761_, 1, v_H_759_);
v___x_762_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_762_, 0, v___f_760_);
lean_ctor_set(v___x_762_, 1, v___f_761_);
return v___x_762_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv(lean_object* v_G_763_, lean_object* v_inst_764_, lean_object* v_A_765_, lean_object* v_inst_766_, lean_object* v_H_767_, lean_object* v_inst_768_){
_start:
{
lean_object* v___x_769_; 
v___x_769_ = lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv___redArg(v_inst_764_, v_H_767_);
return v___x_769_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv___boxed(lean_object* v_G_770_, lean_object* v_inst_771_, lean_object* v_A_772_, lean_object* v_inst_773_, lean_object* v_H_774_, lean_object* v_inst_775_){
_start:
{
lean_object* v_res_776_; 
v_res_776_ = lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv(v_G_770_, v_inst_771_, v_A_772_, v_inst_773_, v_H_774_, v_inst_775_);
lean_dec_ref(v_inst_773_);
return v_res_776_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_restrictHomKerEquiv___redArg(lean_object* v_inst_777_, lean_object* v_H_778_){
_start:
{
lean_object* v___x_779_; 
v___x_779_ = lp_mathlib_MonoidHom_domRestrictHomKerEquiv___redArg(v_inst_777_, v_H_778_);
return v___x_779_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_restrictHomKerEquiv(lean_object* v_G_780_, lean_object* v_inst_781_, lean_object* v_A_782_, lean_object* v_inst_783_, lean_object* v_H_784_, lean_object* v_inst_785_){
_start:
{
lean_object* v___x_786_; 
v___x_786_ = lp_mathlib_MonoidHom_domRestrictHomKerEquiv___redArg(v_inst_781_, v_H_784_);
return v___x_786_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_restrictHomKerEquiv___boxed(lean_object* v_G_787_, lean_object* v_inst_788_, lean_object* v_A_789_, lean_object* v_inst_790_, lean_object* v_H_791_, lean_object* v_inst_792_){
_start:
{
lean_object* v_res_793_; 
v_res_793_ = lp_mathlib_MonoidHom_restrictHomKerEquiv(v_G_787_, v_inst_788_, v_A_789_, v_inst_790_, v_H_791_, v_inst_792_);
lean_dec_ref(v_inst_790_);
return v_res_793_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_restrictHomKerEquiv___redArg(lean_object* v_inst_794_, lean_object* v_H_795_){
_start:
{
lean_object* v___x_796_; 
v___x_796_ = lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv___redArg(v_inst_794_, v_H_795_);
return v___x_796_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_restrictHomKerEquiv(lean_object* v_G_797_, lean_object* v_inst_798_, lean_object* v_A_799_, lean_object* v_inst_800_, lean_object* v_H_801_, lean_object* v_inst_802_){
_start:
{
lean_object* v___x_803_; 
v___x_803_ = lp_mathlib_AddMonoidHom_domRestrictHomKerEquiv___redArg(v_inst_798_, v_H_801_);
return v___x_803_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_restrictHomKerEquiv___boxed(lean_object* v_G_804_, lean_object* v_inst_805_, lean_object* v_A_806_, lean_object* v_inst_807_, lean_object* v_H_808_, lean_object* v_inst_809_){
_start:
{
lean_object* v_res_810_; 
v_res_810_ = lp_mathlib_AddMonoidHom_restrictHomKerEquiv(v_G_804_, v_inst_805_, v_A_806_, v_inst_807_, v_H_808_, v_inst_809_);
lean_dec_ref(v_inst_807_);
return v_res_810_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Pointwise(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Pointwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Subgroup_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Cast_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Coset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_Finset_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
