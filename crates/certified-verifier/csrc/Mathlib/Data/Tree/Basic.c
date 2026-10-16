// Lean compiler output
// Module: Mathlib.Data.Tree.Basic
// Imports: public import Init public meta import Init public import Mathlib.Data.Nat.Notation public import Mathlib.Util.CompileInductive import Batteries.Tactic.Alias
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
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Syntax_isOfKind(lean_object*, lean_object*);
lean_object* l_Lean_Syntax_getArg(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_String_toRawSubstring_x27(lean_object*);
lean_object* l_Lean_addMacroScope(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node2(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_ctorIdx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_ctorIdx___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_ctorIdx(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_ctorIdx___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_ctorElim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_ctorElim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_nil_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_nil_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_node_elim___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_node_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqBinaryTree_decEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqBinaryTree_decEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqBinaryTree_decEq(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqBinaryTree_decEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqBinaryTree___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqBinaryTree___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqBinaryTree(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqBinaryTree___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_instReprBinaryTree_repr___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "BinaryTree.nil"};
static const lean_object* lp_mathlib_instReprBinaryTree_repr___redArg___closed__0 = (const lean_object*)&lp_mathlib_instReprBinaryTree_repr___redArg___closed__0_value;
static const lean_ctor_object lp_mathlib_instReprBinaryTree_repr___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_instReprBinaryTree_repr___redArg___closed__0_value)}};
static const lean_object* lp_mathlib_instReprBinaryTree_repr___redArg___closed__1 = (const lean_object*)&lp_mathlib_instReprBinaryTree_repr___redArg___closed__1_value;
static lean_once_cell_t lp_mathlib_instReprBinaryTree_repr___redArg___closed__2_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instReprBinaryTree_repr___redArg___closed__2;
static lean_once_cell_t lp_mathlib_instReprBinaryTree_repr___redArg___closed__3_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_instReprBinaryTree_repr___redArg___closed__3;
static const lean_string_object lp_mathlib_instReprBinaryTree_repr___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "BinaryTree.node"};
static const lean_object* lp_mathlib_instReprBinaryTree_repr___redArg___closed__4 = (const lean_object*)&lp_mathlib_instReprBinaryTree_repr___redArg___closed__4_value;
static const lean_ctor_object lp_mathlib_instReprBinaryTree_repr___redArg___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_mathlib_instReprBinaryTree_repr___redArg___closed__4_value)}};
static const lean_object* lp_mathlib_instReprBinaryTree_repr___redArg___closed__5 = (const lean_object*)&lp_mathlib_instReprBinaryTree_repr___redArg___closed__5_value;
static const lean_ctor_object lp_mathlib_instReprBinaryTree_repr___redArg___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_instReprBinaryTree_repr___redArg___closed__5_value),((lean_object*)(((size_t)(1) << 1) | 1))}};
static const lean_object* lp_mathlib_instReprBinaryTree_repr___redArg___closed__6 = (const lean_object*)&lp_mathlib_instReprBinaryTree_repr___redArg___closed__6_value;
LEAN_EXPORT lean_object* lp_mathlib_instReprBinaryTree_repr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instReprBinaryTree_repr___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instReprBinaryTree_repr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instReprBinaryTree_repr___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instReprBinaryTree___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instReprBinaryTree(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_rec___redArg_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_rec___redArg_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3____boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_rec_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3_(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_rec_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3____boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_recOn___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_recOn___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_recOn(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_recOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_brecOn_go___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_brecOn_go___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_brecOn_go(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_brecOn___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_brecOn(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___sizeOf__1___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___sizeOf__1___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___sizeOf__1___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___sizeOf__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___sizeOf__inst___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___sizeOf__inst___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___sizeOf__inst(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_nil(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_node___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_node(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_instInhabited(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_traverse___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_BinaryTree_traverse___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_BinaryTree_traverse___redArg___lam__0, .m_arity = 3, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_BinaryTree_traverse___redArg___closed__0 = (const lean_object*)&lp_mathlib_BinaryTree_traverse___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_traverse___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_traverse___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_traverse___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_traverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_traverse___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_traverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Tree_Basic_0__BinaryTree_traverse_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Tree_Basic_0__BinaryTree_traverse_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_numNodes___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_numNodes___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_numNodes(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_numNodes___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_numNodes___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_numNodes___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_numNodes(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_numNodes___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_numLeaves___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_numLeaves___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_numLeaves(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_numLeaves___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_numLeaves___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_numLeaves___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_numLeaves(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_numLeaves___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_height___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_height___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_height(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_height___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_height___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_height___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_height(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_height___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_left___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_left___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_left(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_left___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_left___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_left___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_left(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_left___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_right___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_right___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_right(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_right___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_right___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_right___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_right(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_right___boxed(lean_object*, lean_object*);
static const lean_string_object lp_mathlib_BinaryTree_term___u25b3___00__closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "BinaryTree"};
static const lean_object* lp_mathlib_BinaryTree_term___u25b3___00__closed__0 = (const lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__0_value;
static const lean_string_object lp_mathlib_BinaryTree_term___u25b3___00__closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 7, .m_data = "term_△_"};
static const lean_object* lp_mathlib_BinaryTree_term___u25b3___00__closed__1 = (const lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__1_value;
static const lean_ctor_object lp_mathlib_BinaryTree_term___u25b3___00__closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(74, 28, 132, 201, 129, 202, 51, 66)}};
static const lean_ctor_object lp_mathlib_BinaryTree_term___u25b3___00__closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__2_value_aux_0),((lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__1_value),LEAN_SCALAR_PTR_LITERAL(95, 154, 26, 41, 195, 142, 31, 111)}};
static const lean_object* lp_mathlib_BinaryTree_term___u25b3___00__closed__2 = (const lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__2_value;
static const lean_string_object lp_mathlib_BinaryTree_term___u25b3___00__closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "andthen"};
static const lean_object* lp_mathlib_BinaryTree_term___u25b3___00__closed__3 = (const lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__3_value;
static const lean_ctor_object lp_mathlib_BinaryTree_term___u25b3___00__closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__3_value),LEAN_SCALAR_PTR_LITERAL(40, 255, 78, 30, 143, 119, 117, 174)}};
static const lean_object* lp_mathlib_BinaryTree_term___u25b3___00__closed__4 = (const lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__4_value;
static const lean_string_object lp_mathlib_BinaryTree_term___u25b3___00__closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 3, .m_data = " △ "};
static const lean_object* lp_mathlib_BinaryTree_term___u25b3___00__closed__5 = (const lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__5_value;
static const lean_ctor_object lp_mathlib_BinaryTree_term___u25b3___00__closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 5}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__5_value)}};
static const lean_object* lp_mathlib_BinaryTree_term___u25b3___00__closed__6 = (const lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__6_value;
static const lean_string_object lp_mathlib_BinaryTree_term___u25b3___00__closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "term"};
static const lean_object* lp_mathlib_BinaryTree_term___u25b3___00__closed__7 = (const lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__7_value;
static const lean_ctor_object lp_mathlib_BinaryTree_term___u25b3___00__closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__7_value),LEAN_SCALAR_PTR_LITERAL(187, 230, 181, 162, 253, 146, 122, 119)}};
static const lean_object* lp_mathlib_BinaryTree_term___u25b3___00__closed__8 = (const lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__8_value;
static const lean_ctor_object lp_mathlib_BinaryTree_term___u25b3___00__closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 7}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__8_value),((lean_object*)(((size_t)(65) << 1) | 1))}};
static const lean_object* lp_mathlib_BinaryTree_term___u25b3___00__closed__9 = (const lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__9_value;
static const lean_ctor_object lp_mathlib_BinaryTree_term___u25b3___00__closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 2}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__4_value),((lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__6_value),((lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__9_value)}};
static const lean_object* lp_mathlib_BinaryTree_term___u25b3___00__closed__10 = (const lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__10_value;
static const lean_ctor_object lp_mathlib_BinaryTree_term___u25b3___00__closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*4 + 0, .m_other = 4, .m_tag = 4}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__2_value),((lean_object*)(((size_t)(65) << 1) | 1)),((lean_object*)(((size_t)(66) << 1) | 1)),((lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__10_value)}};
static const lean_object* lp_mathlib_BinaryTree_term___u25b3___00__closed__11 = (const lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__11_value;
LEAN_EXPORT const lean_object* lp_mathlib_BinaryTree_term___u25b3__ = (const lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__11_value;
static const lean_string_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__0 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__0_value;
static const lean_string_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__1 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__1_value;
static const lean_string_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__2 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__2_value;
static const lean_string_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "app"};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__3 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__3_value;
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(69, 118, 10, 41, 220, 156, 243, 179)}};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__4 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__4_value;
static lean_once_cell_t lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__5;
static const lean_string_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "node"};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__6 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__6_value;
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(74, 28, 132, 201, 129, 202, 51, 66)}};
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(165, 197, 227, 116, 17, 134, 221, 248)}};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__7 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__7_value;
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__7_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__8 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__8_value;
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__7_value)}};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__9 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__9_value;
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__9_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__10 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__10_value;
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__8_value),((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__10_value)}};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__11 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__11_value;
static const lean_string_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__12 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__12_value;
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__12_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__13 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__13_value;
static const lean_string_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "tuple"};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__14 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__14_value;
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__15_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__15_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__15_value_aux_0),((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__15_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__15_value_aux_1),((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__15_value_aux_2),((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(191, 24, 88, 245, 200, 250, 27, 217)}};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__15 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__15_value;
static const lean_string_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "hygienicLParen"};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__16 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__16_value;
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__17_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__17_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__17_value_aux_0),((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__17_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__17_value_aux_1),((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__17_value_aux_2),((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__16_value),LEAN_SCALAR_PTR_LITERAL(41, 104, 206, 51, 21, 254, 100, 101)}};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__17 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__17_value;
static const lean_string_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "("};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__18 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__18_value;
static const lean_string_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "hygieneInfo"};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__19 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__19_value;
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__19_value),LEAN_SCALAR_PTR_LITERAL(27, 64, 36, 144, 170, 151, 255, 136)}};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__20 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__20_value;
static const lean_string_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 1, .m_capacity = 1, .m_length = 0, .m_data = ""};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__21 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__21_value;
static lean_once_cell_t lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__22;
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_BinaryTree_term___u25b3___00__closed__0_value),LEAN_SCALAR_PTR_LITERAL(74, 28, 132, 201, 129, 202, 51, 66)}};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__23 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__23_value;
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 0}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__23_value)}};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__24 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__24_value;
static const lean_ctor_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__24_value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__25 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__25_value;
static lean_once_cell_t lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__26_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__26;
static const lean_string_object lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__27 = (const lean_object*)&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__27_value;
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_unitRecOn___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_unitRecOn___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_unitRecOn___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_unitRecOn(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_unitRecOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_unitRecOn___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_unitRecOn___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_unitRecOn(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Tree_unitRecOn___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_ctorIdx___redArg(lean_object* v_x_1_){
_start:
{
if (lean_obj_tag(v_x_1_) == 0)
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
else
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_ctorIdx___redArg___boxed(lean_object* v_x_4_){
_start:
{
lean_object* v_res_5_; 
v_res_5_ = lp_mathlib_BinaryTree_ctorIdx___redArg(v_x_4_);
lean_dec(v_x_4_);
return v_res_5_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_ctorIdx(lean_object* v_00_u03b1_6_, lean_object* v_x_7_){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = lp_mathlib_BinaryTree_ctorIdx___redArg(v_x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_ctorIdx___boxed(lean_object* v_00_u03b1_9_, lean_object* v_x_10_){
_start:
{
lean_object* v_res_11_; 
v_res_11_ = lp_mathlib_BinaryTree_ctorIdx(v_00_u03b1_9_, v_x_10_);
lean_dec(v_x_10_);
return v_res_11_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_ctorElim___redArg(lean_object* v_t_12_, lean_object* v_k_13_){
_start:
{
if (lean_obj_tag(v_t_12_) == 0)
{
return v_k_13_;
}
else
{
lean_object* v_value_14_; lean_object* v_left_15_; lean_object* v_right_16_; lean_object* v___x_17_; 
v_value_14_ = lean_ctor_get(v_t_12_, 0);
lean_inc(v_value_14_);
v_left_15_ = lean_ctor_get(v_t_12_, 1);
lean_inc(v_left_15_);
v_right_16_ = lean_ctor_get(v_t_12_, 2);
lean_inc(v_right_16_);
lean_dec_ref_known(v_t_12_, 3);
v___x_17_ = lean_apply_3(v_k_13_, v_value_14_, v_left_15_, v_right_16_);
return v___x_17_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_ctorElim(lean_object* v_00_u03b1_18_, lean_object* v_motive_19_, lean_object* v_ctorIdx_20_, lean_object* v_t_21_, lean_object* v_h_22_, lean_object* v_k_23_){
_start:
{
lean_object* v___x_24_; 
v___x_24_ = lp_mathlib_BinaryTree_ctorElim___redArg(v_t_21_, v_k_23_);
return v___x_24_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_ctorElim___boxed(lean_object* v_00_u03b1_25_, lean_object* v_motive_26_, lean_object* v_ctorIdx_27_, lean_object* v_t_28_, lean_object* v_h_29_, lean_object* v_k_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib_BinaryTree_ctorElim(v_00_u03b1_25_, v_motive_26_, v_ctorIdx_27_, v_t_28_, v_h_29_, v_k_30_);
lean_dec(v_ctorIdx_27_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_nil_elim___redArg(lean_object* v_t_32_, lean_object* v_nil_33_){
_start:
{
lean_object* v___x_34_; 
v___x_34_ = lp_mathlib_BinaryTree_ctorElim___redArg(v_t_32_, v_nil_33_);
return v___x_34_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_nil_elim(lean_object* v_00_u03b1_35_, lean_object* v_motive_36_, lean_object* v_t_37_, lean_object* v_h_38_, lean_object* v_nil_39_){
_start:
{
lean_object* v___x_40_; 
v___x_40_ = lp_mathlib_BinaryTree_ctorElim___redArg(v_t_37_, v_nil_39_);
return v___x_40_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_node_elim___redArg(lean_object* v_t_41_, lean_object* v_node_42_){
_start:
{
lean_object* v___x_43_; 
v___x_43_ = lp_mathlib_BinaryTree_ctorElim___redArg(v_t_41_, v_node_42_);
return v___x_43_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_node_elim(lean_object* v_00_u03b1_44_, lean_object* v_motive_45_, lean_object* v_t_46_, lean_object* v_h_47_, lean_object* v_node_48_){
_start:
{
lean_object* v___x_49_; 
v___x_49_ = lp_mathlib_BinaryTree_ctorElim___redArg(v_t_46_, v_node_48_);
return v___x_49_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqBinaryTree_decEq___redArg(lean_object* v_inst_50_, lean_object* v_x_51_, lean_object* v_x_52_){
_start:
{
if (lean_obj_tag(v_x_51_) == 0)
{
lean_dec_ref(v_inst_50_);
if (lean_obj_tag(v_x_52_) == 0)
{
uint8_t v___x_53_; 
v___x_53_ = 1;
return v___x_53_;
}
else
{
uint8_t v___x_54_; 
lean_dec_ref_known(v_x_52_, 3);
v___x_54_ = 0;
return v___x_54_;
}
}
else
{
lean_object* v_value_55_; lean_object* v_left_56_; lean_object* v_right_57_; uint8_t v___x_58_; 
v_value_55_ = lean_ctor_get(v_x_51_, 0);
lean_inc(v_value_55_);
v_left_56_ = lean_ctor_get(v_x_51_, 1);
lean_inc(v_left_56_);
v_right_57_ = lean_ctor_get(v_x_51_, 2);
lean_inc(v_right_57_);
lean_dec_ref_known(v_x_51_, 3);
v___x_58_ = 0;
if (lean_obj_tag(v_x_52_) == 0)
{
lean_dec(v_right_57_);
lean_dec(v_left_56_);
lean_dec(v_value_55_);
lean_dec_ref(v_inst_50_);
return v___x_58_;
}
else
{
lean_object* v_value_59_; lean_object* v_left_60_; lean_object* v_right_61_; lean_object* v___x_62_; uint8_t v___x_63_; 
v_value_59_ = lean_ctor_get(v_x_52_, 0);
lean_inc(v_value_59_);
v_left_60_ = lean_ctor_get(v_x_52_, 1);
lean_inc(v_left_60_);
v_right_61_ = lean_ctor_get(v_x_52_, 2);
lean_inc(v_right_61_);
lean_dec_ref_known(v_x_52_, 3);
lean_inc_ref(v_inst_50_);
v___x_62_ = lean_apply_2(v_inst_50_, v_value_55_, v_value_59_);
v___x_63_ = lean_unbox(v___x_62_);
if (v___x_63_ == 0)
{
lean_dec(v_right_61_);
lean_dec(v_left_60_);
lean_dec(v_right_57_);
lean_dec(v_left_56_);
lean_dec_ref(v_inst_50_);
return v___x_58_;
}
else
{
uint8_t v_inst_64_; 
lean_inc_ref(v_inst_50_);
v_inst_64_ = lp_mathlib_instDecidableEqBinaryTree_decEq___redArg(v_inst_50_, v_left_56_, v_left_60_);
if (v_inst_64_ == 0)
{
lean_dec(v_right_61_);
lean_dec(v_right_57_);
lean_dec_ref(v_inst_50_);
return v___x_58_;
}
else
{
uint8_t v_inst_65_; 
v_inst_65_ = lp_mathlib_instDecidableEqBinaryTree_decEq___redArg(v_inst_50_, v_right_57_, v_right_61_);
if (v_inst_65_ == 0)
{
return v___x_58_;
}
else
{
return v_inst_65_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqBinaryTree_decEq___redArg___boxed(lean_object* v_inst_66_, lean_object* v_x_67_, lean_object* v_x_68_){
_start:
{
uint8_t v_res_69_; lean_object* v_r_70_; 
v_res_69_ = lp_mathlib_instDecidableEqBinaryTree_decEq___redArg(v_inst_66_, v_x_67_, v_x_68_);
v_r_70_ = lean_box(v_res_69_);
return v_r_70_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqBinaryTree_decEq(lean_object* v_00_u03b1_71_, lean_object* v_inst_72_, lean_object* v_x_73_, lean_object* v_x_74_){
_start:
{
uint8_t v___x_75_; 
v___x_75_ = lp_mathlib_instDecidableEqBinaryTree_decEq___redArg(v_inst_72_, v_x_73_, v_x_74_);
return v___x_75_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqBinaryTree_decEq___boxed(lean_object* v_00_u03b1_76_, lean_object* v_inst_77_, lean_object* v_x_78_, lean_object* v_x_79_){
_start:
{
uint8_t v_res_80_; lean_object* v_r_81_; 
v_res_80_ = lp_mathlib_instDecidableEqBinaryTree_decEq(v_00_u03b1_76_, v_inst_77_, v_x_78_, v_x_79_);
v_r_81_ = lean_box(v_res_80_);
return v_r_81_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqBinaryTree___redArg(lean_object* v_inst_82_, lean_object* v_x_83_, lean_object* v_x_84_){
_start:
{
uint8_t v___x_85_; 
v___x_85_ = lp_mathlib_instDecidableEqBinaryTree_decEq___redArg(v_inst_82_, v_x_83_, v_x_84_);
return v___x_85_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqBinaryTree___redArg___boxed(lean_object* v_inst_86_, lean_object* v_x_87_, lean_object* v_x_88_){
_start:
{
uint8_t v_res_89_; lean_object* v_r_90_; 
v_res_89_ = lp_mathlib_instDecidableEqBinaryTree___redArg(v_inst_86_, v_x_87_, v_x_88_);
v_r_90_ = lean_box(v_res_89_);
return v_r_90_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instDecidableEqBinaryTree(lean_object* v_00_u03b1_91_, lean_object* v_inst_92_, lean_object* v_x_93_, lean_object* v_x_94_){
_start:
{
uint8_t v___x_95_; 
v___x_95_ = lp_mathlib_instDecidableEqBinaryTree_decEq___redArg(v_inst_92_, v_x_93_, v_x_94_);
return v___x_95_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instDecidableEqBinaryTree___boxed(lean_object* v_00_u03b1_96_, lean_object* v_inst_97_, lean_object* v_x_98_, lean_object* v_x_99_){
_start:
{
uint8_t v_res_100_; lean_object* v_r_101_; 
v_res_100_ = lp_mathlib_instDecidableEqBinaryTree(v_00_u03b1_96_, v_inst_97_, v_x_98_, v_x_99_);
v_r_101_ = lean_box(v_res_100_);
return v_r_101_;
}
}
static lean_object* _init_lp_mathlib_instReprBinaryTree_repr___redArg___closed__2(void){
_start:
{
lean_object* v___x_105_; lean_object* v___x_106_; 
v___x_105_ = lean_unsigned_to_nat(2u);
v___x_106_ = lean_nat_to_int(v___x_105_);
return v___x_106_;
}
}
static lean_object* _init_lp_mathlib_instReprBinaryTree_repr___redArg___closed__3(void){
_start:
{
lean_object* v___x_107_; lean_object* v___x_108_; 
v___x_107_ = lean_unsigned_to_nat(1u);
v___x_108_ = lean_nat_to_int(v___x_107_);
return v___x_108_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instReprBinaryTree_repr___redArg(lean_object* v_inst_115_, lean_object* v_x_116_, lean_object* v_prec_117_){
_start:
{
lean_object* v___y_119_; 
if (lean_obj_tag(v_x_116_) == 0)
{
lean_object* v___x_125_; uint8_t v___x_126_; 
lean_dec_ref(v_inst_115_);
v___x_125_ = lean_unsigned_to_nat(1024u);
v___x_126_ = lean_nat_dec_le(v___x_125_, v_prec_117_);
if (v___x_126_ == 0)
{
lean_object* v___x_127_; 
v___x_127_ = lean_obj_once(&lp_mathlib_instReprBinaryTree_repr___redArg___closed__2, &lp_mathlib_instReprBinaryTree_repr___redArg___closed__2_once, _init_lp_mathlib_instReprBinaryTree_repr___redArg___closed__2);
v___y_119_ = v___x_127_;
goto v___jp_118_;
}
else
{
lean_object* v___x_128_; 
v___x_128_ = lean_obj_once(&lp_mathlib_instReprBinaryTree_repr___redArg___closed__3, &lp_mathlib_instReprBinaryTree_repr___redArg___closed__3_once, _init_lp_mathlib_instReprBinaryTree_repr___redArg___closed__3);
v___y_119_ = v___x_128_;
goto v___jp_118_;
}
}
else
{
lean_object* v_value_129_; lean_object* v_left_130_; lean_object* v_right_131_; lean_object* v___x_132_; lean_object* v___y_134_; uint8_t v___x_149_; 
v_value_129_ = lean_ctor_get(v_x_116_, 0);
lean_inc(v_value_129_);
v_left_130_ = lean_ctor_get(v_x_116_, 1);
lean_inc(v_left_130_);
v_right_131_ = lean_ctor_get(v_x_116_, 2);
lean_inc(v_right_131_);
lean_dec_ref_known(v_x_116_, 3);
v___x_132_ = lean_unsigned_to_nat(1024u);
v___x_149_ = lean_nat_dec_le(v___x_132_, v_prec_117_);
if (v___x_149_ == 0)
{
lean_object* v___x_150_; 
v___x_150_ = lean_obj_once(&lp_mathlib_instReprBinaryTree_repr___redArg___closed__2, &lp_mathlib_instReprBinaryTree_repr___redArg___closed__2_once, _init_lp_mathlib_instReprBinaryTree_repr___redArg___closed__2);
v___y_134_ = v___x_150_;
goto v___jp_133_;
}
else
{
lean_object* v___x_151_; 
v___x_151_ = lean_obj_once(&lp_mathlib_instReprBinaryTree_repr___redArg___closed__3, &lp_mathlib_instReprBinaryTree_repr___redArg___closed__3_once, _init_lp_mathlib_instReprBinaryTree_repr___redArg___closed__3);
v___y_134_ = v___x_151_;
goto v___jp_133_;
}
v___jp_133_:
{
lean_object* v___x_135_; lean_object* v___x_136_; lean_object* v___x_137_; lean_object* v___x_138_; lean_object* v___x_139_; lean_object* v___x_140_; lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; lean_object* v___x_145_; uint8_t v___x_146_; lean_object* v___x_147_; lean_object* v___x_148_; 
v___x_135_ = lean_box(1);
v___x_136_ = ((lean_object*)(lp_mathlib_instReprBinaryTree_repr___redArg___closed__6));
lean_inc_ref_n(v_inst_115_, 2);
v___x_137_ = lean_apply_2(v_inst_115_, v_value_129_, v___x_132_);
v___x_138_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_138_, 0, v___x_136_);
lean_ctor_set(v___x_138_, 1, v___x_137_);
v___x_139_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_139_, 0, v___x_138_);
lean_ctor_set(v___x_139_, 1, v___x_135_);
v___x_140_ = lp_mathlib_instReprBinaryTree_repr___redArg(v_inst_115_, v_left_130_, v___x_132_);
v___x_141_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_141_, 0, v___x_139_);
lean_ctor_set(v___x_141_, 1, v___x_140_);
v___x_142_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_142_, 0, v___x_141_);
lean_ctor_set(v___x_142_, 1, v___x_135_);
v___x_143_ = lp_mathlib_instReprBinaryTree_repr___redArg(v_inst_115_, v_right_131_, v___x_132_);
v___x_144_ = lean_alloc_ctor(5, 2, 0);
lean_ctor_set(v___x_144_, 0, v___x_142_);
lean_ctor_set(v___x_144_, 1, v___x_143_);
lean_inc(v___y_134_);
v___x_145_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_145_, 0, v___y_134_);
lean_ctor_set(v___x_145_, 1, v___x_144_);
v___x_146_ = 0;
v___x_147_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_147_, 0, v___x_145_);
lean_ctor_set_uint8(v___x_147_, sizeof(void*)*1, v___x_146_);
v___x_148_ = l_Repr_addAppParen(v___x_147_, v_prec_117_);
return v___x_148_;
}
}
v___jp_118_:
{
lean_object* v___x_120_; lean_object* v___x_121_; uint8_t v___x_122_; lean_object* v___x_123_; lean_object* v___x_124_; 
v___x_120_ = ((lean_object*)(lp_mathlib_instReprBinaryTree_repr___redArg___closed__1));
lean_inc(v___y_119_);
v___x_121_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_121_, 0, v___y_119_);
lean_ctor_set(v___x_121_, 1, v___x_120_);
v___x_122_ = 0;
v___x_123_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_123_, 0, v___x_121_);
lean_ctor_set_uint8(v___x_123_, sizeof(void*)*1, v___x_122_);
v___x_124_ = l_Repr_addAppParen(v___x_123_, v_prec_117_);
return v___x_124_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instReprBinaryTree_repr___redArg___boxed(lean_object* v_inst_152_, lean_object* v_x_153_, lean_object* v_prec_154_){
_start:
{
lean_object* v_res_155_; 
v_res_155_ = lp_mathlib_instReprBinaryTree_repr___redArg(v_inst_152_, v_x_153_, v_prec_154_);
lean_dec(v_prec_154_);
return v_res_155_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instReprBinaryTree_repr(lean_object* v_00_u03b1_156_, lean_object* v_inst_157_, lean_object* v_x_158_, lean_object* v_prec_159_){
_start:
{
lean_object* v___x_160_; 
v___x_160_ = lp_mathlib_instReprBinaryTree_repr___redArg(v_inst_157_, v_x_158_, v_prec_159_);
return v___x_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instReprBinaryTree_repr___boxed(lean_object* v_00_u03b1_161_, lean_object* v_inst_162_, lean_object* v_x_163_, lean_object* v_prec_164_){
_start:
{
lean_object* v_res_165_; 
v_res_165_ = lp_mathlib_instReprBinaryTree_repr(v_00_u03b1_161_, v_inst_162_, v_x_163_, v_prec_164_);
lean_dec(v_prec_164_);
return v_res_165_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instReprBinaryTree___redArg(lean_object* v_inst_166_){
_start:
{
lean_object* v___x_167_; 
v___x_167_ = lean_alloc_closure((void*)(lp_mathlib_instReprBinaryTree_repr___boxed), 4, 2);
lean_closure_set(v___x_167_, 0, lean_box(0));
lean_closure_set(v___x_167_, 1, v_inst_166_);
return v___x_167_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instReprBinaryTree(lean_object* v_00_u03b1_168_, lean_object* v_inst_169_){
_start:
{
lean_object* v___x_170_; 
v___x_170_ = lean_alloc_closure((void*)(lp_mathlib_instReprBinaryTree_repr___boxed), 4, 2);
lean_closure_set(v___x_170_, 0, lean_box(0));
lean_closure_set(v___x_170_, 1, v_inst_169_);
return v___x_170_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_rec___redArg_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3_(lean_object* v_nil_171_, lean_object* v_node_172_, lean_object* v_t_173_){
_start:
{
if (lean_obj_tag(v_t_173_) == 0)
{
lean_dec(v_node_172_);
lean_inc(v_nil_171_);
return v_nil_171_;
}
else
{
lean_object* v_value_174_; lean_object* v_left_175_; lean_object* v_right_176_; lean_object* v___x_177_; lean_object* v___x_178_; lean_object* v___x_179_; 
v_value_174_ = lean_ctor_get(v_t_173_, 0);
lean_inc(v_value_174_);
v_left_175_ = lean_ctor_get(v_t_173_, 1);
lean_inc_n(v_left_175_, 2);
v_right_176_ = lean_ctor_get(v_t_173_, 2);
lean_inc_n(v_right_176_, 2);
lean_dec_ref_known(v_t_173_, 3);
lean_inc_n(v_node_172_, 2);
v___x_177_ = lp_mathlib_BinaryTree_rec___redArg_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3_(v_nil_171_, v_node_172_, v_left_175_);
v___x_178_ = lp_mathlib_BinaryTree_rec___redArg_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3_(v_nil_171_, v_node_172_, v_right_176_);
v___x_179_ = lean_apply_5(v_node_172_, v_value_174_, v_left_175_, v_right_176_, v___x_177_, v___x_178_);
return v___x_179_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_rec___redArg_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3____boxed(lean_object* v_nil_180_, lean_object* v_node_181_, lean_object* v_t_182_){
_start:
{
lean_object* v_res_183_; 
v_res_183_ = lp_mathlib_BinaryTree_rec___redArg_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3_(v_nil_180_, v_node_181_, v_t_182_);
lean_dec(v_nil_180_);
return v_res_183_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_rec_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3_(lean_object* v_00_u03b1_184_, lean_object* v_motive_185_, lean_object* v_nil_186_, lean_object* v_node_187_, lean_object* v_t_188_){
_start:
{
lean_object* v___x_189_; 
v___x_189_ = lp_mathlib_BinaryTree_rec___redArg_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3_(v_nil_186_, v_node_187_, v_t_188_);
return v___x_189_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_rec_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3____boxed(lean_object* v_00_u03b1_190_, lean_object* v_motive_191_, lean_object* v_nil_192_, lean_object* v_node_193_, lean_object* v_t_194_){
_start:
{
lean_object* v_res_195_; 
v_res_195_ = lp_mathlib_BinaryTree_rec_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3_(v_00_u03b1_190_, v_motive_191_, v_nil_192_, v_node_193_, v_t_194_);
lean_dec(v_nil_192_);
return v_res_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_recOn___redArg(lean_object* v_t_196_, lean_object* v_nil_197_, lean_object* v_node_198_){
_start:
{
lean_object* v___x_199_; 
v___x_199_ = lp_mathlib_BinaryTree_rec___redArg_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3_(v_nil_197_, v_node_198_, v_t_196_);
return v___x_199_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_recOn___redArg___boxed(lean_object* v_t_200_, lean_object* v_nil_201_, lean_object* v_node_202_){
_start:
{
lean_object* v_res_203_; 
v_res_203_ = lp_mathlib_BinaryTree_recOn___redArg(v_t_200_, v_nil_201_, v_node_202_);
lean_dec(v_nil_201_);
return v_res_203_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_recOn(lean_object* v_00_u03b1_204_, lean_object* v_motive_205_, lean_object* v_t_206_, lean_object* v_nil_207_, lean_object* v_node_208_){
_start:
{
lean_object* v___x_209_; 
v___x_209_ = lp_mathlib_BinaryTree_rec___redArg_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3_(v_nil_207_, v_node_208_, v_t_206_);
return v___x_209_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_recOn___boxed(lean_object* v_00_u03b1_210_, lean_object* v_motive_211_, lean_object* v_t_212_, lean_object* v_nil_213_, lean_object* v_node_214_){
_start:
{
lean_object* v_res_215_; 
v_res_215_ = lp_mathlib_BinaryTree_recOn(v_00_u03b1_210_, v_motive_211_, v_t_212_, v_nil_213_, v_node_214_);
lean_dec(v_nil_213_);
return v_res_215_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_brecOn_go___redArg___lam__0(lean_object* v_F__1_216_, lean_object* v_value_217_, lean_object* v_left_218_, lean_object* v_right_219_, lean_object* v_left__ih_220_, lean_object* v_right__ih_221_){
_start:
{
lean_object* v___x_222_; lean_object* v___x_223_; lean_object* v___x_224_; lean_object* v___x_225_; 
v___x_222_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_222_, 0, v_value_217_);
lean_ctor_set(v___x_222_, 1, v_left_218_);
lean_ctor_set(v___x_222_, 2, v_right_219_);
v___x_223_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_223_, 0, v_left__ih_220_);
lean_ctor_set(v___x_223_, 1, v_right__ih_221_);
lean_inc_ref(v___x_223_);
v___x_224_ = lean_apply_2(v_F__1_216_, v___x_222_, v___x_223_);
v___x_225_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_225_, 0, v___x_224_);
lean_ctor_set(v___x_225_, 1, v___x_223_);
return v___x_225_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_brecOn_go___redArg(lean_object* v_t_226_, lean_object* v_F__1_227_){
_start:
{
lean_object* v___f_228_; lean_object* v___x_229_; lean_object* v___x_230_; lean_object* v___x_231_; lean_object* v___x_232_; lean_object* v___x_233_; 
lean_inc(v_F__1_227_);
v___f_228_ = lean_alloc_closure((void*)(lp_mathlib_BinaryTree_brecOn_go___redArg___lam__0), 6, 1);
lean_closure_set(v___f_228_, 0, v_F__1_227_);
v___x_229_ = lean_box(0);
v___x_230_ = lean_box(0);
v___x_231_ = lean_apply_2(v_F__1_227_, v___x_229_, v___x_230_);
v___x_232_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_232_, 0, v___x_231_);
lean_ctor_set(v___x_232_, 1, v___x_230_);
v___x_233_ = lp_mathlib_BinaryTree_rec___redArg_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3_(v___x_232_, v___f_228_, v_t_226_);
lean_dec_ref_known(v___x_232_, 2);
return v___x_233_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_brecOn_go(lean_object* v_00_u03b1_234_, lean_object* v_motive_235_, lean_object* v_t_236_, lean_object* v_F__1_237_){
_start:
{
lean_object* v___x_238_; 
v___x_238_ = lp_mathlib_BinaryTree_brecOn_go___redArg(v_t_236_, v_F__1_237_);
return v___x_238_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_brecOn___redArg(lean_object* v_t_239_, lean_object* v_F__1_240_){
_start:
{
lean_object* v___x_241_; lean_object* v_fst_242_; 
v___x_241_ = lp_mathlib_BinaryTree_brecOn_go___redArg(v_t_239_, v_F__1_240_);
v_fst_242_ = lean_ctor_get(v___x_241_, 0);
lean_inc(v_fst_242_);
lean_dec_ref(v___x_241_);
return v_fst_242_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_brecOn(lean_object* v_00_u03b1_243_, lean_object* v_motive_244_, lean_object* v_t_245_, lean_object* v_F__1_246_){
_start:
{
lean_object* v___x_247_; 
v___x_247_ = lp_mathlib_BinaryTree_brecOn___redArg(v_t_245_, v_F__1_246_);
return v___x_247_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___sizeOf__1___redArg___lam__0(lean_object* v_inst_248_, lean_object* v___x_249_, lean_object* v_value_250_, lean_object* v_left_251_, lean_object* v_right_252_, lean_object* v_left__ih_253_, lean_object* v_right__ih_254_){
_start:
{
lean_object* v___x_255_; lean_object* v___x_256_; lean_object* v___x_257_; lean_object* v___x_258_; 
v___x_255_ = lean_apply_1(v_inst_248_, v_value_250_);
v___x_256_ = lean_nat_add(v___x_249_, v___x_255_);
lean_dec(v___x_255_);
v___x_257_ = lean_nat_add(v___x_256_, v_left__ih_253_);
lean_dec(v___x_256_);
v___x_258_ = lean_nat_add(v___x_257_, v_right__ih_254_);
lean_dec(v___x_257_);
return v___x_258_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___sizeOf__1___redArg___lam__0___boxed(lean_object* v_inst_259_, lean_object* v___x_260_, lean_object* v_value_261_, lean_object* v_left_262_, lean_object* v_right_263_, lean_object* v_left__ih_264_, lean_object* v_right__ih_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_mathlib_BinaryTree___sizeOf__1___redArg___lam__0(v_inst_259_, v___x_260_, v_value_261_, v_left_262_, v_right_263_, v_left__ih_264_, v_right__ih_265_);
lean_dec(v_right__ih_265_);
lean_dec(v_left__ih_264_);
lean_dec(v_right_263_);
lean_dec(v_left_262_);
lean_dec(v___x_260_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___sizeOf__1___redArg(lean_object* v_inst_267_, lean_object* v_t_268_){
_start:
{
lean_object* v___x_269_; lean_object* v___f_270_; lean_object* v___x_271_; 
v___x_269_ = lean_unsigned_to_nat(1u);
v___f_270_ = lean_alloc_closure((void*)(lp_mathlib_BinaryTree___sizeOf__1___redArg___lam__0___boxed), 7, 2);
lean_closure_set(v___f_270_, 0, v_inst_267_);
lean_closure_set(v___f_270_, 1, v___x_269_);
v___x_271_ = lp_mathlib_BinaryTree_rec___redArg_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3_(v___x_269_, v___f_270_, v_t_268_);
return v___x_271_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___sizeOf__1(lean_object* v_00_u03b1_272_, lean_object* v_inst_273_, lean_object* v_t_274_){
_start:
{
lean_object* v___x_275_; 
v___x_275_ = lp_mathlib_BinaryTree___sizeOf__1___redArg(v_inst_273_, v_t_274_);
return v___x_275_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___sizeOf__inst___redArg___lam__0(lean_object* v_inst_276_, lean_object* v_m_277_){
_start:
{
lean_object* v___x_278_; 
v___x_278_ = lp_mathlib_BinaryTree___sizeOf__1___redArg(v_inst_276_, v_m_277_);
return v___x_278_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___sizeOf__inst___redArg(lean_object* v_inst_279_){
_start:
{
lean_object* v___f_280_; 
v___f_280_ = lean_alloc_closure((void*)(lp_mathlib_BinaryTree___sizeOf__inst___redArg___lam__0), 2, 1);
lean_closure_set(v___f_280_, 0, v_inst_279_);
return v___f_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___sizeOf__inst(lean_object* v_00_u03b1_281_, lean_object* v_inst_282_){
_start:
{
lean_object* v___f_283_; 
v___f_283_ = lean_alloc_closure((void*)(lp_mathlib_BinaryTree___sizeOf__inst___redArg___lam__0), 2, 1);
lean_closure_set(v___f_283_, 0, v_inst_282_);
return v___f_283_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_nil(lean_object* v_00_u03b1_284_){
_start:
{
lean_object* v___x_285_; 
v___x_285_ = lean_box(0);
return v___x_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_node___redArg(lean_object* v_value_286_, lean_object* v_left_287_, lean_object* v_right_288_){
_start:
{
lean_object* v___x_289_; 
v___x_289_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_289_, 0, v_value_286_);
lean_ctor_set(v___x_289_, 1, v_left_287_);
lean_ctor_set(v___x_289_, 2, v_right_288_);
return v___x_289_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_node(lean_object* v_00_u03b1_290_, lean_object* v_value_291_, lean_object* v_left_292_, lean_object* v_right_293_){
_start:
{
lean_object* v___x_294_; 
v___x_294_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_294_, 0, v_value_291_);
lean_ctor_set(v___x_294_, 1, v_left_292_);
lean_ctor_set(v___x_294_, 2, v_right_293_);
return v___x_294_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_instInhabited(lean_object* v_00_u03b1_295_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lean_box(0);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_traverse___redArg___lam__0(lean_object* v_value_297_, lean_object* v_left_298_, lean_object* v_right_299_){
_start:
{
lean_object* v___x_300_; 
v___x_300_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_300_, 0, v_value_297_);
lean_ctor_set(v___x_300_, 1, v_left_298_);
lean_ctor_set(v___x_300_, 2, v_right_299_);
return v___x_300_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_traverse___redArg___lam__2(lean_object* v_inst_302_, lean_object* v_f_303_, lean_object* v_left_304_, lean_object* v_x_305_){
_start:
{
lean_object* v___x_306_; 
v___x_306_ = lp_mathlib_BinaryTree_traverse___redArg(v_inst_302_, v_f_303_, v_left_304_);
return v___x_306_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_traverse___redArg(lean_object* v_inst_307_, lean_object* v_f_308_, lean_object* v_x_309_){
_start:
{
if (lean_obj_tag(v_x_309_) == 0)
{
lean_object* v_toPure_310_; lean_object* v___x_311_; lean_object* v___x_312_; 
lean_dec(v_f_308_);
v_toPure_310_ = lean_ctor_get(v_inst_307_, 1);
lean_inc(v_toPure_310_);
lean_dec_ref(v_inst_307_);
v___x_311_ = lean_box(0);
v___x_312_ = lean_apply_2(v_toPure_310_, lean_box(0), v___x_311_);
return v___x_312_;
}
else
{
lean_object* v_toFunctor_313_; lean_object* v_toSeq_314_; lean_object* v_value_315_; lean_object* v_left_316_; lean_object* v_right_317_; lean_object* v_map_318_; lean_object* v___f_319_; lean_object* v___f_320_; lean_object* v___f_321_; lean_object* v___x_322_; lean_object* v___x_323_; lean_object* v___x_324_; lean_object* v___x_325_; 
v_toFunctor_313_ = lean_ctor_get(v_inst_307_, 0);
v_toSeq_314_ = lean_ctor_get(v_inst_307_, 2);
lean_inc_n(v_toSeq_314_, 2);
v_value_315_ = lean_ctor_get(v_x_309_, 0);
lean_inc(v_value_315_);
v_left_316_ = lean_ctor_get(v_x_309_, 1);
lean_inc(v_left_316_);
v_right_317_ = lean_ctor_get(v_x_309_, 2);
lean_inc(v_right_317_);
lean_dec_ref_known(v_x_309_, 3);
v_map_318_ = lean_ctor_get(v_toFunctor_313_, 0);
lean_inc(v_map_318_);
v___f_319_ = ((lean_object*)(lp_mathlib_BinaryTree_traverse___redArg___closed__0));
lean_inc_n(v_f_308_, 2);
lean_inc_ref(v_inst_307_);
v___f_320_ = lean_alloc_closure((void*)(lp_mathlib_BinaryTree_traverse___redArg___lam__1), 4, 3);
lean_closure_set(v___f_320_, 0, v_inst_307_);
lean_closure_set(v___f_320_, 1, v_f_308_);
lean_closure_set(v___f_320_, 2, v_right_317_);
v___f_321_ = lean_alloc_closure((void*)(lp_mathlib_BinaryTree_traverse___redArg___lam__2), 4, 3);
lean_closure_set(v___f_321_, 0, v_inst_307_);
lean_closure_set(v___f_321_, 1, v_f_308_);
lean_closure_set(v___f_321_, 2, v_left_316_);
v___x_322_ = lean_apply_1(v_f_308_, v_value_315_);
v___x_323_ = lean_apply_4(v_map_318_, lean_box(0), lean_box(0), v___f_319_, v___x_322_);
v___x_324_ = lean_apply_4(v_toSeq_314_, lean_box(0), lean_box(0), v___x_323_, v___f_321_);
v___x_325_ = lean_apply_4(v_toSeq_314_, lean_box(0), lean_box(0), v___x_324_, v___f_320_);
return v___x_325_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_traverse___redArg___lam__1(lean_object* v_inst_326_, lean_object* v_f_327_, lean_object* v_right_328_, lean_object* v_x_329_){
_start:
{
lean_object* v___x_330_; 
v___x_330_ = lp_mathlib_BinaryTree_traverse___redArg(v_inst_326_, v_f_327_, v_right_328_);
return v___x_330_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_traverse(lean_object* v_m_331_, lean_object* v_inst_332_, lean_object* v_00_u03b1_333_, lean_object* v_00_u03b2_334_, lean_object* v_f_335_, lean_object* v_x_336_){
_start:
{
lean_object* v___x_337_; 
v___x_337_ = lp_mathlib_BinaryTree_traverse___redArg(v_inst_332_, v_f_335_, v_x_336_);
return v___x_337_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_traverse___redArg(lean_object* v_inst_338_, lean_object* v_f_339_, lean_object* v_t_340_){
_start:
{
lean_object* v___x_341_; 
v___x_341_ = lp_mathlib_BinaryTree_traverse___redArg(v_inst_338_, v_f_339_, v_t_340_);
return v___x_341_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_traverse(lean_object* v_m_342_, lean_object* v_inst_343_, lean_object* v_00_u03b1_344_, lean_object* v_00_u03b2_345_, lean_object* v_f_346_, lean_object* v_t_347_){
_start:
{
lean_object* v___x_348_; 
v___x_348_ = lp_mathlib_BinaryTree_traverse___redArg(v_inst_343_, v_f_346_, v_t_347_);
return v___x_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_map___redArg(lean_object* v_f_349_, lean_object* v_x_350_){
_start:
{
if (lean_obj_tag(v_x_350_) == 0)
{
lean_object* v___x_351_; 
lean_dec(v_f_349_);
v___x_351_ = lean_box(0);
return v___x_351_;
}
else
{
lean_object* v_value_352_; lean_object* v_left_353_; lean_object* v_right_354_; lean_object* v___x_356_; uint8_t v_isShared_357_; uint8_t v_isSharedCheck_364_; 
v_value_352_ = lean_ctor_get(v_x_350_, 0);
v_left_353_ = lean_ctor_get(v_x_350_, 1);
v_right_354_ = lean_ctor_get(v_x_350_, 2);
v_isSharedCheck_364_ = !lean_is_exclusive(v_x_350_);
if (v_isSharedCheck_364_ == 0)
{
v___x_356_ = v_x_350_;
v_isShared_357_ = v_isSharedCheck_364_;
goto v_resetjp_355_;
}
else
{
lean_inc(v_right_354_);
lean_inc(v_left_353_);
lean_inc(v_value_352_);
lean_dec(v_x_350_);
v___x_356_ = lean_box(0);
v_isShared_357_ = v_isSharedCheck_364_;
goto v_resetjp_355_;
}
v_resetjp_355_:
{
lean_object* v___x_358_; lean_object* v___x_359_; lean_object* v___x_360_; lean_object* v___x_362_; 
lean_inc_n(v_f_349_, 2);
v___x_358_ = lean_apply_1(v_f_349_, v_value_352_);
v___x_359_ = lp_mathlib_BinaryTree_map___redArg(v_f_349_, v_left_353_);
v___x_360_ = lp_mathlib_BinaryTree_map___redArg(v_f_349_, v_right_354_);
if (v_isShared_357_ == 0)
{
lean_ctor_set(v___x_356_, 2, v___x_360_);
lean_ctor_set(v___x_356_, 1, v___x_359_);
lean_ctor_set(v___x_356_, 0, v___x_358_);
v___x_362_ = v___x_356_;
goto v_reusejp_361_;
}
else
{
lean_object* v_reuseFailAlloc_363_; 
v_reuseFailAlloc_363_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v_reuseFailAlloc_363_, 0, v___x_358_);
lean_ctor_set(v_reuseFailAlloc_363_, 1, v___x_359_);
lean_ctor_set(v_reuseFailAlloc_363_, 2, v___x_360_);
v___x_362_ = v_reuseFailAlloc_363_;
goto v_reusejp_361_;
}
v_reusejp_361_:
{
return v___x_362_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_map(lean_object* v_00_u03b1_365_, lean_object* v_00_u03b2_366_, lean_object* v_f_367_, lean_object* v_x_368_){
_start:
{
lean_object* v___x_369_; 
v___x_369_ = lp_mathlib_BinaryTree_map___redArg(v_f_367_, v_x_368_);
return v___x_369_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_map___redArg(lean_object* v_f_370_, lean_object* v_t_371_){
_start:
{
lean_object* v___x_372_; 
v___x_372_ = lp_mathlib_BinaryTree_map___redArg(v_f_370_, v_t_371_);
return v___x_372_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_map(lean_object* v_00_u03b1_373_, lean_object* v_00_u03b2_374_, lean_object* v_f_375_, lean_object* v_t_376_){
_start:
{
lean_object* v___x_377_; 
v___x_377_ = lp_mathlib_BinaryTree_map___redArg(v_f_375_, v_t_376_);
return v___x_377_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Tree_Basic_0__BinaryTree_traverse_match__1_splitter___redArg(lean_object* v_x_378_, lean_object* v_h__1_379_, lean_object* v_h__2_380_){
_start:
{
if (lean_obj_tag(v_x_378_) == 0)
{
lean_object* v___x_381_; lean_object* v___x_382_; 
lean_dec(v_h__2_380_);
v___x_381_ = lean_box(0);
v___x_382_ = lean_apply_1(v_h__1_379_, v___x_381_);
return v___x_382_;
}
else
{
lean_object* v_value_383_; lean_object* v_left_384_; lean_object* v_right_385_; lean_object* v___x_386_; 
lean_dec(v_h__1_379_);
v_value_383_ = lean_ctor_get(v_x_378_, 0);
lean_inc(v_value_383_);
v_left_384_ = lean_ctor_get(v_x_378_, 1);
lean_inc(v_left_384_);
v_right_385_ = lean_ctor_get(v_x_378_, 2);
lean_inc(v_right_385_);
lean_dec_ref_known(v_x_378_, 3);
v___x_386_ = lean_apply_3(v_h__2_380_, v_value_383_, v_left_384_, v_right_385_);
return v___x_386_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Tree_Basic_0__BinaryTree_traverse_match__1_splitter(lean_object* v_00_u03b1_387_, lean_object* v_motive_388_, lean_object* v_x_389_, lean_object* v_h__1_390_, lean_object* v_h__2_391_){
_start:
{
if (lean_obj_tag(v_x_389_) == 0)
{
lean_object* v___x_392_; lean_object* v___x_393_; 
lean_dec(v_h__2_391_);
v___x_392_ = lean_box(0);
v___x_393_ = lean_apply_1(v_h__1_390_, v___x_392_);
return v___x_393_;
}
else
{
lean_object* v_value_394_; lean_object* v_left_395_; lean_object* v_right_396_; lean_object* v___x_397_; 
lean_dec(v_h__1_390_);
v_value_394_ = lean_ctor_get(v_x_389_, 0);
lean_inc(v_value_394_);
v_left_395_ = lean_ctor_get(v_x_389_, 1);
lean_inc(v_left_395_);
v_right_396_ = lean_ctor_get(v_x_389_, 2);
lean_inc(v_right_396_);
lean_dec_ref_known(v_x_389_, 3);
v___x_397_ = lean_apply_3(v_h__2_391_, v_value_394_, v_left_395_, v_right_396_);
return v___x_397_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_numNodes___redArg(lean_object* v_x_398_){
_start:
{
if (lean_obj_tag(v_x_398_) == 0)
{
lean_object* v___x_399_; 
v___x_399_ = lean_unsigned_to_nat(0u);
return v___x_399_;
}
else
{
lean_object* v_left_400_; lean_object* v_right_401_; lean_object* v___x_402_; lean_object* v___x_403_; lean_object* v___x_404_; lean_object* v___x_405_; lean_object* v___x_406_; 
v_left_400_ = lean_ctor_get(v_x_398_, 1);
v_right_401_ = lean_ctor_get(v_x_398_, 2);
v___x_402_ = lp_mathlib_BinaryTree_numNodes___redArg(v_left_400_);
v___x_403_ = lp_mathlib_BinaryTree_numNodes___redArg(v_right_401_);
v___x_404_ = lean_nat_add(v___x_402_, v___x_403_);
lean_dec(v___x_403_);
lean_dec(v___x_402_);
v___x_405_ = lean_unsigned_to_nat(1u);
v___x_406_ = lean_nat_add(v___x_404_, v___x_405_);
lean_dec(v___x_404_);
return v___x_406_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_numNodes___redArg___boxed(lean_object* v_x_407_){
_start:
{
lean_object* v_res_408_; 
v_res_408_ = lp_mathlib_BinaryTree_numNodes___redArg(v_x_407_);
lean_dec(v_x_407_);
return v_res_408_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_numNodes(lean_object* v_00_u03b1_409_, lean_object* v_x_410_){
_start:
{
lean_object* v___x_411_; 
v___x_411_ = lp_mathlib_BinaryTree_numNodes___redArg(v_x_410_);
return v___x_411_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_numNodes___boxed(lean_object* v_00_u03b1_412_, lean_object* v_x_413_){
_start:
{
lean_object* v_res_414_; 
v_res_414_ = lp_mathlib_BinaryTree_numNodes(v_00_u03b1_412_, v_x_413_);
lean_dec(v_x_413_);
return v_res_414_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_numNodes___redArg(lean_object* v_t_415_){
_start:
{
lean_object* v___x_416_; 
v___x_416_ = lp_mathlib_BinaryTree_numNodes___redArg(v_t_415_);
return v___x_416_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_numNodes___redArg___boxed(lean_object* v_t_417_){
_start:
{
lean_object* v_res_418_; 
v_res_418_ = lp_mathlib_Tree_numNodes___redArg(v_t_417_);
lean_dec(v_t_417_);
return v_res_418_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_numNodes(lean_object* v_00_u03b1_419_, lean_object* v_t_420_){
_start:
{
lean_object* v___x_421_; 
v___x_421_ = lp_mathlib_BinaryTree_numNodes___redArg(v_t_420_);
return v___x_421_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_numNodes___boxed(lean_object* v_00_u03b1_422_, lean_object* v_t_423_){
_start:
{
lean_object* v_res_424_; 
v_res_424_ = lp_mathlib_Tree_numNodes(v_00_u03b1_422_, v_t_423_);
lean_dec(v_t_423_);
return v_res_424_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_numLeaves___redArg(lean_object* v_x_425_){
_start:
{
if (lean_obj_tag(v_x_425_) == 0)
{
lean_object* v___x_426_; 
v___x_426_ = lean_unsigned_to_nat(1u);
return v___x_426_;
}
else
{
lean_object* v_left_427_; lean_object* v_right_428_; lean_object* v___x_429_; lean_object* v___x_430_; lean_object* v___x_431_; 
v_left_427_ = lean_ctor_get(v_x_425_, 1);
v_right_428_ = lean_ctor_get(v_x_425_, 2);
v___x_429_ = lp_mathlib_BinaryTree_numLeaves___redArg(v_left_427_);
v___x_430_ = lp_mathlib_BinaryTree_numLeaves___redArg(v_right_428_);
v___x_431_ = lean_nat_add(v___x_429_, v___x_430_);
lean_dec(v___x_430_);
lean_dec(v___x_429_);
return v___x_431_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_numLeaves___redArg___boxed(lean_object* v_x_432_){
_start:
{
lean_object* v_res_433_; 
v_res_433_ = lp_mathlib_BinaryTree_numLeaves___redArg(v_x_432_);
lean_dec(v_x_432_);
return v_res_433_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_numLeaves(lean_object* v_00_u03b1_434_, lean_object* v_x_435_){
_start:
{
lean_object* v___x_436_; 
v___x_436_ = lp_mathlib_BinaryTree_numLeaves___redArg(v_x_435_);
return v___x_436_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_numLeaves___boxed(lean_object* v_00_u03b1_437_, lean_object* v_x_438_){
_start:
{
lean_object* v_res_439_; 
v_res_439_ = lp_mathlib_BinaryTree_numLeaves(v_00_u03b1_437_, v_x_438_);
lean_dec(v_x_438_);
return v_res_439_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_numLeaves___redArg(lean_object* v_t_440_){
_start:
{
lean_object* v___x_441_; 
v___x_441_ = lp_mathlib_BinaryTree_numLeaves___redArg(v_t_440_);
return v___x_441_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_numLeaves___redArg___boxed(lean_object* v_t_442_){
_start:
{
lean_object* v_res_443_; 
v_res_443_ = lp_mathlib_Tree_numLeaves___redArg(v_t_442_);
lean_dec(v_t_442_);
return v_res_443_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_numLeaves(lean_object* v_00_u03b1_444_, lean_object* v_t_445_){
_start:
{
lean_object* v___x_446_; 
v___x_446_ = lp_mathlib_BinaryTree_numLeaves___redArg(v_t_445_);
return v___x_446_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_numLeaves___boxed(lean_object* v_00_u03b1_447_, lean_object* v_t_448_){
_start:
{
lean_object* v_res_449_; 
v_res_449_ = lp_mathlib_Tree_numLeaves(v_00_u03b1_447_, v_t_448_);
lean_dec(v_t_448_);
return v_res_449_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_height___redArg(lean_object* v_x_450_){
_start:
{
lean_object* v___y_452_; 
if (lean_obj_tag(v_x_450_) == 0)
{
lean_object* v___x_455_; 
v___x_455_ = lean_unsigned_to_nat(0u);
return v___x_455_;
}
else
{
lean_object* v_left_456_; lean_object* v_right_457_; lean_object* v___x_458_; lean_object* v___x_459_; uint8_t v___x_460_; 
v_left_456_ = lean_ctor_get(v_x_450_, 1);
v_right_457_ = lean_ctor_get(v_x_450_, 2);
v___x_458_ = lp_mathlib_BinaryTree_height___redArg(v_left_456_);
v___x_459_ = lp_mathlib_BinaryTree_height___redArg(v_right_457_);
v___x_460_ = lean_nat_dec_le(v___x_458_, v___x_459_);
if (v___x_460_ == 0)
{
lean_dec(v___x_459_);
v___y_452_ = v___x_458_;
goto v___jp_451_;
}
else
{
lean_dec(v___x_458_);
v___y_452_ = v___x_459_;
goto v___jp_451_;
}
}
v___jp_451_:
{
lean_object* v___x_453_; lean_object* v___x_454_; 
v___x_453_ = lean_unsigned_to_nat(1u);
v___x_454_ = lean_nat_add(v___y_452_, v___x_453_);
lean_dec(v___y_452_);
return v___x_454_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_height___redArg___boxed(lean_object* v_x_461_){
_start:
{
lean_object* v_res_462_; 
v_res_462_ = lp_mathlib_BinaryTree_height___redArg(v_x_461_);
lean_dec(v_x_461_);
return v_res_462_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_height(lean_object* v_00_u03b1_463_, lean_object* v_x_464_){
_start:
{
lean_object* v___x_465_; 
v___x_465_ = lp_mathlib_BinaryTree_height___redArg(v_x_464_);
return v___x_465_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_height___boxed(lean_object* v_00_u03b1_466_, lean_object* v_x_467_){
_start:
{
lean_object* v_res_468_; 
v_res_468_ = lp_mathlib_BinaryTree_height(v_00_u03b1_466_, v_x_467_);
lean_dec(v_x_467_);
return v_res_468_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_height___redArg(lean_object* v_t_469_){
_start:
{
lean_object* v___x_470_; 
v___x_470_ = lp_mathlib_BinaryTree_height___redArg(v_t_469_);
return v___x_470_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_height___redArg___boxed(lean_object* v_t_471_){
_start:
{
lean_object* v_res_472_; 
v_res_472_ = lp_mathlib_Tree_height___redArg(v_t_471_);
lean_dec(v_t_471_);
return v_res_472_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_height(lean_object* v_00_u03b1_473_, lean_object* v_t_474_){
_start:
{
lean_object* v___x_475_; 
v___x_475_ = lp_mathlib_BinaryTree_height___redArg(v_t_474_);
return v___x_475_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_height___boxed(lean_object* v_00_u03b1_476_, lean_object* v_t_477_){
_start:
{
lean_object* v_res_478_; 
v_res_478_ = lp_mathlib_Tree_height(v_00_u03b1_476_, v_t_477_);
lean_dec(v_t_477_);
return v_res_478_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_left___redArg(lean_object* v_x_479_){
_start:
{
if (lean_obj_tag(v_x_479_) == 0)
{
return v_x_479_;
}
else
{
lean_object* v_left_480_; 
v_left_480_ = lean_ctor_get(v_x_479_, 1);
lean_inc(v_left_480_);
return v_left_480_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_left___redArg___boxed(lean_object* v_x_481_){
_start:
{
lean_object* v_res_482_; 
v_res_482_ = lp_mathlib_BinaryTree_left___redArg(v_x_481_);
lean_dec(v_x_481_);
return v_res_482_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_left(lean_object* v_00_u03b1_483_, lean_object* v_x_484_){
_start:
{
if (lean_obj_tag(v_x_484_) == 0)
{
return v_x_484_;
}
else
{
lean_object* v_left_485_; 
v_left_485_ = lean_ctor_get(v_x_484_, 1);
lean_inc(v_left_485_);
return v_left_485_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_left___boxed(lean_object* v_00_u03b1_486_, lean_object* v_x_487_){
_start:
{
lean_object* v_res_488_; 
v_res_488_ = lp_mathlib_BinaryTree_left(v_00_u03b1_486_, v_x_487_);
lean_dec(v_x_487_);
return v_res_488_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_left___redArg(lean_object* v_t_489_){
_start:
{
if (lean_obj_tag(v_t_489_) == 0)
{
return v_t_489_;
}
else
{
lean_object* v_left_490_; 
v_left_490_ = lean_ctor_get(v_t_489_, 1);
lean_inc(v_left_490_);
return v_left_490_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_left___redArg___boxed(lean_object* v_t_491_){
_start:
{
lean_object* v_res_492_; 
v_res_492_ = lp_mathlib_Tree_left___redArg(v_t_491_);
lean_dec(v_t_491_);
return v_res_492_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_left(lean_object* v_00_u03b1_493_, lean_object* v_t_494_){
_start:
{
if (lean_obj_tag(v_t_494_) == 0)
{
return v_t_494_;
}
else
{
lean_object* v_left_495_; 
v_left_495_ = lean_ctor_get(v_t_494_, 1);
lean_inc(v_left_495_);
return v_left_495_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_left___boxed(lean_object* v_00_u03b1_496_, lean_object* v_t_497_){
_start:
{
lean_object* v_res_498_; 
v_res_498_ = lp_mathlib_Tree_left(v_00_u03b1_496_, v_t_497_);
lean_dec(v_t_497_);
return v_res_498_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_right___redArg(lean_object* v_x_499_){
_start:
{
if (lean_obj_tag(v_x_499_) == 0)
{
return v_x_499_;
}
else
{
lean_object* v_right_500_; 
v_right_500_ = lean_ctor_get(v_x_499_, 2);
lean_inc(v_right_500_);
return v_right_500_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_right___redArg___boxed(lean_object* v_x_501_){
_start:
{
lean_object* v_res_502_; 
v_res_502_ = lp_mathlib_BinaryTree_right___redArg(v_x_501_);
lean_dec(v_x_501_);
return v_res_502_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_right(lean_object* v_00_u03b1_503_, lean_object* v_x_504_){
_start:
{
if (lean_obj_tag(v_x_504_) == 0)
{
return v_x_504_;
}
else
{
lean_object* v_right_505_; 
v_right_505_ = lean_ctor_get(v_x_504_, 2);
lean_inc(v_right_505_);
return v_right_505_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_right___boxed(lean_object* v_00_u03b1_506_, lean_object* v_x_507_){
_start:
{
lean_object* v_res_508_; 
v_res_508_ = lp_mathlib_BinaryTree_right(v_00_u03b1_506_, v_x_507_);
lean_dec(v_x_507_);
return v_res_508_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_right___redArg(lean_object* v_t_509_){
_start:
{
if (lean_obj_tag(v_t_509_) == 0)
{
return v_t_509_;
}
else
{
lean_object* v_right_510_; 
v_right_510_ = lean_ctor_get(v_t_509_, 2);
lean_inc(v_right_510_);
return v_right_510_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_right___redArg___boxed(lean_object* v_t_511_){
_start:
{
lean_object* v_res_512_; 
v_res_512_ = lp_mathlib_Tree_right___redArg(v_t_511_);
lean_dec(v_t_511_);
return v_res_512_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_right(lean_object* v_00_u03b1_513_, lean_object* v_t_514_){
_start:
{
if (lean_obj_tag(v_t_514_) == 0)
{
return v_t_514_;
}
else
{
lean_object* v_right_515_; 
v_right_515_ = lean_ctor_get(v_t_514_, 2);
lean_inc(v_right_515_);
return v_right_515_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_right___boxed(lean_object* v_00_u03b1_516_, lean_object* v_t_517_){
_start:
{
lean_object* v_res_518_; 
v_res_518_ = lp_mathlib_Tree_right(v_00_u03b1_516_, v_t_517_);
lean_dec(v_t_517_);
return v_res_518_;
}
}
static lean_object* _init_lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__5(void){
_start:
{
lean_object* v___x_555_; lean_object* v___x_556_; 
v___x_555_ = ((lean_object*)(lp_mathlib_instReprBinaryTree_repr___redArg___closed__4));
v___x_556_ = l_String_toRawSubstring_x27(v___x_555_);
return v___x_556_;
}
}
static lean_object* _init_lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__22(void){
_start:
{
lean_object* v___x_592_; lean_object* v___x_593_; 
v___x_592_ = ((lean_object*)(lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__21));
v___x_593_ = l_String_toRawSubstring_x27(v___x_592_);
return v___x_593_;
}
}
static lean_object* _init_lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__26(void){
_start:
{
lean_object* v___x_601_; 
v___x_601_ = l_Array_mkArray0(lean_box(0));
return v___x_601_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1(lean_object* v_x_603_, lean_object* v_a_604_, lean_object* v_a_605_){
_start:
{
lean_object* v___x_606_; uint8_t v___x_607_; 
v___x_606_ = ((lean_object*)(lp_mathlib_BinaryTree_term___u25b3___00__closed__2));
lean_inc(v_x_603_);
v___x_607_ = l_Lean_Syntax_isOfKind(v_x_603_, v___x_606_);
if (v___x_607_ == 0)
{
lean_object* v___x_608_; lean_object* v___x_609_; 
lean_dec(v_x_603_);
v___x_608_ = lean_box(1);
v___x_609_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_609_, 0, v___x_608_);
lean_ctor_set(v___x_609_, 1, v_a_605_);
return v___x_609_;
}
else
{
lean_object* v_quotContext_610_; lean_object* v_currMacroScope_611_; lean_object* v_ref_612_; lean_object* v___x_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; uint8_t v___x_617_; lean_object* v___x_618_; lean_object* v___x_619_; lean_object* v___x_620_; lean_object* v___x_621_; lean_object* v___x_622_; lean_object* v___x_623_; lean_object* v___x_624_; lean_object* v___x_625_; lean_object* v___x_626_; lean_object* v___x_627_; lean_object* v___x_628_; lean_object* v___x_629_; lean_object* v___x_630_; lean_object* v___x_631_; lean_object* v___x_632_; lean_object* v___x_633_; lean_object* v___x_634_; lean_object* v___x_635_; lean_object* v___x_636_; lean_object* v___x_637_; lean_object* v___x_638_; lean_object* v___x_639_; lean_object* v___x_640_; lean_object* v___x_641_; lean_object* v___x_642_; lean_object* v___x_643_; lean_object* v___x_644_; lean_object* v___x_645_; lean_object* v___x_646_; lean_object* v___x_647_; 
v_quotContext_610_ = lean_ctor_get(v_a_604_, 1);
v_currMacroScope_611_ = lean_ctor_get(v_a_604_, 2);
v_ref_612_ = lean_ctor_get(v_a_604_, 5);
v___x_613_ = lean_unsigned_to_nat(0u);
v___x_614_ = l_Lean_Syntax_getArg(v_x_603_, v___x_613_);
v___x_615_ = lean_unsigned_to_nat(2u);
v___x_616_ = l_Lean_Syntax_getArg(v_x_603_, v___x_615_);
lean_dec(v_x_603_);
v___x_617_ = 0;
v___x_618_ = l_Lean_SourceInfo_fromRef(v_ref_612_, v___x_617_);
v___x_619_ = ((lean_object*)(lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__4));
v___x_620_ = lean_obj_once(&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__5, &lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__5_once, _init_lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__5);
v___x_621_ = ((lean_object*)(lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__7));
lean_inc_n(v_currMacroScope_611_, 2);
lean_inc_n(v_quotContext_610_, 2);
v___x_622_ = l_Lean_addMacroScope(v_quotContext_610_, v___x_621_, v_currMacroScope_611_);
v___x_623_ = ((lean_object*)(lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__11));
lean_inc_n(v___x_618_, 11);
v___x_624_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_624_, 0, v___x_618_);
lean_ctor_set(v___x_624_, 1, v___x_620_);
lean_ctor_set(v___x_624_, 2, v___x_622_);
lean_ctor_set(v___x_624_, 3, v___x_623_);
v___x_625_ = ((lean_object*)(lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__13));
v___x_626_ = ((lean_object*)(lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__15));
v___x_627_ = ((lean_object*)(lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__17));
v___x_628_ = ((lean_object*)(lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__18));
v___x_629_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_629_, 0, v___x_618_);
lean_ctor_set(v___x_629_, 1, v___x_628_);
v___x_630_ = ((lean_object*)(lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__20));
v___x_631_ = lean_obj_once(&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__22, &lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__22_once, _init_lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__22);
v___x_632_ = lean_box(0);
v___x_633_ = l_Lean_addMacroScope(v_quotContext_610_, v___x_632_, v_currMacroScope_611_);
v___x_634_ = ((lean_object*)(lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__25));
v___x_635_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_635_, 0, v___x_618_);
lean_ctor_set(v___x_635_, 1, v___x_631_);
lean_ctor_set(v___x_635_, 2, v___x_633_);
lean_ctor_set(v___x_635_, 3, v___x_634_);
v___x_636_ = l_Lean_Syntax_node1(v___x_618_, v___x_630_, v___x_635_);
v___x_637_ = l_Lean_Syntax_node2(v___x_618_, v___x_627_, v___x_629_, v___x_636_);
v___x_638_ = lean_obj_once(&lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__26, &lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__26_once, _init_lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__26);
v___x_639_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_639_, 0, v___x_618_);
lean_ctor_set(v___x_639_, 1, v___x_625_);
lean_ctor_set(v___x_639_, 2, v___x_638_);
v___x_640_ = ((lean_object*)(lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___closed__27));
v___x_641_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_641_, 0, v___x_618_);
lean_ctor_set(v___x_641_, 1, v___x_640_);
v___x_642_ = l_Lean_Syntax_node3(v___x_618_, v___x_626_, v___x_637_, v___x_639_, v___x_641_);
v___x_643_ = l_Lean_Syntax_node1(v___x_618_, v___x_625_, v___x_642_);
v___x_644_ = l_Lean_Syntax_node2(v___x_618_, v___x_619_, v___x_624_, v___x_643_);
v___x_645_ = l_Lean_Syntax_node2(v___x_618_, v___x_625_, v___x_614_, v___x_616_);
v___x_646_ = l_Lean_Syntax_node2(v___x_618_, v___x_619_, v___x_644_, v___x_645_);
v___x_647_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_647_, 0, v___x_646_);
lean_ctor_set(v___x_647_, 1, v_a_605_);
return v___x_647_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1___boxed(lean_object* v_x_648_, lean_object* v_a_649_, lean_object* v_a_650_){
_start:
{
lean_object* v_res_651_; 
v_res_651_ = lp_mathlib_BinaryTree___aux__Mathlib__Data__Tree__Basic______macroRules__BinaryTree__term___u25b3____1(v_x_648_, v_a_649_, v_a_650_);
lean_dec_ref(v_a_649_);
return v_res_651_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_unitRecOn___redArg___lam__0(lean_object* v_ind_652_, lean_object* v___u_653_, lean_object* v___y_654_, lean_object* v___y_655_, lean_object* v___y_656_, lean_object* v___y_657_){
_start:
{
lean_object* v___x_658_; 
v___x_658_ = lean_apply_4(v_ind_652_, v___y_654_, v___y_655_, v___y_656_, v___y_657_);
return v___x_658_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_unitRecOn___redArg(lean_object* v_t_659_, lean_object* v_base_660_, lean_object* v_ind_661_){
_start:
{
lean_object* v___f_662_; lean_object* v___x_663_; 
v___f_662_ = lean_alloc_closure((void*)(lp_mathlib_BinaryTree_unitRecOn___redArg___lam__0), 6, 1);
lean_closure_set(v___f_662_, 0, v_ind_661_);
v___x_663_ = lp_mathlib_BinaryTree_rec___redArg_00___x40_Mathlib_Data_Tree_Basic_126925257____hygCtx___hyg_3_(v_base_660_, v___f_662_, v_t_659_);
return v___x_663_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_unitRecOn___redArg___boxed(lean_object* v_t_664_, lean_object* v_base_665_, lean_object* v_ind_666_){
_start:
{
lean_object* v_res_667_; 
v_res_667_ = lp_mathlib_BinaryTree_unitRecOn___redArg(v_t_664_, v_base_665_, v_ind_666_);
lean_dec(v_base_665_);
return v_res_667_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_unitRecOn(lean_object* v_motive_668_, lean_object* v_t_669_, lean_object* v_base_670_, lean_object* v_ind_671_){
_start:
{
lean_object* v___x_672_; 
v___x_672_ = lp_mathlib_BinaryTree_unitRecOn___redArg(v_t_669_, v_base_670_, v_ind_671_);
return v___x_672_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BinaryTree_unitRecOn___boxed(lean_object* v_motive_673_, lean_object* v_t_674_, lean_object* v_base_675_, lean_object* v_ind_676_){
_start:
{
lean_object* v_res_677_; 
v_res_677_ = lp_mathlib_BinaryTree_unitRecOn(v_motive_673_, v_t_674_, v_base_675_, v_ind_676_);
lean_dec(v_base_675_);
return v_res_677_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_unitRecOn___redArg(lean_object* v_t_678_, lean_object* v_base_679_, lean_object* v_ind_680_){
_start:
{
lean_object* v___x_681_; 
v___x_681_ = lp_mathlib_BinaryTree_unitRecOn___redArg(v_t_678_, v_base_679_, v_ind_680_);
return v___x_681_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_unitRecOn___redArg___boxed(lean_object* v_t_682_, lean_object* v_base_683_, lean_object* v_ind_684_){
_start:
{
lean_object* v_res_685_; 
v_res_685_ = lp_mathlib_Tree_unitRecOn___redArg(v_t_682_, v_base_683_, v_ind_684_);
lean_dec(v_base_683_);
return v_res_685_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_unitRecOn(lean_object* v_motive_686_, lean_object* v_t_687_, lean_object* v_base_688_, lean_object* v_ind_689_){
_start:
{
lean_object* v___x_690_; 
v___x_690_ = lp_mathlib_BinaryTree_unitRecOn___redArg(v_t_687_, v_base_688_, v_ind_689_);
return v___x_690_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Tree_unitRecOn___boxed(lean_object* v_motive_691_, lean_object* v_t_692_, lean_object* v_base_693_, lean_object* v_ind_694_){
_start:
{
lean_object* v_res_695_; 
v_res_695_ = lp_mathlib_Tree_unitRecOn(v_motive_691_, v_t_692_, v_base_693_, v_ind_694_);
lean_dec(v_base_693_);
return v_res_695_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_CompileInductive(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Tree_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_CompileInductive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Tree_Basic(uint8_t builtin) {
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
lean_object* initialize_mathlib_Mathlib_Data_Nat_Notation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_CompileInductive(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Tree_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_CompileInductive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Tree_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Tree_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Tree_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
