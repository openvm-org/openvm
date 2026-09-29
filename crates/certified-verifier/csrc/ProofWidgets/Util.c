// Lean compiler output
// Module: ProofWidgets.Util
// Imports: public import Init public meta import Init public meta import Lean.PrettyPrinter.Delaborator.Basic
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
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Array_mkArray0(lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* lean_array_fget(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_Expr_getAppNumArgs(lean_object*);
extern lean_object* l_Lean_instInhabitedExpr;
lean_object* l_Lean_Expr_sort___override(lean_object*);
lean_object* lean_mk_array(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_SubExpr_Pos_pushNaryArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_get(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(lean_object*, lean_object*);
lean_object* l_Lean_Expr_cleanupAnnotations(lean_object*);
uint8_t l_Lean_Expr_isApp(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_failure___redArg();
lean_object* l_Lean_Expr_appFnCleanup___redArg(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
uint8_t l_Lean_Expr_isConstOf(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_annotateCurPos___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_addTermInfo___redArg(lean_object*, lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_ST_Prim_mkRef___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_ST_Prim_Ref_get___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_array_size(lean_object*);
lean_object* l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, size_t, size_t, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "term#[_,]"};
static const lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(69, 119, 178, 128, 145, 112, 206, 247)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__1_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "#["};
static const lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__2_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__3 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__3_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__4 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__4_value;
static lean_once_cell_t lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__5_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__5;
static const lean_string_object lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "]"};
static const lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__6 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__6_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__3(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "term_++_"};
static const lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4___closed__0_value),LEAN_SCALAR_PTR_LITERAL(90, 69, 86, 178, 149, 48, 216, 23)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4___closed__1_value;
static const lean_string_object lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "++"};
static const lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4___closed__2 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4___closed__2_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__5(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__5___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_array_object lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___closed__0 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___closed__0_value;
static const lean_ctor_object lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 0, .m_other = 2, .m_tag = 0}, .m_objs = {((lean_object*)&lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___closed__0_value),((lean_object*)&lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___closed__0_value)}};
static const lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___closed__1 = (const lean_object*)&lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1_spec__1___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_once_cell_t lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg___closed__0_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg___closed__0;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 4, .m_capacity = 4, .m_length = 3, .m_data = "nil"};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__1 = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__1_value;
static const lean_string_object lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "List"};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__0 = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__0_value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__2_value_aux_0),((lean_object*)&lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__1_value),LEAN_SCALAR_PTR_LITERAL(90, 150, 134, 113, 145, 38, 173, 251)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__2 = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__2_value;
static const lean_string_object lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "cons"};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__3 = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__3_value;
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__4_value_aux_0),((lean_object*)&lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__3_value),LEAN_SCALAR_PTR_LITERAL(98, 170, 59, 223, 79, 132, 139, 119)}};
static const lean_object* lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__4 = (const lean_object*)&lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__4_value;
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__3___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__3___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabListLiteral___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabListLiteral___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabListLiteral(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabListLiteral___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00Lean_PrettyPrinter_Delaborator_delabArrayLiteral_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00Lean_PrettyPrinter_Delaborator_delabArrayLiteral_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00Lean_PrettyPrinter_Delaborator_delabArrayLiteral_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00Lean_PrettyPrinter_Delaborator_delabArrayLiteral_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 8, .m_capacity = 8, .m_length = 7, .m_data = "toArray"};
static const lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg___closed__0 = (const lean_object*)&lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg___closed__0_value;
static const lean_ctor_object lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg___closed__1_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(245, 188, 225, 225, 165, 5, 251, 132)}};
static const lean_ctor_object lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg___closed__1_value_aux_0),((lean_object*)&lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg___closed__0_value),LEAN_SCALAR_PTR_LITERAL(225, 54, 189, 64, 249, 49, 198, 116)}};
static const lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg___closed__1 = (const lean_object*)&lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg___closed__1_value;
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_withAnnotateTermLikeInfo___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_withAnnotateTermLikeInfo___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_withAnnotateTermLikeInfo(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_withAnnotateTermLikeInfo___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtx___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtx___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtx(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxReaderT___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxReaderT___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxReaderT(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateT___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateT___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateT___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateT___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateT___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_proofwidgets_instMonadSaveCtxStateT___redArg___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_instMonadSaveCtxStateT___redArg___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_instMonadSaveCtxStateT___redArg___closed__0 = (const lean_object*)&lp_proofwidgets_instMonadSaveCtxStateT___redArg___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateT___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateT(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instToJsonUnit__proofWidgets___lam__0(lean_object*);
static const lean_closure_object lp_proofwidgets_instToJsonUnit__proofWidgets___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_instToJsonUnit__proofWidgets___lam__0, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_instToJsonUnit__proofWidgets___closed__0 = (const lean_object*)&lp_proofwidgets_instToJsonUnit__proofWidgets___closed__0_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_instToJsonUnit__proofWidgets = (const lean_object*)&lp_proofwidgets_instToJsonUnit__proofWidgets___closed__0_value;
static const lean_ctor_object lp_proofwidgets_instFromJsonUnit__proofWidgets___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_proofwidgets_instFromJsonUnit__proofWidgets___lam__0___closed__0 = (const lean_object*)&lp_proofwidgets_instFromJsonUnit__proofWidgets___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_proofwidgets_instFromJsonUnit__proofWidgets___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_instFromJsonUnit__proofWidgets___lam__0___boxed(lean_object*);
static const lean_closure_object lp_proofwidgets_instFromJsonUnit__proofWidgets___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_proofwidgets_instFromJsonUnit__proofWidgets___lam__0___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_proofwidgets_instFromJsonUnit__proofWidgets___closed__0 = (const lean_object*)&lp_proofwidgets_instFromJsonUnit__proofWidgets___closed__0_value;
LEAN_EXPORT const lean_object* lp_proofwidgets_instFromJsonUnit__proofWidgets = (const lean_object*)&lp_proofwidgets_instFromJsonUnit__proofWidgets___closed__0_value;
static lean_object* _init_lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__5(void){
_start:
{
lean_object* v___x_8_; 
v___x_8_ = l_Array_mkArray0(lean_box(0));
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0(lean_object* v_info_10_, lean_object* v_toPure_11_, lean_object* v_quotCtx_12_){
_start:
{
lean_object* v___x_13_; lean_object* v___x_14_; lean_object* v___x_15_; lean_object* v___x_16_; lean_object* v___x_17_; lean_object* v___x_18_; lean_object* v___x_19_; lean_object* v___x_20_; lean_object* v___x_21_; lean_object* v___x_22_; 
v___x_13_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__1));
v___x_14_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__2));
lean_inc_n(v_info_10_, 3);
v___x_15_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_15_, 0, v_info_10_);
lean_ctor_set(v___x_15_, 1, v___x_14_);
v___x_16_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__4));
v___x_17_ = lean_obj_once(&lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__5, &lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__5_once, _init_lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__5);
v___x_18_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_18_, 0, v_info_10_);
lean_ctor_set(v___x_18_, 1, v___x_16_);
lean_ctor_set(v___x_18_, 2, v___x_17_);
v___x_19_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___closed__6));
v___x_20_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_20_, 0, v_info_10_);
lean_ctor_set(v___x_20_, 1, v___x_19_);
v___x_21_ = l_Lean_Syntax_node3(v_info_10_, v___x_13_, v___x_15_, v___x_18_, v___x_20_);
v___x_22_ = lean_apply_2(v_toPure_11_, lean_box(0), v___x_21_);
return v___x_22_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___boxed(lean_object* v_info_23_, lean_object* v_toPure_24_, lean_object* v_quotCtx_25_){
_start:
{
lean_object* v_res_26_; 
v_res_26_ = lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0(v_info_23_, v_toPure_24_, v_quotCtx_25_);
lean_dec(v_quotCtx_25_);
return v_res_26_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__1(lean_object* v_toBind_27_, lean_object* v_getContext_28_, lean_object* v___f_29_, lean_object* v_scp_30_){
_start:
{
lean_object* v___x_31_; 
v___x_31_ = lean_apply_4(v_toBind_27_, lean_box(0), lean_box(0), v_getContext_28_, v___f_29_);
return v___x_31_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__1___boxed(lean_object* v_toBind_32_, lean_object* v_getContext_33_, lean_object* v___f_34_, lean_object* v_scp_35_){
_start:
{
lean_object* v_res_36_; 
v_res_36_ = lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__1(v_toBind_32_, v_getContext_33_, v___f_34_, v_scp_35_);
lean_dec(v_scp_35_);
return v_res_36_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__2(lean_object* v_inst_37_, lean_object* v_toPure_38_, lean_object* v_toBind_39_, lean_object* v_info_40_){
_start:
{
lean_object* v_getCurrMacroScope_41_; lean_object* v_getContext_42_; lean_object* v___f_43_; lean_object* v___f_44_; lean_object* v___x_45_; 
v_getCurrMacroScope_41_ = lean_ctor_get(v_inst_37_, 1);
lean_inc(v_getCurrMacroScope_41_);
v_getContext_42_ = lean_ctor_get(v_inst_37_, 2);
lean_inc(v_getContext_42_);
lean_dec_ref(v_inst_37_);
v___f_43_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__0___boxed), 3, 2);
lean_closure_set(v___f_43_, 0, v_info_40_);
lean_closure_set(v___f_43_, 1, v_toPure_38_);
lean_inc(v_toBind_39_);
v___f_44_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_44_, 0, v_toBind_39_);
lean_closure_set(v___f_44_, 1, v_getContext_42_);
lean_closure_set(v___f_44_, 2, v___f_43_);
v___x_45_ = lean_apply_4(v_toBind_39_, lean_box(0), lean_box(0), v_getCurrMacroScope_41_, v___f_44_);
return v___x_45_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__3(uint8_t v___x_46_, lean_object* v_toPure_47_, lean_object* v_____do__lift_48_){
_start:
{
lean_object* v___x_49_; lean_object* v___x_50_; 
v___x_49_ = l_Lean_SourceInfo_fromRef(v_____do__lift_48_, v___x_46_);
v___x_50_ = lean_apply_2(v_toPure_47_, lean_box(0), v___x_49_);
return v___x_50_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__3___boxed(lean_object* v___x_51_, lean_object* v_toPure_52_, lean_object* v_____do__lift_53_){
_start:
{
uint8_t v___x_266__boxed_54_; lean_object* v_res_55_; 
v___x_266__boxed_54_ = lean_unbox(v___x_51_);
v_res_55_ = lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__3(v___x_266__boxed_54_, v_toPure_52_, v_____do__lift_53_);
lean_dec(v_____do__lift_53_);
return v_res_55_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4(lean_object* v_info_60_, lean_object* v_x_61_, lean_object* v_xs_62_, lean_object* v_toPure_63_, lean_object* v_quotCtx_64_){
_start:
{
lean_object* v___x_65_; lean_object* v___x_66_; lean_object* v___x_67_; lean_object* v___x_68_; lean_object* v___x_69_; 
v___x_65_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4___closed__1));
v___x_66_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4___closed__2));
lean_inc(v_info_60_);
v___x_67_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_67_, 0, v_info_60_);
lean_ctor_set(v___x_67_, 1, v___x_66_);
v___x_68_ = l_Lean_Syntax_node3(v_info_60_, v___x_65_, v_x_61_, v___x_67_, v_xs_62_);
v___x_69_ = lean_apply_2(v_toPure_63_, lean_box(0), v___x_68_);
return v___x_69_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4___boxed(lean_object* v_info_70_, lean_object* v_x_71_, lean_object* v_xs_72_, lean_object* v_toPure_73_, lean_object* v_quotCtx_74_){
_start:
{
lean_object* v_res_75_; 
v_res_75_ = lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4(v_info_70_, v_x_71_, v_xs_72_, v_toPure_73_, v_quotCtx_74_);
lean_dec(v_quotCtx_74_);
return v_res_75_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__6(lean_object* v_inst_76_, lean_object* v_x_77_, lean_object* v_xs_78_, lean_object* v_toPure_79_, lean_object* v_toBind_80_, lean_object* v_info_81_){
_start:
{
lean_object* v_getCurrMacroScope_82_; lean_object* v_getContext_83_; lean_object* v___f_84_; lean_object* v___f_85_; lean_object* v___x_86_; 
v_getCurrMacroScope_82_ = lean_ctor_get(v_inst_76_, 1);
lean_inc(v_getCurrMacroScope_82_);
v_getContext_83_ = lean_ctor_get(v_inst_76_, 2);
lean_inc(v_getContext_83_);
lean_dec_ref(v_inst_76_);
v___f_84_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__4___boxed), 5, 4);
lean_closure_set(v___f_84_, 0, v_info_81_);
lean_closure_set(v___f_84_, 1, v_x_77_);
lean_closure_set(v___f_84_, 2, v_xs_78_);
lean_closure_set(v___f_84_, 3, v_toPure_79_);
lean_inc(v_toBind_80_);
v___f_85_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__1___boxed), 4, 3);
lean_closure_set(v___f_85_, 0, v_toBind_80_);
lean_closure_set(v___f_85_, 1, v_getContext_83_);
lean_closure_set(v___f_85_, 2, v___f_84_);
v___x_86_ = lean_apply_4(v_toBind_80_, lean_box(0), lean_box(0), v_getCurrMacroScope_82_, v___f_85_);
return v___x_86_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__5(lean_object* v_toPure_87_, lean_object* v_____do__lift_88_){
_start:
{
uint8_t v___x_89_; lean_object* v___x_90_; lean_object* v___x_91_; 
v___x_89_ = 0;
v___x_90_ = l_Lean_SourceInfo_fromRef(v_____do__lift_88_, v___x_89_);
v___x_91_ = lean_apply_2(v_toPure_87_, lean_box(0), v___x_90_);
return v___x_91_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__5___boxed(lean_object* v_toPure_92_, lean_object* v_____do__lift_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__5(v_toPure_92_, v_____do__lift_93_);
lean_dec(v_____do__lift_93_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__7(lean_object* v_inst_95_, lean_object* v_inst_96_, lean_object* v_inst_97_, lean_object* v_x_98_, lean_object* v_xs_99_){
_start:
{
lean_object* v_toApplicative_100_; lean_object* v_toBind_101_; lean_object* v_getRef_102_; lean_object* v_toPure_103_; lean_object* v___f_104_; lean_object* v___f_105_; lean_object* v___x_106_; lean_object* v___x_107_; 
v_toApplicative_100_ = lean_ctor_get(v_inst_95_, 0);
lean_inc_ref(v_toApplicative_100_);
v_toBind_101_ = lean_ctor_get(v_inst_95_, 1);
lean_inc_n(v_toBind_101_, 3);
lean_dec_ref(v_inst_95_);
v_getRef_102_ = lean_ctor_get(v_inst_96_, 0);
lean_inc(v_getRef_102_);
lean_dec_ref(v_inst_96_);
v_toPure_103_ = lean_ctor_get(v_toApplicative_100_, 1);
lean_inc_n(v_toPure_103_, 2);
lean_dec_ref(v_toApplicative_100_);
v___f_104_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__6), 6, 5);
lean_closure_set(v___f_104_, 0, v_inst_97_);
lean_closure_set(v___f_104_, 1, v_x_98_);
lean_closure_set(v___f_104_, 2, v_xs_99_);
lean_closure_set(v___f_104_, 3, v_toPure_103_);
lean_closure_set(v___f_104_, 4, v_toBind_101_);
v___f_105_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__5___boxed), 2, 1);
lean_closure_set(v___f_105_, 0, v_toPure_103_);
v___x_106_ = lean_apply_4(v_toBind_101_, lean_box(0), lean_box(0), v_getRef_102_, v___f_105_);
v___x_107_ = lean_apply_4(v_toBind_101_, lean_box(0), lean_box(0), v___x_106_, v___f_104_);
return v___x_107_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg(lean_object* v_inst_108_, lean_object* v_inst_109_, lean_object* v_inst_110_, lean_object* v_arr_111_){
_start:
{
lean_object* v___x_112_; lean_object* v___x_113_; uint8_t v___x_114_; 
v___x_112_ = lean_unsigned_to_nat(0u);
v___x_113_ = lean_array_get_size(v_arr_111_);
v___x_114_ = lean_nat_dec_lt(v___x_112_, v___x_113_);
if (v___x_114_ == 0)
{
lean_object* v_toApplicative_115_; lean_object* v_toBind_116_; lean_object* v_getRef_117_; lean_object* v_toPure_118_; lean_object* v___f_119_; lean_object* v___x_120_; lean_object* v___f_121_; lean_object* v___x_122_; lean_object* v___x_123_; 
lean_dec_ref(v_arr_111_);
v_toApplicative_115_ = lean_ctor_get(v_inst_108_, 0);
lean_inc_ref(v_toApplicative_115_);
v_toBind_116_ = lean_ctor_get(v_inst_108_, 1);
lean_inc_n(v_toBind_116_, 3);
lean_dec_ref(v_inst_108_);
v_getRef_117_ = lean_ctor_get(v_inst_109_, 0);
lean_inc(v_getRef_117_);
lean_dec_ref(v_inst_109_);
v_toPure_118_ = lean_ctor_get(v_toApplicative_115_, 1);
lean_inc_n(v_toPure_118_, 2);
lean_dec_ref(v_toApplicative_115_);
v___f_119_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__2), 4, 3);
lean_closure_set(v___f_119_, 0, v_inst_110_);
lean_closure_set(v___f_119_, 1, v_toPure_118_);
lean_closure_set(v___f_119_, 2, v_toBind_116_);
v___x_120_ = lean_box(v___x_114_);
v___f_121_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__3___boxed), 3, 2);
lean_closure_set(v___f_121_, 0, v___x_120_);
lean_closure_set(v___f_121_, 1, v_toPure_118_);
v___x_122_ = lean_apply_4(v_toBind_116_, lean_box(0), lean_box(0), v_getRef_117_, v___f_121_);
v___x_123_ = lean_apply_4(v_toBind_116_, lean_box(0), lean_box(0), v___x_122_, v___f_119_);
return v___x_123_;
}
else
{
lean_object* v___x_124_; lean_object* v___x_125_; uint8_t v___x_126_; 
v___x_124_ = lean_array_fget(v_arr_111_, v___x_112_);
v___x_125_ = lean_unsigned_to_nat(1u);
v___x_126_ = lean_nat_dec_lt(v___x_125_, v___x_113_);
if (v___x_126_ == 0)
{
lean_object* v_toApplicative_127_; lean_object* v_toPure_128_; lean_object* v___x_129_; 
lean_dec_ref(v_arr_111_);
lean_dec_ref(v_inst_110_);
lean_dec_ref(v_inst_109_);
v_toApplicative_127_ = lean_ctor_get(v_inst_108_, 0);
lean_inc_ref(v_toApplicative_127_);
lean_dec_ref(v_inst_108_);
v_toPure_128_ = lean_ctor_get(v_toApplicative_127_, 1);
lean_inc(v_toPure_128_);
lean_dec_ref(v_toApplicative_127_);
v___x_129_ = lean_apply_2(v_toPure_128_, lean_box(0), v___x_124_);
return v___x_129_;
}
else
{
lean_object* v___f_130_; uint8_t v___x_131_; 
lean_inc_ref(v_inst_108_);
v___f_130_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg___lam__7), 5, 3);
lean_closure_set(v___f_130_, 0, v_inst_108_);
lean_closure_set(v___f_130_, 1, v_inst_109_);
lean_closure_set(v___f_130_, 2, v_inst_110_);
v___x_131_ = lean_nat_dec_le(v___x_113_, v___x_113_);
if (v___x_131_ == 0)
{
if (v___x_126_ == 0)
{
lean_object* v_toApplicative_132_; lean_object* v_toPure_133_; lean_object* v___x_134_; 
lean_dec_ref(v___f_130_);
lean_dec_ref(v_arr_111_);
v_toApplicative_132_ = lean_ctor_get(v_inst_108_, 0);
lean_inc_ref(v_toApplicative_132_);
lean_dec_ref(v_inst_108_);
v_toPure_133_ = lean_ctor_get(v_toApplicative_132_, 1);
lean_inc(v_toPure_133_);
lean_dec_ref(v_toApplicative_132_);
v___x_134_ = lean_apply_2(v_toPure_133_, lean_box(0), v___x_124_);
return v___x_134_;
}
else
{
size_t v___x_135_; size_t v___x_136_; lean_object* v___x_137_; 
v___x_135_ = ((size_t)1ULL);
v___x_136_ = lean_usize_of_nat(v___x_113_);
v___x_137_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_108_, v___f_130_, v_arr_111_, v___x_135_, v___x_136_, v___x_124_);
return v___x_137_;
}
}
else
{
size_t v___x_138_; size_t v___x_139_; lean_object* v___x_140_; 
v___x_138_ = ((size_t)1ULL);
v___x_139_ = lean_usize_of_nat(v___x_113_);
v___x_140_ = l___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold(lean_box(0), lean_box(0), lean_box(0), v_inst_108_, v___f_130_, v_arr_111_, v___x_138_, v___x_139_, v___x_124_);
return v___x_140_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_joinArrays(lean_object* v_m_141_, lean_object* v_inst_142_, lean_object* v_inst_143_, lean_object* v_inst_144_, lean_object* v_arr_145_){
_start:
{
lean_object* v___x_146_; 
v___x_146_ = lp_proofwidgets_ProofWidgets_Util_joinArrays___redArg(v_inst_142_, v_inst_143_, v_inst_144_, v_arr_145_);
return v___x_146_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__0(lean_object* v_val_147_, lean_object* v_pending__inls_148_, lean_object* v_toPure_149_, lean_object* v_____r_150_, lean_object* v_ret_151_){
_start:
{
lean_object* v___x_152_; lean_object* v___x_153_; lean_object* v___x_154_; lean_object* v___x_155_; 
v___x_152_ = lean_array_push(v_ret_151_, v_val_147_);
v___x_153_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_153_, 0, v___x_152_);
lean_ctor_set(v___x_153_, 1, v_pending__inls_148_);
v___x_154_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v___x_154_, 0, v___x_153_);
v___x_155_ = lean_apply_2(v_toPure_149_, lean_box(0), v___x_154_);
return v___x_155_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__1(lean_object* v_fst_156_, lean_object* v___f_157_, lean_object* v_____do__lift_158_){
_start:
{
lean_object* v___x_159_; lean_object* v___x_160_; lean_object* v___x_161_; 
v___x_159_ = lean_array_push(v_fst_156_, v_____do__lift_158_);
v___x_160_ = lean_box(0);
v___x_161_ = lean_apply_2(v___f_157_, v___x_160_, v___x_159_);
return v___x_161_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__2(lean_object* v_toPure_162_, lean_object* v_pending__inls_163_, lean_object* v___x_164_, lean_object* v_f_165_, lean_object* v_toBind_166_, lean_object* v_a_167_, lean_object* v_x_168_, lean_object* v___y_169_){
_start:
{
if (lean_obj_tag(v_a_167_) == 0)
{
lean_object* v_fst_170_; lean_object* v_snd_171_; lean_object* v___x_173_; uint8_t v_isShared_174_; uint8_t v_isSharedCheck_188_; 
lean_dec(v_toBind_166_);
lean_dec(v_f_165_);
lean_dec_ref(v_pending__inls_163_);
v_fst_170_ = lean_ctor_get(v___y_169_, 0);
v_snd_171_ = lean_ctor_get(v___y_169_, 1);
v_isSharedCheck_188_ = !lean_is_exclusive(v___y_169_);
if (v_isSharedCheck_188_ == 0)
{
v___x_173_ = v___y_169_;
v_isShared_174_ = v_isSharedCheck_188_;
goto v_resetjp_172_;
}
else
{
lean_inc(v_snd_171_);
lean_inc(v_fst_170_);
lean_dec(v___y_169_);
v___x_173_ = lean_box(0);
v_isShared_174_ = v_isSharedCheck_188_;
goto v_resetjp_172_;
}
v_resetjp_172_:
{
lean_object* v_val_175_; lean_object* v___x_177_; uint8_t v_isShared_178_; uint8_t v_isSharedCheck_187_; 
v_val_175_ = lean_ctor_get(v_a_167_, 0);
v_isSharedCheck_187_ = !lean_is_exclusive(v_a_167_);
if (v_isSharedCheck_187_ == 0)
{
v___x_177_ = v_a_167_;
v_isShared_178_ = v_isSharedCheck_187_;
goto v_resetjp_176_;
}
else
{
lean_inc(v_val_175_);
lean_dec(v_a_167_);
v___x_177_ = lean_box(0);
v_isShared_178_ = v_isSharedCheck_187_;
goto v_resetjp_176_;
}
v_resetjp_176_:
{
lean_object* v___x_179_; lean_object* v___x_181_; 
v___x_179_ = lean_array_push(v_snd_171_, v_val_175_);
if (v_isShared_174_ == 0)
{
lean_ctor_set(v___x_173_, 1, v___x_179_);
v___x_181_ = v___x_173_;
goto v_reusejp_180_;
}
else
{
lean_object* v_reuseFailAlloc_186_; 
v_reuseFailAlloc_186_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_186_, 0, v_fst_170_);
lean_ctor_set(v_reuseFailAlloc_186_, 1, v___x_179_);
v___x_181_ = v_reuseFailAlloc_186_;
goto v_reusejp_180_;
}
v_reusejp_180_:
{
lean_object* v___x_183_; 
if (v_isShared_178_ == 0)
{
lean_ctor_set_tag(v___x_177_, 1);
lean_ctor_set(v___x_177_, 0, v___x_181_);
v___x_183_ = v___x_177_;
goto v_reusejp_182_;
}
else
{
lean_object* v_reuseFailAlloc_185_; 
v_reuseFailAlloc_185_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_185_, 0, v___x_181_);
v___x_183_ = v_reuseFailAlloc_185_;
goto v_reusejp_182_;
}
v_reusejp_182_:
{
lean_object* v___x_184_; 
v___x_184_ = lean_apply_2(v_toPure_162_, lean_box(0), v___x_183_);
return v___x_184_;
}
}
}
}
}
else
{
lean_object* v_fst_189_; lean_object* v_snd_190_; lean_object* v_val_191_; lean_object* v___f_192_; lean_object* v___x_193_; uint8_t v___x_194_; 
v_fst_189_ = lean_ctor_get(v___y_169_, 0);
lean_inc(v_fst_189_);
v_snd_190_ = lean_ctor_get(v___y_169_, 1);
lean_inc(v_snd_190_);
lean_dec_ref(v___y_169_);
v_val_191_ = lean_ctor_get(v_a_167_, 0);
lean_inc_n(v_val_191_, 2);
lean_dec_ref_known(v_a_167_, 1);
lean_inc(v_toPure_162_);
lean_inc_ref(v_pending__inls_163_);
v___f_192_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__0), 5, 3);
lean_closure_set(v___f_192_, 0, v_val_191_);
lean_closure_set(v___f_192_, 1, v_pending__inls_163_);
lean_closure_set(v___f_192_, 2, v_toPure_162_);
v___x_193_ = lean_array_get_size(v_snd_190_);
v___x_194_ = lean_nat_dec_eq(v___x_193_, v___x_164_);
if (v___x_194_ == 0)
{
lean_object* v___f_195_; lean_object* v___x_196_; lean_object* v___x_197_; 
lean_dec(v_val_191_);
lean_dec_ref(v_pending__inls_163_);
lean_dec(v_toPure_162_);
v___f_195_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__1), 3, 2);
lean_closure_set(v___f_195_, 0, v_fst_189_);
lean_closure_set(v___f_195_, 1, v___f_192_);
v___x_196_ = lean_apply_1(v_f_165_, v_snd_190_);
v___x_197_ = lean_apply_4(v_toBind_166_, lean_box(0), lean_box(0), v___x_196_, v___f_195_);
return v___x_197_;
}
else
{
lean_object* v___x_198_; lean_object* v___x_199_; 
lean_dec_ref(v___f_192_);
lean_dec(v_snd_190_);
lean_dec(v_toBind_166_);
lean_dec(v_f_165_);
v___x_198_ = lean_box(0);
v___x_199_ = lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__0(v_val_191_, v_pending__inls_163_, v_toPure_162_, v___x_198_, v_fst_189_);
return v___x_199_;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__2___boxed(lean_object* v_toPure_200_, lean_object* v_pending__inls_201_, lean_object* v___x_202_, lean_object* v_f_203_, lean_object* v_toBind_204_, lean_object* v_a_205_, lean_object* v_x_206_, lean_object* v___y_207_){
_start:
{
lean_object* v_res_208_; 
v_res_208_ = lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__2(v_toPure_200_, v_pending__inls_201_, v___x_202_, v_f_203_, v_toBind_204_, v_a_205_, v_x_206_, v___y_207_);
lean_dec(v___x_202_);
return v_res_208_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__3(lean_object* v_fst_209_, lean_object* v_toPure_210_, lean_object* v_____do__lift_211_){
_start:
{
lean_object* v_ret_212_; lean_object* v___x_213_; 
v_ret_212_ = lean_array_push(v_fst_209_, v_____do__lift_211_);
v___x_213_ = lean_apply_2(v_toPure_210_, lean_box(0), v_ret_212_);
return v___x_213_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__4(lean_object* v___x_214_, lean_object* v_toPure_215_, lean_object* v_f_216_, lean_object* v_toBind_217_, lean_object* v_____s_218_){
_start:
{
lean_object* v_fst_219_; lean_object* v_snd_220_; lean_object* v___x_221_; uint8_t v___x_222_; 
v_fst_219_ = lean_ctor_get(v_____s_218_, 0);
lean_inc(v_fst_219_);
v_snd_220_ = lean_ctor_get(v_____s_218_, 1);
lean_inc(v_snd_220_);
lean_dec_ref(v_____s_218_);
v___x_221_ = lean_array_get_size(v_snd_220_);
v___x_222_ = lean_nat_dec_eq(v___x_221_, v___x_214_);
if (v___x_222_ == 0)
{
lean_object* v___f_223_; lean_object* v___x_224_; lean_object* v___x_225_; 
v___f_223_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__3), 3, 2);
lean_closure_set(v___f_223_, 0, v_fst_219_);
lean_closure_set(v___f_223_, 1, v_toPure_215_);
v___x_224_ = lean_apply_1(v_f_216_, v_snd_220_);
v___x_225_ = lean_apply_4(v_toBind_217_, lean_box(0), lean_box(0), v___x_224_, v___f_223_);
return v___x_225_;
}
else
{
lean_object* v___x_226_; 
lean_dec(v_snd_220_);
lean_dec(v_toBind_217_);
lean_dec(v_f_216_);
v___x_226_ = lean_apply_2(v_toPure_215_, lean_box(0), v_fst_219_);
return v___x_226_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__4___boxed(lean_object* v___x_227_, lean_object* v_toPure_228_, lean_object* v_f_229_, lean_object* v_toBind_230_, lean_object* v_____s_231_){
_start:
{
lean_object* v_res_232_; 
v_res_232_ = lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__4(v___x_227_, v_toPure_228_, v_f_229_, v_toBind_230_, v_____s_231_);
lean_dec(v___x_227_);
return v_res_232_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg(lean_object* v_inst_237_, lean_object* v_arr_238_, lean_object* v_f_239_){
_start:
{
lean_object* v_toApplicative_240_; lean_object* v_toBind_241_; lean_object* v_toPure_242_; lean_object* v___x_243_; lean_object* v_ret_244_; lean_object* v___x_245_; lean_object* v___f_246_; lean_object* v___f_247_; size_t v_sz_248_; size_t v___x_249_; lean_object* v___x_250_; lean_object* v___x_251_; 
v_toApplicative_240_ = lean_ctor_get(v_inst_237_, 0);
v_toBind_241_ = lean_ctor_get(v_inst_237_, 1);
lean_inc_n(v_toBind_241_, 3);
v_toPure_242_ = lean_ctor_get(v_toApplicative_240_, 1);
v___x_243_ = lean_unsigned_to_nat(0u);
v_ret_244_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___closed__0));
v___x_245_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___closed__1));
lean_inc(v_f_239_);
lean_inc_n(v_toPure_242_, 2);
v___f_246_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__2___boxed), 8, 5);
lean_closure_set(v___f_246_, 0, v_toPure_242_);
lean_closure_set(v___f_246_, 1, v_ret_244_);
lean_closure_set(v___f_246_, 2, v___x_243_);
lean_closure_set(v___f_246_, 3, v_f_239_);
lean_closure_set(v___f_246_, 4, v_toBind_241_);
v___f_247_ = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___lam__4___boxed), 5, 4);
lean_closure_set(v___f_247_, 0, v___x_243_);
lean_closure_set(v___f_247_, 1, v_toPure_242_);
lean_closure_set(v___f_247_, 2, v_f_239_);
lean_closure_set(v___f_247_, 3, v_toBind_241_);
v_sz_248_ = lean_array_size(v_arr_238_);
v___x_249_ = ((size_t)0ULL);
v___x_250_ = l___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop(lean_box(0), lean_box(0), lean_box(0), v_inst_237_, v_arr_238_, v___f_246_, v_sz_248_, v___x_249_, v___x_245_);
v___x_251_ = lean_apply_4(v_toBind_241_, lean_box(0), lean_box(0), v___x_250_, v___f_247_);
return v___x_251_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Util_foldInlsM(lean_object* v_00_u03b1_252_, lean_object* v_00_u03b2_253_, lean_object* v_m_254_, lean_object* v_inst_255_, lean_object* v_arr_256_, lean_object* v_f_257_){
_start:
{
lean_object* v___x_258_; 
v___x_258_ = lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg(v_inst_255_, v_arr_256_, v_f_257_);
return v___x_258_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__0___redArg(lean_object* v___y_259_){
_start:
{
lean_object* v_subExpr_261_; lean_object* v_expr_262_; lean_object* v___x_263_; 
v_subExpr_261_ = lean_ctor_get(v___y_259_, 3);
v_expr_262_ = lean_ctor_get(v_subExpr_261_, 0);
lean_inc_ref(v_expr_262_);
v___x_263_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_263_, 0, v_expr_262_);
return v___x_263_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__0___redArg___boxed(lean_object* v___y_264_, lean_object* v___y_265_){
_start:
{
lean_object* v_res_266_; 
v_res_266_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__0___redArg(v___y_264_);
lean_dec_ref(v___y_264_);
return v_res_266_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__0(lean_object* v___y_267_, lean_object* v___y_268_, lean_object* v___y_269_, lean_object* v___y_270_, lean_object* v___y_271_, lean_object* v___y_272_){
_start:
{
lean_object* v___x_274_; 
v___x_274_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__0___redArg(v___y_267_);
return v___x_274_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__0___boxed(lean_object* v___y_275_, lean_object* v___y_276_, lean_object* v___y_277_, lean_object* v___y_278_, lean_object* v___y_279_, lean_object* v___y_280_, lean_object* v___y_281_){
_start:
{
lean_object* v_res_282_; 
v_res_282_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__0(v___y_275_, v___y_276_, v___y_277_, v___y_278_, v___y_279_, v___y_280_);
lean_dec(v___y_280_);
lean_dec_ref(v___y_279_);
lean_dec(v___y_278_);
lean_dec_ref(v___y_277_);
lean_dec(v___y_276_);
lean_dec_ref(v___y_275_);
return v_res_282_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1_spec__1___redArg(lean_object* v___y_283_){
_start:
{
lean_object* v_subExpr_285_; lean_object* v_pos_286_; lean_object* v___x_287_; 
v_subExpr_285_ = lean_ctor_get(v___y_283_, 3);
v_pos_286_ = lean_ctor_get(v_subExpr_285_, 1);
lean_inc(v_pos_286_);
v___x_287_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_287_, 0, v_pos_286_);
return v___x_287_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1_spec__1___redArg___boxed(lean_object* v___y_288_, lean_object* v___y_289_){
_start:
{
lean_object* v_res_290_; 
v_res_290_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1_spec__1___redArg(v___y_288_);
lean_dec_ref(v___y_288_);
return v_res_290_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1_spec__1(lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_, lean_object* v___y_295_, lean_object* v___y_296_){
_start:
{
lean_object* v___x_298_; 
v___x_298_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1_spec__1___redArg(v___y_291_);
return v___x_298_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1_spec__1___boxed(lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_, lean_object* v___y_304_, lean_object* v___y_305_){
_start:
{
lean_object* v_res_306_; 
v_res_306_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1_spec__1(v___y_299_, v___y_300_, v___y_301_, v___y_302_, v___y_303_, v___y_304_);
lean_dec(v___y_304_);
lean_dec_ref(v___y_303_);
lean_dec(v___y_302_);
lean_dec_ref(v___y_301_);
lean_dec(v___y_300_);
lean_dec_ref(v___y_299_);
return v_res_306_;
}
}
static lean_object* _init_lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg___closed__0(void){
_start:
{
lean_object* v___x_307_; lean_object* v_dummy_308_; 
v___x_307_ = lean_box(0);
v_dummy_308_ = l_Lean_Expr_sort___override(v___x_307_);
return v_dummy_308_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg(lean_object* v_argIdx_309_, lean_object* v_x_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_, lean_object* v___y_315_, lean_object* v___y_316_){
_start:
{
lean_object* v___x_318_; lean_object* v_a_319_; lean_object* v___x_320_; lean_object* v_a_321_; lean_object* v_optionsPerPos_322_; lean_object* v_currNamespace_323_; lean_object* v_openDecls_324_; uint8_t v_inPattern_325_; lean_object* v_depth_326_; lean_object* v_lctxInitIndices_327_; lean_object* v_nargs_328_; lean_object* v___x_329_; lean_object* v_dummy_330_; lean_object* v___x_331_; lean_object* v___x_332_; lean_object* v___x_333_; lean_object* v_args_334_; lean_object* v___x_335_; lean_object* v_newPos_336_; lean_object* v___x_337_; lean_object* v___x_338_; lean_object* v___x_339_; lean_object* v___x_340_; 
v___x_318_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__0___redArg(v___y_311_);
v_a_319_ = lean_ctor_get(v___x_318_, 0);
lean_inc(v_a_319_);
lean_dec_ref(v___x_318_);
v___x_320_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1_spec__1___redArg(v___y_311_);
v_a_321_ = lean_ctor_get(v___x_320_, 0);
lean_inc(v_a_321_);
lean_dec_ref(v___x_320_);
v_optionsPerPos_322_ = lean_ctor_get(v___y_311_, 0);
v_currNamespace_323_ = lean_ctor_get(v___y_311_, 1);
v_openDecls_324_ = lean_ctor_get(v___y_311_, 2);
v_inPattern_325_ = lean_ctor_get_uint8(v___y_311_, sizeof(void*)*6);
v_depth_326_ = lean_ctor_get(v___y_311_, 4);
v_lctxInitIndices_327_ = lean_ctor_get(v___y_311_, 5);
v_nargs_328_ = l_Lean_Expr_getAppNumArgs(v_a_319_);
v___x_329_ = l_Lean_instInhabitedExpr;
v_dummy_330_ = lean_obj_once(&lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg___closed__0, &lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg___closed__0_once, _init_lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg___closed__0);
lean_inc(v_nargs_328_);
v___x_331_ = lean_mk_array(v_nargs_328_, v_dummy_330_);
v___x_332_ = lean_unsigned_to_nat(1u);
v___x_333_ = lean_nat_sub(v_nargs_328_, v___x_332_);
lean_dec(v_nargs_328_);
v_args_334_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_319_, v___x_331_, v___x_333_);
v___x_335_ = lean_array_get_size(v_args_334_);
v_newPos_336_ = l_Lean_SubExpr_Pos_pushNaryArg(v___x_335_, v_argIdx_309_, v_a_321_);
lean_dec(v_a_321_);
v___x_337_ = lean_array_get(v___x_329_, v_args_334_, v_argIdx_309_);
lean_dec_ref(v_args_334_);
v___x_338_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_338_, 0, v___x_337_);
lean_ctor_set(v___x_338_, 1, v_newPos_336_);
lean_inc(v_lctxInitIndices_327_);
lean_inc(v_depth_326_);
lean_inc(v_openDecls_324_);
lean_inc(v_currNamespace_323_);
lean_inc(v_optionsPerPos_322_);
v___x_339_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_339_, 0, v_optionsPerPos_322_);
lean_ctor_set(v___x_339_, 1, v_currNamespace_323_);
lean_ctor_set(v___x_339_, 2, v_openDecls_324_);
lean_ctor_set(v___x_339_, 3, v___x_338_);
lean_ctor_set(v___x_339_, 4, v_depth_326_);
lean_ctor_set(v___x_339_, 5, v_lctxInitIndices_327_);
lean_ctor_set_uint8(v___x_339_, sizeof(void*)*6, v_inPattern_325_);
lean_inc(v___y_316_);
lean_inc_ref(v___y_315_);
lean_inc(v___y_314_);
lean_inc_ref(v___y_313_);
lean_inc(v___y_312_);
v___x_340_ = lean_apply_7(v_x_310_, v___x_339_, v___y_312_, v___y_313_, v___y_314_, v___y_315_, v___y_316_, lean_box(0));
return v___x_340_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg___boxed(lean_object* v_argIdx_341_, lean_object* v_x_342_, lean_object* v___y_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_, lean_object* v___y_348_, lean_object* v___y_349_){
_start:
{
lean_object* v_res_350_; 
v_res_350_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg(v_argIdx_341_, v_x_342_, v___y_343_, v___y_344_, v___y_345_, v___y_346_, v___y_347_, v___y_348_);
lean_dec(v___y_348_);
lean_dec_ref(v___y_347_);
lean_dec(v___y_346_);
lean_dec_ref(v___y_345_);
lean_dec(v___y_344_);
lean_dec_ref(v___y_343_);
lean_dec(v_argIdx_341_);
return v_res_350_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1(lean_object* v_00_u03b1_351_, lean_object* v_argIdx_352_, lean_object* v_x_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_, lean_object* v___y_357_, lean_object* v___y_358_, lean_object* v___y_359_){
_start:
{
lean_object* v___x_361_; 
v___x_361_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg(v_argIdx_352_, v_x_353_, v___y_354_, v___y_355_, v___y_356_, v___y_357_, v___y_358_, v___y_359_);
return v___x_361_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___boxed(lean_object* v_00_u03b1_362_, lean_object* v_argIdx_363_, lean_object* v_x_364_, lean_object* v___y_365_, lean_object* v___y_366_, lean_object* v___y_367_, lean_object* v___y_368_, lean_object* v___y_369_, lean_object* v___y_370_, lean_object* v___y_371_){
_start:
{
lean_object* v_res_372_; 
v_res_372_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1(v_00_u03b1_362_, v_argIdx_363_, v_x_364_, v___y_365_, v___y_366_, v___y_367_, v___y_368_, v___y_369_, v___y_370_);
lean_dec(v___y_370_);
lean_dec_ref(v___y_369_);
lean_dec(v___y_368_);
lean_dec_ref(v___y_367_);
lean_dec(v___y_366_);
lean_dec_ref(v___y_365_);
lean_dec(v_argIdx_363_);
return v_res_372_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg(lean_object* v_elem_382_, lean_object* v_acc_383_, lean_object* v_a_384_, lean_object* v_a_385_, lean_object* v_a_386_, lean_object* v_a_387_, lean_object* v_a_388_, lean_object* v_a_389_){
_start:
{
lean_object* v___x_391_; 
v___x_391_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__0___redArg(v_a_384_);
if (lean_obj_tag(v___x_391_) == 0)
{
lean_object* v_a_392_; lean_object* v___x_393_; 
v_a_392_ = lean_ctor_get(v___x_391_, 0);
lean_inc(v_a_392_);
lean_dec_ref_known(v___x_391_, 1);
v___x_393_ = l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(v_a_392_, v_a_387_);
if (lean_obj_tag(v___x_393_) == 0)
{
lean_object* v_a_394_; lean_object* v___x_396_; uint8_t v_isShared_397_; uint8_t v_isSharedCheck_430_; 
v_a_394_ = lean_ctor_get(v___x_393_, 0);
v_isSharedCheck_430_ = !lean_is_exclusive(v___x_393_);
if (v_isSharedCheck_430_ == 0)
{
v___x_396_ = v___x_393_;
v_isShared_397_ = v_isSharedCheck_430_;
goto v_resetjp_395_;
}
else
{
lean_inc(v_a_394_);
lean_dec(v___x_393_);
v___x_396_ = lean_box(0);
v_isShared_397_ = v_isSharedCheck_430_;
goto v_resetjp_395_;
}
v_resetjp_395_:
{
lean_object* v___x_398_; uint8_t v___x_399_; 
v___x_398_ = l_Lean_Expr_cleanupAnnotations(v_a_394_);
v___x_399_ = l_Lean_Expr_isApp(v___x_398_);
if (v___x_399_ == 0)
{
lean_object* v___x_400_; 
lean_dec_ref(v___x_398_);
lean_del_object(v___x_396_);
lean_dec_ref(v_acc_383_);
lean_dec_ref(v_elem_382_);
v___x_400_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_400_;
}
else
{
lean_object* v___x_401_; lean_object* v___x_402_; uint8_t v___x_403_; 
v___x_401_ = l_Lean_Expr_appFnCleanup___redArg(v___x_398_);
v___x_402_ = ((lean_object*)(lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__2));
v___x_403_ = l_Lean_Expr_isConstOf(v___x_401_, v___x_402_);
if (v___x_403_ == 0)
{
uint8_t v___x_404_; 
lean_del_object(v___x_396_);
v___x_404_ = l_Lean_Expr_isApp(v___x_401_);
if (v___x_404_ == 0)
{
lean_object* v___x_405_; 
lean_dec_ref(v___x_401_);
lean_dec_ref(v_acc_383_);
lean_dec_ref(v_elem_382_);
v___x_405_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_405_;
}
else
{
lean_object* v___x_406_; uint8_t v___x_407_; 
v___x_406_ = l_Lean_Expr_appFnCleanup___redArg(v___x_401_);
v___x_407_ = l_Lean_Expr_isApp(v___x_406_);
if (v___x_407_ == 0)
{
lean_object* v___x_408_; 
lean_dec_ref(v___x_406_);
lean_dec_ref(v_acc_383_);
lean_dec_ref(v_elem_382_);
v___x_408_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_408_;
}
else
{
lean_object* v___x_409_; lean_object* v___x_410_; uint8_t v___x_411_; 
v___x_409_ = l_Lean_Expr_appFnCleanup___redArg(v___x_406_);
v___x_410_ = ((lean_object*)(lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___closed__4));
v___x_411_ = l_Lean_Expr_isConstOf(v___x_409_, v___x_410_);
lean_dec_ref(v___x_409_);
if (v___x_411_ == 0)
{
lean_object* v___x_412_; 
lean_dec_ref(v_acc_383_);
lean_dec_ref(v_elem_382_);
v___x_412_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_412_;
}
else
{
lean_object* v___x_413_; lean_object* v___x_414_; 
v___x_413_ = lean_unsigned_to_nat(1u);
lean_inc_ref(v_elem_382_);
v___x_414_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg(v___x_413_, v_elem_382_, v_a_384_, v_a_385_, v_a_386_, v_a_387_, v_a_388_, v_a_389_);
if (lean_obj_tag(v___x_414_) == 0)
{
lean_object* v_a_415_; lean_object* v___x_416_; lean_object* v___x_417_; lean_object* v___x_418_; 
v_a_415_ = lean_ctor_get(v___x_414_, 0);
lean_inc(v_a_415_);
lean_dec_ref_known(v___x_414_, 1);
v___x_416_ = lean_unsigned_to_nat(2u);
v___x_417_ = lean_array_push(v_acc_383_, v_a_415_);
v___x_418_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__3___redArg(v_elem_382_, v___x_417_, v___x_416_, v_a_384_, v_a_385_, v_a_386_, v_a_387_, v_a_388_, v_a_389_);
return v___x_418_;
}
else
{
lean_object* v_a_419_; lean_object* v___x_421_; uint8_t v_isShared_422_; uint8_t v_isSharedCheck_426_; 
lean_dec_ref(v_acc_383_);
lean_dec_ref(v_elem_382_);
v_a_419_ = lean_ctor_get(v___x_414_, 0);
v_isSharedCheck_426_ = !lean_is_exclusive(v___x_414_);
if (v_isSharedCheck_426_ == 0)
{
v___x_421_ = v___x_414_;
v_isShared_422_ = v_isSharedCheck_426_;
goto v_resetjp_420_;
}
else
{
lean_inc(v_a_419_);
lean_dec(v___x_414_);
v___x_421_ = lean_box(0);
v_isShared_422_ = v_isSharedCheck_426_;
goto v_resetjp_420_;
}
v_resetjp_420_:
{
lean_object* v___x_424_; 
if (v_isShared_422_ == 0)
{
v___x_424_ = v___x_421_;
goto v_reusejp_423_;
}
else
{
lean_object* v_reuseFailAlloc_425_; 
v_reuseFailAlloc_425_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_425_, 0, v_a_419_);
v___x_424_ = v_reuseFailAlloc_425_;
goto v_reusejp_423_;
}
v_reusejp_423_:
{
return v___x_424_;
}
}
}
}
}
}
}
else
{
lean_object* v___x_428_; 
lean_dec_ref(v___x_401_);
lean_dec_ref(v_elem_382_);
if (v_isShared_397_ == 0)
{
lean_ctor_set(v___x_396_, 0, v_acc_383_);
v___x_428_ = v___x_396_;
goto v_reusejp_427_;
}
else
{
lean_object* v_reuseFailAlloc_429_; 
v_reuseFailAlloc_429_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_429_, 0, v_acc_383_);
v___x_428_ = v_reuseFailAlloc_429_;
goto v_reusejp_427_;
}
v_reusejp_427_:
{
return v___x_428_;
}
}
}
}
}
else
{
lean_object* v_a_431_; lean_object* v___x_433_; uint8_t v_isShared_434_; uint8_t v_isSharedCheck_438_; 
lean_dec_ref(v_acc_383_);
lean_dec_ref(v_elem_382_);
v_a_431_ = lean_ctor_get(v___x_393_, 0);
v_isSharedCheck_438_ = !lean_is_exclusive(v___x_393_);
if (v_isSharedCheck_438_ == 0)
{
v___x_433_ = v___x_393_;
v_isShared_434_ = v_isSharedCheck_438_;
goto v_resetjp_432_;
}
else
{
lean_inc(v_a_431_);
lean_dec(v___x_393_);
v___x_433_ = lean_box(0);
v_isShared_434_ = v_isSharedCheck_438_;
goto v_resetjp_432_;
}
v_resetjp_432_:
{
lean_object* v___x_436_; 
if (v_isShared_434_ == 0)
{
v___x_436_ = v___x_433_;
goto v_reusejp_435_;
}
else
{
lean_object* v_reuseFailAlloc_437_; 
v_reuseFailAlloc_437_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_437_, 0, v_a_431_);
v___x_436_ = v_reuseFailAlloc_437_;
goto v_reusejp_435_;
}
v_reusejp_435_:
{
return v___x_436_;
}
}
}
}
else
{
lean_object* v_a_439_; lean_object* v___x_441_; uint8_t v_isShared_442_; uint8_t v_isSharedCheck_446_; 
lean_dec_ref(v_acc_383_);
lean_dec_ref(v_elem_382_);
v_a_439_ = lean_ctor_get(v___x_391_, 0);
v_isSharedCheck_446_ = !lean_is_exclusive(v___x_391_);
if (v_isSharedCheck_446_ == 0)
{
v___x_441_ = v___x_391_;
v_isShared_442_ = v_isSharedCheck_446_;
goto v_resetjp_440_;
}
else
{
lean_inc(v_a_439_);
lean_dec(v___x_391_);
v___x_441_ = lean_box(0);
v_isShared_442_ = v_isSharedCheck_446_;
goto v_resetjp_440_;
}
v_resetjp_440_:
{
lean_object* v___x_444_; 
if (v_isShared_442_ == 0)
{
v___x_444_ = v___x_441_;
goto v_reusejp_443_;
}
else
{
lean_object* v_reuseFailAlloc_445_; 
v_reuseFailAlloc_445_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_445_, 0, v_a_439_);
v___x_444_ = v_reuseFailAlloc_445_;
goto v_reusejp_443_;
}
v_reusejp_443_:
{
return v___x_444_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__3___redArg(lean_object* v_elem_447_, lean_object* v___x_448_, lean_object* v_argIdx_449_, lean_object* v___y_450_, lean_object* v___y_451_, lean_object* v___y_452_, lean_object* v___y_453_, lean_object* v___y_454_, lean_object* v___y_455_){
_start:
{
lean_object* v___x_457_; lean_object* v_a_458_; lean_object* v___x_459_; lean_object* v_a_460_; lean_object* v_optionsPerPos_461_; lean_object* v_currNamespace_462_; lean_object* v_openDecls_463_; uint8_t v_inPattern_464_; lean_object* v_depth_465_; lean_object* v_lctxInitIndices_466_; lean_object* v_nargs_467_; lean_object* v___x_468_; lean_object* v_dummy_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v_args_473_; lean_object* v___x_474_; lean_object* v_newPos_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; 
v___x_457_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__0___redArg(v___y_450_);
v_a_458_ = lean_ctor_get(v___x_457_, 0);
lean_inc(v_a_458_);
lean_dec_ref(v___x_457_);
v___x_459_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1_spec__1___redArg(v___y_450_);
v_a_460_ = lean_ctor_get(v___x_459_, 0);
lean_inc(v_a_460_);
lean_dec_ref(v___x_459_);
v_optionsPerPos_461_ = lean_ctor_get(v___y_450_, 0);
v_currNamespace_462_ = lean_ctor_get(v___y_450_, 1);
v_openDecls_463_ = lean_ctor_get(v___y_450_, 2);
v_inPattern_464_ = lean_ctor_get_uint8(v___y_450_, sizeof(void*)*6);
v_depth_465_ = lean_ctor_get(v___y_450_, 4);
v_lctxInitIndices_466_ = lean_ctor_get(v___y_450_, 5);
v_nargs_467_ = l_Lean_Expr_getAppNumArgs(v_a_458_);
v___x_468_ = l_Lean_instInhabitedExpr;
v_dummy_469_ = lean_obj_once(&lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg___closed__0, &lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg___closed__0_once, _init_lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg___closed__0);
lean_inc(v_nargs_467_);
v___x_470_ = lean_mk_array(v_nargs_467_, v_dummy_469_);
v___x_471_ = lean_unsigned_to_nat(1u);
v___x_472_ = lean_nat_sub(v_nargs_467_, v___x_471_);
lean_dec(v_nargs_467_);
v_args_473_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_458_, v___x_470_, v___x_472_);
v___x_474_ = lean_array_get_size(v_args_473_);
v_newPos_475_ = l_Lean_SubExpr_Pos_pushNaryArg(v___x_474_, v_argIdx_449_, v_a_460_);
lean_dec(v_a_460_);
v___x_476_ = lean_array_get(v___x_468_, v_args_473_, v_argIdx_449_);
lean_dec_ref(v_args_473_);
v___x_477_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_477_, 0, v___x_476_);
lean_ctor_set(v___x_477_, 1, v_newPos_475_);
lean_inc(v_lctxInitIndices_466_);
lean_inc(v_depth_465_);
lean_inc(v_openDecls_463_);
lean_inc(v_currNamespace_462_);
lean_inc(v_optionsPerPos_461_);
v___x_478_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_478_, 0, v_optionsPerPos_461_);
lean_ctor_set(v___x_478_, 1, v_currNamespace_462_);
lean_ctor_set(v___x_478_, 2, v_openDecls_463_);
lean_ctor_set(v___x_478_, 3, v___x_477_);
lean_ctor_set(v___x_478_, 4, v_depth_465_);
lean_ctor_set(v___x_478_, 5, v_lctxInitIndices_466_);
lean_ctor_set_uint8(v___x_478_, sizeof(void*)*6, v_inPattern_464_);
v___x_479_ = lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg(v_elem_447_, v___x_448_, v___x_478_, v___y_451_, v___y_452_, v___y_453_, v___y_454_, v___y_455_);
lean_dec_ref_known(v___x_478_, 6);
return v___x_479_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__3___redArg___boxed(lean_object* v_elem_480_, lean_object* v___x_481_, lean_object* v_argIdx_482_, lean_object* v___y_483_, lean_object* v___y_484_, lean_object* v___y_485_, lean_object* v___y_486_, lean_object* v___y_487_, lean_object* v___y_488_, lean_object* v___y_489_){
_start:
{
lean_object* v_res_490_; 
v_res_490_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__3___redArg(v_elem_480_, v___x_481_, v_argIdx_482_, v___y_483_, v___y_484_, v___y_485_, v___y_486_, v___y_487_, v___y_488_);
lean_dec(v___y_488_);
lean_dec_ref(v___y_487_);
lean_dec(v___y_486_);
lean_dec_ref(v___y_485_);
lean_dec(v___y_484_);
lean_dec_ref(v___y_483_);
lean_dec(v_argIdx_482_);
return v_res_490_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg___boxed(lean_object* v_elem_491_, lean_object* v_acc_492_, lean_object* v_a_493_, lean_object* v_a_494_, lean_object* v_a_495_, lean_object* v_a_496_, lean_object* v_a_497_, lean_object* v_a_498_, lean_object* v_a_499_){
_start:
{
lean_object* v_res_500_; 
v_res_500_ = lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg(v_elem_491_, v_acc_492_, v_a_493_, v_a_494_, v_a_495_, v_a_496_, v_a_497_, v_a_498_);
lean_dec(v_a_498_);
lean_dec_ref(v_a_497_);
lean_dec(v_a_496_);
lean_dec_ref(v_a_495_);
lean_dec(v_a_494_);
lean_dec_ref(v_a_493_);
return v_res_500_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go(lean_object* v_00_u03b1_501_, lean_object* v_elem_502_, lean_object* v_acc_503_, lean_object* v_a_504_, lean_object* v_a_505_, lean_object* v_a_506_, lean_object* v_a_507_, lean_object* v_a_508_, lean_object* v_a_509_){
_start:
{
lean_object* v___x_511_; 
v___x_511_ = lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg(v_elem_502_, v_acc_503_, v_a_504_, v_a_505_, v_a_506_, v_a_507_, v_a_508_, v_a_509_);
return v___x_511_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___boxed(lean_object* v_00_u03b1_512_, lean_object* v_elem_513_, lean_object* v_acc_514_, lean_object* v_a_515_, lean_object* v_a_516_, lean_object* v_a_517_, lean_object* v_a_518_, lean_object* v_a_519_, lean_object* v_a_520_, lean_object* v_a_521_){
_start:
{
lean_object* v_res_522_; 
v_res_522_ = lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go(v_00_u03b1_512_, v_elem_513_, v_acc_514_, v_a_515_, v_a_516_, v_a_517_, v_a_518_, v_a_519_, v_a_520_);
lean_dec(v_a_520_);
lean_dec_ref(v_a_519_);
lean_dec(v_a_518_);
lean_dec_ref(v_a_517_);
lean_dec(v_a_516_);
lean_dec_ref(v_a_515_);
return v_res_522_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__3(lean_object* v_00_u03b1_523_, lean_object* v_elem_524_, lean_object* v___x_525_, lean_object* v_argIdx_526_, lean_object* v___y_527_, lean_object* v___y_528_, lean_object* v___y_529_, lean_object* v___y_530_, lean_object* v___y_531_, lean_object* v___y_532_){
_start:
{
lean_object* v___x_534_; 
v___x_534_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__3___redArg(v_elem_524_, v___x_525_, v_argIdx_526_, v___y_527_, v___y_528_, v___y_529_, v___y_530_, v___y_531_, v___y_532_);
return v___x_534_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__3___boxed(lean_object* v_00_u03b1_535_, lean_object* v_elem_536_, lean_object* v___x_537_, lean_object* v_argIdx_538_, lean_object* v___y_539_, lean_object* v___y_540_, lean_object* v___y_541_, lean_object* v___y_542_, lean_object* v___y_543_, lean_object* v___y_544_, lean_object* v___y_545_){
_start:
{
lean_object* v_res_546_; 
v_res_546_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__3(v_00_u03b1_535_, v_elem_536_, v___x_537_, v_argIdx_538_, v___y_539_, v___y_540_, v___y_541_, v___y_542_, v___y_543_, v___y_544_);
lean_dec(v___y_544_);
lean_dec_ref(v___y_543_);
lean_dec(v___y_542_);
lean_dec_ref(v___y_541_);
lean_dec(v___y_540_);
lean_dec_ref(v___y_539_);
lean_dec(v_argIdx_538_);
return v_res_546_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabListLiteral___redArg(lean_object* v_elem_547_, lean_object* v_a_548_, lean_object* v_a_549_, lean_object* v_a_550_, lean_object* v_a_551_, lean_object* v_a_552_, lean_object* v_a_553_){
_start:
{
lean_object* v___x_555_; lean_object* v___x_556_; 
v___x_555_ = ((lean_object*)(lp_proofwidgets_ProofWidgets_Util_foldInlsM___redArg___closed__0));
v___x_556_ = lp_proofwidgets___private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go___redArg(v_elem_547_, v___x_555_, v_a_548_, v_a_549_, v_a_550_, v_a_551_, v_a_552_, v_a_553_);
return v___x_556_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabListLiteral___redArg___boxed(lean_object* v_elem_557_, lean_object* v_a_558_, lean_object* v_a_559_, lean_object* v_a_560_, lean_object* v_a_561_, lean_object* v_a_562_, lean_object* v_a_563_, lean_object* v_a_564_){
_start:
{
lean_object* v_res_565_; 
v_res_565_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabListLiteral___redArg(v_elem_557_, v_a_558_, v_a_559_, v_a_560_, v_a_561_, v_a_562_, v_a_563_);
lean_dec(v_a_563_);
lean_dec_ref(v_a_562_);
lean_dec(v_a_561_);
lean_dec_ref(v_a_560_);
lean_dec(v_a_559_);
lean_dec_ref(v_a_558_);
return v_res_565_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabListLiteral(lean_object* v_00_u03b1_566_, lean_object* v_elem_567_, lean_object* v_a_568_, lean_object* v_a_569_, lean_object* v_a_570_, lean_object* v_a_571_, lean_object* v_a_572_, lean_object* v_a_573_){
_start:
{
lean_object* v___x_575_; 
v___x_575_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabListLiteral___redArg(v_elem_567_, v_a_568_, v_a_569_, v_a_570_, v_a_571_, v_a_572_, v_a_573_);
return v___x_575_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabListLiteral___boxed(lean_object* v_00_u03b1_576_, lean_object* v_elem_577_, lean_object* v_a_578_, lean_object* v_a_579_, lean_object* v_a_580_, lean_object* v_a_581_, lean_object* v_a_582_, lean_object* v_a_583_, lean_object* v_a_584_){
_start:
{
lean_object* v_res_585_; 
v_res_585_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabListLiteral(v_00_u03b1_576_, v_elem_577_, v_a_578_, v_a_579_, v_a_580_, v_a_581_, v_a_582_, v_a_583_);
lean_dec(v_a_583_);
lean_dec_ref(v_a_582_);
lean_dec(v_a_581_);
lean_dec_ref(v_a_580_);
lean_dec(v_a_579_);
lean_dec_ref(v_a_578_);
return v_res_585_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00Lean_PrettyPrinter_Delaborator_delabArrayLiteral_spec__0___redArg(lean_object* v_elem_586_, lean_object* v_argIdx_587_, lean_object* v___y_588_, lean_object* v___y_589_, lean_object* v___y_590_, lean_object* v___y_591_, lean_object* v___y_592_, lean_object* v___y_593_){
_start:
{
lean_object* v___x_595_; lean_object* v_a_596_; lean_object* v___x_597_; lean_object* v_a_598_; lean_object* v_optionsPerPos_599_; lean_object* v_currNamespace_600_; lean_object* v_openDecls_601_; uint8_t v_inPattern_602_; lean_object* v_depth_603_; lean_object* v_lctxInitIndices_604_; lean_object* v_nargs_605_; lean_object* v___x_606_; lean_object* v_dummy_607_; lean_object* v___x_608_; lean_object* v___x_609_; lean_object* v___x_610_; lean_object* v_args_611_; lean_object* v___x_612_; lean_object* v_newPos_613_; lean_object* v___x_614_; lean_object* v___x_615_; lean_object* v___x_616_; lean_object* v___x_617_; 
v___x_595_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__0___redArg(v___y_588_);
v_a_596_ = lean_ctor_get(v___x_595_, 0);
lean_inc(v_a_596_);
lean_dec_ref(v___x_595_);
v___x_597_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1_spec__1___redArg(v___y_588_);
v_a_598_ = lean_ctor_get(v___x_597_, 0);
lean_inc(v_a_598_);
lean_dec_ref(v___x_597_);
v_optionsPerPos_599_ = lean_ctor_get(v___y_588_, 0);
v_currNamespace_600_ = lean_ctor_get(v___y_588_, 1);
v_openDecls_601_ = lean_ctor_get(v___y_588_, 2);
v_inPattern_602_ = lean_ctor_get_uint8(v___y_588_, sizeof(void*)*6);
v_depth_603_ = lean_ctor_get(v___y_588_, 4);
v_lctxInitIndices_604_ = lean_ctor_get(v___y_588_, 5);
v_nargs_605_ = l_Lean_Expr_getAppNumArgs(v_a_596_);
v___x_606_ = l_Lean_instInhabitedExpr;
v_dummy_607_ = lean_obj_once(&lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg___closed__0, &lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg___closed__0_once, _init_lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___redArg___closed__0);
lean_inc(v_nargs_605_);
v___x_608_ = lean_mk_array(v_nargs_605_, v_dummy_607_);
v___x_609_ = lean_unsigned_to_nat(1u);
v___x_610_ = lean_nat_sub(v_nargs_605_, v___x_609_);
lean_dec(v_nargs_605_);
v_args_611_ = l___private_Lean_Expr_0__Lean_Expr_getAppArgsAux(v_a_596_, v___x_608_, v___x_610_);
v___x_612_ = lean_array_get_size(v_args_611_);
v_newPos_613_ = l_Lean_SubExpr_Pos_pushNaryArg(v___x_612_, v_argIdx_587_, v_a_598_);
lean_dec(v_a_598_);
v___x_614_ = lean_array_get(v___x_606_, v_args_611_, v_argIdx_587_);
lean_dec_ref(v_args_611_);
v___x_615_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_615_, 0, v___x_614_);
lean_ctor_set(v___x_615_, 1, v_newPos_613_);
lean_inc(v_lctxInitIndices_604_);
lean_inc(v_depth_603_);
lean_inc(v_openDecls_601_);
lean_inc(v_currNamespace_600_);
lean_inc(v_optionsPerPos_599_);
v___x_616_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_616_, 0, v_optionsPerPos_599_);
lean_ctor_set(v___x_616_, 1, v_currNamespace_600_);
lean_ctor_set(v___x_616_, 2, v_openDecls_601_);
lean_ctor_set(v___x_616_, 3, v___x_615_);
lean_ctor_set(v___x_616_, 4, v_depth_603_);
lean_ctor_set(v___x_616_, 5, v_lctxInitIndices_604_);
lean_ctor_set_uint8(v___x_616_, sizeof(void*)*6, v_inPattern_602_);
v___x_617_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabListLiteral___redArg(v_elem_586_, v___x_616_, v___y_589_, v___y_590_, v___y_591_, v___y_592_, v___y_593_);
lean_dec_ref_known(v___x_616_, 6);
return v___x_617_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00Lean_PrettyPrinter_Delaborator_delabArrayLiteral_spec__0___redArg___boxed(lean_object* v_elem_618_, lean_object* v_argIdx_619_, lean_object* v___y_620_, lean_object* v___y_621_, lean_object* v___y_622_, lean_object* v___y_623_, lean_object* v___y_624_, lean_object* v___y_625_, lean_object* v___y_626_){
_start:
{
lean_object* v_res_627_; 
v_res_627_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00Lean_PrettyPrinter_Delaborator_delabArrayLiteral_spec__0___redArg(v_elem_618_, v_argIdx_619_, v___y_620_, v___y_621_, v___y_622_, v___y_623_, v___y_624_, v___y_625_);
lean_dec(v___y_625_);
lean_dec_ref(v___y_624_);
lean_dec(v___y_623_);
lean_dec_ref(v___y_622_);
lean_dec(v___y_621_);
lean_dec_ref(v___y_620_);
lean_dec(v_argIdx_619_);
return v_res_627_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00Lean_PrettyPrinter_Delaborator_delabArrayLiteral_spec__0(lean_object* v_00_u03b1_628_, lean_object* v_elem_629_, lean_object* v_argIdx_630_, lean_object* v___y_631_, lean_object* v___y_632_, lean_object* v___y_633_, lean_object* v___y_634_, lean_object* v___y_635_, lean_object* v___y_636_){
_start:
{
lean_object* v___x_638_; 
v___x_638_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00Lean_PrettyPrinter_Delaborator_delabArrayLiteral_spec__0___redArg(v_elem_629_, v_argIdx_630_, v___y_631_, v___y_632_, v___y_633_, v___y_634_, v___y_635_, v___y_636_);
return v___x_638_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00Lean_PrettyPrinter_Delaborator_delabArrayLiteral_spec__0___boxed(lean_object* v_00_u03b1_639_, lean_object* v_elem_640_, lean_object* v_argIdx_641_, lean_object* v___y_642_, lean_object* v___y_643_, lean_object* v___y_644_, lean_object* v___y_645_, lean_object* v___y_646_, lean_object* v___y_647_, lean_object* v___y_648_){
_start:
{
lean_object* v_res_649_; 
v_res_649_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00Lean_PrettyPrinter_Delaborator_delabArrayLiteral_spec__0(v_00_u03b1_639_, v_elem_640_, v_argIdx_641_, v___y_642_, v___y_643_, v___y_644_, v___y_645_, v___y_646_, v___y_647_);
lean_dec(v___y_647_);
lean_dec_ref(v___y_646_);
lean_dec(v___y_645_);
lean_dec_ref(v___y_644_);
lean_dec(v___y_643_);
lean_dec_ref(v___y_642_);
lean_dec(v_argIdx_641_);
return v_res_649_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg(lean_object* v_elem_654_, lean_object* v_a_655_, lean_object* v_a_656_, lean_object* v_a_657_, lean_object* v_a_658_, lean_object* v_a_659_, lean_object* v_a_660_){
_start:
{
lean_object* v___x_662_; lean_object* v_a_663_; lean_object* v___x_664_; 
v___x_662_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__0___redArg(v_a_655_);
v_a_663_ = lean_ctor_get(v___x_662_, 0);
lean_inc(v_a_663_);
lean_dec_ref(v___x_662_);
v___x_664_ = l_Lean_Meta_instantiateMVarsIfMVarApp___redArg(v_a_663_, v_a_658_);
if (lean_obj_tag(v___x_664_) == 0)
{
lean_object* v_a_665_; lean_object* v___x_666_; uint8_t v___x_667_; 
v_a_665_ = lean_ctor_get(v___x_664_, 0);
lean_inc(v_a_665_);
lean_dec_ref_known(v___x_664_, 1);
v___x_666_ = l_Lean_Expr_cleanupAnnotations(v_a_665_);
v___x_667_ = l_Lean_Expr_isApp(v___x_666_);
if (v___x_667_ == 0)
{
lean_object* v___x_668_; 
lean_dec_ref(v___x_666_);
lean_dec_ref(v_elem_654_);
v___x_668_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_668_;
}
else
{
lean_object* v___x_669_; uint8_t v___x_670_; 
v___x_669_ = l_Lean_Expr_appFnCleanup___redArg(v___x_666_);
v___x_670_ = l_Lean_Expr_isApp(v___x_669_);
if (v___x_670_ == 0)
{
lean_object* v___x_671_; 
lean_dec_ref(v___x_669_);
lean_dec_ref(v_elem_654_);
v___x_671_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_671_;
}
else
{
lean_object* v___x_672_; lean_object* v___x_673_; uint8_t v___x_674_; 
v___x_672_ = l_Lean_Expr_appFnCleanup___redArg(v___x_669_);
v___x_673_ = ((lean_object*)(lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg___closed__1));
v___x_674_ = l_Lean_Expr_isConstOf(v___x_672_, v___x_673_);
lean_dec_ref(v___x_672_);
if (v___x_674_ == 0)
{
lean_object* v___x_675_; 
lean_dec_ref(v_elem_654_);
v___x_675_ = l_Lean_PrettyPrinter_Delaborator_failure___redArg();
return v___x_675_;
}
else
{
lean_object* v___x_676_; lean_object* v___x_677_; 
v___x_676_ = lean_unsigned_to_nat(1u);
v___x_677_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1___at___00Lean_PrettyPrinter_Delaborator_delabArrayLiteral_spec__0___redArg(v_elem_654_, v___x_676_, v_a_655_, v_a_656_, v_a_657_, v_a_658_, v_a_659_, v_a_660_);
return v___x_677_;
}
}
}
}
else
{
lean_object* v_a_678_; lean_object* v___x_680_; uint8_t v_isShared_681_; uint8_t v_isSharedCheck_685_; 
lean_dec_ref(v_elem_654_);
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
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg___boxed(lean_object* v_elem_686_, lean_object* v_a_687_, lean_object* v_a_688_, lean_object* v_a_689_, lean_object* v_a_690_, lean_object* v_a_691_, lean_object* v_a_692_, lean_object* v_a_693_){
_start:
{
lean_object* v_res_694_; 
v_res_694_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg(v_elem_686_, v_a_687_, v_a_688_, v_a_689_, v_a_690_, v_a_691_, v_a_692_);
lean_dec(v_a_692_);
lean_dec_ref(v_a_691_);
lean_dec(v_a_690_);
lean_dec_ref(v_a_689_);
lean_dec(v_a_688_);
lean_dec_ref(v_a_687_);
return v_res_694_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral(lean_object* v_00_u03b1_695_, lean_object* v_elem_696_, lean_object* v_a_697_, lean_object* v_a_698_, lean_object* v_a_699_, lean_object* v_a_700_, lean_object* v_a_701_, lean_object* v_a_702_){
_start:
{
lean_object* v___x_704_; 
v___x_704_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___redArg(v_elem_696_, v_a_697_, v_a_698_, v_a_699_, v_a_700_, v_a_701_, v_a_702_);
return v___x_704_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral___boxed(lean_object* v_00_u03b1_705_, lean_object* v_elem_706_, lean_object* v_a_707_, lean_object* v_a_708_, lean_object* v_a_709_, lean_object* v_a_710_, lean_object* v_a_711_, lean_object* v_a_712_, lean_object* v_a_713_){
_start:
{
lean_object* v_res_714_; 
v_res_714_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_delabArrayLiteral(v_00_u03b1_705_, v_elem_706_, v_a_707_, v_a_708_, v_a_709_, v_a_710_, v_a_711_, v_a_712_);
lean_dec(v_a_712_);
lean_dec_ref(v_a_711_);
lean_dec(v_a_710_);
lean_dec_ref(v_a_709_);
lean_dec(v_a_708_);
lean_dec_ref(v_a_707_);
return v_res_714_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo___redArg(lean_object* v_stx_715_, lean_object* v_a_716_, lean_object* v_a_717_, lean_object* v_a_718_){
_start:
{
lean_object* v___x_720_; 
v___x_720_ = l_Lean_PrettyPrinter_Delaborator_annotateCurPos___redArg(v_stx_715_, v_a_716_);
if (lean_obj_tag(v___x_720_) == 0)
{
lean_object* v_a_721_; lean_object* v___x_722_; lean_object* v_a_723_; lean_object* v___x_724_; lean_object* v_a_725_; uint8_t v___x_726_; lean_object* v___x_727_; 
v_a_721_ = lean_ctor_get(v___x_720_, 0);
lean_inc_n(v_a_721_, 2);
lean_dec_ref_known(v___x_720_, 1);
v___x_722_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getPos___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withNaryArg___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__1_spec__1___redArg(v_a_716_);
v_a_723_ = lean_ctor_get(v___x_722_, 0);
lean_inc(v_a_723_);
lean_dec_ref(v___x_722_);
v___x_724_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00__private_ProofWidgets_Util_0__Lean_PrettyPrinter_Delaborator_delabListLiteral_go_spec__0___redArg(v_a_716_);
v_a_725_ = lean_ctor_get(v___x_724_, 0);
lean_inc(v_a_725_);
lean_dec_ref(v___x_724_);
v___x_726_ = 0;
v___x_727_ = l_Lean_PrettyPrinter_Delaborator_addTermInfo___redArg(v_a_723_, v_a_721_, v_a_725_, v___x_726_, v_a_717_, v_a_718_);
if (lean_obj_tag(v___x_727_) == 0)
{
lean_object* v___x_729_; uint8_t v_isShared_730_; uint8_t v_isSharedCheck_734_; 
v_isSharedCheck_734_ = !lean_is_exclusive(v___x_727_);
if (v_isSharedCheck_734_ == 0)
{
lean_object* v_unused_735_; 
v_unused_735_ = lean_ctor_get(v___x_727_, 0);
lean_dec(v_unused_735_);
v___x_729_ = v___x_727_;
v_isShared_730_ = v_isSharedCheck_734_;
goto v_resetjp_728_;
}
else
{
lean_dec(v___x_727_);
v___x_729_ = lean_box(0);
v_isShared_730_ = v_isSharedCheck_734_;
goto v_resetjp_728_;
}
v_resetjp_728_:
{
lean_object* v___x_732_; 
if (v_isShared_730_ == 0)
{
lean_ctor_set(v___x_729_, 0, v_a_721_);
v___x_732_ = v___x_729_;
goto v_reusejp_731_;
}
else
{
lean_object* v_reuseFailAlloc_733_; 
v_reuseFailAlloc_733_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_733_, 0, v_a_721_);
v___x_732_ = v_reuseFailAlloc_733_;
goto v_reusejp_731_;
}
v_reusejp_731_:
{
return v___x_732_;
}
}
}
else
{
lean_object* v_a_736_; lean_object* v___x_738_; uint8_t v_isShared_739_; uint8_t v_isSharedCheck_743_; 
lean_dec(v_a_721_);
v_a_736_ = lean_ctor_get(v___x_727_, 0);
v_isSharedCheck_743_ = !lean_is_exclusive(v___x_727_);
if (v_isSharedCheck_743_ == 0)
{
v___x_738_ = v___x_727_;
v_isShared_739_ = v_isSharedCheck_743_;
goto v_resetjp_737_;
}
else
{
lean_inc(v_a_736_);
lean_dec(v___x_727_);
v___x_738_ = lean_box(0);
v_isShared_739_ = v_isSharedCheck_743_;
goto v_resetjp_737_;
}
v_resetjp_737_:
{
lean_object* v___x_741_; 
if (v_isShared_739_ == 0)
{
v___x_741_ = v___x_738_;
goto v_reusejp_740_;
}
else
{
lean_object* v_reuseFailAlloc_742_; 
v_reuseFailAlloc_742_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_742_, 0, v_a_736_);
v___x_741_ = v_reuseFailAlloc_742_;
goto v_reusejp_740_;
}
v_reusejp_740_:
{
return v___x_741_;
}
}
}
}
else
{
return v___x_720_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo___redArg___boxed(lean_object* v_stx_744_, lean_object* v_a_745_, lean_object* v_a_746_, lean_object* v_a_747_, lean_object* v_a_748_){
_start:
{
lean_object* v_res_749_; 
v_res_749_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo___redArg(v_stx_744_, v_a_745_, v_a_746_, v_a_747_);
lean_dec_ref(v_a_747_);
lean_dec(v_a_746_);
lean_dec_ref(v_a_745_);
return v_res_749_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo(lean_object* v_n_750_, lean_object* v_stx_751_, lean_object* v_a_752_, lean_object* v_a_753_, lean_object* v_a_754_, lean_object* v_a_755_, lean_object* v_a_756_, lean_object* v_a_757_){
_start:
{
lean_object* v___x_759_; 
v___x_759_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo___redArg(v_stx_751_, v_a_752_, v_a_753_, v_a_754_);
return v___x_759_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo___boxed(lean_object* v_n_760_, lean_object* v_stx_761_, lean_object* v_a_762_, lean_object* v_a_763_, lean_object* v_a_764_, lean_object* v_a_765_, lean_object* v_a_766_, lean_object* v_a_767_, lean_object* v_a_768_){
_start:
{
lean_object* v_res_769_; 
v_res_769_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo(v_n_760_, v_stx_761_, v_a_762_, v_a_763_, v_a_764_, v_a_765_, v_a_766_, v_a_767_);
lean_dec(v_a_767_);
lean_dec_ref(v_a_766_);
lean_dec(v_a_765_);
lean_dec_ref(v_a_764_);
lean_dec(v_a_763_);
lean_dec_ref(v_a_762_);
lean_dec(v_n_760_);
return v_res_769_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_withAnnotateTermLikeInfo___redArg(lean_object* v_d_770_, lean_object* v_a_771_, lean_object* v_a_772_, lean_object* v_a_773_, lean_object* v_a_774_, lean_object* v_a_775_, lean_object* v_a_776_){
_start:
{
lean_object* v___x_778_; 
lean_inc(v_a_776_);
lean_inc_ref(v_a_775_);
lean_inc(v_a_774_);
lean_inc_ref(v_a_773_);
lean_inc(v_a_772_);
lean_inc_ref(v_a_771_);
v___x_778_ = lean_apply_7(v_d_770_, v_a_771_, v_a_772_, v_a_773_, v_a_774_, v_a_775_, v_a_776_, lean_box(0));
if (lean_obj_tag(v___x_778_) == 0)
{
lean_object* v_a_779_; lean_object* v___x_780_; 
v_a_779_ = lean_ctor_get(v___x_778_, 0);
lean_inc(v_a_779_);
lean_dec_ref_known(v___x_778_, 1);
v___x_780_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_annotateTermLikeInfo___redArg(v_a_779_, v_a_771_, v_a_772_, v_a_773_);
return v___x_780_;
}
else
{
return v___x_778_;
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_withAnnotateTermLikeInfo___redArg___boxed(lean_object* v_d_781_, lean_object* v_a_782_, lean_object* v_a_783_, lean_object* v_a_784_, lean_object* v_a_785_, lean_object* v_a_786_, lean_object* v_a_787_, lean_object* v_a_788_){
_start:
{
lean_object* v_res_789_; 
v_res_789_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_withAnnotateTermLikeInfo___redArg(v_d_781_, v_a_782_, v_a_783_, v_a_784_, v_a_785_, v_a_786_, v_a_787_);
lean_dec(v_a_787_);
lean_dec_ref(v_a_786_);
lean_dec(v_a_785_);
lean_dec_ref(v_a_784_);
lean_dec(v_a_783_);
lean_dec_ref(v_a_782_);
return v_res_789_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_withAnnotateTermLikeInfo(lean_object* v_n_790_, lean_object* v_d_791_, lean_object* v_a_792_, lean_object* v_a_793_, lean_object* v_a_794_, lean_object* v_a_795_, lean_object* v_a_796_, lean_object* v_a_797_){
_start:
{
lean_object* v___x_799_; 
v___x_799_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_withAnnotateTermLikeInfo___redArg(v_d_791_, v_a_792_, v_a_793_, v_a_794_, v_a_795_, v_a_796_, v_a_797_);
return v___x_799_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_Lean_PrettyPrinter_Delaborator_withAnnotateTermLikeInfo___boxed(lean_object* v_n_800_, lean_object* v_d_801_, lean_object* v_a_802_, lean_object* v_a_803_, lean_object* v_a_804_, lean_object* v_a_805_, lean_object* v_a_806_, lean_object* v_a_807_, lean_object* v_a_808_){
_start:
{
lean_object* v_res_809_; 
v_res_809_ = lp_proofwidgets_Lean_PrettyPrinter_Delaborator_withAnnotateTermLikeInfo(v_n_800_, v_d_801_, v_a_802_, v_a_803_, v_a_804_, v_a_805_, v_a_806_, v_a_807_);
lean_dec(v_a_807_);
lean_dec_ref(v_a_806_);
lean_dec(v_a_805_);
lean_dec_ref(v_a_804_);
lean_dec(v_a_803_);
lean_dec_ref(v_a_802_);
lean_dec(v_n_800_);
return v_res_809_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtx___redArg___lam__0(lean_object* v_toPure_810_, lean_object* v_00_u03b1_811_, lean_object* v___y_812_){
_start:
{
lean_object* v___x_813_; 
v___x_813_ = lean_apply_2(v_toPure_810_, lean_box(0), v___y_812_);
return v___x_813_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtx___redArg(lean_object* v_inst_814_){
_start:
{
lean_object* v_toApplicative_815_; lean_object* v_toPure_816_; lean_object* v___f_817_; 
v_toApplicative_815_ = lean_ctor_get(v_inst_814_, 0);
lean_inc_ref(v_toApplicative_815_);
lean_dec_ref(v_inst_814_);
v_toPure_816_ = lean_ctor_get(v_toApplicative_815_, 1);
lean_inc(v_toPure_816_);
lean_dec_ref(v_toApplicative_815_);
v___f_817_ = lean_alloc_closure((void*)(lp_proofwidgets_instMonadSaveCtx___redArg___lam__0), 3, 1);
lean_closure_set(v___f_817_, 0, v_toPure_816_);
return v___f_817_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtx(lean_object* v_m_818_, lean_object* v_inst_819_){
_start:
{
lean_object* v___x_820_; 
v___x_820_ = lp_proofwidgets_instMonadSaveCtx___redArg(v_inst_819_);
return v___x_820_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxReaderT___redArg___lam__0(lean_object* v_inst_821_, lean_object* v_00_u03b1_822_, lean_object* v_act_823_, lean_object* v_ctx_824_){
_start:
{
lean_object* v___x_825_; lean_object* v___x_826_; 
v___x_825_ = lean_apply_1(v_act_823_, v_ctx_824_);
v___x_826_ = lean_apply_2(v_inst_821_, lean_box(0), v___x_825_);
return v___x_826_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxReaderT___redArg(lean_object* v_inst_827_){
_start:
{
lean_object* v___f_828_; 
v___f_828_ = lean_alloc_closure((void*)(lp_proofwidgets_instMonadSaveCtxReaderT___redArg___lam__0), 4, 1);
lean_closure_set(v___f_828_, 0, v_inst_827_);
return v___f_828_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxReaderT(lean_object* v_m_829_, lean_object* v_n_830_, lean_object* v_inst_831_, lean_object* v_00_u03c1_832_){
_start:
{
lean_object* v___f_833_; 
v___f_833_ = lean_alloc_closure((void*)(lp_proofwidgets_instMonadSaveCtxReaderT___redArg___lam__0), 4, 1);
lean_closure_set(v___f_833_, 0, v_inst_831_);
return v___f_833_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateT___redArg___lam__0(lean_object* v_x_834_){
_start:
{
lean_object* v_fst_835_; 
v_fst_835_ = lean_ctor_get(v_x_834_, 0);
lean_inc(v_fst_835_);
return v_fst_835_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateT___redArg___lam__0___boxed(lean_object* v_x_836_){
_start:
{
lean_object* v_res_837_; 
v_res_837_ = lp_proofwidgets_instMonadSaveCtxStateT___redArg___lam__0(v_x_836_);
lean_dec_ref(v_x_836_);
return v_res_837_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateT___redArg___lam__1(lean_object* v_snd_838_, lean_object* v_toPure_839_, lean_object* v_a_840_){
_start:
{
lean_object* v___x_841_; lean_object* v___x_842_; 
v___x_841_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_841_, 0, v_a_840_);
lean_ctor_set(v___x_841_, 1, v_snd_838_);
v___x_842_ = lean_apply_2(v_toPure_839_, lean_box(0), v___x_841_);
return v___x_842_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateT___redArg___lam__2(lean_object* v_toFunctor_843_, lean_object* v_toPure_844_, lean_object* v_act_845_, lean_object* v___f_846_, lean_object* v_inst_847_, lean_object* v_toBind_848_, lean_object* v_____x_849_){
_start:
{
lean_object* v_fst_850_; lean_object* v_snd_851_; lean_object* v_map_852_; lean_object* v___f_853_; lean_object* v___x_854_; lean_object* v___x_855_; lean_object* v___x_856_; lean_object* v___x_857_; 
v_fst_850_ = lean_ctor_get(v_____x_849_, 0);
lean_inc(v_fst_850_);
v_snd_851_ = lean_ctor_get(v_____x_849_, 1);
lean_inc(v_snd_851_);
lean_dec_ref(v_____x_849_);
v_map_852_ = lean_ctor_get(v_toFunctor_843_, 0);
lean_inc(v_map_852_);
lean_dec_ref(v_toFunctor_843_);
v___f_853_ = lean_alloc_closure((void*)(lp_proofwidgets_instMonadSaveCtxStateT___redArg___lam__1), 3, 2);
lean_closure_set(v___f_853_, 0, v_snd_851_);
lean_closure_set(v___f_853_, 1, v_toPure_844_);
v___x_854_ = lean_apply_1(v_act_845_, v_fst_850_);
v___x_855_ = lean_apply_4(v_map_852_, lean_box(0), lean_box(0), v___f_846_, v___x_854_);
v___x_856_ = lean_apply_2(v_inst_847_, lean_box(0), v___x_855_);
v___x_857_ = lean_apply_4(v_toBind_848_, lean_box(0), lean_box(0), v___x_856_, v___f_853_);
return v___x_857_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateT___redArg___lam__3(lean_object* v_toFunctor_858_, lean_object* v_toPure_859_, lean_object* v___f_860_, lean_object* v_inst_861_, lean_object* v_toBind_862_, lean_object* v_00_u03b1_863_, lean_object* v_act_864_, lean_object* v___y_865_){
_start:
{
lean_object* v___f_866_; lean_object* v___x_867_; lean_object* v___x_868_; lean_object* v___x_869_; 
lean_inc(v_toBind_862_);
lean_inc(v_toPure_859_);
v___f_866_ = lean_alloc_closure((void*)(lp_proofwidgets_instMonadSaveCtxStateT___redArg___lam__2), 7, 6);
lean_closure_set(v___f_866_, 0, v_toFunctor_858_);
lean_closure_set(v___f_866_, 1, v_toPure_859_);
lean_closure_set(v___f_866_, 2, v_act_864_);
lean_closure_set(v___f_866_, 3, v___f_860_);
lean_closure_set(v___f_866_, 4, v_inst_861_);
lean_closure_set(v___f_866_, 5, v_toBind_862_);
lean_inc(v___y_865_);
v___x_867_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_867_, 0, v___y_865_);
lean_ctor_set(v___x_867_, 1, v___y_865_);
v___x_868_ = lean_apply_2(v_toPure_859_, lean_box(0), v___x_867_);
v___x_869_ = lean_apply_4(v_toBind_862_, lean_box(0), lean_box(0), v___x_868_, v___f_866_);
return v___x_869_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateT___redArg(lean_object* v_inst_871_, lean_object* v_inst_872_){
_start:
{
lean_object* v_toApplicative_873_; lean_object* v_toBind_874_; lean_object* v_toFunctor_875_; lean_object* v_toPure_876_; lean_object* v___f_877_; lean_object* v___f_878_; 
v_toApplicative_873_ = lean_ctor_get(v_inst_871_, 0);
lean_inc_ref(v_toApplicative_873_);
v_toBind_874_ = lean_ctor_get(v_inst_871_, 1);
lean_inc(v_toBind_874_);
lean_dec_ref(v_inst_871_);
v_toFunctor_875_ = lean_ctor_get(v_toApplicative_873_, 0);
lean_inc_ref(v_toFunctor_875_);
v_toPure_876_ = lean_ctor_get(v_toApplicative_873_, 1);
lean_inc(v_toPure_876_);
lean_dec_ref(v_toApplicative_873_);
v___f_877_ = ((lean_object*)(lp_proofwidgets_instMonadSaveCtxStateT___redArg___closed__0));
v___f_878_ = lean_alloc_closure((void*)(lp_proofwidgets_instMonadSaveCtxStateT___redArg___lam__3), 8, 5);
lean_closure_set(v___f_878_, 0, v_toFunctor_875_);
lean_closure_set(v___f_878_, 1, v_toPure_876_);
lean_closure_set(v___f_878_, 2, v___f_877_);
lean_closure_set(v___f_878_, 3, v_inst_872_);
lean_closure_set(v___f_878_, 4, v_toBind_874_);
return v___f_878_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateT(lean_object* v_m_879_, lean_object* v_n_880_, lean_object* v_inst_881_, lean_object* v_inst_882_, lean_object* v_00_u03c3_883_){
_start:
{
lean_object* v___x_884_; 
v___x_884_ = lp_proofwidgets_instMonadSaveCtxStateT___redArg(v_inst_881_, v_inst_882_);
return v___x_884_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__0(lean_object* v_a_885_, lean_object* v_toPure_886_, lean_object* v_s_887_){
_start:
{
lean_object* v___x_888_; lean_object* v___x_889_; 
v___x_888_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_888_, 0, v_a_885_);
lean_ctor_set(v___x_888_, 1, v_s_887_);
v___x_889_ = lean_apply_2(v_toPure_886_, lean_box(0), v___x_888_);
return v___x_889_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__1(lean_object* v_toPure_890_, lean_object* v_ref_891_, lean_object* v_inst_892_, lean_object* v_toBind_893_, lean_object* v_a_894_){
_start:
{
lean_object* v___f_895_; lean_object* v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; 
v___f_895_ = lean_alloc_closure((void*)(lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__0), 3, 2);
lean_closure_set(v___f_895_, 0, v_a_894_);
lean_closure_set(v___f_895_, 1, v_toPure_890_);
v___x_896_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_896_, 0, lean_box(0));
lean_closure_set(v___x_896_, 1, lean_box(0));
lean_closure_set(v___x_896_, 2, v_ref_891_);
v___x_897_ = lean_apply_2(v_inst_892_, lean_box(0), v___x_896_);
v___x_898_ = lean_apply_4(v_toBind_893_, lean_box(0), lean_box(0), v___x_897_, v___f_895_);
return v___x_898_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__2(lean_object* v_toPure_899_, lean_object* v_inst_900_, lean_object* v_toBind_901_, lean_object* v_act_902_, lean_object* v_ref_903_){
_start:
{
lean_object* v___f_904_; lean_object* v___x_905_; lean_object* v___x_906_; 
lean_inc(v_toBind_901_);
lean_inc(v_ref_903_);
v___f_904_ = lean_alloc_closure((void*)(lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__1), 5, 4);
lean_closure_set(v___f_904_, 0, v_toPure_899_);
lean_closure_set(v___f_904_, 1, v_ref_903_);
lean_closure_set(v___f_904_, 2, v_inst_900_);
lean_closure_set(v___f_904_, 3, v_toBind_901_);
v___x_905_ = lean_apply_1(v_act_902_, v_ref_903_);
v___x_906_ = lean_apply_4(v_toBind_901_, lean_box(0), lean_box(0), v___x_905_, v___f_904_);
return v___x_906_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__3(lean_object* v_toPure_907_, lean_object* v_____x_908_){
_start:
{
lean_object* v_fst_909_; lean_object* v___x_910_; 
v_fst_909_ = lean_ctor_get(v_____x_908_, 0);
lean_inc(v_fst_909_);
lean_dec_ref(v_____x_908_);
v___x_910_ = lean_apply_2(v_toPure_907_, lean_box(0), v_fst_909_);
return v___x_910_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__4(lean_object* v_toApplicative_911_, lean_object* v_inst_912_, lean_object* v_toBind_913_, lean_object* v_act_914_, lean_object* v_inst_915_, lean_object* v_a_916_){
_start:
{
lean_object* v_toPure_917_; lean_object* v___x_918_; lean_object* v___x_919_; lean_object* v___f_920_; lean_object* v___f_921_; lean_object* v___x_922_; lean_object* v___x_923_; lean_object* v___x_924_; 
v_toPure_917_ = lean_ctor_get(v_toApplicative_911_, 1);
lean_inc_n(v_toPure_917_, 2);
lean_dec_ref(v_toApplicative_911_);
v___x_918_ = lean_alloc_closure((void*)(l_ST_Prim_mkRef___boxed), 4, 3);
lean_closure_set(v___x_918_, 0, lean_box(0));
lean_closure_set(v___x_918_, 1, lean_box(0));
lean_closure_set(v___x_918_, 2, v_a_916_);
lean_inc(v_inst_912_);
v___x_919_ = lean_apply_2(v_inst_912_, lean_box(0), v___x_918_);
lean_inc_n(v_toBind_913_, 2);
v___f_920_ = lean_alloc_closure((void*)(lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__2), 5, 4);
lean_closure_set(v___f_920_, 0, v_toPure_917_);
lean_closure_set(v___f_920_, 1, v_inst_912_);
lean_closure_set(v___f_920_, 2, v_toBind_913_);
lean_closure_set(v___f_920_, 3, v_act_914_);
v___f_921_ = lean_alloc_closure((void*)(lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__3), 2, 1);
lean_closure_set(v___f_921_, 0, v_toPure_917_);
v___x_922_ = lean_apply_4(v_toBind_913_, lean_box(0), lean_box(0), v___x_919_, v___f_920_);
v___x_923_ = lean_apply_4(v_toBind_913_, lean_box(0), lean_box(0), v___x_922_, v___f_921_);
v___x_924_ = lean_apply_2(v_inst_915_, lean_box(0), v___x_923_);
return v___x_924_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__5(lean_object* v_inst_925_, lean_object* v_inst_926_, lean_object* v_inst_927_, lean_object* v_00_u03b1_928_, lean_object* v_act_929_, lean_object* v___y_930_){
_start:
{
lean_object* v_toApplicative_931_; lean_object* v_toBind_932_; lean_object* v___f_933_; lean_object* v___x_934_; lean_object* v___x_935_; lean_object* v___x_936_; 
v_toApplicative_931_ = lean_ctor_get(v_inst_925_, 0);
lean_inc_ref(v_toApplicative_931_);
v_toBind_932_ = lean_ctor_get(v_inst_925_, 1);
lean_inc_n(v_toBind_932_, 2);
lean_dec_ref(v_inst_925_);
lean_inc(v_inst_926_);
v___f_933_ = lean_alloc_closure((void*)(lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__4), 6, 5);
lean_closure_set(v___f_933_, 0, v_toApplicative_931_);
lean_closure_set(v___f_933_, 1, v_inst_926_);
lean_closure_set(v___f_933_, 2, v_toBind_932_);
lean_closure_set(v___f_933_, 3, v_act_929_);
lean_closure_set(v___f_933_, 4, v_inst_927_);
lean_inc(v___y_930_);
v___x_934_ = lean_alloc_closure((void*)(l_ST_Prim_Ref_get___boxed), 4, 3);
lean_closure_set(v___x_934_, 0, lean_box(0));
lean_closure_set(v___x_934_, 1, lean_box(0));
lean_closure_set(v___x_934_, 2, v___y_930_);
v___x_935_ = lean_apply_2(v_inst_926_, lean_box(0), v___x_934_);
v___x_936_ = lean_apply_4(v_toBind_932_, lean_box(0), lean_box(0), v___x_935_, v___f_933_);
return v___x_936_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__5___boxed(lean_object* v_inst_937_, lean_object* v_inst_938_, lean_object* v_inst_939_, lean_object* v_00_u03b1_940_, lean_object* v_act_941_, lean_object* v___y_942_){
_start:
{
lean_object* v_res_943_; 
v_res_943_ = lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__5(v_inst_937_, v_inst_938_, v_inst_939_, v_00_u03b1_940_, v_act_941_, v___y_942_);
lean_dec(v___y_942_);
return v_res_943_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg(lean_object* v_inst_944_, lean_object* v_inst_945_, lean_object* v_inst_946_){
_start:
{
lean_object* v___f_947_; 
v___f_947_ = lean_alloc_closure((void*)(lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__5___boxed), 6, 3);
lean_closure_set(v___f_947_, 0, v_inst_944_);
lean_closure_set(v___f_947_, 1, v_inst_946_);
lean_closure_set(v___f_947_, 2, v_inst_945_);
return v___f_947_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST(lean_object* v_m_948_, lean_object* v_n_949_, lean_object* v_inst_950_, lean_object* v_inst_951_, lean_object* v_00_u03c9_952_, lean_object* v_00_u03c3_953_, lean_object* v_inst_954_){
_start:
{
lean_object* v___f_955_; 
v___f_955_ = lean_alloc_closure((void*)(lp_proofwidgets_instMonadSaveCtxStateRefT_x27OfMonadLiftTST___redArg___lam__5___boxed), 6, 3);
lean_closure_set(v___f_955_, 0, v_inst_950_);
lean_closure_set(v___f_955_, 1, v_inst_954_);
lean_closure_set(v___f_955_, 2, v_inst_951_);
return v___f_955_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instToJsonUnit__proofWidgets___lam__0(lean_object* v_x_956_){
_start:
{
lean_object* v___x_957_; 
v___x_957_ = lean_box(0);
return v___x_957_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instFromJsonUnit__proofWidgets___lam__0(lean_object* v_x_962_){
_start:
{
lean_object* v___x_963_; 
v___x_963_ = ((lean_object*)(lp_proofwidgets_instFromJsonUnit__proofWidgets___lam__0___closed__0));
return v___x_963_;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_instFromJsonUnit__proofWidgets___lam__0___boxed(lean_object* v_x_964_){
_start:
{
lean_object* v_res_965_; 
v_res_965_ = lp_proofwidgets_instFromJsonUnit__proofWidgets___lam__0(v_x_964_);
lean_dec(v_x_964_);
return v_res_965_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_proofwidgets_ProofWidgets_Util(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize_runtime_module();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_PrettyPrinter_Delaborator_Basic(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_proofwidgets_ProofWidgets_Util(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_PrettyPrinter_Delaborator_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_PrettyPrinter_Delaborator_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_proofwidgets_ProofWidgets_Util(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_PrettyPrinter_Delaborator_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_proofwidgets_ProofWidgets_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_proofwidgets_ProofWidgets_Util(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_proofwidgets_ProofWidgets_Util(builtin);
}
#ifdef __cplusplus
}
#endif
