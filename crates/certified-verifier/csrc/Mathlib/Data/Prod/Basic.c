// Lean compiler output
// Module: Mathlib.Data.Prod.Basic
// Imports: public import Init public meta import Init public import Mathlib.Logic.Function.Defs public import Mathlib.Logic.Function.Iterate public import Mathlib.Tactic.Inhabit public import Batteries.Tactic.Trans public meta import Lean.PrettyPrinter.Delaborator.Builtins
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
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_SubExpr_Pos_push(lean_object*, lean_object*);
lean_object* l_Lean_SourceInfo_fromRef(lean_object*, uint8_t);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* l_Lean_Syntax_node1(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Syntax_node3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_delab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPExplicit___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_getPPFieldNotation___boxed(lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr3(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_register_option(lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_PrettyPrinter_Delaborator_whenPPOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_mk_injArrow___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_mk_injArrow(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_mk_injArrow___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Prod_Lex_decidable___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_decidable___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Prod_Lex_decidable(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_decidable___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__0_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "pp"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__0_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__0_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__1_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 12, .m_capacity = 12, .m_length = 11, .m_data = "numericProj"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__1_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__1_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__2_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "prod"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__2_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__2_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__3_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__0_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(249, 51, 192, 169, 230, 180, 160, 93)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__3_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__3_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__1_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(145, 113, 157, 154, 1, 229, 69, 81)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__3_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__3_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__2_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(97, 153, 56, 67, 174, 125, 112, 126)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__3_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__3_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__4_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 72, .m_capacity = 72, .m_length = 71, .m_data = "enable pretty printing `Prod.fst x` as `x.1` and `Prod.snd x` as `x.2`."};
static const lean_object* lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__4_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__4_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__5_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 0}, .m_objs = {((lean_object*)(((size_t)(1) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__4_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value),((lean_object*)(((size_t)(0) << 1) | 1))}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__5_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__5_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__6_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Prod"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__6_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__6_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value;
static const lean_string_object lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__7_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 15, .m_capacity = 15, .m_length = 14, .m_data = "PrettyPrinting"};
static const lean_object* lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__7_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__7_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__8_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__6_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(121, 119, 164, 206, 221, 118, 48, 212)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__8_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__8_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__7_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(83, 140, 66, 177, 49, 137, 42, 224)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__8_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__8_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value_aux_1),((lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__0_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(129, 49, 157, 58, 162, 206, 67, 74)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__8_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value_aux_3 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__8_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value_aux_2),((lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__1_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(25, 128, 22, 22, 16, 43, 68, 214)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__8_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__8_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value_aux_3),((lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__2_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value),LEAN_SCALAR_PTR_LITERAL(57, 189, 220, 147, 27, 117, 106, 108)}};
static const lean_object* lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__8_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4_ = (const lean_object*)&lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__8_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_pp_numericProj_prod;
LEAN_EXPORT uint8_t lp_mathlib_Prod_PrettyPrinting_getPPNumericProjProd(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_getPPNumericProjProd___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__0___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__0___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__0_value;
static const lean_string_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__1 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__1_value;
static const lean_string_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Term"};
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__2 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__2_value;
static const lean_string_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "proj"};
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__3 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__3_value;
static const lean_ctor_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__2_value),LEAN_SCALAR_PTR_LITERAL(75, 170, 162, 138, 136, 204, 251, 229)}};
static const lean_ctor_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__3_value),LEAN_SCALAR_PTR_LITERAL(103, 149, 207, 196, 17, 4, 77, 74)}};
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__4 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__4_value;
static const lean_string_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "."};
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__5 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__5_value;
static const lean_string_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 9, .m_capacity = 9, .m_length = 8, .m_data = "fieldIdx"};
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__6 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__6_value;
static const lean_ctor_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__6_value),LEAN_SCALAR_PTR_LITERAL(243, 141, 165, 29, 238, 211, 61, 163)}};
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__7 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__7_value;
static const lean_string_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "1"};
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__8 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__8_value;
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Prod_PrettyPrinting_getPPNumericProjProd___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__0 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__0_value;
static const lean_closure_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPFieldNotation___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__1 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__1_value;
static const lean_closure_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_getPPExplicit___boxed, .m_arity = 1, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__2 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__2_value;
static const lean_closure_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_delab___boxed, .m_arity = 7, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__3 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__3_value;
static const lean_closure_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___boxed, .m_arity = 8, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__3_value)} };
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__4 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__4_value;
static const lean_closure_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__4_value)} };
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__5 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__5_value;
static const lean_closure_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__2_value),((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__5_value)} };
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__6 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__6_value;
static const lean_closure_object lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__1_value),((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__6_value)} };
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__7 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__7_value;
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Prod_PrettyPrinting_delabProdSnd___lam__0___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = "2"};
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdSnd___lam__0___closed__0 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdSnd___lam__0___closed__0_value;
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdSnd___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdSnd___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_closure_object lp_mathlib_Prod_PrettyPrinting_delabProdSnd___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*1, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_mathlib_Prod_PrettyPrinting_delabProdSnd___lam__0___boxed, .m_arity = 8, .m_num_fixed = 1, .m_objs = {((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__3_value)} };
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdSnd___closed__0 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdSnd___closed__0_value;
static const lean_closure_object lp_mathlib_Prod_PrettyPrinting_delabProdSnd___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_withOverApp___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)(((size_t)(3) << 1) | 1)),((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdSnd___closed__0_value)} };
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdSnd___closed__1 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdSnd___closed__1_value;
static const lean_closure_object lp_mathlib_Prod_PrettyPrinting_delabProdSnd___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenNotPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__2_value),((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdSnd___closed__1_value)} };
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdSnd___closed__2 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdSnd___closed__2_value;
static const lean_closure_object lp_mathlib_Prod_PrettyPrinting_delabProdSnd___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*2, .m_other = 0, .m_tag = 245}, .m_fun = (void*)l_Lean_PrettyPrinter_Delaborator_whenPPOption___boxed, .m_arity = 9, .m_num_fixed = 2, .m_objs = {((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__1_value),((lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdSnd___closed__2_value)} };
static const lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdSnd___closed__3 = (const lean_object*)&lp_mathlib_Prod_PrettyPrinting_delabProdSnd___closed__3_value;
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdSnd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdSnd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_mk_injArrow___redArg(lean_object* v_w_1_){
_start:
{
lean_object* v___x_2_; 
v___x_2_ = lean_apply_2(v_w_1_, lean_box(0), lean_box(0));
return v___x_2_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_mk_injArrow(lean_object* v_00_u03b1_3_, lean_object* v_00_u03b2_4_, lean_object* v_x_u2081_5_, lean_object* v_y_u2081_6_, lean_object* v_x_u2082_7_, lean_object* v_y_u2082_8_, lean_object* v_h_9_, lean_object* v_P_10_, lean_object* v_w_11_){
_start:
{
lean_object* v___x_12_; 
v___x_12_ = lean_apply_2(v_w_11_, lean_box(0), lean_box(0));
return v___x_12_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_mk_injArrow___boxed(lean_object* v_00_u03b1_13_, lean_object* v_00_u03b2_14_, lean_object* v_x_u2081_15_, lean_object* v_y_u2081_16_, lean_object* v_x_u2082_17_, lean_object* v_y_u2082_18_, lean_object* v_h_19_, lean_object* v_P_20_, lean_object* v_w_21_){
_start:
{
lean_object* v_res_22_; 
v_res_22_ = lp_mathlib_Prod_mk_injArrow(v_00_u03b1_13_, v_00_u03b2_14_, v_x_u2081_15_, v_y_u2081_16_, v_x_u2082_17_, v_y_u2082_18_, v_h_19_, v_P_20_, v_w_21_);
lean_dec(v_y_u2082_18_);
lean_dec(v_x_u2082_17_);
lean_dec(v_y_u2081_16_);
lean_dec(v_x_u2081_15_);
return v_res_22_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Prod_Lex_decidable___redArg(lean_object* v_inst_23_, lean_object* v_inst_24_, lean_object* v_inst_25_, lean_object* v_x_26_, lean_object* v_x_27_){
_start:
{
lean_object* v_fst_28_; lean_object* v_snd_29_; lean_object* v_fst_30_; lean_object* v_snd_31_; lean_object* v___x_32_; uint8_t v___x_33_; 
v_fst_28_ = lean_ctor_get(v_x_26_, 0);
lean_inc_n(v_fst_28_, 2);
v_snd_29_ = lean_ctor_get(v_x_26_, 1);
lean_inc(v_snd_29_);
lean_dec_ref(v_x_26_);
v_fst_30_ = lean_ctor_get(v_x_27_, 0);
lean_inc_n(v_fst_30_, 2);
v_snd_31_ = lean_ctor_get(v_x_27_, 1);
lean_inc(v_snd_31_);
lean_dec_ref(v_x_27_);
v___x_32_ = lean_apply_2(v_inst_24_, v_fst_28_, v_fst_30_);
v___x_33_ = lean_unbox(v___x_32_);
if (v___x_33_ == 0)
{
lean_object* v___x_34_; uint8_t v___x_35_; 
v___x_34_ = lean_apply_2(v_inst_23_, v_fst_28_, v_fst_30_);
v___x_35_ = lean_unbox(v___x_34_);
if (v___x_35_ == 0)
{
uint8_t v___x_36_; 
lean_dec(v_snd_31_);
lean_dec(v_snd_29_);
lean_dec_ref(v_inst_25_);
v___x_36_ = lean_unbox(v___x_34_);
return v___x_36_;
}
else
{
lean_object* v___x_37_; uint8_t v___x_38_; 
v___x_37_ = lean_apply_2(v_inst_25_, v_snd_29_, v_snd_31_);
v___x_38_ = lean_unbox(v___x_37_);
return v___x_38_;
}
}
else
{
uint8_t v___x_39_; 
lean_dec(v_snd_31_);
lean_dec(v_fst_30_);
lean_dec(v_snd_29_);
lean_dec(v_fst_28_);
lean_dec_ref(v_inst_25_);
lean_dec_ref(v_inst_23_);
v___x_39_ = lean_unbox(v___x_32_);
return v___x_39_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_decidable___redArg___boxed(lean_object* v_inst_40_, lean_object* v_inst_41_, lean_object* v_inst_42_, lean_object* v_x_43_, lean_object* v_x_44_){
_start:
{
uint8_t v_res_45_; lean_object* v_r_46_; 
v_res_45_ = lp_mathlib_Prod_Lex_decidable___redArg(v_inst_40_, v_inst_41_, v_inst_42_, v_x_43_, v_x_44_);
v_r_46_ = lean_box(v_res_45_);
return v_r_46_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Prod_Lex_decidable(lean_object* v_00_u03b1_47_, lean_object* v_00_u03b2_48_, lean_object* v_inst_49_, lean_object* v_r_50_, lean_object* v_s_51_, lean_object* v_inst_52_, lean_object* v_inst_53_, lean_object* v_x_54_, lean_object* v_x_55_){
_start:
{
uint8_t v___x_56_; 
v___x_56_ = lp_mathlib_Prod_Lex_decidable___redArg(v_inst_49_, v_inst_52_, v_inst_53_, v_x_54_, v_x_55_);
return v___x_56_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_Lex_decidable___boxed(lean_object* v_00_u03b1_57_, lean_object* v_00_u03b2_58_, lean_object* v_inst_59_, lean_object* v_r_60_, lean_object* v_s_61_, lean_object* v_inst_62_, lean_object* v_inst_63_, lean_object* v_x_64_, lean_object* v_x_65_){
_start:
{
uint8_t v_res_66_; lean_object* v_r_67_; 
v_res_66_ = lp_mathlib_Prod_Lex_decidable(v_00_u03b1_57_, v_00_u03b2_58_, v_inst_59_, v_r_60_, v_s_61_, v_inst_62_, v_inst_63_, v_x_64_, v_x_65_);
v_r_67_ = lean_box(v_res_66_);
return v_r_67_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__spec__0(lean_object* v_name_68_, lean_object* v_decl_69_, lean_object* v_ref_70_){
_start:
{
lean_object* v_defValue_72_; lean_object* v_descr_73_; lean_object* v_deprecation_x3f_74_; lean_object* v___x_75_; uint8_t v___x_76_; lean_object* v___x_77_; lean_object* v___x_78_; 
v_defValue_72_ = lean_ctor_get(v_decl_69_, 0);
v_descr_73_ = lean_ctor_get(v_decl_69_, 1);
v_deprecation_x3f_74_ = lean_ctor_get(v_decl_69_, 2);
v___x_75_ = lean_alloc_ctor(1, 0, 1);
v___x_76_ = lean_unbox(v_defValue_72_);
lean_ctor_set_uint8(v___x_75_, 0, v___x_76_);
lean_inc(v_deprecation_x3f_74_);
lean_inc_ref(v_descr_73_);
lean_inc_n(v_name_68_, 2);
v___x_77_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v___x_77_, 0, v_name_68_);
lean_ctor_set(v___x_77_, 1, v_ref_70_);
lean_ctor_set(v___x_77_, 2, v___x_75_);
lean_ctor_set(v___x_77_, 3, v_descr_73_);
lean_ctor_set(v___x_77_, 4, v_deprecation_x3f_74_);
v___x_78_ = lean_register_option(v_name_68_, v___x_77_);
if (lean_obj_tag(v___x_78_) == 0)
{
lean_object* v___x_80_; uint8_t v_isShared_81_; uint8_t v_isSharedCheck_86_; 
v_isSharedCheck_86_ = !lean_is_exclusive(v___x_78_);
if (v_isSharedCheck_86_ == 0)
{
lean_object* v_unused_87_; 
v_unused_87_ = lean_ctor_get(v___x_78_, 0);
lean_dec(v_unused_87_);
v___x_80_ = v___x_78_;
v_isShared_81_ = v_isSharedCheck_86_;
goto v_resetjp_79_;
}
else
{
lean_dec(v___x_78_);
v___x_80_ = lean_box(0);
v_isShared_81_ = v_isSharedCheck_86_;
goto v_resetjp_79_;
}
v_resetjp_79_:
{
lean_object* v___x_82_; lean_object* v___x_84_; 
lean_inc(v_defValue_72_);
v___x_82_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_82_, 0, v_name_68_);
lean_ctor_set(v___x_82_, 1, v_defValue_72_);
if (v_isShared_81_ == 0)
{
lean_ctor_set(v___x_80_, 0, v___x_82_);
v___x_84_ = v___x_80_;
goto v_reusejp_83_;
}
else
{
lean_object* v_reuseFailAlloc_85_; 
v_reuseFailAlloc_85_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_85_, 0, v___x_82_);
v___x_84_ = v_reuseFailAlloc_85_;
goto v_reusejp_83_;
}
v_reusejp_83_:
{
return v___x_84_;
}
}
}
else
{
lean_object* v_a_88_; lean_object* v___x_90_; uint8_t v_isShared_91_; uint8_t v_isSharedCheck_95_; 
lean_dec(v_name_68_);
v_a_88_ = lean_ctor_get(v___x_78_, 0);
v_isSharedCheck_95_ = !lean_is_exclusive(v___x_78_);
if (v_isSharedCheck_95_ == 0)
{
v___x_90_ = v___x_78_;
v_isShared_91_ = v_isSharedCheck_95_;
goto v_resetjp_89_;
}
else
{
lean_inc(v_a_88_);
lean_dec(v___x_78_);
v___x_90_ = lean_box(0);
v_isShared_91_ = v_isSharedCheck_95_;
goto v_resetjp_89_;
}
v_resetjp_89_:
{
lean_object* v___x_93_; 
if (v_isShared_91_ == 0)
{
v___x_93_ = v___x_90_;
goto v_reusejp_92_;
}
else
{
lean_object* v_reuseFailAlloc_94_; 
v_reuseFailAlloc_94_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_94_, 0, v_a_88_);
v___x_93_ = v_reuseFailAlloc_94_;
goto v_reusejp_92_;
}
v_reusejp_92_:
{
return v___x_93_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__spec__0___boxed(lean_object* v_name_96_, lean_object* v_decl_97_, lean_object* v_ref_98_, lean_object* v_a_99_){
_start:
{
lean_object* v_res_100_; 
v_res_100_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__spec__0(v_name_96_, v_decl_97_, v_ref_98_);
lean_dec_ref(v_decl_97_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4_(){
_start:
{
lean_object* v___x_123_; lean_object* v___x_124_; lean_object* v___x_125_; lean_object* v___x_126_; 
v___x_123_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__3_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4_));
v___x_124_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__5_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4_));
v___x_125_ = ((lean_object*)(lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn___closed__8_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4_));
v___x_126_ = lp_mathlib_Lean_Option_register___at___00__private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4__spec__0(v___x_123_, v___x_124_, v___x_125_);
return v___x_126_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4____boxed(lean_object* v_a_127_){
_start:
{
lean_object* v_res_128_; 
v_res_128_ = lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4_();
return v_res_128_;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Prod_PrettyPrinting_getPPNumericProjProd(lean_object* v_o_129_){
_start:
{
lean_object* v___x_130_; lean_object* v_name_131_; lean_object* v_defValue_132_; lean_object* v_map_133_; lean_object* v___x_134_; 
v___x_130_ = lp_mathlib_Prod_PrettyPrinting_pp_numericProj_prod;
v_name_131_ = lean_ctor_get(v___x_130_, 0);
v_defValue_132_ = lean_ctor_get(v___x_130_, 1);
v_map_133_ = lean_ctor_get(v_o_129_, 0);
v___x_134_ = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(v_map_133_, v_name_131_);
if (lean_obj_tag(v___x_134_) == 0)
{
uint8_t v___x_135_; 
v___x_135_ = lean_unbox(v_defValue_132_);
return v___x_135_;
}
else
{
lean_object* v_val_136_; 
v_val_136_ = lean_ctor_get(v___x_134_, 0);
lean_inc(v_val_136_);
lean_dec_ref_known(v___x_134_, 1);
if (lean_obj_tag(v_val_136_) == 1)
{
uint8_t v_v_137_; 
v_v_137_ = lean_ctor_get_uint8(v_val_136_, 0);
lean_dec_ref_known(v_val_136_, 0);
return v_v_137_;
}
else
{
uint8_t v___x_138_; 
lean_dec(v_val_136_);
v___x_138_ = lean_unbox(v_defValue_132_);
return v___x_138_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_getPPNumericProjProd___boxed(lean_object* v_o_139_){
_start:
{
uint8_t v_res_140_; lean_object* v_r_141_; 
v_res_140_ = lp_mathlib_Prod_PrettyPrinting_getPPNumericProjProd(v_o_139_);
lean_dec_ref(v_o_139_);
v_r_141_ = lean_box(v_res_140_);
return v_r_141_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__1___redArg(lean_object* v_child_142_, lean_object* v_childIdx_143_, lean_object* v_x_144_, lean_object* v___y_145_, lean_object* v___y_146_, lean_object* v___y_147_, lean_object* v___y_148_, lean_object* v___y_149_, lean_object* v___y_150_){
_start:
{
lean_object* v_subExpr_152_; lean_object* v_optionsPerPos_153_; lean_object* v_currNamespace_154_; lean_object* v_openDecls_155_; uint8_t v_inPattern_156_; lean_object* v_depth_157_; lean_object* v_lctxInitIndices_158_; lean_object* v_pos_159_; lean_object* v___x_160_; lean_object* v___x_161_; lean_object* v___x_162_; lean_object* v___x_163_; 
v_subExpr_152_ = lean_ctor_get(v___y_145_, 3);
v_optionsPerPos_153_ = lean_ctor_get(v___y_145_, 0);
v_currNamespace_154_ = lean_ctor_get(v___y_145_, 1);
v_openDecls_155_ = lean_ctor_get(v___y_145_, 2);
v_inPattern_156_ = lean_ctor_get_uint8(v___y_145_, sizeof(void*)*6);
v_depth_157_ = lean_ctor_get(v___y_145_, 4);
v_lctxInitIndices_158_ = lean_ctor_get(v___y_145_, 5);
v_pos_159_ = lean_ctor_get(v_subExpr_152_, 1);
v___x_160_ = l_Lean_SubExpr_Pos_push(v_pos_159_, v_childIdx_143_);
v___x_161_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_161_, 0, v_child_142_);
lean_ctor_set(v___x_161_, 1, v___x_160_);
lean_inc(v_lctxInitIndices_158_);
lean_inc(v_depth_157_);
lean_inc(v_openDecls_155_);
lean_inc(v_currNamespace_154_);
lean_inc(v_optionsPerPos_153_);
v___x_162_ = lean_alloc_ctor(0, 6, 1);
lean_ctor_set(v___x_162_, 0, v_optionsPerPos_153_);
lean_ctor_set(v___x_162_, 1, v_currNamespace_154_);
lean_ctor_set(v___x_162_, 2, v_openDecls_155_);
lean_ctor_set(v___x_162_, 3, v___x_161_);
lean_ctor_set(v___x_162_, 4, v_depth_157_);
lean_ctor_set(v___x_162_, 5, v_lctxInitIndices_158_);
lean_ctor_set_uint8(v___x_162_, sizeof(void*)*6, v_inPattern_156_);
lean_inc(v___y_150_);
lean_inc_ref(v___y_149_);
lean_inc(v___y_148_);
lean_inc_ref(v___y_147_);
lean_inc(v___y_146_);
v___x_163_ = lean_apply_7(v_x_144_, v___x_162_, v___y_146_, v___y_147_, v___y_148_, v___y_149_, v___y_150_, lean_box(0));
return v___x_163_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__1___redArg___boxed(lean_object* v_child_164_, lean_object* v_childIdx_165_, lean_object* v_x_166_, lean_object* v___y_167_, lean_object* v___y_168_, lean_object* v___y_169_, lean_object* v___y_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_){
_start:
{
lean_object* v_res_174_; 
v_res_174_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__1___redArg(v_child_164_, v_childIdx_165_, v_x_166_, v___y_167_, v___y_168_, v___y_169_, v___y_170_, v___y_171_, v___y_172_);
lean_dec(v___y_172_);
lean_dec_ref(v___y_171_);
lean_dec(v___y_170_);
lean_dec_ref(v___y_169_);
lean_dec(v___y_168_);
lean_dec_ref(v___y_167_);
return v_res_174_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__0___redArg(lean_object* v___y_175_){
_start:
{
lean_object* v_subExpr_177_; lean_object* v_expr_178_; lean_object* v___x_179_; 
v_subExpr_177_ = lean_ctor_get(v___y_175_, 3);
v_expr_178_ = lean_ctor_get(v_subExpr_177_, 0);
lean_inc_ref(v_expr_178_);
v___x_179_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_179_, 0, v_expr_178_);
return v___x_179_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__0___redArg___boxed(lean_object* v___y_180_, lean_object* v___y_181_){
_start:
{
lean_object* v_res_182_; 
v_res_182_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__0___redArg(v___y_180_);
lean_dec_ref(v___y_180_);
return v_res_182_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0___redArg(lean_object* v_x_183_, lean_object* v___y_184_, lean_object* v___y_185_, lean_object* v___y_186_, lean_object* v___y_187_, lean_object* v___y_188_, lean_object* v___y_189_){
_start:
{
lean_object* v___x_191_; lean_object* v_a_192_; lean_object* v___x_193_; lean_object* v___x_194_; lean_object* v___x_195_; 
v___x_191_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__0___redArg(v___y_184_);
v_a_192_ = lean_ctor_get(v___x_191_, 0);
lean_inc(v_a_192_);
lean_dec_ref(v___x_191_);
v___x_193_ = l_Lean_Expr_appArg_x21(v_a_192_);
lean_dec(v_a_192_);
v___x_194_ = lean_unsigned_to_nat(1u);
v___x_195_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__1___redArg(v___x_193_, v___x_194_, v_x_183_, v___y_184_, v___y_185_, v___y_186_, v___y_187_, v___y_188_, v___y_189_);
return v___x_195_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0___redArg___boxed(lean_object* v_x_196_, lean_object* v___y_197_, lean_object* v___y_198_, lean_object* v___y_199_, lean_object* v___y_200_, lean_object* v___y_201_, lean_object* v___y_202_, lean_object* v___y_203_){
_start:
{
lean_object* v_res_204_; 
v_res_204_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0___redArg(v_x_196_, v___y_197_, v___y_198_, v___y_199_, v___y_200_, v___y_201_, v___y_202_);
lean_dec(v___y_202_);
lean_dec_ref(v___y_201_);
lean_dec(v___y_200_);
lean_dec_ref(v___y_199_);
lean_dec(v___y_198_);
lean_dec_ref(v___y_197_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0(lean_object* v___x_219_, lean_object* v___y_220_, lean_object* v___y_221_, lean_object* v___y_222_, lean_object* v___y_223_, lean_object* v___y_224_, lean_object* v___y_225_){
_start:
{
lean_object* v___x_227_; 
v___x_227_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0___redArg(v___x_219_, v___y_220_, v___y_221_, v___y_222_, v___y_223_, v___y_224_, v___y_225_);
if (lean_obj_tag(v___x_227_) == 0)
{
lean_object* v_a_228_; lean_object* v___x_230_; uint8_t v_isShared_231_; uint8_t v_isSharedCheck_246_; 
v_a_228_ = lean_ctor_get(v___x_227_, 0);
v_isSharedCheck_246_ = !lean_is_exclusive(v___x_227_);
if (v_isSharedCheck_246_ == 0)
{
v___x_230_ = v___x_227_;
v_isShared_231_ = v_isSharedCheck_246_;
goto v_resetjp_229_;
}
else
{
lean_inc(v_a_228_);
lean_dec(v___x_227_);
v___x_230_ = lean_box(0);
v_isShared_231_ = v_isSharedCheck_246_;
goto v_resetjp_229_;
}
v_resetjp_229_:
{
lean_object* v_ref_232_; uint8_t v___x_233_; lean_object* v___x_234_; lean_object* v___x_235_; lean_object* v___x_236_; lean_object* v___x_237_; lean_object* v___x_238_; lean_object* v___x_239_; lean_object* v___x_240_; lean_object* v___x_241_; lean_object* v___x_242_; lean_object* v___x_244_; 
v_ref_232_ = lean_ctor_get(v___y_224_, 5);
v___x_233_ = 0;
v___x_234_ = l_Lean_SourceInfo_fromRef(v_ref_232_, v___x_233_);
v___x_235_ = ((lean_object*)(lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__4));
v___x_236_ = ((lean_object*)(lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__5));
lean_inc_n(v___x_234_, 3);
v___x_237_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_237_, 0, v___x_234_);
lean_ctor_set(v___x_237_, 1, v___x_236_);
v___x_238_ = ((lean_object*)(lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__7));
v___x_239_ = ((lean_object*)(lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__8));
v___x_240_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_240_, 0, v___x_234_);
lean_ctor_set(v___x_240_, 1, v___x_239_);
v___x_241_ = l_Lean_Syntax_node1(v___x_234_, v___x_238_, v___x_240_);
v___x_242_ = l_Lean_Syntax_node3(v___x_234_, v___x_235_, v_a_228_, v___x_237_, v___x_241_);
if (v_isShared_231_ == 0)
{
lean_ctor_set(v___x_230_, 0, v___x_242_);
v___x_244_ = v___x_230_;
goto v_reusejp_243_;
}
else
{
lean_object* v_reuseFailAlloc_245_; 
v_reuseFailAlloc_245_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_245_, 0, v___x_242_);
v___x_244_ = v_reuseFailAlloc_245_;
goto v_reusejp_243_;
}
v_reusejp_243_:
{
return v___x_244_;
}
}
}
else
{
return v___x_227_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___boxed(lean_object* v___x_247_, lean_object* v___y_248_, lean_object* v___y_249_, lean_object* v___y_250_, lean_object* v___y_251_, lean_object* v___y_252_, lean_object* v___y_253_, lean_object* v___y_254_){
_start:
{
lean_object* v_res_255_; 
v_res_255_ = lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0(v___x_247_, v___y_248_, v___y_249_, v___y_250_, v___y_251_, v___y_252_, v___y_253_);
lean_dec(v___y_253_);
lean_dec_ref(v___y_252_);
lean_dec(v___y_251_);
lean_dec_ref(v___y_250_);
lean_dec(v___y_249_);
lean_dec_ref(v___y_248_);
return v_res_255_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst(lean_object* v_a_271_, lean_object* v_a_272_, lean_object* v_a_273_, lean_object* v_a_274_, lean_object* v_a_275_, lean_object* v_a_276_){
_start:
{
lean_object* v___x_278_; lean_object* v___x_279_; lean_object* v___x_280_; 
v___x_278_ = ((lean_object*)(lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__0));
v___x_279_ = ((lean_object*)(lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__7));
v___x_280_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_278_, v___x_279_, v_a_271_, v_a_272_, v_a_273_, v_a_274_, v_a_275_, v_a_276_);
return v___x_280_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdFst___boxed(lean_object* v_a_281_, lean_object* v_a_282_, lean_object* v_a_283_, lean_object* v_a_284_, lean_object* v_a_285_, lean_object* v_a_286_, lean_object* v_a_287_){
_start:
{
lean_object* v_res_288_; 
v_res_288_ = lp_mathlib_Prod_PrettyPrinting_delabProdFst(v_a_281_, v_a_282_, v_a_283_, v_a_284_, v_a_285_, v_a_286_);
lean_dec(v_a_286_);
lean_dec_ref(v_a_285_);
lean_dec(v_a_284_);
lean_dec_ref(v_a_283_);
lean_dec(v_a_282_);
lean_dec_ref(v_a_281_);
return v_res_288_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__0(lean_object* v___y_289_, lean_object* v___y_290_, lean_object* v___y_291_, lean_object* v___y_292_, lean_object* v___y_293_, lean_object* v___y_294_){
_start:
{
lean_object* v___x_296_; 
v___x_296_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__0___redArg(v___y_289_);
return v___x_296_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__0___boxed(lean_object* v___y_297_, lean_object* v___y_298_, lean_object* v___y_299_, lean_object* v___y_300_, lean_object* v___y_301_, lean_object* v___y_302_, lean_object* v___y_303_){
_start:
{
lean_object* v_res_304_; 
v_res_304_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_getExpr___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__0(v___y_297_, v___y_298_, v___y_299_, v___y_300_, v___y_301_, v___y_302_);
lean_dec(v___y_302_);
lean_dec_ref(v___y_301_);
lean_dec(v___y_300_);
lean_dec_ref(v___y_299_);
lean_dec(v___y_298_);
lean_dec_ref(v___y_297_);
return v_res_304_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__1(lean_object* v_00_u03b1_305_, lean_object* v_child_306_, lean_object* v_childIdx_307_, lean_object* v_x_308_, lean_object* v___y_309_, lean_object* v___y_310_, lean_object* v___y_311_, lean_object* v___y_312_, lean_object* v___y_313_, lean_object* v___y_314_){
_start:
{
lean_object* v___x_316_; 
v___x_316_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__1___redArg(v_child_306_, v_childIdx_307_, v_x_308_, v___y_309_, v___y_310_, v___y_311_, v___y_312_, v___y_313_, v___y_314_);
return v___x_316_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__1___boxed(lean_object* v_00_u03b1_317_, lean_object* v_child_318_, lean_object* v_childIdx_319_, lean_object* v_x_320_, lean_object* v___y_321_, lean_object* v___y_322_, lean_object* v___y_323_, lean_object* v___y_324_, lean_object* v___y_325_, lean_object* v___y_326_, lean_object* v___y_327_){
_start:
{
lean_object* v_res_328_; 
v_res_328_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_descend___at___00Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0_spec__1(v_00_u03b1_317_, v_child_318_, v_childIdx_319_, v_x_320_, v___y_321_, v___y_322_, v___y_323_, v___y_324_, v___y_325_, v___y_326_);
lean_dec(v___y_326_);
lean_dec_ref(v___y_325_);
lean_dec(v___y_324_);
lean_dec_ref(v___y_323_);
lean_dec(v___y_322_);
lean_dec_ref(v___y_321_);
return v_res_328_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0(lean_object* v_00_u03b1_329_, lean_object* v_x_330_, lean_object* v___y_331_, lean_object* v___y_332_, lean_object* v___y_333_, lean_object* v___y_334_, lean_object* v___y_335_, lean_object* v___y_336_){
_start:
{
lean_object* v___x_338_; 
v___x_338_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0___redArg(v_x_330_, v___y_331_, v___y_332_, v___y_333_, v___y_334_, v___y_335_, v___y_336_);
return v___x_338_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0___boxed(lean_object* v_00_u03b1_339_, lean_object* v_x_340_, lean_object* v___y_341_, lean_object* v___y_342_, lean_object* v___y_343_, lean_object* v___y_344_, lean_object* v___y_345_, lean_object* v___y_346_, lean_object* v___y_347_){
_start:
{
lean_object* v_res_348_; 
v_res_348_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0(v_00_u03b1_339_, v_x_340_, v___y_341_, v___y_342_, v___y_343_, v___y_344_, v___y_345_, v___y_346_);
lean_dec(v___y_346_);
lean_dec_ref(v___y_345_);
lean_dec(v___y_344_);
lean_dec_ref(v___y_343_);
lean_dec(v___y_342_);
lean_dec_ref(v___y_341_);
return v_res_348_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdSnd___lam__0(lean_object* v___x_350_, lean_object* v___y_351_, lean_object* v___y_352_, lean_object* v___y_353_, lean_object* v___y_354_, lean_object* v___y_355_, lean_object* v___y_356_){
_start:
{
lean_object* v___x_358_; 
v___x_358_ = lp_mathlib_Lean_PrettyPrinter_Delaborator_SubExpr_withAppArg___at___00Prod_PrettyPrinting_delabProdFst_spec__0___redArg(v___x_350_, v___y_351_, v___y_352_, v___y_353_, v___y_354_, v___y_355_, v___y_356_);
if (lean_obj_tag(v___x_358_) == 0)
{
lean_object* v_a_359_; lean_object* v___x_361_; uint8_t v_isShared_362_; uint8_t v_isSharedCheck_377_; 
v_a_359_ = lean_ctor_get(v___x_358_, 0);
v_isSharedCheck_377_ = !lean_is_exclusive(v___x_358_);
if (v_isSharedCheck_377_ == 0)
{
v___x_361_ = v___x_358_;
v_isShared_362_ = v_isSharedCheck_377_;
goto v_resetjp_360_;
}
else
{
lean_inc(v_a_359_);
lean_dec(v___x_358_);
v___x_361_ = lean_box(0);
v_isShared_362_ = v_isSharedCheck_377_;
goto v_resetjp_360_;
}
v_resetjp_360_:
{
lean_object* v_ref_363_; uint8_t v___x_364_; lean_object* v___x_365_; lean_object* v___x_366_; lean_object* v___x_367_; lean_object* v___x_368_; lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_375_; 
v_ref_363_ = lean_ctor_get(v___y_355_, 5);
v___x_364_ = 0;
v___x_365_ = l_Lean_SourceInfo_fromRef(v_ref_363_, v___x_364_);
v___x_366_ = ((lean_object*)(lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__4));
v___x_367_ = ((lean_object*)(lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__5));
lean_inc_n(v___x_365_, 3);
v___x_368_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_368_, 0, v___x_365_);
lean_ctor_set(v___x_368_, 1, v___x_367_);
v___x_369_ = ((lean_object*)(lp_mathlib_Prod_PrettyPrinting_delabProdFst___lam__0___closed__7));
v___x_370_ = ((lean_object*)(lp_mathlib_Prod_PrettyPrinting_delabProdSnd___lam__0___closed__0));
v___x_371_ = lean_alloc_ctor(2, 2, 0);
lean_ctor_set(v___x_371_, 0, v___x_365_);
lean_ctor_set(v___x_371_, 1, v___x_370_);
v___x_372_ = l_Lean_Syntax_node1(v___x_365_, v___x_369_, v___x_371_);
v___x_373_ = l_Lean_Syntax_node3(v___x_365_, v___x_366_, v_a_359_, v___x_368_, v___x_372_);
if (v_isShared_362_ == 0)
{
lean_ctor_set(v___x_361_, 0, v___x_373_);
v___x_375_ = v___x_361_;
goto v_reusejp_374_;
}
else
{
lean_object* v_reuseFailAlloc_376_; 
v_reuseFailAlloc_376_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_376_, 0, v___x_373_);
v___x_375_ = v_reuseFailAlloc_376_;
goto v_reusejp_374_;
}
v_reusejp_374_:
{
return v___x_375_;
}
}
}
else
{
return v___x_358_;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdSnd___lam__0___boxed(lean_object* v___x_378_, lean_object* v___y_379_, lean_object* v___y_380_, lean_object* v___y_381_, lean_object* v___y_382_, lean_object* v___y_383_, lean_object* v___y_384_, lean_object* v___y_385_){
_start:
{
lean_object* v_res_386_; 
v_res_386_ = lp_mathlib_Prod_PrettyPrinting_delabProdSnd___lam__0(v___x_378_, v___y_379_, v___y_380_, v___y_381_, v___y_382_, v___y_383_, v___y_384_);
lean_dec(v___y_384_);
lean_dec_ref(v___y_383_);
lean_dec(v___y_382_);
lean_dec_ref(v___y_381_);
lean_dec(v___y_380_);
lean_dec_ref(v___y_379_);
return v_res_386_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdSnd(lean_object* v_a_398_, lean_object* v_a_399_, lean_object* v_a_400_, lean_object* v_a_401_, lean_object* v_a_402_, lean_object* v_a_403_){
_start:
{
lean_object* v___x_405_; lean_object* v___x_406_; lean_object* v___x_407_; 
v___x_405_ = ((lean_object*)(lp_mathlib_Prod_PrettyPrinting_delabProdFst___closed__0));
v___x_406_ = ((lean_object*)(lp_mathlib_Prod_PrettyPrinting_delabProdSnd___closed__3));
v___x_407_ = l_Lean_PrettyPrinter_Delaborator_whenPPOption(v___x_405_, v___x_406_, v_a_398_, v_a_399_, v_a_400_, v_a_401_, v_a_402_, v_a_403_);
return v___x_407_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_PrettyPrinting_delabProdSnd___boxed(lean_object* v_a_408_, lean_object* v_a_409_, lean_object* v_a_410_, lean_object* v_a_411_, lean_object* v_a_412_, lean_object* v_a_413_, lean_object* v_a_414_){
_start:
{
lean_object* v_res_415_; 
v_res_415_ = lp_mathlib_Prod_PrettyPrinting_delabProdSnd(v_a_408_, v_a_409_, v_a_410_, v_a_411_, v_a_412_, v_a_413_);
lean_dec(v_a_413_);
lean_dec_ref(v_a_412_);
lean_dec(v_a_411_);
lean_dec_ref(v_a_410_);
lean_dec(v_a_409_);
lean_dec_ref(v_a_408_);
return v_res_415_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Defs(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Inhabit(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_Trans(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Data_Prod_Basic(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Inhabit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_Trans(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_PrettyPrinter_Delaborator_Builtins(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Data_Prod_Basic(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_PrettyPrinter_Delaborator_Builtins(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Data_Prod_Basic_0__Prod_PrettyPrinting_initFn_00___x40_Mathlib_Data_Prod_Basic_1114368346____hygCtx___hyg_4_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_Prod_PrettyPrinting_pp_numericProj_prod = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_Prod_PrettyPrinting_pp_numericProj_prod);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Function_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Function_Iterate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Inhabit(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Trans(uint8_t builtin);
lean_object* initialize_Lean_PrettyPrinter_Delaborator_Builtins(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Prod_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Function_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Inhabit(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Trans(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_PrettyPrinter_Delaborator_Builtins(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Data_Prod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Data_Prod_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Data_Prod_Basic(builtin);
}
#ifdef __cplusplus
}
#endif
