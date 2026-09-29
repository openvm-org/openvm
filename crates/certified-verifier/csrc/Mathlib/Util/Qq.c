// Lean compiler output
// Module: Mathlib.Util.Qq
// Imports: public import Init public meta import Init public import Mathlib.Init public import Qq public import Qq.Typ
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
lean_object* lean_st_ref_get(lean_object*);
lean_object* l_Lean_Name_mkStr4(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_instantiate_level_mvars(lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
lean_object* lean_infer_type(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_getLevel(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Level_dec(lean_object*);
lean_object* l_Lean_stringToMessageData(lean_object*);
lean_object* l_Lean_indentExpr(lean_object*);
lean_object* l_Lean_mkAtom(lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
lean_object* lean_string_utf8_byte_size(lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
lean_object* l_Lean_mkNatLit(lean_object*);
lean_object* l_Lean_mkIntLit(lean_object*);
lean_object* l_Lean_Meta_findLocalDeclWithType_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_fvar___override(lean_object*);
lean_object* l_Lean_Meta_mkDecideProof(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* l_Lean_Level_succ___override(lean_object*);
lean_object* l_Lean_Level_max___override(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_getLevelQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_getLevelQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Qq_getLevelQ_x27_spec__0___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Qq_getLevelQ_x27_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Qq_getLevelQ_x27_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Qq_getLevelQ_x27_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_getLevelQ_x27_spec__1_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_getLevelQ_x27_spec__1_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Qq_getLevelQ_x27_spec__1___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Qq_getLevelQ_x27_spec__1___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Qq_getLevelQ_x27___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 11, .m_capacity = 11, .m_length = 10, .m_data = "not a Type"};
static const lean_object* lp_mathlib_Qq_getLevelQ_x27___closed__0 = (const lean_object*)&lp_mathlib_Qq_getLevelQ_x27___closed__0_value;
static lean_once_cell_t lp_mathlib_Qq_getLevelQ_x27___closed__1_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_getLevelQ_x27___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Qq_getLevelQ_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_getLevelQ_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Qq_getLevelQ_x27_spec__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Qq_getLevelQ_x27_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_inferTypeQ_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_inferTypeQ_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_findLocalDeclWithTypeQ_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_findLocalDeclWithTypeQ_x3f___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_findLocalDeclWithTypeQ_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_findLocalDeclWithTypeQ_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_mkDecideProofQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_mkDecideProofQ___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "Lean"};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__0 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__0_value;
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Parser"};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__1 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__1_value;
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Tactic"};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__2 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__2_value;
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "tacticSeq"};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__3 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__3_value;
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__4_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__4_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__4_value_aux_0),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__4_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__4_value_aux_1),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__4_value_aux_2),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__3_value),LEAN_SCALAR_PTR_LITERAL(212, 140, 85, 215, 241, 69, 7, 118)}};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__4 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__4_value;
static const lean_array_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_array_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 246}, .m_size = 0, .m_capacity = 0, .m_data = {}};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__5 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__5_value;
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "tacticSeq1Indented"};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__6 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__6_value;
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__7_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__7_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__7_value_aux_0),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__7_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__7_value_aux_1),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__7_value_aux_2),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__6_value),LEAN_SCALAR_PTR_LITERAL(223, 90, 160, 238, 133, 180, 23, 239)}};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__7 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__7_value;
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 5, .m_capacity = 5, .m_length = 4, .m_data = "null"};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__8 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__8_value;
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__8_value),LEAN_SCALAR_PTR_LITERAL(24, 58, 49, 223, 146, 207, 197, 136)}};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__9 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__9_value;
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 6, .m_capacity = 6, .m_length = 5, .m_data = "exact"};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__10 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__10_value;
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__0_value),LEAN_SCALAR_PTR_LITERAL(70, 193, 83, 126, 233, 67, 208, 165)}};
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__11_value_aux_1 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__1_value),LEAN_SCALAR_PTR_LITERAL(103, 136, 125, 166, 167, 98, 71, 111)}};
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__11_value_aux_2 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__11_value_aux_1),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__2_value),LEAN_SCALAR_PTR_LITERAL(166, 58, 35, 182, 187, 130, 147, 254)}};
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__11_value_aux_2),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__10_value),LEAN_SCALAR_PTR_LITERAL(108, 106, 111, 83, 219, 207, 32, 208)}};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__11 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__11_value;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__12_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__12;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__13_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__13;
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "Qq"};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__14 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__14_value;
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "termQ(__)"};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__15 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__15_value;
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__16_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__14_value),LEAN_SCALAR_PTR_LITERAL(172, 246, 49, 188, 140, 116, 47, 174)}};
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__16_value_aux_0),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__15_value),LEAN_SCALAR_PTR_LITERAL(101, 189, 163, 187, 112, 4, 232, 151)}};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__16 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__16_value;
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 3, .m_capacity = 3, .m_length = 2, .m_data = "q("};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__17 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__17_value;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__18_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__18;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__19_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__19;
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "inferInstance"};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__20 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__20_value;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__21_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__21;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__22_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__22;
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__20_value),LEAN_SCALAR_PTR_LITERAL(17, 162, 120, 176, 98, 85, 114, 76)}};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__23 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__23_value;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__24_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__24;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__25_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__25;
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*3 + 0, .m_other = 3, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(2) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__9_value),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__5_value)}};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__26 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__26_value;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__27_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__27;
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 2, .m_capacity = 2, .m_length = 1, .m_data = ")"};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__28 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__28_value;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__29_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__29;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__30_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__30;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__31_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__31;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__32_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__32;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__33_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__33;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__34_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__34;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__35_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__35;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__36_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__36;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__37_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__37;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__38_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__38;
static lean_once_cell_t lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__39_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__39;
LEAN_EXPORT lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__1;
LEAN_EXPORT lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__3;
LEAN_EXPORT lean_object* lp_mathlib_Qq_mkSetLiteralQ___auto__5;
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "EmptyCollection"};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___closed__0 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__0_value;
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 16, .m_capacity = 16, .m_length = 15, .m_data = "emptyCollection"};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___closed__1 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__1_value;
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___closed__2_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__0_value),LEAN_SCALAR_PTR_LITERAL(236, 209, 69, 209, 212, 29, 83, 196)}};
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__2_value_aux_0),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__1_value),LEAN_SCALAR_PTR_LITERAL(3, 53, 136, 5, 91, 228, 156, 207)}};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___closed__2 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__2_value;
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "Singleton"};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___closed__3 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__3_value;
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 10, .m_capacity = 10, .m_length = 9, .m_data = "singleton"};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___closed__4 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__4_value;
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___closed__5_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__3_value),LEAN_SCALAR_PTR_LITERAL(190, 73, 36, 155, 228, 35, 161, 122)}};
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__5_value_aux_0),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__4_value),LEAN_SCALAR_PTR_LITERAL(185, 48, 115, 60, 21, 14, 217, 215)}};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___closed__5 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__5_value;
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__0_value),LEAN_SCALAR_PTR_LITERAL(236, 209, 69, 209, 212, 29, 83, 196)}};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___closed__6 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__6_value;
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__3_value),LEAN_SCALAR_PTR_LITERAL(190, 73, 36, 155, 228, 35, 161, 122)}};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___closed__7 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__7_value;
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "Insert"};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___closed__8 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__8_value;
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__8_value),LEAN_SCALAR_PTR_LITERAL(126, 209, 156, 174, 188, 62, 109, 85)}};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___closed__9 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__9_value;
static const lean_string_object lp_mathlib_Qq_mkSetLiteralQ___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "insert"};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___closed__10 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__10_value;
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___closed__11_value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__8_value),LEAN_SCALAR_PTR_LITERAL(126, 209, 156, 174, 188, 62, 109, 85)}};
static const lean_ctor_object lp_mathlib_Qq_mkSetLiteralQ___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__11_value_aux_0),((lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__10_value),LEAN_SCALAR_PTR_LITERAL(12, 132, 219, 243, 180, 219, 203, 85)}};
static const lean_object* lp_mathlib_Qq_mkSetLiteralQ___closed__11 = (const lean_object*)&lp_mathlib_Qq_mkSetLiteralQ___closed__11_value;
LEAN_EXPORT lean_object* lp_mathlib_Qq_mkSetLiteralQ(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_mkNatLitQ(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_mkIntLitQ(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_mkIntLitQ___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Qq_getLevelQ(lean_object* v_e_1_, lean_object* v_a_2_, lean_object* v_a_3_, lean_object* v_a_4_, lean_object* v_a_5_){
_start:
{
lean_object* v___x_7_; 
lean_inc_ref(v_e_1_);
v___x_7_ = l_Lean_Meta_getLevel(v_e_1_, v_a_2_, v_a_3_, v_a_4_, v_a_5_);
if (lean_obj_tag(v___x_7_) == 0)
{
lean_object* v_a_8_; lean_object* v___x_10_; uint8_t v_isShared_11_; uint8_t v_isSharedCheck_16_; 
v_a_8_ = lean_ctor_get(v___x_7_, 0);
v_isSharedCheck_16_ = !lean_is_exclusive(v___x_7_);
if (v_isSharedCheck_16_ == 0)
{
v___x_10_ = v___x_7_;
v_isShared_11_ = v_isSharedCheck_16_;
goto v_resetjp_9_;
}
else
{
lean_inc(v_a_8_);
lean_dec(v___x_7_);
v___x_10_ = lean_box(0);
v_isShared_11_ = v_isSharedCheck_16_;
goto v_resetjp_9_;
}
v_resetjp_9_:
{
lean_object* v___x_12_; lean_object* v___x_14_; 
v___x_12_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_12_, 0, v_a_8_);
lean_ctor_set(v___x_12_, 1, v_e_1_);
if (v_isShared_11_ == 0)
{
lean_ctor_set(v___x_10_, 0, v___x_12_);
v___x_14_ = v___x_10_;
goto v_reusejp_13_;
}
else
{
lean_object* v_reuseFailAlloc_15_; 
v_reuseFailAlloc_15_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_15_, 0, v___x_12_);
v___x_14_ = v_reuseFailAlloc_15_;
goto v_reusejp_13_;
}
v_reusejp_13_:
{
return v___x_14_;
}
}
}
else
{
lean_object* v_a_17_; lean_object* v___x_19_; uint8_t v_isShared_20_; uint8_t v_isSharedCheck_24_; 
lean_dec_ref(v_e_1_);
v_a_17_ = lean_ctor_get(v___x_7_, 0);
v_isSharedCheck_24_ = !lean_is_exclusive(v___x_7_);
if (v_isSharedCheck_24_ == 0)
{
v___x_19_ = v___x_7_;
v_isShared_20_ = v_isSharedCheck_24_;
goto v_resetjp_18_;
}
else
{
lean_inc(v_a_17_);
lean_dec(v___x_7_);
v___x_19_ = lean_box(0);
v_isShared_20_ = v_isSharedCheck_24_;
goto v_resetjp_18_;
}
v_resetjp_18_:
{
lean_object* v___x_22_; 
if (v_isShared_20_ == 0)
{
v___x_22_ = v___x_19_;
goto v_reusejp_21_;
}
else
{
lean_object* v_reuseFailAlloc_23_; 
v_reuseFailAlloc_23_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_23_, 0, v_a_17_);
v___x_22_ = v_reuseFailAlloc_23_;
goto v_reusejp_21_;
}
v_reusejp_21_:
{
return v___x_22_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_getLevelQ___boxed(lean_object* v_e_25_, lean_object* v_a_26_, lean_object* v_a_27_, lean_object* v_a_28_, lean_object* v_a_29_, lean_object* v_a_30_){
_start:
{
lean_object* v_res_31_; 
v_res_31_ = lp_mathlib_Qq_getLevelQ(v_e_25_, v_a_26_, v_a_27_, v_a_28_, v_a_29_);
lean_dec(v_a_29_);
lean_dec_ref(v_a_28_);
lean_dec(v_a_27_);
lean_dec_ref(v_a_26_);
return v_res_31_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Qq_getLevelQ_x27_spec__0___redArg(lean_object* v_l_32_, lean_object* v___y_33_){
_start:
{
lean_object* v___x_35_; lean_object* v_mctx_36_; lean_object* v___x_37_; lean_object* v_fst_38_; lean_object* v_snd_39_; lean_object* v___x_40_; lean_object* v_cache_41_; lean_object* v_zetaDeltaFVarIds_42_; lean_object* v_postponed_43_; lean_object* v_diag_44_; lean_object* v___x_46_; uint8_t v_isShared_47_; uint8_t v_isSharedCheck_53_; 
v___x_35_ = lean_st_ref_get(v___y_33_);
v_mctx_36_ = lean_ctor_get(v___x_35_, 0);
lean_inc_ref(v_mctx_36_);
lean_dec(v___x_35_);
v___x_37_ = lean_instantiate_level_mvars(v_mctx_36_, v_l_32_);
v_fst_38_ = lean_ctor_get(v___x_37_, 0);
lean_inc(v_fst_38_);
v_snd_39_ = lean_ctor_get(v___x_37_, 1);
lean_inc(v_snd_39_);
lean_dec_ref(v___x_37_);
v___x_40_ = lean_st_ref_take(v___y_33_);
v_cache_41_ = lean_ctor_get(v___x_40_, 1);
v_zetaDeltaFVarIds_42_ = lean_ctor_get(v___x_40_, 2);
v_postponed_43_ = lean_ctor_get(v___x_40_, 3);
v_diag_44_ = lean_ctor_get(v___x_40_, 4);
v_isSharedCheck_53_ = !lean_is_exclusive(v___x_40_);
if (v_isSharedCheck_53_ == 0)
{
lean_object* v_unused_54_; 
v_unused_54_ = lean_ctor_get(v___x_40_, 0);
lean_dec(v_unused_54_);
v___x_46_ = v___x_40_;
v_isShared_47_ = v_isSharedCheck_53_;
goto v_resetjp_45_;
}
else
{
lean_inc(v_diag_44_);
lean_inc(v_postponed_43_);
lean_inc(v_zetaDeltaFVarIds_42_);
lean_inc(v_cache_41_);
lean_dec(v___x_40_);
v___x_46_ = lean_box(0);
v_isShared_47_ = v_isSharedCheck_53_;
goto v_resetjp_45_;
}
v_resetjp_45_:
{
lean_object* v___x_49_; 
if (v_isShared_47_ == 0)
{
lean_ctor_set(v___x_46_, 0, v_fst_38_);
v___x_49_ = v___x_46_;
goto v_reusejp_48_;
}
else
{
lean_object* v_reuseFailAlloc_52_; 
v_reuseFailAlloc_52_ = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(v_reuseFailAlloc_52_, 0, v_fst_38_);
lean_ctor_set(v_reuseFailAlloc_52_, 1, v_cache_41_);
lean_ctor_set(v_reuseFailAlloc_52_, 2, v_zetaDeltaFVarIds_42_);
lean_ctor_set(v_reuseFailAlloc_52_, 3, v_postponed_43_);
lean_ctor_set(v_reuseFailAlloc_52_, 4, v_diag_44_);
v___x_49_ = v_reuseFailAlloc_52_;
goto v_reusejp_48_;
}
v_reusejp_48_:
{
lean_object* v___x_50_; lean_object* v___x_51_; 
v___x_50_ = lean_st_ref_set(v___y_33_, v___x_49_);
v___x_51_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_51_, 0, v_snd_39_);
return v___x_51_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Qq_getLevelQ_x27_spec__0___redArg___boxed(lean_object* v_l_55_, lean_object* v___y_56_, lean_object* v___y_57_){
_start:
{
lean_object* v_res_58_; 
v_res_58_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Qq_getLevelQ_x27_spec__0___redArg(v_l_55_, v___y_56_);
lean_dec(v___y_56_);
return v_res_58_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Qq_getLevelQ_x27_spec__0(lean_object* v_l_59_, lean_object* v___y_60_, lean_object* v___y_61_, lean_object* v___y_62_, lean_object* v___y_63_){
_start:
{
lean_object* v___x_65_; 
v___x_65_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Qq_getLevelQ_x27_spec__0___redArg(v_l_59_, v___y_61_);
return v___x_65_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_instantiateLevelMVars___at___00Qq_getLevelQ_x27_spec__0___boxed(lean_object* v_l_66_, lean_object* v___y_67_, lean_object* v___y_68_, lean_object* v___y_69_, lean_object* v___y_70_, lean_object* v___y_71_){
_start:
{
lean_object* v_res_72_; 
v_res_72_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Qq_getLevelQ_x27_spec__0(v_l_66_, v___y_67_, v___y_68_, v___y_69_, v___y_70_);
lean_dec(v___y_70_);
lean_dec_ref(v___y_69_);
lean_dec(v___y_68_);
lean_dec_ref(v___y_67_);
return v_res_72_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_getLevelQ_x27_spec__1_spec__1(lean_object* v_msgData_73_, lean_object* v___y_74_, lean_object* v___y_75_, lean_object* v___y_76_, lean_object* v___y_77_){
_start:
{
lean_object* v___x_79_; lean_object* v_env_80_; lean_object* v___x_81_; lean_object* v_mctx_82_; lean_object* v_lctx_83_; lean_object* v_options_84_; lean_object* v___x_85_; lean_object* v___x_86_; lean_object* v___x_87_; 
v___x_79_ = lean_st_ref_get(v___y_77_);
v_env_80_ = lean_ctor_get(v___x_79_, 0);
lean_inc_ref(v_env_80_);
lean_dec(v___x_79_);
v___x_81_ = lean_st_ref_get(v___y_75_);
v_mctx_82_ = lean_ctor_get(v___x_81_, 0);
lean_inc_ref(v_mctx_82_);
lean_dec(v___x_81_);
v_lctx_83_ = lean_ctor_get(v___y_74_, 2);
v_options_84_ = lean_ctor_get(v___y_76_, 2);
lean_inc_ref(v_options_84_);
lean_inc_ref(v_lctx_83_);
v___x_85_ = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(v___x_85_, 0, v_env_80_);
lean_ctor_set(v___x_85_, 1, v_mctx_82_);
lean_ctor_set(v___x_85_, 2, v_lctx_83_);
lean_ctor_set(v___x_85_, 3, v_options_84_);
v___x_86_ = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(v___x_86_, 0, v___x_85_);
lean_ctor_set(v___x_86_, 1, v_msgData_73_);
v___x_87_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v___x_87_, 0, v___x_86_);
return v___x_87_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_getLevelQ_x27_spec__1_spec__1___boxed(lean_object* v_msgData_88_, lean_object* v___y_89_, lean_object* v___y_90_, lean_object* v___y_91_, lean_object* v___y_92_, lean_object* v___y_93_){
_start:
{
lean_object* v_res_94_; 
v_res_94_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_getLevelQ_x27_spec__1_spec__1(v_msgData_88_, v___y_89_, v___y_90_, v___y_91_, v___y_92_);
lean_dec(v___y_92_);
lean_dec_ref(v___y_91_);
lean_dec(v___y_90_);
lean_dec_ref(v___y_89_);
return v_res_94_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Qq_getLevelQ_x27_spec__1___redArg(lean_object* v_msg_95_, lean_object* v___y_96_, lean_object* v___y_97_, lean_object* v___y_98_, lean_object* v___y_99_){
_start:
{
lean_object* v_ref_101_; lean_object* v___x_102_; lean_object* v_a_103_; lean_object* v___x_105_; uint8_t v_isShared_106_; uint8_t v_isSharedCheck_111_; 
v_ref_101_ = lean_ctor_get(v___y_98_, 5);
v___x_102_ = lp_mathlib_Lean_addMessageContextFull___at___00Lean_throwError___at___00Qq_getLevelQ_x27_spec__1_spec__1(v_msg_95_, v___y_96_, v___y_97_, v___y_98_, v___y_99_);
v_a_103_ = lean_ctor_get(v___x_102_, 0);
v_isSharedCheck_111_ = !lean_is_exclusive(v___x_102_);
if (v_isSharedCheck_111_ == 0)
{
v___x_105_ = v___x_102_;
v_isShared_106_ = v_isSharedCheck_111_;
goto v_resetjp_104_;
}
else
{
lean_inc(v_a_103_);
lean_dec(v___x_102_);
v___x_105_ = lean_box(0);
v_isShared_106_ = v_isSharedCheck_111_;
goto v_resetjp_104_;
}
v_resetjp_104_:
{
lean_object* v___x_107_; lean_object* v___x_109_; 
lean_inc(v_ref_101_);
v___x_107_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_107_, 0, v_ref_101_);
lean_ctor_set(v___x_107_, 1, v_a_103_);
if (v_isShared_106_ == 0)
{
lean_ctor_set_tag(v___x_105_, 1);
lean_ctor_set(v___x_105_, 0, v___x_107_);
v___x_109_ = v___x_105_;
goto v_reusejp_108_;
}
else
{
lean_object* v_reuseFailAlloc_110_; 
v_reuseFailAlloc_110_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_110_, 0, v___x_107_);
v___x_109_ = v_reuseFailAlloc_110_;
goto v_reusejp_108_;
}
v_reusejp_108_:
{
return v___x_109_;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Qq_getLevelQ_x27_spec__1___redArg___boxed(lean_object* v_msg_112_, lean_object* v___y_113_, lean_object* v___y_114_, lean_object* v___y_115_, lean_object* v___y_116_, lean_object* v___y_117_){
_start:
{
lean_object* v_res_118_; 
v_res_118_ = lp_mathlib_Lean_throwError___at___00Qq_getLevelQ_x27_spec__1___redArg(v_msg_112_, v___y_113_, v___y_114_, v___y_115_, v___y_116_);
lean_dec(v___y_116_);
lean_dec_ref(v___y_115_);
lean_dec(v___y_114_);
lean_dec_ref(v___y_113_);
return v_res_118_;
}
}
static lean_object* _init_lp_mathlib_Qq_getLevelQ_x27___closed__1(void){
_start:
{
lean_object* v___x_120_; lean_object* v___x_121_; 
v___x_120_ = ((lean_object*)(lp_mathlib_Qq_getLevelQ_x27___closed__0));
v___x_121_ = l_Lean_stringToMessageData(v___x_120_);
return v___x_121_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_getLevelQ_x27(lean_object* v_e_122_, lean_object* v_a_123_, lean_object* v_a_124_, lean_object* v_a_125_, lean_object* v_a_126_){
_start:
{
lean_object* v___x_128_; 
lean_inc_ref(v_e_122_);
v___x_128_ = l_Lean_Meta_getLevel(v_e_122_, v_a_123_, v_a_124_, v_a_125_, v_a_126_);
if (lean_obj_tag(v___x_128_) == 0)
{
lean_object* v_a_129_; lean_object* v___x_130_; lean_object* v_a_131_; lean_object* v___x_133_; uint8_t v_isShared_134_; uint8_t v_isSharedCheck_145_; 
v_a_129_ = lean_ctor_get(v___x_128_, 0);
lean_inc(v_a_129_);
lean_dec_ref_known(v___x_128_, 1);
v___x_130_ = lp_mathlib_Lean_instantiateLevelMVars___at___00Qq_getLevelQ_x27_spec__0___redArg(v_a_129_, v_a_124_);
v_a_131_ = lean_ctor_get(v___x_130_, 0);
v_isSharedCheck_145_ = !lean_is_exclusive(v___x_130_);
if (v_isSharedCheck_145_ == 0)
{
v___x_133_ = v___x_130_;
v_isShared_134_ = v_isSharedCheck_145_;
goto v_resetjp_132_;
}
else
{
lean_inc(v_a_131_);
lean_dec(v___x_130_);
v___x_133_ = lean_box(0);
v_isShared_134_ = v_isSharedCheck_145_;
goto v_resetjp_132_;
}
v_resetjp_132_:
{
lean_object* v___x_135_; 
v___x_135_ = l_Lean_Level_dec(v_a_131_);
lean_dec(v_a_131_);
if (lean_obj_tag(v___x_135_) == 1)
{
lean_object* v_val_136_; lean_object* v___x_137_; lean_object* v___x_139_; 
v_val_136_ = lean_ctor_get(v___x_135_, 0);
lean_inc(v_val_136_);
lean_dec_ref_known(v___x_135_, 1);
v___x_137_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_137_, 0, v_val_136_);
lean_ctor_set(v___x_137_, 1, v_e_122_);
if (v_isShared_134_ == 0)
{
lean_ctor_set(v___x_133_, 0, v___x_137_);
v___x_139_ = v___x_133_;
goto v_reusejp_138_;
}
else
{
lean_object* v_reuseFailAlloc_140_; 
v_reuseFailAlloc_140_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_140_, 0, v___x_137_);
v___x_139_ = v_reuseFailAlloc_140_;
goto v_reusejp_138_;
}
v_reusejp_138_:
{
return v___x_139_;
}
}
else
{
lean_object* v___x_141_; lean_object* v___x_142_; lean_object* v___x_143_; lean_object* v___x_144_; 
lean_dec(v___x_135_);
lean_del_object(v___x_133_);
v___x_141_ = lean_obj_once(&lp_mathlib_Qq_getLevelQ_x27___closed__1, &lp_mathlib_Qq_getLevelQ_x27___closed__1_once, _init_lp_mathlib_Qq_getLevelQ_x27___closed__1);
v___x_142_ = l_Lean_indentExpr(v_e_122_);
v___x_143_ = lean_alloc_ctor(7, 2, 0);
lean_ctor_set(v___x_143_, 0, v___x_141_);
lean_ctor_set(v___x_143_, 1, v___x_142_);
v___x_144_ = lp_mathlib_Lean_throwError___at___00Qq_getLevelQ_x27_spec__1___redArg(v___x_143_, v_a_123_, v_a_124_, v_a_125_, v_a_126_);
return v___x_144_;
}
}
}
else
{
lean_object* v_a_146_; lean_object* v___x_148_; uint8_t v_isShared_149_; uint8_t v_isSharedCheck_153_; 
lean_dec_ref(v_e_122_);
v_a_146_ = lean_ctor_get(v___x_128_, 0);
v_isSharedCheck_153_ = !lean_is_exclusive(v___x_128_);
if (v_isSharedCheck_153_ == 0)
{
v___x_148_ = v___x_128_;
v_isShared_149_ = v_isSharedCheck_153_;
goto v_resetjp_147_;
}
else
{
lean_inc(v_a_146_);
lean_dec(v___x_128_);
v___x_148_ = lean_box(0);
v_isShared_149_ = v_isSharedCheck_153_;
goto v_resetjp_147_;
}
v_resetjp_147_:
{
lean_object* v___x_151_; 
if (v_isShared_149_ == 0)
{
v___x_151_ = v___x_148_;
goto v_reusejp_150_;
}
else
{
lean_object* v_reuseFailAlloc_152_; 
v_reuseFailAlloc_152_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_152_, 0, v_a_146_);
v___x_151_ = v_reuseFailAlloc_152_;
goto v_reusejp_150_;
}
v_reusejp_150_:
{
return v___x_151_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_getLevelQ_x27___boxed(lean_object* v_e_154_, lean_object* v_a_155_, lean_object* v_a_156_, lean_object* v_a_157_, lean_object* v_a_158_, lean_object* v_a_159_){
_start:
{
lean_object* v_res_160_; 
v_res_160_ = lp_mathlib_Qq_getLevelQ_x27(v_e_154_, v_a_155_, v_a_156_, v_a_157_, v_a_158_);
lean_dec(v_a_158_);
lean_dec_ref(v_a_157_);
lean_dec(v_a_156_);
lean_dec_ref(v_a_155_);
return v_res_160_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Qq_getLevelQ_x27_spec__1(lean_object* v_00_u03b1_161_, lean_object* v_msg_162_, lean_object* v___y_163_, lean_object* v___y_164_, lean_object* v___y_165_, lean_object* v___y_166_){
_start:
{
lean_object* v___x_168_; 
v___x_168_ = lp_mathlib_Lean_throwError___at___00Qq_getLevelQ_x27_spec__1___redArg(v_msg_162_, v___y_163_, v___y_164_, v___y_165_, v___y_166_);
return v___x_168_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_throwError___at___00Qq_getLevelQ_x27_spec__1___boxed(lean_object* v_00_u03b1_169_, lean_object* v_msg_170_, lean_object* v___y_171_, lean_object* v___y_172_, lean_object* v___y_173_, lean_object* v___y_174_, lean_object* v___y_175_){
_start:
{
lean_object* v_res_176_; 
v_res_176_ = lp_mathlib_Lean_throwError___at___00Qq_getLevelQ_x27_spec__1(v_00_u03b1_169_, v_msg_170_, v___y_171_, v___y_172_, v___y_173_, v___y_174_);
lean_dec(v___y_174_);
lean_dec_ref(v___y_173_);
lean_dec(v___y_172_);
lean_dec_ref(v___y_171_);
return v_res_176_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_inferTypeQ_x27(lean_object* v_e_177_, lean_object* v_a_178_, lean_object* v_a_179_, lean_object* v_a_180_, lean_object* v_a_181_){
_start:
{
lean_object* v___x_183_; 
lean_inc(v_a_181_);
lean_inc_ref(v_a_180_);
lean_inc(v_a_179_);
lean_inc_ref(v_a_178_);
lean_inc_ref(v_e_177_);
v___x_183_ = lean_infer_type(v_e_177_, v_a_178_, v_a_179_, v_a_180_, v_a_181_);
if (lean_obj_tag(v___x_183_) == 0)
{
lean_object* v_a_184_; lean_object* v___x_185_; 
v_a_184_ = lean_ctor_get(v___x_183_, 0);
lean_inc(v_a_184_);
lean_dec_ref_known(v___x_183_, 1);
v___x_185_ = lp_mathlib_Qq_getLevelQ_x27(v_a_184_, v_a_178_, v_a_179_, v_a_180_, v_a_181_);
if (lean_obj_tag(v___x_185_) == 0)
{
lean_object* v_a_186_; lean_object* v___x_188_; uint8_t v_isShared_189_; uint8_t v_isSharedCheck_203_; 
v_a_186_ = lean_ctor_get(v___x_185_, 0);
v_isSharedCheck_203_ = !lean_is_exclusive(v___x_185_);
if (v_isSharedCheck_203_ == 0)
{
v___x_188_ = v___x_185_;
v_isShared_189_ = v_isSharedCheck_203_;
goto v_resetjp_187_;
}
else
{
lean_inc(v_a_186_);
lean_dec(v___x_185_);
v___x_188_ = lean_box(0);
v_isShared_189_ = v_isSharedCheck_203_;
goto v_resetjp_187_;
}
v_resetjp_187_:
{
lean_object* v_fst_190_; lean_object* v_snd_191_; lean_object* v___x_193_; uint8_t v_isShared_194_; uint8_t v_isSharedCheck_202_; 
v_fst_190_ = lean_ctor_get(v_a_186_, 0);
v_snd_191_ = lean_ctor_get(v_a_186_, 1);
v_isSharedCheck_202_ = !lean_is_exclusive(v_a_186_);
if (v_isSharedCheck_202_ == 0)
{
v___x_193_ = v_a_186_;
v_isShared_194_ = v_isSharedCheck_202_;
goto v_resetjp_192_;
}
else
{
lean_inc(v_snd_191_);
lean_inc(v_fst_190_);
lean_dec(v_a_186_);
v___x_193_ = lean_box(0);
v_isShared_194_ = v_isSharedCheck_202_;
goto v_resetjp_192_;
}
v_resetjp_192_:
{
lean_object* v___x_196_; 
if (v_isShared_194_ == 0)
{
lean_ctor_set(v___x_193_, 1, v_e_177_);
lean_ctor_set(v___x_193_, 0, v_snd_191_);
v___x_196_ = v___x_193_;
goto v_reusejp_195_;
}
else
{
lean_object* v_reuseFailAlloc_201_; 
v_reuseFailAlloc_201_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v_reuseFailAlloc_201_, 0, v_snd_191_);
lean_ctor_set(v_reuseFailAlloc_201_, 1, v_e_177_);
v___x_196_ = v_reuseFailAlloc_201_;
goto v_reusejp_195_;
}
v_reusejp_195_:
{
lean_object* v___x_197_; lean_object* v___x_199_; 
v___x_197_ = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(v___x_197_, 0, v_fst_190_);
lean_ctor_set(v___x_197_, 1, v___x_196_);
if (v_isShared_189_ == 0)
{
lean_ctor_set(v___x_188_, 0, v___x_197_);
v___x_199_ = v___x_188_;
goto v_reusejp_198_;
}
else
{
lean_object* v_reuseFailAlloc_200_; 
v_reuseFailAlloc_200_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_200_, 0, v___x_197_);
v___x_199_ = v_reuseFailAlloc_200_;
goto v_reusejp_198_;
}
v_reusejp_198_:
{
return v___x_199_;
}
}
}
}
}
else
{
lean_object* v_a_204_; lean_object* v___x_206_; uint8_t v_isShared_207_; uint8_t v_isSharedCheck_211_; 
lean_dec_ref(v_e_177_);
v_a_204_ = lean_ctor_get(v___x_185_, 0);
v_isSharedCheck_211_ = !lean_is_exclusive(v___x_185_);
if (v_isSharedCheck_211_ == 0)
{
v___x_206_ = v___x_185_;
v_isShared_207_ = v_isSharedCheck_211_;
goto v_resetjp_205_;
}
else
{
lean_inc(v_a_204_);
lean_dec(v___x_185_);
v___x_206_ = lean_box(0);
v_isShared_207_ = v_isSharedCheck_211_;
goto v_resetjp_205_;
}
v_resetjp_205_:
{
lean_object* v___x_209_; 
if (v_isShared_207_ == 0)
{
v___x_209_ = v___x_206_;
goto v_reusejp_208_;
}
else
{
lean_object* v_reuseFailAlloc_210_; 
v_reuseFailAlloc_210_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_210_, 0, v_a_204_);
v___x_209_ = v_reuseFailAlloc_210_;
goto v_reusejp_208_;
}
v_reusejp_208_:
{
return v___x_209_;
}
}
}
}
else
{
lean_object* v_a_212_; lean_object* v___x_214_; uint8_t v_isShared_215_; uint8_t v_isSharedCheck_219_; 
lean_dec_ref(v_e_177_);
v_a_212_ = lean_ctor_get(v___x_183_, 0);
v_isSharedCheck_219_ = !lean_is_exclusive(v___x_183_);
if (v_isSharedCheck_219_ == 0)
{
v___x_214_ = v___x_183_;
v_isShared_215_ = v_isSharedCheck_219_;
goto v_resetjp_213_;
}
else
{
lean_inc(v_a_212_);
lean_dec(v___x_183_);
v___x_214_ = lean_box(0);
v_isShared_215_ = v_isSharedCheck_219_;
goto v_resetjp_213_;
}
v_resetjp_213_:
{
lean_object* v___x_217_; 
if (v_isShared_215_ == 0)
{
v___x_217_ = v___x_214_;
goto v_reusejp_216_;
}
else
{
lean_object* v_reuseFailAlloc_218_; 
v_reuseFailAlloc_218_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_218_, 0, v_a_212_);
v___x_217_ = v_reuseFailAlloc_218_;
goto v_reusejp_216_;
}
v_reusejp_216_:
{
return v___x_217_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_inferTypeQ_x27___boxed(lean_object* v_e_220_, lean_object* v_a_221_, lean_object* v_a_222_, lean_object* v_a_223_, lean_object* v_a_224_, lean_object* v_a_225_){
_start:
{
lean_object* v_res_226_; 
v_res_226_ = lp_mathlib_Qq_inferTypeQ_x27(v_e_220_, v_a_221_, v_a_222_, v_a_223_, v_a_224_);
lean_dec(v_a_224_);
lean_dec_ref(v_a_223_);
lean_dec(v_a_222_);
lean_dec_ref(v_a_221_);
return v_res_226_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_findLocalDeclWithTypeQ_x3f___redArg(lean_object* v_sort_227_, lean_object* v_a_228_, lean_object* v_a_229_, lean_object* v_a_230_, lean_object* v_a_231_){
_start:
{
lean_object* v___x_233_; 
v___x_233_ = l_Lean_Meta_findLocalDeclWithType_x3f(v_sort_227_, v_a_228_, v_a_229_, v_a_230_, v_a_231_);
if (lean_obj_tag(v___x_233_) == 0)
{
lean_object* v_a_234_; lean_object* v___x_236_; uint8_t v_isShared_237_; uint8_t v_isSharedCheck_254_; 
v_a_234_ = lean_ctor_get(v___x_233_, 0);
v_isSharedCheck_254_ = !lean_is_exclusive(v___x_233_);
if (v_isSharedCheck_254_ == 0)
{
v___x_236_ = v___x_233_;
v_isShared_237_ = v_isSharedCheck_254_;
goto v_resetjp_235_;
}
else
{
lean_inc(v_a_234_);
lean_dec(v___x_233_);
v___x_236_ = lean_box(0);
v_isShared_237_ = v_isSharedCheck_254_;
goto v_resetjp_235_;
}
v_resetjp_235_:
{
if (lean_obj_tag(v_a_234_) == 1)
{
lean_object* v_val_238_; lean_object* v___x_240_; uint8_t v_isShared_241_; uint8_t v_isSharedCheck_249_; 
v_val_238_ = lean_ctor_get(v_a_234_, 0);
v_isSharedCheck_249_ = !lean_is_exclusive(v_a_234_);
if (v_isSharedCheck_249_ == 0)
{
v___x_240_ = v_a_234_;
v_isShared_241_ = v_isSharedCheck_249_;
goto v_resetjp_239_;
}
else
{
lean_inc(v_val_238_);
lean_dec(v_a_234_);
v___x_240_ = lean_box(0);
v_isShared_241_ = v_isSharedCheck_249_;
goto v_resetjp_239_;
}
v_resetjp_239_:
{
lean_object* v___x_242_; lean_object* v___x_244_; 
v___x_242_ = l_Lean_Expr_fvar___override(v_val_238_);
if (v_isShared_241_ == 0)
{
lean_ctor_set(v___x_240_, 0, v___x_242_);
v___x_244_ = v___x_240_;
goto v_reusejp_243_;
}
else
{
lean_object* v_reuseFailAlloc_248_; 
v_reuseFailAlloc_248_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_248_, 0, v___x_242_);
v___x_244_ = v_reuseFailAlloc_248_;
goto v_reusejp_243_;
}
v_reusejp_243_:
{
lean_object* v___x_246_; 
if (v_isShared_237_ == 0)
{
lean_ctor_set(v___x_236_, 0, v___x_244_);
v___x_246_ = v___x_236_;
goto v_reusejp_245_;
}
else
{
lean_object* v_reuseFailAlloc_247_; 
v_reuseFailAlloc_247_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_247_, 0, v___x_244_);
v___x_246_ = v_reuseFailAlloc_247_;
goto v_reusejp_245_;
}
v_reusejp_245_:
{
return v___x_246_;
}
}
}
}
else
{
lean_object* v___x_250_; lean_object* v___x_252_; 
lean_dec(v_a_234_);
v___x_250_ = lean_box(0);
if (v_isShared_237_ == 0)
{
lean_ctor_set(v___x_236_, 0, v___x_250_);
v___x_252_ = v___x_236_;
goto v_reusejp_251_;
}
else
{
lean_object* v_reuseFailAlloc_253_; 
v_reuseFailAlloc_253_ = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(v_reuseFailAlloc_253_, 0, v___x_250_);
v___x_252_ = v_reuseFailAlloc_253_;
goto v_reusejp_251_;
}
v_reusejp_251_:
{
return v___x_252_;
}
}
}
}
else
{
lean_object* v_a_255_; lean_object* v___x_257_; uint8_t v_isShared_258_; uint8_t v_isSharedCheck_262_; 
v_a_255_ = lean_ctor_get(v___x_233_, 0);
v_isSharedCheck_262_ = !lean_is_exclusive(v___x_233_);
if (v_isSharedCheck_262_ == 0)
{
v___x_257_ = v___x_233_;
v_isShared_258_ = v_isSharedCheck_262_;
goto v_resetjp_256_;
}
else
{
lean_inc(v_a_255_);
lean_dec(v___x_233_);
v___x_257_ = lean_box(0);
v_isShared_258_ = v_isSharedCheck_262_;
goto v_resetjp_256_;
}
v_resetjp_256_:
{
lean_object* v___x_260_; 
if (v_isShared_258_ == 0)
{
v___x_260_ = v___x_257_;
goto v_reusejp_259_;
}
else
{
lean_object* v_reuseFailAlloc_261_; 
v_reuseFailAlloc_261_ = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(v_reuseFailAlloc_261_, 0, v_a_255_);
v___x_260_ = v_reuseFailAlloc_261_;
goto v_reusejp_259_;
}
v_reusejp_259_:
{
return v___x_260_;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_findLocalDeclWithTypeQ_x3f___redArg___boxed(lean_object* v_sort_263_, lean_object* v_a_264_, lean_object* v_a_265_, lean_object* v_a_266_, lean_object* v_a_267_, lean_object* v_a_268_){
_start:
{
lean_object* v_res_269_; 
v_res_269_ = lp_mathlib_Qq_findLocalDeclWithTypeQ_x3f___redArg(v_sort_263_, v_a_264_, v_a_265_, v_a_266_, v_a_267_);
lean_dec(v_a_267_);
lean_dec_ref(v_a_266_);
lean_dec(v_a_265_);
lean_dec_ref(v_a_264_);
return v_res_269_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_findLocalDeclWithTypeQ_x3f(lean_object* v_u_270_, lean_object* v_sort_271_, lean_object* v_a_272_, lean_object* v_a_273_, lean_object* v_a_274_, lean_object* v_a_275_){
_start:
{
lean_object* v___x_277_; 
v___x_277_ = lp_mathlib_Qq_findLocalDeclWithTypeQ_x3f___redArg(v_sort_271_, v_a_272_, v_a_273_, v_a_274_, v_a_275_);
return v___x_277_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_findLocalDeclWithTypeQ_x3f___boxed(lean_object* v_u_278_, lean_object* v_sort_279_, lean_object* v_a_280_, lean_object* v_a_281_, lean_object* v_a_282_, lean_object* v_a_283_, lean_object* v_a_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_mathlib_Qq_findLocalDeclWithTypeQ_x3f(v_u_278_, v_sort_279_, v_a_280_, v_a_281_, v_a_282_, v_a_283_);
lean_dec(v_a_283_);
lean_dec_ref(v_a_282_);
lean_dec(v_a_281_);
lean_dec_ref(v_a_280_);
lean_dec(v_u_278_);
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_mkDecideProofQ(lean_object* v_p_286_, lean_object* v_a_287_, lean_object* v_a_288_, lean_object* v_a_289_, lean_object* v_a_290_){
_start:
{
lean_object* v___x_292_; 
v___x_292_ = l_Lean_Meta_mkDecideProof(v_p_286_, v_a_287_, v_a_288_, v_a_289_, v_a_290_);
return v___x_292_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_mkDecideProofQ___boxed(lean_object* v_p_293_, lean_object* v_a_294_, lean_object* v_a_295_, lean_object* v_a_296_, lean_object* v_a_297_, lean_object* v_a_298_){
_start:
{
lean_object* v_res_299_; 
v_res_299_ = lp_mathlib_Qq_mkDecideProofQ(v_p_293_, v_a_294_, v_a_295_, v_a_296_, v_a_297_);
lean_dec(v_a_297_);
lean_dec_ref(v_a_296_);
lean_dec(v_a_295_);
lean_dec_ref(v_a_294_);
return v_res_299_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__12(void){
_start:
{
lean_object* v___x_326_; lean_object* v___x_327_; 
v___x_326_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__10));
v___x_327_ = l_Lean_mkAtom(v___x_326_);
return v___x_327_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__13(void){
_start:
{
lean_object* v___x_328_; lean_object* v___x_329_; lean_object* v___x_330_; 
v___x_328_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__12, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__12_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__12);
v___x_329_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__5));
v___x_330_ = lean_array_push(v___x_329_, v___x_328_);
return v___x_330_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__18(void){
_start:
{
lean_object* v___x_337_; lean_object* v___x_338_; 
v___x_337_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__17));
v___x_338_ = l_Lean_mkAtom(v___x_337_);
return v___x_338_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__19(void){
_start:
{
lean_object* v___x_339_; lean_object* v___x_340_; lean_object* v___x_341_; 
v___x_339_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__18, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__18_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__18);
v___x_340_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__5));
v___x_341_ = lean_array_push(v___x_340_, v___x_339_);
return v___x_341_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__21(void){
_start:
{
lean_object* v___x_343_; lean_object* v___x_344_; 
v___x_343_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__20));
v___x_344_ = lean_string_utf8_byte_size(v___x_343_);
return v___x_344_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__22(void){
_start:
{
lean_object* v___x_345_; lean_object* v___x_346_; lean_object* v___x_347_; lean_object* v___x_348_; 
v___x_345_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__21, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__21_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__21);
v___x_346_ = lean_unsigned_to_nat(0u);
v___x_347_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__20));
v___x_348_ = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(v___x_348_, 0, v___x_347_);
lean_ctor_set(v___x_348_, 1, v___x_346_);
lean_ctor_set(v___x_348_, 2, v___x_345_);
return v___x_348_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__24(void){
_start:
{
lean_object* v___x_351_; lean_object* v___x_352_; lean_object* v___x_353_; lean_object* v___x_354_; lean_object* v___x_355_; 
v___x_351_ = lean_box(0);
v___x_352_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__23));
v___x_353_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__22, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__22_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__22);
v___x_354_ = lean_box(2);
v___x_355_ = lean_alloc_ctor(3, 4, 0);
lean_ctor_set(v___x_355_, 0, v___x_354_);
lean_ctor_set(v___x_355_, 1, v___x_353_);
lean_ctor_set(v___x_355_, 2, v___x_352_);
lean_ctor_set(v___x_355_, 3, v___x_351_);
return v___x_355_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__25(void){
_start:
{
lean_object* v___x_356_; lean_object* v___x_357_; lean_object* v___x_358_; 
v___x_356_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__24, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__24_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__24);
v___x_357_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__19, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__19_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__19);
v___x_358_ = lean_array_push(v___x_357_, v___x_356_);
return v___x_358_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__27(void){
_start:
{
lean_object* v___x_363_; lean_object* v___x_364_; lean_object* v___x_365_; 
v___x_363_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__26));
v___x_364_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__25, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__25_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__25);
v___x_365_ = lean_array_push(v___x_364_, v___x_363_);
return v___x_365_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__29(void){
_start:
{
lean_object* v___x_367_; lean_object* v___x_368_; 
v___x_367_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__28));
v___x_368_ = l_Lean_mkAtom(v___x_367_);
return v___x_368_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__30(void){
_start:
{
lean_object* v___x_369_; lean_object* v___x_370_; lean_object* v___x_371_; 
v___x_369_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__29, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__29_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__29);
v___x_370_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__27, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__27_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__27);
v___x_371_ = lean_array_push(v___x_370_, v___x_369_);
return v___x_371_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__31(void){
_start:
{
lean_object* v___x_372_; lean_object* v___x_373_; lean_object* v___x_374_; lean_object* v___x_375_; 
v___x_372_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__30, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__30_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__30);
v___x_373_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__16));
v___x_374_ = lean_box(2);
v___x_375_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_375_, 0, v___x_374_);
lean_ctor_set(v___x_375_, 1, v___x_373_);
lean_ctor_set(v___x_375_, 2, v___x_372_);
return v___x_375_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__32(void){
_start:
{
lean_object* v___x_376_; lean_object* v___x_377_; lean_object* v___x_378_; 
v___x_376_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__31, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__31_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__31);
v___x_377_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__13, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__13_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__13);
v___x_378_ = lean_array_push(v___x_377_, v___x_376_);
return v___x_378_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__33(void){
_start:
{
lean_object* v___x_379_; lean_object* v___x_380_; lean_object* v___x_381_; lean_object* v___x_382_; 
v___x_379_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__32, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__32_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__32);
v___x_380_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__11));
v___x_381_ = lean_box(2);
v___x_382_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_382_, 0, v___x_381_);
lean_ctor_set(v___x_382_, 1, v___x_380_);
lean_ctor_set(v___x_382_, 2, v___x_379_);
return v___x_382_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__34(void){
_start:
{
lean_object* v___x_383_; lean_object* v___x_384_; lean_object* v___x_385_; 
v___x_383_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__33, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__33_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__33);
v___x_384_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__5));
v___x_385_ = lean_array_push(v___x_384_, v___x_383_);
return v___x_385_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__35(void){
_start:
{
lean_object* v___x_386_; lean_object* v___x_387_; lean_object* v___x_388_; lean_object* v___x_389_; 
v___x_386_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__34, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__34_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__34);
v___x_387_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__9));
v___x_388_ = lean_box(2);
v___x_389_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_389_, 0, v___x_388_);
lean_ctor_set(v___x_389_, 1, v___x_387_);
lean_ctor_set(v___x_389_, 2, v___x_386_);
return v___x_389_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__36(void){
_start:
{
lean_object* v___x_390_; lean_object* v___x_391_; lean_object* v___x_392_; 
v___x_390_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__35, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__35_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__35);
v___x_391_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__5));
v___x_392_ = lean_array_push(v___x_391_, v___x_390_);
return v___x_392_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__37(void){
_start:
{
lean_object* v___x_393_; lean_object* v___x_394_; lean_object* v___x_395_; lean_object* v___x_396_; 
v___x_393_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__36, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__36_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__36);
v___x_394_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__7));
v___x_395_ = lean_box(2);
v___x_396_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_396_, 0, v___x_395_);
lean_ctor_set(v___x_396_, 1, v___x_394_);
lean_ctor_set(v___x_396_, 2, v___x_393_);
return v___x_396_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__38(void){
_start:
{
lean_object* v___x_397_; lean_object* v___x_398_; lean_object* v___x_399_; 
v___x_397_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__37, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__37_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__37);
v___x_398_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__5));
v___x_399_ = lean_array_push(v___x_398_, v___x_397_);
return v___x_399_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__39(void){
_start:
{
lean_object* v___x_400_; lean_object* v___x_401_; lean_object* v___x_402_; lean_object* v___x_403_; 
v___x_400_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__38, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__38_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__38);
v___x_401_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__4));
v___x_402_ = lean_box(2);
v___x_403_ = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(v___x_403_, 0, v___x_402_);
lean_ctor_set(v___x_403_, 1, v___x_401_);
lean_ctor_set(v___x_403_, 2, v___x_400_);
return v___x_403_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1(void){
_start:
{
lean_object* v___x_404_; 
v___x_404_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__39, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__39_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__39);
return v___x_404_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__3(void){
_start:
{
lean_object* v___x_405_; 
v___x_405_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__39, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__39_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__39);
return v___x_405_;
}
}
static lean_object* _init_lp_mathlib_Qq_mkSetLiteralQ___auto__5(void){
_start:
{
lean_object* v___x_406_; 
v___x_406_ = lean_obj_once(&lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__39, &lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__39_once, _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__39);
return v___x_406_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_mkSetLiteralQ(lean_object* v_u_428_, lean_object* v_v_429_, lean_object* v_00_u03b1_430_, lean_object* v_00_u03b2_431_, lean_object* v_elems_432_, lean_object* v_x_433_, lean_object* v_x_434_, lean_object* v_x_435_){
_start:
{
if (lean_obj_tag(v_elems_432_) == 0)
{
lean_object* v___x_436_; lean_object* v___x_437_; lean_object* v___x_438_; lean_object* v___x_439_; lean_object* v___x_440_; lean_object* v___x_441_; 
lean_dec_ref(v_x_435_);
lean_dec_ref(v_x_434_);
lean_dec_ref(v_00_u03b1_430_);
lean_dec(v_u_428_);
v___x_436_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___closed__2));
v___x_437_ = lean_box(0);
v___x_438_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_438_, 0, v_v_429_);
lean_ctor_set(v___x_438_, 1, v___x_437_);
v___x_439_ = l_Lean_Expr_const___override(v___x_436_, v___x_438_);
v___x_440_ = l_Lean_Expr_app___override(v___x_439_, v_00_u03b2_431_);
v___x_441_ = l_Lean_Expr_app___override(v___x_440_, v_x_433_);
return v___x_441_;
}
else
{
lean_object* v_tail_442_; 
v_tail_442_ = lean_ctor_get(v_elems_432_, 1);
if (lean_obj_tag(v_tail_442_) == 0)
{
lean_object* v_head_443_; lean_object* v___x_445_; uint8_t v_isShared_446_; uint8_t v_isSharedCheck_458_; 
lean_dec_ref(v_x_435_);
lean_dec_ref(v_x_433_);
v_head_443_ = lean_ctor_get(v_elems_432_, 0);
v_isSharedCheck_458_ = !lean_is_exclusive(v_elems_432_);
if (v_isSharedCheck_458_ == 0)
{
lean_object* v_unused_459_; 
v_unused_459_ = lean_ctor_get(v_elems_432_, 1);
lean_dec(v_unused_459_);
v___x_445_ = v_elems_432_;
v_isShared_446_ = v_isSharedCheck_458_;
goto v_resetjp_444_;
}
else
{
lean_inc(v_head_443_);
lean_dec(v_elems_432_);
v___x_445_ = lean_box(0);
v_isShared_446_ = v_isSharedCheck_458_;
goto v_resetjp_444_;
}
v_resetjp_444_:
{
lean_object* v___x_447_; lean_object* v___x_448_; lean_object* v___x_450_; 
v___x_447_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___closed__5));
v___x_448_ = lean_box(0);
if (v_isShared_446_ == 0)
{
lean_ctor_set(v___x_445_, 1, v___x_448_);
lean_ctor_set(v___x_445_, 0, v_v_429_);
v___x_450_ = v___x_445_;
goto v_reusejp_449_;
}
else
{
lean_object* v_reuseFailAlloc_457_; 
v_reuseFailAlloc_457_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_457_, 0, v_v_429_);
lean_ctor_set(v_reuseFailAlloc_457_, 1, v___x_448_);
v___x_450_ = v_reuseFailAlloc_457_;
goto v_reusejp_449_;
}
v_reusejp_449_:
{
lean_object* v___x_451_; lean_object* v___x_452_; lean_object* v___x_453_; lean_object* v___x_454_; lean_object* v___x_455_; lean_object* v___x_456_; 
v___x_451_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_451_, 0, v_u_428_);
lean_ctor_set(v___x_451_, 1, v___x_450_);
v___x_452_ = l_Lean_Expr_const___override(v___x_447_, v___x_451_);
v___x_453_ = l_Lean_Expr_app___override(v___x_452_, v_00_u03b1_430_);
v___x_454_ = l_Lean_Expr_app___override(v___x_453_, v_00_u03b2_431_);
v___x_455_ = l_Lean_Expr_app___override(v___x_454_, v_x_434_);
v___x_456_ = l_Lean_Expr_app___override(v___x_455_, v_head_443_);
return v___x_456_;
}
}
}
else
{
lean_object* v_head_460_; lean_object* v___x_462_; uint8_t v_isShared_463_; uint8_t v_isSharedCheck_502_; 
lean_inc(v_tail_442_);
v_head_460_ = lean_ctor_get(v_elems_432_, 0);
v_isSharedCheck_502_ = !lean_is_exclusive(v_elems_432_);
if (v_isSharedCheck_502_ == 0)
{
lean_object* v_unused_503_; 
v_unused_503_ = lean_ctor_get(v_elems_432_, 1);
lean_dec(v_unused_503_);
v___x_462_ = v_elems_432_;
v_isShared_463_ = v_isSharedCheck_502_;
goto v_resetjp_461_;
}
else
{
lean_inc(v_head_460_);
lean_dec(v_elems_432_);
v___x_462_ = lean_box(0);
v_isShared_463_ = v_isSharedCheck_502_;
goto v_resetjp_461_;
}
v_resetjp_461_:
{
lean_object* v___x_464_; lean_object* v___x_465_; lean_object* v___x_467_; 
v___x_464_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___closed__6));
v___x_465_ = lean_box(0);
lean_inc(v_v_429_);
if (v_isShared_463_ == 0)
{
lean_ctor_set(v___x_462_, 1, v___x_465_);
lean_ctor_set(v___x_462_, 0, v_v_429_);
v___x_467_ = v___x_462_;
goto v_reusejp_466_;
}
else
{
lean_object* v_reuseFailAlloc_501_; 
v_reuseFailAlloc_501_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v_reuseFailAlloc_501_, 0, v_v_429_);
lean_ctor_set(v_reuseFailAlloc_501_, 1, v___x_465_);
v___x_467_ = v_reuseFailAlloc_501_;
goto v_reusejp_466_;
}
v_reusejp_466_:
{
lean_object* v___x_468_; lean_object* v___x_469_; lean_object* v___x_470_; lean_object* v___x_471_; lean_object* v___x_472_; lean_object* v___x_473_; lean_object* v___x_474_; lean_object* v___x_475_; lean_object* v___x_476_; lean_object* v___x_477_; lean_object* v___x_478_; lean_object* v___x_479_; lean_object* v___x_480_; lean_object* v___x_481_; lean_object* v___x_482_; lean_object* v___x_483_; lean_object* v___x_484_; lean_object* v___x_485_; lean_object* v___x_486_; lean_object* v___x_487_; lean_object* v___x_488_; lean_object* v___x_489_; lean_object* v___x_490_; lean_object* v___x_491_; lean_object* v___x_492_; lean_object* v_a_493_; lean_object* v___x_494_; lean_object* v___x_495_; lean_object* v___x_496_; lean_object* v___x_497_; lean_object* v___x_498_; lean_object* v___x_499_; lean_object* v___x_500_; 
lean_inc_ref(v___x_467_);
v___x_468_ = l_Lean_Expr_const___override(v___x_464_, v___x_467_);
lean_inc_ref_n(v_00_u03b2_431_, 4);
v___x_469_ = l_Lean_Expr_app___override(v___x_468_, v_00_u03b2_431_);
v___x_470_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___auto__1___closed__23));
lean_inc(v_v_429_);
v___x_471_ = l_Lean_Level_succ___override(v_v_429_);
lean_inc(v___x_471_);
v___x_472_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_472_, 0, v___x_471_);
lean_ctor_set(v___x_472_, 1, v___x_465_);
v___x_473_ = l_Lean_Expr_const___override(v___x_470_, v___x_472_);
v___x_474_ = l_Lean_Expr_app___override(v___x_473_, v___x_469_);
v___x_475_ = l_Lean_Expr_app___override(v___x_474_, v_x_433_);
v___x_476_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___closed__7));
lean_inc_n(v_u_428_, 2);
v___x_477_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_477_, 0, v_u_428_);
lean_ctor_set(v___x_477_, 1, v___x_467_);
lean_inc_ref_n(v___x_477_, 2);
v___x_478_ = l_Lean_Expr_const___override(v___x_476_, v___x_477_);
lean_inc_ref_n(v_00_u03b1_430_, 3);
v___x_479_ = l_Lean_Expr_app___override(v___x_478_, v_00_u03b1_430_);
v___x_480_ = l_Lean_Expr_app___override(v___x_479_, v_00_u03b2_431_);
v___x_481_ = l_Lean_Level_succ___override(v_u_428_);
v___x_482_ = l_Lean_Level_max___override(v___x_481_, v___x_471_);
v___x_483_ = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(v___x_483_, 0, v___x_482_);
lean_ctor_set(v___x_483_, 1, v___x_465_);
v___x_484_ = l_Lean_Expr_const___override(v___x_470_, v___x_483_);
lean_inc_ref(v___x_484_);
v___x_485_ = l_Lean_Expr_app___override(v___x_484_, v___x_480_);
v___x_486_ = l_Lean_Expr_app___override(v___x_485_, v_x_434_);
v___x_487_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___closed__9));
v___x_488_ = l_Lean_Expr_const___override(v___x_487_, v___x_477_);
v___x_489_ = l_Lean_Expr_app___override(v___x_488_, v_00_u03b1_430_);
v___x_490_ = l_Lean_Expr_app___override(v___x_489_, v_00_u03b2_431_);
v___x_491_ = l_Lean_Expr_app___override(v___x_484_, v___x_490_);
lean_inc_ref(v_x_435_);
v___x_492_ = l_Lean_Expr_app___override(v___x_491_, v_x_435_);
v_a_493_ = lp_mathlib_Qq_mkSetLiteralQ(v_u_428_, v_v_429_, v_00_u03b1_430_, v_00_u03b2_431_, v_tail_442_, v___x_475_, v___x_486_, v___x_492_);
v___x_494_ = ((lean_object*)(lp_mathlib_Qq_mkSetLiteralQ___closed__11));
v___x_495_ = l_Lean_Expr_const___override(v___x_494_, v___x_477_);
v___x_496_ = l_Lean_Expr_app___override(v___x_495_, v_00_u03b1_430_);
v___x_497_ = l_Lean_Expr_app___override(v___x_496_, v_00_u03b2_431_);
v___x_498_ = l_Lean_Expr_app___override(v___x_497_, v_x_435_);
v___x_499_ = l_Lean_Expr_app___override(v___x_498_, v_head_460_);
v___x_500_ = l_Lean_Expr_app___override(v___x_499_, v_a_493_);
return v___x_500_;
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_mkNatLitQ(lean_object* v_n_504_){
_start:
{
lean_object* v___x_505_; 
v___x_505_ = l_Lean_mkNatLit(v_n_504_);
return v___x_505_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_mkIntLitQ(lean_object* v_n_506_){
_start:
{
lean_object* v___x_507_; 
v___x_507_ = l_Lean_mkIntLit(v_n_506_);
return v___x_507_;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Qq_mkIntLitQ___boxed(lean_object* v_n_508_){
_start:
{
lean_object* v_res_509_; 
v_res_509_ = lp_mathlib_Qq_mkIntLitQ(v_n_508_);
lean_dec(v_n_508_);
return v_res_509_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq(uint8_t builtin);
lean_object* runtime_initialize_Qq_Qq_Typ(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Util_Qq(uint8_t builtin) {
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
res = runtime_initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Qq_Qq_Typ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Util_Qq(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Qq_mkSetLiteralQ___auto__1 = _init_lp_mathlib_Qq_mkSetLiteralQ___auto__1();
lean_mark_persistent(lp_mathlib_Qq_mkSetLiteralQ___auto__1);
lp_mathlib_Qq_mkSetLiteralQ___auto__3 = _init_lp_mathlib_Qq_mkSetLiteralQ___auto__3();
lean_mark_persistent(lp_mathlib_Qq_mkSetLiteralQ___auto__3);
lp_mathlib_Qq_mkSetLiteralQ___auto__5 = _init_lp_mathlib_Qq_mkSetLiteralQ___auto__5();
lean_mark_persistent(lp_mathlib_Qq_mkSetLiteralQ___auto__5);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Qq_Qq(uint8_t builtin);
lean_object* initialize_Qq_Qq_Typ(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Util_Qq(uint8_t builtin) {
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
res = initialize_Qq_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Qq_Qq_Typ(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Util_Qq(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Util_Qq(builtin);
}
#ifdef __cplusplus
}
#endif
