// Lean compiler output
// Module: Mathlib.Init
// Imports: public import Init public meta import Init public import Lean.Linter.Sets public import Lean.LibrarySuggestions.Default public import Mathlib.Lean.Linter public import Mathlib.Tactic.AdaptationNote public import Mathlib.Tactic.Lemma public import Mathlib.Tactic.Linter.AuxLemma public import Mathlib.Tactic.Linter.DeprecatedSyntaxLinter public import Mathlib.Tactic.Linter.DirectoryDependency public import Mathlib.Tactic.Linter.DocPrime public import Mathlib.Tactic.Linter.DocString public import Mathlib.Tactic.Linter.EmptyLine public import Mathlib.Tactic.Linter.GlobalAttributeIn public import Mathlib.Tactic.Linter.HashCommandLinter public import Mathlib.Tactic.Linter.HaveILetI public import Mathlib.Tactic.Linter.Header public import Mathlib.Tactic.Linter.InternalConstructor public import Mathlib.Tactic.Linter.FlexibleLinter public import Mathlib.Tactic.Linter.Multigoal public import Mathlib.Tactic.Linter.OldObtain public import Mathlib.Tactic.Linter.OverlappingInstances public import Mathlib.Tactic.Linter.PrivateModule public import Mathlib.Tactic.Linter.TacticDocumentation public import Mathlib.Tactic.Linter.UnusedTacticExtension public import Mathlib.Tactic.Linter.UnusedTactic public import Mathlib.Tactic.Linter.UnusedInstancesInType public import Mathlib.Tactic.Linter.Style public import Mathlib.Tactic.Linter.Whitespace public import Mathlib.Tactic.TacticAnalysis.Declarations public import Mathlib.Tactic.TypeStar public import Batteries.Tactic.HelpCmd public import Batteries.Util.ProofWanted public import ImportGraph.Tools public import Mathlib.Tactic.Linter.Lint public import Mathlib.Tactic.MinImports public import Mathlib.Util.CodeActions
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
lean_object* l_Lean_Linter_registerSet(lean_object*, lean_object*);
static const lean_string_object lp_mathlib___private_Mathlib_Init_0__initFn___closed__0_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 7, .m_capacity = 7, .m_length = 6, .m_data = "linter"};
static const lean_object* lp_mathlib___private_Mathlib_Init_0__initFn___closed__0_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3_ = (const lean_object*)&lp_mathlib___private_Mathlib_Init_0__initFn___closed__0_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3__value;
static const lean_string_object lp_mathlib___private_Mathlib_Init_0__initFn___closed__1_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 19, .m_capacity = 19, .m_length = 18, .m_data = "mathlibStandardSet"};
static const lean_object* lp_mathlib___private_Mathlib_Init_0__initFn___closed__1_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3_ = (const lean_object*)&lp_mathlib___private_Mathlib_Init_0__initFn___closed__1_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Init_0__initFn___closed__2_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Init_0__initFn___closed__0_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Init_0__initFn___closed__2_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Init_0__initFn___closed__2_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Init_0__initFn___closed__1_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3__value),LEAN_SCALAR_PTR_LITERAL(157, 32, 44, 82, 232, 108, 144, 52)}};
static const lean_object* lp_mathlib___private_Mathlib_Init_0__initFn___closed__2_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3_ = (const lean_object*)&lp_mathlib___private_Mathlib_Init_0__initFn___closed__2_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linter_mathlibStandardSet;
static const lean_string_object lp_mathlib___private_Mathlib_Init_0__initFn___closed__0_00___x40_Mathlib_Init_2344344261____hygCtx___hyg_3__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 21, .m_capacity = 21, .m_length = 20, .m_data = "nightlyRegressionSet"};
static const lean_object* lp_mathlib___private_Mathlib_Init_0__initFn___closed__0_00___x40_Mathlib_Init_2344344261____hygCtx___hyg_3_ = (const lean_object*)&lp_mathlib___private_Mathlib_Init_0__initFn___closed__0_00___x40_Mathlib_Init_2344344261____hygCtx___hyg_3__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Init_0__initFn___closed__1_00___x40_Mathlib_Init_2344344261____hygCtx___hyg_3__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Init_0__initFn___closed__0_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Init_0__initFn___closed__1_00___x40_Mathlib_Init_2344344261____hygCtx___hyg_3__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Init_0__initFn___closed__1_00___x40_Mathlib_Init_2344344261____hygCtx___hyg_3__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Init_0__initFn___closed__0_00___x40_Mathlib_Init_2344344261____hygCtx___hyg_3__value),LEAN_SCALAR_PTR_LITERAL(36, 26, 63, 205, 21, 0, 90, 36)}};
static const lean_object* lp_mathlib___private_Mathlib_Init_0__initFn___closed__1_00___x40_Mathlib_Init_2344344261____hygCtx___hyg_3_ = (const lean_object*)&lp_mathlib___private_Mathlib_Init_0__initFn___closed__1_00___x40_Mathlib_Init_2344344261____hygCtx___hyg_3__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_2344344261____hygCtx___hyg_3_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_2344344261____hygCtx___hyg_3____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linter_nightlyRegressionSet;
static const lean_string_object lp_mathlib___private_Mathlib_Init_0__initFn___closed__0_00___x40_Mathlib_Init_2984349899____hygCtx___hyg_3__value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 14, .m_capacity = 14, .m_length = 13, .m_data = "weeklyLintSet"};
static const lean_object* lp_mathlib___private_Mathlib_Init_0__initFn___closed__0_00___x40_Mathlib_Init_2984349899____hygCtx___hyg_3_ = (const lean_object*)&lp_mathlib___private_Mathlib_Init_0__initFn___closed__0_00___x40_Mathlib_Init_2984349899____hygCtx___hyg_3__value;
static const lean_ctor_object lp_mathlib___private_Mathlib_Init_0__initFn___closed__1_00___x40_Mathlib_Init_2984349899____hygCtx___hyg_3__value_aux_0 = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)(((size_t)(0) << 1) | 1)),((lean_object*)&lp_mathlib___private_Mathlib_Init_0__initFn___closed__0_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3__value),LEAN_SCALAR_PTR_LITERAL(186, 218, 113, 226, 101, 176, 32, 79)}};
static const lean_ctor_object lp_mathlib___private_Mathlib_Init_0__initFn___closed__1_00___x40_Mathlib_Init_2984349899____hygCtx___hyg_3__value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*2 + 8, .m_other = 2, .m_tag = 1}, .m_objs = {((lean_object*)&lp_mathlib___private_Mathlib_Init_0__initFn___closed__1_00___x40_Mathlib_Init_2984349899____hygCtx___hyg_3__value_aux_0),((lean_object*)&lp_mathlib___private_Mathlib_Init_0__initFn___closed__0_00___x40_Mathlib_Init_2984349899____hygCtx___hyg_3__value),LEAN_SCALAR_PTR_LITERAL(46, 169, 219, 10, 141, 8, 222, 60)}};
static const lean_object* lp_mathlib___private_Mathlib_Init_0__initFn___closed__1_00___x40_Mathlib_Init_2984349899____hygCtx___hyg_3_ = (const lean_object*)&lp_mathlib___private_Mathlib_Init_0__initFn___closed__1_00___x40_Mathlib_Init_2984349899____hygCtx___hyg_3__value;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_2984349899____hygCtx___hyg_3_();
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_2984349899____hygCtx___hyg_3____boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_linter_weeklyLintSet;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3_(){
_start:
{
lean_object* v___x_7_; lean_object* v___x_8_; 
v___x_7_ = ((lean_object*)(lp_mathlib___private_Mathlib_Init_0__initFn___closed__2_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3_));
v___x_8_ = l_Lean_Linter_registerSet(v___x_7_, v___x_7_);
return v___x_8_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3____boxed(lean_object* v_a_9_){
_start:
{
lean_object* v_res_10_; 
v_res_10_ = lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3_();
return v_res_10_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_2344344261____hygCtx___hyg_3_(){
_start:
{
lean_object* v___x_16_; lean_object* v___x_17_; 
v___x_16_ = ((lean_object*)(lp_mathlib___private_Mathlib_Init_0__initFn___closed__1_00___x40_Mathlib_Init_2344344261____hygCtx___hyg_3_));
v___x_17_ = l_Lean_Linter_registerSet(v___x_16_, v___x_16_);
return v___x_17_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_2344344261____hygCtx___hyg_3____boxed(lean_object* v_a_18_){
_start:
{
lean_object* v_res_19_; 
v_res_19_ = lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_2344344261____hygCtx___hyg_3_();
return v_res_19_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_2984349899____hygCtx___hyg_3_(){
_start:
{
lean_object* v___x_25_; lean_object* v___x_26_; 
v___x_25_ = ((lean_object*)(lp_mathlib___private_Mathlib_Init_0__initFn___closed__1_00___x40_Mathlib_Init_2984349899____hygCtx___hyg_3_));
v___x_26_ = l_Lean_Linter_registerSet(v___x_25_, v___x_25_);
return v___x_26_;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_2984349899____hygCtx___hyg_3____boxed(lean_object* v_a_27_){
_start:
{
lean_object* v_res_28_; 
v_res_28_ = lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_2984349899____hygCtx___hyg_3_();
return v_res_28_;
}
}
lean_object* runtime_initialize_Init(uint8_t builtin);
lean_object* runtime_initialize_Lean_Linter_Sets(uint8_t builtin);
lean_object* runtime_initialize_Lean_LibrarySuggestions_Default(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Lean_Linter(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_AdaptationNote(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Lemma(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_AuxLemma(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_DirectoryDependency(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_DocPrime(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_DocString(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_EmptyLine(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_GlobalAttributeIn(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_HashCommandLinter(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_HaveILetI(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_InternalConstructor(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_FlexibleLinter(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Multigoal(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_OldObtain(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_OverlappingInstances(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_PrivateModule(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_TacticDocumentation(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_UnusedTacticExtension(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_UnusedTactic(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_UnusedInstancesInType(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Style(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Whitespace(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_TacticAnalysis_Declarations(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_TypeStar(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Tactic_HelpCmd(uint8_t builtin);
lean_object* runtime_initialize_batteries_Batteries_Util_ProofWanted(uint8_t builtin);
lean_object* runtime_initialize_importGraph_ImportGraph_Tools(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_Linter_Lint(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Tactic_MinImports(uint8_t builtin);
lean_object* runtime_initialize_mathlib_Mathlib_Util_CodeActions(uint8_t builtin);
void lean_initialize();
static bool _G_runtime_initialized = false;
LEAN_EXPORT lean_object* runtime_initialize_mathlib_Mathlib_Init(uint8_t builtin) {
lean_object * res;
if (_G_runtime_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_runtime_initialized = true;
lean_initialize();
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_Linter_Sets(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_Lean_LibrarySuggestions_Default(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Lean_Linter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_AdaptationNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Lemma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_AuxLemma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_DirectoryDependency(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_DocPrime(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_DocString(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_EmptyLine(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_GlobalAttributeIn(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_HashCommandLinter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_HaveILetI(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_InternalConstructor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_FlexibleLinter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Multigoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_OldObtain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_OverlappingInstances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_PrivateModule(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_TacticDocumentation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_UnusedTacticExtension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_UnusedTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_UnusedInstancesInType(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Style(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Whitespace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_TacticAnalysis_Declarations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_TypeStar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Tactic_HelpCmd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_batteries_Batteries_Util_ProofWanted(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_importGraph_ImportGraph_Tools(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_Linter_Lint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Tactic_MinImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Util_CodeActions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* runtime_initialize_Init(uint8_t builtin);
static bool _G_meta_initialized = false;
LEAN_EXPORT lean_object* meta_initialize_mathlib_Mathlib_Init(uint8_t builtin) {
lean_object * res;
if (_G_meta_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_meta_initialized = true;
res = runtime_initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_3682792609____hygCtx___hyg_3_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_linter_mathlibStandardSet = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_linter_mathlibStandardSet);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_2344344261____hygCtx___hyg_3_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_linter_nightlyRegressionSet = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_linter_nightlyRegressionSet);
lean_dec_ref(res);
res = lp_mathlib___private_Mathlib_Init_0__initFn_00___x40_Mathlib_Init_2984349899____hygCtx___hyg_3_();
if (lean_io_result_is_error(res)) return res;
lp_mathlib_linter_weeklyLintSet = lean_io_result_get_value(res);
lean_mark_persistent(lp_mathlib_linter_weeklyLintSet);
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Linter_Sets(uint8_t builtin);
lean_object* initialize_Lean_LibrarySuggestions_Default(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Lean_Linter(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_AdaptationNote(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Lemma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_AuxLemma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_DirectoryDependency(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_DocPrime(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_DocString(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_EmptyLine(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_GlobalAttributeIn(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_HashCommandLinter(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_HaveILetI(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Header(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_InternalConstructor(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_FlexibleLinter(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Multigoal(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_OldObtain(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_OverlappingInstances(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_PrivateModule(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_TacticDocumentation(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_UnusedTacticExtension(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_UnusedTactic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_UnusedInstancesInType(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Style(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Whitespace(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_TacticAnalysis_Declarations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_TypeStar(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_HelpCmd(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Util_ProofWanted(uint8_t builtin);
lean_object* initialize_importGraph_ImportGraph_Tools(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Linter_Lint(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_MinImports(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_CodeActions(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Linter_Sets(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_LibrarySuggestions_Default(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Lean_Linter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_AdaptationNote(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Lemma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_AuxLemma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_DeprecatedSyntaxLinter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_DirectoryDependency(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_DocPrime(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_DocString(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_EmptyLine(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_GlobalAttributeIn(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_HashCommandLinter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_HaveILetI(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Header(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_InternalConstructor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_FlexibleLinter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Multigoal(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_OldObtain(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_OverlappingInstances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_PrivateModule(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_TacticDocumentation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_UnusedTacticExtension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_UnusedTactic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_UnusedInstancesInType(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Style(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Whitespace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_TacticAnalysis_Declarations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_TypeStar(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_HelpCmd(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Util_ProofWanted(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_importGraph_ImportGraph_Tools(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Linter_Lint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_MinImports(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_CodeActions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = runtime_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = meta_initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return initialize_mathlib_Mathlib_Init(builtin);
}
#ifdef __cplusplus
}
#endif
