// Lean compiler output
// Module: Recursion.Spec.VerifierAirId
// Imports: public import Init public meta import Init
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
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Repr_addAppParen(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorIdx(uint8_t);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorIdx___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorElim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorElim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorElim(lean_object*, lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorElim___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_verifierPvs_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_verifierPvs_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_verifierPvs_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_verifierPvs_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_vmPvs_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_vmPvs_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_vmPvs_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_vmPvs_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_unsetPvs_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_unsetPvs_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_unsetPvs_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_unsetPvs_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_proofShape_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_proofShape_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_proofShape_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_proofShape_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_publicValues_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_publicValues_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_publicValues_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_publicValues_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_transcript_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_transcript_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_transcript_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_transcript_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_poseidon2_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_poseidon2_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_poseidon2_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_poseidon2_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_merkleVerify_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_merkleVerify_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_merkleVerify_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_merkleVerify_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrInput_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrInput_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrInput_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrInput_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayer_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayer_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayer_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayer_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayerSumcheck_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayerSumcheck_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayerSumcheck_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayerSumcheck_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrXiSampler_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrXiSampler_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrXiSampler_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrXiSampler_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_symbolicExpression_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_symbolicExpression_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_symbolicExpression_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_symbolicExpression_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_fractionsFolder_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_fractionsFolder_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_fractionsFolder_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_fractionsFolder_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateSumcheck_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateSumcheck_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateSumcheck_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateSumcheck_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_multilinearSumcheck_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_multilinearSumcheck_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_multilinearSumcheck_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_multilinearSumcheck_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNs_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNs_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNs_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNs_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eq3b_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eq3b_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eq3b_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eq3b_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUni_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUni_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUni_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUni_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUniReceiver_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUniReceiver_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUniReceiver_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUniReceiver_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqUni_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqUni_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqUni_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqUni_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expressionClaim_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expressionClaim_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expressionClaim_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expressionClaim_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_interactionsFolding_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_interactionsFolding_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_interactionsFolding_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_interactionsFolding_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_constraintsFolding_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_constraintsFolding_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_constraintsFolding_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_constraintsFolding_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNeg_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNeg_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNeg_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNeg_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_openingClaims_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_openingClaims_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_openingClaims_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_openingClaims_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateRound_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateRound_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateRound_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateRound_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_sumcheckRounds_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_sumcheckRounds_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_sumcheckRounds_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_sumcheckRounds_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_stackingClaims_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_stackingClaims_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_stackingClaims_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_stackingClaims_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBase_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBase_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBase_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBase_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBits_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBits_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBits_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBits_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirRound_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirRound_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirRound_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirRound_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirSumcheck_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirSumcheck_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirSumcheck_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirSumcheck_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirQuery_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirQuery_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirQuery_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirQuery_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_initialOpenedValues_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_initialOpenedValues_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_initialOpenedValues_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_initialOpenedValues_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_nonInitialOpenedValues_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_nonInitialOpenedValues_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_nonInitialOpenedValues_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_nonInitialOpenedValues_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirFolding_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirFolding_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirFolding_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirFolding_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyMleEval_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyMleEval_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyMleEval_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyMleEval_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyQueryEval_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyQueryEval_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyQueryEval_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyQueryEval_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_rangeChecker_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_rangeChecker_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_rangeChecker_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_rangeChecker_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_powerChecker_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_powerChecker_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_powerChecker_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_powerChecker_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expBitsLen_elim___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expBitsLen_elim___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expBitsLen_elim(lean_object*, uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expBitsLen_elim___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ofNat(lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ofNat___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_workspace_Recursion_Spec_VerifierCircuit_instDecidableEqVerifierAirId(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instDecidableEqVerifierAirId___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_workspace_Recursion_Spec_VerifierCircuit_instBEqVerifierAirId_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instBEqVerifierAirId_beq___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_workspace_Recursion_Spec_VerifierCircuit_instBEqVerifierAirId___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_workspace_Recursion_Spec_VerifierCircuit_instBEqVerifierAirId_beq___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instBEqVerifierAirId___closed__0 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instBEqVerifierAirId___closed__0_value;
LEAN_EXPORT const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instBEqVerifierAirId = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instBEqVerifierAirId___closed__0_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.verifierPvs"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__0 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__0_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__1_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__0_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__1 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__1_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__2_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 51, .m_capacity = 51, .m_length = 50, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.vmPvs"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__2 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__2_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__3_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__2_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__3 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__3_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__4_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.unsetPvs"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__4 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__4_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__5_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__4_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__5 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__5_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__6_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.proofShape"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__6 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__6_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__7_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__6_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__7 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__7_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__8_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 57, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.publicValues"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__8 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__8_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__9_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__8_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__9 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__9_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__10_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.transcript"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__10 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__10_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__11_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__10_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__11 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__11_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__12_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.poseidon2"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__12 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__12_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__13_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__12_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__13 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__13_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__14_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 57, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.merkleVerify"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__14 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__14_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__15_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__14_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__15 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__15_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__16_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.gkrInput"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__16 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__16_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__17_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__16_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__17 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__17_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__18_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 54, .m_capacity = 54, .m_length = 53, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.gkrLayer"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__18 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__18_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__19_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__18_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__19 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__19_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__20_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 62, .m_capacity = 62, .m_length = 61, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.gkrLayerSumcheck"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__20 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__20_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__21_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__20_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__21 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__21_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__22_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 57, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.gkrXiSampler"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__22 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__22_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__23_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__22_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__23 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__23_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__24_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 64, .m_capacity = 64, .m_length = 63, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.symbolicExpression"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__24 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__24_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__25_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__24_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__25 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__25_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__26_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 61, .m_capacity = 61, .m_length = 60, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.fractionsFolder"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__26 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__26_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__27_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__26_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__27 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__27_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__28_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 64, .m_capacity = 64, .m_length = 63, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.univariateSumcheck"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__28 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__28_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__29_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__28_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__29 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__29_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__30_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 65, .m_capacity = 65, .m_length = 64, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.multilinearSumcheck"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__30 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__30_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__31_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__30_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__31 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__31_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__32_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 50, .m_capacity = 50, .m_length = 49, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.eqNs"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__32 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__32_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__33_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__32_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__33 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__33_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__34_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 50, .m_capacity = 50, .m_length = 49, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.eq3b"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__34 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__34_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__35_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__34_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__35 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__35_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__36_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.eqSharpUni"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__36 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__36_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__37_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__36_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__37 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__37_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__38_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 64, .m_capacity = 64, .m_length = 63, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.eqSharpUniReceiver"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__38 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__38_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__39_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__38_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__39 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__39_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__40_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 51, .m_capacity = 51, .m_length = 50, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.eqUni"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__40 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__40_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__41_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__40_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__41 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__41_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__42_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 61, .m_capacity = 61, .m_length = 60, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.expressionClaim"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__42 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__42_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__43_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__42_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__43 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__43_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__44_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 65, .m_capacity = 65, .m_length = 64, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.interactionsFolding"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__44 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__44_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__45_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__44_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__45 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__45_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__46_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 64, .m_capacity = 64, .m_length = 63, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.constraintsFolding"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__46 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__46_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__47_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__46_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__47 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__47_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__48_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 51, .m_capacity = 51, .m_length = 50, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.eqNeg"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__48 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__48_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__49_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__48_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__49 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__49_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__50_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 59, .m_capacity = 59, .m_length = 58, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.openingClaims"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__50 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__50_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__51_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__50_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__51 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__51_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__52_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 61, .m_capacity = 61, .m_length = 60, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.univariateRound"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__52 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__52_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__53_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__52_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__53 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__53_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__54_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.sumcheckRounds"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__54 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__54_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__55_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__54_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__55 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__55_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__56_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 60, .m_capacity = 60, .m_length = 59, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.stackingClaims"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__56 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__56_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__57_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__56_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__57 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__57_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__58_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 52, .m_capacity = 52, .m_length = 51, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.eqBase"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__58 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__58_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__59_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__58_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__59 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__59_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__60_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 52, .m_capacity = 52, .m_length = 51, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.eqBits"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__60 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__60_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__61_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__60_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__61 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__61_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__62_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.whirRound"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__62 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__62_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__63_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__62_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__63 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__63_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__64_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 57, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.whirSumcheck"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__64 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__64_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__65_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__64_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__65 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__65_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__66_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 55, .m_capacity = 55, .m_length = 54, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.whirQuery"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__66 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__66_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__67_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__66_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__67 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__67_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__68_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 65, .m_capacity = 65, .m_length = 64, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.initialOpenedValues"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__68 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__68_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__69_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__68_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__69 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__69_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__70_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 68, .m_capacity = 68, .m_length = 67, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.nonInitialOpenedValues"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__70 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__70_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__71_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__70_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__71 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__71_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__72_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 57, .m_capacity = 57, .m_length = 56, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.whirFolding"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__72 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__72_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__73_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__72_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__73 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__73_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__74_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 62, .m_capacity = 62, .m_length = 61, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.finalPolyMleEval"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__74 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__74_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__75_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__74_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__75 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__75_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__76_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 64, .m_capacity = 64, .m_length = 63, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.finalPolyQueryEval"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__76 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__76_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__77_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__76_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__77 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__77_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__78_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 57, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.rangeChecker"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__78 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__78_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__79_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__78_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__79 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__79_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__80_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 58, .m_capacity = 58, .m_length = 57, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.powerChecker"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__80 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__80_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__81_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__80_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__81 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__81_value;
static const lean_string_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__82_value = {.m_header = {.m_rc = 0, .m_cs_sz = 0, .m_other = 0, .m_tag = 249}, .m_size = 56, .m_capacity = 56, .m_length = 55, .m_data = "Recursion.Spec.VerifierCircuit.VerifierAirId.expBitsLen"};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__82 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__82_value;
static const lean_ctor_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__83_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_ctor_object) + sizeof(void*)*1 + 0, .m_other = 1, .m_tag = 3}, .m_objs = {((lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__82_value)}};
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__83 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__83_value;
static lean_once_cell_t lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84;
static lean_once_cell_t lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once = LEAN_ONCE_CELL_INITIALIZER;
static lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85;
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr(uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___boxed(lean_object*, lean_object*);
static const lean_closure_object lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId___closed__0_value = {.m_header = {.m_rc = 0, .m_cs_sz = sizeof(lean_closure_object) + sizeof(void*)*0, .m_other = 0, .m_tag = 245}, .m_fun = (void*)lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___boxed, .m_arity = 2, .m_num_fixed = 0, .m_objs = {} };
static const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId___closed__0 = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId___closed__0_value;
LEAN_EXPORT const lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId = (const lean_object*)&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId___closed__0_value;
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorIdx(uint8_t v_x_1_){
_start:
{
switch(v_x_1_)
{
case 0:
{
lean_object* v___x_2_; 
v___x_2_ = lean_unsigned_to_nat(0u);
return v___x_2_;
}
case 1:
{
lean_object* v___x_3_; 
v___x_3_ = lean_unsigned_to_nat(1u);
return v___x_3_;
}
case 2:
{
lean_object* v___x_4_; 
v___x_4_ = lean_unsigned_to_nat(2u);
return v___x_4_;
}
case 3:
{
lean_object* v___x_5_; 
v___x_5_ = lean_unsigned_to_nat(3u);
return v___x_5_;
}
case 4:
{
lean_object* v___x_6_; 
v___x_6_ = lean_unsigned_to_nat(4u);
return v___x_6_;
}
case 5:
{
lean_object* v___x_7_; 
v___x_7_ = lean_unsigned_to_nat(5u);
return v___x_7_;
}
case 6:
{
lean_object* v___x_8_; 
v___x_8_ = lean_unsigned_to_nat(6u);
return v___x_8_;
}
case 7:
{
lean_object* v___x_9_; 
v___x_9_ = lean_unsigned_to_nat(7u);
return v___x_9_;
}
case 8:
{
lean_object* v___x_10_; 
v___x_10_ = lean_unsigned_to_nat(8u);
return v___x_10_;
}
case 9:
{
lean_object* v___x_11_; 
v___x_11_ = lean_unsigned_to_nat(9u);
return v___x_11_;
}
case 10:
{
lean_object* v___x_12_; 
v___x_12_ = lean_unsigned_to_nat(10u);
return v___x_12_;
}
case 11:
{
lean_object* v___x_13_; 
v___x_13_ = lean_unsigned_to_nat(11u);
return v___x_13_;
}
case 12:
{
lean_object* v___x_14_; 
v___x_14_ = lean_unsigned_to_nat(12u);
return v___x_14_;
}
case 13:
{
lean_object* v___x_15_; 
v___x_15_ = lean_unsigned_to_nat(13u);
return v___x_15_;
}
case 14:
{
lean_object* v___x_16_; 
v___x_16_ = lean_unsigned_to_nat(14u);
return v___x_16_;
}
case 15:
{
lean_object* v___x_17_; 
v___x_17_ = lean_unsigned_to_nat(15u);
return v___x_17_;
}
case 16:
{
lean_object* v___x_18_; 
v___x_18_ = lean_unsigned_to_nat(16u);
return v___x_18_;
}
case 17:
{
lean_object* v___x_19_; 
v___x_19_ = lean_unsigned_to_nat(17u);
return v___x_19_;
}
case 18:
{
lean_object* v___x_20_; 
v___x_20_ = lean_unsigned_to_nat(18u);
return v___x_20_;
}
case 19:
{
lean_object* v___x_21_; 
v___x_21_ = lean_unsigned_to_nat(19u);
return v___x_21_;
}
case 20:
{
lean_object* v___x_22_; 
v___x_22_ = lean_unsigned_to_nat(20u);
return v___x_22_;
}
case 21:
{
lean_object* v___x_23_; 
v___x_23_ = lean_unsigned_to_nat(21u);
return v___x_23_;
}
case 22:
{
lean_object* v___x_24_; 
v___x_24_ = lean_unsigned_to_nat(22u);
return v___x_24_;
}
case 23:
{
lean_object* v___x_25_; 
v___x_25_ = lean_unsigned_to_nat(23u);
return v___x_25_;
}
case 24:
{
lean_object* v___x_26_; 
v___x_26_ = lean_unsigned_to_nat(24u);
return v___x_26_;
}
case 25:
{
lean_object* v___x_27_; 
v___x_27_ = lean_unsigned_to_nat(25u);
return v___x_27_;
}
case 26:
{
lean_object* v___x_28_; 
v___x_28_ = lean_unsigned_to_nat(26u);
return v___x_28_;
}
case 27:
{
lean_object* v___x_29_; 
v___x_29_ = lean_unsigned_to_nat(27u);
return v___x_29_;
}
case 28:
{
lean_object* v___x_30_; 
v___x_30_ = lean_unsigned_to_nat(28u);
return v___x_30_;
}
case 29:
{
lean_object* v___x_31_; 
v___x_31_ = lean_unsigned_to_nat(29u);
return v___x_31_;
}
case 30:
{
lean_object* v___x_32_; 
v___x_32_ = lean_unsigned_to_nat(30u);
return v___x_32_;
}
case 31:
{
lean_object* v___x_33_; 
v___x_33_ = lean_unsigned_to_nat(31u);
return v___x_33_;
}
case 32:
{
lean_object* v___x_34_; 
v___x_34_ = lean_unsigned_to_nat(32u);
return v___x_34_;
}
case 33:
{
lean_object* v___x_35_; 
v___x_35_ = lean_unsigned_to_nat(33u);
return v___x_35_;
}
case 34:
{
lean_object* v___x_36_; 
v___x_36_ = lean_unsigned_to_nat(34u);
return v___x_36_;
}
case 35:
{
lean_object* v___x_37_; 
v___x_37_ = lean_unsigned_to_nat(35u);
return v___x_37_;
}
case 36:
{
lean_object* v___x_38_; 
v___x_38_ = lean_unsigned_to_nat(36u);
return v___x_38_;
}
case 37:
{
lean_object* v___x_39_; 
v___x_39_ = lean_unsigned_to_nat(37u);
return v___x_39_;
}
case 38:
{
lean_object* v___x_40_; 
v___x_40_ = lean_unsigned_to_nat(38u);
return v___x_40_;
}
case 39:
{
lean_object* v___x_41_; 
v___x_41_ = lean_unsigned_to_nat(39u);
return v___x_41_;
}
case 40:
{
lean_object* v___x_42_; 
v___x_42_ = lean_unsigned_to_nat(40u);
return v___x_42_;
}
default: 
{
lean_object* v___x_43_; 
v___x_43_ = lean_unsigned_to_nat(41u);
return v___x_43_;
}
}
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorIdx___boxed(lean_object* v_x_44_){
_start:
{
uint8_t v_x_boxed_45_; lean_object* v_res_46_; 
v_x_boxed_45_ = lean_unbox(v_x_44_);
v_res_46_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorIdx(v_x_boxed_45_);
return v_res_46_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorElim___redArg(lean_object* v_k_47_){
_start:
{
lean_inc(v_k_47_);
return v_k_47_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorElim___redArg___boxed(lean_object* v_k_48_){
_start:
{
lean_object* v_res_49_; 
v_res_49_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorElim___redArg(v_k_48_);
lean_dec(v_k_48_);
return v_res_49_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorElim(lean_object* v_motive_50_, lean_object* v_ctorIdx_51_, uint8_t v_t_52_, lean_object* v_h_53_, lean_object* v_k_54_){
_start:
{
lean_inc(v_k_54_);
return v_k_54_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorElim___boxed(lean_object* v_motive_55_, lean_object* v_ctorIdx_56_, lean_object* v_t_57_, lean_object* v_h_58_, lean_object* v_k_59_){
_start:
{
uint8_t v_t_boxed_60_; lean_object* v_res_61_; 
v_t_boxed_60_ = lean_unbox(v_t_57_);
v_res_61_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorElim(v_motive_55_, v_ctorIdx_56_, v_t_boxed_60_, v_h_58_, v_k_59_);
lean_dec(v_k_59_);
lean_dec(v_ctorIdx_56_);
return v_res_61_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_verifierPvs_elim___redArg(lean_object* v_verifierPvs_62_){
_start:
{
lean_inc(v_verifierPvs_62_);
return v_verifierPvs_62_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_verifierPvs_elim___redArg___boxed(lean_object* v_verifierPvs_63_){
_start:
{
lean_object* v_res_64_; 
v_res_64_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_verifierPvs_elim___redArg(v_verifierPvs_63_);
lean_dec(v_verifierPvs_63_);
return v_res_64_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_verifierPvs_elim(lean_object* v_motive_65_, uint8_t v_t_66_, lean_object* v_h_67_, lean_object* v_verifierPvs_68_){
_start:
{
lean_inc(v_verifierPvs_68_);
return v_verifierPvs_68_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_verifierPvs_elim___boxed(lean_object* v_motive_69_, lean_object* v_t_70_, lean_object* v_h_71_, lean_object* v_verifierPvs_72_){
_start:
{
uint8_t v_t_boxed_73_; lean_object* v_res_74_; 
v_t_boxed_73_ = lean_unbox(v_t_70_);
v_res_74_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_verifierPvs_elim(v_motive_69_, v_t_boxed_73_, v_h_71_, v_verifierPvs_72_);
lean_dec(v_verifierPvs_72_);
return v_res_74_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_vmPvs_elim___redArg(lean_object* v_vmPvs_75_){
_start:
{
lean_inc(v_vmPvs_75_);
return v_vmPvs_75_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_vmPvs_elim___redArg___boxed(lean_object* v_vmPvs_76_){
_start:
{
lean_object* v_res_77_; 
v_res_77_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_vmPvs_elim___redArg(v_vmPvs_76_);
lean_dec(v_vmPvs_76_);
return v_res_77_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_vmPvs_elim(lean_object* v_motive_78_, uint8_t v_t_79_, lean_object* v_h_80_, lean_object* v_vmPvs_81_){
_start:
{
lean_inc(v_vmPvs_81_);
return v_vmPvs_81_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_vmPvs_elim___boxed(lean_object* v_motive_82_, lean_object* v_t_83_, lean_object* v_h_84_, lean_object* v_vmPvs_85_){
_start:
{
uint8_t v_t_boxed_86_; lean_object* v_res_87_; 
v_t_boxed_86_ = lean_unbox(v_t_83_);
v_res_87_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_vmPvs_elim(v_motive_82_, v_t_boxed_86_, v_h_84_, v_vmPvs_85_);
lean_dec(v_vmPvs_85_);
return v_res_87_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_unsetPvs_elim___redArg(lean_object* v_unsetPvs_88_){
_start:
{
lean_inc(v_unsetPvs_88_);
return v_unsetPvs_88_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_unsetPvs_elim___redArg___boxed(lean_object* v_unsetPvs_89_){
_start:
{
lean_object* v_res_90_; 
v_res_90_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_unsetPvs_elim___redArg(v_unsetPvs_89_);
lean_dec(v_unsetPvs_89_);
return v_res_90_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_unsetPvs_elim(lean_object* v_motive_91_, uint8_t v_t_92_, lean_object* v_h_93_, lean_object* v_unsetPvs_94_){
_start:
{
lean_inc(v_unsetPvs_94_);
return v_unsetPvs_94_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_unsetPvs_elim___boxed(lean_object* v_motive_95_, lean_object* v_t_96_, lean_object* v_h_97_, lean_object* v_unsetPvs_98_){
_start:
{
uint8_t v_t_boxed_99_; lean_object* v_res_100_; 
v_t_boxed_99_ = lean_unbox(v_t_96_);
v_res_100_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_unsetPvs_elim(v_motive_95_, v_t_boxed_99_, v_h_97_, v_unsetPvs_98_);
lean_dec(v_unsetPvs_98_);
return v_res_100_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_proofShape_elim___redArg(lean_object* v_proofShape_101_){
_start:
{
lean_inc(v_proofShape_101_);
return v_proofShape_101_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_proofShape_elim___redArg___boxed(lean_object* v_proofShape_102_){
_start:
{
lean_object* v_res_103_; 
v_res_103_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_proofShape_elim___redArg(v_proofShape_102_);
lean_dec(v_proofShape_102_);
return v_res_103_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_proofShape_elim(lean_object* v_motive_104_, uint8_t v_t_105_, lean_object* v_h_106_, lean_object* v_proofShape_107_){
_start:
{
lean_inc(v_proofShape_107_);
return v_proofShape_107_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_proofShape_elim___boxed(lean_object* v_motive_108_, lean_object* v_t_109_, lean_object* v_h_110_, lean_object* v_proofShape_111_){
_start:
{
uint8_t v_t_boxed_112_; lean_object* v_res_113_; 
v_t_boxed_112_ = lean_unbox(v_t_109_);
v_res_113_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_proofShape_elim(v_motive_108_, v_t_boxed_112_, v_h_110_, v_proofShape_111_);
lean_dec(v_proofShape_111_);
return v_res_113_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_publicValues_elim___redArg(lean_object* v_publicValues_114_){
_start:
{
lean_inc(v_publicValues_114_);
return v_publicValues_114_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_publicValues_elim___redArg___boxed(lean_object* v_publicValues_115_){
_start:
{
lean_object* v_res_116_; 
v_res_116_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_publicValues_elim___redArg(v_publicValues_115_);
lean_dec(v_publicValues_115_);
return v_res_116_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_publicValues_elim(lean_object* v_motive_117_, uint8_t v_t_118_, lean_object* v_h_119_, lean_object* v_publicValues_120_){
_start:
{
lean_inc(v_publicValues_120_);
return v_publicValues_120_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_publicValues_elim___boxed(lean_object* v_motive_121_, lean_object* v_t_122_, lean_object* v_h_123_, lean_object* v_publicValues_124_){
_start:
{
uint8_t v_t_boxed_125_; lean_object* v_res_126_; 
v_t_boxed_125_ = lean_unbox(v_t_122_);
v_res_126_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_publicValues_elim(v_motive_121_, v_t_boxed_125_, v_h_123_, v_publicValues_124_);
lean_dec(v_publicValues_124_);
return v_res_126_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_transcript_elim___redArg(lean_object* v_transcript_127_){
_start:
{
lean_inc(v_transcript_127_);
return v_transcript_127_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_transcript_elim___redArg___boxed(lean_object* v_transcript_128_){
_start:
{
lean_object* v_res_129_; 
v_res_129_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_transcript_elim___redArg(v_transcript_128_);
lean_dec(v_transcript_128_);
return v_res_129_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_transcript_elim(lean_object* v_motive_130_, uint8_t v_t_131_, lean_object* v_h_132_, lean_object* v_transcript_133_){
_start:
{
lean_inc(v_transcript_133_);
return v_transcript_133_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_transcript_elim___boxed(lean_object* v_motive_134_, lean_object* v_t_135_, lean_object* v_h_136_, lean_object* v_transcript_137_){
_start:
{
uint8_t v_t_boxed_138_; lean_object* v_res_139_; 
v_t_boxed_138_ = lean_unbox(v_t_135_);
v_res_139_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_transcript_elim(v_motive_134_, v_t_boxed_138_, v_h_136_, v_transcript_137_);
lean_dec(v_transcript_137_);
return v_res_139_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_poseidon2_elim___redArg(lean_object* v_poseidon2_140_){
_start:
{
lean_inc(v_poseidon2_140_);
return v_poseidon2_140_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_poseidon2_elim___redArg___boxed(lean_object* v_poseidon2_141_){
_start:
{
lean_object* v_res_142_; 
v_res_142_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_poseidon2_elim___redArg(v_poseidon2_141_);
lean_dec(v_poseidon2_141_);
return v_res_142_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_poseidon2_elim(lean_object* v_motive_143_, uint8_t v_t_144_, lean_object* v_h_145_, lean_object* v_poseidon2_146_){
_start:
{
lean_inc(v_poseidon2_146_);
return v_poseidon2_146_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_poseidon2_elim___boxed(lean_object* v_motive_147_, lean_object* v_t_148_, lean_object* v_h_149_, lean_object* v_poseidon2_150_){
_start:
{
uint8_t v_t_boxed_151_; lean_object* v_res_152_; 
v_t_boxed_151_ = lean_unbox(v_t_148_);
v_res_152_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_poseidon2_elim(v_motive_147_, v_t_boxed_151_, v_h_149_, v_poseidon2_150_);
lean_dec(v_poseidon2_150_);
return v_res_152_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_merkleVerify_elim___redArg(lean_object* v_merkleVerify_153_){
_start:
{
lean_inc(v_merkleVerify_153_);
return v_merkleVerify_153_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_merkleVerify_elim___redArg___boxed(lean_object* v_merkleVerify_154_){
_start:
{
lean_object* v_res_155_; 
v_res_155_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_merkleVerify_elim___redArg(v_merkleVerify_154_);
lean_dec(v_merkleVerify_154_);
return v_res_155_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_merkleVerify_elim(lean_object* v_motive_156_, uint8_t v_t_157_, lean_object* v_h_158_, lean_object* v_merkleVerify_159_){
_start:
{
lean_inc(v_merkleVerify_159_);
return v_merkleVerify_159_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_merkleVerify_elim___boxed(lean_object* v_motive_160_, lean_object* v_t_161_, lean_object* v_h_162_, lean_object* v_merkleVerify_163_){
_start:
{
uint8_t v_t_boxed_164_; lean_object* v_res_165_; 
v_t_boxed_164_ = lean_unbox(v_t_161_);
v_res_165_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_merkleVerify_elim(v_motive_160_, v_t_boxed_164_, v_h_162_, v_merkleVerify_163_);
lean_dec(v_merkleVerify_163_);
return v_res_165_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrInput_elim___redArg(lean_object* v_gkrInput_166_){
_start:
{
lean_inc(v_gkrInput_166_);
return v_gkrInput_166_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrInput_elim___redArg___boxed(lean_object* v_gkrInput_167_){
_start:
{
lean_object* v_res_168_; 
v_res_168_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrInput_elim___redArg(v_gkrInput_167_);
lean_dec(v_gkrInput_167_);
return v_res_168_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrInput_elim(lean_object* v_motive_169_, uint8_t v_t_170_, lean_object* v_h_171_, lean_object* v_gkrInput_172_){
_start:
{
lean_inc(v_gkrInput_172_);
return v_gkrInput_172_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrInput_elim___boxed(lean_object* v_motive_173_, lean_object* v_t_174_, lean_object* v_h_175_, lean_object* v_gkrInput_176_){
_start:
{
uint8_t v_t_boxed_177_; lean_object* v_res_178_; 
v_t_boxed_177_ = lean_unbox(v_t_174_);
v_res_178_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrInput_elim(v_motive_173_, v_t_boxed_177_, v_h_175_, v_gkrInput_176_);
lean_dec(v_gkrInput_176_);
return v_res_178_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayer_elim___redArg(lean_object* v_gkrLayer_179_){
_start:
{
lean_inc(v_gkrLayer_179_);
return v_gkrLayer_179_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayer_elim___redArg___boxed(lean_object* v_gkrLayer_180_){
_start:
{
lean_object* v_res_181_; 
v_res_181_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayer_elim___redArg(v_gkrLayer_180_);
lean_dec(v_gkrLayer_180_);
return v_res_181_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayer_elim(lean_object* v_motive_182_, uint8_t v_t_183_, lean_object* v_h_184_, lean_object* v_gkrLayer_185_){
_start:
{
lean_inc(v_gkrLayer_185_);
return v_gkrLayer_185_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayer_elim___boxed(lean_object* v_motive_186_, lean_object* v_t_187_, lean_object* v_h_188_, lean_object* v_gkrLayer_189_){
_start:
{
uint8_t v_t_boxed_190_; lean_object* v_res_191_; 
v_t_boxed_190_ = lean_unbox(v_t_187_);
v_res_191_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayer_elim(v_motive_186_, v_t_boxed_190_, v_h_188_, v_gkrLayer_189_);
lean_dec(v_gkrLayer_189_);
return v_res_191_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayerSumcheck_elim___redArg(lean_object* v_gkrLayerSumcheck_192_){
_start:
{
lean_inc(v_gkrLayerSumcheck_192_);
return v_gkrLayerSumcheck_192_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayerSumcheck_elim___redArg___boxed(lean_object* v_gkrLayerSumcheck_193_){
_start:
{
lean_object* v_res_194_; 
v_res_194_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayerSumcheck_elim___redArg(v_gkrLayerSumcheck_193_);
lean_dec(v_gkrLayerSumcheck_193_);
return v_res_194_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayerSumcheck_elim(lean_object* v_motive_195_, uint8_t v_t_196_, lean_object* v_h_197_, lean_object* v_gkrLayerSumcheck_198_){
_start:
{
lean_inc(v_gkrLayerSumcheck_198_);
return v_gkrLayerSumcheck_198_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayerSumcheck_elim___boxed(lean_object* v_motive_199_, lean_object* v_t_200_, lean_object* v_h_201_, lean_object* v_gkrLayerSumcheck_202_){
_start:
{
uint8_t v_t_boxed_203_; lean_object* v_res_204_; 
v_t_boxed_203_ = lean_unbox(v_t_200_);
v_res_204_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrLayerSumcheck_elim(v_motive_199_, v_t_boxed_203_, v_h_201_, v_gkrLayerSumcheck_202_);
lean_dec(v_gkrLayerSumcheck_202_);
return v_res_204_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrXiSampler_elim___redArg(lean_object* v_gkrXiSampler_205_){
_start:
{
lean_inc(v_gkrXiSampler_205_);
return v_gkrXiSampler_205_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrXiSampler_elim___redArg___boxed(lean_object* v_gkrXiSampler_206_){
_start:
{
lean_object* v_res_207_; 
v_res_207_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrXiSampler_elim___redArg(v_gkrXiSampler_206_);
lean_dec(v_gkrXiSampler_206_);
return v_res_207_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrXiSampler_elim(lean_object* v_motive_208_, uint8_t v_t_209_, lean_object* v_h_210_, lean_object* v_gkrXiSampler_211_){
_start:
{
lean_inc(v_gkrXiSampler_211_);
return v_gkrXiSampler_211_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrXiSampler_elim___boxed(lean_object* v_motive_212_, lean_object* v_t_213_, lean_object* v_h_214_, lean_object* v_gkrXiSampler_215_){
_start:
{
uint8_t v_t_boxed_216_; lean_object* v_res_217_; 
v_t_boxed_216_ = lean_unbox(v_t_213_);
v_res_217_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_gkrXiSampler_elim(v_motive_212_, v_t_boxed_216_, v_h_214_, v_gkrXiSampler_215_);
lean_dec(v_gkrXiSampler_215_);
return v_res_217_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_symbolicExpression_elim___redArg(lean_object* v_symbolicExpression_218_){
_start:
{
lean_inc(v_symbolicExpression_218_);
return v_symbolicExpression_218_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_symbolicExpression_elim___redArg___boxed(lean_object* v_symbolicExpression_219_){
_start:
{
lean_object* v_res_220_; 
v_res_220_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_symbolicExpression_elim___redArg(v_symbolicExpression_219_);
lean_dec(v_symbolicExpression_219_);
return v_res_220_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_symbolicExpression_elim(lean_object* v_motive_221_, uint8_t v_t_222_, lean_object* v_h_223_, lean_object* v_symbolicExpression_224_){
_start:
{
lean_inc(v_symbolicExpression_224_);
return v_symbolicExpression_224_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_symbolicExpression_elim___boxed(lean_object* v_motive_225_, lean_object* v_t_226_, lean_object* v_h_227_, lean_object* v_symbolicExpression_228_){
_start:
{
uint8_t v_t_boxed_229_; lean_object* v_res_230_; 
v_t_boxed_229_ = lean_unbox(v_t_226_);
v_res_230_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_symbolicExpression_elim(v_motive_225_, v_t_boxed_229_, v_h_227_, v_symbolicExpression_228_);
lean_dec(v_symbolicExpression_228_);
return v_res_230_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_fractionsFolder_elim___redArg(lean_object* v_fractionsFolder_231_){
_start:
{
lean_inc(v_fractionsFolder_231_);
return v_fractionsFolder_231_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_fractionsFolder_elim___redArg___boxed(lean_object* v_fractionsFolder_232_){
_start:
{
lean_object* v_res_233_; 
v_res_233_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_fractionsFolder_elim___redArg(v_fractionsFolder_232_);
lean_dec(v_fractionsFolder_232_);
return v_res_233_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_fractionsFolder_elim(lean_object* v_motive_234_, uint8_t v_t_235_, lean_object* v_h_236_, lean_object* v_fractionsFolder_237_){
_start:
{
lean_inc(v_fractionsFolder_237_);
return v_fractionsFolder_237_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_fractionsFolder_elim___boxed(lean_object* v_motive_238_, lean_object* v_t_239_, lean_object* v_h_240_, lean_object* v_fractionsFolder_241_){
_start:
{
uint8_t v_t_boxed_242_; lean_object* v_res_243_; 
v_t_boxed_242_ = lean_unbox(v_t_239_);
v_res_243_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_fractionsFolder_elim(v_motive_238_, v_t_boxed_242_, v_h_240_, v_fractionsFolder_241_);
lean_dec(v_fractionsFolder_241_);
return v_res_243_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateSumcheck_elim___redArg(lean_object* v_univariateSumcheck_244_){
_start:
{
lean_inc(v_univariateSumcheck_244_);
return v_univariateSumcheck_244_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateSumcheck_elim___redArg___boxed(lean_object* v_univariateSumcheck_245_){
_start:
{
lean_object* v_res_246_; 
v_res_246_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateSumcheck_elim___redArg(v_univariateSumcheck_245_);
lean_dec(v_univariateSumcheck_245_);
return v_res_246_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateSumcheck_elim(lean_object* v_motive_247_, uint8_t v_t_248_, lean_object* v_h_249_, lean_object* v_univariateSumcheck_250_){
_start:
{
lean_inc(v_univariateSumcheck_250_);
return v_univariateSumcheck_250_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateSumcheck_elim___boxed(lean_object* v_motive_251_, lean_object* v_t_252_, lean_object* v_h_253_, lean_object* v_univariateSumcheck_254_){
_start:
{
uint8_t v_t_boxed_255_; lean_object* v_res_256_; 
v_t_boxed_255_ = lean_unbox(v_t_252_);
v_res_256_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateSumcheck_elim(v_motive_251_, v_t_boxed_255_, v_h_253_, v_univariateSumcheck_254_);
lean_dec(v_univariateSumcheck_254_);
return v_res_256_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_multilinearSumcheck_elim___redArg(lean_object* v_multilinearSumcheck_257_){
_start:
{
lean_inc(v_multilinearSumcheck_257_);
return v_multilinearSumcheck_257_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_multilinearSumcheck_elim___redArg___boxed(lean_object* v_multilinearSumcheck_258_){
_start:
{
lean_object* v_res_259_; 
v_res_259_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_multilinearSumcheck_elim___redArg(v_multilinearSumcheck_258_);
lean_dec(v_multilinearSumcheck_258_);
return v_res_259_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_multilinearSumcheck_elim(lean_object* v_motive_260_, uint8_t v_t_261_, lean_object* v_h_262_, lean_object* v_multilinearSumcheck_263_){
_start:
{
lean_inc(v_multilinearSumcheck_263_);
return v_multilinearSumcheck_263_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_multilinearSumcheck_elim___boxed(lean_object* v_motive_264_, lean_object* v_t_265_, lean_object* v_h_266_, lean_object* v_multilinearSumcheck_267_){
_start:
{
uint8_t v_t_boxed_268_; lean_object* v_res_269_; 
v_t_boxed_268_ = lean_unbox(v_t_265_);
v_res_269_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_multilinearSumcheck_elim(v_motive_264_, v_t_boxed_268_, v_h_266_, v_multilinearSumcheck_267_);
lean_dec(v_multilinearSumcheck_267_);
return v_res_269_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNs_elim___redArg(lean_object* v_eqNs_270_){
_start:
{
lean_inc(v_eqNs_270_);
return v_eqNs_270_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNs_elim___redArg___boxed(lean_object* v_eqNs_271_){
_start:
{
lean_object* v_res_272_; 
v_res_272_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNs_elim___redArg(v_eqNs_271_);
lean_dec(v_eqNs_271_);
return v_res_272_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNs_elim(lean_object* v_motive_273_, uint8_t v_t_274_, lean_object* v_h_275_, lean_object* v_eqNs_276_){
_start:
{
lean_inc(v_eqNs_276_);
return v_eqNs_276_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNs_elim___boxed(lean_object* v_motive_277_, lean_object* v_t_278_, lean_object* v_h_279_, lean_object* v_eqNs_280_){
_start:
{
uint8_t v_t_boxed_281_; lean_object* v_res_282_; 
v_t_boxed_281_ = lean_unbox(v_t_278_);
v_res_282_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNs_elim(v_motive_277_, v_t_boxed_281_, v_h_279_, v_eqNs_280_);
lean_dec(v_eqNs_280_);
return v_res_282_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eq3b_elim___redArg(lean_object* v_eq3b_283_){
_start:
{
lean_inc(v_eq3b_283_);
return v_eq3b_283_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eq3b_elim___redArg___boxed(lean_object* v_eq3b_284_){
_start:
{
lean_object* v_res_285_; 
v_res_285_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eq3b_elim___redArg(v_eq3b_284_);
lean_dec(v_eq3b_284_);
return v_res_285_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eq3b_elim(lean_object* v_motive_286_, uint8_t v_t_287_, lean_object* v_h_288_, lean_object* v_eq3b_289_){
_start:
{
lean_inc(v_eq3b_289_);
return v_eq3b_289_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eq3b_elim___boxed(lean_object* v_motive_290_, lean_object* v_t_291_, lean_object* v_h_292_, lean_object* v_eq3b_293_){
_start:
{
uint8_t v_t_boxed_294_; lean_object* v_res_295_; 
v_t_boxed_294_ = lean_unbox(v_t_291_);
v_res_295_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eq3b_elim(v_motive_290_, v_t_boxed_294_, v_h_292_, v_eq3b_293_);
lean_dec(v_eq3b_293_);
return v_res_295_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUni_elim___redArg(lean_object* v_eqSharpUni_296_){
_start:
{
lean_inc(v_eqSharpUni_296_);
return v_eqSharpUni_296_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUni_elim___redArg___boxed(lean_object* v_eqSharpUni_297_){
_start:
{
lean_object* v_res_298_; 
v_res_298_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUni_elim___redArg(v_eqSharpUni_297_);
lean_dec(v_eqSharpUni_297_);
return v_res_298_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUni_elim(lean_object* v_motive_299_, uint8_t v_t_300_, lean_object* v_h_301_, lean_object* v_eqSharpUni_302_){
_start:
{
lean_inc(v_eqSharpUni_302_);
return v_eqSharpUni_302_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUni_elim___boxed(lean_object* v_motive_303_, lean_object* v_t_304_, lean_object* v_h_305_, lean_object* v_eqSharpUni_306_){
_start:
{
uint8_t v_t_boxed_307_; lean_object* v_res_308_; 
v_t_boxed_307_ = lean_unbox(v_t_304_);
v_res_308_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUni_elim(v_motive_303_, v_t_boxed_307_, v_h_305_, v_eqSharpUni_306_);
lean_dec(v_eqSharpUni_306_);
return v_res_308_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUniReceiver_elim___redArg(lean_object* v_eqSharpUniReceiver_309_){
_start:
{
lean_inc(v_eqSharpUniReceiver_309_);
return v_eqSharpUniReceiver_309_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUniReceiver_elim___redArg___boxed(lean_object* v_eqSharpUniReceiver_310_){
_start:
{
lean_object* v_res_311_; 
v_res_311_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUniReceiver_elim___redArg(v_eqSharpUniReceiver_310_);
lean_dec(v_eqSharpUniReceiver_310_);
return v_res_311_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUniReceiver_elim(lean_object* v_motive_312_, uint8_t v_t_313_, lean_object* v_h_314_, lean_object* v_eqSharpUniReceiver_315_){
_start:
{
lean_inc(v_eqSharpUniReceiver_315_);
return v_eqSharpUniReceiver_315_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUniReceiver_elim___boxed(lean_object* v_motive_316_, lean_object* v_t_317_, lean_object* v_h_318_, lean_object* v_eqSharpUniReceiver_319_){
_start:
{
uint8_t v_t_boxed_320_; lean_object* v_res_321_; 
v_t_boxed_320_ = lean_unbox(v_t_317_);
v_res_321_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqSharpUniReceiver_elim(v_motive_316_, v_t_boxed_320_, v_h_318_, v_eqSharpUniReceiver_319_);
lean_dec(v_eqSharpUniReceiver_319_);
return v_res_321_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqUni_elim___redArg(lean_object* v_eqUni_322_){
_start:
{
lean_inc(v_eqUni_322_);
return v_eqUni_322_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqUni_elim___redArg___boxed(lean_object* v_eqUni_323_){
_start:
{
lean_object* v_res_324_; 
v_res_324_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqUni_elim___redArg(v_eqUni_323_);
lean_dec(v_eqUni_323_);
return v_res_324_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqUni_elim(lean_object* v_motive_325_, uint8_t v_t_326_, lean_object* v_h_327_, lean_object* v_eqUni_328_){
_start:
{
lean_inc(v_eqUni_328_);
return v_eqUni_328_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqUni_elim___boxed(lean_object* v_motive_329_, lean_object* v_t_330_, lean_object* v_h_331_, lean_object* v_eqUni_332_){
_start:
{
uint8_t v_t_boxed_333_; lean_object* v_res_334_; 
v_t_boxed_333_ = lean_unbox(v_t_330_);
v_res_334_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqUni_elim(v_motive_329_, v_t_boxed_333_, v_h_331_, v_eqUni_332_);
lean_dec(v_eqUni_332_);
return v_res_334_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expressionClaim_elim___redArg(lean_object* v_expressionClaim_335_){
_start:
{
lean_inc(v_expressionClaim_335_);
return v_expressionClaim_335_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expressionClaim_elim___redArg___boxed(lean_object* v_expressionClaim_336_){
_start:
{
lean_object* v_res_337_; 
v_res_337_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expressionClaim_elim___redArg(v_expressionClaim_336_);
lean_dec(v_expressionClaim_336_);
return v_res_337_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expressionClaim_elim(lean_object* v_motive_338_, uint8_t v_t_339_, lean_object* v_h_340_, lean_object* v_expressionClaim_341_){
_start:
{
lean_inc(v_expressionClaim_341_);
return v_expressionClaim_341_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expressionClaim_elim___boxed(lean_object* v_motive_342_, lean_object* v_t_343_, lean_object* v_h_344_, lean_object* v_expressionClaim_345_){
_start:
{
uint8_t v_t_boxed_346_; lean_object* v_res_347_; 
v_t_boxed_346_ = lean_unbox(v_t_343_);
v_res_347_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expressionClaim_elim(v_motive_342_, v_t_boxed_346_, v_h_344_, v_expressionClaim_345_);
lean_dec(v_expressionClaim_345_);
return v_res_347_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_interactionsFolding_elim___redArg(lean_object* v_interactionsFolding_348_){
_start:
{
lean_inc(v_interactionsFolding_348_);
return v_interactionsFolding_348_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_interactionsFolding_elim___redArg___boxed(lean_object* v_interactionsFolding_349_){
_start:
{
lean_object* v_res_350_; 
v_res_350_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_interactionsFolding_elim___redArg(v_interactionsFolding_349_);
lean_dec(v_interactionsFolding_349_);
return v_res_350_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_interactionsFolding_elim(lean_object* v_motive_351_, uint8_t v_t_352_, lean_object* v_h_353_, lean_object* v_interactionsFolding_354_){
_start:
{
lean_inc(v_interactionsFolding_354_);
return v_interactionsFolding_354_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_interactionsFolding_elim___boxed(lean_object* v_motive_355_, lean_object* v_t_356_, lean_object* v_h_357_, lean_object* v_interactionsFolding_358_){
_start:
{
uint8_t v_t_boxed_359_; lean_object* v_res_360_; 
v_t_boxed_359_ = lean_unbox(v_t_356_);
v_res_360_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_interactionsFolding_elim(v_motive_355_, v_t_boxed_359_, v_h_357_, v_interactionsFolding_358_);
lean_dec(v_interactionsFolding_358_);
return v_res_360_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_constraintsFolding_elim___redArg(lean_object* v_constraintsFolding_361_){
_start:
{
lean_inc(v_constraintsFolding_361_);
return v_constraintsFolding_361_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_constraintsFolding_elim___redArg___boxed(lean_object* v_constraintsFolding_362_){
_start:
{
lean_object* v_res_363_; 
v_res_363_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_constraintsFolding_elim___redArg(v_constraintsFolding_362_);
lean_dec(v_constraintsFolding_362_);
return v_res_363_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_constraintsFolding_elim(lean_object* v_motive_364_, uint8_t v_t_365_, lean_object* v_h_366_, lean_object* v_constraintsFolding_367_){
_start:
{
lean_inc(v_constraintsFolding_367_);
return v_constraintsFolding_367_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_constraintsFolding_elim___boxed(lean_object* v_motive_368_, lean_object* v_t_369_, lean_object* v_h_370_, lean_object* v_constraintsFolding_371_){
_start:
{
uint8_t v_t_boxed_372_; lean_object* v_res_373_; 
v_t_boxed_372_ = lean_unbox(v_t_369_);
v_res_373_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_constraintsFolding_elim(v_motive_368_, v_t_boxed_372_, v_h_370_, v_constraintsFolding_371_);
lean_dec(v_constraintsFolding_371_);
return v_res_373_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNeg_elim___redArg(lean_object* v_eqNeg_374_){
_start:
{
lean_inc(v_eqNeg_374_);
return v_eqNeg_374_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNeg_elim___redArg___boxed(lean_object* v_eqNeg_375_){
_start:
{
lean_object* v_res_376_; 
v_res_376_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNeg_elim___redArg(v_eqNeg_375_);
lean_dec(v_eqNeg_375_);
return v_res_376_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNeg_elim(lean_object* v_motive_377_, uint8_t v_t_378_, lean_object* v_h_379_, lean_object* v_eqNeg_380_){
_start:
{
lean_inc(v_eqNeg_380_);
return v_eqNeg_380_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNeg_elim___boxed(lean_object* v_motive_381_, lean_object* v_t_382_, lean_object* v_h_383_, lean_object* v_eqNeg_384_){
_start:
{
uint8_t v_t_boxed_385_; lean_object* v_res_386_; 
v_t_boxed_385_ = lean_unbox(v_t_382_);
v_res_386_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqNeg_elim(v_motive_381_, v_t_boxed_385_, v_h_383_, v_eqNeg_384_);
lean_dec(v_eqNeg_384_);
return v_res_386_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_openingClaims_elim___redArg(lean_object* v_openingClaims_387_){
_start:
{
lean_inc(v_openingClaims_387_);
return v_openingClaims_387_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_openingClaims_elim___redArg___boxed(lean_object* v_openingClaims_388_){
_start:
{
lean_object* v_res_389_; 
v_res_389_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_openingClaims_elim___redArg(v_openingClaims_388_);
lean_dec(v_openingClaims_388_);
return v_res_389_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_openingClaims_elim(lean_object* v_motive_390_, uint8_t v_t_391_, lean_object* v_h_392_, lean_object* v_openingClaims_393_){
_start:
{
lean_inc(v_openingClaims_393_);
return v_openingClaims_393_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_openingClaims_elim___boxed(lean_object* v_motive_394_, lean_object* v_t_395_, lean_object* v_h_396_, lean_object* v_openingClaims_397_){
_start:
{
uint8_t v_t_boxed_398_; lean_object* v_res_399_; 
v_t_boxed_398_ = lean_unbox(v_t_395_);
v_res_399_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_openingClaims_elim(v_motive_394_, v_t_boxed_398_, v_h_396_, v_openingClaims_397_);
lean_dec(v_openingClaims_397_);
return v_res_399_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateRound_elim___redArg(lean_object* v_univariateRound_400_){
_start:
{
lean_inc(v_univariateRound_400_);
return v_univariateRound_400_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateRound_elim___redArg___boxed(lean_object* v_univariateRound_401_){
_start:
{
lean_object* v_res_402_; 
v_res_402_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateRound_elim___redArg(v_univariateRound_401_);
lean_dec(v_univariateRound_401_);
return v_res_402_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateRound_elim(lean_object* v_motive_403_, uint8_t v_t_404_, lean_object* v_h_405_, lean_object* v_univariateRound_406_){
_start:
{
lean_inc(v_univariateRound_406_);
return v_univariateRound_406_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateRound_elim___boxed(lean_object* v_motive_407_, lean_object* v_t_408_, lean_object* v_h_409_, lean_object* v_univariateRound_410_){
_start:
{
uint8_t v_t_boxed_411_; lean_object* v_res_412_; 
v_t_boxed_411_ = lean_unbox(v_t_408_);
v_res_412_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_univariateRound_elim(v_motive_407_, v_t_boxed_411_, v_h_409_, v_univariateRound_410_);
lean_dec(v_univariateRound_410_);
return v_res_412_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_sumcheckRounds_elim___redArg(lean_object* v_sumcheckRounds_413_){
_start:
{
lean_inc(v_sumcheckRounds_413_);
return v_sumcheckRounds_413_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_sumcheckRounds_elim___redArg___boxed(lean_object* v_sumcheckRounds_414_){
_start:
{
lean_object* v_res_415_; 
v_res_415_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_sumcheckRounds_elim___redArg(v_sumcheckRounds_414_);
lean_dec(v_sumcheckRounds_414_);
return v_res_415_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_sumcheckRounds_elim(lean_object* v_motive_416_, uint8_t v_t_417_, lean_object* v_h_418_, lean_object* v_sumcheckRounds_419_){
_start:
{
lean_inc(v_sumcheckRounds_419_);
return v_sumcheckRounds_419_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_sumcheckRounds_elim___boxed(lean_object* v_motive_420_, lean_object* v_t_421_, lean_object* v_h_422_, lean_object* v_sumcheckRounds_423_){
_start:
{
uint8_t v_t_boxed_424_; lean_object* v_res_425_; 
v_t_boxed_424_ = lean_unbox(v_t_421_);
v_res_425_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_sumcheckRounds_elim(v_motive_420_, v_t_boxed_424_, v_h_422_, v_sumcheckRounds_423_);
lean_dec(v_sumcheckRounds_423_);
return v_res_425_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_stackingClaims_elim___redArg(lean_object* v_stackingClaims_426_){
_start:
{
lean_inc(v_stackingClaims_426_);
return v_stackingClaims_426_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_stackingClaims_elim___redArg___boxed(lean_object* v_stackingClaims_427_){
_start:
{
lean_object* v_res_428_; 
v_res_428_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_stackingClaims_elim___redArg(v_stackingClaims_427_);
lean_dec(v_stackingClaims_427_);
return v_res_428_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_stackingClaims_elim(lean_object* v_motive_429_, uint8_t v_t_430_, lean_object* v_h_431_, lean_object* v_stackingClaims_432_){
_start:
{
lean_inc(v_stackingClaims_432_);
return v_stackingClaims_432_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_stackingClaims_elim___boxed(lean_object* v_motive_433_, lean_object* v_t_434_, lean_object* v_h_435_, lean_object* v_stackingClaims_436_){
_start:
{
uint8_t v_t_boxed_437_; lean_object* v_res_438_; 
v_t_boxed_437_ = lean_unbox(v_t_434_);
v_res_438_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_stackingClaims_elim(v_motive_433_, v_t_boxed_437_, v_h_435_, v_stackingClaims_436_);
lean_dec(v_stackingClaims_436_);
return v_res_438_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBase_elim___redArg(lean_object* v_eqBase_439_){
_start:
{
lean_inc(v_eqBase_439_);
return v_eqBase_439_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBase_elim___redArg___boxed(lean_object* v_eqBase_440_){
_start:
{
lean_object* v_res_441_; 
v_res_441_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBase_elim___redArg(v_eqBase_440_);
lean_dec(v_eqBase_440_);
return v_res_441_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBase_elim(lean_object* v_motive_442_, uint8_t v_t_443_, lean_object* v_h_444_, lean_object* v_eqBase_445_){
_start:
{
lean_inc(v_eqBase_445_);
return v_eqBase_445_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBase_elim___boxed(lean_object* v_motive_446_, lean_object* v_t_447_, lean_object* v_h_448_, lean_object* v_eqBase_449_){
_start:
{
uint8_t v_t_boxed_450_; lean_object* v_res_451_; 
v_t_boxed_450_ = lean_unbox(v_t_447_);
v_res_451_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBase_elim(v_motive_446_, v_t_boxed_450_, v_h_448_, v_eqBase_449_);
lean_dec(v_eqBase_449_);
return v_res_451_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBits_elim___redArg(lean_object* v_eqBits_452_){
_start:
{
lean_inc(v_eqBits_452_);
return v_eqBits_452_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBits_elim___redArg___boxed(lean_object* v_eqBits_453_){
_start:
{
lean_object* v_res_454_; 
v_res_454_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBits_elim___redArg(v_eqBits_453_);
lean_dec(v_eqBits_453_);
return v_res_454_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBits_elim(lean_object* v_motive_455_, uint8_t v_t_456_, lean_object* v_h_457_, lean_object* v_eqBits_458_){
_start:
{
lean_inc(v_eqBits_458_);
return v_eqBits_458_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBits_elim___boxed(lean_object* v_motive_459_, lean_object* v_t_460_, lean_object* v_h_461_, lean_object* v_eqBits_462_){
_start:
{
uint8_t v_t_boxed_463_; lean_object* v_res_464_; 
v_t_boxed_463_ = lean_unbox(v_t_460_);
v_res_464_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_eqBits_elim(v_motive_459_, v_t_boxed_463_, v_h_461_, v_eqBits_462_);
lean_dec(v_eqBits_462_);
return v_res_464_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirRound_elim___redArg(lean_object* v_whirRound_465_){
_start:
{
lean_inc(v_whirRound_465_);
return v_whirRound_465_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirRound_elim___redArg___boxed(lean_object* v_whirRound_466_){
_start:
{
lean_object* v_res_467_; 
v_res_467_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirRound_elim___redArg(v_whirRound_466_);
lean_dec(v_whirRound_466_);
return v_res_467_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirRound_elim(lean_object* v_motive_468_, uint8_t v_t_469_, lean_object* v_h_470_, lean_object* v_whirRound_471_){
_start:
{
lean_inc(v_whirRound_471_);
return v_whirRound_471_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirRound_elim___boxed(lean_object* v_motive_472_, lean_object* v_t_473_, lean_object* v_h_474_, lean_object* v_whirRound_475_){
_start:
{
uint8_t v_t_boxed_476_; lean_object* v_res_477_; 
v_t_boxed_476_ = lean_unbox(v_t_473_);
v_res_477_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirRound_elim(v_motive_472_, v_t_boxed_476_, v_h_474_, v_whirRound_475_);
lean_dec(v_whirRound_475_);
return v_res_477_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirSumcheck_elim___redArg(lean_object* v_whirSumcheck_478_){
_start:
{
lean_inc(v_whirSumcheck_478_);
return v_whirSumcheck_478_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirSumcheck_elim___redArg___boxed(lean_object* v_whirSumcheck_479_){
_start:
{
lean_object* v_res_480_; 
v_res_480_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirSumcheck_elim___redArg(v_whirSumcheck_479_);
lean_dec(v_whirSumcheck_479_);
return v_res_480_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirSumcheck_elim(lean_object* v_motive_481_, uint8_t v_t_482_, lean_object* v_h_483_, lean_object* v_whirSumcheck_484_){
_start:
{
lean_inc(v_whirSumcheck_484_);
return v_whirSumcheck_484_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirSumcheck_elim___boxed(lean_object* v_motive_485_, lean_object* v_t_486_, lean_object* v_h_487_, lean_object* v_whirSumcheck_488_){
_start:
{
uint8_t v_t_boxed_489_; lean_object* v_res_490_; 
v_t_boxed_489_ = lean_unbox(v_t_486_);
v_res_490_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirSumcheck_elim(v_motive_485_, v_t_boxed_489_, v_h_487_, v_whirSumcheck_488_);
lean_dec(v_whirSumcheck_488_);
return v_res_490_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirQuery_elim___redArg(lean_object* v_whirQuery_491_){
_start:
{
lean_inc(v_whirQuery_491_);
return v_whirQuery_491_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirQuery_elim___redArg___boxed(lean_object* v_whirQuery_492_){
_start:
{
lean_object* v_res_493_; 
v_res_493_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirQuery_elim___redArg(v_whirQuery_492_);
lean_dec(v_whirQuery_492_);
return v_res_493_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirQuery_elim(lean_object* v_motive_494_, uint8_t v_t_495_, lean_object* v_h_496_, lean_object* v_whirQuery_497_){
_start:
{
lean_inc(v_whirQuery_497_);
return v_whirQuery_497_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirQuery_elim___boxed(lean_object* v_motive_498_, lean_object* v_t_499_, lean_object* v_h_500_, lean_object* v_whirQuery_501_){
_start:
{
uint8_t v_t_boxed_502_; lean_object* v_res_503_; 
v_t_boxed_502_ = lean_unbox(v_t_499_);
v_res_503_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirQuery_elim(v_motive_498_, v_t_boxed_502_, v_h_500_, v_whirQuery_501_);
lean_dec(v_whirQuery_501_);
return v_res_503_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_initialOpenedValues_elim___redArg(lean_object* v_initialOpenedValues_504_){
_start:
{
lean_inc(v_initialOpenedValues_504_);
return v_initialOpenedValues_504_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_initialOpenedValues_elim___redArg___boxed(lean_object* v_initialOpenedValues_505_){
_start:
{
lean_object* v_res_506_; 
v_res_506_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_initialOpenedValues_elim___redArg(v_initialOpenedValues_505_);
lean_dec(v_initialOpenedValues_505_);
return v_res_506_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_initialOpenedValues_elim(lean_object* v_motive_507_, uint8_t v_t_508_, lean_object* v_h_509_, lean_object* v_initialOpenedValues_510_){
_start:
{
lean_inc(v_initialOpenedValues_510_);
return v_initialOpenedValues_510_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_initialOpenedValues_elim___boxed(lean_object* v_motive_511_, lean_object* v_t_512_, lean_object* v_h_513_, lean_object* v_initialOpenedValues_514_){
_start:
{
uint8_t v_t_boxed_515_; lean_object* v_res_516_; 
v_t_boxed_515_ = lean_unbox(v_t_512_);
v_res_516_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_initialOpenedValues_elim(v_motive_511_, v_t_boxed_515_, v_h_513_, v_initialOpenedValues_514_);
lean_dec(v_initialOpenedValues_514_);
return v_res_516_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_nonInitialOpenedValues_elim___redArg(lean_object* v_nonInitialOpenedValues_517_){
_start:
{
lean_inc(v_nonInitialOpenedValues_517_);
return v_nonInitialOpenedValues_517_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_nonInitialOpenedValues_elim___redArg___boxed(lean_object* v_nonInitialOpenedValues_518_){
_start:
{
lean_object* v_res_519_; 
v_res_519_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_nonInitialOpenedValues_elim___redArg(v_nonInitialOpenedValues_518_);
lean_dec(v_nonInitialOpenedValues_518_);
return v_res_519_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_nonInitialOpenedValues_elim(lean_object* v_motive_520_, uint8_t v_t_521_, lean_object* v_h_522_, lean_object* v_nonInitialOpenedValues_523_){
_start:
{
lean_inc(v_nonInitialOpenedValues_523_);
return v_nonInitialOpenedValues_523_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_nonInitialOpenedValues_elim___boxed(lean_object* v_motive_524_, lean_object* v_t_525_, lean_object* v_h_526_, lean_object* v_nonInitialOpenedValues_527_){
_start:
{
uint8_t v_t_boxed_528_; lean_object* v_res_529_; 
v_t_boxed_528_ = lean_unbox(v_t_525_);
v_res_529_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_nonInitialOpenedValues_elim(v_motive_524_, v_t_boxed_528_, v_h_526_, v_nonInitialOpenedValues_527_);
lean_dec(v_nonInitialOpenedValues_527_);
return v_res_529_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirFolding_elim___redArg(lean_object* v_whirFolding_530_){
_start:
{
lean_inc(v_whirFolding_530_);
return v_whirFolding_530_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirFolding_elim___redArg___boxed(lean_object* v_whirFolding_531_){
_start:
{
lean_object* v_res_532_; 
v_res_532_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirFolding_elim___redArg(v_whirFolding_531_);
lean_dec(v_whirFolding_531_);
return v_res_532_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirFolding_elim(lean_object* v_motive_533_, uint8_t v_t_534_, lean_object* v_h_535_, lean_object* v_whirFolding_536_){
_start:
{
lean_inc(v_whirFolding_536_);
return v_whirFolding_536_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirFolding_elim___boxed(lean_object* v_motive_537_, lean_object* v_t_538_, lean_object* v_h_539_, lean_object* v_whirFolding_540_){
_start:
{
uint8_t v_t_boxed_541_; lean_object* v_res_542_; 
v_t_boxed_541_ = lean_unbox(v_t_538_);
v_res_542_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_whirFolding_elim(v_motive_537_, v_t_boxed_541_, v_h_539_, v_whirFolding_540_);
lean_dec(v_whirFolding_540_);
return v_res_542_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyMleEval_elim___redArg(lean_object* v_finalPolyMleEval_543_){
_start:
{
lean_inc(v_finalPolyMleEval_543_);
return v_finalPolyMleEval_543_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyMleEval_elim___redArg___boxed(lean_object* v_finalPolyMleEval_544_){
_start:
{
lean_object* v_res_545_; 
v_res_545_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyMleEval_elim___redArg(v_finalPolyMleEval_544_);
lean_dec(v_finalPolyMleEval_544_);
return v_res_545_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyMleEval_elim(lean_object* v_motive_546_, uint8_t v_t_547_, lean_object* v_h_548_, lean_object* v_finalPolyMleEval_549_){
_start:
{
lean_inc(v_finalPolyMleEval_549_);
return v_finalPolyMleEval_549_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyMleEval_elim___boxed(lean_object* v_motive_550_, lean_object* v_t_551_, lean_object* v_h_552_, lean_object* v_finalPolyMleEval_553_){
_start:
{
uint8_t v_t_boxed_554_; lean_object* v_res_555_; 
v_t_boxed_554_ = lean_unbox(v_t_551_);
v_res_555_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyMleEval_elim(v_motive_550_, v_t_boxed_554_, v_h_552_, v_finalPolyMleEval_553_);
lean_dec(v_finalPolyMleEval_553_);
return v_res_555_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyQueryEval_elim___redArg(lean_object* v_finalPolyQueryEval_556_){
_start:
{
lean_inc(v_finalPolyQueryEval_556_);
return v_finalPolyQueryEval_556_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyQueryEval_elim___redArg___boxed(lean_object* v_finalPolyQueryEval_557_){
_start:
{
lean_object* v_res_558_; 
v_res_558_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyQueryEval_elim___redArg(v_finalPolyQueryEval_557_);
lean_dec(v_finalPolyQueryEval_557_);
return v_res_558_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyQueryEval_elim(lean_object* v_motive_559_, uint8_t v_t_560_, lean_object* v_h_561_, lean_object* v_finalPolyQueryEval_562_){
_start:
{
lean_inc(v_finalPolyQueryEval_562_);
return v_finalPolyQueryEval_562_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyQueryEval_elim___boxed(lean_object* v_motive_563_, lean_object* v_t_564_, lean_object* v_h_565_, lean_object* v_finalPolyQueryEval_566_){
_start:
{
uint8_t v_t_boxed_567_; lean_object* v_res_568_; 
v_t_boxed_567_ = lean_unbox(v_t_564_);
v_res_568_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_finalPolyQueryEval_elim(v_motive_563_, v_t_boxed_567_, v_h_565_, v_finalPolyQueryEval_566_);
lean_dec(v_finalPolyQueryEval_566_);
return v_res_568_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_rangeChecker_elim___redArg(lean_object* v_rangeChecker_569_){
_start:
{
lean_inc(v_rangeChecker_569_);
return v_rangeChecker_569_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_rangeChecker_elim___redArg___boxed(lean_object* v_rangeChecker_570_){
_start:
{
lean_object* v_res_571_; 
v_res_571_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_rangeChecker_elim___redArg(v_rangeChecker_570_);
lean_dec(v_rangeChecker_570_);
return v_res_571_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_rangeChecker_elim(lean_object* v_motive_572_, uint8_t v_t_573_, lean_object* v_h_574_, lean_object* v_rangeChecker_575_){
_start:
{
lean_inc(v_rangeChecker_575_);
return v_rangeChecker_575_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_rangeChecker_elim___boxed(lean_object* v_motive_576_, lean_object* v_t_577_, lean_object* v_h_578_, lean_object* v_rangeChecker_579_){
_start:
{
uint8_t v_t_boxed_580_; lean_object* v_res_581_; 
v_t_boxed_580_ = lean_unbox(v_t_577_);
v_res_581_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_rangeChecker_elim(v_motive_576_, v_t_boxed_580_, v_h_578_, v_rangeChecker_579_);
lean_dec(v_rangeChecker_579_);
return v_res_581_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_powerChecker_elim___redArg(lean_object* v_powerChecker_582_){
_start:
{
lean_inc(v_powerChecker_582_);
return v_powerChecker_582_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_powerChecker_elim___redArg___boxed(lean_object* v_powerChecker_583_){
_start:
{
lean_object* v_res_584_; 
v_res_584_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_powerChecker_elim___redArg(v_powerChecker_583_);
lean_dec(v_powerChecker_583_);
return v_res_584_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_powerChecker_elim(lean_object* v_motive_585_, uint8_t v_t_586_, lean_object* v_h_587_, lean_object* v_powerChecker_588_){
_start:
{
lean_inc(v_powerChecker_588_);
return v_powerChecker_588_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_powerChecker_elim___boxed(lean_object* v_motive_589_, lean_object* v_t_590_, lean_object* v_h_591_, lean_object* v_powerChecker_592_){
_start:
{
uint8_t v_t_boxed_593_; lean_object* v_res_594_; 
v_t_boxed_593_ = lean_unbox(v_t_590_);
v_res_594_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_powerChecker_elim(v_motive_589_, v_t_boxed_593_, v_h_591_, v_powerChecker_592_);
lean_dec(v_powerChecker_592_);
return v_res_594_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expBitsLen_elim___redArg(lean_object* v_expBitsLen_595_){
_start:
{
lean_inc(v_expBitsLen_595_);
return v_expBitsLen_595_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expBitsLen_elim___redArg___boxed(lean_object* v_expBitsLen_596_){
_start:
{
lean_object* v_res_597_; 
v_res_597_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expBitsLen_elim___redArg(v_expBitsLen_596_);
lean_dec(v_expBitsLen_596_);
return v_res_597_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expBitsLen_elim(lean_object* v_motive_598_, uint8_t v_t_599_, lean_object* v_h_600_, lean_object* v_expBitsLen_601_){
_start:
{
lean_inc(v_expBitsLen_601_);
return v_expBitsLen_601_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expBitsLen_elim___boxed(lean_object* v_motive_602_, lean_object* v_t_603_, lean_object* v_h_604_, lean_object* v_expBitsLen_605_){
_start:
{
uint8_t v_t_boxed_606_; lean_object* v_res_607_; 
v_t_boxed_606_ = lean_unbox(v_t_603_);
v_res_607_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_expBitsLen_elim(v_motive_602_, v_t_boxed_606_, v_h_604_, v_expBitsLen_605_);
lean_dec(v_expBitsLen_605_);
return v_res_607_;
}
}
LEAN_EXPORT uint8_t lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ofNat(lean_object* v_n_608_){
_start:
{
lean_object* v___x_609_; uint8_t v___x_610_; 
v___x_609_ = lean_unsigned_to_nat(20u);
v___x_610_ = lean_nat_dec_le(v_n_608_, v___x_609_);
if (v___x_610_ == 0)
{
lean_object* v___x_611_; uint8_t v___x_612_; 
v___x_611_ = lean_unsigned_to_nat(30u);
v___x_612_ = lean_nat_dec_le(v_n_608_, v___x_611_);
if (v___x_612_ == 0)
{
lean_object* v___x_613_; uint8_t v___x_614_; 
v___x_613_ = lean_unsigned_to_nat(35u);
v___x_614_ = lean_nat_dec_le(v_n_608_, v___x_613_);
if (v___x_614_ == 0)
{
lean_object* v___x_615_; uint8_t v___x_616_; 
v___x_615_ = lean_unsigned_to_nat(38u);
v___x_616_ = lean_nat_dec_le(v_n_608_, v___x_615_);
if (v___x_616_ == 0)
{
lean_object* v___x_617_; uint8_t v___x_618_; 
v___x_617_ = lean_unsigned_to_nat(39u);
v___x_618_ = lean_nat_dec_le(v_n_608_, v___x_617_);
if (v___x_618_ == 0)
{
lean_object* v___x_619_; uint8_t v___x_620_; 
v___x_619_ = lean_unsigned_to_nat(40u);
v___x_620_ = lean_nat_dec_le(v_n_608_, v___x_619_);
if (v___x_620_ == 0)
{
uint8_t v___x_621_; 
v___x_621_ = 41;
return v___x_621_;
}
else
{
uint8_t v___x_622_; 
v___x_622_ = 40;
return v___x_622_;
}
}
else
{
uint8_t v___x_623_; 
v___x_623_ = 39;
return v___x_623_;
}
}
else
{
lean_object* v___x_624_; uint8_t v___x_625_; 
v___x_624_ = lean_unsigned_to_nat(36u);
v___x_625_ = lean_nat_dec_le(v_n_608_, v___x_624_);
if (v___x_625_ == 0)
{
lean_object* v___x_626_; uint8_t v___x_627_; 
v___x_626_ = lean_unsigned_to_nat(37u);
v___x_627_ = lean_nat_dec_le(v_n_608_, v___x_626_);
if (v___x_627_ == 0)
{
uint8_t v___x_628_; 
v___x_628_ = 38;
return v___x_628_;
}
else
{
uint8_t v___x_629_; 
v___x_629_ = 37;
return v___x_629_;
}
}
else
{
uint8_t v___x_630_; 
v___x_630_ = 36;
return v___x_630_;
}
}
}
else
{
lean_object* v___x_631_; uint8_t v___x_632_; 
v___x_631_ = lean_unsigned_to_nat(32u);
v___x_632_ = lean_nat_dec_le(v_n_608_, v___x_631_);
if (v___x_632_ == 0)
{
lean_object* v___x_633_; uint8_t v___x_634_; 
v___x_633_ = lean_unsigned_to_nat(33u);
v___x_634_ = lean_nat_dec_le(v_n_608_, v___x_633_);
if (v___x_634_ == 0)
{
lean_object* v___x_635_; uint8_t v___x_636_; 
v___x_635_ = lean_unsigned_to_nat(34u);
v___x_636_ = lean_nat_dec_le(v_n_608_, v___x_635_);
if (v___x_636_ == 0)
{
uint8_t v___x_637_; 
v___x_637_ = 35;
return v___x_637_;
}
else
{
uint8_t v___x_638_; 
v___x_638_ = 34;
return v___x_638_;
}
}
else
{
uint8_t v___x_639_; 
v___x_639_ = 33;
return v___x_639_;
}
}
else
{
lean_object* v___x_640_; uint8_t v___x_641_; 
v___x_640_ = lean_unsigned_to_nat(31u);
v___x_641_ = lean_nat_dec_le(v_n_608_, v___x_640_);
if (v___x_641_ == 0)
{
uint8_t v___x_642_; 
v___x_642_ = 32;
return v___x_642_;
}
else
{
uint8_t v___x_643_; 
v___x_643_ = 31;
return v___x_643_;
}
}
}
}
else
{
lean_object* v___x_644_; uint8_t v___x_645_; 
v___x_644_ = lean_unsigned_to_nat(25u);
v___x_645_ = lean_nat_dec_le(v_n_608_, v___x_644_);
if (v___x_645_ == 0)
{
lean_object* v___x_646_; uint8_t v___x_647_; 
v___x_646_ = lean_unsigned_to_nat(27u);
v___x_647_ = lean_nat_dec_le(v_n_608_, v___x_646_);
if (v___x_647_ == 0)
{
lean_object* v___x_648_; uint8_t v___x_649_; 
v___x_648_ = lean_unsigned_to_nat(28u);
v___x_649_ = lean_nat_dec_le(v_n_608_, v___x_648_);
if (v___x_649_ == 0)
{
lean_object* v___x_650_; uint8_t v___x_651_; 
v___x_650_ = lean_unsigned_to_nat(29u);
v___x_651_ = lean_nat_dec_le(v_n_608_, v___x_650_);
if (v___x_651_ == 0)
{
uint8_t v___x_652_; 
v___x_652_ = 30;
return v___x_652_;
}
else
{
uint8_t v___x_653_; 
v___x_653_ = 29;
return v___x_653_;
}
}
else
{
uint8_t v___x_654_; 
v___x_654_ = 28;
return v___x_654_;
}
}
else
{
lean_object* v___x_655_; uint8_t v___x_656_; 
v___x_655_ = lean_unsigned_to_nat(26u);
v___x_656_ = lean_nat_dec_le(v_n_608_, v___x_655_);
if (v___x_656_ == 0)
{
uint8_t v___x_657_; 
v___x_657_ = 27;
return v___x_657_;
}
else
{
uint8_t v___x_658_; 
v___x_658_ = 26;
return v___x_658_;
}
}
}
else
{
lean_object* v___x_659_; uint8_t v___x_660_; 
v___x_659_ = lean_unsigned_to_nat(22u);
v___x_660_ = lean_nat_dec_le(v_n_608_, v___x_659_);
if (v___x_660_ == 0)
{
lean_object* v___x_661_; uint8_t v___x_662_; 
v___x_661_ = lean_unsigned_to_nat(23u);
v___x_662_ = lean_nat_dec_le(v_n_608_, v___x_661_);
if (v___x_662_ == 0)
{
lean_object* v___x_663_; uint8_t v___x_664_; 
v___x_663_ = lean_unsigned_to_nat(24u);
v___x_664_ = lean_nat_dec_le(v_n_608_, v___x_663_);
if (v___x_664_ == 0)
{
uint8_t v___x_665_; 
v___x_665_ = 25;
return v___x_665_;
}
else
{
uint8_t v___x_666_; 
v___x_666_ = 24;
return v___x_666_;
}
}
else
{
uint8_t v___x_667_; 
v___x_667_ = 23;
return v___x_667_;
}
}
else
{
lean_object* v___x_668_; uint8_t v___x_669_; 
v___x_668_ = lean_unsigned_to_nat(21u);
v___x_669_ = lean_nat_dec_le(v_n_608_, v___x_668_);
if (v___x_669_ == 0)
{
uint8_t v___x_670_; 
v___x_670_ = 22;
return v___x_670_;
}
else
{
uint8_t v___x_671_; 
v___x_671_ = 21;
return v___x_671_;
}
}
}
}
}
else
{
lean_object* v___x_672_; uint8_t v___x_673_; 
v___x_672_ = lean_unsigned_to_nat(9u);
v___x_673_ = lean_nat_dec_le(v_n_608_, v___x_672_);
if (v___x_673_ == 0)
{
lean_object* v___x_674_; uint8_t v___x_675_; 
v___x_674_ = lean_unsigned_to_nat(14u);
v___x_675_ = lean_nat_dec_le(v_n_608_, v___x_674_);
if (v___x_675_ == 0)
{
lean_object* v___x_676_; uint8_t v___x_677_; 
v___x_676_ = lean_unsigned_to_nat(17u);
v___x_677_ = lean_nat_dec_le(v_n_608_, v___x_676_);
if (v___x_677_ == 0)
{
lean_object* v___x_678_; uint8_t v___x_679_; 
v___x_678_ = lean_unsigned_to_nat(18u);
v___x_679_ = lean_nat_dec_le(v_n_608_, v___x_678_);
if (v___x_679_ == 0)
{
lean_object* v___x_680_; uint8_t v___x_681_; 
v___x_680_ = lean_unsigned_to_nat(19u);
v___x_681_ = lean_nat_dec_le(v_n_608_, v___x_680_);
if (v___x_681_ == 0)
{
uint8_t v___x_682_; 
v___x_682_ = 20;
return v___x_682_;
}
else
{
uint8_t v___x_683_; 
v___x_683_ = 19;
return v___x_683_;
}
}
else
{
uint8_t v___x_684_; 
v___x_684_ = 18;
return v___x_684_;
}
}
else
{
lean_object* v___x_685_; uint8_t v___x_686_; 
v___x_685_ = lean_unsigned_to_nat(15u);
v___x_686_ = lean_nat_dec_le(v_n_608_, v___x_685_);
if (v___x_686_ == 0)
{
lean_object* v___x_687_; uint8_t v___x_688_; 
v___x_687_ = lean_unsigned_to_nat(16u);
v___x_688_ = lean_nat_dec_le(v_n_608_, v___x_687_);
if (v___x_688_ == 0)
{
uint8_t v___x_689_; 
v___x_689_ = 17;
return v___x_689_;
}
else
{
uint8_t v___x_690_; 
v___x_690_ = 16;
return v___x_690_;
}
}
else
{
uint8_t v___x_691_; 
v___x_691_ = 15;
return v___x_691_;
}
}
}
else
{
lean_object* v___x_692_; uint8_t v___x_693_; 
v___x_692_ = lean_unsigned_to_nat(11u);
v___x_693_ = lean_nat_dec_le(v_n_608_, v___x_692_);
if (v___x_693_ == 0)
{
lean_object* v___x_694_; uint8_t v___x_695_; 
v___x_694_ = lean_unsigned_to_nat(12u);
v___x_695_ = lean_nat_dec_le(v_n_608_, v___x_694_);
if (v___x_695_ == 0)
{
lean_object* v___x_696_; uint8_t v___x_697_; 
v___x_696_ = lean_unsigned_to_nat(13u);
v___x_697_ = lean_nat_dec_le(v_n_608_, v___x_696_);
if (v___x_697_ == 0)
{
uint8_t v___x_698_; 
v___x_698_ = 14;
return v___x_698_;
}
else
{
uint8_t v___x_699_; 
v___x_699_ = 13;
return v___x_699_;
}
}
else
{
uint8_t v___x_700_; 
v___x_700_ = 12;
return v___x_700_;
}
}
else
{
lean_object* v___x_701_; uint8_t v___x_702_; 
v___x_701_ = lean_unsigned_to_nat(10u);
v___x_702_ = lean_nat_dec_le(v_n_608_, v___x_701_);
if (v___x_702_ == 0)
{
uint8_t v___x_703_; 
v___x_703_ = 11;
return v___x_703_;
}
else
{
uint8_t v___x_704_; 
v___x_704_ = 10;
return v___x_704_;
}
}
}
}
else
{
lean_object* v___x_705_; uint8_t v___x_706_; 
v___x_705_ = lean_unsigned_to_nat(4u);
v___x_706_ = lean_nat_dec_le(v_n_608_, v___x_705_);
if (v___x_706_ == 0)
{
lean_object* v___x_707_; uint8_t v___x_708_; 
v___x_707_ = lean_unsigned_to_nat(6u);
v___x_708_ = lean_nat_dec_le(v_n_608_, v___x_707_);
if (v___x_708_ == 0)
{
lean_object* v___x_709_; uint8_t v___x_710_; 
v___x_709_ = lean_unsigned_to_nat(7u);
v___x_710_ = lean_nat_dec_le(v_n_608_, v___x_709_);
if (v___x_710_ == 0)
{
lean_object* v___x_711_; uint8_t v___x_712_; 
v___x_711_ = lean_unsigned_to_nat(8u);
v___x_712_ = lean_nat_dec_le(v_n_608_, v___x_711_);
if (v___x_712_ == 0)
{
uint8_t v___x_713_; 
v___x_713_ = 9;
return v___x_713_;
}
else
{
uint8_t v___x_714_; 
v___x_714_ = 8;
return v___x_714_;
}
}
else
{
uint8_t v___x_715_; 
v___x_715_ = 7;
return v___x_715_;
}
}
else
{
lean_object* v___x_716_; uint8_t v___x_717_; 
v___x_716_ = lean_unsigned_to_nat(5u);
v___x_717_ = lean_nat_dec_le(v_n_608_, v___x_716_);
if (v___x_717_ == 0)
{
uint8_t v___x_718_; 
v___x_718_ = 6;
return v___x_718_;
}
else
{
uint8_t v___x_719_; 
v___x_719_ = 5;
return v___x_719_;
}
}
}
else
{
lean_object* v___x_720_; uint8_t v___x_721_; 
v___x_720_ = lean_unsigned_to_nat(1u);
v___x_721_ = lean_nat_dec_le(v_n_608_, v___x_720_);
if (v___x_721_ == 0)
{
lean_object* v___x_722_; uint8_t v___x_723_; 
v___x_722_ = lean_unsigned_to_nat(2u);
v___x_723_ = lean_nat_dec_le(v_n_608_, v___x_722_);
if (v___x_723_ == 0)
{
lean_object* v___x_724_; uint8_t v___x_725_; 
v___x_724_ = lean_unsigned_to_nat(3u);
v___x_725_ = lean_nat_dec_le(v_n_608_, v___x_724_);
if (v___x_725_ == 0)
{
uint8_t v___x_726_; 
v___x_726_ = 4;
return v___x_726_;
}
else
{
uint8_t v___x_727_; 
v___x_727_ = 3;
return v___x_727_;
}
}
else
{
uint8_t v___x_728_; 
v___x_728_ = 2;
return v___x_728_;
}
}
else
{
lean_object* v___x_729_; uint8_t v___x_730_; 
v___x_729_ = lean_unsigned_to_nat(0u);
v___x_730_ = lean_nat_dec_le(v_n_608_, v___x_729_);
if (v___x_730_ == 0)
{
uint8_t v___x_731_; 
v___x_731_ = 1;
return v___x_731_;
}
else
{
uint8_t v___x_732_; 
v___x_732_ = 0;
return v___x_732_;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ofNat___boxed(lean_object* v_n_733_){
_start:
{
uint8_t v_res_734_; lean_object* v_r_735_; 
v_res_734_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ofNat(v_n_733_);
lean_dec(v_n_733_);
v_r_735_ = lean_box(v_res_734_);
return v_r_735_;
}
}
LEAN_EXPORT uint8_t lp_workspace_Recursion_Spec_VerifierCircuit_instDecidableEqVerifierAirId(uint8_t v_x_736_, uint8_t v_y_737_){
_start:
{
lean_object* v___x_738_; lean_object* v___x_739_; uint8_t v___x_740_; 
v___x_738_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorIdx(v_x_736_);
v___x_739_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorIdx(v_y_737_);
v___x_740_ = lean_nat_dec_eq(v___x_738_, v___x_739_);
lean_dec(v___x_739_);
lean_dec(v___x_738_);
return v___x_740_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instDecidableEqVerifierAirId___boxed(lean_object* v_x_741_, lean_object* v_y_742_){
_start:
{
uint8_t v_x_13__boxed_743_; uint8_t v_y_14__boxed_744_; uint8_t v_res_745_; lean_object* v_r_746_; 
v_x_13__boxed_743_ = lean_unbox(v_x_741_);
v_y_14__boxed_744_ = lean_unbox(v_y_742_);
v_res_745_ = lp_workspace_Recursion_Spec_VerifierCircuit_instDecidableEqVerifierAirId(v_x_13__boxed_743_, v_y_14__boxed_744_);
v_r_746_ = lean_box(v_res_745_);
return v_r_746_;
}
}
LEAN_EXPORT uint8_t lp_workspace_Recursion_Spec_VerifierCircuit_instBEqVerifierAirId_beq(uint8_t v_x_747_, uint8_t v_y_748_){
_start:
{
lean_object* v___x_749_; lean_object* v___x_750_; uint8_t v___x_751_; 
v___x_749_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorIdx(v_x_747_);
v___x_750_ = lp_workspace_Recursion_Spec_VerifierCircuit_VerifierAirId_ctorIdx(v_y_748_);
v___x_751_ = lean_nat_dec_eq(v___x_749_, v___x_750_);
lean_dec(v___x_750_);
lean_dec(v___x_749_);
return v___x_751_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instBEqVerifierAirId_beq___boxed(lean_object* v_x_752_, lean_object* v_y_753_){
_start:
{
uint8_t v_x_17__boxed_754_; uint8_t v_y_18__boxed_755_; uint8_t v_res_756_; lean_object* v_r_757_; 
v_x_17__boxed_754_ = lean_unbox(v_x_752_);
v_y_18__boxed_755_ = lean_unbox(v_y_753_);
v_res_756_ = lp_workspace_Recursion_Spec_VerifierCircuit_instBEqVerifierAirId_beq(v_x_17__boxed_754_, v_y_18__boxed_755_);
v_r_757_ = lean_box(v_res_756_);
return v_r_757_;
}
}
static lean_object* _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84(void){
_start:
{
lean_object* v___x_886_; lean_object* v___x_887_; 
v___x_886_ = lean_unsigned_to_nat(2u);
v___x_887_ = lean_nat_to_int(v___x_886_);
return v___x_887_;
}
}
static lean_object* _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85(void){
_start:
{
lean_object* v___x_888_; lean_object* v___x_889_; 
v___x_888_ = lean_unsigned_to_nat(1u);
v___x_889_ = lean_nat_to_int(v___x_888_);
return v___x_889_;
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr(uint8_t v_x_890_, lean_object* v_prec_891_){
_start:
{
lean_object* v___y_893_; lean_object* v___y_900_; lean_object* v___y_907_; lean_object* v___y_914_; lean_object* v___y_921_; lean_object* v___y_928_; lean_object* v___y_935_; lean_object* v___y_942_; lean_object* v___y_949_; lean_object* v___y_956_; lean_object* v___y_963_; lean_object* v___y_970_; lean_object* v___y_977_; lean_object* v___y_984_; lean_object* v___y_991_; lean_object* v___y_998_; lean_object* v___y_1005_; lean_object* v___y_1012_; lean_object* v___y_1019_; lean_object* v___y_1026_; lean_object* v___y_1033_; lean_object* v___y_1040_; lean_object* v___y_1047_; lean_object* v___y_1054_; lean_object* v___y_1061_; lean_object* v___y_1068_; lean_object* v___y_1075_; lean_object* v___y_1082_; lean_object* v___y_1089_; lean_object* v___y_1096_; lean_object* v___y_1103_; lean_object* v___y_1110_; lean_object* v___y_1117_; lean_object* v___y_1124_; lean_object* v___y_1131_; lean_object* v___y_1138_; lean_object* v___y_1145_; lean_object* v___y_1152_; lean_object* v___y_1159_; lean_object* v___y_1166_; lean_object* v___y_1173_; lean_object* v___y_1180_; 
switch(v_x_890_)
{
case 0:
{
lean_object* v___x_1186_; uint8_t v___x_1187_; 
v___x_1186_ = lean_unsigned_to_nat(1024u);
v___x_1187_ = lean_nat_dec_le(v___x_1186_, v_prec_891_);
if (v___x_1187_ == 0)
{
lean_object* v___x_1188_; 
v___x_1188_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_893_ = v___x_1188_;
goto v___jp_892_;
}
else
{
lean_object* v___x_1189_; 
v___x_1189_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_893_ = v___x_1189_;
goto v___jp_892_;
}
}
case 1:
{
lean_object* v___x_1190_; uint8_t v___x_1191_; 
v___x_1190_ = lean_unsigned_to_nat(1024u);
v___x_1191_ = lean_nat_dec_le(v___x_1190_, v_prec_891_);
if (v___x_1191_ == 0)
{
lean_object* v___x_1192_; 
v___x_1192_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_900_ = v___x_1192_;
goto v___jp_899_;
}
else
{
lean_object* v___x_1193_; 
v___x_1193_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_900_ = v___x_1193_;
goto v___jp_899_;
}
}
case 2:
{
lean_object* v___x_1194_; uint8_t v___x_1195_; 
v___x_1194_ = lean_unsigned_to_nat(1024u);
v___x_1195_ = lean_nat_dec_le(v___x_1194_, v_prec_891_);
if (v___x_1195_ == 0)
{
lean_object* v___x_1196_; 
v___x_1196_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_907_ = v___x_1196_;
goto v___jp_906_;
}
else
{
lean_object* v___x_1197_; 
v___x_1197_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_907_ = v___x_1197_;
goto v___jp_906_;
}
}
case 3:
{
lean_object* v___x_1198_; uint8_t v___x_1199_; 
v___x_1198_ = lean_unsigned_to_nat(1024u);
v___x_1199_ = lean_nat_dec_le(v___x_1198_, v_prec_891_);
if (v___x_1199_ == 0)
{
lean_object* v___x_1200_; 
v___x_1200_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_914_ = v___x_1200_;
goto v___jp_913_;
}
else
{
lean_object* v___x_1201_; 
v___x_1201_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_914_ = v___x_1201_;
goto v___jp_913_;
}
}
case 4:
{
lean_object* v___x_1202_; uint8_t v___x_1203_; 
v___x_1202_ = lean_unsigned_to_nat(1024u);
v___x_1203_ = lean_nat_dec_le(v___x_1202_, v_prec_891_);
if (v___x_1203_ == 0)
{
lean_object* v___x_1204_; 
v___x_1204_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_921_ = v___x_1204_;
goto v___jp_920_;
}
else
{
lean_object* v___x_1205_; 
v___x_1205_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_921_ = v___x_1205_;
goto v___jp_920_;
}
}
case 5:
{
lean_object* v___x_1206_; uint8_t v___x_1207_; 
v___x_1206_ = lean_unsigned_to_nat(1024u);
v___x_1207_ = lean_nat_dec_le(v___x_1206_, v_prec_891_);
if (v___x_1207_ == 0)
{
lean_object* v___x_1208_; 
v___x_1208_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_928_ = v___x_1208_;
goto v___jp_927_;
}
else
{
lean_object* v___x_1209_; 
v___x_1209_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_928_ = v___x_1209_;
goto v___jp_927_;
}
}
case 6:
{
lean_object* v___x_1210_; uint8_t v___x_1211_; 
v___x_1210_ = lean_unsigned_to_nat(1024u);
v___x_1211_ = lean_nat_dec_le(v___x_1210_, v_prec_891_);
if (v___x_1211_ == 0)
{
lean_object* v___x_1212_; 
v___x_1212_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_935_ = v___x_1212_;
goto v___jp_934_;
}
else
{
lean_object* v___x_1213_; 
v___x_1213_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_935_ = v___x_1213_;
goto v___jp_934_;
}
}
case 7:
{
lean_object* v___x_1214_; uint8_t v___x_1215_; 
v___x_1214_ = lean_unsigned_to_nat(1024u);
v___x_1215_ = lean_nat_dec_le(v___x_1214_, v_prec_891_);
if (v___x_1215_ == 0)
{
lean_object* v___x_1216_; 
v___x_1216_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_942_ = v___x_1216_;
goto v___jp_941_;
}
else
{
lean_object* v___x_1217_; 
v___x_1217_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_942_ = v___x_1217_;
goto v___jp_941_;
}
}
case 8:
{
lean_object* v___x_1218_; uint8_t v___x_1219_; 
v___x_1218_ = lean_unsigned_to_nat(1024u);
v___x_1219_ = lean_nat_dec_le(v___x_1218_, v_prec_891_);
if (v___x_1219_ == 0)
{
lean_object* v___x_1220_; 
v___x_1220_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_949_ = v___x_1220_;
goto v___jp_948_;
}
else
{
lean_object* v___x_1221_; 
v___x_1221_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_949_ = v___x_1221_;
goto v___jp_948_;
}
}
case 9:
{
lean_object* v___x_1222_; uint8_t v___x_1223_; 
v___x_1222_ = lean_unsigned_to_nat(1024u);
v___x_1223_ = lean_nat_dec_le(v___x_1222_, v_prec_891_);
if (v___x_1223_ == 0)
{
lean_object* v___x_1224_; 
v___x_1224_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_956_ = v___x_1224_;
goto v___jp_955_;
}
else
{
lean_object* v___x_1225_; 
v___x_1225_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_956_ = v___x_1225_;
goto v___jp_955_;
}
}
case 10:
{
lean_object* v___x_1226_; uint8_t v___x_1227_; 
v___x_1226_ = lean_unsigned_to_nat(1024u);
v___x_1227_ = lean_nat_dec_le(v___x_1226_, v_prec_891_);
if (v___x_1227_ == 0)
{
lean_object* v___x_1228_; 
v___x_1228_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_963_ = v___x_1228_;
goto v___jp_962_;
}
else
{
lean_object* v___x_1229_; 
v___x_1229_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_963_ = v___x_1229_;
goto v___jp_962_;
}
}
case 11:
{
lean_object* v___x_1230_; uint8_t v___x_1231_; 
v___x_1230_ = lean_unsigned_to_nat(1024u);
v___x_1231_ = lean_nat_dec_le(v___x_1230_, v_prec_891_);
if (v___x_1231_ == 0)
{
lean_object* v___x_1232_; 
v___x_1232_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_970_ = v___x_1232_;
goto v___jp_969_;
}
else
{
lean_object* v___x_1233_; 
v___x_1233_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_970_ = v___x_1233_;
goto v___jp_969_;
}
}
case 12:
{
lean_object* v___x_1234_; uint8_t v___x_1235_; 
v___x_1234_ = lean_unsigned_to_nat(1024u);
v___x_1235_ = lean_nat_dec_le(v___x_1234_, v_prec_891_);
if (v___x_1235_ == 0)
{
lean_object* v___x_1236_; 
v___x_1236_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_977_ = v___x_1236_;
goto v___jp_976_;
}
else
{
lean_object* v___x_1237_; 
v___x_1237_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_977_ = v___x_1237_;
goto v___jp_976_;
}
}
case 13:
{
lean_object* v___x_1238_; uint8_t v___x_1239_; 
v___x_1238_ = lean_unsigned_to_nat(1024u);
v___x_1239_ = lean_nat_dec_le(v___x_1238_, v_prec_891_);
if (v___x_1239_ == 0)
{
lean_object* v___x_1240_; 
v___x_1240_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_984_ = v___x_1240_;
goto v___jp_983_;
}
else
{
lean_object* v___x_1241_; 
v___x_1241_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_984_ = v___x_1241_;
goto v___jp_983_;
}
}
case 14:
{
lean_object* v___x_1242_; uint8_t v___x_1243_; 
v___x_1242_ = lean_unsigned_to_nat(1024u);
v___x_1243_ = lean_nat_dec_le(v___x_1242_, v_prec_891_);
if (v___x_1243_ == 0)
{
lean_object* v___x_1244_; 
v___x_1244_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_991_ = v___x_1244_;
goto v___jp_990_;
}
else
{
lean_object* v___x_1245_; 
v___x_1245_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_991_ = v___x_1245_;
goto v___jp_990_;
}
}
case 15:
{
lean_object* v___x_1246_; uint8_t v___x_1247_; 
v___x_1246_ = lean_unsigned_to_nat(1024u);
v___x_1247_ = lean_nat_dec_le(v___x_1246_, v_prec_891_);
if (v___x_1247_ == 0)
{
lean_object* v___x_1248_; 
v___x_1248_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_998_ = v___x_1248_;
goto v___jp_997_;
}
else
{
lean_object* v___x_1249_; 
v___x_1249_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_998_ = v___x_1249_;
goto v___jp_997_;
}
}
case 16:
{
lean_object* v___x_1250_; uint8_t v___x_1251_; 
v___x_1250_ = lean_unsigned_to_nat(1024u);
v___x_1251_ = lean_nat_dec_le(v___x_1250_, v_prec_891_);
if (v___x_1251_ == 0)
{
lean_object* v___x_1252_; 
v___x_1252_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1005_ = v___x_1252_;
goto v___jp_1004_;
}
else
{
lean_object* v___x_1253_; 
v___x_1253_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1005_ = v___x_1253_;
goto v___jp_1004_;
}
}
case 17:
{
lean_object* v___x_1254_; uint8_t v___x_1255_; 
v___x_1254_ = lean_unsigned_to_nat(1024u);
v___x_1255_ = lean_nat_dec_le(v___x_1254_, v_prec_891_);
if (v___x_1255_ == 0)
{
lean_object* v___x_1256_; 
v___x_1256_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1012_ = v___x_1256_;
goto v___jp_1011_;
}
else
{
lean_object* v___x_1257_; 
v___x_1257_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1012_ = v___x_1257_;
goto v___jp_1011_;
}
}
case 18:
{
lean_object* v___x_1258_; uint8_t v___x_1259_; 
v___x_1258_ = lean_unsigned_to_nat(1024u);
v___x_1259_ = lean_nat_dec_le(v___x_1258_, v_prec_891_);
if (v___x_1259_ == 0)
{
lean_object* v___x_1260_; 
v___x_1260_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1019_ = v___x_1260_;
goto v___jp_1018_;
}
else
{
lean_object* v___x_1261_; 
v___x_1261_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1019_ = v___x_1261_;
goto v___jp_1018_;
}
}
case 19:
{
lean_object* v___x_1262_; uint8_t v___x_1263_; 
v___x_1262_ = lean_unsigned_to_nat(1024u);
v___x_1263_ = lean_nat_dec_le(v___x_1262_, v_prec_891_);
if (v___x_1263_ == 0)
{
lean_object* v___x_1264_; 
v___x_1264_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1026_ = v___x_1264_;
goto v___jp_1025_;
}
else
{
lean_object* v___x_1265_; 
v___x_1265_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1026_ = v___x_1265_;
goto v___jp_1025_;
}
}
case 20:
{
lean_object* v___x_1266_; uint8_t v___x_1267_; 
v___x_1266_ = lean_unsigned_to_nat(1024u);
v___x_1267_ = lean_nat_dec_le(v___x_1266_, v_prec_891_);
if (v___x_1267_ == 0)
{
lean_object* v___x_1268_; 
v___x_1268_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1033_ = v___x_1268_;
goto v___jp_1032_;
}
else
{
lean_object* v___x_1269_; 
v___x_1269_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1033_ = v___x_1269_;
goto v___jp_1032_;
}
}
case 21:
{
lean_object* v___x_1270_; uint8_t v___x_1271_; 
v___x_1270_ = lean_unsigned_to_nat(1024u);
v___x_1271_ = lean_nat_dec_le(v___x_1270_, v_prec_891_);
if (v___x_1271_ == 0)
{
lean_object* v___x_1272_; 
v___x_1272_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1040_ = v___x_1272_;
goto v___jp_1039_;
}
else
{
lean_object* v___x_1273_; 
v___x_1273_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1040_ = v___x_1273_;
goto v___jp_1039_;
}
}
case 22:
{
lean_object* v___x_1274_; uint8_t v___x_1275_; 
v___x_1274_ = lean_unsigned_to_nat(1024u);
v___x_1275_ = lean_nat_dec_le(v___x_1274_, v_prec_891_);
if (v___x_1275_ == 0)
{
lean_object* v___x_1276_; 
v___x_1276_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1047_ = v___x_1276_;
goto v___jp_1046_;
}
else
{
lean_object* v___x_1277_; 
v___x_1277_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1047_ = v___x_1277_;
goto v___jp_1046_;
}
}
case 23:
{
lean_object* v___x_1278_; uint8_t v___x_1279_; 
v___x_1278_ = lean_unsigned_to_nat(1024u);
v___x_1279_ = lean_nat_dec_le(v___x_1278_, v_prec_891_);
if (v___x_1279_ == 0)
{
lean_object* v___x_1280_; 
v___x_1280_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1054_ = v___x_1280_;
goto v___jp_1053_;
}
else
{
lean_object* v___x_1281_; 
v___x_1281_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1054_ = v___x_1281_;
goto v___jp_1053_;
}
}
case 24:
{
lean_object* v___x_1282_; uint8_t v___x_1283_; 
v___x_1282_ = lean_unsigned_to_nat(1024u);
v___x_1283_ = lean_nat_dec_le(v___x_1282_, v_prec_891_);
if (v___x_1283_ == 0)
{
lean_object* v___x_1284_; 
v___x_1284_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1061_ = v___x_1284_;
goto v___jp_1060_;
}
else
{
lean_object* v___x_1285_; 
v___x_1285_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1061_ = v___x_1285_;
goto v___jp_1060_;
}
}
case 25:
{
lean_object* v___x_1286_; uint8_t v___x_1287_; 
v___x_1286_ = lean_unsigned_to_nat(1024u);
v___x_1287_ = lean_nat_dec_le(v___x_1286_, v_prec_891_);
if (v___x_1287_ == 0)
{
lean_object* v___x_1288_; 
v___x_1288_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1068_ = v___x_1288_;
goto v___jp_1067_;
}
else
{
lean_object* v___x_1289_; 
v___x_1289_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1068_ = v___x_1289_;
goto v___jp_1067_;
}
}
case 26:
{
lean_object* v___x_1290_; uint8_t v___x_1291_; 
v___x_1290_ = lean_unsigned_to_nat(1024u);
v___x_1291_ = lean_nat_dec_le(v___x_1290_, v_prec_891_);
if (v___x_1291_ == 0)
{
lean_object* v___x_1292_; 
v___x_1292_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1075_ = v___x_1292_;
goto v___jp_1074_;
}
else
{
lean_object* v___x_1293_; 
v___x_1293_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1075_ = v___x_1293_;
goto v___jp_1074_;
}
}
case 27:
{
lean_object* v___x_1294_; uint8_t v___x_1295_; 
v___x_1294_ = lean_unsigned_to_nat(1024u);
v___x_1295_ = lean_nat_dec_le(v___x_1294_, v_prec_891_);
if (v___x_1295_ == 0)
{
lean_object* v___x_1296_; 
v___x_1296_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1082_ = v___x_1296_;
goto v___jp_1081_;
}
else
{
lean_object* v___x_1297_; 
v___x_1297_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1082_ = v___x_1297_;
goto v___jp_1081_;
}
}
case 28:
{
lean_object* v___x_1298_; uint8_t v___x_1299_; 
v___x_1298_ = lean_unsigned_to_nat(1024u);
v___x_1299_ = lean_nat_dec_le(v___x_1298_, v_prec_891_);
if (v___x_1299_ == 0)
{
lean_object* v___x_1300_; 
v___x_1300_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1089_ = v___x_1300_;
goto v___jp_1088_;
}
else
{
lean_object* v___x_1301_; 
v___x_1301_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1089_ = v___x_1301_;
goto v___jp_1088_;
}
}
case 29:
{
lean_object* v___x_1302_; uint8_t v___x_1303_; 
v___x_1302_ = lean_unsigned_to_nat(1024u);
v___x_1303_ = lean_nat_dec_le(v___x_1302_, v_prec_891_);
if (v___x_1303_ == 0)
{
lean_object* v___x_1304_; 
v___x_1304_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1096_ = v___x_1304_;
goto v___jp_1095_;
}
else
{
lean_object* v___x_1305_; 
v___x_1305_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1096_ = v___x_1305_;
goto v___jp_1095_;
}
}
case 30:
{
lean_object* v___x_1306_; uint8_t v___x_1307_; 
v___x_1306_ = lean_unsigned_to_nat(1024u);
v___x_1307_ = lean_nat_dec_le(v___x_1306_, v_prec_891_);
if (v___x_1307_ == 0)
{
lean_object* v___x_1308_; 
v___x_1308_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1103_ = v___x_1308_;
goto v___jp_1102_;
}
else
{
lean_object* v___x_1309_; 
v___x_1309_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1103_ = v___x_1309_;
goto v___jp_1102_;
}
}
case 31:
{
lean_object* v___x_1310_; uint8_t v___x_1311_; 
v___x_1310_ = lean_unsigned_to_nat(1024u);
v___x_1311_ = lean_nat_dec_le(v___x_1310_, v_prec_891_);
if (v___x_1311_ == 0)
{
lean_object* v___x_1312_; 
v___x_1312_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1110_ = v___x_1312_;
goto v___jp_1109_;
}
else
{
lean_object* v___x_1313_; 
v___x_1313_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1110_ = v___x_1313_;
goto v___jp_1109_;
}
}
case 32:
{
lean_object* v___x_1314_; uint8_t v___x_1315_; 
v___x_1314_ = lean_unsigned_to_nat(1024u);
v___x_1315_ = lean_nat_dec_le(v___x_1314_, v_prec_891_);
if (v___x_1315_ == 0)
{
lean_object* v___x_1316_; 
v___x_1316_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1117_ = v___x_1316_;
goto v___jp_1116_;
}
else
{
lean_object* v___x_1317_; 
v___x_1317_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1117_ = v___x_1317_;
goto v___jp_1116_;
}
}
case 33:
{
lean_object* v___x_1318_; uint8_t v___x_1319_; 
v___x_1318_ = lean_unsigned_to_nat(1024u);
v___x_1319_ = lean_nat_dec_le(v___x_1318_, v_prec_891_);
if (v___x_1319_ == 0)
{
lean_object* v___x_1320_; 
v___x_1320_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1124_ = v___x_1320_;
goto v___jp_1123_;
}
else
{
lean_object* v___x_1321_; 
v___x_1321_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1124_ = v___x_1321_;
goto v___jp_1123_;
}
}
case 34:
{
lean_object* v___x_1322_; uint8_t v___x_1323_; 
v___x_1322_ = lean_unsigned_to_nat(1024u);
v___x_1323_ = lean_nat_dec_le(v___x_1322_, v_prec_891_);
if (v___x_1323_ == 0)
{
lean_object* v___x_1324_; 
v___x_1324_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1131_ = v___x_1324_;
goto v___jp_1130_;
}
else
{
lean_object* v___x_1325_; 
v___x_1325_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1131_ = v___x_1325_;
goto v___jp_1130_;
}
}
case 35:
{
lean_object* v___x_1326_; uint8_t v___x_1327_; 
v___x_1326_ = lean_unsigned_to_nat(1024u);
v___x_1327_ = lean_nat_dec_le(v___x_1326_, v_prec_891_);
if (v___x_1327_ == 0)
{
lean_object* v___x_1328_; 
v___x_1328_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1138_ = v___x_1328_;
goto v___jp_1137_;
}
else
{
lean_object* v___x_1329_; 
v___x_1329_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1138_ = v___x_1329_;
goto v___jp_1137_;
}
}
case 36:
{
lean_object* v___x_1330_; uint8_t v___x_1331_; 
v___x_1330_ = lean_unsigned_to_nat(1024u);
v___x_1331_ = lean_nat_dec_le(v___x_1330_, v_prec_891_);
if (v___x_1331_ == 0)
{
lean_object* v___x_1332_; 
v___x_1332_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1145_ = v___x_1332_;
goto v___jp_1144_;
}
else
{
lean_object* v___x_1333_; 
v___x_1333_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1145_ = v___x_1333_;
goto v___jp_1144_;
}
}
case 37:
{
lean_object* v___x_1334_; uint8_t v___x_1335_; 
v___x_1334_ = lean_unsigned_to_nat(1024u);
v___x_1335_ = lean_nat_dec_le(v___x_1334_, v_prec_891_);
if (v___x_1335_ == 0)
{
lean_object* v___x_1336_; 
v___x_1336_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1152_ = v___x_1336_;
goto v___jp_1151_;
}
else
{
lean_object* v___x_1337_; 
v___x_1337_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1152_ = v___x_1337_;
goto v___jp_1151_;
}
}
case 38:
{
lean_object* v___x_1338_; uint8_t v___x_1339_; 
v___x_1338_ = lean_unsigned_to_nat(1024u);
v___x_1339_ = lean_nat_dec_le(v___x_1338_, v_prec_891_);
if (v___x_1339_ == 0)
{
lean_object* v___x_1340_; 
v___x_1340_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1159_ = v___x_1340_;
goto v___jp_1158_;
}
else
{
lean_object* v___x_1341_; 
v___x_1341_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1159_ = v___x_1341_;
goto v___jp_1158_;
}
}
case 39:
{
lean_object* v___x_1342_; uint8_t v___x_1343_; 
v___x_1342_ = lean_unsigned_to_nat(1024u);
v___x_1343_ = lean_nat_dec_le(v___x_1342_, v_prec_891_);
if (v___x_1343_ == 0)
{
lean_object* v___x_1344_; 
v___x_1344_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1166_ = v___x_1344_;
goto v___jp_1165_;
}
else
{
lean_object* v___x_1345_; 
v___x_1345_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1166_ = v___x_1345_;
goto v___jp_1165_;
}
}
case 40:
{
lean_object* v___x_1346_; uint8_t v___x_1347_; 
v___x_1346_ = lean_unsigned_to_nat(1024u);
v___x_1347_ = lean_nat_dec_le(v___x_1346_, v_prec_891_);
if (v___x_1347_ == 0)
{
lean_object* v___x_1348_; 
v___x_1348_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1173_ = v___x_1348_;
goto v___jp_1172_;
}
else
{
lean_object* v___x_1349_; 
v___x_1349_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1173_ = v___x_1349_;
goto v___jp_1172_;
}
}
default: 
{
lean_object* v___x_1350_; uint8_t v___x_1351_; 
v___x_1350_ = lean_unsigned_to_nat(1024u);
v___x_1351_ = lean_nat_dec_le(v___x_1350_, v_prec_891_);
if (v___x_1351_ == 0)
{
lean_object* v___x_1352_; 
v___x_1352_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__84);
v___y_1180_ = v___x_1352_;
goto v___jp_1179_;
}
else
{
lean_object* v___x_1353_; 
v___x_1353_ = lean_obj_once(&lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85, &lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85_once, _init_lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__85);
v___y_1180_ = v___x_1353_;
goto v___jp_1179_;
}
}
}
v___jp_892_:
{
lean_object* v___x_894_; lean_object* v___x_895_; uint8_t v___x_896_; lean_object* v___x_897_; lean_object* v___x_898_; 
v___x_894_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__1));
lean_inc(v___y_893_);
v___x_895_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_895_, 0, v___y_893_);
lean_ctor_set(v___x_895_, 1, v___x_894_);
v___x_896_ = 0;
v___x_897_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_897_, 0, v___x_895_);
lean_ctor_set_uint8(v___x_897_, sizeof(void*)*1, v___x_896_);
v___x_898_ = l_Repr_addAppParen(v___x_897_, v_prec_891_);
return v___x_898_;
}
v___jp_899_:
{
lean_object* v___x_901_; lean_object* v___x_902_; uint8_t v___x_903_; lean_object* v___x_904_; lean_object* v___x_905_; 
v___x_901_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__3));
lean_inc(v___y_900_);
v___x_902_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_902_, 0, v___y_900_);
lean_ctor_set(v___x_902_, 1, v___x_901_);
v___x_903_ = 0;
v___x_904_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_904_, 0, v___x_902_);
lean_ctor_set_uint8(v___x_904_, sizeof(void*)*1, v___x_903_);
v___x_905_ = l_Repr_addAppParen(v___x_904_, v_prec_891_);
return v___x_905_;
}
v___jp_906_:
{
lean_object* v___x_908_; lean_object* v___x_909_; uint8_t v___x_910_; lean_object* v___x_911_; lean_object* v___x_912_; 
v___x_908_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__5));
lean_inc(v___y_907_);
v___x_909_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_909_, 0, v___y_907_);
lean_ctor_set(v___x_909_, 1, v___x_908_);
v___x_910_ = 0;
v___x_911_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_911_, 0, v___x_909_);
lean_ctor_set_uint8(v___x_911_, sizeof(void*)*1, v___x_910_);
v___x_912_ = l_Repr_addAppParen(v___x_911_, v_prec_891_);
return v___x_912_;
}
v___jp_913_:
{
lean_object* v___x_915_; lean_object* v___x_916_; uint8_t v___x_917_; lean_object* v___x_918_; lean_object* v___x_919_; 
v___x_915_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__7));
lean_inc(v___y_914_);
v___x_916_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_916_, 0, v___y_914_);
lean_ctor_set(v___x_916_, 1, v___x_915_);
v___x_917_ = 0;
v___x_918_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_918_, 0, v___x_916_);
lean_ctor_set_uint8(v___x_918_, sizeof(void*)*1, v___x_917_);
v___x_919_ = l_Repr_addAppParen(v___x_918_, v_prec_891_);
return v___x_919_;
}
v___jp_920_:
{
lean_object* v___x_922_; lean_object* v___x_923_; uint8_t v___x_924_; lean_object* v___x_925_; lean_object* v___x_926_; 
v___x_922_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__9));
lean_inc(v___y_921_);
v___x_923_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_923_, 0, v___y_921_);
lean_ctor_set(v___x_923_, 1, v___x_922_);
v___x_924_ = 0;
v___x_925_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_925_, 0, v___x_923_);
lean_ctor_set_uint8(v___x_925_, sizeof(void*)*1, v___x_924_);
v___x_926_ = l_Repr_addAppParen(v___x_925_, v_prec_891_);
return v___x_926_;
}
v___jp_927_:
{
lean_object* v___x_929_; lean_object* v___x_930_; uint8_t v___x_931_; lean_object* v___x_932_; lean_object* v___x_933_; 
v___x_929_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__11));
lean_inc(v___y_928_);
v___x_930_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_930_, 0, v___y_928_);
lean_ctor_set(v___x_930_, 1, v___x_929_);
v___x_931_ = 0;
v___x_932_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_932_, 0, v___x_930_);
lean_ctor_set_uint8(v___x_932_, sizeof(void*)*1, v___x_931_);
v___x_933_ = l_Repr_addAppParen(v___x_932_, v_prec_891_);
return v___x_933_;
}
v___jp_934_:
{
lean_object* v___x_936_; lean_object* v___x_937_; uint8_t v___x_938_; lean_object* v___x_939_; lean_object* v___x_940_; 
v___x_936_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__13));
lean_inc(v___y_935_);
v___x_937_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_937_, 0, v___y_935_);
lean_ctor_set(v___x_937_, 1, v___x_936_);
v___x_938_ = 0;
v___x_939_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_939_, 0, v___x_937_);
lean_ctor_set_uint8(v___x_939_, sizeof(void*)*1, v___x_938_);
v___x_940_ = l_Repr_addAppParen(v___x_939_, v_prec_891_);
return v___x_940_;
}
v___jp_941_:
{
lean_object* v___x_943_; lean_object* v___x_944_; uint8_t v___x_945_; lean_object* v___x_946_; lean_object* v___x_947_; 
v___x_943_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__15));
lean_inc(v___y_942_);
v___x_944_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_944_, 0, v___y_942_);
lean_ctor_set(v___x_944_, 1, v___x_943_);
v___x_945_ = 0;
v___x_946_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_946_, 0, v___x_944_);
lean_ctor_set_uint8(v___x_946_, sizeof(void*)*1, v___x_945_);
v___x_947_ = l_Repr_addAppParen(v___x_946_, v_prec_891_);
return v___x_947_;
}
v___jp_948_:
{
lean_object* v___x_950_; lean_object* v___x_951_; uint8_t v___x_952_; lean_object* v___x_953_; lean_object* v___x_954_; 
v___x_950_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__17));
lean_inc(v___y_949_);
v___x_951_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_951_, 0, v___y_949_);
lean_ctor_set(v___x_951_, 1, v___x_950_);
v___x_952_ = 0;
v___x_953_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_953_, 0, v___x_951_);
lean_ctor_set_uint8(v___x_953_, sizeof(void*)*1, v___x_952_);
v___x_954_ = l_Repr_addAppParen(v___x_953_, v_prec_891_);
return v___x_954_;
}
v___jp_955_:
{
lean_object* v___x_957_; lean_object* v___x_958_; uint8_t v___x_959_; lean_object* v___x_960_; lean_object* v___x_961_; 
v___x_957_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__19));
lean_inc(v___y_956_);
v___x_958_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_958_, 0, v___y_956_);
lean_ctor_set(v___x_958_, 1, v___x_957_);
v___x_959_ = 0;
v___x_960_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_960_, 0, v___x_958_);
lean_ctor_set_uint8(v___x_960_, sizeof(void*)*1, v___x_959_);
v___x_961_ = l_Repr_addAppParen(v___x_960_, v_prec_891_);
return v___x_961_;
}
v___jp_962_:
{
lean_object* v___x_964_; lean_object* v___x_965_; uint8_t v___x_966_; lean_object* v___x_967_; lean_object* v___x_968_; 
v___x_964_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__21));
lean_inc(v___y_963_);
v___x_965_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_965_, 0, v___y_963_);
lean_ctor_set(v___x_965_, 1, v___x_964_);
v___x_966_ = 0;
v___x_967_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_967_, 0, v___x_965_);
lean_ctor_set_uint8(v___x_967_, sizeof(void*)*1, v___x_966_);
v___x_968_ = l_Repr_addAppParen(v___x_967_, v_prec_891_);
return v___x_968_;
}
v___jp_969_:
{
lean_object* v___x_971_; lean_object* v___x_972_; uint8_t v___x_973_; lean_object* v___x_974_; lean_object* v___x_975_; 
v___x_971_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__23));
lean_inc(v___y_970_);
v___x_972_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_972_, 0, v___y_970_);
lean_ctor_set(v___x_972_, 1, v___x_971_);
v___x_973_ = 0;
v___x_974_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_974_, 0, v___x_972_);
lean_ctor_set_uint8(v___x_974_, sizeof(void*)*1, v___x_973_);
v___x_975_ = l_Repr_addAppParen(v___x_974_, v_prec_891_);
return v___x_975_;
}
v___jp_976_:
{
lean_object* v___x_978_; lean_object* v___x_979_; uint8_t v___x_980_; lean_object* v___x_981_; lean_object* v___x_982_; 
v___x_978_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__25));
lean_inc(v___y_977_);
v___x_979_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_979_, 0, v___y_977_);
lean_ctor_set(v___x_979_, 1, v___x_978_);
v___x_980_ = 0;
v___x_981_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_981_, 0, v___x_979_);
lean_ctor_set_uint8(v___x_981_, sizeof(void*)*1, v___x_980_);
v___x_982_ = l_Repr_addAppParen(v___x_981_, v_prec_891_);
return v___x_982_;
}
v___jp_983_:
{
lean_object* v___x_985_; lean_object* v___x_986_; uint8_t v___x_987_; lean_object* v___x_988_; lean_object* v___x_989_; 
v___x_985_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__27));
lean_inc(v___y_984_);
v___x_986_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_986_, 0, v___y_984_);
lean_ctor_set(v___x_986_, 1, v___x_985_);
v___x_987_ = 0;
v___x_988_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_988_, 0, v___x_986_);
lean_ctor_set_uint8(v___x_988_, sizeof(void*)*1, v___x_987_);
v___x_989_ = l_Repr_addAppParen(v___x_988_, v_prec_891_);
return v___x_989_;
}
v___jp_990_:
{
lean_object* v___x_992_; lean_object* v___x_993_; uint8_t v___x_994_; lean_object* v___x_995_; lean_object* v___x_996_; 
v___x_992_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__29));
lean_inc(v___y_991_);
v___x_993_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_993_, 0, v___y_991_);
lean_ctor_set(v___x_993_, 1, v___x_992_);
v___x_994_ = 0;
v___x_995_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_995_, 0, v___x_993_);
lean_ctor_set_uint8(v___x_995_, sizeof(void*)*1, v___x_994_);
v___x_996_ = l_Repr_addAppParen(v___x_995_, v_prec_891_);
return v___x_996_;
}
v___jp_997_:
{
lean_object* v___x_999_; lean_object* v___x_1000_; uint8_t v___x_1001_; lean_object* v___x_1002_; lean_object* v___x_1003_; 
v___x_999_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__31));
lean_inc(v___y_998_);
v___x_1000_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1000_, 0, v___y_998_);
lean_ctor_set(v___x_1000_, 1, v___x_999_);
v___x_1001_ = 0;
v___x_1002_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1002_, 0, v___x_1000_);
lean_ctor_set_uint8(v___x_1002_, sizeof(void*)*1, v___x_1001_);
v___x_1003_ = l_Repr_addAppParen(v___x_1002_, v_prec_891_);
return v___x_1003_;
}
v___jp_1004_:
{
lean_object* v___x_1006_; lean_object* v___x_1007_; uint8_t v___x_1008_; lean_object* v___x_1009_; lean_object* v___x_1010_; 
v___x_1006_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__33));
lean_inc(v___y_1005_);
v___x_1007_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1007_, 0, v___y_1005_);
lean_ctor_set(v___x_1007_, 1, v___x_1006_);
v___x_1008_ = 0;
v___x_1009_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1009_, 0, v___x_1007_);
lean_ctor_set_uint8(v___x_1009_, sizeof(void*)*1, v___x_1008_);
v___x_1010_ = l_Repr_addAppParen(v___x_1009_, v_prec_891_);
return v___x_1010_;
}
v___jp_1011_:
{
lean_object* v___x_1013_; lean_object* v___x_1014_; uint8_t v___x_1015_; lean_object* v___x_1016_; lean_object* v___x_1017_; 
v___x_1013_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__35));
lean_inc(v___y_1012_);
v___x_1014_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1014_, 0, v___y_1012_);
lean_ctor_set(v___x_1014_, 1, v___x_1013_);
v___x_1015_ = 0;
v___x_1016_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1016_, 0, v___x_1014_);
lean_ctor_set_uint8(v___x_1016_, sizeof(void*)*1, v___x_1015_);
v___x_1017_ = l_Repr_addAppParen(v___x_1016_, v_prec_891_);
return v___x_1017_;
}
v___jp_1018_:
{
lean_object* v___x_1020_; lean_object* v___x_1021_; uint8_t v___x_1022_; lean_object* v___x_1023_; lean_object* v___x_1024_; 
v___x_1020_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__37));
lean_inc(v___y_1019_);
v___x_1021_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1021_, 0, v___y_1019_);
lean_ctor_set(v___x_1021_, 1, v___x_1020_);
v___x_1022_ = 0;
v___x_1023_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1023_, 0, v___x_1021_);
lean_ctor_set_uint8(v___x_1023_, sizeof(void*)*1, v___x_1022_);
v___x_1024_ = l_Repr_addAppParen(v___x_1023_, v_prec_891_);
return v___x_1024_;
}
v___jp_1025_:
{
lean_object* v___x_1027_; lean_object* v___x_1028_; uint8_t v___x_1029_; lean_object* v___x_1030_; lean_object* v___x_1031_; 
v___x_1027_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__39));
lean_inc(v___y_1026_);
v___x_1028_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1028_, 0, v___y_1026_);
lean_ctor_set(v___x_1028_, 1, v___x_1027_);
v___x_1029_ = 0;
v___x_1030_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1030_, 0, v___x_1028_);
lean_ctor_set_uint8(v___x_1030_, sizeof(void*)*1, v___x_1029_);
v___x_1031_ = l_Repr_addAppParen(v___x_1030_, v_prec_891_);
return v___x_1031_;
}
v___jp_1032_:
{
lean_object* v___x_1034_; lean_object* v___x_1035_; uint8_t v___x_1036_; lean_object* v___x_1037_; lean_object* v___x_1038_; 
v___x_1034_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__41));
lean_inc(v___y_1033_);
v___x_1035_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1035_, 0, v___y_1033_);
lean_ctor_set(v___x_1035_, 1, v___x_1034_);
v___x_1036_ = 0;
v___x_1037_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1037_, 0, v___x_1035_);
lean_ctor_set_uint8(v___x_1037_, sizeof(void*)*1, v___x_1036_);
v___x_1038_ = l_Repr_addAppParen(v___x_1037_, v_prec_891_);
return v___x_1038_;
}
v___jp_1039_:
{
lean_object* v___x_1041_; lean_object* v___x_1042_; uint8_t v___x_1043_; lean_object* v___x_1044_; lean_object* v___x_1045_; 
v___x_1041_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__43));
lean_inc(v___y_1040_);
v___x_1042_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1042_, 0, v___y_1040_);
lean_ctor_set(v___x_1042_, 1, v___x_1041_);
v___x_1043_ = 0;
v___x_1044_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1044_, 0, v___x_1042_);
lean_ctor_set_uint8(v___x_1044_, sizeof(void*)*1, v___x_1043_);
v___x_1045_ = l_Repr_addAppParen(v___x_1044_, v_prec_891_);
return v___x_1045_;
}
v___jp_1046_:
{
lean_object* v___x_1048_; lean_object* v___x_1049_; uint8_t v___x_1050_; lean_object* v___x_1051_; lean_object* v___x_1052_; 
v___x_1048_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__45));
lean_inc(v___y_1047_);
v___x_1049_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1049_, 0, v___y_1047_);
lean_ctor_set(v___x_1049_, 1, v___x_1048_);
v___x_1050_ = 0;
v___x_1051_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1051_, 0, v___x_1049_);
lean_ctor_set_uint8(v___x_1051_, sizeof(void*)*1, v___x_1050_);
v___x_1052_ = l_Repr_addAppParen(v___x_1051_, v_prec_891_);
return v___x_1052_;
}
v___jp_1053_:
{
lean_object* v___x_1055_; lean_object* v___x_1056_; uint8_t v___x_1057_; lean_object* v___x_1058_; lean_object* v___x_1059_; 
v___x_1055_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__47));
lean_inc(v___y_1054_);
v___x_1056_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1056_, 0, v___y_1054_);
lean_ctor_set(v___x_1056_, 1, v___x_1055_);
v___x_1057_ = 0;
v___x_1058_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1058_, 0, v___x_1056_);
lean_ctor_set_uint8(v___x_1058_, sizeof(void*)*1, v___x_1057_);
v___x_1059_ = l_Repr_addAppParen(v___x_1058_, v_prec_891_);
return v___x_1059_;
}
v___jp_1060_:
{
lean_object* v___x_1062_; lean_object* v___x_1063_; uint8_t v___x_1064_; lean_object* v___x_1065_; lean_object* v___x_1066_; 
v___x_1062_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__49));
lean_inc(v___y_1061_);
v___x_1063_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1063_, 0, v___y_1061_);
lean_ctor_set(v___x_1063_, 1, v___x_1062_);
v___x_1064_ = 0;
v___x_1065_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1065_, 0, v___x_1063_);
lean_ctor_set_uint8(v___x_1065_, sizeof(void*)*1, v___x_1064_);
v___x_1066_ = l_Repr_addAppParen(v___x_1065_, v_prec_891_);
return v___x_1066_;
}
v___jp_1067_:
{
lean_object* v___x_1069_; lean_object* v___x_1070_; uint8_t v___x_1071_; lean_object* v___x_1072_; lean_object* v___x_1073_; 
v___x_1069_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__51));
lean_inc(v___y_1068_);
v___x_1070_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1070_, 0, v___y_1068_);
lean_ctor_set(v___x_1070_, 1, v___x_1069_);
v___x_1071_ = 0;
v___x_1072_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1072_, 0, v___x_1070_);
lean_ctor_set_uint8(v___x_1072_, sizeof(void*)*1, v___x_1071_);
v___x_1073_ = l_Repr_addAppParen(v___x_1072_, v_prec_891_);
return v___x_1073_;
}
v___jp_1074_:
{
lean_object* v___x_1076_; lean_object* v___x_1077_; uint8_t v___x_1078_; lean_object* v___x_1079_; lean_object* v___x_1080_; 
v___x_1076_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__53));
lean_inc(v___y_1075_);
v___x_1077_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1077_, 0, v___y_1075_);
lean_ctor_set(v___x_1077_, 1, v___x_1076_);
v___x_1078_ = 0;
v___x_1079_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1079_, 0, v___x_1077_);
lean_ctor_set_uint8(v___x_1079_, sizeof(void*)*1, v___x_1078_);
v___x_1080_ = l_Repr_addAppParen(v___x_1079_, v_prec_891_);
return v___x_1080_;
}
v___jp_1081_:
{
lean_object* v___x_1083_; lean_object* v___x_1084_; uint8_t v___x_1085_; lean_object* v___x_1086_; lean_object* v___x_1087_; 
v___x_1083_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__55));
lean_inc(v___y_1082_);
v___x_1084_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1084_, 0, v___y_1082_);
lean_ctor_set(v___x_1084_, 1, v___x_1083_);
v___x_1085_ = 0;
v___x_1086_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1086_, 0, v___x_1084_);
lean_ctor_set_uint8(v___x_1086_, sizeof(void*)*1, v___x_1085_);
v___x_1087_ = l_Repr_addAppParen(v___x_1086_, v_prec_891_);
return v___x_1087_;
}
v___jp_1088_:
{
lean_object* v___x_1090_; lean_object* v___x_1091_; uint8_t v___x_1092_; lean_object* v___x_1093_; lean_object* v___x_1094_; 
v___x_1090_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__57));
lean_inc(v___y_1089_);
v___x_1091_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1091_, 0, v___y_1089_);
lean_ctor_set(v___x_1091_, 1, v___x_1090_);
v___x_1092_ = 0;
v___x_1093_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1093_, 0, v___x_1091_);
lean_ctor_set_uint8(v___x_1093_, sizeof(void*)*1, v___x_1092_);
v___x_1094_ = l_Repr_addAppParen(v___x_1093_, v_prec_891_);
return v___x_1094_;
}
v___jp_1095_:
{
lean_object* v___x_1097_; lean_object* v___x_1098_; uint8_t v___x_1099_; lean_object* v___x_1100_; lean_object* v___x_1101_; 
v___x_1097_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__59));
lean_inc(v___y_1096_);
v___x_1098_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1098_, 0, v___y_1096_);
lean_ctor_set(v___x_1098_, 1, v___x_1097_);
v___x_1099_ = 0;
v___x_1100_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1100_, 0, v___x_1098_);
lean_ctor_set_uint8(v___x_1100_, sizeof(void*)*1, v___x_1099_);
v___x_1101_ = l_Repr_addAppParen(v___x_1100_, v_prec_891_);
return v___x_1101_;
}
v___jp_1102_:
{
lean_object* v___x_1104_; lean_object* v___x_1105_; uint8_t v___x_1106_; lean_object* v___x_1107_; lean_object* v___x_1108_; 
v___x_1104_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__61));
lean_inc(v___y_1103_);
v___x_1105_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1105_, 0, v___y_1103_);
lean_ctor_set(v___x_1105_, 1, v___x_1104_);
v___x_1106_ = 0;
v___x_1107_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1107_, 0, v___x_1105_);
lean_ctor_set_uint8(v___x_1107_, sizeof(void*)*1, v___x_1106_);
v___x_1108_ = l_Repr_addAppParen(v___x_1107_, v_prec_891_);
return v___x_1108_;
}
v___jp_1109_:
{
lean_object* v___x_1111_; lean_object* v___x_1112_; uint8_t v___x_1113_; lean_object* v___x_1114_; lean_object* v___x_1115_; 
v___x_1111_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__63));
lean_inc(v___y_1110_);
v___x_1112_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1112_, 0, v___y_1110_);
lean_ctor_set(v___x_1112_, 1, v___x_1111_);
v___x_1113_ = 0;
v___x_1114_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1114_, 0, v___x_1112_);
lean_ctor_set_uint8(v___x_1114_, sizeof(void*)*1, v___x_1113_);
v___x_1115_ = l_Repr_addAppParen(v___x_1114_, v_prec_891_);
return v___x_1115_;
}
v___jp_1116_:
{
lean_object* v___x_1118_; lean_object* v___x_1119_; uint8_t v___x_1120_; lean_object* v___x_1121_; lean_object* v___x_1122_; 
v___x_1118_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__65));
lean_inc(v___y_1117_);
v___x_1119_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1119_, 0, v___y_1117_);
lean_ctor_set(v___x_1119_, 1, v___x_1118_);
v___x_1120_ = 0;
v___x_1121_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1121_, 0, v___x_1119_);
lean_ctor_set_uint8(v___x_1121_, sizeof(void*)*1, v___x_1120_);
v___x_1122_ = l_Repr_addAppParen(v___x_1121_, v_prec_891_);
return v___x_1122_;
}
v___jp_1123_:
{
lean_object* v___x_1125_; lean_object* v___x_1126_; uint8_t v___x_1127_; lean_object* v___x_1128_; lean_object* v___x_1129_; 
v___x_1125_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__67));
lean_inc(v___y_1124_);
v___x_1126_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1126_, 0, v___y_1124_);
lean_ctor_set(v___x_1126_, 1, v___x_1125_);
v___x_1127_ = 0;
v___x_1128_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1128_, 0, v___x_1126_);
lean_ctor_set_uint8(v___x_1128_, sizeof(void*)*1, v___x_1127_);
v___x_1129_ = l_Repr_addAppParen(v___x_1128_, v_prec_891_);
return v___x_1129_;
}
v___jp_1130_:
{
lean_object* v___x_1132_; lean_object* v___x_1133_; uint8_t v___x_1134_; lean_object* v___x_1135_; lean_object* v___x_1136_; 
v___x_1132_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__69));
lean_inc(v___y_1131_);
v___x_1133_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1133_, 0, v___y_1131_);
lean_ctor_set(v___x_1133_, 1, v___x_1132_);
v___x_1134_ = 0;
v___x_1135_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1135_, 0, v___x_1133_);
lean_ctor_set_uint8(v___x_1135_, sizeof(void*)*1, v___x_1134_);
v___x_1136_ = l_Repr_addAppParen(v___x_1135_, v_prec_891_);
return v___x_1136_;
}
v___jp_1137_:
{
lean_object* v___x_1139_; lean_object* v___x_1140_; uint8_t v___x_1141_; lean_object* v___x_1142_; lean_object* v___x_1143_; 
v___x_1139_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__71));
lean_inc(v___y_1138_);
v___x_1140_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1140_, 0, v___y_1138_);
lean_ctor_set(v___x_1140_, 1, v___x_1139_);
v___x_1141_ = 0;
v___x_1142_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1142_, 0, v___x_1140_);
lean_ctor_set_uint8(v___x_1142_, sizeof(void*)*1, v___x_1141_);
v___x_1143_ = l_Repr_addAppParen(v___x_1142_, v_prec_891_);
return v___x_1143_;
}
v___jp_1144_:
{
lean_object* v___x_1146_; lean_object* v___x_1147_; uint8_t v___x_1148_; lean_object* v___x_1149_; lean_object* v___x_1150_; 
v___x_1146_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__73));
lean_inc(v___y_1145_);
v___x_1147_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1147_, 0, v___y_1145_);
lean_ctor_set(v___x_1147_, 1, v___x_1146_);
v___x_1148_ = 0;
v___x_1149_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1149_, 0, v___x_1147_);
lean_ctor_set_uint8(v___x_1149_, sizeof(void*)*1, v___x_1148_);
v___x_1150_ = l_Repr_addAppParen(v___x_1149_, v_prec_891_);
return v___x_1150_;
}
v___jp_1151_:
{
lean_object* v___x_1153_; lean_object* v___x_1154_; uint8_t v___x_1155_; lean_object* v___x_1156_; lean_object* v___x_1157_; 
v___x_1153_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__75));
lean_inc(v___y_1152_);
v___x_1154_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1154_, 0, v___y_1152_);
lean_ctor_set(v___x_1154_, 1, v___x_1153_);
v___x_1155_ = 0;
v___x_1156_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1156_, 0, v___x_1154_);
lean_ctor_set_uint8(v___x_1156_, sizeof(void*)*1, v___x_1155_);
v___x_1157_ = l_Repr_addAppParen(v___x_1156_, v_prec_891_);
return v___x_1157_;
}
v___jp_1158_:
{
lean_object* v___x_1160_; lean_object* v___x_1161_; uint8_t v___x_1162_; lean_object* v___x_1163_; lean_object* v___x_1164_; 
v___x_1160_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__77));
lean_inc(v___y_1159_);
v___x_1161_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1161_, 0, v___y_1159_);
lean_ctor_set(v___x_1161_, 1, v___x_1160_);
v___x_1162_ = 0;
v___x_1163_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1163_, 0, v___x_1161_);
lean_ctor_set_uint8(v___x_1163_, sizeof(void*)*1, v___x_1162_);
v___x_1164_ = l_Repr_addAppParen(v___x_1163_, v_prec_891_);
return v___x_1164_;
}
v___jp_1165_:
{
lean_object* v___x_1167_; lean_object* v___x_1168_; uint8_t v___x_1169_; lean_object* v___x_1170_; lean_object* v___x_1171_; 
v___x_1167_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__79));
lean_inc(v___y_1166_);
v___x_1168_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1168_, 0, v___y_1166_);
lean_ctor_set(v___x_1168_, 1, v___x_1167_);
v___x_1169_ = 0;
v___x_1170_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1170_, 0, v___x_1168_);
lean_ctor_set_uint8(v___x_1170_, sizeof(void*)*1, v___x_1169_);
v___x_1171_ = l_Repr_addAppParen(v___x_1170_, v_prec_891_);
return v___x_1171_;
}
v___jp_1172_:
{
lean_object* v___x_1174_; lean_object* v___x_1175_; uint8_t v___x_1176_; lean_object* v___x_1177_; lean_object* v___x_1178_; 
v___x_1174_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__81));
lean_inc(v___y_1173_);
v___x_1175_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1175_, 0, v___y_1173_);
lean_ctor_set(v___x_1175_, 1, v___x_1174_);
v___x_1176_ = 0;
v___x_1177_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1177_, 0, v___x_1175_);
lean_ctor_set_uint8(v___x_1177_, sizeof(void*)*1, v___x_1176_);
v___x_1178_ = l_Repr_addAppParen(v___x_1177_, v_prec_891_);
return v___x_1178_;
}
v___jp_1179_:
{
lean_object* v___x_1181_; lean_object* v___x_1182_; uint8_t v___x_1183_; lean_object* v___x_1184_; lean_object* v___x_1185_; 
v___x_1181_ = ((lean_object*)(lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___closed__83));
lean_inc(v___y_1180_);
v___x_1182_ = lean_alloc_ctor(4, 2, 0);
lean_ctor_set(v___x_1182_, 0, v___y_1180_);
lean_ctor_set(v___x_1182_, 1, v___x_1181_);
v___x_1183_ = 0;
v___x_1184_ = lean_alloc_ctor(6, 1, 1);
lean_ctor_set(v___x_1184_, 0, v___x_1182_);
lean_ctor_set_uint8(v___x_1184_, sizeof(void*)*1, v___x_1183_);
v___x_1185_ = l_Repr_addAppParen(v___x_1184_, v_prec_891_);
return v___x_1185_;
}
}
}
LEAN_EXPORT lean_object* lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr___boxed(lean_object* v_x_1354_, lean_object* v_prec_1355_){
_start:
{
uint8_t v_x_2361__boxed_1356_; lean_object* v_res_1357_; 
v_x_2361__boxed_1356_ = lean_unbox(v_x_1354_);
v_res_1357_ = lp_workspace_Recursion_Spec_VerifierCircuit_instReprVerifierAirId_repr(v_x_2361__boxed_1356_, v_prec_1355_);
lean_dec(v_prec_1355_);
return v_res_1357_;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Init(uint8_t builtin);
void lean_initialize_runtime_module();
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_workspace_Recursion_Spec_VerifierAirId(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
lean_initialize_runtime_module();
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
