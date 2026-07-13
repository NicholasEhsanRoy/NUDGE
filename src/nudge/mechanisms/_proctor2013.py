"""AUTO-GENERATED — do not edit by hand. Regenerate with
``scripts/vv/gen_proctor2013.py``.

The Proctor et al. 2013 Alzheimer's-disease amyloid-β / tau / p53 / microglia ODE
model as a differentiable JAX vector field, transcribed directly from the CC0
BioModels SBML ``BIOMD0000000488``.  Mathematical facts (the reaction network +
published rate constants); see the citation in ``ad_qsp.py``.

64 dynamic states, 5 constant boundary species,
73 kinetic parameters, 112 reactions.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

SPECIES = (
    'Mdm2',
    'p53',
    'Mdm2_p53',
    'Mdm2_mRNA',
    'p53_mRNA',
    'ATMA',
    'ATMI',
    'p53_P',
    'Mdm2_P',
    'IR',
    'ROS',
    'damDNA',
    'E1',
    'E2',
    'E1_Ub',
    'E2_Ub',
    'Proteasome',
    'Ub',
    'p53DUB',
    'Mdm2DUB',
    'DUB',
    'Mdm2_p53_Ub',
    'Mdm2_p53_Ub2',
    'Mdm2_p53_Ub3',
    'Mdm2_p53_Ub4',
    'Mdm2_P1_p53_Ub4',
    'Mdm2_Ub',
    'Mdm2_Ub2',
    'Mdm2_Ub3',
    'Mdm2_Ub4',
    'Mdm2_P_Ub',
    'Mdm2_P_Ub2',
    'Mdm2_P_Ub3',
    'Mdm2_P_Ub4',
    'p53_Ub4_Proteasome',
    'Mdm2_Ub4_Proteasome',
    'Mdm2_P_Ub4_Proteasome',
    'GSK3b',
    'GSK3b_p53',
    'GSK3b_p53_P',
    'Abeta',
    'AggAbeta_Proteasome',
    'AbetaPlaque',
    'Tau',
    'Tau_P1',
    'Tau_P2',
    'MT_Tau',
    'AggTau',
    'AggTau_Proteasome',
    'Proteasome_Tau',
    'PP1',
    'NFT',
    'AbetaDimer',
    'AbetaPlaque_GliaA',
    'GliaI',
    'GliaM1',
    'GliaM2',
    'GliaA',
    'antiAb',
    'Abeta_antiAb',
    'AbetaDimer_antiAb',
    'degAbetaGlia',
    'disaggPlaque1',
    'disaggPlaque2',
)

Y0 = np.array([
    5, 5, 95, 10, 10, 0, 200, 0, 0, 0, 0, 0, 100, 100, 0, 0, 500, 4000, 200, 200, 200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 500, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 50, 0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0,
], dtype=np.float64)

PARAM_NAMES = (
    'ksynp53mRNA',
    'kdegp53mRNA',
    'ksynMdm2mRNA',
    'kdegMdm2mRNA',
    'ksynMdm2mRNAGSK3bp53',
    'ksynp53',
    'kdegp53',
    'kbinMdm2p53',
    'krelMdm2p53',
    'kbinGSK3bp53',
    'krelGSK3bp53',
    'ksynMdm2',
    'kdegMdm2',
    'kbinE1Ub',
    'kbinE2Ub',
    'kp53Ub',
    'kp53PolyUb',
    'kbinProt',
    'kactDUBp53',
    'kactDUBProtp53',
    'kactDUBMdm2',
    'kMdm2Ub',
    'kMdm2PUb',
    'kMdm2PolyUb',
    'kdam',
    'krepair',
    'kactATM',
    'kinactATM',
    'kphosp53',
    'kdephosp53',
    'kphosMdm2',
    'kdephosMdm2',
    'kphosMdm2GSK3b',
    'kphosMdm2GSK3bp53',
    'kphospTauGSK3bp53',
    'kphospTauGSK3b',
    'kdephospTau',
    'kbinMTTau',
    'krelMTTau',
    'ksynTau',
    'kbinTauProt',
    'kdegTau20SProt',
    'kaggTau',
    'kaggTauP1',
    'kaggTauP2',
    'ktangfor',
    'kinhibprot',
    'ksynp53mRNAAbeta',
    'kdamROS',
    'kgenROSAbeta',
    'kgenROSPlaque',
    'kgenROSGlia',
    'kproteff',
    'kremROS',
    'kprodAbeta',
    'kprodAbeta2',
    'kdegAbeta',
    'kaggAbeta',
    'kdisaggAbeta',
    'kdisaggAbeta1',
    'kdisaggAbeta2',
    'kdegAbetaGlia',
    'kpf',
    'kpg',
    'kpghalf',
    'kactglia1',
    'kactglia2',
    'kinactglia1',
    'kinactglia2',
    'kbinAbetaGlia',
    'krelAbetaGlia',
    'kdegAntiAb',
    'kbinAbantiAb',
)

PARAM_VALUES = np.array([
    0.001, 0.0001, 0.0005, 0.0005, 0.0007, 0.007, 0.005, 0.001155, 1.155e-05, 2e-06, 0.002, 0.000495, 0.01, 0.0002, 0.001, 5e-05, 0.01, 2e-06, 1e-07, 0.0001, 1e-07, 4.56e-06, 6.84e-06, 0.00456, 0.08, 2e-05, 0.0001, 0.0005, 0.0002, 0.5, 2, 0.5, 0.005, 0.5, 0.1, 0.0002, 0.01, 0.1, 0.0001, 8e-05, 1.925e-07, 0.01, 1e-08, 1e-08, 1e-07, 0.001, 1e-07, 1e-05, 1e-05, 2e-05, 1e-05, 1e-05, 1, 7e-05, 1.86e-05, 1.86e-05, 1.5e-05, 3e-06, 1e-06, 0.0002, 1e-06, 0.005, 0.2, 0.15, 10, 6e-07, 6e-07, 5e-06, 5e-06, 1e-05, 5e-05, 2.75e-06, 1e-06,
], dtype=np.float64)

BOUNDARY = {
    'ATP': 10000,
    'ADP': 1000,
    'AMP': 1000,
    'Source': 1,
    'Sink': 1,
}

N_STATES = len(SPECIES)
N_PARAMS = len(PARAM_NAMES)


def rhs(y, p):
    """dy/dt = S . v(y) for the Proctor 2013 network (y: (64,), p: (73,)).

    ``y`` are the dynamic species (order = ``SPECIES``); ``p`` the kinetic
    parameters (order = ``PARAM_NAMES``). Boundary species are constants.
    """
    Mdm2 = y[0]
    p53 = y[1]
    Mdm2_p53 = y[2]
    Mdm2_mRNA = y[3]
    p53_mRNA = y[4]
    ATMA = y[5]
    ATMI = y[6]
    p53_P = y[7]
    Mdm2_P = y[8]
    IR = y[9]
    ROS = y[10]
    damDNA = y[11]
    E1 = y[12]
    E2 = y[13]
    E1_Ub = y[14]
    E2_Ub = y[15]
    Proteasome = y[16]
    Ub = y[17]
    p53DUB = y[18]
    Mdm2DUB = y[19]
    DUB = y[20]
    Mdm2_p53_Ub = y[21]
    Mdm2_p53_Ub2 = y[22]
    Mdm2_p53_Ub3 = y[23]
    Mdm2_p53_Ub4 = y[24]
    Mdm2_P1_p53_Ub4 = y[25]
    Mdm2_Ub = y[26]
    Mdm2_Ub2 = y[27]
    Mdm2_Ub3 = y[28]
    Mdm2_Ub4 = y[29]
    Mdm2_P_Ub = y[30]
    Mdm2_P_Ub2 = y[31]
    Mdm2_P_Ub3 = y[32]
    Mdm2_P_Ub4 = y[33]
    p53_Ub4_Proteasome = y[34]
    Mdm2_Ub4_Proteasome = y[35]
    Mdm2_P_Ub4_Proteasome = y[36]
    GSK3b = y[37]
    GSK3b_p53 = y[38]
    GSK3b_p53_P = y[39]
    Abeta = y[40]
    AggAbeta_Proteasome = y[41]
    AbetaPlaque = y[42]
    Tau = y[43]
    Tau_P1 = y[44]
    Tau_P2 = y[45]
    MT_Tau = y[46]
    AggTau = y[47]
    AggTau_Proteasome = y[48]
    Proteasome_Tau = y[49]
    PP1 = y[50]
    NFT = y[51]
    AbetaDimer = y[52]
    AbetaPlaque_GliaA = y[53]
    GliaI = y[54]
    GliaM1 = y[55]
    GliaM2 = y[56]
    GliaA = y[57]
    antiAb = y[58]
    Abeta_antiAb = y[59]
    AbetaDimer_antiAb = y[60]
    degAbetaGlia = y[61]
    disaggPlaque1 = y[62]
    disaggPlaque2 = y[63]
    # boundary species (held constant)
    ATP = 10000
    ADP = 1000
    AMP = 1000
    Source = 1
    Sink = 1
    # kinetic parameters
    ksynp53mRNA = p[0]
    kdegp53mRNA = p[1]
    ksynMdm2mRNA = p[2]
    kdegMdm2mRNA = p[3]
    ksynMdm2mRNAGSK3bp53 = p[4]
    ksynp53 = p[5]
    kdegp53 = p[6]
    kbinMdm2p53 = p[7]
    krelMdm2p53 = p[8]
    kbinGSK3bp53 = p[9]
    krelGSK3bp53 = p[10]
    ksynMdm2 = p[11]
    kdegMdm2 = p[12]
    kbinE1Ub = p[13]
    kbinE2Ub = p[14]
    kp53Ub = p[15]
    kp53PolyUb = p[16]
    kbinProt = p[17]
    kactDUBp53 = p[18]
    kactDUBProtp53 = p[19]
    kactDUBMdm2 = p[20]
    kMdm2Ub = p[21]
    kMdm2PUb = p[22]
    kMdm2PolyUb = p[23]
    kdam = p[24]
    krepair = p[25]
    kactATM = p[26]
    kinactATM = p[27]
    kphosp53 = p[28]
    kdephosp53 = p[29]
    kphosMdm2 = p[30]
    kdephosMdm2 = p[31]
    kphosMdm2GSK3b = p[32]
    kphosMdm2GSK3bp53 = p[33]
    kphospTauGSK3bp53 = p[34]
    kphospTauGSK3b = p[35]
    kdephospTau = p[36]
    kbinMTTau = p[37]
    krelMTTau = p[38]
    ksynTau = p[39]
    kbinTauProt = p[40]
    kdegTau20SProt = p[41]
    kaggTau = p[42]
    kaggTauP1 = p[43]
    kaggTauP2 = p[44]
    ktangfor = p[45]
    kinhibprot = p[46]
    ksynp53mRNAAbeta = p[47]
    kdamROS = p[48]
    kgenROSAbeta = p[49]
    kgenROSPlaque = p[50]
    kgenROSGlia = p[51]
    kproteff = p[52]
    kremROS = p[53]
    kprodAbeta = p[54]
    kprodAbeta2 = p[55]
    kdegAbeta = p[56]
    kaggAbeta = p[57]
    kdisaggAbeta = p[58]
    kdisaggAbeta1 = p[59]
    kdisaggAbeta2 = p[60]
    kdegAbetaGlia = p[61]
    kpf = p[62]
    kpg = p[63]
    kpghalf = p[64]
    kactglia1 = p[65]
    kactglia2 = p[66]
    kinactglia1 = p[67]
    kinactglia2 = p[68]
    kbinAbetaGlia = p[69]
    krelAbetaGlia = p[70]
    kdegAntiAb = p[71]
    kbinAbantiAb = p[72]
    # reaction rates (SBML kinetic laws)
    r0 = (ksynp53mRNA * Source)  # p53mRNASynthesis
    r1 = (kdegp53mRNA * p53_mRNA)  # p53mRNADegradation
    r2 = (ksynMdm2 * Mdm2_mRNA)  # Mdm2Synthesis
    r3 = (ksynMdm2mRNA * p53)  # Mdm2mRNASynthesis1
    r4 = (ksynMdm2mRNA * p53_P)  # Mdm2mRNASynthesis2
    r5 = (ksynMdm2mRNAGSK3bp53 * GSK3b_p53)  # Mdm2mRNASynthesis3
    r6 = (ksynMdm2mRNAGSK3bp53 * GSK3b_p53_P)  # Mdm2mRNASynthesis4
    r7 = (kdegMdm2mRNA * Mdm2_mRNA)  # Mdm2mRNADegradation
    r8 = (kbinMdm2p53 * p53 * Mdm2)  # P53Mdm2Binding
    r9 = (krelMdm2p53 * Mdm2_p53)  # P53Mdm2Release
    r10 = (kbinGSK3bp53 * GSK3b * p53)  # GSK3p53Binding
    r11 = (krelGSK3bp53 * GSK3b_p53)  # GSK3p53Release
    r12 = (kbinGSK3bp53 * GSK3b * p53_P)  # GSK3p53PBinding
    r13 = (krelGSK3bp53 * GSK3b_p53_P)  # GSK3_p53PRelease
    r14 = ((kbinE1Ub * E1 * Ub * ATP) / (5000 + ATP))  # E1UbBinding
    r15 = (kbinE2Ub * E2 * E1_Ub)  # E2UbBinding
    r16 = (kMdm2Ub * Mdm2 * E2_Ub)  # Mdm2Ubiquitination
    r17 = (kMdm2PolyUb * Mdm2_Ub * E2_Ub)  # Mdm2polyUbiquitination1
    r18 = (kMdm2PolyUb * Mdm2_Ub2 * E2_Ub)  # Mdm2polyUbiquitination2
    r19 = (kMdm2PolyUb * Mdm2_Ub3 * E2_Ub)  # Mdm2polyUbiquitination3
    r20 = (kactDUBMdm2 * Mdm2_Ub4 * Mdm2DUB)  # Mdm2Deubiquitination4
    r21 = (kactDUBMdm2 * Mdm2_Ub3 * Mdm2DUB)  # Mdm2Deubiquitination3
    r22 = (kactDUBMdm2 * Mdm2_Ub2 * Mdm2DUB)  # Mdm2Deubiquitination2
    r23 = (kactDUBMdm2 * Mdm2_Ub * Mdm2DUB)  # Mdm2Deubiquitination1
    r24 = (kbinProt * Mdm2_Ub4 * Proteasome)  # Mdm2ProteasomeBinding1
    r25 = (kdegMdm2 * Mdm2_Ub4_Proteasome * kproteff)  # Mdm2Degradation
    r26 = (ksynp53 * p53_mRNA)  # p53Synthesis
    r27 = (kp53Ub * E2_Ub * Mdm2_p53)  # p53Monoubiquitination
    r28 = (kp53PolyUb * Mdm2_p53_Ub * E2_Ub)  # p53Polyubiquitination1
    r29 = (kp53PolyUb * Mdm2_p53_Ub2 * E2_Ub)  # p53Polyubiquitination2
    r30 = (kp53PolyUb * Mdm2_p53_Ub3 * E2_Ub)  # p53Polyubiquitination3
    r31 = (kactDUBp53 * Mdm2_p53_Ub4 * p53DUB)  # p53Deubiqutination4
    r32 = (kactDUBp53 * Mdm2_p53_Ub3 * p53DUB)  # p53Deubiquitination3
    r33 = (kactDUBp53 * Mdm2_p53_Ub2 * p53DUB)  # p53Deubiquitination2
    r34 = (kactDUBp53 * Mdm2_p53_Ub * p53DUB)  # p53Deubiquitination1
    r35 = (kphosMdm2GSK3b * Mdm2_p53_Ub4 * GSK3b)  # Mdm2GSK3phosphorylation1
    r36 = (kphosMdm2GSK3bp53 * Mdm2_p53_Ub4 * GSK3b_p53)  # Mdm2GSK3phosphorylation2
    r37 = (kphosMdm2GSK3bp53 * Mdm2_p53_Ub4 * GSK3b_p53_P)  # Mdm2GSK3phosphorylation3
    r38 = (kbinProt * Mdm2_P1_p53_Ub4 * Proteasome)  # p53ProteasomeBinding1
    r39 = ((kdegp53 * kproteff * p53_Ub4_Proteasome * ATP) / (5000 + ATP))  # Degradationp53_Ub4
    r40 = (kbinMTTau * Tau)  # TauMTbinding
    r41 = (krelMTTau * MT_Tau)  # TauMTrelease
    r42 = (kphospTauGSK3bp53 * GSK3b_p53 * Tau)  # Tauphosphorylation1
    r43 = (kphospTauGSK3bp53 * GSK3b_p53 * Tau_P1)  # Tauphosphorylation2
    r44 = (kphospTauGSK3bp53 * GSK3b_p53_P * Tau)  # Tauphosphorylation3
    r45 = (kphospTauGSK3bp53 * GSK3b_p53_P * Tau_P1)  # Tauphosphorylation4
    r46 = (kphospTauGSK3b * GSK3b * Tau)  # Tauphosphorylation5
    r47 = (kphospTauGSK3b * GSK3b * Tau_P1)  # Tauphosphorylation6
    r48 = (kdephospTau * Tau_P2 * PP1)  # Taudephosphorylation1
    r49 = (kdephospTau * Tau_P1 * PP1)  # Taudephosphorylation2
    r50 = (kaggTauP1 * (Tau_P1 ** 2) * 0.5)  # TauP1Aggregation1
    r51 = (kaggTauP1 * Tau_P1 * AggTau)  # TauP1Aggregation2
    r52 = (kaggTauP2 * (Tau_P2 ** 2) * 0.5)  # TauP2Aggregation1
    r53 = (kaggTauP2 * Tau_P2 * AggTau)  # TauP2Aggregation2
    r54 = (kaggTau * (Tau ** 2) * 0.5)  # TauAggregation1
    r55 = (kaggTau * Tau * AggTau)  # TauAggregation2
    r56 = (ktangfor * (AggTau ** 2) * 0.5)  # TangleFormation1
    r57 = (ktangfor * AggTau * NFT)  # TangleFormation2
    r58 = (kinhibprot * AggTau * Proteasome)  # ProteasomeInhibitionAggTau
    r59 = (kprodAbeta * Source)  # Abetaproduction1
    r60 = (kprodAbeta2 * GSK3b_p53)  # Abetaproduction2
    r61 = (kprodAbeta2 * GSK3b_p53_P)  # Abetaproduction3
    r62 = (kinhibprot * AbetaDimer * Proteasome)  # ProteasomeInhibitionAbeta
    r63 = (kdegAbeta * Abeta)  # AbetaDegradation
    r64 = (ksynp53mRNAAbeta * Abeta)  # p53transcriptionViaAbeta
    r65 = (kdam * IR)  # DNAdamage
    r66 = (krepair * damDNA)  # DNArepair
    r67 = (kactATM * damDNA * ATMI)  # ATMactivation
    r68 = (kphosp53 * p53 * ATMA)  # p53phosphorylation
    r69 = (kdephosp53 * p53_P)  # p53dephosphorylation
    r70 = (kphosMdm2 * Mdm2 * ATMA)  # Mdm2phosphorylation
    r71 = (kdephosMdm2 * Mdm2_P)  # Mdm2dephosphorylation
    r72 = (kMdm2PUb * Mdm2_P * E2_Ub)  # Mdm2PUbiquitination
    r73 = (kMdm2PolyUb * Mdm2_P_Ub * E2_Ub)  # Mdm2PpolyUbiquitination1
    r74 = (kMdm2PolyUb * Mdm2_P_Ub2 * E2_Ub)  # Mdm2PpolyUbiquitination2
    r75 = (kMdm2PolyUb * Mdm2_P_Ub3 * E2_Ub)  # Mdm2PpolyUbiquitination3
    r76 = (kactDUBMdm2 * Mdm2_P_Ub4 * Mdm2DUB)  # Mdm2PDeubiquitination4
    r77 = (kactDUBMdm2 * Mdm2_P_Ub3 * Mdm2DUB)  # Mdm2PDeubiquitination3
    r78 = (kactDUBMdm2 * Mdm2_P_Ub2 * Mdm2DUB)  # Mdm2PDeubiquitination2
    r79 = (kactDUBMdm2 * Mdm2_P_Ub * Mdm2DUB)  # Mdm2PDeubiquitination1
    r80 = (kbinProt * Mdm2_P_Ub4 * Proteasome)  # Mdm2PProteasomeBinding1
    r81 = (kdegMdm2 * Mdm2_P_Ub4_Proteasome * kproteff)  # Mdm2PDegradation
    r82 = (kinactATM * ATMA)  # ATMInactivation
    r83 = (kgenROSAbeta * Abeta)  # AbetaROSproduction1
    r84 = (kgenROSPlaque * AbetaPlaque)  # PlaqueROSproduction
    r85 = (kgenROSAbeta * AggAbeta_Proteasome)  # AggAbetaROSproduction2
    r86 = (kdamROS * ROS)  # ROSDNAdamage
    r87 = (ksynTau * Source)  # TauSynthesis
    r88 = (kbinTauProt * Tau * Proteasome)  # TauProteasomeBinding
    r89 = (kdegTau20SProt * Proteasome_Tau)  # Tau20SProteasomeDegradation
    r90 = (kaggAbeta * (Abeta ** 2) * 0.5)  # AbetaAggregation1
    r91 = (kpf * (AbetaDimer ** 2) * 0.5)  # AbetaPlaqueFormation1
    r92 = ((kpg * AbetaDimer * (AbetaPlaque ** 2)) / ((kpghalf ** 2) + (AbetaPlaque ** 2)))  # AbetaPlaqueGrowth
    r93 = (kdisaggAbeta * AbetaDimer)  # AbetaDisaggregation1
    r94 = (kdisaggAbeta1 * AbetaPlaque)  # AbetaDisaggregation3
    r95 = (kdisaggAbeta2 * antiAb * AbetaPlaque)  # AbetaDisaggregation4
    r96 = (kbinAbantiAb * Abeta * antiAb)  # Abeta_antiAbBinding
    r97 = (kbinAbantiAb * AbetaDimer * antiAb)  # AbetaDimer_antiAbBinding
    r98 = (10 * kdegAbeta * Abeta_antiAb)  # Abeta_antiAbDegredation
    r99 = (10 * kdegAbeta * AbetaDimer_antiAb)  # AbetaDimer_antiAbDegredation
    r100 = (kactglia1 * GliaI * AbetaPlaque)  # GliaActivationStep1
    r101 = (kactglia1 * GliaM1 * AbetaPlaque)  # GliaActivationStep2
    r102 = (kactglia2 * GliaM2 * antiAb)  # GliaActivationStep3
    r103 = (kinactglia1 * GliaA)  # GliaInactivationStep1
    r104 = (kinactglia2 * GliaM2)  # GliaInactivationStep2
    r105 = (kinactglia2 * GliaM1)  # GliaInactivationStep3
    r106 = (kbinAbetaGlia * AbetaPlaque * GliaA)  # AbetaBindingToGlia
    r107 = (krelAbetaGlia * AbetaPlaque_GliaA)  # AbetaReleaseFromGlia
    r108 = (kdegAbetaGlia * AbetaPlaque_GliaA)  # AbetaPlaqueClearanceByGlia
    r109 = (kgenROSGlia * AbetaPlaque_GliaA)  # ROSgenerationByGlia
    r110 = (kdegAntiAb * antiAb)  # antiAbRemoval
    r111 = (kremROS * ROS)  # ROSremoval
    # dy/dt accumulation
    d_Mdm2 = r2 - r8 + r9 - r16 + r23 + r38 - r70 + r71
    d_p53 = - r3 + r3 - r8 + r9 - r10 + r11 + r26 - r68 + r69
    d_Mdm2_p53 = r8 - r9 - r27 + r34
    d_Mdm2_mRNA = - r2 + r2 + r3 + r4 + r5 + r6 - r7
    d_p53_mRNA = r0 - r1 - r26 + r26 + r64
    d_ATMA = r67 - r68 + r68 - r70 + r70 - r82
    d_ATMI = - r67 + r82
    d_p53_P = - r4 + r4 - r12 + r13 + r68 - r69
    d_Mdm2_P = r70 - r71 - r72 + r79
    d_IR = - r65 + r65
    d_ROS = r83 + r84 + r85 - r86 + r86 + r109 - r111
    d_damDNA = r65 - r66 - r67 + r67 + r86
    d_E1 = - r14 + r15
    d_E2 = - r15 + r16 + r17 + r18 + r19 + r27 + r28 + r29 + r30 + r72 + r73 + r74 + r75
    d_E1_Ub = r14 - r15
    d_E2_Ub = r15 - r16 - r17 - r18 - r19 - r27 - r28 - r29 - r30 - r72 - r73 - r74 - r75
    d_Proteasome = - r24 + r25 - r38 + r39 - r58 - r62 - r80 + r81 - r88 + r89
    d_Ub = - r14 + r20 + r21 + r22 + r23 + 4 * r25 + r31 + r32 + r33 + r34 + 4 * r39 + r76 + r77 + r78 + r79 + 4 * r81
    d_p53DUB = - r31 + r31 - r32 + r32 - r33 + r33 - r34 + r34
    d_Mdm2DUB = - r20 + r20 - r21 + r21 - r22 + r22 - r23 + r23 - r76 + r76 - r77 + r77 - r78 + r78 - r79 + r79
    d_DUB = 0.0 * DUB
    d_Mdm2_p53_Ub = r27 - r28 + r33 - r34
    d_Mdm2_p53_Ub2 = r28 - r29 + r32 - r33
    d_Mdm2_p53_Ub3 = r29 - r30 + r31 - r32
    d_Mdm2_p53_Ub4 = r30 - r31 - r35 - r36 - r37
    d_Mdm2_P1_p53_Ub4 = r35 + r36 + r37 - r38
    d_Mdm2_Ub = r16 - r17 + r22 - r23
    d_Mdm2_Ub2 = r17 - r18 + r21 - r22
    d_Mdm2_Ub3 = r18 - r19 + r20 - r21
    d_Mdm2_Ub4 = r19 - r20 - r24
    d_Mdm2_P_Ub = r72 - r73 + r78 - r79
    d_Mdm2_P_Ub2 = r73 - r74 + r77 - r78
    d_Mdm2_P_Ub3 = r74 - r75 + r76 - r77
    d_Mdm2_P_Ub4 = r75 - r76 - r80
    d_p53_Ub4_Proteasome = r38 - r39
    d_Mdm2_Ub4_Proteasome = r24 - r25
    d_Mdm2_P_Ub4_Proteasome = r80 - r81
    d_GSK3b = - r10 + r11 - r12 + r13 - r35 + r35 - r46 + r46 - r47 + r47
    d_GSK3b_p53 = - r5 + r5 + r10 - r11 - r36 + r36 - r42 + r42 - r43 + r43 - r60 + r60
    d_GSK3b_p53_P = - r6 + r6 + r12 - r13 - r37 + r37 - r44 + r44 - r45 + r45 - r61 + r61
    d_Abeta = r59 + r60 + r61 - r63 - r64 + r64 - r83 + r83 - 2 * r90 + 2 * r93 - r96
    d_AggAbeta_Proteasome = r62 - r85 + r85
    d_AbetaPlaque = - r84 + r84 + r91 - r92 + 2 * r92 - r94 - r95 - r100 + r100 - r101 + r101 - r106 + r107
    d_Tau = - r40 + r41 - r42 - r44 - r46 + r49 - 2 * r54 - r55 + r87 - r88
    d_Tau_P1 = r42 - r43 + r44 - r45 + r46 - r47 + r48 - r49 - 2 * r50 - r51
    d_Tau_P2 = r43 + r45 + r47 - r48 - 2 * r52 - r53
    d_MT_Tau = r40 - r41
    d_AggTau = 2 * r50 - r51 + 2 * r51 + 2 * r52 - r53 + 2 * r53 + 2 * r54 - r55 + 2 * r55 - 2 * r56 - r57 - r58
    d_AggTau_Proteasome = r58
    d_Proteasome_Tau = r88 - r89
    d_PP1 = - r48 + r48 - r49 + r49
    d_NFT = 2 * r56 - r57 + 2 * r57
    d_AbetaDimer = - r62 + r90 - 2 * r91 - r92 - r93 + r94 + r95 - r97
    d_AbetaPlaque_GliaA = r106 - r107 - r108 - r109 + r109
    d_GliaI = - r100 + r105
    d_GliaM1 = r100 - r101 + r104 - r105
    d_GliaM2 = r101 - r102 + r103 - r104
    d_GliaA = r102 - r103 - r106 + r107 + r108
    d_antiAb = - r95 + r95 - r96 - r97 + r98 + r99 - r102 + r102 - r110
    d_Abeta_antiAb = r96 - r98
    d_AbetaDimer_antiAb = r97 - r99
    d_degAbetaGlia = r108
    d_disaggPlaque1 = r94
    d_disaggPlaque2 = r95
    return jnp.stack([
        d_Mdm2,
        d_p53,
        d_Mdm2_p53,
        d_Mdm2_mRNA,
        d_p53_mRNA,
        d_ATMA,
        d_ATMI,
        d_p53_P,
        d_Mdm2_P,
        d_IR,
        d_ROS,
        d_damDNA,
        d_E1,
        d_E2,
        d_E1_Ub,
        d_E2_Ub,
        d_Proteasome,
        d_Ub,
        d_p53DUB,
        d_Mdm2DUB,
        d_DUB,
        d_Mdm2_p53_Ub,
        d_Mdm2_p53_Ub2,
        d_Mdm2_p53_Ub3,
        d_Mdm2_p53_Ub4,
        d_Mdm2_P1_p53_Ub4,
        d_Mdm2_Ub,
        d_Mdm2_Ub2,
        d_Mdm2_Ub3,
        d_Mdm2_Ub4,
        d_Mdm2_P_Ub,
        d_Mdm2_P_Ub2,
        d_Mdm2_P_Ub3,
        d_Mdm2_P_Ub4,
        d_p53_Ub4_Proteasome,
        d_Mdm2_Ub4_Proteasome,
        d_Mdm2_P_Ub4_Proteasome,
        d_GSK3b,
        d_GSK3b_p53,
        d_GSK3b_p53_P,
        d_Abeta,
        d_AggAbeta_Proteasome,
        d_AbetaPlaque,
        d_Tau,
        d_Tau_P1,
        d_Tau_P2,
        d_MT_Tau,
        d_AggTau,
        d_AggTau_Proteasome,
        d_Proteasome_Tau,
        d_PP1,
        d_NFT,
        d_AbetaDimer,
        d_AbetaPlaque_GliaA,
        d_GliaI,
        d_GliaM1,
        d_GliaM2,
        d_GliaA,
        d_antiAb,
        d_Abeta_antiAb,
        d_AbetaDimer_antiAb,
        d_degAbetaGlia,
        d_disaggPlaque1,
        d_disaggPlaque2,
    ])
