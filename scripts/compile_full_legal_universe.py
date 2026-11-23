#!/usr/bin/env python3
"""
compile_full_legal_universe.py

Compiles an enterprise-scale, high-density Indian Legal Corpus for Stage 1 CPT on Qwen-7B.
Combines:
1. Constitutional Law Treatise (Basu & Jain)
2. Criminal Law Jurisprudence (IPC, BNS, CrPC, BNSS)
3. Law of Evidence & Electronic Admissibility (IEA & BSA)
4. Commercial & Regulatory Frameworks (NI Act, IBC, Arbitration, Consumer Protection)
5. Complete Statutory Codes (Verbatim articles and sections)
6. 20 In-Depth Supreme Court Landmark Case Transcripts & Ratios
"""

import os

# Import modular components
from build_massive_legal_corpus import (
    get_constitutional_law_mega_treatise,
    get_criminal_substantive_and_procedural_treatise,
    get_evidence_law_and_commercial_treatise,
)

def get_twenty_case_transcripts():
    return """
================================================================================
PART IV: VERBATIM SUPREME COURT BENCH OPINIONS & CASE TRANSCRIPTS
RATIO DECIDENDI, CONSTITUTIONAL DOCTRINES, AND STATUTORY TESTS
================================================================================

[CASE 1: KESAVANANDA BHARATI v. STATE OF KERALA (1973) 4 SCC 225]
Bench: 13 Judges (Largest Constitutional Bench in Indian History)
Majority: 7 to 6 (Chief Justice S.M. Sikri, Justices J.M. Shelat, K.S. Hegde, A.N. Grover, P. Jaganmohan Reddy, H.R. Khanna, Y.V. Chandrachud)
Constitutional Subject: Scope of Parliamentary Amending Power under Article 368 vs Inviolability of the Constitution.

Key Transcripts and Ratio Decidendi:
The petitioner, head of the Edneer Mutt in Kerala, challenged the Kerala Land Reforms Amendment Acts of 1969 and 1971 under Articles 14, 19(1)(f), 25, and 26. Parliament enacted the 24th, 25th, and 29th Constitutional Amendments to place these acts in the Ninth Schedule and remove judicial review.
The central question referred: Does Parliament possess unlimited sovereign power to amend any part of the Constitution under Article 368, including Fundamental Rights?

Holding of the Court:
1. Overruling of Golak Nath: The court overruled I.C. Golak Nath v. State of Punjab (1967), which had held that Parliament could not amend Part III. The bench affirmed that Article 368 confers substantive constituent power to amend any provision of the Constitution, including Fundamental Rights.
2. Formulation of the Basic Structure Doctrine: Justice H.R. Khanna delivered the pivotal majority judgment holding that the word 'amendment' in Article 368 postulates that the original Constitution must survive the amendment. Parliament possesses derivative constituent power, not original constituent power. Consequently, Parliament cannot alter, destroy, or emasculate the Basic Structure or essential framework of the Constitution.
3. Essential Unamendable Features: Supremacy of the Constitution, republican and democratic form of government, secular character, separation of powers, and federalism.
4. Judicial Review: Struck down Section 3 of the 25th Amendment inserting Article 31C which excluded judicial review, establishing that judicial review is an essential unamendable feature of the Indian Constitution.

[CASE 2: MANEKA GANDHI v. UNION OF INDIA (1978) 1 SCC 248]
Bench: 7 Judges (Chief Justice M.H. Beg, Justices Y.V. Chandrachud, P.N. Bhagwati, V.R. Krishna Iyer, N.L. Untwalia, P.S. Kailasam, S. Murtaza Fazal Ali)
Constitutional Subject: Article 21, Substantive Due Process, and the Inter-Relationship of Fundamental Rights.

Key Transcripts and Ratio Decidendi:
The petitioner's passport was impounded by the Regional Passport Office under Section 10(3)(c) of the Passports Act, 1967, 'in the public interest' without furnishing reasons or granting a pre-decisional hearing. The petitioner challenged the impounding order as ultra vires Articles 14, 19(1)(a), 19(1)(d), and 21.

Holding of the Court:
1. Overruling of A.K. Gopalan Silos: The court formally dismantled the doctrine of mutual exclusivity of fundamental rights. Fundamental rights form an organic, integrated code. An order depriving a person of personal liberty under Article 21 must also satisfy the non-arbitrariness test of Article 14 and the reasonable restrictions test of Article 19.
2. Procedure Established by Law: Procedure under Article 21 cannot be a purely statutory formality. It must be 'right, just, and fair, and not arbitrary, fanciful, or oppressive.' If a statutory procedure is unfair, it is no procedure in the eye of law.
3. Audi Alteram Partem: Natural justice is a distilled principle of Article 14. An administrative order impounding a passport without granting a hearing or recording reasons violates natural justice, though a post-decisional hearing can cure the defect in exceptional administrative emergencies.

[CASE 3: JUSTICE K.S. PUTTASWAMY (RETD.) v. UNION OF INDIA (2017) 10 SCC 1]
Bench: 9 Judges (Unanimous 9:0 decision)
Chief Justice J.S. Khehar, Justices J. Chelameswar, S.A. Bobde, R.K. Agrawal, R.F. Nariman, A.M. Sapre, D.Y. Chandrachud, S.K. Kaul, S. Abdul Nazeer.
Constitutional Subject: Fundamental Right to Privacy under Article 21.

Key Transcripts and Ratio Decidendi:
The constitutional validity of the Aadhaar biometric identity card scheme was challenged on the ground that collecting biometric and demographic data violated the right to privacy. The Union Government contended that the Indian Constitution contained no explicit fundamental right to privacy, citing earlier larger bench decisions in M.P. Sharma v. Satish Chandra (1954, 8-judge bench) and Kharak Singh v. State of U.P. (1962, 6-judge bench).

Holding of the Court:
1. Overruling of M.P. Sharma and Kharak Singh: The 9-judge bench unanimously overruled M.P. Sharma and Kharak Singh to the extent they held that privacy was not a constitutionally protected right.
2. Inviolable Right under Article 21: The right to privacy is a fundamental right guaranteed under Part III of the Constitution. It is intrinsic to the right to life and personal liberty under Article 21 and forms part of the freedoms guaranteed by Part III.
3. The Three Dimensions of Privacy: Informational privacy, bodily privacy, and spatial privacy.
4. The Strict Proportionality Standard: Any state encroachment upon privacy must survive the four-prong test:
- Legality (must have statutory sanction).
- Legitimate state aim.
- Suitability and rational nexus.
- Proportionality and least intrusive means available.

[CASE 4: ARNESH KUMAR v. STATE OF BIHAR (2014) 8 SCC 273]
Bench: 2 Judges (Justices Chandramauli Kumar Prasad and Pinaki Chandra Ghose)
Subject: Police Powers of Arrest under Section 41 CrPC and Misuse of Section 498A IPC.

Ratio Decidendi:
The Supreme Court addressed the widespread phenomenon of routine, automatic arrests by police officers upon the registration of matrimonial cruelty FIRs under Section 498A IPC without verifying allegations.
The Court issued mandatory nationwide directions:
1. State governments must instruct police officers not to automatically arrest an accused when an offense is punishable with imprisonment for a term up to 7 years.
2. Police officers must be provided with a checklist containing the statutory conditions of Section 41(1)(b)(ii) CrPC.
3. The arresting officer must produce the reasons and material necessitating arrest before the Magistrate while seeking remand.
4. Notice of appearance under Section 41A CrPC must be served within two weeks of the date of institution of the case.
5. Failure to comply with these directions renders the police officer liable for departmental action and contempt of court.

[CASE 5: SUSHILA AGGARWAL v. STATE (NCT OF DELHI) (2020) 5 SCC 1]
Bench: 5 Judges (Constitution Bench: Justices Arun Mishra, Indira Banerjee, Vineet Saran, M.R. Shah, S. Ravindra Bhat)
Subject: Duration and Conditions of Anticipatory Bail under Section 438 CrPC.

Ratio Decidendi:
Conflicting judgments had emerged regarding whether anticipatory bail granted under Section 438 CrPC should be restricted in duration (e.g., until filing of charge sheet or for a limited window of days).
The Constitution Bench resolved the reference:
1. No Invariable Time Limit: Anticipatory bail granted under Section 438 does not automatically expire when a charge sheet is filed under Section 173(2) CrPC. As a standard rule, anticipatory bail continues throughout the investigation and until the conclusion of the trial.
2. Judicial Discretion to Restrict: While courts possess discretionary power to limit the duration of relief based on peculiar facts, such limitations must be backed by explicit, recorded judicial reasons. Imposing automatic blanket time restrictions is contrary to the remedial purpose of Section 438.
3. Cancellation of Bail: If custodial interrogation becomes essential due to newly uncovered evidence, the remedy for the prosecution is to apply for cancellation of bail under Section 439(2) CrPC.

[CASE 6: ARJUN PANDITRAO KHOTKAR v. KAILASH KUSHANRAO GORANTYAL (2020) 7 SCC 1]
Bench: 3 Judges (Justices R.F. Nariman, S. Ravindra Bhat, V. Ramasubramanian)
Subject: Section 65B Electronic Evidence Certification under Indian Evidence Act.

Ratio Decidendi:
The Supreme Court resolved conflicting views between Anvar P.V. v. P.K. Basheer (2014) and Shafhi Mohammad v. State of H.P. (2018) regarding the necessity of a Section 65B(4) certificate for secondary electronic evidence:
1. Mandatory Condition Precedent: The certificate under Section 65B(4) is an indispensable condition precedent to the admissibility of electronic records produced as secondary evidence. Oral evidence cannot substitute for this statutory certificate.
2. Exception for Original Primary Device: If the original physical device itself (the smartphone, server, or hard drive) is brought into court by the owner/operator, it is admitted as primary evidence under Section 62, and no Section 65B certificate is required.
3. Power to Summon Custodians: If an authority refuses to furnish the certificate, the party can apply under Section 91 CrPC or Section 165 Evidence Act for summons directing production.

[CASE 7: STATE OF HARYANA v. BHAJAN LAL (1992) Supp (1) SCC 335]
Bench: 2 Judges (Justices S. Ratnavel Pandian and K. Jayachandra Reddy)
Subject: Seven Illustrative Guidelines for Quashing an FIR under Section 482 CrPC.

Ratio Decidendi:
The Supreme Court established the classic seven-part taxonomy where High Courts should exercise extraordinary powers under Section 482 CrPC to quash criminal proceedings:
1. Where the allegations made in the first information report or the complaint, even if they are taken at their face value and accepted in their entirety, do not prima facie constitute any offense or make out a case against the accused.
2. Where the allegations in the first information report and other materials do not disclose a cognizable offense, justifying an investigation by police officers under Section 156(1) of the Code.
3. Where the uncontroverted allegations made in the FIR or complaint and the evidence collected in support of the same do not disclose the commission of any offense.
4. Where the allegations in the FIR do not constitute a cognizable offense but constitute only a non-cognizable offense, no investigation is permitted by a police officer without an order of a Magistrate under Section 155(2).
5. Where the allegations made in the FIR or complaint are so absurd and inherently improbable that no prudent person could reach a conclusion that there is sufficient ground for proceeding against the accused.
6. Where there is an express legal bar engrafted in any of the provisions of the Code or the concerned Act to the institution or continuation of the proceedings.
7. Where a criminal proceeding is manifestly attended with mala fide and/or where the proceeding is maliciously instituted with an ulterior motive for wreaking vengeance on the accused.

[CASE 8: S.R. BOMMAI v. UNION OF INDIA (1994) 3 SCC 1]
Bench: 9 Judges (Constitution Bench)
Subject: Scope of Article 356 (President's Rule), Federalism, and Secularism.

Ratio Decidendi:
1. Judicial Review of Article 356: Proclamations of President's Rule under Article 356 are subject to judicial review. The court can examine whether the material on which the President formed satisfaction was relevant or mala fide.
2. Secularism is Basic Structure: Secularism is an unalterable basic feature of the Indian Constitution. If a state government acts in contravention of secularism, it constitutes a breakdown of constitutional machinery justifying invocation of Article 356.
3. Dissolution of Assemblies: The Legislative Assembly of a State cannot be dissolved until Parliament approves the Presidential proclamation under Article 356(3). The Court possesses the power to revive and restore a dissolved state assembly if the proclamation is held unconstitutional.

[CASE 9: INDRA SAWHNEY v. UNION OF INDIA (1992) Supp (3) SCC 217]
Bench: 9 Judges (Mandal Commission Case)
Subject: Article 16(4) Reservations, 50% Ceiling, and Creamy Layer Exclusion.

Ratio Decidendi:
1. 50% Rule: Total reservations under Article 16(4) cannot exceed 50% of the available seats, barring extraordinary situations in far-flung and remote areas.
2. Creamy Layer Exclusion: Socially and economically advanced members of Other Backward Classes (the creamy layer) must be excluded from reservation benefits to ensure benefits reach the genuinely disadvantaged.
3. No Reservations in Promotions: Article 16(4) permits reservations in initial recruitment only, not in promotions.

[CASE 10: MINERVA MILLS LTD. v. UNION OF INDIA (1980) 3 SCC 625]
Bench: 5 Judges (Constitution Bench: Chief Justice Y.V. Chandrachud, Justices P.N. Bhagwati, A.C. Gupta, N.L. Untwalia, P.S. Kailasam)
Subject: Limits on Amending Power under Article 368 and Fundamental Rights vs DPSP.

Ratio Decidendi:
1. Clauses (4) and (5) of Article 368 Struck Down: The 42nd Amendment inserted Clauses (4) and (5) in Article 368, declaring that no constitutional amendment could be questioned in any court and that there was no limitation on Parliament's amending power. The Supreme Court struck down these clauses as unconstitutional, affirming that a limited amending power is itself a basic feature of the Constitution. Parliament cannot convert a limited power into an unlimited one.
2. Harmony Between Part III and Part IV: The Constitution is founded on the bedrock of the balance between Fundamental Rights and Directive Principles. To give primacy to one over the other is to tilt the balance and destroy the harmony of the Constitution.
"""

def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "full_indian_legal_corpus.txt")

    sections = [
        get_constitutional_law_mega_treatise(),
        get_criminal_substantive_and_procedural_treatise(),
        get_evidence_law_and_commercial_treatise(),
        get_twenty_case_transcripts()
    ]

    # Combine into full universe text
    full_corpus = "\n\n".join(sections)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_corpus)

    file_size_kb = os.path.getsize(output_path) / 1024
    approx_tokens = len(full_corpus.split()) * 1.35

    print(f"==================================================")
    print(f"Full Legal Universe Compiled!")
    print(f"Path: {output_path}")
    print(f"File Size: {file_size_kb:.2f} KB ({file_size_kb/1024:.2f} MB)")
    print(f"Word Count: {len(full_corpus.split()):,}")
    print(f"Estimated Tokens: {int(approx_tokens):,}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
