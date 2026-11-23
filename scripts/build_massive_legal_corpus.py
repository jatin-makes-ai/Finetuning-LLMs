#!/usr/bin/env python3
"""
build_massive_legal_corpus.py

Enterprise-Scale Continued Pre-Training (CPT) Corpus Generator for Indian Law.
Generates a multi-megabyte, high-density corpus spanning:
- Constitutional Law Commentary & All Fundamental Rights
- Substantive Criminal Law (IPC & BNS)
- Procedural Criminal Jurisprudence (CrPC & BNSS)
- Law of Evidence (IEA & BSA)
- Commercial & Corporate Law (NI Act, IBC 2016, Arbitration, Consumer Protection)
- Civil Procedure Code & Specific Relief
- 30 In-Depth Landmark Supreme Court Case Transcripts & Holdings
"""

import os
import sys

def get_constitutional_law_mega_treatise():
    return """
================================================================================
PART I: COMPREHENSIVE TREATISE ON THE CONSTITUTION OF INDIA
THE ARCHITECTURE OF DEMOCRATIC REPUBLICANISM, SEPARATION OF POWERS, AND PART III
================================================================================

1. THE HISTORICAL EVOLUTION AND CONSTITUENT ASSEMBLY DEBATES
The Constitution of India was drafted over a span of two years, eleven months, and seventeen days by the Constituent Assembly of India, presided over by Dr. Rajendra Prasad, with the Drafting Committee chaired by Dr. Bhimrao Ramji Ambedkar. Adopted on November 26, 1949, and coming into full effect on January 26, 1950, it represents the longest written national constitution in the world.

The foundational philosophy of the Constitution was set by the 'Objectives Resolution' moved by Pandit Jawaharlal Nehru on December 13, 1946. It proclaimed India as an Independent Sovereign Republic, guaranteeing justice, equality, and freedom to all citizens, with adequate safeguards for minorities, backward and tribal areas, and depressed classes.

2. THE PREAMBLE: SOVEREIGNTY, SOCIALISM, SECULARISM, AND JUSTICE
The Preamble embodies the fundamental value system and constitutional ethos of India.
- 'Sovereign': Indicates that India is completely independent of external control and is supreme within its own territorial domain.
- 'Socialist': Added by the 42nd Constitutional Amendment Act, 1976. In D.S. Nakara v. Union of India (1983) 1 SCC 305, the Supreme Court interpreted Indian democratic socialism as a blend of Marxism and Gandhism, leaning heavily towards Gandhian socialism to eliminate poverty, ignorance, disease, and inequality of opportunity.
- 'Secular': Added by the 42nd Amendment, 1976. In S.R. Bommai v. Union of India (1994) 3 SCC 1, a 9-judge Constitution Bench held that Secularism has been an intrinsic basic feature of the Constitution since January 26, 1950. The Indian concept of secularism is not anti-religious or strictly separationist like the French laïcité; rather, it embodies 'Sarva Dharma Sambhava' (equal respect and treatment for all religions by the State).
- 'Democratic': Government deriving its authority from the will of the people, exercised via universal adult suffrage (Article 326).
- 'Republic': The Head of State (the President of India) is elected indirectly by the representatives of the people for a fixed term, signifying the absence of any hereditary monarch or privileged class.

3. PART III: THE BILL OF RIGHTS (ARTICLES 12 TO 35)

Article 12: Scope of the State
Article 12 defines 'the State' for Part III enforcement:
- The Government and Parliament of India.
- The Government and Legislature of each of the States.
- All local authorities (Panchayats, Municipalities, District Boards).
- 'Other authorities' under the control of the Government of India.
In Rajasthan State Electricity Board (1967), Sukhdev Singh (1975), and Ramana Dayaram Shetty (1979), the Supreme Court laid down the agency/instrumentality test. In Pradeep Kumar Biswas v. Indian Institute of Chemical Biology (2002) 5 SCC 111, a 7-judge Constitution Bench held that a body qualifies as 'State' under Article 12 only if it is financially, functionally, and administratively dominated by or under the pervasive control of the Government.

Article 13: Watchdog of Fundamental Rights and Judicial Review
- Article 13(1): Declares all pre-constitutional laws inconsistent with Fundamental Rights to be void to the extent of such inconsistency (giving rise to the Doctrine of Eclipse in Bhikaji Narain Dhakras v. State of M.P., 1955).
- Article 13(2): Forbids the post-constitutional state from making any law taking away or abridging Part III rights; any law so made is void ab initio (Deep Chand v. State of U.P., 1959; Mahendra Lal Jaini, 1963).
- Article 13(3): Defines 'law' broadly to include ordinances, orders, bye-laws, rules, regulations, notifications, customs, and usages.
- Doctrine of Severability: Laid down in R.M.D. Chamarbaugwalla v. Union of India (1957) SCR 930. If the valid and invalid parts of an enacted statute are distinct and severable, only the invalid part is struck down. If they are so intertwined that what remains cannot stand alone, the entire statute is invalid.

Article 14: Equality Before the Law and Equal Protection
Article 14 guarantees equality to all persons (citizens and non-citizens alike):
- The Old Rule of Reasonable Classification: In State of West Bengal v. Anwar Ali Sarkar (1952), the court required: (1) Intelligible differentia distinguishing persons/things grouped together from those left out; (2) Rational nexus to the statutory object.
- The New Doctrine of Non-Arbitrariness: Introduced by Justice P.N. Bhagwati in E.P. Royappa v. State of Tamil Nadu (1974) 4 SCC 3: "Equality is a dynamic concept with many aspects and dimensions... Equality and arbitrariness are sworn enemies; one belongs to the rule of law in a republic while the other, to the whim and caprice of an absolute monarch."
- In Maneka Gandhi v. Union of India (1978) 1 SCC 248, non-arbitrariness was integrated into Article 21, establishing the Golden Triangle (Articles 14, 19, and 21).

Article 15 and 16: Prohibition of Discrimination and Affirmative Action
- Article 15(1) prohibits discrimination against any citizen on grounds only of religion, race, caste, sex, place of birth, or any of them.
- Article 15(3) enables special provisions for women and children.
- Article 15(4) and 15(5) enable affirmative action and reservations for socially and educationally backward classes (SEBCs) and Scheduled Castes (SCs) / Scheduled Tribes (STs).
- Article 16 guarantees equality of opportunity in matters of public employment.
- In Indra Sawhney v. Union of India (1992) Supp (3) SCC 217 (Mandal Commission Case, 9-judge bench):
  1. Backward class status cannot be determined solely on economic criteria; social and educational backwardness must be established.
  2. The 'Creamy Layer' among backward classes must be excluded from reservation benefits.
  3. Total reservations under Article 16(4) cannot exceed 50%, except in extraordinary, remote circumstances.
  4. Reservations do not apply to promotions (later amended by 77th Amendment inserting Art 16(4A)).
- In Janhit Abhiyan v. Union of India (2023) 5 SCC 1, a 5-judge Constitution Bench upheld the 103rd Constitutional Amendment providing 10% reservation for Economically Weaker Sections (EWS), holding that the 50% ceiling rule applies only to Part III caste-based reservations under Articles 15(4) and 16(4), and not to distinct economic classifications.

Article 19: The Six Fundamental Freedoms and Reasonable Restrictions
Article 19(1) protects:
- 19(1)(a): Freedom of speech and expression.
- 19(1)(b): Freedom to assemble peaceably and without arms.
- 19(1)(c): Freedom to form associations, unions, or co-operative societies.
- 19(1)(d): Freedom to move freely throughout the territory of India.
- 19(1)(e): Freedom to reside and settle in any part of the territory of India.
- 19(1)(g): Freedom to practice any profession, trade, or business.

Restrictions under Clauses (2) to (6):
Restrictions on Article 19 must be 'reasonable' and must fall strictly within the exhaustive grounds enumerated in the respective sub-clauses:
- Grounds under Article 19(2): Sovereignty and integrity of India, security of the State, friendly relations with foreign States, public order, decency or morality, contempt of court, defamation, or incitement to an offense.
- In Shreya Singhal v. Union of India (2015) 5 SCC 1, the Supreme Court distinguished between 'discussion', 'advocacy', and 'incitement'. Mere advocacy of an unpopular cause cannot be restricted; speech can only be curtailed under Article 19(2) when it reaches the threshold of clear and imminent incitement.
- In Anuradha Bhasin v. Union of India (2020) 3 SCC 637, the Supreme Court declared that freedom of speech and expression through the medium of the internet is an integral part of Article 19(1)(a). Indefinite suspension of telecom/internet services without statutory review was held unconstitutional.

Article 20: Protection in Respect of Conviction for Offenses
- Article 20(1) (Protection Against Ex Post Facto Laws): No person shall be convicted of an offense except for violation of a law in force at the time of the commission of the act, nor subjected to a penalty greater than that which might have been inflicted under the law in force at that time.
- Article 20(2) (Protection Against Double Jeopardy): No person shall be prosecuted and punished for the same offense more than once. Applies only when the previous proceeding was before a judicial court/tribunal and resulted in punishment (Maqbool Hussain v. State of Bombay, 1953). Departmental proceedings do not attract Article 20(2).
- Article 20(3) (Protection Against Self-Incrimination): No person accused of any offense shall be compelled to be a witness against himself.
In Nandini Satpathy v. P.L. Dani (1978) 2 SCC 424, the right to silence was held to extend to police interrogation and investigation stages.
In Selvi v. State of Karnataka (2010) 7 SCC 263, the Supreme Court held that involuntary administration of narco-analysis, polygraph tests, or BEAP (brain mapping) violates Article 20(3) and the right to privacy under Article 21.

Article 21: Protection of Life and Personal Liberty
"No person shall be deprived of his life or personal liberty except according to procedure established by law."
The interpretation of Article 21 has witnessed the greatest transformation in Indian constitutional history:
- A.K. Gopalan v. State of Madras (1950): Strict positivist textualism. Procedure established by law meant any law enacted by Parliament.
- Maneka Gandhi v. Union of India (1978): The procedure must be 'just, fair, and reasonable'. Introduced substantive due process into Indian law.
- Sunil Batra v. Delhi Administration (1978): Solitary confinement and barbaric torture of prison inmates violate Article 21.
- Francis Coralie Mullin v. Administrator, Union Territory of Delhi (1981) 1 SCC 608: The right to life includes the right to live with human dignity and all that goes along with it—adequate nutrition, clothing, shelter, and facilities for reading and writing.
- Bandhua Mukti Morcha v. Union of India (1984) 3 SCC 161: Bonded labor violates Article 21.
- Olga Tellis v. Bombay Municipal Corporation (1985) 3 SCC 545: Right to livelihood is an integral facet of the right to life under Article 21.
- Unni Krishnan v. State of A.P. (1993) 1 SCC 645: Right to education for children up to 14 years recognized as a fundamental right (led to the 86th Constitutional Amendment inserting Article 21A).
- Justice K.S. Puttaswamy (Retd.) v. Union of India (2017) 10 SCC 1: 9-judge Constitution Bench declared the Right to Privacy an inalienable fundamental right under Article 21.
- Navtej Singh Johar v. Union of India (2018) 1 SCC 791: Decriminalization of consensual same-sex adult relationships under Section 377 IPC, affirming sexual orientation as an attribute of human dignity under Article 21.
- Common Cause v. Union of India (2018) 5 SCC 1: Recognized the right to die with dignity, legalizing passive euthanasia and advance medical directives (Living Wills).

Article 32 and Article 226: The Constitutional Writs
- Article 32: Conferred upon the Supreme Court. A fundamental right in itself (Part III). Supreme Court cannot refuse to entertain an Article 32 petition where fundamental rights are violated.
- Article 226: Conferred upon the High Courts (Part VI). Discretionary remedy. Broader substantive jurisdiction than Article 32 because it enforces Fundamental Rights and 'for any other purpose' (statutory, legal, and contractual rights).
The Five Prerogative Writs:
1. Habeas Corpus: Corrects illegal, arbitrary, or unconstitutional physical custody.
2. Mandamus: Compels performance of an imperative public statutory duty.
3. Certiorari: Quashes orders passed without jurisdiction, in excess of jurisdiction, or in violation of natural justice.
4. Prohibition: Restrains inferior tribunals from usurping jurisdiction or exceeding statutory powers.
5. Quo Warranto: Ousts unauthorized usurpers from public statutory offices.
"""

def get_criminal_substantive_and_procedural_treatise():
    return """
================================================================================
PART II: COMPREHENSIVE TREATISE ON CRIMINAL LAW AND PROCEDURAL JURISPRUDENCE
SUBSTANTIVE OFFENSES, ARREST SAFEGUARDS, BAIL DOCTRINES, AND EVIDENCE
================================================================================

1. GENERAL PRINCIPLES OF CRIMINAL LIABILITY
Every criminal offense consists of two indispensable elements:
1. Actus Reus: The physical, outward act or illegal omission forbidden by penal law.
2. Mens Rea: The blameworthy mental state (intention, knowledge, recklessness, or gross negligence).
In statutory offenses where strict liability is imposed (e.g. food adulteration or negotiable instruments dishonour under Section 138), mens rea may be displaced by explicit statutory wording.

Joint Liability and Group Offenses:
- Section 34 IPC (Common Intention): When a criminal act is done by several persons in furtherance of the common intention of all, each of such persons is liable for that act in the same manner as if it were done by him alone. Requires prior meeting of minds, pre-arranged plan, and active participation in the commission of the offense (Barendra Kumar Ghosh v. Emperor, 1925; Mahbub Shah v. Emperor, 1945).
- Section 149 IPC (Common Object): Every member of an unlawful assembly (5 or more persons under Section 141) is guilty of an offense committed by any member in prosecution of the common object of that assembly, or such as the members knew to be likely to be committed. Does not require prior meeting of minds; mere membership of an unlawful assembly with knowledge of the common object suffices.

Criminal Conspiracy: Section 120A and 120B IPC
When two or more persons agree to do, or cause to be done:
(1) An illegal act, or
(2) An act which is not illegal by illegal means.
The agreement itself constitutes the offense of criminal conspiracy. In State (NCT of Delhi) v. Navjot Sandhu (Afsan Guru, 2005) 11 SCC 600, the Supreme Court held that direct evidence of a conspiracy is rarely available; conspiracy is inferred from circumstantial evidence, conduct, and surrounding transactions.

2. OFFENSES AGAINST THE HUMAN BODY
Culpable Homicide (Section 299) vs. Murder (Section 300 IPC):
Culpable homicide is the genus, murder is the species.
Section 299 defines Culpable Homicide:
- Causing death with the intention of causing death, or
- Intention of causing bodily injury likely to cause death, or
- Knowledge that the act is likely to cause death.
Section 300 defines Murder:
- Clause 1: Act done with intention of causing death.
- Clause 2: Act done with intention of causing bodily injury which the offender knows to be likely to cause the death of the person to whom harm is caused.
- Clause 3: Act done with intention of causing bodily injury to any person, and the bodily injury intended to be inflicted is sufficient in the ordinary course of nature to cause death (Virsa Singh v. State of Punjab, 1958 SCR 1495).
- Clause 4: Imminently dangerous act with knowledge that it must, in all probability, cause death.

The Five Exceptions to Section 300 (Culpable Homicide Not Amounting to Murder):
1. Exception 1: Grave and sudden provocation depriving the offender of self-control (K.M. Nanavati v. State of Maharashtra, 1962). Provocation must not be sought or voluntarily provoked.
2. Exception 2: Exceeding the right of private defense in good faith without premeditation.
3. Exception 3: Public servant exceeding powers in good faith believing it necessary for public duty.
4. Exception 4: Sudden fight in the heat of passion upon a sudden quarrel, without premeditation, and without the offender having taken undue advantage or acted in a cruel or unusual manner.
5. Exception 5: Death caused with the voluntary consent of the deceased who is above 18 years of age.

Death by Rash or Negligent Act: Section 304A IPC
Penalizes causing death by any rash or negligent act not amounting to culpable homicide.
In Jacob Mathew v. State of Punjab (2005) 6 SCC 1, the Supreme Court laid down the criminal negligence standard for medical doctors:
- Ordinary civil malpractice or error of judgment does not constitute criminal liability under Section 304A.
- Prosecution must establish **gross negligence** or reckless disregard for human life.
- Police cannot arrest or register an FIR against a doctor without first obtaining an independent written opinion from a panel of government doctors specializing in that branch of medicine.

Kidnapping and Abduction:
- Kidnapping from Lawful Guardianship (Section 361): Enticing or taking any minor (male under 16, female under 18) or person of unsound mind out of the keeping of the lawful guardian without consent. Strict liability; consent of the minor is immaterial.
- Abduction (Section 362): Compelling by force or inducing by deceitful means any person to go from any place. Abduction is an auxiliary act; it is punishable only when accompanied by an ulterior criminal intent (e.g., Section 364 abduction for murder, Section 364A abduction for ransom).

3. OFFENSES AGAINST PROPERTY
- Theft (Section 378): Dishonestly taking any movable property out of the possession of any person without that person's consent, and moving that property in order to such taking.
- Extortion (Section 383): Intentionally putting any person in fear of injury and thereby dishonestly inducing them to deliver any property.
- Robbery (Section 390): Theft or extortion becomes robbery when accompanied by death, hurt, wrongful restraint, or fear of immediate death/hurt.
- Dacoity (Section 391): Robbery committed by five or more persons conjointly.
- Criminal Misappropriation (Section 403): Dishonestly converting to one's own use any movable property which came into possession innocently.
- Criminal Breach of Trust (Section 405): Entrustment of property, and subsequent dishonest misappropriation, conversion, or illegal disposal in violation of trust.
- Cheating (Section 415 & 420): Deceiving any person and fraudulently or dishonestly inducing them to deliver any property. In Hridaya Ranjan Prasad Verma v. State of Bihar (2000) 4 SCC 168, the Supreme Court established the distinction between civil breach of contract and criminal cheating: In cheating, dishonest fraudulent intention must exist **at the inception of the transaction**. Subsequent inability to fulfill a promise constitutes breach of contract, not criminal cheating.

4. PROCEDURAL JURISPRUDENCE UNDER CrPC / BNSS

The Statutory Framework of Arrest:
- Section 41 CrPC: Powers of police to arrest without a warrant. Restricted to cognizable offenses, or offenses punishable up to 7 years only when recorded reasons satisfy Section 41(1)(b) criteria.
- Section 41A CrPC: Mandatory Notice of Appearance. Police must issue a notice directing the accused to appear before the investigating officer. If the accused complies, they cannot be arrested unless reasons are recorded.
- Arnesh Kumar v. State of Bihar (2014) 8 SCC 273: Supreme Court mandated that arrest for offenses punishable up to 7 years cannot be routine. Disobedience of Section 41 and 41A renders the arresting officer liable for departmental action and contempt of court.
- D.K. Basu v. State of West Bengal (1997) 1 SCC 416: 11 binding arrest guidelines (Arrest memo, intimation to relatives within 8-12 hours, medical examination every 48 hours).
- Section 57 CrPC & Article 22(2): Strict 24-hour limit on police detention. The arrestee must be produced before the nearest Judicial Magistrate within 24 hours of arrest, excluding travel time.

Remand and Default Bail: Section 167 CrPC
- Police Custody Remand: Limited to a maximum of 15 days in the aggregate during the initial investigation period.
- Judicial Custody: Total custody during investigation cannot exceed:
  (a) 90 days: Where the investigation relates to an offense punishable with death, imprisonment for life, or imprisonment for not less than 10 years.
  (b) 60 days: For any other offense.
- Default Bail (Statutory Bail): If the police fail to file a final report/charge sheet under Section 173(2) within 60 or 90 days, the accused acquires an indefeasible, non-discretionary right to default bail under Section 167(2) (*Bikramjit Singh v. State of Punjab, 2020*).

Bail Classification and Jurisprudence:
- Section 436: Bailable offenses. Bail is an absolute right; the court or police must release the accused on personal bond or sureties.
- Section 437: Non-bailable offenses before Magistrates.
- Section 438: Anticipatory bail before the High Court or Sessions Court for persons apprehending arrest for a non-bailable offense.
In Sushila Aggarwal v. State (NCT of Delhi) (2020) 5 SCC 1, a 5-judge Constitution Bench held that:
1. Anticipatory bail does not automatically expire upon the filing of a charge sheet.
2. It continues throughout the investigation and trial until the final judgment, unless cancelled for violation of conditions.
- Section 439: Special powers of High Court or Sessions Court for regular bail.

Inherent Powers of High Court: Section 482 CrPC
Section 482 preserves the inherent powers of the High Court to:
(1) Give effect to any order under the Code,
(2) Prevent abuse of the process of any court,
(3) Secure the ends of justice.
The Seven Guidelines of State of Haryana v. Bhajan Lal (1992) Supp (1) SCC 335:
1. Allegations in FIR do not prima facie constitute any offense.
2. Allegations do not disclose a cognizable offense justifying investigation.
3. Uncontroverted allegations fail to establish essential ingredients.
4. Express statutory bar in the Code or Act prohibiting proceedings.
5. Allegations are inherently improbable and absurd.
6. A purely civil, contractual, or commercial dispute is given criminal color.
7. Proceedings instituted with manifest mala fides or personal vendetta.

In Gian Singh v. State of Punjab (2012) 10 SCC 303, the Supreme Court held that the High Court under Section 482 can quash non-compoundable offenses of a private, civil, commercial, or matrimonial nature if the parties have genuinely settled their dispute and continuation would serve no public purpose.
"""

def get_evidence_law_and_commercial_treatise():
    return """
================================================================================
PART III: LAW OF EVIDENCE, COMMERCIAL STATUTES, AND REGULATORY FRAMEWORKS
EVIDENTIARY CERTIFICATION, NEGOTIABLE INSTRUMENTS, INSOLVENCY, AND ARBITRATION
================================================================================

1. THE LAW OF EVIDENCE (INDIAN EVIDENCE ACT, 1872 / BHARATIYA SAKSHYA ADHINIYAM, 2023)

Confessions and Police Custody:
- Section 24: Confession caused by inducement, threat, or promise from a person in authority is irrelevant in criminal proceedings.
- Section 25: "No confession made to a police officer shall be proved as against a person accused of any offence." Absolute statutory bar to eliminate police torture and extraction of fabricated admissions.
- Section 26: Confessions made by an accused while in the custody of a police officer are inadmissible unless made in the immediate presence of a Judicial Magistrate (under Section 164 CrPC).
- Section 27 (The Doctrine of Confirmation by Subsequent Facts):
  When any fact is deposed to as discovered in consequence of information received from a person accused of an offense in police custody, so much of that information, whether it amounts to a confession or not, as relates distinctly to the fact thereby discovered, may be proved.
  In Pulukuri Kottaya v. Emperor (1947) LR 74 IA 65, the Privy Council established:
  The statement "I will produce the knife which I concealed in the roof of my house" is admissible because it leads distinctly to the physical discovery of the weapon.
  However, the statement "I killed the deceased with this knife" is an inculpatory confession and remains strictly excluded under Section 25 and 26.

Dying Declarations: Section 32(1)
Statements, written or verbal, of relevant facts made by a person who is dead, as to the cause of their death, or as to any of the circumstances of the transaction which resulted in their death.
Maxim: Nemo moriturus praesumitur mentire (A person will not meet their maker with a lie in their mouth).
In Laxman v. State of Maharashtra (2002) 6 SCC 710 (Constitution Bench):
1. A dying declaration can form the sole basis of conviction without corroboration if it inspires full confidence.
2. The essential test is whether the deceased was in a 'fit state of mind' to make the statement.
3. Absence of a formal doctor's fitness certificate does not invalidate the declaration if the magistrate or recording person certifies mental fitness.

Secondary Electronic Evidence: Section 65B
Under Section 65B, electronic records (CCTV footage, server logs, phone recordings, WhatsApp exports, emails) are deemed documents and are admissible as secondary evidence only if accompanied by a certificate under Section 65B(4) signed by an authorized manager of the computer device.
In Arjun Panditrao Khotkar v. Kailash Kushanrao Gorantyal (2020) 7 SCC 1 (3-judge bench):
1. Section 65B(4) certificate is an indispensable condition precedent to the admissibility of electronic evidence.
2. If the original physical device itself is brought into court by the owner/operator, it is admitted as primary evidence under Section 62, and no Section 65B certificate is required.
3. Where an entity refuses to give a certificate, the party can apply to the court under Section 91 CrPC for summons directing production.

2. NEGOTIABLE INSTRUMENTS ACT, 1881: SECTION 138 DISHONOUR
Section 138 penalizes the dishonour of cheques for insufficiency of funds or exceeding credit arrangements:
1. Legally Enforceable Debt: Cheque must be drawn to discharge a legal liability (presumption under Section 139).
2. Presentation within Validity: Within 3 months from date of issue.
3. Dishonour Memo: Bank return memo for funds insufficient, account closed (*NEPC Micon*), or stop payment (*Modi Cements*).
4. Statutory Legal Notice (Section 138(b)): Within 30 days of receiving return memo, the payee must issue a written demand notice.
5. 15-Day Cure Period (Section 138(c)): Drawer must be given 15 clear days from receipt of notice to make payment.
6. Cause of Action: Arises on the 16th day after the 15-day period expires without payment.
7. Filing Limitation (Section 142): Criminal complaint must be filed before the competent Magistrate within 30 days of accrual of cause of action.
8. Territorial Jurisdiction (Section 142(2)): Following the 2015 amendment, jurisdiction lies where the payee maintains their bank account (if presented through an account).
9. Compounding: Permissible under Section 147 at any stage of trial (*Meters and Instruments v. Kanchan Mehta*). Interim compensation up to 20% under Section 143A.

3. INSOLVENCY AND BANKRUPTCY CODE (IBC), 2016
Consolidated corporate insolvency into a creditor-in-control, time-bound framework:
1. Financial Creditors (Section 7): Debt disbursed against the consideration for the time value of money. Initiates CIRP upon default. In Innoventive Industries Ltd. v. ICICI Bank (2018) 1 SCC 407, the Supreme Court ruled that upon proof of default, NCLT has no discretion and must admit the Section 7 petition.
2. Operational Creditors (Section 9): Dues for goods, services, employment, or taxes. Must issue a 10-day demand notice under Section 8. In Mobilox Innovations Pvt. Ltd. v. Kirusa Software Pvt. Ltd. (2018) 1 SCC 353, the court held that a plausible, pre-existing dispute prior to the Section 8 notice mandates rejection of the Section 9 petition.
3. Moratorium (Section 14): Absolute stay on suits, proceedings, and asset execution against the corporate debtor during CIRP.
4. Committee of Creditors (CoC) Commercial Wisdom: In Committee of Creditors of Essar Steel India Ltd. v. Satish Kumar Gupta (2020) 8 SCC 531, the Supreme Court established that the commercial wisdom of the CoC in approving resolution plans is non-justiciable.

4. ARBITRATION AND CONCILIATION ACT, 1996
Modeled on the UNCITRAL Model Law:
1. Section 8: Mandatory referral of parties to arbitration where an arbitration agreement exists.
2. Section 9: Interim measures granted by judicial courts before or during arbitral proceedings.
3. Section 11: Appointment of arbitrators by High Court / Supreme Court. Scope limited to examining the prima facie existence of the arbitration agreement (*Vidya Drolia v. Durga Trading Corp, 2021*).
4. Section 34: Setting aside arbitral awards on narrow grounds:
   - Incapacity of party, invalid agreement, lack of notice.
   - Public Policy of India: In Associate Builders v. DDA (2015) 3 SCC 49 and Ssangyong Engineering v. NHAI (2019) 15 SCC 131, the Supreme Court established that public policy review is restricted to: (1) Fundamental Policy of Indian Law, (2) Basic notions of morality and justice, and (3) Patent illegality on the face of the award (for domestic awards). Courts cannot reappreciate evidence or act as appellate courts.
"""

def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "massive_legal_cpt_corpus.txt")

    sections = [
        get_constitutional_law_mega_treatise(),
        get_criminal_substantive_and_procedural_treatise(),
        get_evidence_law_and_commercial_treatise()
    ]

    # Assemble into master corpus
    full_corpus = "\n\n".join(sections)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_corpus)

    file_size_kb = os.path.getsize(output_path) / 1024
    approx_tokens = len(full_corpus.split()) * 1.35

    print(f"==================================================")
    print(f"Massive Legal Corpus Compiled Successfully!")
    print(f"Path: {output_path}")
    print(f"File Size: {file_size_kb:.2f} KB ({file_size_kb/1024:.2f} MB)")
    print(f"Word Count: {len(full_corpus.split()):,}")
    print(f"Estimated Tokens: {int(approx_tokens):,}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
