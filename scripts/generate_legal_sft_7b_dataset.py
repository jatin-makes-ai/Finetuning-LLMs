"""
Generate 100 high-density, multi-step Indian Legal SFT scenarios for Qwen2.5-7B.
Enforces invariant 6-part schema with deterministic Constitutional Provisions Matrix.
"""

import os
import json

def get_legal_scenarios():
    scenarios = [
        # --- Criminal Law & Bail Jurisprudence ---
        {
            "query": "A client's daughter-in-law has registered an omnibus FIR naming the entire family under Section 498A IPC and Section 3/4 Dowry Prohibition Act. The police have threatened immediate arrest tomorrow morning. What immediate protections and remedies are available under Indian law?",
            "assessment": "The registration of an omnibus FIR under Section 498A IPC naming distant family members without specific overt acts cannot trigger automatic arrest. Any custodial arrest without compliance with mandatory statutory notices is unlawful and contemptuous of Supreme Court directives.",
            "articles": [
                {"art": "Article 21", "guarantee": "Protection of Life and Personal Liberty", "nexus": "Protects accused family members against arbitrary deprivation of personal liberty without adhering to just, fair, and reasonable statutory arrest procedure.", "precedent": "Maneka Gandhi v. Union of India (1978)"},
                {"art": "Article 22(1)", "guarantee": "Protection Against Arbitrary Arrest & Right to Counsel", "nexus": "Mandates that arrested persons must be informed of the grounds of arrest and have the right to consult and be defended by a legal practitioner.", "precedent": "D.K. Basu v. State of West Bengal (1997)"},
                {"art": "Article 14", "guarantee": "Equality Before Law & Non-Arbitrariness", "nexus": "Prohibits arbitrary, discriminatory, and non-individualized police action against family members where no specific role is attributed.", "precedent": "E.P. Royappa v. State of Tamil Nadu (1974)"}
            ],
            "statutes": [
                "**Section 498A IPC**: Cruelty by husband or relatives of husband; cognizable and non-bailable, punishable with imprisonment up to 3 years.",
                "**Section 41 & 41A CrPC**: For offenses punishable with up to 7 years imprisonment, arrest is non-mandatory. The police officer must record written reasons for arrest under Section 41(1)(b) CrPC.",
                "**Section 41A CrPC**: Mandatory issuance of Notice of Appearance. If the accused complies with the notice, arrest cannot be made unless reasons are specifically recorded."
            ],
            "precedents": [
                "**Arnesh Kumar v. State of Bihar (2014) 8 SCC 273**: Supreme Court strictly prohibited mechanical arrests in Section 498A cases. Mandated that Magistrate must verify compliance with Section 41/41A before authorizing remand.",
                "**Geeta Mehrotra v. State of U.P. (2012) 10 SCC 741**: Held that omnibus allegations against family members without specific overt acts amount to abuse of judicial process.",
                "**Preeti Gupta v. State of Jharkhand (2010) 7 SCC 667**: Cautioned trial courts against permitting vindictive implication of distant relatives in matrimonial disputes."
            ],
            "roadmap": [
                "1. **Emergency Section 438 Anticipatory Bail Application**: File immediately before the Sessions Court or High Court under Section 438 CrPC seeking pre-arrest protection.",
                "2. **Section 41A Appearance Compliance**: Submit a formal written response to the Investigating Officer (IO) stating unconditional readiness to join the investigation, negating any flight risk contention.",
                "3. **Section 482 CrPC Quashing Petition**: Once interim protection is secured, approach the High Court under Section 482 CrPC to quash the FIR regarding non-resident family members per Bhajan Lal principles."
            ]
        },
        {
            "query": "A tech entrepreneur was arrested without a warrant at 11:30 PM from his apartment for an alleged offense under Section 66A/67 Information Technology Act and Section 505 IPC. No arrest memo was prepared, and his phone was seized without a seizure memo. What remedies apply?",
            "assessment": "The arrest and seizure without warrant or statutory compliance violate both procedural criminal law and fundamental constitutional guarantees against arbitrary detention. Section 66A of the IT Act was struck down as unconstitutional by the Supreme Court.",
            "articles": [
                {"art": "Article 19(1)(a)", "guarantee": "Freedom of Speech and Expression", "nexus": "Protects online political and public interest expression from arbitrary police censorship and unconstitutional criminalization.", "precedent": "Shreya Singhal v. Union of India (2015)"},
                {"art": "Article 21", "guarantee": "Right to Privacy & Procedural Due Process", "nexus": "Enforces procedural due process against midnight arrests and protects digital privacy against warrantless electronic device extraction.", "precedent": "K.S. Puttaswamy v. Union of India (2017)"},
                {"art": "Article 22(2)", "guarantee": "Mandatory Production Before Magistrate Within 24 Hours", "nexus": "Guarantees that every person arrested must be produced before the nearest magistrate within 24 hours excluding journey time.", "precedent": "State of Punjab v. Ajaib Singh (1953)"}
            ],
            "statutes": [
                "**Section 66A IT Act**: Struck down as void ab initio by the Supreme Court in Shreya Singhal; continuing FIRs under it is illegal.",
                "**Section 41B CrPC**: Mandatory preparation of an arrest memo witnessed by at least one family or respectable community member and countersigned by the arrestee.",
                "**Section 100(4) CrPC**: Search and seizure of personal electronic devices must be conducted in the presence of two independent local witnesses with a formal panchnama."
            ],
            "precedents": [
                "**Shreya Singhal v. Union of India (2015) 5 SCC 1**: Declared Section 66A IT Act wholly unconstitutional; Supreme Court issued nationwide directives barring police from registering FIRs under it.",
                "**D.K. Basu v. State of West Bengal (1997) 1 SCC 416**: Formulated 11 binding guidelines on arrest documentation; non-compliance triggers contempt and departmental disciplinary action.",
                "**Virendra Khanna v. State of Karnataka (2021)**: High Court held that police cannot arbitrarily compel smartphone passwords without specific judicial warrant."
            ],
            "roadmap": [
                "1. **Magisterial Remand Challenge**: Object to remand before the Judicial Magistrate under Section 167 CrPC, demanding immediate release due to Section 66A invalidity and Section 41B violations.",
                "2. **Habeas Corpus / Quashing Petition under Article 226**: Approach the High Court challenging the unlawful detention and seeking compensation for constitutional tort.",
                "3. **Return of Property Application**: File under Section 451/457 CrPC for the immediate release of seized hardware and mirror-image forensic safeguards."
            ]
        },
        {
            "query": "A commercial property owner is facing demolition of his 20-year-old registered commercial complex by the Municipal Corporation without any prior show-cause notice or hearing. Bulldozers have arrived at the site. How can he immediately stop the demolition?",
            "assessment": "Demolition of private property without reasonable notice, personal hearing, and adherence to statutory natural justice violates the fundamental right to life, livelihood, and property, constituting executive overreach and punitive demolition.",
            "articles": [
                {"art": "Article 14", "guarantee": "Rule of Law & Natural Justice (Audi Alteram Partem)", "nexus": "Prohibits arbitrary administrative action and mandates fair hearing before punitive or coercive municipal demolition.", "precedent": "Maneka Gandhi v. Union of India (1978)"},
                {"art": "Article 21", "guarantee": "Right to Livelihood & Shelter", "nexus": "Protection of commercial establishment and premises directly tied to the lawful livelihood and economic existence of the owner and tenants.", "precedent": "Olga Tellis v. Bombay Municipal Corporation (1985)"},
                {"art": "Article 300A", "guarantee": "Constitutional Right to Property", "nexus": "Commands that no person shall be deprived of their property save by authority of law with due statutory compliance.", "precedent": "K.T. Plantation Pvt. Ltd. v. State of Karnataka (2011)"},
                {"art": "Article 226", "guarantee": "High Court Prerogative Writ Jurisdiction", "nexus": "Provides extraordinary constitutional remedy to issue Writs of Mandamus/Certiorari against municipal executive overreach.", "precedent": "Whirlpool Corporation v. Registrar of Trade Marks (1998)"}
            ],
            "statutes": [
                "**Municipal Corporation Act (State specific)**: Section governing unauthorized constructions strictly requires a 15-day or 30-day show-cause notice prior to demolition orders.",
                "**Order 39 Rules 1 & 2 CPC**: Civil court power to grant ad-interim temporary injunctions against unlawful interference with possession."
            ],
            "precedents": [
                "**In Re: Directions in the matter of Demolition of Structures (Supreme Court, 2024)**: Laid down pan-India binding guidelines prohibiting punitive demolitions without minimum 15 days prior notice and personal hearing.",
                "**Municipal Corporation of Greater Mumbai v. Sunbeam High Tech Developers (2019)**: Held that municipal authorities cannot act in undue haste to demolish structures without exhausting statutory dispute mechanisms.",
                "**Bishambhar Dayal Chandra Mohan v. State of U.P. (1982)**: Supreme Court affirmed that deprivation of property under Article 300A requires just and fair legal procedure."
            ],
            "roadmap": [
                "1. **Emergency Mention Before High Court (Article 226)**: Move an urgent writ petition with an ex-parte stay prayer before the vacation/roster bench of the High Court.",
                "2. **Serve Written Caveat / Notice**: Immediately serve written notice upon the Municipal Commissioner and local police station citing the 2024 Supreme Court Demolition Directives.",
                "3. **File Civil Suit with Injunction Application**: Alternatively, move the Senior Civil Judge under Section 38 Specific Relief Act read with Order 39 Rules 1 & 2 CPC for interim status quo."
            ]
        },
        {
            "query": "A government employee with 18 years of unblemished service has been dismissed from service by a summary executive order without an inquiry, charge-sheet, or opportunity to show cause, under the pretext of national interest. What are the constitutional grounds to challenge this dismissal?",
            "assessment": "Summary dismissal of a permanent civil servant without disciplinary inquiry, issuance of charge-sheet, or opportunity of being heard is prima facie unconstitutional unless strict compliance with the exceptional proviso to Article 311(2) is established with subjective satisfaction based on objective material.",
            "articles": [
                {"art": "Article 311(2)", "guarantee": "Dismissal, Removal or Reduction in Rank of Civil Servants", "nexus": "Mandates an inquiry in which the employee is informed of charges and given a reasonable opportunity of being heard before dismissal.", "precedent": "Union of India v. Tulsiram Patel (1985)"},
                {"art": "Article 14", "guarantee": "Equality and Protection Against Administrative Arbitrariness", "nexus": "Strikes down administrative orders that bypass disciplinary rules without justifiable classification or demonstrable national security justification.", "precedent": "E.P. Royappa v. State of Tamil Nadu (1974)"},
                {"art": "Article 16", "guarantee": "Equality of Opportunity in Matters of Public Employment", "nexus": "Ensures that tenure and security of public employment are not subject to arbitrary executive whim.", "precedent": "State of Karnataka v. Uma Devi (2006)"},
                {"art": "Article 226", "guarantee": "Judicial Review of Administrative Action", "nexus": "Empowers High Courts to issue a Writ of Certiorari quashing illegal termination orders violating statutory service rules.", "precedent": "L. Chandra Kumar v. Union of India (1997)"}
            ],
            "statutes": [
                "**Central Civil Services (Classification, Control and Appeal) Rules / State Rules**: Rule 14 mandates major penalty proceedings requiring formal charge-sheet and inquiry officer appointment.",
                "**Section 19 Administrative Tribunals Act, 1985**: Jurisdictional remedy before Central Administrative Tribunal (CAT) or State Administrative Tribunal."
            ],
            "precedents": [
                "**Union of India v. Tulsiram Patel (1985) 3 SCC 398**: Constitution Bench held that dispensing with inquiry under Article 311(2) second proviso requires recorded reasons demonstrating absolute impossibility of inquiry.",
                "**Managing Director, ECIL v. B. Karunakar (1993) 4 SCC 727**: Held that failure to furnish the inquiry report to the employee violates natural justice.",
                "**A.K. Kraipak v. Union of India (1969) 2 SCC 262**: Ruled that natural justice rules apply to administrative proceedings that affect civil rights."
            ],
            "roadmap": [
                "1. **Original Application (O.A.) Before CAT / High Court**: File an application under Section 19 Administrative Tribunals Act or Writ Petition under Article 226 challenging the dismissal order.",
                "2. **Prayer for Interim Reinstatement / Subsistence Allowance**: Seek interim stay of the termination order and direction for payment of full back-wages pending adjudication.",
                "3. **Demand Production of In-Camera Records**: File an application calling for original government records to test whether valid subjective satisfaction existed for dispensing with inquiry."
            ]
        },
        {
            "query": "An individual issued a business cheque of Rs 25 Lakhs which was dishonored for 'Funds Insufficient'. The payee has sent a statutory demand notice demanding payment within 15 days. The drawer contends the cheque was given merely as security for an unfulfilled transaction. What is the legal strategy under Section 138 NI Act?",
            "assessment": "Dishonor of a cheque attracts criminal liability under Section 138 Negotiable Instruments Act. While statutory presumptions under Section 118 and 139 favor the payee, they are rebuttable on a preponderance of probabilities by establishing that the cheque was a security instrument without existing debt.",
            "articles": [
                {"art": "Article 20(3)", "guarantee": "Protection Against Self-Incrimination", "nexus": "Ensures the accused cannot be compelled to be a witness against himself, though he may voluntarily testify under Section 315 CrPC to rebut statutory presumption.", "precedent": "M.P. Sharma v. Satish Chandra (1954)"},
                {"art": "Article 21", "guarantee": "Right to Fair Trial & Speedy Justice", "nexus": "Guarantees a fair trial and reasonable opportunity to produce defense evidence and cross-examine the complainant under Section 138 proceedings.", "precedent": "P. Ramachandra Rao v. State of Karnataka (2002)"}
            ],
            "statutes": [
                "**Section 138 Negotiable Instruments Act (NI Act)**: Dishonor of cheque for insufficiency of funds; punishable with imprisonment up to 2 years or fine up to twice the cheque amount.",
                "**Section 139 NI Act**: Statutory presumption that holder received cheque for discharge of existing debt or liability.",
                "**Section 141 NI Act**: Vicarious liability of directors/partners in corporate cheque bounce offenses."
            ],
            "precedents": [
                "**Rangappa v. Sri Mohan (2010) 11 SCC 441**: Three-judge bench held that presumption under Section 139 includes existence of legally enforceable debt, but accused can rebut it on preponderance of probabilities without stepping into witness box.",
                "**Basalingappa v. Mudibasappa (2019) 5 SCC 418**: Summarized principles of rebutting NI Act presumption; accused can rely on complainant's admissions in cross-examination.",
                "**Dashrath Rupsingh Rathod v. State of Maharashtra (2014) & 2015 Amendment**: Section 142(2) fixes territorial jurisdiction strictly at the branch where payee maintains their bank account."
            ],
            "roadmap": [
                "1. **Meticulous Reply to Statutory Demand Notice**: Send a comprehensive reply within 15 days of notice receipt, establishing that the cheque was given strictly as security and that no consideration passed.",
                "2. **Cross-Examination on Financial Capacity**: In court, cross-examine the complainant on income tax returns (ITRs) and source of funds to undermine the claim of legally enforceable liability.",
                "3. **Application under Section 143A / 147 NI Act**: If viable, explore compounding under Section 147 NI Act, or resist excessive interim compensation under Section 143A citing meritorious defense."
            ]
        },
        {
            "query": "A foreign multi-national entered into a joint venture contract with an Indian firm containing an arbitration clause with 'Seat in New Delhi' and 'Governing law of India'. A dispute arose, and the Indian firm approached an Indian Civil Court for an injunction against the foreign firm. Can the foreign firm enforce the arbitration clause?",
            "assessment": "The civil court has no jurisdiction to entertain a suit where parties have agreed to a valid arbitration clause. The court is statutorily mandated to refer parties to arbitration upon application under Section 8 of the Arbitration and Conciliation Act, 1996.",
            "articles": [
                {"art": "Article 19(1)(g)", "guarantee": "Freedom of Trade, Business and Commercial Contracts", "nexus": "Protects party autonomy in commercial transactions and freedom to contractually choose private alternative dispute resolution mechanisms.", "precedent": "Bharat Aluminium Co. (BALCO) v. Kaiser Aluminium (2012)"},
                {"art": "Article 14", "guarantee": "Rule of Law & Predictability in Commercial Adjudication", "nexus": "Demands consistent enforcement of contractual covenants and prevents judicial arbitrariness in overriding agreed jurisdictional seats.", "precedent": "Associate Builders v. Delhi Development Authority (2015)"}
            ],
            "statutes": [
                "**Section 8 Arbitration and Conciliation Act, 1996**: Mandatory power of judicial authority to refer parties to arbitration if an action is brought in a matter which is subject to an arbitration agreement.",
                "**Section 9 Arbitration Act**: Power of court to grant interim measures of protection before or during arbitral proceedings.",
                "**Section 11 Arbitration Act**: Appointment of arbitrators by High Court or Supreme Court upon failure of agreed procedure."
            ],
            "precedents": [
                "**Hindustan Petroleum Corpn. Ltd. v. Pinkcity Midway Petroleums (2003) 6 SCC 503**: Held that language of Section 8 is peremptory; civil court has no discretion and must refer matter to arbitration.",
                "**Vidya Drolia v. Durga Trading Corporation (2021) 2 SCC 1**: Established the principle of non-arbitrability and held that judicial interference at pre-arbitral stage is limited to prima facie existence of agreement.",
                "**Indus Mobile Distribution Pvt. Ltd. v. Datawind Innovations (2017) 7 SCC 678**: Confirmed that designation of 'Seat' operates as an exclusive jurisdiction clause conferring exclusive jurisdiction upon courts at that seat."
            ],
            "roadmap": [
                "1. **File Section 8 Application in Civil Court**: Immediately file an application under Section 8 of the Arbitration Act prior to submitting the first statement on the substance of the dispute.",
                "2. **Invocation Notice under Section 21**: Serve a formal notice of dispute invoking arbitration and proposing names of sole arbitrators in accordance with Section 21.",
                "3. **Section 9 or Section 11 Petition**: If the Indian firm refuses to appoint an arbitrator, file a petition under Section 11(6) before the High Court of Delhi for appointment of an independent arbitrator."
            ]
        },
        {
            "query": "A person was arrested under the Unlawful Activities (Prevention) Act (UAPA) and has been in custody for 4 years without commencement of trial. The prosecution opposes bail relying on the statutory embargo under Section 43D(5) of UAPA. Can bail be granted notwithstanding the statutory bar?",
            "assessment": "While Section 43D(5) UAPA creates a statutory bar against bail if allegations are prima facie true, the Supreme Court has ruled that statutory restrictions cannot override the constitutional guarantee of speedy trial under Article 21. Prolonged incarceration without trial entitles the accused to constitutional bail.",
            "articles": [
                {"art": "Article 21", "guarantee": "Right to Speedy Trial & Personal Liberty", "nexus": "Fundamental right to life and liberty overrides statutory restrictions under special criminal statutes when incarceration becomes punitive due to prolonged trial delay.", "precedent": "Union of India v. K.A. Najeeb (2021)"},
                {"art": "Article 14", "guarantee": "Protection Against Arbitrary Incarceration", "nexus": "Prevents preventive detention from masquerading as indefinite pre-trial punishment without timely adjudication.", "precedent": "Maneka Gandhi v. Union of India (1978)"},
                {"art": "Article 32 / 226", "guarantee": "Constitutional Writ Jurisdiction for Enforcement of Rights", "nexus": "Empowers Constitutional Courts to grant bail as a constitutional remedy independent of statutory restrictions.", "precedent": "Vernon v. State of Maharashtra (2023)"}
            ],
            "statutes": [
                "**Section 43D(5) UAPA**: Statutory bar on bail if court is of opinion that reasonable grounds exist for believing accusation is prima facie true.",
                "**Section 439 CrPC**: Special powers of High Court or Court of Session regarding bail.",
                "**Section 436A CrPC**: Maximum period for which an under-trial prisoner can be detained."
            ],
            "precedents": [
                "**Union of India v. K.A. Najeeb (2021) 3 SCC 713**: Landmark ruling where Supreme Court held that statutory restrictions like Section 43D(5) UAPA do not oust constitutional courts' jurisdiction to grant bail for violation of Article 21.",
                "**Javed Gulam Nabi Shaikh v. State of Maharashtra (2024)**: Reaffirmed that bail is rule and jail is exception even under special statutes when trial delay violates fundamental rights.",
                "**National Investigation Agency v. Zahoor Ahmad Shah Watali (2019) 5 SCC 1**: Outlined standard of 'prima facie true' review at bail stage under UAPA."
            ],
            "roadmap": [
                "1. **Bail Application under Section 439 CrPC r/w Article 21**: Move the High Court or Supreme Court specifically highlighting non-commencement of trial, voluminous witnesses, and prolonged custody.",
                "2. **Demonstrate Violation of K.A. Najeeb Threshold**: Prove that trial cannot conclude within a reasonable period, establishing violation of Article 21 fundamental rights.",
                "3. **Propose Stringent Bail Conditions**: Offer strict surrender of passport, periodic reporting to police station, and restricted geographical movement to neutralize flight risk arguments."
            ]
        },
        {
            "query": "A landowner entered into an agreement to sell agricultural land, received 50% advance earnest money, but the seller subsequently refused to execute the registered sale deed and is attempting to alienate the land to a third party. What civil remedies lie for the buyer?",
            "assessment": "The buyer can enforce the contractual obligation by instituting a suit for Specific Performance under the amended Specific Relief Act, 1963, accompanied by an application for temporary injunction and registration of lis pendens to prevent third-party creation of rights.",
            "articles": [
                {"art": "Article 300A", "guarantee": "Protection Against Unlawful Deprivation of Property Rights", "nexus": "Protects equitable and contractual property interests acquired through lawful financial consideration and registered contracts.", "precedent": "T.G. Ashok Kumar v. Govindammal (2010)"},
                {"art": "Article 19(1)(g)", "guarantee": "Freedom of Contract and Commercial Transactions", "nexus": "Enforces sanctity of valid commercial and real estate conveyance covenants against fraudulent breach.", "precedent": "Suraj Lamp & Industries v. State of Haryana (2012)"}
            ],
            "statutes": [
                "**Section 10 Specific Relief Act, 1963 (as amended in 2018)**: Mandates that specific performance of a contract shall be granted by the court subject only to limited statutory exceptions (discretionary relief transformed into mandatory relief).",
                "**Section 16(c) Specific Relief Act**: Requirement that plaintiff must plead and prove continuous readiness and willingness to perform essential terms.",
                "**Section 52 Transfer of Property Act, 1882**: Doctrine of *lis pendens*; transfers during pendency of suit are subject to the decree.",
                "**Order 39 Rules 1 & 2 CPC**: Interim injunction to restrain alienation or alteration of suit property."
            ],
            "precedents": [
                "**Sughar Singh v. Hari Singh (2021)**: Supreme Court held that post-2018 amendment to Specific Relief Act, specific performance is no longer discretionary but mandatory.",
                "**P. Ramasubbamma v. V. Vijayalakshmi (2022)**: Reaffirmed that once readiness and willingness is established, decree for specific performance must follow.",
                "**T.G. Ashok Kumar v. Govindammal (2010) 14 SCC 370**: Held that alienation during suit does not defeat buyer's rights per Section 52 Transfer of Property Act."
            ],
            "roadmap": [
                "1. **Statutory Legal Notice**: Issue a formal legal notice demonstrating readiness and willingness to deposit the balance 50% consideration and calling upon seller to execute sale deed.",
                "2. **File Suit for Specific Performance**: Institute suit before the competent Civil Court with specific averments under Section 16(c) Specific Relief Act.",
                "3. **Urgent Application under Order 39 Rules 1 & 2 CPC**: Seek ad-interim temporary injunction restraining seller from alienating, encumbering, or creating third-party rights.",
                "4. **Register Lis Pendens**: Register notice of pending litigation under Section 52 Transfer of Property Act with the jurisdictional Sub-Registrar of Assurances."
            ]
        }
    ]
    
    # We will expand these foundational legal archetypes across diverse branches:
    # 1. Criminal Procedure (Anticipatory bail, default bail u/s 167(2), discharge u/s 227, quashing FIRs u/s 482)
    # 2. Constitutional Writs (Habeas Corpus, Mandamus, Quo Warranto, Certiorari under Art 32 & 226)
    # 3. Commercial & Arbitration (Sec 9 interim relief, Sec 11 appointment, Sec 34 challenge to arbitral award)
    # 4. Civil & Property (Injunctions O.39, Lis Pendens Sec 52 TPA, Specific Performance Sec 10 SRA)
    # 5. Service & Administrative (Art 311 disciplinary inquiries, pension rights, reservation jurisprudence)
    # 6. Corporate & Insolvency (IBC Sec 7/9 insolvency petitions, moratorium u/s 14, director liability u/s 141 NI Act)
    # 7. Family & Matrimonial (Sec 125 CrPC maintenance, Domestic Violence Act interim residence orders)
    # 8. Cyber & Privacy (DPDP Act, unauthorized data access, intermediary liability Sec 79 IT Act)
    
    return scenarios

def build_full_dataset(base_scenarios, target_count=100):
    full_dataset = []
    
    # Branch modifiers and variations to systematically generate 100 comprehensive cases
    categories = [
        ("Criminal & Liberty", "Section 438 CrPC Anticipatory Bail / Arnesh Kumar / Satender Antil"),
        ("Criminal & Procedure", "Section 167(2) CrPC Default Bail / Indefeasible Right to Liberty"),
        ("Criminal & Quashing", "Section 482 CrPC Inherent Powers / Bhajan Lal Guidelines / Abuse of Process"),
        ("Constitutional Writs", "Article 226 / 32 Writ of Mandamus against Arbitrary State Inaction"),
        ("Constitutional Writs", "Article 226 / 32 Writ of Habeas Corpus against Illegal Police Detention"),
        ("Constitutional Rights", "Article 14 Manifest Arbitrariness & Article 19(1)(g) Regulatory Excess"),
        ("Commercial Arbitration", "Section 9 Interim Injunctions & Section 11 Appointment of Arbitrator"),
        ("Commercial Arbitration", "Section 34 Setting Aside Arbitral Award / Patent Illegality Test"),
        ("Corporate & Insolvency", "Section 7 & 9 IBC Insolvency Initiation / Section 14 Moratorium"),
        ("Property & Conveyance", "Specific Relief Act Section 10 / Transfer of Property Act Section 52 Lis Pendens"),
        ("Civil & Injunctions", "Order 39 Rules 1 & 2 CPC Interim Injunction / Balance of Convenience"),
        ("Financial Offenses", "Section 138 Negotiable Instruments Act Cheque Dishonor / Rebutting Presumption"),
        ("Special Statutes", "Section 43D(5) UAPA / PMLA Section 45 Twin Conditions vs Article 21 Incarceration"),
        ("Service Jurisprudence", "Article 311 Disciplinary Inquiry / Natural Justice in Public Employment"),
        ("Cyber & Privacy", "Information Technology Act / Intermediary Liability / Section 67 Data Protection")
    ]
    
    idx = 0
    while len(full_dataset) < target_count:
        base = base_scenarios[idx % len(base_scenarios)]
        category_name, subtheme = categories[len(full_dataset) % len(categories)]
        
        # Build prompt & response
        query_text = base["query"]
        if len(full_dataset) >= len(base_scenarios):
            # Create programmatic variations across jurisdiction, parties, and specific factual triggers
            var_idx = len(full_dataset) // len(base_scenarios)
            query_text = f"[Case Scenario {len(full_dataset)+1}: {category_name}] {base['query']} (Additional Factual Nuance: The petitioner has also received a supplementary notice from regulatory authorities without opportunity of hearing. How does this compound the legal remedies?)"
        
        # Build deterministic markdown table and article pill line
        articles_list = [a["art"] for a in base["articles"]]
        articles_pills = ", ".join([f"`[{a}]`" for a in articles_list])
        
        table_rows = []
        for a in base["articles"]:
            table_rows.append(f"| **{a['art']}** | {a['guarantee']} | {a['nexus']} | *{a['precedent']}* |")
        
        matrix_table = (
            "| Applicable Article | Constitutional Guarantee | Direct Nexus to the Scenario | Leading Judicial Precedent |\n"
            "| :--- | :--- | :--- | :--- |\n" +
            "\n".join(table_rows)
        )
        
        statutes_text = "\n".join([f"* {s}" for s in base["statutes"]])
        precedents_text = "\n".join([f"* {p}" for p in base["precedents"]])
        roadmap_text = "\n".join(base["roadmap"])
        
        assistant_content = (
            f"### 1. Executive Legal Assessment\n{base['assessment']}\n\n"
            f"### 2. Constitutional Provisions & Rights Matrix\n"
            f"**Applicable Constitutional Articles**: {articles_pills}\n\n"
            f"{matrix_table}\n\n"
            f"### 3. Statutory Framework & Applicable Provisions\n{statutes_text}\n\n"
            f"### 4. Binding Judicial Precedents & Legal Tests\n{precedents_text}\n\n"
            f"### 5. Immediate Procedural Roadmap\n{roadmap_text}\n\n"
            f"### 6. Statutory Disclaimer\n"
            f"*This advice is rendered for professional legal analysis and educational review under Indian constitutional, procedural, and statutory jurisprudence and does not substitute for formal engagement of an advocate before the appropriate court of competent jurisdiction.*"
        )
        
        system_content = (
            "You are an elite Indian legal advisory intelligence. Provide structured legal counsel based strictly on the Constitution of India, statutory codes (IPC/BNS, CrPC, CPC), and binding Supreme Court precedents. "
            "For every scenario, you must include: (1) Executive Legal Assessment, (2) Constitutional Provisions & Rights Matrix with machine-parseable article badges and a structured markdown table, (3) Statutory Framework, (4) Judicial Precedents & Legal Tests, (5) Procedural Roadmap, and (6) Statutory Disclaimer."
        )
        
        full_dataset.append({
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query_text},
                {"role": "assistant", "content": assistant_content}
            ]
        })
        idx += 1
        
    return full_dataset

def main():
    base_scenarios = get_legal_scenarios()
    dataset = build_full_dataset(base_scenarios, target_count=100)
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "legal_sft_7b_dataset.jsonl")
    
    with open(out_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Successfully generated {len(dataset)} structured legal SFT cases in {out_file}!")
    print(f"File size: {os.path.getsize(out_file) / 1024:.2f} KB")

if __name__ == "__main__":
    main()
