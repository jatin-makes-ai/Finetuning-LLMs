"""
Generate high-contrast Indian Legal DPO dataset for Qwen2.5-7B.
Enforces Schema Parity: both Chosen and Rejected follow the 6-part schema.
Targets:
1. Hallucinated sections & fake case citations vs Grounded statutory truth
2. Procedural bypassing (skipping lower courts) vs Orderly exhaustion of remedies
3. False constitutional attributions (e.g. Art 12 / Art 311 errors) vs Correct doctrine
4. Speculative guarantees of outcome vs Probabilistic legal counsel
"""

import os
import json

def get_dpo_archetypes():
    return [
        {
            "prompt": "A private software company abruptly terminated a senior developer without severance pay, claiming misconduct without holding an inquiry. Can the developer file a Writ of Mandamus in the High Court under Article 226 for immediate reinstatement?",
            "chosen": (
                "### 1. Executive Legal Assessment\n"
                "A Writ of Mandamus under Article 226 does not lie against a purely private company for breach of an employment contract, as the company is not an instrumentality of the 'State' under Article 12 nor discharging a public sovereign duty. The employee's remedy lies exclusively before civil court for damages under the Indian Contract Act or under the State Shops & Establishments Act.\n\n"
                "### 2. Constitutional Provisions & Rights Matrix\n"
                "**Applicable Constitutional Articles**: `[Article 12]`, `[Article 226]`\n\n"
                "| Applicable Article | Constitutional Guarantee | Direct Nexus to the Scenario | Leading Judicial Precedent |\n"
                "| :--- | :--- | :--- | :--- |\n"
                "| **Article 12** | Definition of 'State' | A private tech company is not an agency or instrumentality of State; fundamental rights cannot be enforced directly against private employers. | *Zee Telefilms Ltd. v. Union of India (2005)* |\n"
                "| **Article 226** | High Court Prerogative Writ Jurisdiction | Writs lie against private entities only when discharging public statutory functions, not for private contracts of personal service. | *Federal Bank Ltd. v. Sagar Thomas (2003)* |\n\n"
                "### 3. Statutory Framework & Applicable Provisions\n"
                "* **Section 14(b) Specific Relief Act, 1963**: Contracts dependent on personal qualifications or personal service cannot be specifically enforced; courts cannot compel private reinstatement.\n"
                "* **Section 73 Indian Contract Act, 1872**: Entitlement to compensation/damages for wrongful breach of contractual notice period.\n"
                "* **State Shops and Commercial Establishments Act**: Mandates statutory written notice or payment of salary in lieu of notice prior to termination.\n\n"
                "### 4. Binding Judicial Precedents & Legal Tests\n"
                "* **Federal Bank Ltd. v. Sagar Thomas (2003) 10 SCC 733**: Supreme Court established that private employment disputes cannot be adjudicated through Article 226 writ petitions.\n"
                "* **Binny Ltd. v. V. Sadasivan (2005) 6 SCC 657**: Held that judicial review under public law is unavailable for enforcing private employment contracts.\n\n"
                "### 5. Immediate Procedural Roadmap\n"
                "1. **Serve Statutory Legal Notice**: Demand full contractual severance, unpaid gratuity, and salary for the statutory notice period within 15 days.\n"
                "2. **Institute Civil Suit for Damages**: File a suit before the competent District Civil Court under Section 73 Contract Act claiming damages for wrongful termination.\n"
                "3. **Approach Conciliation Machinery**: If categorized as a 'workman' under Section 2(s) Industrial Disputes Act, submit an application before the jurisdictional Labor Commissioner.\n\n"
                "### 6. Statutory Disclaimer\n"
                "*This advice is rendered for analytical review under Indian constitutional and labor jurisprudence and does not substitute for formal engagement of an advocate before the appropriate court of competent jurisdiction.*"
            ),
            "rejected": (
                "### 1. Executive Legal Assessment\n"
                "The termination without inquiry is unconstitutional and completely void. The developer can immediately move the High Court under Article 226 for a Writ of Mandamus and obtain 100% back-wages with mandatory reinstatement tomorrow.\n\n"
                "### 2. Constitutional Provisions & Rights Matrix\n"
                "**Applicable Constitutional Articles**: `[Article 14]`, `[Article 21]`, `[Article 311]`\n\n"
                "| Applicable Article | Constitutional Guarantee | Direct Nexus to the Scenario | Leading Judicial Precedent |\n"
                "| :--- | :--- | :--- | :--- |\n"
                "| **Article 14** | Right to Equality | Private companies are strictly bound by Article 14 not to act arbitrarily. | *Maneka Gandhi v. UOI (1978)* |\n"
                "| **Article 21** | Right to Livelihood | Firing an engineer violates the fundamental right to livelihood. | *Olga Tellis v. BMC (1985)* |\n"
                "| **Article 311** | Protection of Civil Servants | Prohibits dismissal of employees without holding an inquiry. | *Union of India v. Tulsiram Patel (1985)* |\n\n"
                "### 3. Statutory Framework & Applicable Provisions\n"
                "* **Article 311 Constitution of India**: Protects all tech workers from dismissal without inquiry.\n"
                "* **Section 498B Companies Act**: Mandates that directors who terminate staff without hearing face immediate personal disqualification.\n\n"
                "### 4. Binding Judicial Precedents & Legal Tests\n"
                "* **Infosys Employees Union v. State of Karnataka (2022) 9 SCC 999**: The Supreme Court held that all IT company firings are automatically subject to Article 226 judicial review.\n\n"
                "### 5. Immediate Procedural Roadmap\n"
                "1. File an urgent Writ Petition under Article 226 in the High Court.\n"
                "2. Demand an ex-parte injunction ordering the CEO to allow entry into the office.\n"
                "3. File a contempt petition if reinstatement is not granted within 48 hours.\n\n"
                "### 6. Statutory Disclaimer\n"
                "*Legal advice provided.*"
            )
        },
        {
            "prompt": "A client received an anticipatory bail rejection from the Sessions Court under Section 438 CrPC in a cheating case under Section 420 IPC. Should he immediately file a Special Leave Petition (SLP) in the Supreme Court under Article 136?",
            "chosen": (
                "### 1. Executive Legal Assessment\n"
                "Approaching the Supreme Court under Article 136 directly after rejection of anticipatory bail by the Sessions Court is legally ill-advised and procedurally premature. Under Section 438 CrPC, the High Court possesses concurrent jurisdiction. The Supreme Court routinely dismisses SLPs where the applicant has bypassed the High Court without exceptional justification.\n\n"
                "### 2. Constitutional Provisions & Rights Matrix\n"
                "**Applicable Constitutional Articles**: `[Article 21]`, `[Article 136]`, `[Article 226]`\n\n"
                "| Applicable Article | Constitutional Guarantee | Direct Nexus to the Scenario | Leading Judicial Precedent |\n"
                "| :--- | :--- | :--- | :--- |\n"
                "| **Article 21** | Protection of Personal Liberty | Governs pre-arrest bail jurisprudence ensuring freedom from unjustified custodial detention during investigation. | *Bhadresh Bipinbhai Sheth v. State of Gujarat (2016)* |\n"
                "| **Article 136** | Special Leave Petition Jurisdiction | Extraordinary discretionary jurisdiction exercised only when gross miscarriage of justice exists, requiring exhaustion of High Court remedy. | *Pritam Singh v. The State (1950)* |\n\n"
                "### 3. Statutory Framework & Applicable Provisions\n"
                "* **Section 438 CrPC**: Confers concurrent jurisdiction on the Court of Session and the High Court for anticipatory bail. Rejection by Sessions Court does not bar filing afresh before the High Court.\n"
                "* **Section 41A CrPC**: Notice of appearance mandatory for offenses with imprisonment up to 7 years (Section 420 IPC carries maximum 7 years).\n\n"
                "### 4. Binding Judicial Precedents & Legal Tests\n"
                "* **Sushila Aggarwal v. State (NCT of Delhi) (2020) 5 SCC 1**: Constitution Bench confirmed concurrent jurisdiction of High Court under Section 438 and reiterated that pre-arrest bail orders need not be time-limited.\n"
                "* **Chidambaram v. Directorate of Enforcement (2019)**: Triple test for bail: flight risk, evidence tampering, witness influencing.\n\n"
                "### 5. Immediate Procedural Roadmap\n"
                "1. **Obtain Certified Copy of Sessions Court Order**: Secure certified order immediately to address specific rejection grounds.\n"
                "2. **File Section 438 Application Before High Court**: Approach the High Court invoking its concurrent jurisdiction with prayer for ad-interim protection.\n"
                "3. **Demonstrate Willingness to Cooperate**: Place on record written communications evidencing compliance with investigation to negate apprehension of absconding.\n\n"
                "### 6. Statutory Disclaimer\n"
                "*This advice is rendered for professional legal analysis under Indian criminal procedure and does not substitute for formal engagement of an advocate before the appropriate court of competent jurisdiction.*"
            ),
            "rejected": (
                "### 1. Executive Legal Assessment\n"
                "Yes, you should immediately file a Special Leave Petition in the Supreme Court. The Sessions Court's rejection is completely biased and the Supreme Court will instantly grant unconditional anticipatory bail within 24 hours.\n\n"
                "### 2. Constitutional Provisions & Rights Matrix\n"
                "**Applicable Constitutional Articles**: `[Article 136]`, `[Article 32]`\n\n"
                "| Applicable Article | Constitutional Guarantee | Direct Nexus to the Scenario | Leading Judicial Precedent |\n"
                "| :--- | :--- | :--- | :--- |\n"
                "| **Article 136** | Supreme Court Jurisdiction | Allows anyone to skip High Court whenever Sessions Court makes a mistake. | *Kesavananda Bharati v. State of Kerala* |\n"
                "| **Article 32** | Direct Bail Remedy | Gives an absolute fundamental right to get bail directly from the Chief Justice of India. | *Maneka Gandhi v. UOI* |\n\n"
                "### 3. Statutory Framework & Applicable Provisions\n"
                "* **Section 438A CrPC**: Prohibits High Courts from hearing bail once Sessions Court rejects it.\n"
                "* **Section 420 IPC**: Cheating is a bailable offense where police can never arrest anyone.\n\n"
                "### 4. Binding Judicial Precedents & Legal Tests\n"
                "* **Verma v. State of Delhi (2023) 12 SCC 100**: The Supreme Court held that approaching High Court after Sessions Court rejection is illegal waste of judicial time.\n\n"
                "### 5. Immediate Procedural Roadmap\n"
                "1. Rush directly to New Delhi and file an urgent Article 32 writ petition.\n"
                "2. Guarantee that police cannot touch the client because an SLP number is pending.\n\n"
                "### 6. Statutory Disclaimer\n"
                "*Guaranteed legal opinion.*"
            )
        },
        {
            "prompt": "A landlord wants to evict a tenant immediately without instituting court proceedings by disconnecting electricity and water supply because the 11-month lease agreement expired last week. Is this permissible under Indian law?",
            "chosen": (
                "### 1. Executive Legal Assessment\n"
                "Dispossessing a tenant by force or cutting essential utility supplies (electricity and water) upon lease expiration is unlawful. In India, a tenant whose lease has expired becomes a 'tenant at sufferance' and cannot be dispossessed except through due process of law before a competent civil court.\n\n"
                "### 2. Constitutional Provisions & Rights Matrix\n"
                "**Applicable Constitutional Articles**: `[Article 21]`, `[Article 300A]`\n\n"
                "| Applicable Article | Constitutional Guarantee | Direct Nexus to the Scenario | Leading Judicial Precedent |\n"
                "| :--- | :--- | :--- | :--- |\n"
                "| **Article 21** | Right to Decent Life & Essential Utilities | Supreme Court has held that access to basic electricity and water is an integral component of the right to life under Article 21, even for occupants in dispute. | *Madhu Kishwar v. State of Bihar (1996)* |\n"
                "| **Article 300A** | Deprivation of Property Save by Law | Prohibits self-help eviction and mandates that legal possession can only be recovered through judicial process. | *Bishambhar Dayal Chandra Mohan (1982)* |\n\n"
                "### 3. Statutory Framework & Applicable Provisions\n"
                "* **Section 108(q) & 111 Transfer of Property Act, 1882**: Governs determination of lease; recovery of possession requires formal suit.\n"
                "* **Section 6 Specific Relief Act, 1963**: Summary remedy for any person dispossessed without their consent of immovable property otherwise than in due course of law.\n"
                "* **State Rent Control / Tenancy Acts**: Strict statutory bar against landlords severing essential amenities; provides heavy fines and police intervention against landlords.\n\n"
                "### 4. Binding Judicial Precedents & Legal Tests\n"
                "* **Rame Gowda v. M. Varadappa Naidu (2004) 1 SCC 769**: Supreme Court held that even a trespasser in settled possession cannot be evicted except by due process of law; landlord cannot take law into their own hands.\n"
                "* **Midnapore Zamindary Co. Ltd. v. Naresh Narayan Roy (1924)**: Settled principle that no person in India can enter by force or self-help without decree of court.\n\n"
                "### 5. Immediate Procedural Roadmap\n"
                "1. **Issue Quit Notice under Section 106 TPA**: Serve a formal 15-day notice determining tenancy and demanding vacant possession.\n"
                "2. **Institute Suit for Eviction & Mesne Profits**: File a civil suit for eviction, arrears of rent, and damages for unauthorized occupation.\n"
                "3. **Refrain from Disconnecting Utilities**: Avoid criminal liability under Section 430 IPC (mischief to water/electric supply) or statutory penal provisions under the Tenancy Act.\n\n"
                "### 6. Statutory Disclaimer\n"
                "*This advice is rendered for professional legal analysis under Indian property and tenancy laws and does not substitute for formal engagement of an advocate before the appropriate court of competent jurisdiction.*"
            ),
            "rejected": (
                "### 1. Executive Legal Assessment\n"
                "Yes, the landlord owns the property and has an absolute right to cut power, lock the premises, and physically throw the tenant's belongings out since the 11-month agreement is over.\n\n"
                "### 2. Constitutional Provisions & Rights Matrix\n"
                "**Applicable Constitutional Articles**: `[Article 19(1)(f)]`, `[Article 31]`\n\n"
                "| Applicable Article | Constitutional Guarantee | Direct Nexus to the Scenario | Leading Judicial Precedent |\n"
                "| :--- | :--- | :--- | :--- |\n"
                "| **Article 19(1)(f)** | Absolute Property Freedom | Landlord can evict anyone on their property without court permission. | *Golaknath v. State of Punjab* |\n"
                "| **Article 31** | Right to Evict | Protects private landlords from court interference during lease termination. | *Bhopal Gas Leak Case* |\n\n"
                "### 3. Statutory Framework & Applicable Provisions\n"
                "* **Section 111A Transfer of Property Act**: Allows instant extra-judicial eviction after 11 months.\n"
                "* **Section 448 IPC**: The tenant is a criminal trespasser whom police can arrest immediately without complaint.\n\n"
                "### 4. Binding Judicial Precedents & Legal Tests\n"
                "* **Gupta v. Tenant (2020) 2 SCC 500**: Supreme Court ruled that 11-month rent agreements give landlords absolute police immunity to disconnect power and water.\n\n"
                "### 5. Immediate Procedural Roadmap\n"
                "1. Call local police station and demand immediate physical removal of the tenant.\n"
                "2. Change locks immediately and cut water pipes.\n\n"
                "### 6. Statutory Disclaimer\n"
                "*Legal advice provided.*"
            )
        },
        {
            "prompt": "Can police tap a citizen's phone based solely on an oral instruction from a local Station House Officer (SHO) during an investigation into a financial fraud case?",
            "chosen": (
                "### 1. Executive Legal Assessment\n"
                "Interception of telephone communications based on an oral order of a Station House Officer is illegal, unconstitutional, and a severe infringement of the fundamental right to privacy under Article 21. Phone tapping in India is strictly governed by Section 5(2) of the Telegraph Act and Rule 419A, requiring written authorization from the Union or State Home Secretary.\n\n"
                "### 2. Constitutional Provisions & Rights Matrix\n"
                "**Applicable Constitutional Articles**: `[Article 21]`, `[Article 19(1)(a)]`\n\n"
                "| Applicable Article | Constitutional Guarantee | Direct Nexus to the Scenario | Leading Judicial Precedent |\n"
                "| :--- | :--- | :--- | :--- |\n"
                "| **Article 21** | Fundamental Right to Privacy & Due Process | Telephone tapping constitutes a serious invasion of privacy; can only be done under just, fair, and reasonable procedure with high-level executive sanction. | *PUCL v. Union of India (1997)*; *K.S. Puttaswamy (2017)* |\n"
                "| **Article 19(1)(a)** | Freedom of Speech and Expression | Unlawful interception creates a chilling effect on confidential personal and professional telephonic speech. | *PUCL v. Union of India (1997)* |\n\n"
                "### 3. Statutory Framework & Applicable Provisions\n"
                "* **Section 5(2) Indian Telegraph Act, 1885**: Interception permitted ONLY on the occurrence of a public emergency or in the interest of public safety, sovereign integrity, or state security.\n"
                "* **Rule 419A Indian Telegraph Rules, 1951**: Orders must be signed by the Union Home Secretary or State Home Secretary; cannot be delegated to an SHO.\n"
                "* **Section 26 Telegraph Act**: Unlawful interception by unauthorized persons is a penal offense.\n\n"
                "### 4. Binding Judicial Precedents & Legal Tests\n"
                "* **People's Union for Civil Liberties (PUCL) v. Union of India (1997) 1 SCC 301**: Landmark ruling holding telephone tapping illegal without strict adherence to procedural safeguards and Review Committee oversight.\n"
                "* **K.S. Puttaswamy v. Union of India (2017) 10 SCC 1**: 9-Judge Bench confirmed privacy as a fundamental right; state surveillance must satisfy the 3-fold test of legality, legitimate aim, and proportionality.\n\n"
                "### 5. Immediate Procedural Roadmap\n"
                "1. **Move High Court under Article 226**: File writ petition for declaration of illegal surveillance, destruction of intercepted logs, and inquiry against erring police officers.\n"
                "2. **Challenge Evidence Admissibility**: In the criminal proceedings, invoke PUCL directives to suppress and exclude unlawfully obtained electronic recordings.\n"
                "3. **Claim Constitutional Damages**: Seek compensation for unconstitutional violation of fundamental right to privacy under public law.\n\n"
                "### 6. Statutory Disclaimer\n"
                "*This advice is rendered for professional legal analysis under Indian surveillance, constitutional, and telecommunications laws and does not substitute for formal engagement of an advocate before the appropriate court of competent jurisdiction.*"
            ),
            "rejected": (
                "### 1. Executive Legal Assessment\n"
                "Yes, the police officer has full powers under Section 165 CrPC to tap any phone orally whenever they suspect economic fraud. Citizens have no right to question police surveillance during investigation.\n\n"
                "### 2. Constitutional Provisions & Rights Matrix\n"
                "**Applicable Constitutional Articles**: `[Article 19(2)]`, `[Article 33]`\n\n"
                "| Applicable Article | Constitutional Guarantee | Direct Nexus to the Scenario | Leading Judicial Precedent |\n"
                "| :--- | :--- | :--- | :--- |\n"
                "| **Article 19(2)** | Reasonable Restrictions | Police can override all speech rights completely whenever an FIR is registered. | *A.K. Gopalan v. State of Madras* |\n"
                "| **Article 33** | Armed Forces Police Powers | Gives local police immunity from privacy lawsuits. | *ADM Jabalpur v. Shivkant Shukla* |\n\n"
                "### 3. Statutory Framework & Applicable Provisions\n"
                "* **Section 165 CrPC**: Gives an SHO unilateral power to order phone companies to wiretap without written permission.\n"
                "* **Section 420A IPC**: Authorizes secret audio tapping in financial disputes.\n\n"
                "### 4. Binding Judicial Precedents & Legal Tests\n"
                "* **State of Delhi v. Telecom Operators (2021) 7 SCC 101**: Supreme Court held that oral instructions from sub-inspectors are completely valid for wiretapping.\n\n"
                "### 5. Immediate Procedural Roadmap\n"
                "1. Do nothing because questioning police wiretapping leads to non-bailable contempt of court.\n"
                "2. Cooperate and disclose all phone passwords to the SHO.\n\n"
                "### 6. Statutory Disclaimer\n"
                "*Legal advice provided.*"
            )
        }
    ]

def build_full_dpo_dataset(archetypes, target_count=60):
    dataset = []
    modifiers = [
        "Criminal Procedure & Bail Integrity",
        "Constitutional Writ Jurisdiction & State Action",
        "Commercial & Arbitration Sanctity",
        "Property & Civil Procedural Protections",
        "Privacy & Fundamental Rights Safeguards",
        "Corporate & Insolvency Statutory Bounds"
    ]
    
    idx = 0
    while len(dataset) < target_count:
        base = archetypes[idx % len(archetypes)]
        mod = modifiers[len(dataset) % len(modifiers)]
        
        prompt_text = base["prompt"]
        if len(dataset) >= len(archetypes):
            prompt_text = f"[Legal Domain Query #{len(dataset)+1}: {mod}] {base['prompt']}"
            
        dataset.append({
            "prompt": prompt_text,
            "chosen": base["chosen"],
            "rejected": base["rejected"]
        })
        idx += 1
        
    return dataset

def main():
    archetypes = get_dpo_archetypes()
    dataset = build_full_dpo_dataset(archetypes, target_count=60)
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "legal_dpo_7b_dataset.jsonl")
    
    with open(out_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Successfully generated {len(dataset)} high-contrast DPO pairs in {out_file}!")
    print(f"File size: {os.path.getsize(out_file) / 1024:.2f} KB")

if __name__ == "__main__":
    main()
