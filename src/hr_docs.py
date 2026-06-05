"""
Realistic HR policy documents for Acme Corp.
Used as the knowledge base for the RAG pipeline.
"""

HR_DOCUMENTS = [
    {
        "title": "Annual Leave & Time Off Policy",
        "content": """
Annual Leave & Time Off Policy — Acme Corp

VACATION DAYS
Full-time employees accrue 15 paid vacation days per year for the first three years of service.
After 3 years of continuous employment, accrual increases to 20 days per year.
After 7 years, accrual increases to 25 days per year.
Part-time employees accrue vacation on a pro-rata basis relative to their contracted hours.
Accrual begins on the employee's first day. The leave year runs January 1 to December 31.
Up to 5 unused vacation days may be carried over to the following year and must be used by March 31.
Unused carried-over days are forfeited after March 31.

REQUESTING VACATION
All vacation requests must be submitted through the HR portal (hr.acmecorp.com) at least 10 business days in advance.
Emergency requests are reviewed case-by-case by your direct manager.
Approval depends on team coverage and operational requirements.
Blackout periods (e.g., Q4 year-end close: November 25 – December 10) require department-head approval.

SICK LEAVE
Employees receive 10 paid sick days per calendar year.
Sick leave resets on January 1 and does not carry over or pay out.
A doctor's note is required for absences exceeding 3 consecutive working days.
Sick leave cannot be substituted for vacation leave.

PERSONAL DAYS
3 personal days per year for appointments, religious observances, or personal needs.
Must be requested at least 48 hours in advance where possible.
Personal days do not carry over to the next year.

PARENTAL LEAVE
Primary caregivers receive 16 weeks of fully paid parental leave.
Secondary caregivers receive 4 weeks of fully paid parental leave.
Leave can begin any time within the first 12 months following birth or adoption.
Notify HR and your manager at least 8 weeks before the intended start date.

BEREAVEMENT LEAVE
5 paid days for the death of an immediate family member (spouse, child, parent, sibling).
3 paid days for grandparents, grandchildren, or in-laws.
1 paid day for other extended family members.
Additional unpaid leave may be granted at manager discretion.

JURY DUTY & CIVIC LEAVE
Paid leave for jury duty up to 15 working days per calendar year.
Employees must provide the court summons to HR within 2 business days of receipt.
""".strip()
    },
    {
        "title": "Remote Work & Hybrid Policy",
        "content": """
Remote Work & Hybrid Policy — Acme Corp

OVERVIEW
Acme Corp supports hybrid and remote work arrangements that balance flexibility with collaboration.
This policy applies to all full-time and part-time employees whose roles can be performed remotely.

ELIGIBILITY
Employees must have completed their 90-day probationary period.
The role must not require physical presence (lab work, on-site client service, hardware support).
The employee must have demonstrated satisfactory performance (rating of 3 or above).

STANDARD HYBRID SCHEDULE
Default hybrid schedule: 3 days per week in the office.
Core in-office days: Tuesday, Wednesday, Thursday.
Remote days: Monday and Friday (subject to manager confirmation).
Teams may agree on alternate schedules with manager and HR approval.

HOME OFFICE REQUIREMENTS
A dedicated, distraction-free workspace with a door (or equivalent privacy).
Reliable broadband internet: minimum 25 Mbps download, 10 Mbps upload.
Company-issued laptop or approved personal device with required security software installed.
A working webcam, headset, and microphone for video calls.
Acme Corp provides a one-time home office setup allowance of $600 for eligible equipment (receipts required).

CORE HOURS & AVAILABILITY
All employees must be available and responsive between 9:00 AM and 5:00 PM local time on weekdays.
Status must be kept current in Slack (online/away/do not disturb).
Response time for non-urgent messages: within 2 hours during core hours.
All team meetings must be attended on video unless medical accommodation is in place.

SECURITY WHILE REMOTE
Use the company VPN at all times when accessing internal systems.
Never use public Wi-Fi (coffee shops, airports) without the VPN active.
Lock your screen when stepping away from your desk.
Do not share your screen in public spaces.
Violations may result in immediate suspension of remote work privileges pending investigation.

HOW TO REQUEST A REMOTE ARRANGEMENT
Discuss with your direct manager first.
Submit a formal Remote Work Agreement form through the HR portal.
Arrangements are reviewed every 6 months and can be revoked for performance or security reasons.
""".strip()
    },
    {
        "title": "Code of Conduct",
        "content": """
Code of Conduct — Acme Corp

PURPOSE
This Code sets the standard of professional behaviour expected of all Acme Corp employees,
contractors, and representatives. It reflects our values: integrity, respect, accountability, and excellence.

PROFESSIONAL BEHAVIOUR
Be punctual, reliable, and meet your commitments.
Communicate respectfully in all written and verbal interactions.
Represent Acme Corp professionally at client sites, events, and online.
Treat all colleagues, clients, vendors, and partners with courtesy and dignity.

CONFLICTS OF INTEREST
Disclose in writing to HR any situation where personal interests may conflict with Acme Corp's interests.
Examples: outside employment with a competitor, financial interest in a supplier, personal relationship with a direct report.
Failure to disclose known conflicts is a disciplinary offence.

CONFIDENTIALITY
Protect all Acme Corp confidential information: business plans, customer data, financial results, source code, and personnel records.
Do not share confidential information with third parties without written authorisation.
This obligation continues after your employment ends.

SOCIAL MEDIA
Do not disclose confidential company information on any social media platform.
When posting as an Acme Corp employee, add: "Views are my own, not Acme Corp's."
Disparaging posts about colleagues, clients, or the company are prohibited.

GIFTS & ENTERTAINMENT
You may give or accept gifts or entertainment worth up to $100 per occasion without reporting.
Gifts above $100 must be reported to HR within 5 business days and returned or donated to charity.
Never accept cash or cash equivalents (gift cards, vouchers) from vendors or clients.

DISCIPLINARY PROCESS
Minor violations: verbal warning → written warning → final written warning → termination.
Serious misconduct (fraud, harassment, gross negligence, data theft): immediate termination without prior warning.
All disciplinary actions are documented and held in the employee's HR file.

REPORTING VIOLATIONS
Report concerns to your manager, HR, or anonymously via ethics@acmecorp.com.
You can also call the Ethics Hotline: 1-800-555-ETHICS (available 24/7).
Retaliation against anyone who reports a concern in good faith is strictly prohibited.
""".strip()
    },
    {
        "title": "Performance Review Process",
        "content": """
Performance Review Process — Acme Corp

REVIEW CYCLE
Mid-year check-in: conducted in July. Focus: goal progress, feedback, and course corrections.
Annual review: conducted in December. Full assessment of performance, goals, and compensation impact.
Monthly 1:1 meetings between managers and direct reports are required year-round.

GOAL SETTING
At the start of each year, employees and managers jointly set 3–5 SMART goals.
Goals must be documented in the HR portal by January 31.
Goals may be revised mid-year with mutual agreement, documented through the HR portal.
Individual goals must align to team and company OKRs.

PERFORMANCE RATING SCALE
5 – Exceptional: Significantly exceeds all goals with outstanding impact.
4 – Exceeds Expectations: Consistently surpasses goals; high performer.
3 – Meets Expectations: Fully achieves all goals; solid contributor.
2 – Needs Improvement: Partially meets goals; a formal development plan is required.
1 – Unsatisfactory: Falls significantly short of expectations; at risk of termination.

REVIEW PROCESS STEPS
Step 1: Employee completes self-assessment in HR portal (due November 30).
Step 2: Manager writes assessment and selects a provisional rating.
Step 3: Calibration session with HR and peer managers to ensure consistency across teams.
Step 4: Manager delivers final review in a face-to-face or video meeting.
Step 5: Employee acknowledges the review in the HR portal within 5 business days.

COMPENSATION IMPACT
Rating 5 (Exceptional): salary increase of 5–8% plus eligibility for equity refresh.
Rating 4 (Exceeds Expectations): salary increase of 3–5%.
Rating 3 (Meets Expectations): salary increase of 2–3%.
Rating 2 (Needs Improvement): 0–1% increase; mandatory Performance Improvement Plan (PIP).
Rating 1 (Unsatisfactory): no increase; PIP required; risk of termination.

PERFORMANCE IMPROVEMENT PLAN (PIP)
Issued to employees with a rating of 2 or below.
PIP duration: 60–90 days.
Outlines specific, measurable improvement targets and check-in cadence.
Failure to meet PIP targets may result in role reassignment or termination.

PROMOTIONS
Promotion decisions are made once per year following the December review.
Standard eligibility: rating of 4 or 5 for two consecutive years.
All promotions require department-head sign-off and HR approval.
""".strip()
    },
    {
        "title": "Benefits & Compensation",
        "content": """
Benefits & Compensation Overview — Acme Corp

COMPENSATION PHILOSOPHY
Acme Corp benchmarks salaries at the 65th percentile of market rates for equivalent roles.
Benchmarking is updated annually using third-party compensation surveys.
Salaries reflect role scope, level, performance, location, and market conditions.

HEALTH INSURANCE
Comprehensive medical, dental, and vision coverage for all full-time employees and eligible dependants.
Medical: Acme Corp covers 80% of the premium for employee-only plans; 60% for family plans.
Coverage starts on the first day of employment.
Open enrollment: every November. New employees must enrol within 30 days of start date.

DENTAL COVERAGE
Two cleanings and annual check-ups per year at no additional cost.
Basic restorations (fillings): 80% covered.
Major restorations (crowns, bridges): 50% covered, up to an annual maximum of $2,000.

VISION COVERAGE
One annual eye exam fully covered.
Up to $200 per year towards frames, prescription lenses, or contact lenses.

RETIREMENT — 401(K) PLAN
Acme Corp matches 100% on the first 2% of salary contributed, plus 50% on the next 2%.
Maximum company match: 3% of base salary.
Eligibility: after 90 days of employment.
Vesting schedule: 25% Year 1, 50% Year 2, 75% Year 3, 100% Year 4.

WELLNESS ALLOWANCE
$600 per year, reimbursable for gym memberships, fitness classes, sports equipment, or mental health apps.
Submit receipts through Concur with the category "Wellness Benefit" within 60 days of purchase.

EMPLOYEE ASSISTANCE PROGRAMME (EAP)
6 free confidential counselling sessions per year via our third-party provider, BetterHelp.
Covers personal, financial, and legal guidance.
Access via eap.acmecorp.com or call 1-800-EAP-HELP.

LIFE & DISABILITY INSURANCE
Basic life insurance: 2× annual base salary, fully company-paid.
Short-term disability: 60% of base salary for up to 12 weeks.
Long-term disability: 60% of base salary after a 90-day elimination period.

ADDITIONAL PERKS
Learning & Development budget: $1,500 per year for courses, certifications, or conferences.
Employee referral bonus: $2,000 for a successful hire (paid after the referred employee passes 90 days).
Commuter benefit: pre-tax transit or parking account up to $300/month.
Company-wide paid holiday shutdown: December 25 – January 1.
""".strip()
    },
    {
        "title": "Anti-Harassment & Discrimination Policy",
        "content": """
Anti-Harassment & Discrimination Policy — Acme Corp

ZERO TOLERANCE
Acme Corp is committed to a workplace free from harassment, discrimination, and retaliation.
This policy applies to all employees, contractors, interns, clients, and visitors.
Violations are treated with the utmost seriousness and may result in immediate termination.

PROTECTED CHARACTERISTICS
Harassment or discrimination based on any of the following is strictly prohibited:
Race, colour, ethnicity, national origin, religion, age, gender, gender identity,
sexual orientation, disability, pregnancy, marital status, or any other legally protected characteristic.

WHAT COUNTS AS HARASSMENT
Unwelcome verbal conduct: offensive jokes, slurs, name-calling, threats, or intimidation.
Unwelcome physical conduct: unwanted touching, blocking movement, physical threats.
Visual conduct: displaying or sharing offensive images, memes, or objects.
Sexual conduct: advances, requests for sexual favours, or sexually suggestive communication.
Cyberbullying: harassment via email, Slack, social media, or any digital channel.
Creating a hostile work environment through any of the above.

HOW TO REPORT
Contact your manager (if they are not the subject of the complaint).
Contact HR directly at hr@acmecorp.com or by phone at +1-415-555-0199.
Report anonymously via the Ethics Hotline: ethics@acmecorp.com or 1-800-555-ETHICS.
Reports can be made at any time; there is no deadline to come forward.

INVESTIGATION PROCESS
HR acknowledges the complaint within 2 business days.
A neutral investigator (internal HR or external counsel) is assigned.
Both the complainant and respondent have the opportunity to present their account.
Investigation is typically completed within 30 days; complex cases may take up to 60 days.
Findings and any disciplinary action are communicated to both parties in writing.

NON-RETALIATION
Retaliation against anyone who reports, participates in an investigation, or opposes prohibited conduct
is itself a serious violation of this policy.
Retaliatory actions include: demotion, reassignment, poor performance reviews, exclusion, or termination.
Any suspected retaliation must be reported immediately to HR.

CONSEQUENCES
Substantiated violations result in disciplinary action proportionate to severity:
ranging from mandatory training and a written warning to immediate termination.
Acme Corp reserves the right to pursue civil or criminal action where appropriate.
""".strip()
    },
]
