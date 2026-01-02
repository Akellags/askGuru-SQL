
from openai import OpenAI
import json

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8000/v1"
)

# Complete prompt format (as used in training)
user_prompt = """You are a Text-to-SQL generator for Oracle EBS.
Return ONLY Oracle SQL. No markdown. No explanations. No comments.

# Candidate Tables
- AP_INVOICES: INVOICE_ID, VENDOR_ID, INVOICE_NUM, STATUS, AMOUNT
- AP_SUPPLIERS: VENDOR_ID, SEGMENT1, VENDOR_NAME

# Join Graph
- AP_INVOICES.VENDOR_ID = AP_SUPPLIERS.VENDOR_ID

# Relevant Columns
- AP_INVOICES.STATUS: VARCHAR2
- AP_INVOICES.AMOUNT: NUMBER
- AP_SUPPLIERS.VENDOR_NAME: VARCHAR2

[User Question]
Count total amount of paid invoices by supplier

[SQL]
"""

prompt=""" 
You are a Oracle EBS expert. Generate executable SQL based on the user's question.
Only output SQL query. Do not invent columns - use only those in the schema.
CRITICAL: Columns marked with **[ESSENTIAL]** are mandatory for proper joins and aggregations - prioritize them.

[User Question]
Give me the list of invoices where the purchase order payment terms is different from invoice payment terms  Filters/Constraints: operating unit vision operation only  Return ONLY these columns: invoice number, invoice date, puchase order number, purchase order payment terms, invoice payment terms

[Database Schema]
AP_SUPPLIERS (
  VENDOR_ID NUMBER -- Primary key; unique identifier for each supplier (vendor).
  VENDOR_NAME VARCHAR2 -- Supplier name; full name of the vendor.
  SET_OF_BOOKS_ID NUMBER -- Set of Books identifier; associates supplier to a ledger.
  SEGMENT1 VARCHAR2 -- Supplier number; a unique user-defined identifier for the supplier.
  VENDOR_TYPE_LOOKUP_CODE VARCHAR2 -- Supplier type/classification (e.g., EMPLOYEE, EXTERNAL, STATUTORY).
  INVOICE_CURRENCY_CODE VARCHAR2 -- Default currency used for invoices from this supplier.
  DISTRIBUTION_SET_ID NUMBER -- Identifier for the default distribution set for this supplier.
  AMOUNT_INCLUDES_TAX_FLAG VARCHAR2 -- Indicates if amounts include tax (Y/N).
  EMPLOYEE_ID NUMBER -- If supplier is an employee, links to employee record (FK to PER_ALL_PEOPLE_F).
  HOLD_ALL_PAYMENTS_FLAG VARCHAR2 -- Indicates if all payments are held for this supplier (Y/N).
  HOLD_FUTURE_PAYMENTS_FLAG VARCHAR2 -- Indicates if future payments are held (Y/N).
  ONE_TIME_FLAG VARCHAR2 -- Identifies if supplier is a one-time vendor (Y/N).
)

AP_INVOICES_ALL (
  **[ESSENTIAL]** TERMS_ID NUMBER -- Payment terms id
  INVOICE_NUM VARCHAR2 -- Invoice number.
  INVOICE_DATE DATE -- Invoice date.
  INVOICE_AMOUNT NUMBER -- Invoice total amount.
  VENDOR_ID NUMBER -- Supplier identifier.
  VENDOR_SITE_ID NUMBER -- Supplier site identifier.
  ORG_ID NUMBER -- Organization identifier (multi-org support).
  PAYMENT_STATUS_FLAG VARCHAR2 -- Payment status (Y=Paid, N=Unpaid, P=Partial).
  GL_DATE DATE -- Accounting date default for invoice distributions.
  SET_OF_BOOKS_ID NUMBER -- Set of books identifier.
  CREATION_DATE DATE -- Date the record was created.
  CREATED_BY NUMBER -- User who created the record (FND_USER.USER_ID).
)

HR_OPERATING_UNITS (
  ORGANIZATION_ID NUMBER(15) -- Primary key; the operating unit identifier (commonly referred to as org_id).
  SET_OF_BOOKS_ID VARCHAR2(150) -- Ledger ID to which the org_id is associated.
  BUSINESS_GROUP_ID NUMBER(15) -- Business group ID to which the operating unit is associated.
  NAME VARCHAR2(240) -- Name of the operating unit.
  DATE_FROM DATE -- Date from which the operating unit is effective.
  SHORT_CODE VARCHAR2(150) -- Short code for the operating unit.
  DEFAULT_LEGAL_CONTEXT_ID VARCHAR2(150) -- Default legal context ID.
  USABLE_FLAG VARCHAR2(150) -- Indicates whether the operating unit is usable.
)

AP_INVOICE_DISTRIBUTIONS_ALL (
  ORG_ID NUMBER -- Organization identifier.
  SET_OF_BOOKS_ID NUMBER -- Set of books identifier.
  UNIT_PRICE NUMBER -- Unit price (for PO/receipt matched distributions and corrections).
  ACCOUNTING_DATE DATE -- Accounting date.
  ACCOUNTING_EVENT_ID NUMBER -- Accounting event identifier (refers to the event that accounted the distribution).
  ACCRUAL_POSTED_FLAG VARCHAR2 -- Accrual accounting status (Y/N).
  ADJUSTMENT_REASON VARCHAR2 -- Reason for expense adjustment.
  AMOUNT NUMBER -- Invoice distribution amount.
  AMOUNT_VARIANCE NUMBER -- Amount variance in entered currency (for matched service/receipt).
  ASSET_BOOK_TYPE_CODE VARCHAR2 -- Asset book type (defaults from invoice line).
  ASSET_CATEGORY_ID NUMBER -- Asset category (defaults from invoice line).
  ASSETS_ADDITION_FLAG VARCHAR2 -- Flag indicating if distribution transferred into Oracle Assets (U, Y, or N).
)

AP_SUPPLIER_SITES_ALL (
  VENDOR_ID NUMBER -- Foreign key linking to AP_SUPPLIERS; associates the site with a supplier.
  VENDOR_SITE_ID NUMBER -- Primary key; unique identifier for the supplier site.
  ORG_ID NUMBER -- Operating unit identifier for multi-org support.
  VENDOR_SITE_CODE VARCHAR2 -- Site code name (e.g., 'HQ', 'Warehouse').
  ADDRESS_LINE1 VARCHAR2 -- First line of the supplier address.
  ADDRESS_LINE2 VARCHAR2 -- Second line of the supplier address.
  ADDRESS_LINE3 VARCHAR2 -- Third line of the supplier address.
  ADDRESS_LINE4 VARCHAR2 -- Fourth line of the supplier address (if applicable).
  ADDRESS_STYLE VARCHAR2 -- Format/style of the address as per regional conventions.
  INVOICE_CURRENCY_CODE VARCHAR2 -- Default currency used for invoices from this site.
  PARTY_SITE_ID NUMBER -- TCA Party Site ID; links to Trading Community Architecture (TCA).
)

PO_HEADERS_ALL (
  VENDOR_ID NUMBER -- Supplier identifier (FK to AP_SUPPLIERS).
  VENDOR_SITE_ID NUMBER -- Supplier site identifier (FK to AP_SUPPLIER_SITES_ALL).
  ORG_ID NUMBER -- Organization identifier (multi-org).
  CURRENCY_CODE VARCHAR2 -- Currency code for the PO (FK to FND_CURRENCIES).
  CREATION_DATE DATE -- Date the record was created.
  PO_HEADER_ID NUMBER -- Primary key; unique identifier of the PO header.
  ACCEPTANCE_DUE_DATE DATE -- Date by which the supplier should accept the purchase order.
  AGENT_ID NUMBER -- Buyer unique identifier (FK to PER_ALL_PEOPLE_F).
  APPROVED_DATE DATE -- Date the purchase order was last approved.
  APPROVED_FLAG VARCHAR2 -- Indicates whether the purchase order is approved (Y/N).
  AUTHORIZATION_STATUS VARCHAR2 -- Authorization status of the purchase order.
  CANCEL_FLAG VARCHAR2 -- Indicates whether the purchase order is cancelled (Y/N).
)

PO_LINES_ALL (
  ORG_ID NUMBER -- Organization identifier (multi-org).
  PO_HEADER_ID NUMBER -- Document header unique identifier (FK to PO_HEADERS_ALL).
  PO_LINE_ID NUMBER -- Primary key; document line unique identifier.
  ITEM_ID NUMBER -- Item unique identifier (FK to MTL_SYSTEM_ITEMS_B).
  QUANTITY NUMBER -- Quantity ordered for the line item.
  UNIT_PRICE NUMBER -- Unit price for the line item.
  AMOUNT NUMBER -- Budget amount for temporary labor standard PO lines.
  BASE_QTY NUMBER -- OPM: Ordered quantity converted from transaction UOM to base UOM.
  BASE_UNIT_PRICE NUMBER -- Base unit price for the item.
  BASE_UOM VARCHAR2 -- OPM: Base unit of measure (UOM) for the item ordered.
  CANCEL_DATE DATE -- Date the PO line was cancelled.
  CANCEL_FLAG VARCHAR2 -- Indicates if the line is cancelled (Y/N).
)

PO_LINE_LOCATIONS_ALL (
  **[ESSENTIAL]** SHIP_TO_ORGANIZATION_ID NUMBER -- mandatory for joining with mtl_system_items or org_organization_definitions
  ORG_ID NUMBER -- Organization identifier (multi-org).
  CREATION_DATE DATE -- Date when the row was created.
  PO_HEADER_ID NUMBER -- Document header unique identifier (FK to PO_HEADERS_ALL).
  PO_LINE_ID NUMBER -- Document line unique identifier (FK to PO_LINES_ALL).
  LINE_LOCATION_ID NUMBER -- Primary key; unique identifier of the shipment schedule.
  QUANTITY NUMBER -- Quantity ordered or break quantity.
  AMOUNT NUMBER -- Amount on the shipment for service lines.
  AMOUNT_BILLED NUMBER -- Amount billed for service lines.
  AMOUNT_CANCELLED NUMBER -- Amount cancelled for service lines.
  AMOUNT_RECEIVED NUMBER -- Amount received for service lines.
  AMOUNT_REJECTED NUMBER -- Amount rejected for service lines.
)

JOIN HINTS:
  - AP_INVOICES_ALL.VENDOR_ID = AP_SUPPLIERS.VENDOR_ID
  - HR_OPERATING_UNITS.ORGANIZATION_ID = AP_INVOICES_ALL.ORG_ID
  - -- TODO: join HR_OPERATING_UNITS to AP_INVOICE_DISTRIBUTIONS_ALL
  - -- TODO: join AP_INVOICE_DISTRIBUTIONS_ALL to AP_SUPPLIER_SITES_ALL
  - AP_SUPPLIER_SITES_ALL.VENDOR_SITE_ID = PO_HEADERS_ALL.VENDOR_SITE_ID
  - PO_HEADERS_ALL.PO_HEADER_ID = PO_LINES_ALL.PO_HEADER_ID
  - PO_LINES_ALL.PO_LINE_ID = PO_LINE_LOCATIONS_ALL.PO_LINE_ID

[Reference Information]
[Rules]
- **CRITICAL**: Examples marked with [* VERY SIMILAR] are semantically matched to your question. Follow their SQL structure, JOIN patterns, WHERE logic, aggregation strategy, and CTE usage EXACTLY. Do NOT simplify or refactor them - they are proven patterns that work for this data model.
- Examples marked with * are most similar to the current question - follow their patterns and structure closely. Mimic their JOIN order, subquery approach (CTE vs NOT EXISTS vs LEFT JOIN), aggregation, and filtering logic.
- **MANDATORY**: Always include all JOIN key columns in the SELECT clause. Every ID or key used to JOIN tables must be present in the final result set.
- Do not invent columns; use only those in [Database Schema].
- When computing closing balance, use this expression exactly:
  NVL(BEGIN_BALANCE_DR,0) - NVL(BEGIN_BALANCE_CR,0)
  + NVL(PERIOD_NET_DR,0) - NVL(PERIOD_NET_CR,0)
- Do NOT reference a column alias inside CASE unless it is computed in a subquery.
  EITHER inline the full expression inside CASE (preferred),
  OR compute it in a subquery and reference it in the outer SELECT.
- Always guard divisions:  / NULLIF(ABS(<denominator>), 0)
- Prefer PERIOD_NAME-based filters (e.g., 'Jan-03','Jun-03'), not literal START_DATEs.
- CRITICAL: Month abbreviations in PERIOD_NAME must ALWAYS be in title case format: 'Jan-03', 'Feb-03', 'Mar-03', 'Apr-03', 'May-03', 'Jun-03', 'Jul-03', 'Aug-03', 'Sep-03', 'Oct-03', 'Nov-03', 'Dec-03'. Do NOT use uppercase (MAR-03, APR-03) or lowercase (mar-03, apr-03).
- When in any join that has MTL_SYSTEM_ITEMS table must necessary have 2 joins. namely

Follow JOIN HINTS. Do not invent columns. Guard divisions with NULLIF.

### Few Shot examples
Question 1: GIVE ME THE PAYMENT DETAILS FOR INVOICE NUMBER CRAC Apr 06 09
Tables: AP_CHECKS_ALL, AP_INVOICES_ALL, AP_INVOICE_PAYMENTS_ALL
SQL:
SELECT
    aip.INVOICE_PAYMENT_ID,
    aip.payment_num,
    aip.CREATION_date, /*--THERE IS NO PAYMENT DATE COLUMN*/
    aip.amount,
    aCA.currency_code,
    ACA.exchange_rate,
    ACA.exchange_rate_type,
    aCA.payment_method_code,
    aip.check_id,
    aca.check_number,
    aca.check_date,
    aca.bank_account_id,
     ACA.PAYMENT_METHOD_CODE,
    aca.STATUS_LOOKUP_CODE,
    aca.creation_date,
    aca.last_update_date,
    aca.created_by,
    aca.last_updated_by
FROM
    ap_invoice_payments_all aip
JOIN
    ap_checks_all aca ON aip.check_id = aca.check_id
JOIN
    ap_invoices_all ai ON aip.invoice_id = ai.invoice_id
WHERE
    ai.invoice_num = 'CRAC Apr 06 09'
ORDER BY
    aip.CREATION_DATE
Question 2: get the list of oustanding invoices as on 31 dec 2007 for vision operations backdate
Tables: AP_INVOICES_ALL, AP_INVOICE_DISTRIBUTIONS_ALL, AP_PAYMENT_SCHEDULES_ALL, AP_SUPPLIERS, CUTOFF_PARAM, DUAL, HR_OPERATING_UNITS, INVOICE_BALANCES, PAYMENTS_AFTER_CUTOFF, PREPAY_AFTER_CUTOFF
SQL:
WITH cutoff_param AS (
    /* Define the cutoff date and operating unit name as parameters */
    SELECT
        TO_DATE('2007-12-31', 'YYYY-MM-DD') AS cutoff_date,
        'Vision Operations' AS operating_unit_name
    FROM dual
),  payments_after_cutoff AS (
    /* Payments made after the cutoff date and restricted to the specified operating unit */
    SELECT
        aip.INVOICE_ID,
        SUM(NVL(aip.AMOUNT, 0)) AS payment_after_cutoff
    FROM
        ap_invoice_payments_all aip
        JOIN ap_invoices_all ai ON aip.INVOICE_ID = ai.INVOICE_ID
        JOIN hr_operating_units hr ON ai.ORG_ID = hr.ORGANIZATION_ID
        JOIN cutoff_param c ON 1 = 1
    WHERE
        aip.ACCOUNTING_DATE > c.cutoff_date
        AND UPPER(hr.NAME) = UPPER(c.operating_unit_name)
    GROUP BY
        aip.INVOICE_ID
),  prepay_after_cutoff AS (
    /* Prepayment adjustments made after the cutoff date and restricted to the specified operating unit */
    SELECT
        aid.INVOICE_ID,
        SUM(NVL(aid.AMOUNT, 0)) AS prepay_after_cutoff
    FROM
        ap_invoice_distributions_all aid
        JOIN ap_invoices_all ai ON aid.INVOICE_ID = ai.INVOICE_ID
        JOIN hr_operating_units hr ON ai.ORG_ID = hr.ORGANIZATION_ID
        JOIN cutoff_param c ON 1 = 1
    WHERE
        aid.LINE_TYPE_LOOKUP_CODE = 'PREPAY'
        AND aid.ACCOUNTING_DATE > c.cutoff_date
        AND UPPER(hr.NAME) = UPPER(c.operating_unit_name)
    GROUP BY
        aid.INVOICE_ID
),  invoice_balances AS (
    /* Total invoice balances (sum of gross_amount across all payment schedules per invoice),
       restricted to the specified operating unit */
    SELECT
        ai.INVOICE_ID,
        ai.INVOICE_NUM,
        ai.INVOICE_DATE,
        ai.VENDOR_ID,
        SUM(aps.GROSS_AMOUNT) AS GROSS_AMOUNT
    FROM
        ap_invoices_all ai
        JOIN ap_payment_schedules_all aps ON ai.INVOICE_ID = aps.INVOICE_ID
        JOIN hr_operating_units hr ON ai.ORG_ID = hr.ORGANIZATION_ID
        JOIN cutoff_param c ON 1 = 1
    WHERE
        UPPER(hr.NAME) = UPPER(c.operating_unit_name)
    GROUP BY
        ai.INVOICE_ID,
        ai.INVOICE_NUM,
        ai.INVOICE_DATE,
        ai.VENDOR_ID
)  SELECT
    /* Final result: unpaid balances per invoice as of the cutoff date */
    s.VENDOR_NAME,
    ib.INVOICE_ID,
    ib.INVOICE_NUM,
    ib.INVOICE_DATE,
    ib.GROSS_AMOUNT,
    NVL(pac.payment_after_cutoff, 0) AS payment_after_cutoff,
    NVL(ppc.prepay_after_cutoff, 0) AS prepay_after_cutoff,
    (ib.GROSS_AMOUNT - NVL(pac.payment_after_cutoff, 0) - NVL(ppc.prepay_after_cutoff, 0)) AS UNPAID_BALANCE_AS_OF_CUTOFF
FROM
    invoice_balances ib
    JOIN ap_suppliers s ON ib.VENDOR_ID = s.VENDOR_ID
    LEFT JOIN payments_after_cutoff pac ON ib.INVOICE_ID = pac.INVOICE_ID
    LEFT JOIN prepay_after_cutoff ppc ON ib.INVOICE_ID = ppc.INVOICE_ID
WHERE
    /* Exclude invoices where unpaid balance is fully cleared */
    (ib.GROSS_AMOUNT - NVL(pac.payment_after_cutoff, 0) - NVL(ppc.prepay_after_cutoff, 0)) <> 0
ORDER BY
    UNPAID_BALANCE_AS_OF_CUTOFF
Question 3: Get the list of payments and advance adjustments against a specific invoice abc123
Tables: AP_INVOICE_PAYMENTS_ALL, AP_CHECKS_ALL, AP_INVOICES_ALL, AP_INVOICE_DISTRIBUTIONS_ALL
SQL:
SELECT ai.INVOICE_ID AS standard_invoice_id,
       ai.INVOICE_NUM AS standard_invoice_number,
       aip.amount AS payment_amount,
       aip.accounting_date AS payment_accounting_date,
       TO_CHAR(ap.check_number) AS payment_reference_number,
       ap.payment_method_lookup_code AS payment_method,
       'PAYMENT' AS settlement_type
  FROM ap_invoices_all ai
       JOIN ap_invoice_payments_all aip ON ai.invoice_id = aip.invoice_id
       JOIN ap_checks_all ap ON aip.check_id = ap.check_id
 WHERE ai.invoice_num = 'abc123' AND ai.cancelled_date IS NULL
UNION ALL
SELECT ai.INVOICE_ID AS standard_invoice_id,
       ai.INVOICE_NUM AS standard_invoice_number,
       aid.amount AS adjustment_amount,
       aid.accounting_date AS adjustment_accounting_date,
       ppai.invoice_num AS adjustment_reference_number,
       CAST(NULL AS VARCHAR2(30)) AS payment_method,
       'PREPAYMENT' AS settlement_type
  FROM ap_invoice_distributions_all aid
       JOIN ap_invoices_all ai ON aid.invoice_id = ai.invoice_id
       JOIN ap_invoice_distributions_all ppd ON aid.prepay_distribution_id = ppd.invoice_distribution_id
       JOIN ap_invoices_all ppai ON ppd.invoice_id = ppai.invoice_id
 WHERE aid.line_type_lookup_code = 'PREPAY' AND NVL(aid.reversal_flag, 'N') <> 'Y' AND ai.cancelled_date IS NULL AND ai.invoice_num = 'abc123'
ORDER BY payment_accounting_date
Question 4: getting some selected invoice details provided po number is given SAY PO NUMBER AS 4682. sql plan to connect ap_invoice_headers_all with po_headers_all table or joining purchase order tables with invoice tables
Tables: AP_INVOICES_ALL, AP_INVOICE_DISTRIBUTIONS_ALL, AP_SUPPLIERS, PO_DISTRIBUTIONS_ALL, PO_HEADERS_ALL, PO_LINES_ALL
SQL:
SELECT
    ai.invoice_num AS invoice_number,
    ai.invoice_date AS invoice_date,
    aps.vendor_name,
    pol.item_description AS item_details
FROM
    ap_invoices_all ai
JOIN
    ap_invoice_distributions_all aid ON ai.invoice_id = aid.invoice_id
JOIN
    ap_suppliers aps ON ai.vendor_id = aps.vendor_id
JOIN
    po_distributions_all pod ON aid.po_distribution_id = pod.po_distribution_id
JOIN
    po_lines_all pol ON pod.po_line_id = pol.po_line_id
JOIN
    po_headers_all poh ON poh.po_header_id = pol.po_line_id
WHERE
    poh.segment1 = '4682' /*-- segment1 is the PO number */
ORDER BY
    ai.invoice_date
Question 5: Get the list of payments and advance adjustments against a specific invoice abc123
Tables: AP_CHECKS_ALL, AP_INVOICES_ALL, AP_INVOICE_DISTRIBUTIONS_ALL, AP_INVOICE_PAYMENTS_ALL
SQL:
SELECT ai.invoice_id AS standard_invoice_id,
       ai.invoice_num,
       aip.amount AS settlement_amount,
       aip.accounting_date,
       TO_CHAR (ap.check_number) AS reference_number,
       ap.payment_method_lookup_code,
       'PAYMENT' AS settlement_type
  FROM ap_invoices_all ai
       JOIN ap_invoice_payments_all aip
          ON ai.invoice_id = aip.invoice_id
       JOIN ap_checks_all ap
          ON aip.check_id = ap.check_id
 WHERE ai.invoice_num = 'abc123' AND ai.cancelled_date IS NULL
UNION ALL
SELECT ai.invoice_id AS standard_invoice_id,
       ai.invoice_num,
       aid.amount AS settlement_amount,
       aid.accounting_date,
       ppai.invoice_num AS reference_number,
       CAST (NULL AS VARCHAR2 (30)) AS payment_method,
       'PREPAYMENT' AS settlement_type
  FROM ap_invoice_distributions_all aid
       JOIN ap_invoices_all ai
          ON aid.invoice_id = ai.invoice_id
       JOIN ap_invoice_distributions_all ppd
          ON aid.prepay_distribution_id = ppd.invoice_distribution_id
       JOIN ap_invoices_all ppai
          ON ppd.invoice_id = ppai.invoice_id
 WHERE     aid.line_type_lookup_code = 'PREPAY'
       AND NVL (aid.reversal_flag, 'N') <> 'Y'
       AND ai.cancelled_date IS NULL
       AND ai.invoice_num = 'abc123'
ORDER BY accounting_date
Question 6: LIST APPROVERS WHO APPROVED AN INVOCE WHEN INVOIE NUMBER IS GIVEN SAY 'Approv'
Tables: AP_INVOICES_ALL, AP_INV_APRVL_HIST_ALL
SQL:
SELECT
AH.APPROVAL_HISTORY_ID, AI.INVOICE_ID,AI.INVOICE_AMOUNT,AI.CREATION_DATE INV_CREATION_DATE,
       AH.RESPONSE, /*---WHEN THE VALUE IN (WFAPPROVED,MANUALLY APPROVED,APPROVED) IT IS APPROVED, ELSE PENDING APPROVAL */
       AH.ITERATION,
       AH.APPROVER_NAME,
       AH.INVOICE_ID,
       ah.approver_comments,
       AH.CREATION_DATE AS ACTION_DATE /*---DATE WHEN INVOICE IS LASTED ACTED,IF RESPONSE IN      (WFAPPROVED,MANUALLY APPROVED,APPROVED) IT MEANS IT IS APPROVAL DATE */
FROM AP_INV_APRVL_HIST_ALL AH, AP_INVOICES_ALL AI
WHERE 1=1
AND AH.INVOICE_ID = AI.INVOICE_ID
AND AI.INVOICE_ID  IN (SELECT INVOICE_ID FROM AP_INVOICES_ALL WHERE INVOICE_NUM = 'Approv')
ORDER BY AI.INVOICE_ID,AH.APPROVAL_HISTORY_ID, AH.ITERATION
### End of Few Shot examples

[User Question]
Give me the list of invoices where the purchase order payment terms is different from invoice payment terms  Filters/Constraints: operating unit vision operation only  Return ONLY these columns: invoice number, invoice date, puchase order number, purchase order payment terms, invoice payment terms

[SQL]
"""

response = client.completions.create(
    model="/llamaSFT/outputs/merged_oracle_llama70b_awq4",
    prompt=prompt,
    max_tokens=512,
    temperature=0.0,
    top_p=1.0
)

generated_sql = response.choices[0].text.strip()
print("Generated SQL:")
print(generated_sql)


