1. Get Journal details of header and lines
    select LED.NAME ledger_name,jhead.je_header_id,jhead.je_category,jhead.je_source,jhead.period_name,jhead.name,jhead.CURRENCY_CODE,jhead.status,jhead.actual_flag,
    jhead.default_effective_date,jhead.creation_date, jhead.posted_date,jhead.JE_FROM_SLA_FLAG,
    jline.JE_LINE_NUM,jline.CODE_COMBINATION_ID,jline.EFFECTIVE_DATE,jline.ENTERED_DR,jline.ENTERED_CR,jline.ACCOUNTED_DR,jline.ACCOUNTED_CR,
    jline.DESCRIPTION,jline.GL_SL_LINK_ID,jline.GL_SL_LINK_TABLE,
    gl_flexfields_pkg.get_description_sql(led.chart_of_accounts_id,1,gcc.segment1) bu_name,
    gl_flexfields_pkg.get_description_sql( led.chart_of_accounts_id,3,gcc.segment3) acct_desc
    from 
    gl_je_headers jhead,
    gl_je_lines jline,
    gl_ledgers led,
    gl_code_combinations gcc
    where 1=1
    and jhead.je_header_id = jline.je_header_id
    and led.ledger_id = jhead.ledger_id
    and jhead.actual_flag = 'A'
    and jhead.status = 'P' /*--for posted only journal.. only posted journal have accounting impact on trail balance */
    and jline.code_combination_id = gcc.code_combination_id
###
2. get trial balances of multiple ledgers when leger short names are provided as follows:
WITH trial_balance AS (
    SELECT
        LED.ledger_id, -- Using ledger_id for pivot
        LED.short_name AS ledger_short_name,
        TO_CHAR(GP.START_DATE, 'YYYY-MM') AS period,
        GCC.segment3 AS account,
        gl_flexfields_pkg.get_description_sql(LED.chart_of_accounts_id, 3, GCC.segment3) AS account_desc,
        SUM(NVL(BAL.BEGIN_BALANCE_DR, 0) - NVL(BAL.BEGIN_BALANCE_CR, 0) + 
            NVL(BAL.PERIOD_NET_DR, 0) - NVL(BAL.PERIOD_NET_CR, 0)) AS closing_balance      
    FROM
        gl_balances BAL
    JOIN gl_code_combinations GCC ON BAL.code_combination_id = GCC.code_combination_id
    JOIN GL_LEDGERS LED ON BAL.LEDGER_ID = LED.LEDGER_ID
    JOIN GL_PERIODS GP ON BAL.PERIOD_NAME = GP.PERIOD_NAME
                     AND GP.period_set_name = LED.period_set_name
                     AND GP.period_type = LED.accounted_period_type
    WHERE
        LED.chart_of_accounts_id = 101 /*-- Replace with your COA ID*/
        AND LED.period_set_name = 'Accounting' /*-- Replace with your Period Set Name*/
        AND LED.accounted_period_type = 'Month' /*-- Replace with your Period Type*/
        AND GP.PERIOD_NAME = 'Jan-05' /*-- Replace with your required period*/
        AND BAL.ACTUAL_FLAG = 'A'
        AND BAL.TEMPLATE_ID IS NULL
        AND BAL.TRANSLATED_FLAG IS NULL
        AND BAL.CURRENCY_CODE = LED.CURRENCY_CODE
    GROUP BY
        LED.ledger_id,
        LED.short_name,
        TO_CHAR(GP.START_DATE, 'YYYY-MM'),
        GCC.segment3,
        gl_flexfields_pkg.get_description_sql(LED.chart_of_accounts_id, 3, GCC.segment3)
)
SELECT * FROM (
    SELECT 
        account,
        account_desc,
        ledger_short_name,
        closing_balance
    FROM trial_balance
)
PIVOT (
    MAX(closing_balance) FOR ledger_short_name IN (
        'Vision Operations-' AS "Ledger A Closing",
        'Vision BR SL (USD)-' AS "Ledger B Closing",
        'Vision IFRS/IAS Ops-' AS "Ledger C Closing"
    )
)
ORDER BY account

NOTES: IF THE QESTION OR USER QUERY  HAS BUDGET BALANCE  OR BUDGET AS WORD THEN THE WHERE CLAUSE SHOULD HAVE  ACTUAL_FLAG = 'B' OR THE STRING AS ENCUMERBANCE
BALANCE THEN THE ACTUAL_FLAG = 'E'
###    
3.  to get the balance at the end of a given single period period in the ledger currency for any account (segment3) for vision operations ledger
    
    WITH OpeningBalance AS (
    SELECT
        gcc.segment3 AS account,
        SUM(NVL(bal.BEGIN_BALANCE_DR, 0) - NVL(bal.BEGIN_BALANCE_CR, 0)) AS opening_balance
    FROM
        gl_balances bal
    JOIN
        gl_code_combinations gcc ON bal.code_combination_id = gcc.code_combination_id
    JOIN
        GL_LEDGERS led ON bal.LEDGER_ID = led.LEDGER_ID
    JOIN
        GL_PERIODS gp ON bal.PERIOD_NAME = gp.PERIOD_NAME
                     AND gp.period_set_name = led.period_set_name
                     AND gp.period_type = led.accounted_period_type
    WHERE
        led.name  = 'Vision Operations (USA)'
        AND gcc.account_type = 'E'
        AND gp.PERIOD_NAME = 'Mar-08'  /*-- Only take opening balance from first period*/
        AND bal.ACTUAL_FLAG = 'A'
        AND bal.TEMPLATE_ID IS NULL
        AND bal.TRANSLATED_FLAG IS NULL
        AND bal.CURRENCY_CODE = led.CURRENCY_CODE
    GROUP BY
        gcc.segment3
),
PeriodMovements AS (
    SELECT
        gcc.segment3 AS account,
        SUM(NVL(bal.PERIOD_NET_DR, 0)) AS total_debit,
        SUM(NVL(bal.PERIOD_NET_CR, 0)) AS total_credit
    FROM
        gl_balances bal
    JOIN
        gl_code_combinations gcc ON bal.code_combination_id = gcc.code_combination_id
    JOIN
        GL_LEDGERS led ON bal.LEDGER_ID = led.LEDGER_ID
    JOIN
        GL_PERIODS gp ON bal.PERIOD_NAME = gp.PERIOD_NAME
                     AND gp.period_set_name = led.period_set_name
                     AND gp.period_type = led.accounted_period_type
    WHERE
        led.name  = 'Vision Operations (USA)'
        AND gcc.account_type = 'E'
        AND gp.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = 'Mar-08' AND period_set_name = led.period_set_name AND period_type = led.accounted_period_type)
        AND gp.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = 'Mar-08' AND period_set_name = led.period_set_name AND period_type = led.accounted_period_type)
        AND bal.ACTUAL_FLAG = 'A'
        AND bal.TEMPLATE_ID IS NULL
        AND bal.TRANSLATED_FLAG IS NULL
        AND bal.CURRENCY_CODE = led.CURRENCY_CODE
    GROUP BY
        gcc.segment3
)
SELECT
    pm.account,
    gl_flexfields_pkg.get_description_sql(led.chart_of_accounts_id, 3, pm.account) AS account_desc,
    NVL(ob.opening_balance, 0) AS opening_balance,  -- Opening balance from Mar-08 only
    NVL(pm.total_debit, 0) AS debit,  /*-- Total period debits from Mar-08 to Mar-08*/
    NVL(pm.total_credit, 0) AS credit,  /*-- Total period credits from Mar-08 to Mar-08*/
    NVL(ob.opening_balance, 0) + NVL(pm.total_debit, 0) - NVL(pm.total_credit, 0) AS closing_balance  /*-- Correct closing balance calculation*/
FROM
    PeriodMovements pm
LEFT JOIN
    OpeningBalance ob ON pm.account = ob.account
JOIN
    GL_LEDGERS led ON led.name  = 'Vision Operations (USA)'
ORDER BY
    pm.account
###
4.  get net movment of any account in a single period say Mar-08  for vision operations ledgers: 
    WITH OpeningBalance AS (
    SELECT
        gcc.segment3 AS account,
        SUM(NVL(bal.BEGIN_BALANCE_DR, 0) - NVL(bal.BEGIN_BALANCE_CR, 0)) AS opening_balance
    FROM
        gl_balances bal
    JOIN
        gl_code_combinations gcc ON bal.code_combination_id = gcc.code_combination_id
    JOIN
        GL_LEDGERS led ON bal.LEDGER_ID = led.LEDGER_ID
    JOIN
        GL_PERIODS gp ON bal.PERIOD_NAME = gp.PERIOD_NAME
                     AND gp.period_set_name = led.period_set_name
                     AND gp.period_type = led.accounted_period_type
    WHERE
        led.name  = 'Vision Operations (USA)'
        AND gcc.account_type = 'E'
        AND gp.PERIOD_NAME = 'Mar-08'  /*-- Only take opening balance from first period*/
        AND bal.ACTUAL_FLAG = 'A'
        AND bal.TEMPLATE_ID IS NULL
        AND bal.TRANSLATED_FLAG IS NULL
        AND bal.CURRENCY_CODE = led.CURRENCY_CODE
    GROUP BY
        gcc.segment3
),
PeriodMovements AS (
    SELECT
        gcc.segment3 AS account,
        SUM(NVL(bal.PERIOD_NET_DR, 0)) AS total_debit,
        SUM(NVL(bal.PERIOD_NET_CR, 0)) AS total_credit
    FROM
        gl_balances bal
    JOIN
        gl_code_combinations gcc ON bal.code_combination_id = gcc.code_combination_id
    JOIN
        GL_LEDGERS led ON bal.LEDGER_ID = led.LEDGER_ID
    JOIN
        GL_PERIODS gp ON bal.PERIOD_NAME = gp.PERIOD_NAME
                     AND gp.period_set_name = led.period_set_name
                     AND gp.period_type = led.accounted_period_type
    WHERE
        led.name  = 'Vision Operations (USA)'
        AND gcc.account_type = 'E'
        AND gp.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = 'Mar-08' AND period_set_name = led.period_set_name AND period_type = led.accounted_period_type)
        AND gp.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = 'Mar-08' AND period_set_name = led.period_set_name AND period_type = led.accounted_period_type)
        AND bal.ACTUAL_FLAG = 'A'
        AND bal.TEMPLATE_ID IS NULL
        AND bal.TRANSLATED_FLAG IS NULL
        AND bal.CURRENCY_CODE = led.CURRENCY_CODE
    GROUP BY
        gcc.segment3
)
SELECT
    pm.account,
    gl_flexfields_pkg.get_description_sql(led.chart_of_accounts_id, 3, pm.account) AS account_desc,
    NVL(ob.opening_balance, 0) AS opening_balance,  /*-- Opening balance from Mar-08 only*/
    NVL(pm.total_debit, 0) AS debit,  /*-- Total period debits from Mar-08 to Mar-08*/
    NVL(pm.total_credit, 0) AS credit,  /*-- Total period credits from Mar-08 to Mar-08*/
    NVL(ob.opening_balance, 0) + NVL(pm.total_debit, 0) - NVL(pm.total_credit, 0) AS closing_balance  /*-- Correct closing balance calculation*/
FROM
    PeriodMovements pm
LEFT JOIN
    OpeningBalance ob ON pm.account = ob.account
JOIN
    GL_LEDGERS led ON led.name  = 'Vision Operations (USA)'
ORDER BY
    pm.account
###
5.  to get opening balance at the start period and  period movement  for the period range and closing balance at the end of period 
    across range of period say mar 08 to sep 08 we need to use following query
WITH OpeningBalance AS (
    SELECT
        gcc.segment3 AS account,
        SUM(NVL(bal.BEGIN_BALANCE_DR, 0) - NVL(bal.BEGIN_BALANCE_CR, 0)) AS opening_balance
    FROM
        gl_balances bal
    JOIN
        gl_code_combinations gcc ON bal.code_combination_id = gcc.code_combination_id
    JOIN
        GL_LEDGERS led ON bal.LEDGER_ID = led.LEDGER_ID
    JOIN
        GL_PERIODS gp ON bal.PERIOD_NAME = gp.PERIOD_NAME
                     AND gp.period_set_name = led.period_set_name
                     AND gp.period_type = led.accounted_period_type
    WHERE
        led.name  = 'Vision Operations (USA)'
        AND gcc.account_type = 'E'
        AND gp.PERIOD_NAME = 'Mar-08'  /*-- Only take opening balance from first period*/
        AND bal.ACTUAL_FLAG = 'A'
        AND bal.TEMPLATE_ID IS NULL
        AND bal.TRANSLATED_FLAG IS NULL
        AND bal.CURRENCY_CODE = led.CURRENCY_CODE
    GROUP BY
        gcc.segment3
),
PeriodMovements AS (
    SELECT
        gcc.segment3 AS account,
        SUM(NVL(bal.PERIOD_NET_DR, 0)) AS total_debit,
        SUM(NVL(bal.PERIOD_NET_CR, 0)) AS total_credit
    FROM
        gl_balances bal
    JOIN
        gl_code_combinations gcc ON bal.code_combination_id = gcc.code_combination_id
    JOIN
        GL_LEDGERS led ON bal.LEDGER_ID = led.LEDGER_ID
    JOIN
        GL_PERIODS gp ON bal.PERIOD_NAME = gp.PERIOD_NAME
                     AND gp.period_set_name = led.period_set_name
                     AND gp.period_type = led.accounted_period_type
    WHERE
        led.name  = 'Vision Operations (USA)'
        AND gcc.account_type = 'E'
        AND gp.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = 'Mar-08' AND period_set_name = led.period_set_name AND period_type = led.accounted_period_type)
        AND gp.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = 'Sep-08' AND period_set_name = led.period_set_name AND period_type = led.accounted_period_type)
        AND bal.ACTUAL_FLAG = 'A'
        AND bal.TEMPLATE_ID IS NULL
        AND bal.TRANSLATED_FLAG IS NULL
        AND bal.CURRENCY_CODE = led.CURRENCY_CODE
    GROUP BY
        gcc.segment3
)
SELECT
    pm.account,
    gl_flexfields_pkg.get_description_sql(led.chart_of_accounts_id, 3, pm.account) AS account_desc,
    NVL(ob.opening_balance, 0) AS opening_balance,  /*-- Opening balance from Mar-08 only*/
    NVL(pm.total_debit, 0) AS debit,  /*-- Total period debits from Mar-08 to Sep-08*/
    NVL(pm.total_credit, 0) AS credit,  /*-- Total period credits from Mar-08 to Sep-08*/
    NVL(ob.opening_balance, 0) + NVL(pm.total_debit, 0) - NVL(pm.total_credit, 0) AS closing_balance  /*-- Correct closing balance calculation*/
FROM
    PeriodMovements pm
LEFT JOIN
    OpeningBalance ob ON pm.account = ob.account
JOIN
    GL_LEDGERS led ON led.name  = 'Vision Operations (USA)'
ORDER BY
    pm.account;
###
6.can you give me month wise opening balance, movement and  closing balance only between mar 08 and apr 08 for all expense accounts at segment2 level for ledger id 1"
WITH period_movement AS (
    SELECT
        TO_CHAR(GP.START_DATE, 'YYYY-MM') AS period,
        GCC.segment2 AS account_segment2,
        SUM(NVL(BAL.BEGIN_BALANCE_DR, 0) - NVL(BAL.BEGIN_BALANCE_CR, 0)) AS opening_balance,
        SUM(NVL(BAL.PERIOD_NET_DR, 0) - NVL(BAL.PERIOD_NET_CR, 0)) AS movement_in_accounting_period,
        SUM(NVL(BAL.BEGIN_BALANCE_DR, 0) - NVL(BAL.BEGIN_BALANCE_CR, 0) + NVL(BAL.PERIOD_NET_DR, 0) - NVL(BAL.PERIOD_NET_CR, 0)) AS closing_balance
    FROM
        gl_balances BAL
    JOIN
        gl_code_combinations GCC ON BAL.code_combination_id = GCC.code_combination_id
    JOIN
        GL_LEDGERS LED ON BAL.LEDGER_ID = LED.LEDGER_ID
    JOIN
        GL_PERIODS GP ON BAL.PERIOD_NAME = GP.PERIOD_NAME
                     AND GP.period_set_name = LED.period_set_name
                     AND GP.period_type = LED.accounted_period_type
    WHERE
        LED.name = 'Vision Operations (USA)'
        AND GCC.account_type = 'E'
        AND GP.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = 'Mar-08' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
        AND GP.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = 'Apr-08' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
        AND BAL.ACTUAL_FLAG = 'A'
        AND BAL.TEMPLATE_ID IS NULL
        AND BAL.TRANSLATED_FLAG IS NULL
        AND BAL.CURRENCY_CODE = LED.CURRENCY_CODE
    GROUP BY
        TO_CHAR(GP.START_DATE, 'YYYY-MM'),
        GCC.segment2
)
SELECT
    period,
    account_segment2,
    MAX(CASE WHEN period = '2008-03' THEN opening_balance END) AS "Mar-08 Opening",
    MAX(CASE WHEN period = '2008-03' THEN movement_in_accounting_period END) AS "Mar-08 Movement",
    MAX(CASE WHEN period = '2008-03' THEN closing_balance END) AS "Mar-08 Closing",
    MAX(CASE WHEN period = '2008-04' THEN opening_balance END) AS "Apr-08 Opening",
    MAX(CASE WHEN period = '2008-04' THEN movement_in_accounting_period END) AS "Apr-08 Movement",
    MAX(CASE WHEN period = '2008-04' THEN closing_balance END) AS "Apr-08 Closing"
FROM
    period_movement
GROUP BY
    period,account_segment2
ORDER BY
    account_segment2

NOTES: IF THE QESTION OR USER QUERY  HAS BUDGET BALANCE  OR BUDGET AS WORD THEN THE WHERE CLAUSE SHOULD HAVE  ACTUAL_FLAG = 'B' OR THE STRING AS ENCUMERBANCE
BALANCE THEN THE ACTUAL_FLAG = 'E'
###    
7.give me the movement as per TB between Jan-2002 to Mar-2002 for Vision Operations (USA) for actua only 
 select bal.LEDGER_ID,LED.NAME LEDGER_NAME,BAL.CURRENCY_CODE,gcc.segment3,sum(nvl(BAL.PERIOD_NET_DR,0)-nvl(BAL.PERIOD_NET_CR,0)+
    nvl(BAL.BEGIN_BALANCE_DR,0)-nvl(BAL.BEGIN_BALANCE_CR,0)) movement
    from gl_balances bal, gl_code_combinations gcc, GL_LEDGERS LED,GL_PERIODS GP
    where bAL.code_combination_id = gcc.code_combination_id
    AND BAL.currency_code = LED.CURRENCY_CODE
    and bal.ledger_id = led.ledger_id
    AND ACTuAL_FLAG = 'A'
    AND BAL.TEMPLATE_ID IS NULL
    AND BAL.TRANSLATED_FLAG IS NULL
    AND GP.PERIOD_YEAR= BAL.PERIOD_YEAR
    AND GP.PERIOD_NUM= BAL.PERIOD_NUM
    AND GP.PERIOD_SET_NAME = LED.PERIOD_SET_NAME
    AND GP.PERIOD_TYPE = LED.ACCOUNTED_PERIOD_TYPE
    and gp.start_date >= (select start_date from gl_periods where period_name = 'Jan-2002' and PERIOD_SET_NAME = LED.PERIOD_SET_NAME and period_type = led.accounted_period_type)
    and gp.end_date <= (select end_date from gl_periods where period_name = 'Mar-2002' and PERIOD_SET_NAME = LED.PERIOD_SET_NAME and period_type = led.accounted_period_type)
    and LED.NAME = 'Vision Operations (USA)'
    group by bal.LEDGER_ID,LED.NAME,BAL.CURRENCY_CODE,gcc.segment3
###
8.give me month wise period movement actuals for ACCOUNT 7360 for ledger Vision Operations between jan 2008 and mar 2008 along with growth month on month 
WITH period_movement AS (
    SELECT
        TO_CHAR(GP.START_DATE, 'YYYY-MM') AS period,
        SUM(NVL(BAL.PERIOD_NET_DR, 0) - NVL(BAL.PERIOD_NET_CR, 0)) AS movement_in_accounting_period
    FROM
        gl_balances BAL
    JOIN
        gl_code_combinations GCC ON BAL.code_combination_id = GCC.code_combination_id
    JOIN
        GL_LEDGERS LED ON BAL.LEDGER_ID = LED.LEDGER_ID
    JOIN
        GL_PERIODS GP ON BAL.PERIOD_NAME = GP.PERIOD_NAME 
                     AND GP.period_set_name = LED.period_set_name
                     AND GP.period_type = LED.accounted_period_type
    WHERE
        GCC.segment3 = '7360'
        AND LED.NAME = 'Vision Operations (USA)'
        AND GP.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = 'Jan-08' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
        AND GP.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = 'Mar-08' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
        AND BAL.ACTUAL_FLAG = 'A'
        AND BAL.TEMPLATE_ID IS NULL
        AND BAL.TRANSLATED_FLAG IS NULL
        and bal.currency_code = led.currency_code
    GROUP BY
        TO_CHAR(GP.START_DATE, 'YYYY-MM')
)
SELECT
    period,
    movement_in_accounting_period,
    (movement_in_accounting_period - LAG(movement_in_accounting_period) OVER (ORDER BY period)) /
    NULLIF(LAG(movement_in_accounting_period) OVER (ORDER BY period), 0) * 100 AS month_on_month_growth
FROM
    period_movement
ORDER BY
    period;
###
8.1. give me trial balance for vision operations ledger period wise
WITH trial_balance AS (
    SELECT
        TO_CHAR(GP.START_DATE, 'YYYY-MM') AS period,
        GCC.segment3 AS account,
        gl_flexfields_pkg.get_description_sql(LED.chart_of_accounts_id, 3, GCC.segment3) AS account_desc,
        SUM(NVL(BAL.BEGIN_BALANCE_DR, 0) - NVL(BAL.BEGIN_BALANCE_CR, 0)) AS opening_balance,
        SUM(NVL(BAL.PERIOD_NET_DR, 0)) AS debit,
        SUM(NVL(BAL.PERIOD_NET_CR, 0)) AS credit,
        SUM(NVL(BAL.BEGIN_BALANCE_DR, 0) - NVL(BAL.BEGIN_BALANCE_CR, 0) + NVL(BAL.PERIOD_NET_DR, 0) - NVL(BAL.PERIOD_NET_CR, 0)) AS closing_balance      
    FROM
        gl_balances BAL
    JOIN
        gl_code_combinations GCC ON BAL.code_combination_id = GCC.code_combination_id
    JOIN
        GL_LEDGERS LED ON BAL.LEDGER_ID = LED.LEDGER_ID
    JOIN
        GL_PERIODS GP ON BAL.PERIOD_NAME = GP.PERIOD_NAME
                     AND GP.period_set_name = LED.period_set_name
                     AND GP.period_type = LED.accounted_period_type
    WHERE
        LED.NAME = 'Vision Operations (USA)'
        AND GP.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = 'Jan-08' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
        AND GP.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = 'Mar-08' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
        AND BAL.ACTUAL_FLAG = 'A'
        AND BAL.TEMPLATE_ID IS NULL
        AND BAL.TRANSLATED_FLAG IS NULL
        AND BAL.CURRENCY_CODE = LED.CURRENCY_CODE
    GROUP BY
        TO_CHAR(GP.START_DATE, 'YYYY-MM'),
        GCC.segment3,
        gl_flexfields_pkg.get_description_sql(LED.chart_of_accounts_id, 3, GCC.segment3)
)
SELECT 
    account,
    account_desc,
    MAX(CASE WHEN period = '2008-01' THEN opening_balance END) AS "Jan-08 Opening",
    MAX(CASE WHEN period = '2008-02' THEN opening_balance END) AS "Feb-08 Opening",
    MAX(CASE WHEN period = '2008-03' THEN opening_balance END) AS "Mar-08 Opening",
    
    MAX(CASE WHEN period = '2008-01' THEN debit END) AS "Jan-08 Debit",
    MAX(CASE WHEN period = '2008-02' THEN debit END) AS "Feb-08 Debit",
    MAX(CASE WHEN period = '2008-03' THEN debit END) AS "Mar-08 Debit",

    MAX(CASE WHEN period = '2008-01' THEN credit END) AS "Jan-08 Credit",
    MAX(CASE WHEN period = '2008-02' THEN credit END) AS "Feb-08 Credit",
    MAX(CASE WHEN period = '2008-03' THEN credit END) AS "Mar-08 Credit",

    MAX(CASE WHEN period = '2008-01' THEN closing_balance END) AS "Jan-08 Closing",
    MAX(CASE WHEN period = '2008-02' THEN closing_balance END) AS "Feb-08 Closing",
    MAX(CASE WHEN period = '2008-03' THEN closing_balance END) AS "Mar-08 Closing"
FROM trial_balance
GROUP BY 
    account, 
    account_desc
ORDER BY 
    account;
###
9. to get the converted translated trial balance into a different currency for example usd trial balance into ggp following the is query
WITH trial_balance AS ( 
    SELECT
        TO_CHAR(GP.END_DATE, 'YYYY-MM') AS period,
        GCC.SEGMENT3 AS account,
        GL_FLEXFIELDS_PKG.GET_DESCRIPTION_SQL(LED.CHART_OF_ACCOUNTS_ID, 3, GCC.SEGMENT3) AS account_desc,
        SUM(NVL(BAL.BEGIN_BALANCE_DR, 0) - NVL(BAL.BEGIN_BALANCE_CR, 0) + NVL(BAL.PERIOD_NET_DR, 0) - NVL(BAL.PERIOD_NET_CR, 0)) AS closing_balance,
        GP.END_DATE AS period_end_date,
        LED.CURRENCY_CODE AS ledger_currency
    FROM
        GL_BALANCES BAL
    JOIN
        GL_CODE_COMBINATIONS GCC ON BAL.CODE_COMBINATION_ID = GCC.CODE_COMBINATION_ID
    JOIN
        GL_LEDGERS LED ON BAL.LEDGER_ID = LED.LEDGER_ID
    JOIN
        GL_PERIODS GP ON BAL.PERIOD_NAME = GP.PERIOD_NAME
                     AND GP.PERIOD_SET_NAME = LED.PERIOD_SET_NAME
                     AND GP.PERIOD_TYPE = LED.ACCOUNTED_PERIOD_TYPE
    WHERE
        LED.NAME = 'Vision Operations (USA)'
        AND GP.START_DATE >= (SELECT START_DATE FROM GL_PERIODS WHERE PERIOD_NAME = 'Jan-08' AND PERIOD_SET_NAME = LED.PERIOD_SET_NAME AND PERIOD_TYPE = LED.ACCOUNTED_PERIOD_TYPE)
        AND GP.END_DATE <= (SELECT END_DATE FROM GL_PERIODS WHERE PERIOD_NAME = 'Mar-08' AND PERIOD_SET_NAME = LED.PERIOD_SET_NAME AND PERIOD_TYPE = LED.ACCOUNTED_PERIOD_TYPE)
        AND BAL.ACTUAL_FLAG = 'A'
        AND BAL.TEMPLATE_ID IS NULL
        AND BAL.TRANSLATED_FLAG IS NULL
        AND BAL.CURRENCY_CODE = LED.CURRENCY_CODE
    GROUP BY
        TO_CHAR(GP.END_DATE, 'YYYY-MM'),
        GCC.SEGMENT3,
        GL_FLEXFIELDS_PKG.GET_DESCRIPTION_SQL(LED.CHART_OF_ACCOUNTS_ID, 3, GCC.SEGMENT3),
        GP.END_DATE,
        LED.CURRENCY_CODE
)
SELECT 
    TB.account,
    TB.account_desc,   
    -- USD Closing Balances
    MAX(CASE WHEN TB.period = '2008-01' THEN TB.closing_balance END) AS "Jan-08 Closing (USD)",
    MAX(CASE WHEN TB.period = '2008-02' THEN TB.closing_balance END) AS "Feb-08 Closing (USD)",
    MAX(CASE WHEN TB.period = '2008-03' THEN TB.closing_balance END) AS "Mar-08 Closing (USD)",

    -- Converted GBP Balances
    MAX(CASE WHEN TB.period = '2008-01' THEN TB.closing_balance * GR.CONVERSION_RATE END) AS "Jan-08 Closing (GBP)",
    MAX(CASE WHEN TB.period = '2008-02' THEN TB.closing_balance * GR.CONVERSION_RATE END) AS "Feb-08 Closing (GBP)",
    MAX(CASE WHEN TB.period = '2008-03' THEN TB.closing_balance * GR.CONVERSION_RATE END) AS "Mar-08 Closing (GBP)"
FROM trial_balance TB
LEFT JOIN GL_DAILY_RATES GR 
    ON GR.CONVERSION_DATE = TB.period_end_date
    AND GR.CONVERSION_TYPE = 'Corporate'
    AND GR.FROM_CURRENCY = TB.ledger_currency
    AND GR.TO_CURRENCY = 'GBP'
GROUP BY 
    TB.account, 
    TB.account_desc
ORDER BY 
    TB.account;
###
9.1 TO GET THE LIST OF CHILD ACCOUNTS UNDER ONE PARENT ACCOUNT:
SELECT
    FLEX_VALUE AS ACCOUNT_CODE,
    DESCRIPTION AS ACCOUNT_DESCRIPTION,
    PARENT_FLEX_VALUE AS PARENT_ACCOUNT_CODE
FROM
    FND_FLEX_VALUE_CHILDREN_V
WHERE
    FLEX_VALUE_SET_ID = (
        SELECT FIS.FLEX_VALUE_SET_ID
        FROM FND_ID_FLEX_STRUCTURES FFS
        JOIN GL_LEDGERS GLL ON GLL.CHART_OF_ACCOUNTS_ID = FFS.id_flex_num
        JOIN FND_ID_FLEX_SEGMENTS FIS
            ON FIS.ID_FLEX_NUM = FFS.ID_FLEX_NUM
            AND FIS.ID_FLEX_CODE = FFS.ID_FLEX_CODE
            AND FIS.APPLICATION_ID = FFS.APPLICATION_ID
        WHERE
            FFS.ID_FLEX_NUM = 101
            AND FFS.ID_FLEX_CODE = 'GL#'
            AND FFS.APPLICATION_ID = 101
            AND FIS.SEGMENT_NAME = 'Account'
            AND GLL.NAME = 'Vision Operations (USA)'
    )
###
10. TO GET THE LIST OF ACCOUNTS UNDER A PARENT ACCOUNT  WHEN PARENT VALUE AND LEDGER AND MONTH IS SPECIFIED
WITH AccountHierarchy AS (
    SELECT 
        FLEX_VALUE AS ACCOUNT_CODE,
        DESCRIPTION AS ACCOUNT_DESCRIPTION,
        PARENT_FLEX_VALUE AS PARENT_ACCOUNT_CODE
    FROM 
        FND_FLEX_VALUE_CHILDREN_V
    WHERE  
        FLEX_VALUE_SET_ID = (    
            SELECT FIS.FLEX_VALUE_SET_ID
            FROM FND_ID_FLEX_STRUCTURES FFS
            JOIN GL_LEDGERS GLL ON GLL.CHART_OF_ACCOUNTS_ID = FFS.id_flex_num
            JOIN FND_ID_FLEX_SEGMENTS FIS 
                ON FIS.ID_FLEX_NUM = FFS.ID_FLEX_NUM 
                AND FIS.ID_FLEX_CODE = FFS.ID_FLEX_CODE
                AND FIS.APPLICATION_ID = FFS.APPLICATION_ID
            WHERE 1=1
               -- FFS.ID_FLEX_NUM = 101 
                AND FFS.ID_FLEX_CODE = 'GL#' 
                AND FFS.APPLICATION_ID = 101
                AND FIS.SEGMENT_NAME = 'Account'
                AND GLL.NAME = 'Vision Operations (USA)'  /*-- Parameter for Ledger Name*/
        )  
    START WITH PARENT_FLEX_VALUE = 'PT'  /*-- Parameter for Parent Account*/
    CONNECT BY NOCYCLE  
    PRIOR FLEX_VALUE = PARENT_FLEX_VALUE AND PRIOR FLEX_VALUE_SET_ID = FLEX_VALUE_SET_ID
),
Balances AS (
    SELECT
        gcc.segment3 AS account,
        gl_flexfields_pkg.get_description_sql(led.chart_of_accounts_id, 3, gcc.segment3) AS account_desc,
        SUM(NVL(bal.BEGIN_BALANCE_DR, 0) - NVL(bal.BEGIN_BALANCE_CR, 0)) AS opening_balance,
        SUM(NVL(bal.PERIOD_NET_DR, 0)) AS debit,
        SUM(NVL(bal.PERIOD_NET_CR, 0)) AS credit,
        SUM(NVL(bal.BEGIN_BALANCE_DR, 0) - NVL(bal.BEGIN_BALANCE_CR, 0) + NVL(bal.PERIOD_NET_DR, 0) - NVL(bal.PERIOD_NET_CR, 0)) AS closing_balance      
    FROM
        gl_balances bal
    JOIN
        gl_code_combinations gcc ON bal.code_combination_id = gcc.code_combination_id
    JOIN
        GL_LEDGERS led ON bal.LEDGER_ID = led.LEDGER_ID and led.chart_of_accounts_id = gcc.chart_of_accounts_id
    JOIN
        GL_PERIODS gp ON bal.PERIOD_NAME = gp.PERIOD_NAME
                     AND gp.period_set_name = led.period_set_name
                     AND gp.period_type = led.accounted_period_type
                     and gp.period_year = bal.period_year
                     and gp.period_num= bal.period_num
                     and gp.period_type = bal.period_type
    WHERE 1=1
        and LED.NAME = 'Vision Operations (USA)'  /*-- Parameter for Ledger Name*/
        AND gp.PERIOD_NAME = 'Mar-08' -- Parameter for Period Name
        AND bal.ACTUAL_FLAG = 'A'
        AND bal.TEMPLATE_ID IS NULL
        AND bal.TRANSLATED_FLAG IS NULL
        AND bal.CURRENCY_CODE = led.CURRENCY_CODE
           GROUP BY
        gcc.segment3,
        gl_flexfields_pkg.get_description_sql(led.chart_of_accounts_id, 3, gcc.segment3)
        order by gcc.segment3
)
SELECT   -- Ensures unique rows
    b.account,
    b.account_desc,
    NVL(b.opening_balance, 0) AS opening_balance,
    NVL(b.debit, 0) AS debit,
    NVL(b.credit, 0) AS credit,
    NVL(b.closing_balance, 0) AS closing_balance
FROM
    Balances b
JOIN
    AccountHierarchy ah ON b.account = ah.ACCOUNT_CODE
ORDER BY
    b.account
###
11. GET THE LIST OF ACCOUNTS OF A GIVEN TYPE SAY EXPENSE FOR MARCH 08FOLLOWING THE IS QUERY
SELECT
    gcc.segment3 AS account,
    gl_flexfields_pkg.get_description_sql(led.chart_of_accounts_id, 3, gcc.segment3) AS account_desc,
    SUM(NVL(bal.BEGIN_BALANCE_DR, 0) - NVL(bal.BEGIN_BALANCE_CR, 0)) AS opening_balance,
    SUM(NVL(bal.PERIOD_NET_DR, 0)) AS debit,
    SUM(NVL(bal.PERIOD_NET_CR, 0)) AS credit,
    SUM(NVL(bal.BEGIN_BALANCE_DR, 0) - NVL(bal.BEGIN_BALANCE_CR, 0) + NVL(bal.PERIOD_NET_DR, 0) - NVL(bal.PERIOD_NET_CR, 0)) AS closing_balance      
FROM
    gl_balances bal
JOIN
    gl_code_combinations gcc ON bal.code_combination_id = gcc.code_combination_id
JOIN
    GL_LEDGERS led ON bal.LEDGER_ID = led.LEDGER_ID
JOIN
    GL_PERIODS gp ON bal.PERIOD_NAME = gp.PERIOD_NAME
                 AND gp.period_set_name = led.period_set_name
                 AND gp.period_type = led.accounted_period_type
WHERE
    LED.NAME = 'Vision Operations (USA)'
    AND gp.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = 'Mar-08' AND period_set_name = led.period_set_name AND period_type = led.accounted_period_type)
    AND gp.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = 'Mar-08' AND period_set_name = led.period_set_name AND period_type = led.accounted_period_type)
    AND bal.ACTUAL_FLAG = 'A'
    AND bal.TEMPLATE_ID IS NULL
    AND bal.TRANSLATED_FLAG IS NULL
    AND bal.CURRENCY_CODE = led.CURRENCY_CODE
    and gcc.account_type = 'E'
   GROUP BY
    gcc.segment3,
    gl_flexfields_pkg.get_description_sql(led.chart_of_accounts_id, 3, gcc.segment3)
ORDER BY
    account
###
12. get pivoted retsults means rows as columns0> give me month wise period movement actuals for ACCOUNT 7360 for ledger 
    Vision Operations between jan 2008 and mar 2008 along with growth month on month 
WITH period_movement AS (
    SELECT
        TO_CHAR(GP.START_DATE, 'YYYY-MM') AS period,
        SUM(NVL(BAL.PERIOD_NET_DR, 0) - NVL(BAL.PERIOD_NET_CR, 0)) AS movement_in_accounting_period
    FROM
        gl_balances BAL
    JOIN
        gl_code_combinations GCC ON BAL.code_combination_id = GCC.code_combination_id
    JOIN
        GL_LEDGERS LED ON BAL.LEDGER_ID = LED.LEDGER_ID
    JOIN
        GL_PERIODS GP ON BAL.PERIOD_NAME = GP.PERIOD_NAME 
                     AND GP.period_set_name = LED.period_set_name
                     AND GP.period_type = LED.accounted_period_type
    WHERE
        GCC.segment3 = '7360'
        AND LED.NAME = 'Vision Operations (USA)'
        AND GP.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = 'Jan-08' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
        AND GP.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = 'Mar-08' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
        AND BAL.ACTUAL_FLAG = 'A'
        AND BAL.TEMPLATE_ID IS NULL
        AND BAL.TRANSLATED_FLAG IS NULL
        and bal.currency_code = led.currency_code
    GROUP BY
        TO_CHAR(GP.START_DATE, 'YYYY-MM')
),
period_growth AS (
    SELECT
        period,
        movement_in_accounting_period,
        ROUND(
            (movement_in_accounting_period - LAG(movement_in_accounting_period) OVER (ORDER BY period)) /
            NULLIF(LAG(movement_in_accounting_period) OVER (ORDER BY period), 0) * 100,
            2
        ) AS month_on_month_growth
    FROM period_movement
)
SELECT
    'Movement' AS metric,
    MAX(CASE WHEN period = '2008-01' THEN movement_in_accounting_period END) AS "Jan-08",
    MAX(CASE WHEN period = '2008-02' THEN movement_in_accounting_period END) AS "Feb-08",
    MAX(CASE WHEN period = '2008-03' THEN movement_in_accounting_period END) AS "Mar-08"
FROM period_growth
UNION ALL
SELECT
    'Growth (%)' AS metric,
    NULL AS "Jan-08",
    MAX(CASE WHEN period = '2008-02' THEN month_on_month_growth END) AS "Feb-08",
    MAX(CASE WHEN period = '2008-03' THEN month_on_month_growth END) AS "Mar-08"
FROM period_growth;

NOTES: IF THE QESTION OR USER QUERY  HAS BUDGET BALANCE  OR BUDGET AS WORD THEN THE WHERE CLAUSE SHOULD HAVE  ACTUAL_FLAG = 'B' OR THE STRING AS ENCUMERBANCE
BALANCE THEN THE ACTUAL_FLAG = 'E'
###
14. how to get chart of account structure of a given ledger say Vision Operations (USA) as each account will have multiple segments
WITH QualifierData AS (
    SELECT
        fifs.id_flex_num AS chart_of_accounts_id,
        fifs.ID_FLEX_STRUCTURE_CODE AS chart_of_accounts_name,
        SAV.APPLICATION_COLUMN_NAME,
        fifs.id_flex_code,
        fifs.application_id,
        fics.segment_num,
        fics.segment_name,
        fics.flex_value_set_id,
        fvs.flex_value_set_name,
        GLL.LEDGER_ID,
        SAV.SEGMENT_ATTRIBUTE_TYPE,
        COUNT(CASE WHEN SAV.SEGMENT_ATTRIBUTE_TYPE != 'GL_GLOBAL' THEN 1 END) 
        OVER (PARTITION BY fifs.id_flex_num, fics.segment_num) AS NonGlobalCount  /*-- Count of non-GL_GLOBAL qualifiers*/
    FROM
        fnd_id_flex_structures fifs
    JOIN
        fnd_id_flex_segments fics ON fifs.id_flex_num = fics.id_flex_num
    JOIN
        fnd_flex_value_sets fvs ON fics.flex_value_set_id = fvs.flex_value_set_id
    JOIN    
        fnd_segment_attribute_values sav 
        ON fifs.application_id = sav.application_id
        AND SAV.ATTRIBUTE_VALUE = 'Y'
        AND fifs.id_flex_code = sav.id_flex_code
        AND fics.application_column_name = sav.application_column_name  
        AND fics.id_flex_num = sav.id_flex_num 
        AND fics.id_flex_code = sav.id_flex_code
    JOIN 
        gl_ledgers gll ON fifs.id_flex_num = gll.chart_of_accounts_id      
    WHERE
        fifs.id_flex_code = 'GL#'  /*-- Only General Ledger COA*/
        AND fifs.application_id = 101  /*-- Application ID filter*/
        AND gll.name = 'Vision Operations (USA)'
)
SELECT *
FROM QualifierData
WHERE 
    /*-- Keep only GL_GLOBAL if no other qualifiers exist*/
    NOT (SEGMENT_ATTRIBUTE_TYPE = 'GL_GLOBAL' AND NonGlobalCount > 0)
ORDER BY chart_of_accounts_id, segment_num;

NOTES:
    The result is typically as follows

    CHART_OF_ACCOUNTS_ID	APPLICATION_COLUMN_NAME	SEGMENT_NUM	SEGMENT_NAME	FLEX_VALUE_SET_ID	FLEX_VALUE_SET_NAME	LEDGER_ID	SEGMENT_ATTRIBUTE_TYPE
    101	SEGMENT1	1	Company	1002470	Operations Company	1	GL_BALANCING
    101	SEGMENT2	2	Department	1002471	Operations Department	1	FA_COST_CTR
    101	SEGMENT3	3	Account	1002472	Operations Account	1	GL_ACCOUNT
    101	SEGMENT4	4	Sub-Account	1002473	Operations Sub-Account	1	GL_GLOBAL
    101	SEGMENT5	5	Product	1002474	Operations Product	1	GL_GLOBAL

    note the user may want account balance by segments and may refer to them by APPLICATION_COLUMN_NAME such as coompany for segment1,
    department for segment2,product for segment5,sub_account for segment4 etc.,
###
15 can you give me month wise opening balance, movement and  closing balance only between mar 08 and apr 08 for all expense 
    accounts at department segment level for ledger id 1
WITH period_movement AS (
    SELECT
        TO_CHAR(GP.START_DATE, 'YYYY-MM') AS period,
        GCC.segment2 AS department_segment,
        SUM(NVL(BAL.BEGIN_BALANCE_DR, 0) - NVL(BAL.BEGIN_BALANCE_CR, 0)) AS opening_balance,
        SUM(NVL(BAL.PERIOD_NET_DR, 0) - NVL(BAL.PERIOD_NET_CR, 0)) AS movement_in_accounting_period,
        SUM(NVL(BAL.BEGIN_BALANCE_DR, 0) - NVL(BAL.BEGIN_BALANCE_CR, 0) + NVL(BAL.PERIOD_NET_DR, 0) - NVL(BAL.PERIOD_NET_CR, 0)) AS closing_balance
    FROM
        gl_balances BAL
    JOIN
        gl_code_combinations GCC ON BAL.code_combination_id = GCC.code_combination_id
    JOIN
        GL_LEDGERS LED ON BAL.LEDGER_ID = LED.LEDGER_ID
    JOIN
        GL_PERIODS GP ON BAL.PERIOD_NAME = GP.PERIOD_NAME
                     AND GP.period_set_name = LED.period_set_name
                     AND GP.period_type = LED.accounted_period_type
    WHERE
        LED.name = 'Vision Operations (USA)'
        AND GCC.account_type = 'E'
        AND GP.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = 'Mar-08' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
        AND GP.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = 'Apr-08' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
        AND BAL.ACTUAL_FLAG = 'A'
        AND BAL.TEMPLATE_ID IS NULL
        AND BAL.TRANSLATED_FLAG IS NULL
        AND BAL.CURRENCY_CODE = LED.CURRENCY_CODE
    GROUP BY
        TO_CHAR(GP.START_DATE, 'YYYY-MM'),
        GCC.segment2 --- segment1 fro company ,segment2 for department,segment5 for product,segment4 for sub account  etc.,
)
SELECT
    department_segment,
    MAX(CASE WHEN period = '2008-03' THEN opening_balance END) AS "Mar-08 Opening",
    MAX(CASE WHEN period = '2008-03' THEN movement_in_accounting_period END) AS "Mar-08 Movement",
    MAX(CASE WHEN period = '2008-03' THEN closing_balance END) AS "Mar-08 Closing",
    MAX(CASE WHEN period = '2008-04' THEN opening_balance END) AS "Apr-08 Opening",
    MAX(CASE WHEN period = '2008-04' THEN movement_in_accounting_period END) AS "Apr-08 Movement",
    MAX(CASE WHEN period = '2008-04' THEN closing_balance END) AS "Apr-08 Closing"
FROM
    period_movement
GROUP BY
    department_segment
ORDER BY
    department_segment
###
16.  get the qualifier for accounting flex fields like balancing segment cost centre etc., 
WITH QualifierData AS (
    SELECT
        fifs.id_flex_num AS chart_of_accounts_id,
        fifs.ID_FLEX_STRUCTURE_CODE AS chart_of_accounts_name,
        SAV.APPLICATION_COLUMN_NAME,
        fifs.id_flex_code,
        fifs.application_id,
        fics.segment_num,
        fics.segment_name,
        fics.flex_value_set_id,
        fvs.flex_value_set_name,
        GLL.LEDGER_ID,
        SAV.SEGMENT_ATTRIBUTE_TYPE,
        COUNT(CASE WHEN SAV.SEGMENT_ATTRIBUTE_TYPE != 'GL_GLOBAL' THEN 1 END) 
        OVER (PARTITION BY fifs.id_flex_num, fics.segment_num) AS NonGlobalCount  /*-- Count of non-GL_GLOBAL qualifiers */
    FROM
        fnd_id_flex_structures fifs
    JOIN
        fnd_id_flex_segments fics ON fifs.id_flex_num = fics.id_flex_num
    JOIN
        fnd_flex_value_sets fvs ON fics.flex_value_set_id = fvs.flex_value_set_id
    JOIN    
        fnd_segment_attribute_values sav 
        ON fifs.application_id = sav.application_id
        AND SAV.ATTRIBUTE_VALUE = 'Y'
        AND fifs.id_flex_code = sav.id_flex_code
        AND fics.application_column_name = sav.application_column_name  
        AND fics.id_flex_num = sav.id_flex_num 
        AND fics.id_flex_code = sav.id_flex_code
    JOIN 
        gl_ledgers gll ON fifs.id_flex_num = gll.chart_of_accounts_id      
    WHERE
        fifs.id_flex_code = 'GL#'  /*-- Only General Ledger COA*/
        AND fifs.application_id = 101  /*-- Application ID filter*/
        AND gll.name = 'Vision Operations (USA)'
)
SELECT *
FROM QualifierData
WHERE 
    /*-- Keep only GL_GLOBAL if no other qualifiers exist*/
    NOT (SEGMENT_ATTRIBUTE_TYPE = 'GL_GLOBAL' AND NonGlobalCount > 0)
ORDER BY chart_of_accounts_id, segment_num;
###
17.1 Calculate total purchase for item AS54888:
   SELECT SUM(pl.unit_price * NVL(h.rate, 1) * NVL(pd.quantity_ordered, 0)) AS total_purchase
   FROM po_distributions_all pd
   JOIN po_lines_all pl ON pl.po_line_id = pd.po_line_id
   JOIN po_headers_all h ON pl.po_header_id = h.po_header_id   
   left JOIN mtl_system_items_b m ON pd.destination_organization_id = m.organization_id   
        AND pl.item_id = m.inventory_item_id
   WHERE (UPPER(m.segment1) LIKE UPPER('%AS54888%') OR UPPER(m.description) LIKE UPPER('%AS54888%'))
###
17.2 Calculate total purchase for item AS54888 for Vision Operations:
   SELECT SUM(pl.unit_price * NVL(h.rate, 1) * NVL(pd.quantity_ordered, 0)) AS total_purchase
   FROM po_distributions_all pd
   JOIN po_lines_all pl ON pl.po_line_id = pd.po_line_id
   JOIN po_headers_all h ON pl.po_header_id = h.po_header_id
   left JOIN mtl_system_items_b m ON pd.destination_organization_id = m.organization_id   
        AND pl.item_id = m.inventory_item_id
   JOIN hr_operating_units hou on hou.organization_id = h.org_id
   WHERE (UPPER(m.segment1) LIKE UPPER('%AS54888%') OR UPPER(m.description) LIKE UPPER('%AS54888%'))
   and h.NAME = 'Vision Operations';
###  
17.3 Calculate total purchase for item AS54888 for Vision Operations along with quanity in primary unit of measure
  SELECT SUM(pl.unit_price * NVL(h.rate, 1) * NVL(pd.quantity_ordered, 0)) AS total_purchase,primary_uom_code,UNIT_MEAS_LOOKUP_CODE as order_uom_code,
sum(pl.quantity*NVL(um.conversion_rate,1)) as total_QTY,sum(pl.quantity) as total_in_tran_uom
   FROM po_distributions_all pd
   JOIN po_lines_all pl ON pl.po_line_id = pd.po_line_id
   JOIN po_headers_all h ON pl.po_header_id = h.po_header_id
   left JOIN mtl_system_items_b m ON pd.destination_organization_id = m.organization_id   
   AND pl.item_id = m.inventory_item_id
   LEFT JOIN  mtl_uom_conversions um on um.inventory_item_id = m.inventory_item_id     
   JOIN hr_operating_units hou on hou.organization_id = h.org_id
   WHERE  h.NAME = 'Vision Operations'
   and  (UPPER(m.segment1) LIKE UPPER('%AS54888%') OR UPPER(m.description) LIKE UPPER('%AS54888%'))
   group by primary_uom_code,UNIT_MEAS_LOOKUP_CODE
###
17.4  gete the receipt details for item AS54888 for Vision Operations along with po details
      SELECT
    ph.segment1 AS po_number,
    ph.creation_date AS po_creation_date,
    hou.name AS operating_unit,
    rt.transaction_date AS receive_date,
    msi.segment1 AS item_code,
    SUM(pd.quantity_ordered) AS total_quantity_ordered,
    rt.quantity AS received_quantity
FROM
    rcv_transactions rt
JOIN po_distributions_all pd
    ON rt.po_distribution_id = pd.po_distribution_id
JOIN po_lines_all pl
    ON pd.po_line_id = pl.po_line_id
    AND pd.po_header_id = pl.po_header_id
LEFT JOIN mtl_system_items_b msi
    ON pl.item_id = msi.inventory_item_id
    AND pd.destination_organization_id = msi.organization_id
JOIN po_headers_all ph
    ON pl.po_header_id = ph.po_header_id
JOIN hr_operating_units hou
    ON ph.org_id = hou.organization_id
JOIN org_organization_definitions ood
    ON rt.organization_id = ood.organization_id
WHERE
    rt.transaction_type = 'RECEIVE' /* 'DELILVER' IN CASE OF DELIVERY DETAILS 'RETURN TO VENDOR IN CASE RETURN DETAILS*/
GROUP BY
    ph.segment1,
    ph.creation_date,
    hou.name,
    rt.transaction_date,
    msi.segment1,
    rt.quantity
###
   17.5 give me details of purchaes order 4682 the items purchase, qty ,unit rate and item code, warehouse name etc.,
SELECT
    aps.vendor_name,
    ph.segment1 AS po_number,
    ph.creation_date AS po_creation_date,
    hou.name AS operating_unit,
    org.organization_name AS warehouse_name,
    NVL(msi.segment1, pl.item_description) AS item_code,
    SUM(pll.quantity) AS total_ordered_qty,
    pl.unit_price,
    SUM(pll.quantity * pl.unit_price) AS total_line_amount
FROM
    po_headers_all ph
    JOIN po_lines_all pl
        ON ph.po_header_id = pl.po_header_id
    JOIN po_line_locations_all pll
        ON pl.po_line_id = pll.po_line_id
    LEFT JOIN mtl_system_items_b msi
        ON pl.item_id = msi.inventory_item_id
        AND pll.ship_to_organization_id = msi.organization_id
    LEFT JOIN ap_suppliers aps
        ON ph.vendor_id = aps.vendor_id
    JOIN org_organization_definitions org
        ON org.organization_id = pll.ship_to_organization_id
    JOIN hr_operating_units hou
        ON ph.org_id = hou.organization_id
WHERE
    ph.segment1 = '4683'
    AND ph.authorization_status = 'APPROVED'
GROUP BY
    aps.vendor_name,
    ph.segment1,
    ph.creation_date,
    hou.name,
    org.organization_name,
    NVL(msi.segment1, pl.item_description),
    pl.unit_price,
    pl.po_line_id
###
17.6 Calculate total purchases expense account wise in vision operations
SELECT
    hou.name,gl_flexfields_pkg.get_description_sql(led.chart_of_accounts_id, 3, gcc.segment3) AS expense_account_desc,
    gcc.segment3 AS expense_account,
    SUM(pl.unit_price * NVL(h.rate, 1) * NVL(pd.quantity_ordered, 0)) AS total_purchase
FROM
    po_distributions_all pd
JOIN po_lines_all pl ON pl.po_line_id = pd.po_line_id
JOIN po_headers_all h ON pl.po_header_id = h.po_header_id
LEFT JOIN mtl_system_items_b m ON pd.destination_organization_id = m.organization_id
            AND pl.item_id = m.inventory_item_id
JOIN hr_operating_units hou ON hou.organization_id = h.org_id
JOIN gl_ledgers led ON hou.set_of_books_id = led.ledger_id
JOIN gl_code_combinations gcc ON pd.code_combination_id = gcc.code_combination_id
WHERE  h.NAME = 'Vision Operations'
GROUP BY
    gl_flexfields_pkg.get_description_sql(led.chart_of_accounts_id, 3, gcc.segment3),
    gcc.segment3,hou.name
###
17.7 use purchase orders and give sum of expenses, expense account wise, vendor wise for the year 2003 for vision operations
SELECT
    gl_flexfields_pkg.get_description_sql(led.chart_of_accounts_id, 3, gcc.segment3) AS expense_account_desc,
    gcc.segment3 AS expense_account,
    aps.vendor_name,
    SUM(pl.unit_price * NVL(h.rate, 1) * NVL(pd.quantity_ordered, 0)) AS total_expenses
FROM
    po_distributions_all pd
JOIN po_lines_all pl ON pl.po_line_id = pd.po_line_id
JOIN po_headers_all h ON pl.po_header_id = h.po_header_id
JOIN ap_suppliers aps ON h.vendor_id = aps.vendor_id
JOIN gl_code_combinations gcc ON pd.code_combination_id = gcc.code_combination_id
JOIN hr_operating_units ON h.org_id = hr_operating_units.organization_id
JOIN gl_ledgers led ON h.org_id = hr_operating_units.organization_id AND hr_operating_units.set_of_books_id = led.ledger_id
WHERE
    hr_operating_units.name = 'Vision Operations'
    AND EXTRACT(YEAR FROM h.creation_date) = 2003
GROUP BY
    gl_flexfields_pkg.get_description_sql(led.chart_of_accounts_id, 3, gcc.segment3),
    gcc.segment3,
    aps.vendor_name
ORDER BY
    expense_account_desc,
    aps.vendor_name
###
17.8 give me grn details of po number: 4586
SELECT
    h.segment1 AS po_number,
    h.creation_date AS po_creation_date,
    aps.vendor_name,
    pl.line_num AS po_line_number,
    m.segment1 AS item_code,
    m.description AS item_description,
    rt.transaction_date AS receipt_date,
    rt.quantity,
    rsh.receipt_num AS grn_number,
    hou.name AS operating_unit
FROM
    po_headers_all h
JOIN po_lines_all pl
    ON h.po_header_id = pl.po_header_id
JOIN po_distributions_all pd
    ON pl.po_line_id = pd.po_line_id
JOIN rcv_transactions rt
    ON pd.po_distribution_id = rt.po_distribution_id
JOIN rcv_shipment_headers rsh
    ON rt.shipment_header_id = rsh.shipment_header_id
LEFT JOIN mtl_system_items_b m
    ON pl.item_id = m.inventory_item_id
    AND pd.destination_organization_id = m.organization_id
LEFT JOIN ap_suppliers aps
    ON h.vendor_id = aps.vendor_id
JOIN hr_operating_units hou
    ON h.org_id = hou.organization_id
WHERE
    h.segment1 = '4586'
    AND rt.transaction_type = 'RECEIVE'
ORDER BY
    rt.transaction_date
###
17.8.1 Give me the list of requisition numbers, line items and requisition distributions that are not still not converted into purchase orders in vision operations
SELECT
    prh.segment1 AS requisition_number,
    prl.line_num AS line_item,
    prd.distribution_id AS requisition_distribution_id
FROM
    po_requisition_headers_all prh
JOIN
    po_requisition_lines_all prl ON prh.requisition_header_id = prl.requisition_header_id
JOIN
    po_req_distributions_all prd ON prl.requisition_line_id = prd.requisition_line_id
LEFT JOIN
    po_distributions_all pd ON prd.distribution_id = pd.req_distribution_id
JOIN
    hr_operating_units hou ON prh.org_id = hou.organization_id
WHERE
    pd.po_distribution_id IS NULL
    AND h.NAME = 'Vision Operations'
ORDER BY
    prh.segment1, prl.line_num, prd.distribution_id
###
17.9 give me the list of approvers for purhcase order no: 4586 
SELECT
    h.segment1 AS po_number,PAH.OBJECT_REVISION_NUM PO_ITERATION,
    pah.sequence_num AS approval_sequence,
    pah.action_code,
    pah.action_date,
   h.authorization_status current_status,
    pah.note,
    pah.employee_id,
    fu.user_name,
    fu.description AS approver_name
FROM
    po_headers_all h
JOIN po_action_history pah
    ON h.po_header_id = pah.object_id
    AND pah.object_type_code = 'PO' 
    and object_sub_type_code = 'STANDARD'
LEFT JOIN fnd_user fu
    ON pah.employee_id = fu.employee_id
WHERE
    h.segment1 = '4586'
ORDER BY
    pah.object_revision_num,pah.sequence_num
###    
18.1List closed shipments by item and vendor:
   SELECT loc.closed_date, loc.shipment_type, m.description AS item_description, v.vendor_name
   FROM po_line_locations_all loc
   JOIN po_lines_all l ON loc.po_line_id = l.po_line_id
   JOIN po_distributions_all pd on loc.line_location_id = pd.line_location_id
   left JOIN mtl_system_items_b m ON l.item_id = m.inventory_item_id AND pd.destination_organization_id = m.organization_id
   JOIN po_headers_all h ON l.po_header_id = h.po_header_id
   JOIN ap_suppliers v ON h.vendor_id = v.vendor_id
   WHERE 1=1 AND NVL(LOC.CLOSED_FLAG,'N') = 'Y';
###
18.2. give me requisition details for requistion number 1234 including po number if converted
 SELECT
    h.segment1 AS req_number,
    h.creation_date AS req_date,
    l.line_num,
    l.item_description,
    l.unit_meas_lookup_code,
    l.unit_price,
    l.quantity,
    l.need_by_date,
    l.suggested_vendor_name,
    l.destination_type_code,
    l.destination_organization_id,
    l.closed_code,
    l.closed_date,
    l.org_id,
    gcc.segment3 AS natural_account,
    gl_flexfields_pkg.get_description_sql(gl.chart_of_accounts_id, 3, gcc.segment3) AS account_description,
    mi.segment1 AS item_code,
    fu.user_name AS requested_by,
    poh.segment1 AS po_number_if_converted
FROM
    po_requisition_headers_all h
JOIN po_requisition_lines_all l
    ON h.requisition_header_id = l.requisition_header_id and h.segment1 = '1234'
LEFT JOIN po_req_distributions_all d
    ON l.requisition_line_id = d.requisition_line_id
LEFT JOIN mtl_system_items_b mi
    ON l.item_id = mi.inventory_item_id
    AND l.destination_organization_id = mi.organization_id
LEFT JOIN gl_code_combinations gcc
    ON d.code_combination_id = gcc.code_combination_id
LEFT JOIN hr_operating_units hou
    ON h.org_id = hou.organization_id
LEFT JOIN gl_ledgers gl
    ON hou.set_of_books_id = gl.ledger_id
LEFT JOIN fnd_user fu
    ON l.to_person_id = fu.employee_id
LEFT JOIN po_distributions_all pd
    ON pd.req_distribution_id = d.distribution_id
    AND NVL(pd.QUANTITY_CANCELLED, 0) = 0
LEFT JOIN po_headers_all poh
    ON pd.po_header_id = poh.po_header_id
WHERE
    h.type_lookup_code = 'PURCHASE'
    AND h.authorization_status = 'APPROVED'
ORDER BY
    h.segment1, l.line_num
###
18.3 give me total purchase amounts along with quantities for the item as54888 between years 2000 and 2008 in vision operations include received quantities only
SELECT SUM (pl.unit_price * NVL (ph.rate, 1) * pd.quantity_delivered)
          AS total_purchase_amount,
       SUM (pd.quantity_delivered) AS total_received_quantity
  FROM po_distributions_all pd
       JOIN po_lines_all pl
          ON pd.po_line_id = pl.po_line_id
       JOIN po_headers_all ph
          ON pl.po_header_id = ph.po_header_id
       JOIN mtl_system_items_b msi
          ON     pd.destination_organization_id = msi.organization_id
             AND pl.item_id = msi.inventory_item_id
       JOIN hr_operating_units hou
          ON ph.org_id = hou.organization_id
 WHERE     (   UPPER (msi.segment1) LIKE UPPER ('%AS54888%')
            OR UPPER (msi.description) LIKE UPPER ('%AS54888%'))
       AND hou.name = 'Vision Operations'
       AND ph.creation_date BETWEEN to_date ('2000-01-01','YYYY-MM-DD')
                                AND to_date('2008-12-31','YYYY-MM-DD')
###
19.give me top 10 vendors for AS54888, show total purchased value
   SELECT * FROM ( SELECT v.vendor_name, SUM(CASE WHEN l.line_type_id = 1 AND d.destination_type_code = 'INVENTORY' THEN 
   l.unit_price * nvl(h.rate, 1) * d.quantity_ordered ELSE nvl(h.rate, 1) * d.amount_ordered END) AS total_purchased_value 
   FROM po_headers_all h JOIN po_lines_all l ON h.po_header_id = l.po_header_id JOIN po_distributions_all d ON l.po_line_id = d.po_line_id 
   JOIN ap_suppliers v ON h.vendor_id = v.vendor_id left JOIN mtl_system_items_b msi ON l.item_id = msi.inventory_item_id 
   AND d.destination_organization_id = msi.organization_id WHERE UPPER(msi.segment1) LIKE UPPER('%AS54888%') OR UPPER(msi.description) LIKE UPPER('%AS54888%')
   GROUP BY v.vendor_name ORDER BY total_purchased_value DESC  ) WHERE ROWNUM <= 10   
###
20. question: can you give list of top 10 vendors from whom we purchased most in mar-07 along with purchase value 
SELECT *
  FROM (  SELECT aps.vendor_name,
                 SUM (pda.quantity_ordered * pla.unit_price * nvl(pha.rate,1))
                    AS total_purchase_value
            FROM po_headers_all pha
                 JOIN po_lines_all pla
                    ON pha.po_header_id = pla.po_header_id
                 JOIN po_distributions_all pda
                    ON pla.po_line_id = pda.po_line_id
                 JOIN ap_suppliers aps
                    ON pha.vendor_id = aps.vendor_id
                    join hr_operating_units hou on pha.org_id = hou.organization_id
                 JOIN gl_periods gp
                    ON pha.creation_date BETWEEN gp.start_date AND gp.end_date
           WHERE     pha.approved_flag = 'Y'
                 AND gp.period_name = 'Mar-07'
                 AND gp.period_set_name = (SELECT period_set_name
                                             FROM gl_ledgers
                                            WHERE ledger_id = hou.set_of_books_id)
                 AND gp.period_type = (SELECT accounted_period_type
                                         FROM gl_ledgers
                                        WHERE ledger_id = hou.set_of_books_id)
        GROUP BY aps.vendor_name
        ORDER BY total_purchase_value DESC)
 WHERE ROWNUM <= 10
###
21. recurring puchases of same item from same vendor more than 2 times in a month in 2007
SELECT
    pol.item_id,
    pol.item_description,
    poh.vendor_id,
    aps.vendor_name,
    count(*)
        FROM
    po_headers_all poh
JOIN
    po_lines_all pol ON poh.po_header_id = pol.po_header_id
JOIN
    ap_suppliers aps ON poh.vendor_id = aps.vendor_id
WHERE 1=1
     and poh.approved_flag = 'Y'
    AND poh.creation_date BETWEEN DATE '2007-01-01' AND DATE '2007-12-31'
GROUP BY
    pol.item_id, pol.item_description, poh.vendor_id, aps.vendor_name, TO_CHAR(poh.creation_date, 'YYYY-MM')        
HAVING
    COUNT(item_id) > 2
    order by item_id,vendor_id
###
22. dealing with Puchase order tables and getting ledger details from them
select ledger_id from gl_ledgers gll,hr_operating_units hu,po_headers_all poh
   where poh.org_id = hu.organization_id
   and hu.set_of_books_id = gll.ledger_id
###   
23. dealing to get start date and endate of periods connected to a ledgers 
select period_name,period_year,start_date,end_date,period_num,quarter_num,adjustment_period_flag,year_start_date,quarter_start_date
from
gl_periods glp,gl_ledgers gll
where glp.period_set_name = gll.PERIOD_SET_NAME
and glp.period_type= gll.accounted_period_type
###
24. getting subledger data having org_id for selected periods by connecting to gl_ledgers
SELECT
    pha.segment1 AS po_number,
    pha.creation_date AS po_creation_date,
    pha.vendor_id,
    pha.currency_code,
    pha.rate,
    pla.item_id,
    pla.item_description,
    pla.quantity,
    pla.unit_price,
    pda.quantity_ordered,
    pda.quantity_delivered,
    pda.quantity_billed,
    rt.transaction_id,
    rt.transaction_type,
    rt.transaction_date,
    rt.shipment_header_id,
    rt.shipment_line_id
FROM
    po_headers_all pha
JOIN
    po_lines_all pla ON pha.po_header_id = pla.po_header_id
JOIN
    po_distributions_all pda ON pla.po_line_id = pda.po_line_id
JOIN
    rcv_transactions rt ON rt.po_distribution_id = pda.po_distribution_id
JOIN 
    hr_operating_units hu ON hu.organization_id = pha.org_id
JOIN 
    gl_ledgers gll ON gll.ledger_id = hu.set_of_books_id 
JOIN 
    gl_periods glp ON glp.period_set_name = gll.period_set_name 
                   AND glp.period_type = gll.accounted_period_type
                   AND glp.period_name = 'Jan-05'  /* Filters GL_PERIODS once instead of using subqueries*/
WHERE
    rt.transaction_date BETWEEN glp.start_date AND glp.end_date
ORDER BY
    pha.segment1,
    pla.line_num,
    pda.distribution_num,
    rt.transaction_date;
###
25. to get the purchase order data between two periods 
SELECT
    pha.segment1 AS po_number,
    pha.creation_date AS po_creation_date,
    pha.vendor_id,
    pha.currency_code,
    pha.rate,
    pla.item_id,
    pla.item_description,
    pla.quantity,
    pla.unit_price,
    pda.quantity_ordered,
    pda.quantity_delivered,
    pda.quantity_billed,
    rt.transaction_id,
    rt.transaction_type,
    rt.transaction_date,
    rt.shipment_header_id,
    rt.shipment_line_id
FROM
    po_headers_all pha
JOIN
    po_lines_all pla ON pha.po_header_id = pla.po_header_id
JOIN
    po_distributions_all pda ON pla.po_line_id = pda.po_line_id
JOIN
    rcv_transactions rt ON rt.po_distribution_id = pda.po_distribution_id
JOIN 
    hr_operating_units hu ON hu.organization_id = pha.org_id
JOIN 
    gl_ledgers gll ON gll.ledger_id = hu.set_of_books_id
JOIN 
    gl_periods glp_start ON glp_start.period_set_name = gll.period_set_name 
                         AND glp_start.period_type = gll.accounted_period_type
                         AND glp_start.period_name = 'Jan-05'  /* Start Period*/
JOIN 
    gl_periods glp_end ON glp_end.period_set_name = gll.period_set_name 
                       AND glp_end.period_type = gll.accounted_period_type
                       AND glp_end.period_name = 'Mar-05'  /* End Period*/
WHERE
    rt.transaction_date BETWEEN glp_start.start_date AND glp_end.end_date
ORDER BY
    pha.segment1,
    pla.line_num,
    pda.distribution_num,
    rt.transaction_date;
###
26. query to get purchase beyond promised or NEED by date or never delivered
WITH receive_info AS (
    SELECT
        rt.po_distribution_id,
        MIN(rt.transaction_date) AS first_receive_date,
        SUM(rt.quantity) AS total_received_qty
    FROM
        rcv_transactions rt
    WHERE
        rt.transaction_type = 'RECEIVE'
    GROUP BY
        rt.po_distribution_id
)
SELECT
    pha.segment1 AS po_number,
    pha.creation_date AS po_creation_date,
    pha.vendor_id,
    pha.currency_code,
    pha.rate,
    pla.item_id,
    CASE 
        WHEN pla.item_id IS NULL THEN pla.item_description
        ELSE msi.segment1
    END AS item_identifier,
    pla.quantity AS po_line_qty,
    pla.unit_price,
    pda.quantity_ordered,
    pda.quantity_delivered,
    pda.quantity_billed,
    rcv.first_receive_date,
    rcv.total_received_qty,
    pll.promised_date,
    pll.need_by_date,
    CASE
        WHEN rcv.first_receive_date IS NULL THEN 'NOT_RECEIVED'
        WHEN rcv.first_receive_date > pll.promised_date THEN 'LATE_AGAINST_PROMISED'
        WHEN rcv.first_receive_date > pll.need_by_date THEN 'LATE_AGAINST_NEED_BY'
        ELSE 'RECEIVED_ON_TIME'
    END AS delivery_status,
    (NVL(rcv.first_receive_date, SYSDATE) - pll.promised_date) AS days_late_vs_promised,
    (NVL(rcv.first_receive_date, SYSDATE) - pll.need_by_date) AS days_late_vs_need_by
FROM
    po_headers_all pha
JOIN po_lines_all pla
    ON pha.po_header_id = pla.po_header_id and pha.authorization_status = 'APPROVED'
JOIN po_distributions_all pda
    ON pla.po_line_id = pda.po_line_id
JOIN po_line_locations_all pll
    ON pla.po_line_id = pll.po_line_id
    AND pll.line_location_id = pda.line_location_id
LEFT JOIN receive_info rcv
    ON rcv.po_distribution_id = pda.po_distribution_id
LEFT JOIN (select inventory_item_id,segment1,description ,organization_id from  mtl_system_items_b) msi
    ON pla.item_id = msi.inventory_item_id
    AND msi.organization_id = pll.ship_to_organization_id
WHERE
    -- Show only late or undelivered lines
    rcv.first_receive_date IS NULL
    OR rcv.first_receive_date > pll.promised_date
    OR rcv.first_receive_date > pll.need_by_date
ORDER BY
    pha.segment1,
    pla.line_num,
    pda.distribution_num
###
27. give me list recepits that are received but not delivered or deliverd only after certain period say 60 sdyas
items received but not tansferred to inventory by more than 2 months 
WITH received_txns AS (
    SELECT
        rt.transaction_id,
        rt.po_distribution_id,
        rt.shipment_line_id,
        rt.quantity AS received_qty,
        rt.transaction_date AS receive_date
    FROM
        rcv_transactions rt
    WHERE
        rt.transaction_type = 'RECEIVE'
),
delivered_txns AS (
    SELECT
        rt.parent_transaction_id,
        rt.quantity AS delivered_qty,
        rt.transaction_date AS deliver_date
    FROM
        rcv_transactions rt
    WHERE
        rt.transaction_type = 'DELIVER'
),
receive_with_delivery_status AS (
    SELECT
        r.transaction_id,
        r.po_distribution_id,
        r.shipment_line_id,
        r.received_qty,
        r.receive_date,
        d.delivered_qty,
        d.deliver_date,
        CASE
            WHEN d.parent_transaction_id IS NULL THEN 'NOT_DELIVERED'
            WHEN (d.deliver_date - r.receive_date) > 60 THEN 'DELIVERED_LATE'
            ELSE 'DELIVERED_ON_TIME'
        END AS delivery_status,
       ROUND( (NVL(d.deliver_date, SYSDATE) - r.receive_date)) AS days_to_delivery
    FROM
        received_txns r
    LEFT JOIN delivered_txns d
        ON r.transaction_id = d.parent_transaction_id
    WHERE
        d.parent_transaction_id IS NULL
        OR (d.deliver_date - r.receive_date) > 60
)
SELECT
    pha.segment1 AS po_number,
    pha.creation_date AS po_date,
    pla.line_num AS po_line_number,
    msi.segment1 AS item_code,
    msi.description AS item_description,
    pda.distribution_num,
    rwds.received_qty,
    rwds.delivered_qty,
    rwds.receive_date,
    rwds.deliver_date,
    rwds.delivery_status,
    rwds.days_to_delivery
FROM
    receive_with_delivery_status rwds
JOIN po_distributions_all pda
    ON rwds.po_distribution_id = pda.po_distribution_id
JOIN po_lines_all pla
    ON pda.po_line_id = pla.po_line_id
JOIN po_headers_all pha
    ON pla.po_header_id = pha.po_header_id
LEFT JOIN mtl_system_items_b msi
    ON pla.item_id = msi.inventory_item_id
    AND pda.destination_organization_id = msi.organization_id
ORDER BY
    pha.segment1,
    pla.line_num,
    rwds.receive_date
###
28. item lying in receiving location  by return to receivng without return to supplier for long time over 2 months
WITH return_to_receiving AS (
    SELECT
        rt.transaction_id,
        rt.po_distribution_id,
        rt.shipment_line_id,
        rt.transaction_date AS rtr_date,
        rt.quantity AS rtr_qty
    FROM
        rcv_transactions rt
    WHERE
        rt.transaction_type = 'RETURN TO RECEIVING'
),
return_to_vendor AS (
    SELECT
        rt.po_distribution_id,
        rt.shipment_line_id,
        rt.transaction_date AS rtv_date,
        rt.quantity AS rtv_qty
    FROM
        rcv_transactions rt
    WHERE
        rt.transaction_type = 'RETURN TO VENDOR'
),
rtr_rtv_combined AS (
    SELECT
        rtr.po_distribution_id,
        rtr.shipment_line_id,
        rtr.rtr_qty,
        rtr.rtr_date,
        rtv.rtv_qty,
        rtv.rtv_date,
        (NVL(rtv.rtv_date, SYSDATE) - rtr.rtr_date) AS days_delay,
        CASE
            WHEN rtv.rtv_date IS NULL THEN 'NOT_RETURNED'
            WHEN (rtv.rtv_date - rtr.rtr_date) > 15 THEN 'RETURNED_LATE'
            ELSE 'RETURNED_ON_TIME'
        END AS return_status
    FROM
        return_to_receiving rtr
    LEFT JOIN return_to_vendor rtv
        ON rtr.po_distribution_id = rtv.po_distribution_id
        AND rtr.shipment_line_id = rtv.shipment_line_id
    WHERE
        rtv.rtv_date IS NULL OR (rtv.rtv_date - rtr.rtr_date) > 15
)
SELECT
    pha.segment1 AS po_number,
    pha.creation_date AS po_date,
    pla.line_num AS po_line_number,
    msi.segment1 AS item_code,
    msi.description AS item_description,
    pda.distribution_num,
    rc.rtr_qty AS qty_returned_to_receiving,
    rc.rtv_qty AS qty_returned_to_vendor,
    rc.rtr_date,
    rc.rtv_date,
    rc.days_delay,
    rc.return_status
FROM
    rtr_rtv_combined rc
JOIN po_distributions_all pda
    ON rc.po_distribution_id = pda.po_distribution_id
JOIN po_lines_all pla
    ON pda.po_line_id = pla.po_line_id
JOIN po_headers_all pha
    ON pla.po_header_id = pha.po_header_id
LEFT JOIN mtl_system_items_b msi
    ON pla.item_id = msi.inventory_item_id
    AND msi.organization_id = pda.destination_organization_id
ORDER BY
    pha.segment1,
    pla.line_num,
    rc.rtr_date
###
31. give me list recepits that are returned to staging but not returned to vendor
SELECT
    pha.segment1                        AS po_number,
    pha.creation_date                   AS po_date,
    pla.line_num                        AS po_line_number,
    pla.item_id,
    msi.segment1                        AS item_code,
    msi.description                     AS item_description,
    pda.distribution_num,
    NVL(rt_stagging.received_qty, 0)    AS quantity_received,
    NVL(rt_vendor.delivered_qty, 0)  AS quantity_delivered,
    (NVL(rt_stagging.received_qty, 0) - NVL(rt_vendor.delivered_qty, 0)) AS quantity_pending_delivery
FROM
    po_headers_all pha
    JOIN po_lines_all pla
        ON pha.po_header_id = pla.po_header_id
    JOIN po_distributions_all pda
        ON pla.po_line_id = pda.po_line_id
    LEFT JOIN mtl_system_items_b msi
        ON pla.item_id = msi.inventory_item_id
        AND msi.organization_id = pda.destination_organization_id
    LEFT JOIN (
        SELECT
            po_distribution_id,
            SUM(quantity) AS received_qty
        FROM
            rcv_transactions
        WHERE
            transaction_type = 'RETURN TO RECEIVING'
        GROUP BY
            po_distribution_id
    ) rt_stagging
        ON rt_stagging.po_distribution_id = pda.po_distribution_id
    LEFT JOIN (
        SELECT
            po_distribution_id,
            SUM(quantity) AS delivered_qty
        FROM
            rcv_transactions
        WHERE
            transaction_type = 'RETURN TO VENDOR'
        GROUP BY
            po_distribution_id
    ) rt_vendor 
        ON rt_vendor.po_distribution_id = pda.po_distribution_id
WHERE
    (NVL(rt_stagging.received_qty, 0) - NVL(rt_vendor.delivered_qty, 0)) > 0
ORDER BY
    pha.segment1, pla.line_num;
###    
32. give me list of purchase orders where the net quantity is lying in the staging locatoins
SELECT
    pha.segment1                                                              AS po_number,
    pha.creation_date                                                         AS po_date,
    pla.line_num                                                              AS po_line_number,
    pla.item_id,
    msi.segment1                                                              AS item_code,
    msi.description                                                           AS item_description,
    pda.distribution_num,
    NVL(rt_received.received_qty, 0)                                          AS quantity_received,
    NVL(rt_delivered.delivered_qty, 0)                                        AS quantity_delivered,
    NVL(rt_ret_to_receiving.qty_return_to_receiving, 0)                       AS return_to_receiving_qty,
    NVL(rt_ret_to_vendor.qty_return_to_vendor, 0)                             AS return_to_vendor_qty,
    (
        NVL(rt_received.received_qty, 0)
        - NVL(rt_delivered.delivered_qty, 0)
        + NVL(rt_ret_to_receiving.qty_return_to_receiving, 0)
        - NVL(rt_ret_to_vendor.qty_return_to_vendor, 0)
    )                                                                         AS net_qty_in_staging
FROM
    po_headers_all pha
    JOIN po_lines_all pla
        ON pha.po_header_id = pla.po_header_id
    JOIN po_distributions_all pda
        ON pla.po_line_id = pda.po_line_id
    LEFT JOIN mtl_system_items_b msi
        ON pla.item_id = msi.inventory_item_id
        AND msi.organization_id = pda.destination_organization_id
    LEFT JOIN (
        SELECT
            po_distribution_id,
            SUM(quantity) AS received_qty
        FROM
            rcv_transactions
        WHERE
            transaction_type = 'RECEIVE'
        GROUP BY
            po_distribution_id
    ) rt_received
        ON rt_received.po_distribution_id = pda.po_distribution_id
    LEFT JOIN (
        SELECT
            po_distribution_id,
            SUM(quantity) AS delivered_qty
        FROM
            rcv_transactions
        WHERE
            transaction_type = 'DELIVER'
        GROUP BY
            po_distribution_id
    ) rt_delivered
        ON rt_delivered.po_distribution_id = pda.po_distribution_id
    LEFT JOIN (
        SELECT
            po_distribution_id,
            SUM(quantity) AS qty_return_to_receiving
        FROM
            rcv_transactions
        WHERE
            transaction_type = 'RETURN TO RECEIVING'
        GROUP BY
            po_distribution_id
    ) rt_ret_to_receiving
        ON rt_ret_to_receiving.po_distribution_id = pda.po_distribution_id
    LEFT JOIN (
        SELECT
            po_distribution_id,
            SUM(quantity) AS qty_return_to_vendor
        FROM
            rcv_transactions
        WHERE
            transaction_type = 'RETURN TO VENDOR'
        GROUP BY
            po_distribution_id
    ) rt_ret_to_vendor
        ON rt_ret_to_vendor.po_distribution_id = pda.po_distribution_id
WHERE
    (
        NVL(rt_received.received_qty, 0)
        - NVL(rt_delivered.delivered_qty, 0)
        + NVL(rt_ret_to_receiving.qty_return_to_receiving, 0)
        - NVL(rt_ret_to_vendor.qty_return_to_vendor, 0)
    ) > 0
ORDER BY
    pha.segment1,
    pla.line_num;
###
33. give me list of purchase orders where purchase order quantity is less than net received quantity
SELECT
    pha.segment1                                                                                                                              AS
    po_number,
    pha.creation_date                                                                                                                         AS
    po_date,
    pla.line_num                                                                                                                              AS
    po_line_number,
    pla.item_id,
    msi.segment1                                                                                                                              AS
    item_code,
    msi.description                                                                                                                           AS
    item_description,
    pda.distribution_num,
    nvl(rt_received.received_qty, 0)                                                                                                          AS
    quantity_received,
    nvl(rt_ret_to_vendor.qty_return_to_vendor, 0)                                                                                             AS
    return_to_vendor_qty,
    ( nvl(rt_received.received_qty, 0) - nvl
    (rt_ret_to_vendor.qty_return_to_vendor, 0) ) AS net_qty_receceved
FROM
         po_headers_all pha
    JOIN po_lines_all         pla ON pha.po_header_id = pla.po_header_id
    JOIN po_distributions_all pda ON pla.po_line_id = pda.po_line_id
    LEFT JOIN mtl_system_items_b   msi ON pla.item_id = msi.inventory_item_id
                                        AND msi.organization_id = pda.destination_organization_id
    LEFT JOIN (
        SELECT
            po_distribution_id,
            SUM(quantity) AS received_qty
        FROM
            rcv_transactions
        WHERE
            transaction_type = 'RECEIVE'
        GROUP BY
            po_distribution_id
    )                    rt_received ON rt_received.po_distribution_id = pda.po_distribution_id
    LEFT JOIN (
        SELECT
            po_distribution_id,
            SUM(quantity) AS qty_return_to_vendor
        FROM
            rcv_transactions
        WHERE
            transaction_type = 'RETURN TO VENDOR'
        GROUP BY
            po_distribution_id
    )                    rt_ret_to_vendor ON rt_ret_to_vendor.po_distribution_id = pda.po_distribution_id
WHERE
    ( nvl(rt_received.received_qty, 0)  - nvl
    (rt_ret_to_vendor.qty_return_to_vendor, 0) ) > pla.quantity
ORDER BY
    pha.segment1,
    pla.line_num
###
34. give the list of items purchased in Jan 07 where the price is over the same items previous purchase before jan 07
SELECT 
    pol.item_id,
    pol.item_description,
    poh.po_header_id,
    poh.segment1 AS po_number,
    poh.creation_date AS purchase_date,
    pol.unit_price AS current_price,
    prev_pol.unit_price AS previous_price,
    poh.vendor_id,
    poh.approved_flag
FROM 
    po_headers_all poh
JOIN 
    po_lines_all pol ON poh.po_header_id = pol.po_header_id
JOIN 
    (SELECT 
         pol1.item_id, 
         pol1.unit_price, 
         pol1.po_header_id, 
         poh1.creation_date, 
         ROW_NUMBER() OVER (PARTITION BY pol1.item_id ORDER BY poh1.creation_date DESC) AS rn
     FROM 
         po_headers_all poh1
     JOIN 
         po_lines_all pol1 ON poh1.po_header_id = pol1.po_header_id
     WHERE 
         poh1.approved_flag = 'Y' 
         AND poh1.creation_date < DATE '2007-01-01'
    ) prev_pol 
    ON pol.item_id = prev_pol.item_id AND prev_pol.rn = 1
WHERE 
    poh.approved_flag = 'Y' 
    AND poh.creation_date BETWEEN DATE '2007-01-01' AND DATE '2007-01-31'
    AND pol.unit_price > prev_pol.unit_price
ORDER BY 
    pol.item_id, poh.creation_date;

Notes: unless otherwise specified in question, purchase amount is always in local currency only it has arrived at by unit_price* rate* rate column in po headers
###
34. give the list of items purchased in jan07 to dec 07 that is purchased frequently atleast once or twice a month consider one purchase order as one purchase approved purchaes orders only give months in separate columns
WITH purchase_data AS (
    SELECT
        pol.item_id,
        pol.item_description,
        poh.segment1 AS po_number,
        poh.creation_date AS purchase_date,
        TO_NUMBER(TO_CHAR(poh.creation_date, 'MM')) AS purchase_month, 
        TO_CHAR(poh.creation_date, 'YYYY') AS purchase_year
    FROM
        po_headers_all poh
    JOIN
        po_lines_all pol ON poh.po_header_id = pol.po_header_id
    WHERE
        poh.approved_flag = 'Y'
        AND poh.creation_date BETWEEN DATE '2007-01-01' AND DATE '2007-12-31'
),
monthly_purchase_count AS (
    SELECT
        item_id,
        item_description,
        purchase_month,
        COUNT(DISTINCT po_number) AS purchase_count /*-- Count distinct POs */
    FROM
        purchase_data
    GROUP BY
        item_id,
        item_description,
        purchase_month
),
frequent_items AS (
    SELECT
        item_id,
        item_description,
        purchase_month,
        purchase_count
    FROM
        monthly_purchase_count
    WHERE
        purchase_count >= 1 
)
SELECT
    item_id,
    item_description,
    SUM(CASE WHEN purchase_month = 1 THEN purchase_count ELSE 0 END) AS Jan,
    SUM(CASE WHEN purchase_month = 2 THEN purchase_count ELSE 0 END) AS Feb,
    SUM(CASE WHEN purchase_month = 3 THEN purchase_count ELSE 0 END) AS Mar,
    SUM(CASE WHEN purchase_month = 4 THEN purchase_count ELSE 0 END) AS Apr,
    SUM(CASE WHEN purchase_month = 5 THEN purchase_count ELSE 0 END) AS May,
    SUM(CASE WHEN purchase_month = 6 THEN purchase_count ELSE 0 END) AS Jun,
    SUM(CASE WHEN purchase_month = 7 THEN purchase_count ELSE 0 END) AS Jul,
    SUM(CASE WHEN purchase_month = 8 THEN purchase_count ELSE 0 END) AS Aug,
    SUM(CASE WHEN purchase_month = 9 THEN purchase_count ELSE 0 END) AS Sep,
    SUM(CASE WHEN purchase_month = 10 THEN purchase_count ELSE 0 END) AS Oct,
    SUM(CASE WHEN purchase_month = 11 THEN purchase_count ELSE 0 END) AS Nov,
    SUM(CASE WHEN purchase_month = 12 THEN purchase_count ELSE 0 END) AS Dec
FROM
    frequent_items
GROUP BY
    item_id,
    item_description
ORDER BY
    item_id;
###
35. CAN YOU GIVE ME DETAILS OF RECEIPTS MADE AGSINST PO NUMBER '4682'
  SELECT rt.transaction_id,
         rt.transaction_type,
         rt.transaction_date,
         rt.shipment_header_id,
         rt.shipment_line_id,
         rt.po_distribution_id,
         rt.quantity,
         rt.unit_of_measure,
         NVL (msi.segment1, pol.item_description) AS item_code,
         rt.comments
    FROM rcv_transactions rt
         JOIN po_distributions_all pda
            ON rt.po_distribution_id = pda.po_distribution_id
         JOIN po_lines_all pol
            ON pda.po_line_id = pol.po_line_id
         JOIN po_headers_all poh
            ON pol.po_header_id = poh.po_header_id
         LEFT JOIN mtl_system_items_b msi
            ON     pol.item_id = msi.inventory_item_id
               AND pda.destination_organization_id = msi.organization_id
   WHERE poh.segment1 = '4682'
   AND rt.transaction_type= 'RECEIVE'
ORDER BY rt.transaction_date
###
35.5 GIVE ME CASES WHERE BILLED VALUE IS MORE THAN PURCHASE ORDER VALUE 
WITH Purchase_Order AS (
    SELECT
        poh.po_header_id,
        poh.org_id AS po_operating_unit,
        poh.vendor_id AS po_vendor_id,
        poh.vendor_site_id AS poh_vendor_site_id,
        poh.creation_date,
        poh.segment1 AS po_number,
        poh.currency_code,
        pla.po_line_id,
        pla.item_id,
        pla.UNIT_PRICE,
        pla.QUANTITY,
        pod.po_distribution_id,
        pod.QUANTITY_ORDERED,
        pod.QUANTITY_DELIVERED,
        pod.QUANTITY_BILLED,
        pod.amount_billed,
        pod.DESTINATION_ORGANIZATION_ID
    FROM po_headers_all poh
    JOIN po_lines_all pla ON poh.po_header_id = pla.po_header_id
    JOIN po_line_locations_all pll ON pla.po_line_id = pll.po_line_id
    JOIN po_distributions_all pod ON pll.line_location_id = pod.line_location_id
    WHERE poh.type_lookup_code = 'STANDARD'
      AND poh.org_id = 204
),
PO_Vendors AS (
    SELECT 
        aps.vendor_id,
        apss.vendor_site_id AS ven_vendor_site_id,
        aps.vendor_name
    FROM ap_suppliers aps
    JOIN ap_supplier_sites_all apss ON aps.vendor_id = apss.vendor_id
),
EBS_INV_Items AS (
    SELECT 
        inventory_item_id, 
        organization_id, 
        segment1 AS mtl_item_code
    FROM mtl_system_items
),
Warehouses AS (
    SELECT 
        od.organization_id AS warehouse_id, 
        hou.name AS operating_unit_name, 
        gl.name AS ledger_name
    FROM org_organization_definitions od
    JOIN hr_operating_units hou ON od.operating_unit = hou.organization_id
    JOIN gl_ledgers gl ON gl.ledger_id = od.set_of_books_id
)
SELECT 
    pv.vendor_name,
    po.po_operating_unit,
    po.po_number,
    ei.mtl_item_code,
    po.currency_code,
    SUM(NVL(po.QUANTITY, 1) * po.unit_price) AS amount_ordered,
    SUM(po.amount_billed) AS amount_billed,
    SUM(po.QUANTITY_DELIVERED) AS quantity_delivered,
    SUM(po.QUANTITY_BILLED) AS quantity_billed,
    SUM(po.QUANTITY_ORDERED) AS quantity_ordered
FROM Purchase_Order po
JOIN PO_Vendors pv ON po.poh_vendor_site_id = pv.ven_vendor_site_id
JOIN EBS_INV_Items ei 
     ON po.item_id = ei.inventory_item_id 
    AND po.DESTINATION_ORGANIZATION_ID = ei.organization_id
JOIN Warehouses w ON po.DESTINATION_ORGANIZATION_ID = w.warehouse_id
WHERE po.QUANTITY_DELIVERED > po.QUANTITY_ORDERED
  AND po.creation_date BETWEEN TO_DATE('01-JAN-2000', 'DD-MON-YYYY') AND SYSDATE
GROUP BY 
    pv.vendor_name,
    po.po_operating_unit,
    po.po_number,
    ei.mtl_item_code,
    po.currency_code
###
36.GIVE ME THE PAYMENT DETAILS FOR INVOICE NUMBER CRAC Apr 06 09
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
###    
37. give me list of suppllier invoices that are NOT paid before the due date by more than 60 days
SELECT 
    ai.invoice_id,
    ai.invoice_num,
    ai.invoice_date,
    ass.due_date,
    ass.amount_remaining,
    ai.vendor_id,
    aps.vendor_name,
    ai.creation_date AS invoice_creation_date,
    ai.last_update_date AS invoice_last_update_date,
    ass.payment_num,
   TRUNC(NVL(aip.creation_date, SYSDATE)) AS payment_date,
  ROUND(NVL(aip.creation_date, SYSDATE) - ass.due_date) AS delayed_days
FROM 
    ap_invoices_all ai
JOIN 
    ap_suppliers aps 
    ON ai.vendor_id = aps.vendor_id
JOIN 
    ap_payment_schedules_all ass 
    ON ai.invoice_id = ass.invoice_id
LEFT JOIN 
    ap_invoice_payments_all aip 
    ON ass.invoice_id = aip.invoice_id 
    AND ass.payment_num = aip.payment_num
LEFT JOIN 
    ap_checks_all aca 
    ON aip.check_id = aca.check_id
WHERE 
    ai.cancelled_date IS NULL
    and ai.invoice_amount > 0
        and ROUND(NVL(aip.creation_date, SYSDATE) - ass.due_date) >60
ORDER BY 
ass.payment_num,
    ai.invoice_id,
    aip.payment_num 
###
38. give me list of suppllier invoices that are  paid before the due date
SELECT 
    ai.invoice_id,
    ai.invoice_num,
    ai.invoice_date,
    ass.due_date,
    ass.amount_remaining,
    ai.vendor_id,
    aps.vendor_name,
    ai.creation_date AS invoice_creation_date,
    ai.last_update_date AS invoice_last_update_date,
    ass.payment_num,
   aip.creation_date AS payment_date,
  ass.due_date -  aip.creation_date  AS ADVANCED_days
FROM 
    ap_invoices_all ai
JOIN 
    ap_suppliers aps 
    ON ai.vendor_id = aps.vendor_id
JOIN 
    ap_payment_schedules_all ass 
    ON ai.invoice_id = ass.invoice_id
JOIN 
    ap_invoice_payments_all aip 
    ON ass.invoice_id = aip.invoice_id 
    AND ass.payment_num = aip.payment_num
JOIN 
    ap_checks_all aca 
    ON aip.check_id = aca.check_id
WHERE 
    ai.cancelled_date IS NULL
    and ai.invoice_amount > 0
     AND   ass.due_date -  aip.creation_date >0 
ORDER BY 
ass.payment_num,
    ai.invoice_id,
    aip.payment_num 
###
39. getting some selected invoice details provided po number is given SAY PO NUMBER AS 4682, sql plan to connect ap_invoice_headers_all with po_headers_all table or joining purchase order tables with invoice tables
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
###
40. get the prepayments applied against an invoice
  SELECT STD_AI.INVOICE_ID,
         std_ai.INVOICE_NUM AS STANDARD_INVOICE_NUM,
         std_ai.INVOICE_DATE AS STANDARD_INVOICE_DATE,
         std_ai.INVOICE_AMOUNT AS STANDARD_INVOICE_AMOUNT,
         std_ai.VENDOR_ID AS VENDOR_ID,
         s.VENDOR_NAME AS VENDOR_NAME,
         pre_ai.INVOICE_NUM AS PREPAYMENT_INVOICE_NUM,
         pre_ai.INVOICE_DATE AS PREPAYMENT_INVOICE_DATE,
         pre_ai.INVOICE_AMOUNT AS PREPAYMENT_INVOICE_AMOUNT,
         SUM (aid.AMOUNT) AS TOTAL_ADJUSTED_AMOUNT
    FROM ap_invoice_distributions_all aid
         JOIN ap_invoices_all std_ai
            ON aid.INVOICE_ID = std_ai.INVOICE_ID
         JOIN ap_invoice_distributions_all prepay_dist
            ON aid.PREPAY_DISTRIBUTION_ID = prepay_dist.INVOICE_DISTRIBUTION_ID
         JOIN ap_invoices_all pre_ai
            ON prepay_dist.INVOICE_ID = pre_ai.INVOICE_ID
         JOIN ap_suppliers s
            ON std_ai.VENDOR_ID = s.VENDOR_ID
   WHERE     std_ai.INVOICE_TYPE_LOOKUP_CODE IN ('STANDARD', 'EXPENSE REPORT')
         AND aid.LINE_TYPE_LOOKUP_CODE = 'PREPAY'
         AND pre_ai.INVOICE_TYPE_LOOKUP_CODE = 'PREPAYMENT'
GROUP BY STD_AI.INVOICE_ID,
         std_ai.INVOICE_NUM,
         std_ai.INVOICE_DATE,
         std_ai.INVOICE_AMOUNT,
         std_ai.VENDOR_ID,
         s.VENDOR_NAME,
         pre_ai.INVOICE_NUM,
         pre_ai.INVOICE_DATE,
         pre_ai.INVOICE_AMOUNT
###
41. get the list of oustanding invoices as on 31 dec 2007 for vision operations backdate
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
###       
42. invoie payment status as of today for invoie abc123:
SELECT ai.INVOICE_ID,
       ai.INVOICE_NUM,
       ai.INVOICE_DATE,
       ai.CREATION_DATE,
       ai.INVOICE_AMOUNT,
       ai.INVOICE_CURRENCY_CODE,
       s.VENDOR_NAME,
       ssa.VENDOR_SITE_CODE,
       NVL (payments.TOTAL_ACTUAL_PAYMENT, 0) AS TOTAL_ACTUAL_PAYMENT,
       NVL (prepays.TOTAL_PREPAY_ADJUSTED, 0) AS TOTAL_PREPAY_ADJUSTED,
       NVL (schedules.TOTAL_OUTSTANDING_AMOUNT, 0) AS OUTSTANDING_AMOUNT
  FROM ap_invoices_all ai
       LEFT JOIN ap_suppliers s
          ON ai.VENDOR_ID = s.VENDOR_ID
       LEFT JOIN ap_supplier_sites_all ssa
          ON ai.VENDOR_SITE_ID = ssa.VENDOR_SITE_ID
       LEFT JOIN (  SELECT aip.INVOICE_ID,
                           SUM (aip.AMOUNT) AS TOTAL_ACTUAL_PAYMENT
                      FROM ap_invoice_payments_all aip
                  GROUP BY aip.INVOICE_ID) payments
          ON ai.INVOICE_ID = payments.INVOICE_ID
       LEFT JOIN (  SELECT aid.INVOICE_ID,
                           SUM (aid.AMOUNT) AS TOTAL_PREPAY_ADJUSTED
                      FROM ap_invoice_distributions_all aid
                           JOIN ap_invoice_distributions_all prepay_dist
                              ON aid.PREPAY_DISTRIBUTION_ID =
                                    prepay_dist.INVOICE_DISTRIBUTION_ID
                           JOIN ap_invoices_all pre_ai
                              ON prepay_dist.INVOICE_ID = pre_ai.INVOICE_ID
                     WHERE     aid.LINE_TYPE_LOOKUP_CODE = 'PREPAY'
                           AND pre_ai.INVOICE_TYPE_LOOKUP_CODE = 'PREPAYMENT'
                  GROUP BY aid.INVOICE_ID) prepays
          ON ai.INVOICE_ID = prepays.INVOICE_ID
       LEFT JOIN (  SELECT aps.INVOICE_ID,
                           SUM (aps.AMOUNT_REMAINING)
                              AS TOTAL_OUTSTANDING_AMOUNT
                      FROM ap_payment_schedules_all aps
                  GROUP BY aps.INVOICE_ID) schedules
          ON ai.INVOICE_ID = schedules.INVOICE_ID
 WHERE ai.INVOICE_NUM = 'ABC'
###
43. List overdue invoices by days, including vendor and operating unit  as of today or current or  latest or system date
SELECT
    ai.invoice_num,
    s.vendor_name,
    aps.due_date,
    sysdate - aps.due_date AS days_overdue,
    h.name                 AS operating_unit
FROM
         ap_invoices_all ai
    JOIN ap_payment_schedules_all aps ON ai.invoice_id = aps.invoice_id
    JOIN ap_suppliers             s ON ai.vendor_id = s.vendor_id
    JOIN hr_operating_units       h ON ai.org_id = h.organization_id
WHERE
        aps.due_date < sysdate
    AND aps.amount_remaining > 0
    AND h.name = 'VISION OPERATIONS '
    AND ai.wfapproval_status IN ( 'WFAPPROVED', 'MANUALLY APPROVED', 'NOT REQUIRED' )
    AND ai.cancelled_date IS NULL 
###
43.1 GIVE ME THE OUTSTANDING INVOICES FOR CONSOLIDATED SUPPLIES AS OF TODAY for vision operations
SELECT ai.INVOICE_NUM,
       s.VENDOR_NAME,
       aps.DUE_DATE,
       aps.amount_remaining,
       ai.invoice_currency_code,
       SYSDATE - aps.DUE_DATE AS DAYS_OVERDUE,
       h.NAME AS OPERATING_UNIT
  FROM ap_invoices_all ai
       JOIN ap_payment_schedules_all aps
           ON ai.INVOICE_ID = aps.INVOICE_ID
       JOIN ap_suppliers s
           ON ai.VENDOR_ID = s.VENDOR_ID
       JOIN hr_operating_units h
           ON ai.ORG_ID = h.ORGANIZATION_ID
 WHERE aps.DUE_DATE < SYSDATE
   AND aps.AMOUNT_REMAINING > 0
   AND h.NAME = 'VISION OPERATIONS'
   AND ai.WFAPPROVAL_STATUS IN ('WFAPPROVED', 'MANUALLY APPROVED', 'NOT REQUIRED')
   AND ai.CANCELLED_DATE IS NULL
   AND s.VENDOR_NAME = 'Consolidated Supplies'  
ORDER BY aps.DUE_DATE
###
44. Get list of invoices paid before due date
SELECT ai.INVOICE_ID, ai.INVOICE_NUM, ai.INVOICE_DATE, aps.DUE_DATE, aps.AMOUNT_REMAINING, ai.VENDOR_ID, s.VENDOR_NAME, ai.CREATION_DATE AS INVOICE_CREATION_DATE, ai.LAST_UPDATE_DATE AS INVOICE_LAST_UPDATE_DATE, aps.PAYMENT_NUM, aip.CREATION_DATE AS PAYMENT_DATE, aps.DUE_DATE - aip.CREATION_DATE AS ADVANCED_DAYS FROM ap_invoices_all ai JOIN ap_suppliers s ON ai.VENDOR_ID = s.VENDOR_ID JOIN ap_payment_schedules_all aps ON ai.INVOICE_ID = aps.INVOICE_ID JOIN ap_invoice_payments_all aip ON aps.INVOICE_ID = aip.INVOICE_ID AND aps.PAYMENT_NUM = aip.PAYMENT_NUM JOIN ap_checks_all aca ON aip.CHECK_ID = aca.CHECK_ID WHERE ai.CANCELLED_DATE IS NULL AND ai.INVOICE_AMOUNT > 0 AND aps.DUE_DATE - aip.CREATION_DATE > 0 ORDER BY aps.PAYMENT_NUM, ai.INVOICE_ID, aip.PAYMENT_NUM
###
45.  list of approvers against specific  invoice
SELECT ai.INVOICE_NUM, CASE WHEN ai.WFAPPROVAL_STATUS = 'NOT REQUIRED' THEN 'NOT REQUIRED' WHEN ai.WFAPPROVAL_STATUS = 'REQUIRED' THEN 'NOT SUBMITTED FOR APPROVAL' ELSE aih.APPROVER_NAME END AS APPROVER, CASE WHEN ai.WFAPPROVAL_STATUS = 'NOT REQUIRED' THEN 'NOT REQUIRED' WHEN ai.WFAPPROVAL_STATUS = 'REQUIRED' THEN 'NOT SUBMITTED FOR APPROVAL' ELSE aih.RESPONSE END AS RESPONSE, CASE WHEN ai.WFAPPROVAL_STATUS = 'NOT REQUIRED' THEN 'NOT REQUIRED' WHEN ai.WFAPPROVAL_STATUS = 'REQUIRED' THEN 'NOT SUBMITTED FOR APPROVAL' ELSE aih.APPROVER_COMMENTS END AS APPROVER_COMMENTS, CASE WHEN ai.WFAPPROVAL_STATUS IN ('NOT REQUIRED', 'REQUIRED') THEN NULL ELSE aih.CREATION_DATE END AS ACTION_DATE FROM AP_INVOICES_ALL ai LEFT JOIN AP_INV_APRVL_HIST_ALL aih ON ai.INVOICE_ID = aih.INVOICE_ID AND aih.ITERATION = (SELECT MAX(iteration) FROM AP_INV_APRVL_HIST_ALL WHERE INVOICE_ID = ai.INVOICE_ID) WHERE ai.INVOICE_NUM = 'abc123' ORDER BY aih.APPROVAL_HISTORY_ID
###
46. LIST OF APPROVERS AND APPROVAL ACTIONS 
SELECT AH.APPROVAL_HISTORY_ID, 
       AH.RESPONSE, /*---WHEN THE VALUE IN (WFAPPROVED,MANUALLY APPROVED,APPROVED) IT IS APPROVED, ELSE PENDING APPROVAL */
       AH.ITERATION, 
       AH.APPROVER_NAME, 
       AH.INVOICE_ID, 
       AH.CREATION_DATE AS ACTION_DATE /*---DATE WHEN INVOICE IS LASTED ACTED,IF RESPONSE IN (WFAPPROVED,MANUALLY APPROVED,APPROVED) IT MEANS IT IS APPROVAL DATE*/
FROM AP_INV_APRVL_HIST_ALL AH
WHERE AH.HISTORY_TYPE = 'DOCUMENTAPPROVAL'
  AND AH.APPROVAL_HISTORY_ID = (
      SELECT MAX(AH2.APPROVAL_HISTORY_ID)
      FROM AP_INV_APRVL_HIST_ALL AH2
      WHERE AH2.INVOICE_ID = AH.INVOICE_ID
  );
###
47. LIST APPROVERS WHO APPROVED AN INVOCE WHEN INVOIE NUMBER IS GIVEN SAY 'Approv'
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
###
48. list of invoices not approved even after 10 days from creation date,
WITH q1 AS (
    SELECT AH.APPROVAL_HISTORY_ID, 
           AH.RESPONSE, 
           AH.ITERATION, 
           AH.APPROVER_NAME, 
           AH.INVOICE_ID, 
           AH.CREATION_DATE AS ACTION_DATE, 
           AH.LAST_UPDATE_DATE
    FROM AP_INV_APRVL_HIST_ALL AH
    WHERE AH.HISTORY_TYPE = 'DOCUMENTAPPROVAL'
      AND AH.APPROVAL_HISTORY_ID = (
          SELECT MAX(AH2.APPROVAL_HISTORY_ID)
          FROM AP_INV_APRVL_HIST_ALL AH2
          WHERE AH2.INVOICE_ID = AH.INVOICE_ID
      )
), 
q2 AS (
    SELECT INVOICE_ID, CREATION_DATE 
    FROM AP_INVOICES_ALL
    where wfapproval_status not in ('NOT REQUIRED')
)
SELECT q2.INVOICE_ID, 
       q2.CREATION_DATE, 
       q1.RESPONSE, 
       q1.ACTION_DATE, 
       q1.LAST_UPDATE_DATE AS FINAL_APPROVAL_DATE,
       CASE 
           WHEN q1.RESPONSE IN ('WFAPPROVED', 'MANUALLY APPROVED', 'APPROVED','NOT REQUIRED') 
                AND (q1.ACTION_DATE - q2.CREATION_DATE) > 10 /*-- x days*/
           THEN 'Approved but took more than 10 days'
           WHEN q1.RESPONSE NOT IN ('WFAPPROVED', 'MANUALLY APPROVED', 'APPROVED','NOT REQUIRED') 
                AND (SYSDATE - q2.CREATION_DATE) > 10 /*--x days*/
           THEN 'Pending for more than 10 days'
           ELSE 'Does not meet criteria'
       END AS STATUS
FROM q1 
JOIN q2 ON q1.INVOICE_ID = q2.INVOICE_ID
WHERE (q1.RESPONSE IN ('WFAPPROVED', 'MANUALLY APPROVED', 'APPROVED','NOT REQUIRED') 
        AND (q1.ACTION_DATE - q2.CREATION_DATE) > 10)
   OR (q1.RESPONSE NOT IN ('WFAPPROVED', 'MANUALLY APPROVED', 'APPROVED','NOT REQUIRED') 
        AND (SYSDATE - q2.CREATION_DATE) > 10); 
###     
48. Get the list of payments and advance adjustments against a specific invoice abc123 
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
###    
49. Get Journal details of header and lines
    select LED.NAME ledger_name,jhead.je_header_id,jhead.je_category,jhead.je_source,jhead.period_name,jhead.name,jhead.CURRENCY_CODE,jhead.status,jhead.actual_flag, jhead.default_effective_date,jhead.creation_date, jhead.posted_date,jhead.JE_FROM_SLA_FLAG, jline.JE_LINE_NUM,jline.CODE_COMBINATION_ID,jline.EFFECTIVE_DATE,jline.ENTERED_DR,jline.ENTERED_CR,jline.ACCOUNTED_DR,jline.ACCOUNTED_CR, jline.DESCRIPTION,jline.GL_SL_LINK_ID,jline.GL_SL_LINK_TABLE, gl_flexfields_pkg.get_description_sql(led.chart_of_accounts_id,1,gcc.segment1) bu_name, gl_flexfields_pkg.get_description_sql( led.chart_of_accounts_id,3,gcc.segment3) acct_desc from gl_je_headers jhead, gl_je_lines jline, gl_ledgers led, gl_code_combinations gcc where 1=1 and jhead.je_header_id = jline.je_header_id and led.ledger_id = jhead.ledger_id and jhead.actual_flag = 'A' and jhead.status = 'P'  and jline.code_combination_id = gcc.code_combination_id
###    
50. question: if inference has Validate params say validate params in get the list of oustanding invoices as on 31 dec 2007 for vision operations backdate
    Answer: 
    /*-- identify params  as operating unit and value is vision operations */
    /*-- identify 31 dec 2007 as date*/
select name from hr_operating_units where name = 'vision operations'
###
51. question : if inference has Validate params say validate params in Calculate total purchase for item as54888 for the year 2006
       ANSWER
        /*--identify params  as inventory item and the value is as54888for which data is required
        2006 as year for which data is required*/
        select segment1
        from mtl_system_items_b 
        where (upper(segment1) like UPPER('%as54888%') 
        or upper(description ) like  UPPER('%as54888%') ) and rownum =1
###
