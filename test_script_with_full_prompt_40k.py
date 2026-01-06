
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

db_schema = """
    
    The below tables are related to purchase order related data. PO_HEADERS_ALL has a child table called PO_LINES_ALL
    AND PO_LINES_ALL identified by PO_LINE_ID primary key and has a child table PO_LINE_LOCATIONS_ALL 
    AND PO_LINE_LOCATIONS_ALL identified line_locations_id primary key and has a child table PO_DISTRIBUTIONS_ALL
    PO_HEADERS_ALL - critical columns like segment1 for po_number, currency, vendor_id, rate, po_approval_status, po_creation_date are found here
    PO_LINES_ALL - item_id, description, quantity, unit_rate, unit_of_measure and need_by_date are found here
    PO_LINE_LOCATIONS_ALL - location_id to which the delivery has to be made is found here
    PO_DISTRIBUTIONS_ALL - identified by distributions_id primary key, expense_account where house to which the delivery to be made (destination_organization_id), quantity_ordered, quantity_delivered, quantity_billed


    Necessary joins:
    po_headers_all.po_headers_id = po_lines_all.po_header_id for joining 2 tables - po_headers_all and po_lines_all
    po_headers_all.po_headers_id = po_line_locations_all.po_header_id for joining 2 tables - po_headers_all and po_line_locations_all
    po_headers_all.po_headers_id = PO_DISTRIBUTIONS_ALL.po_header_id for joining 2 tables - po_headers_all and PO_DISTRIBUTIONS_ALL
    po_headers_all.vendor_id = ap_suppliers.vendor_id for joining 2 tables - po_headers_all and ap_suppliers
    
    po_lines_all.po_line_id = po_line_locations_all.po_line_id for joining 2 tables - po_line_locations_all and po_lines_all
    po_lines_all.po_line_id = PO_DISTRIBUTIONS_ALL.po_line_id for joining 2 tables - PO_DISTRIBUTIONS_ALL and po_lines_all
    po_lines_all.item_id = MTL_SYSTEM_ITEMS_B.inventory_item_id for joining 2 tables - MTL_SYSTEM_ITEMS_B and po_lines_all
    po_line_locations_all.line_location_id = PO_DISTRIBUTIONS_ALL.line_location_id for joining 2 tables - PO_DISTRIBUTIONS_ALL and po_line_locations_all
    PO_DISTRIBUTIONS_ALL.destination_organization_id = MTL_SYSTEM_ITEMS_B.organization_id for joining 2 tables - PO_DISTRIBUTIONS_ALL and MTL_SYSTEM_ITEMS_B

    Always use alias names for tables when using in select statement, where condition, order by, joins and groupby clauses.
    Always generate syntactically correct SQL queries based on the schema provided above.
    
    The below tables are related to purchase order reltated data. PO_HEADERS_ALL has a child table called PO_LINES_ALL
    AND PO_LINES_ALL identified by PO_LINE_ID primary key and has a child table PO_LINE_LOCATIONS_ALL 
    AND PO_LINE_LOCATIONS_ALL identified line_locations_id primary key and has a child table PO_DISTRIBUTIONS_ALL
    PO_HEADERS_ALL - critical columns like segment1 for po_number, currency, vendor_id, rate, po_approval_status, po_creation_date are found here
    PO_LINES_ALL - item_id, description, quantity, unit_rate, unit_of_measure and need_by_date are found here
    PO_LINE_LOCATIONS_ALL - location_id to which the delivery has to be made is found here
    PO_DISTRIBUTIONS_ALL - identified by distributions_id primary key, expense_account where house to which the delivery to be made (destination_organization_id), quantity_ordered, quantity_delivered, quantity_billed
    Column names and column purpose details are provided along with ddl statment
    
    Necessary joins
    po_headers_all.po_headers_id = po_lines_all.po_header_id for joining 2 tables - po_headers_all and po_lines_all
    po_headers_all.po_headers_id = po_line_locations_all.po_header_id for joining 2 tables - po_headers_all and po_line_locations_all
    po_headers_all.po_headers_id = PO_DISTRIBUTIONS_ALL.po_header_id for joining 2 tables - po_headers_all and PO_DISTRIBUTIONS_ALL
    po_headers_all.vendor_id = ap_suppliers.vendor_id for joining 2 tables - po_headers_all and ap_suppliers
    
    po_lines_all.po_line_id = po_line_locationss_all.po_line_id for joining 2 tables - po_line_locationss_all and po_lines_all
    po_lines_all.po_line_id = PO_DISTRIBUTIONS_ALL.po_line_id for joining 2 tables - PO_DISTRIBUTIONS_ALL and po_lines_all
    po_lines_all.item_id = MTL_SYSTEM_ITEMS_B.inventory_item_id for joining 2 tables - MTL_SYSTEM_ITEMS_B and po_lines_all
    po_line_locations_all.line_location_id = PO_DISTRIBUTIONS_ALL.line_location_id for joining 2 tables - PO_DISTRIBUTIONS_ALL and po_line_locations_all
    PO_DISTRIBUTIONS_ALL.destination_organization_id = MTL_SYSTEM_ITEMS_B.organization_id for joining 2 tables - PO_DISTRIBUTIONS_ALL and MTL_SYSTEM_ITEMS_B

    it is necessary tto join MTL_SYSTEM_ITEMS_B with both columns namely organization_id and inventory_item_id when using with purchasing table as one item in inventory system has multiple rows, one for each organziation id. 
    Hence the unique is the composite key of  organization_id and inventory_item_id

    always use alias names for tables when using in select statement, where condition, order by, joins and groupby cluases

    it is a must to join relavent table columns of different tables such as item_id to item_id or vendor_id to vendor_id etc. 
    non relavent joins are po_header_id to vendor_id
    The column must exist in the table when using joins or selects or groupby clauses. For example do not use po_headers_all.item_id when item_id column doesnot exist in the po_headers_table. This a common mistake you are doing, please remember
    
    There is a column called ORG_ID in all the purchase order tables such as PO_HEADERS_ALL, PO_LINES_ALL, PO_LINE_LOCATIONS_ALL,PO_DISTRIBUTIONS_ALL
    which stands for the business unit or the operating unit id which is available from HR_OPERATING_UNITS table as organization_id. To get the 
    business unit name, we join po_headers_all.org_id = hr_operating_units.organization_id. The org_id column in purchase tables such as PO_HEADERS_ALL, PO_LINES_ALL, PO_LINE_LOCATIONS_ALL,PO_DISTRIBUTIONS_ALL
    should not be confuesd for whearehouse id. To get the whearehouse we have to join PO_DISTRIBUTIONS_ALL.destination_organization_id = MTL_SYSTEM_ITEMS_B.organization_id
    
    Do not join org_id of purchase order tables, namely, po_headers_all, po_lines_all, po_distributions_all with organization_id of MTL_SYSTEM_ITEMS_B because ord_id purchasing tables stands for business division or business unit. 
    Where as organization_id in MTL_SYSTEM_ITEMS_B stands for warehouse. The organization_id of MTL_SYSTEM_ITEMS_B with distribution_organization_id of po_distributions_all table. 
    Po_lines_all will not have organization_id column (which means warehouse) as one line in po_lines_all can be deliverd to multiple warehouses as given by po_distributions_all.destination_organization_id. 
    This means there is one too many relation between po_lines_all and po_distributions_all
    
    when the question has some item related query, always use following condition from MTL_SYSTEM_ITEMS_B.segment1 like item in the question or 
    MTL_SYSTEM_ITEMS_B.description like item in the question for example UPPER(MTL_SYSTEM_ITEMS_B.description) like UPPER('%AS54888%') or UPPER(MTL_SYSTEM_ITEMS_B.segment1) like UPPER('%AS54888%')
    

    If the user question has purchase amount related query, then always use PO_LINES_ALL.unit_rate * PO_HEADERS_ALL.exchage_rate * PO_DISTRIBUTIONS_ALL.quantity_ordered if po_lines_all.line_type=1 or PO_DISTRIBUTIONS_ALL.destination_type_code = 'INVENTORY' 
    If not, which means po_lines_all.line_type != 1 and PO_DISTRIBUTIONS_ALL.destination_type_code != 'INVENTORY' then get the purchase amount using the following formula PO_HEADERS_ALL.exchage_rate * PO_DISTRIBUTIONS_ALL.AMOUNT_ordered 
    
    If the question has rate word, then take it as unit_rate or unit_price and not rate in the po_headers_all
    
    If po_lines_all.item_id is null then never join with MTL_SYSTEM_ITEMS_B or you may use outer join

    when user enters from period to to period for some data requirements  in purchasing then you must join the gl_period table.
    for example if user asks purchase between say feb 2008 and mar 2008

    gl_periods has followign important columns
    period_set_name -> should be joined with gl_ledgers-> period_set_name for joining with gl_periods table for for any transaction/journals ledger calender period data 
    period_name should be joined with transaction/journal period_name 
    start_date and endate define start date and end date of a period_name 
    period_type should be joined with with gl_ledger period_type for joining with gl_periods table for for any transaction/journals ledger calender period data
    period_year defines accounting year of the calendar
    period_num- do not by this. please use start date and end date always for joins for data between periods and dates
    period_name are case sensitive, examples - Feb-08 is not same as FEB-08. 
    it is ALWAYS necessary to use gl_ledger columns period_set_name,period_type  FOR JOINING WITH GL_PERIODS
    
    Gl_code_combinations table-> how to get account type.. typically user may ask balances by accunt type. account type is available from gl_code_combinations
    following are the values for account type:

    R for revenue-sales or income
    A for assets both current and fixed assets
    O for ownership includes paid up share capital and free reserves
    L FOR liability includes short and long term liabilities inlcudes provisions and captial reserves
        
        when gl_periods table is used it is mandatory to use the below joins
        WHEN ORG ID IS ONLY THERE IN TRANSACTION TABLES THEN USE JOIN TRANSACTIONS TABLE ORG_ID COLUMN WITH HR_OPERATING_UNITS AND THEN GET GL_LDEGER
    FOR EXAMPLE:
    PO_HEAERS_ALL.ORG_ID = HR_OPERATING_UNINTS.ORGANIZATION_ID
    AND HR_OPERATING_UNITS.SET_OF_BOOKS_ID = GL.LEDGER.LEDGER_ID
    AND GL_PERIODS.period_set_name = GL.LEDGER.period_set_name
    and GL_PERIODS.period_type = GL.LEDGER.ACCOUNTED_PERIOD_TYPE
    and gl.period.aADJUSTMENT_PERIOD_FLAG = 'N'
    and  h.CREATION_DATE <= (SELECT end_date FROM gl_periods WHERE UPPER(table.column_name) like 'upper('str%1')
    and  h.CREATION_DATE => (SELECT start_date FROM gl_periods WHERE UPPER(table.column_name) like 'upper('str%1')
    
        
        TABLE po_headers_all 
            this table represents purchase order (PO) header level information. One purchase order can have many lines as given by PO_LINES_ALL
            at the header table level we can get verndor on whom the purchase order is placed given by vendor id column, purchase order creation date, currency in which the purchase order issued given by CURRENCY_CODE column
            and exchange rate for conversion into accounting currency (country currency). The table also has the approval status given by approvded_flag="Y" considered as approved
            Only approved purchase oders will be taken for reporting. If approved_flag !="Y" it is considered as purchase order is in process. In process PO cannot be received
        TABLE PO_LINES_ALL 
            In this purchase order lines table. On purchase order can have many lines. Each line is for one item that is being purchased. The order quantity, unit rate, unit of mesuarement, line amount for a given item in the lines is avalibale from this table. 
            One PO line can have many line locations given by po_line_locations_all table
            Joins with po_headers_all using PO_HEADER_ID, joins with PO_LINE_LOCATIONS_ALL.po_line_id using po_line_id, joins with PO_DISTRIBUTIONS_ALL.po_line_id using po_line_id, joins with MTL_SYSTEM_ITEMS_B.INVENTORY_ITEM_ID using item_id
        TABLE PO_LINE_LOCATIONS_ALL 
            the line locations all table represents location where the delivery has to be made by the suppliare
            One line in po_lines_all delivery accross multiple locations as give by PO_LINES_LOCATIONS_ALL
            one location can have delivery accross multiple wharehouses in that location. The where house data is give by PO_DISTRIBUTIONS_ALL
            Joins with po_headers_all using PO_HEADER_ID, joins with PO_LINES_ALL.po_line_id using po_line_id, joins with PO_DISTRIBUTIONS_ALL.line_location_id using line_location_id 
        TABLE PO_DISTRIBUTIONS_ALL 
            this table is child table to po_line_locations_all table. One line location may have multiple po distributions
            each distribution can carry a whearhourse name give by destination_organization_id 
            in addition the disctrbution will also have CODE_COMBINATION_ID which stands for GL account against which this purchase is debited
            additionally the disctributions table will also have the name of the person for whom the purchases are being made as given by DELIVER_TO_PERSON_ID 
            it is necessary to join DESTINATION_ORGANIZATION_ID with MTL_SYSTEM_ITEMS_B.ORGANIZATION_ID when join involves MTL_SYSTEM_ITEMS_B for getting item names etc
            Joins with po_headers_all using PO_HEADER_ID, joins with PO_LINE_LOCATIONS_ALL.line_location_id using line_location_id, joins with PO_LINES_ALL.po_line_id using po_line_id, joins with MTL_SYSTEM_ITEMS_B.ORGANIZATION_ID using DESTINATION_ORGANIZATION_ID
        TABLE ap_suppliers
            This table is Vendor master table
            This table has 4 important columns namely Vendor_id unique identifier, segment1 as vendor number, vendor_name has name of the vendor, vendor_type_lookup_code as vendor classification
            Such as employees vendors, external vendors, statutory vendors etc
        TABLE MTL_SYSTEM_ITEMS_B 
            This table is item master table
            This table has 2 identifier columns for any joins namely inventory item id and organization id
            This table defines item management properties and item identification properties and item type properties. Segment1 column of the table gives item name
            INVENTORY_ITEM_ID and ORGANIZATION_ID together constitute unique key for this table. it is always necessary to use both columns in joins when using this table. 

            wWhen the user requests data between a 'from period' and a 'to period' in purchasing, you must join the gl_periods table.
        
            it is necessary to always take exchange_rate as nvl(exchange_rate,1 ) or if exchange rate column value is rate as seen in 
        soem tables then it is nvl(rate,1)

        so get value of purchases it po_lines_all.qty*po_lines_all.unit_price*nvl(rate,1)

    always join po_headers_all with hr_operating_units as below. there is no ledger_id in po_headers_all table hence while trying to get leger id please use join with hr_operating_units, such as
        po_headers_all.org_id = hr_operaring_units.organization_id
        and hr_operating_units.set_of_books_id = gl_ledgers.ledger_id and so on...

        GRN - Goods Received Note reated tables
        TABLE RCV_TRANSACTIONS  stores historical information about receiving transactions that you have performed. When you enter a receiving transaction and the receiving transaction processor processes your transaction, the transaction is recorded in this table.
        TABLE MTL_MATERIAL_TRANSACTIONS  stores a record of every material transaction or cost update performed in Inventory.
        TABLE MTL_TRANSACTION_ACCOUNTS  holds the accounting information for each material transaction in MTL_MATERIAL_TRANSACTIONS. Inventory uses this information to track the financial impact of your quantity moves.
        
        Notes on Purchase order receipts
        Purchase order receipts in 2 steps using receive trasaction type, where marteirals are received staged in staging location. 
        In the next step, know as delivery where goods will move from staging location to warehouse. 
        user might ask questions about goods lying in staging locations which means goods are only received but there is no other transaction type which is deliverd.
        The data can be found out by searching rcv_transactions where there is a po_distribution_id aganist a receiving transaction but the doesn't exist at the delivery transaction

        Notes on Purchase order returns to vendor
        Purchase order returns in 2 steps using return to receiving transaction type, where marteirals are returned to staging location. 
        In the next step, know as retunr to verndor where mateirial is picked from stagging location and returned to verndor. Thus the matrieal goes out of the premisis. 
        user might ask questions about goods lying in staging locations which means goods are returned but there is no other transaction type which is returned to vendor.
        The data can be found out by searching rcv_transactions where there is a po_distribution_id aganist a receiving transaction but the doesn't exist at the delivery transaction
        In all there are 4 transaction types that are used in rcv_transactions table in a purchase receiving cycle. 
            RECEIVING -> material that is coming into the organization from outside supplier. Stock in the organization increased. n
            DELIVER -> material thats moved from internal loaction to another internal location. from staging to warehouse. No chantge in the quantity of stock in the organziation except the location
            RETURN TO RECEIVING -> material ment to be returned supplier. internal movement of the goods between warehouse to staging location. No chantge in the quantity of stock in the organziation except the location
            RETURN TO VENDOR -> material return back to supplier. External movement of the matrial from staging location to supplier. Change in the orgnaization stock reduced
            
        When the user might ask net received quantity or net supplied quantity which is basically what we have received from supplier - what we have returend to supplier. No internal movements in the form of transaction types Delivered or RETURN TO RECIVING considered
        ### Important Joins and Considerations:
        - *gl_periods.period_set_name* → Must be joined with *gl_ledgers.period_set_name* to retrieve ledger calendar period data for transactions and journals.
        - *gl_periods.period_name* → Must be joined with the corresponding transaction/journal period_name.
        - *gl_periods.start_date and end_date* → Define the period's start and end dates. *Always use these for filtering transactions between periods*.
        - *gl_periods.period_type* → Must be joined with *gl_ledgers.period_type* for proper calendar alignment.
        - *gl_periods.period_year* → Defines the accounting year but should *not* be used for filtering.
        - *gl_periods.period_num* → *Do not use this for filtering*; always rely on start_date and end_date.
        - *Period names are case-sensitive* → 'Feb-08' is *not* the same as 'FEB-08'.

        ### *Mandatory Joins for gl_periods When ledger_id is Available (e.g., GL tables)*
        If ledger_id is available in the table, you must include the following joins:
        AND gl_periods.period_set_name = gl_ledgers.period_set_name
        AND gl_periods.period_type = gl_ledgers.accounted_period_type
        AND gl_periods.adjustment_period_flag = 'N'
        AND h.creation_date <= (SELECT end_date FROM gl_periods WHERE UPPER(table.column_name) LIKE UPPER('str%1'))
        AND h.creation_date >= (SELECT start_date FROM gl_periods WHERE UPPER(table.column_name) LIKE UPPER('str%1'))

        Mandatory Joins for gl_periods When ledger_id is Not Directly Available (e.g., Purchasing tables)
        For purchasing-related tables like po_headers_all, where ledger_id is not directly available, obtain it via org_id using hr_operating_units:
        AND po_headers_all.org_id = hr_operating_units.organization_id
        AND hr_operating_units.set_of_books_id = gl_ledgers.ledger_id
        AND gl_periods.period_set_name = gl_ledgers.period_set_name
        AND gl_periods.period_type = gl_ledgers.accounted_period_type
        AND gl_periods.adjustment_period_flag = 'N'
        AND h.creation_date <= (SELECT end_date FROM gl_periods WHERE UPPER(table.column_name) LIKE UPPER('str%1'))
        AND h.creation_date >= (SELECT start_date FROM gl_periods WHERE UPPER(table.column_name) LIKE UPPER('str%1'))

        Additional Considerations
        ALWAYS join gl_periods with gl_ledgers.period_set_name and gl_ledgers.period_type.
        NEVER use period_num for date range filtering. Instead, always use start_date and end_date.
        In all cases where date filtering is needed, use subqueries to retrieve start_date and end_date.

    The below tables GL_JE_HEADERS, gl_je_lines, GL_BALANCES, gl_periods  are reated to General Ledger 
        GL_JE_HEADERS together with gl_je_lines capture journal voucher transactional data  in accounts.
        gl_je_headers alias headers shall captures jounal ID(je_header_id), journal ledger ID in which journal is being created, journal name, description of the journal, currency of the journal and defult effective date of journal.
        The gl_je_lines alias lines table connects to gl_je_headers with je_header_id column. There je_line_num constitutes the line_number that is unique with in a journal. hence je_header_id and je_line_num together constitute primary key for journal lines information.the gl_je_lines importantly has following information,   
        je_header_id 
        je_line_num 
        entered_dr amount entered in non-ledger currency (every ledger has currency in which is it maintained, typically local currency of the country)
        entered_cr  amount entered in non-ledger currency (every ledger has currency in which is it maintained, typically local currency of the country)
        accounted_dr amount converted into ledger currency using currency (every ledger has currency in which is it maintained, typically local currency of the country)
        entered_cr amount converted into ledger currency using currency (every ledger has currency in which is it maintained, typically local currency of the country)
        code_combination_id  accounting head under which the line shall be posted. the id has to be essentially joined with gl_code_combinations table with code_combination_id join. typically segment3 is the natural account. you will find natual account code in segement3 column of gl_code_combination_table. For converting code into descritpion which every business user wants , we need to join this with hard coded values different for each client by using following folowwing function:
        apps.gl_flexfields_pkg.get_description_sql( gcc.chart_of_accounts_id,3,gcc.segment3) Segment3_desc,
        GL_JE_HEADERS stores journal entries. There is a one-to-many relationship between journal entry batches and journal entries. Each row in this table includes the associated batch ID, the journal entry name and description, and other information about the journal entry. This table corresponds to the Journals window of the Enter Journals form. STATUS is 'U' for unposted and 'P' for posted. Other statuses indicate that an error condition was found.  only posted journal are effective and increate ledger balances.
        CONVERSION_FLAG equal to 'N' indicates that you manually changed a converted amount in the Journal Entry Lines zone of a foreign currency journal entry. In this case, the posting program does not re-convert your foreign amounts. This can happen only if your user profile option MULTIPLE_RATES_PER_JE is 'Yes'. BALANCING_SEGMENT_VALUE is null if there is only one balancing segment value in your journal entry. If there is more than one, BALANCING_SEGMENT_VALUE is the greatest balancing segment value in your journal entry.

        new

        Purpose:
            - Stores summarized financial account balance information
            - Tracks cumulative balances for general ledger accounts across different periods and currencies

            Key Columns:
            1. LEDGER_ID
            - Identifies the specific ledger/set of books
            - Typically links to GL_LEDGERS table

            2. PERIOD_NAME
            - Represents accounting period which is represented by month and year 
            - Links to GL_PERIODS table

            3. CURRENCY_CODE
            - Stores the currency for the balance
            - Typically links to FND_CURRENCIES table

            4. CODE_COMBINATION_ID
            - Unique identifier for the account combination
            - Links to GL_CODE_COMBINATIONS table

            5. ACTUAL_FLAG
            - Indicates balance type (actual, budget, etc.)

            6. BALANCE_TYPE
            - Represents balance type (period, quarter, year-to-date)

            Balance Amount Columns:
            - PERIOD_NET_DR
            - PERIOD_NET_CR
            - PERIOD_NET_TOTAL
            - PERIOD_BEGINNING_BALANCE
            - YEAR_TO_DATE_BALANCE

            Additional Attributes:
            - Creation Date
            - Last Update Date
            - Last Updated By

            Primary Key: Combination of LEDGER_ID, PERIOD_NAME, CURRENCY_CODE, CODE_COMBINATION_ID

            Typical Usage:
            - Financial reporting
            - Account reconciliation
            - Balance tracking across periods
            
        Table GL_BALANCES stores actual, budget, and encumbrance balances for detail and summary accounts. This table stores ledger currency, foreign currency, and statistical balances for each accounting period that has ever been opened. ACTUAL_FLAG is either 'A', 'B', or 'E' for actual, budget, or encumbrance balances, respectively. If ACTUAL_FLAG is 'B', then BUDGET_VERSION_ID is required. If ACTUAL_FLAG is 'E', then ENCUMBRANCE_TYPE_ID is required. GL_BALANCES stores period activity for an account in the PERIOD_NET_DR and PERIOD_NET_CR columns. The table stores the period beginning balances in BEGIN_BALANCE_DR and BEGIN_BALANCE_CR. An account's year-to-date balance is calculated as BEGIN_BALANCE_DR - BEGIN_BALANCE_CR + PERIOD_NET_DR - PERIOD_NET_CR. Detail and summary foreign currency balances that are the result of posted foreign currency journal entries have TRANSLATED_FLAG set to 'R', to indicate that the row is a candidate for revaluation.
        For foreign currency rows, the begin balance and period net columns contain the foreign currency balance, while the begin balance and period net BEQ columns contain the converted ledger currency balance. Detail foreign currency balances that are the result of foreign currency translation have TRANSLATED_FLAG set to 'Y' or 'N'. 'N' indicates that the translation is out of date (i.e., the account needs to be re-translated). 'Y' indicates that the translation is current. Summary foreign currency balances that are the result of foreign currency translation have TRANSLATED_FLAG set to NULL. All summary account balances have TEMPLATE_ID not NULL. The columns that end in ADB are not used. Also, the REVALUATION_STATUS column is not used.
        
        gl_periods has followign important columns

        period_set_name -> should be joined with gl_ledgers-> period_set_name for joining with gl_periods table for for any transaction/journals ledger calender period data 
        period_name should be joined with transaction/journal period_name 
        start_date and endate define start date and end date of a period_name 
        period_type should be joined with with gl_ledger period_type for joining with gl_periods table for for any transaction/journals ledger calender period data
        period_year defines accounting year of the calendar
        period_num- do not by this. please use start date and end date always for joins for data between periods and dates
        period_name are case sensitive, examples - Feb-08 is not same as FEB-08. 
        it is ALWAYS necessary to use gl_ledger columns period_set_name,period_type  FOR JOINING WITH GL_PERIODS
        
        when gl_periods table is used it is mandatory to use the below joins
        WHEN ORG ID IS ONLY THERE IN TRANSACTION TABLES THEN USE JOIN TRANSACTIONS TABLE ORG_ID COLUMN WITH HR_OPERATING_UNITS AND THEN GET GL_LDEGER
    FOR EXAMPLE:

    PO_HEAERS_ALL.ORG_ID = HR_OPERATING_UNINTS.ORGANIZATION_ID
    AND HR_OPERATING_UNITS.SET_OF_BOOKS_ID = GL.LEDGER.LEDGER_ID
    AND GL_PERIODS.period_set_name = GL.LEDGER.period_set_name
    and GL_PERIODS.period_type = GL.LEDGER.ACCOUNTED_PERIOD_TYPE
    and gl.period.aADJUSTMENT_PERIOD_FLAG = 'N'
    and  h.CREATION_DATE <= (SELECT end_date FROM gl_periods WHERE UPPER(table.column_name) like 'upper('str%1')
    and  h.CREATION_DATE => (SELECT start_date FROM gl_periods WHERE UPPER(table.column_name) like 'upper('str%1')
    
    === Query Generation Guidelines ===
    1. VERY IMPORTANT 
    User provided parameters identified from the question are enclosed in opening <p> and closing tags </p> in the few shot examples.
    Similarly you enclose these tags while generating the query. 
    We have noticed you are adding tags for user provided parameters in the query generated by you. Make sure that the tags are present in the query generated by you.
    You must also enclose these tags while generating the query.
    DO NOT ADD ANY TAGS TO ANY OTHER FILTER CONDITIONS THAT ARE NOT PROVIDERD AS PARAMETERS IN THE QUESTION
    
    1.1 Adhere to  SQL syntax and best practices.
    - Use proper casing for keywords (e.g., SELECT, FROM, WHERE).
    - Use ANSI JOINs and ensure ON conditions accurately reference foreign and primary keys.
    - Use aliasing for all tables in SELECT, JOIN, WHERE, and GROUP BY clauses to improve readability and avoid ambiguity.
    
    2. For item-related queries, always filter by item name or description:
    - UPPER(m.segment1) LIKE UPPER('%item%')
    - UPPER(m.description) LIKE UPPER('%item%')
    
    3. Ensure joins reference appropriate columns:
    - Use po_headers_all.po_header_id = po_lines_all.po_header_id for header-line joins.
    - Match items by mtl_system_items_b.inventory_item_id = po_lines_all.item_id AND mtl_system_items_b.organization_id = po_distributions_all.destination_organization_id.

    4. Calculate ordered amounts based on line types and destination types:
    - For inventory lines, calculate as:
        po_lines_all.unit_rate * po_headers_all.rate * po_distributions_all.quantity_ordered
    - For non-inventory lines:
        po_headers_all.rate * po_distributions_all.amount_ordered

    5. Use NVL() to handle NULL values where appropriate:
    - NVL(po_headers_all.rate, 1)
    - NVL(po_distributions_all.quantity_ordered, 0)
    
    6. Aggregate functions must be correctly grouped:
    - Always group by vendor, item, or shipment when using SUM() or COUNT().

    7. When context is insufficient to generate the full query, return an intermediate SQL query:
    -- Intermediate SQL: SELECT DISTINCT m.segment1 FROM mtl_system_items_b m

    8. Maintain response formatting:
    SELECT ...
    FROM ...
    JOIN ...
    WHERE ...
    GROUP BY ...

    9. Correct common SQL issues:
    - Avoid implicit joins or non-standard joins.
    - Use COALESCE or NVL for null checks.
    - Use EXISTS instead of IN for subqueries when filtering large datasets.

    10. When encountering errors or ambiguous conditions, provide an explanation instead of generating incorrect SQL.

    11. When choosing column names in select or in joins or other conditions such as group by or orderby striclty confine to column names of the table only already provided as part of table schema
    12. Please do not invent, assume, create any columns that are not part of table defininions. Cross check with table schema before using the column name
    13. Please do not and never use more 30 charecters for alias names or comman table expressions. 30 is the max size
    14. Do not put ; at the end of the sql statement
    15. Do not use TO_TRUNC function as its not part of PostgreSQL database
    16. If the resultant query has any formula column that has null or 0 as denominator handle it appropriately 
    18. Do not add comments inbetween the query statements as it may cause syntax errors
    19. Use NULLIF when ever there is division by zero or null. Do not generate the query with out NULLIF when ever there is division by zero or null
    
    ### Additional Considerations:
        - Trim extra spaces in user input before applying conditions.
        - Do not put ; at the end of the sql statement
        - Ensure consistent logic across all SQL queries.
        - Always wrap string values in single quotes `' '`.
        - Avoid BETWEEN for date-based filters involving period_name—use subqueries for start_date and end_date instead.

    Modify all generated SQL queries to apply case sensitivity and string matching rules dynamically to all VARCHAR2 columns based on the following parameters: These rules apply consistently to all string parameters found in the WHERE conditions.

    ### **General Rules:**
    1. **Do not modify the original string values** provided by the user. Use them exactly as entered.
    2. **If multiple string parameters exist in the WHERE clause, apply the same rule to all.**
    3. **Use wildcards (`%`) for `like_string` conditions** by enclosing the string with `%` at the start, end, and in whitespace spaces between words.
        Example if the user inputs a string 'allied manufacturing' it has to convert it to '%allied%manufacturing%' so that the whitespace is filled with %
    4. **Do not change any month and year which is given as string by the user in the input query. 
    5. **table.column_name is case sensitive. Do not change. EXAMPLE: If the user inputs 'allied manufacturing' it has to passed as 'allied manufacturing'
    6. User provided parameters identified from the question are enclosed in opening <p> and closing tags </p> in the few shot examples.
    Similarly you enclose these tags while generating the query. 

    Tags handeling in the few shots while generating the queries
    1. User provided parameters identified from the question are enclosed in opening <p> and closing tags </p> in the few shot examples.
    Similarly you enclose these tags while generating the query. 

    ### **Wildcard Handling for `like_string`:**
    - **Use `%` at the start, end, and around whitespace in the string.**  
    - Example: `'str 1'` → `'%str%1%'`
    - **Do not modify numeric or non-string parameters.**  
    - **Always use `LIKE` for `like_string`, regardless of case sensitivity.**

    These rules must be enforced **consistently** for `VARCHAR2` columns in all generated SQL queries based on the parameters.

    === Few Shot Example Queries (Oracle Compliant) ===
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
    and jhead.status = 'P' 
    and jline.code_combination_id = gcc.code_combination_id

2 get trial balances of multiple ledgers when leger short names are provided as follows:
WITH trial_balance AS (
    SELECT
        LED.ledger_id, 
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
        LED.chart_of_accounts_id = 101
        AND LED.period_set_name ='Accounting' 
        AND LED.accounted_period_type ='Month' 
        AND GP.PERIOD_NAME ='<p>Jan-05</p>' 
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

IF THE QESTION OR USER QUERY  HAS BUDGET BALANCE  OR BUDGET AS WORD THEN THE WHERE CLAUSE SHOULD HAVE  ACTUAL_FLAG = 'B' OR THE STRING AS ENCUMERBANCE
BALANCE THEN THE ACTUAL_FLAG = 'E'
    
3.  to get the balance at the end of a given single period period in the ledger currency for any account (segment3) we need to do following:
    
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
        led.LEDGER_ID = 1
        AND gcc.account_type = 'E'
        AND gp.PERIOD_NAME ='<p>Mar-08</p>'  
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
        led.LEDGER_ID = 1
        AND gcc.account_type = 'E'
        AND gp.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = '<p>Mar-08</p>' AND period_set_name = led.period_set_name AND period_type = led.accounted_period_type)
        AND gp.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = '<p>Mar-08</p>' AND period_set_name = led.period_set_name AND period_type = led.accounted_period_type)
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
    NVL(pm.total_debit, 0) AS debit,  
    NVL(pm.total_credit, 0) AS credit, 
    NVL(ob.opening_balance, 0) + NVL(pm.total_debit, 0) - NVL(pm.total_credit, 0) AS closing_balance  
FROM
    PeriodMovements pm
LEFT JOIN
    OpeningBalance ob ON pm.account = ob.account
JOIN
    GL_LEDGERS led ON led.LEDGER_ID = 1
ORDER BY
    pm.account

4.  get net movment of any account in a single period say Mar-08 : 
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
        led.LEDGER_ID = 1
        AND gcc.account_type = 'E'
        AND gp.PERIOD_NAME ='<p>Mar-08</p>' 
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
        led.LEDGER_ID = 1
        AND gcc.account_type = 'E'
        AND gp.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = '<p>Mar-08</p>' AND period_set_name = led.period_set_name AND period_type = led.accounted_period_type)
        AND gp.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = '<p>Mar-08</p>' AND period_set_name = led.period_set_name AND period_type = led.accounted_period_type)
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
    NVL(ob.opening_balance, 0) AS opening_balance,  
    NVL(pm.total_debit, 0) AS debit,  
    NVL(pm.total_credit, 0) AS credit,  
    NVL(ob.opening_balance, 0) + NVL(pm.total_debit, 0) - NVL(pm.total_credit, 0) AS closing_balance 
FROM
    PeriodMovements pm
LEFT JOIN
    OpeningBalance ob ON pm.account = ob.account
JOIN
    GL_LEDGERS led ON led.LEDGER_ID = 1
ORDER BY
    pm.account

5.  to get opening balance at the start period and  period movement  for the period range and closing balance at the end of period across range of period say mar 08 to sep 08 we need to use following query
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
        led.LEDGER_ID = 1
        AND gcc.account_type ='E'
        AND gp.PERIOD_NAME ='<p>Mar-08</p>'  
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
        led.LEDGER_ID = 1
        AND gcc.account_type ='E'
        AND gp.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = '<p>Mar-08</p>' AND period_set_name = led.period_set_name AND period_type = led.accounted_period_type)
        AND gp.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = '<p>Sep-08</p>' AND period_set_name = led.period_set_name AND period_type = led.accounted_period_type)
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
    NVL(ob.opening_balance, 0) AS opening_balance, 
    NVL(pm.total_debit, 0) AS debit,  
    NVL(pm.total_credit, 0) AS credit,  
    NVL(ob.opening_balance, 0) + NVL(pm.total_debit, 0) - NVL(pm.total_credit, 0) AS closing_balance 
FROM
    PeriodMovements pm
LEFT JOIN
    OpeningBalance ob ON pm.account = ob.account
JOIN
    GL_LEDGERS led ON led.LEDGER_ID = 1
ORDER BY
    pm.account;


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
        LED.LEDGER_ID = 1
        AND GCC.account_type ='E'
        AND GP.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = '<p>Mar-08</p>' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
        AND GP.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = '<p>Apr-08</p>' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
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
    MAX(CASE WHEN period ='2008-03' THEN opening_balance END) AS "Mar-08 Opening",
    MAX(CASE WHEN period ='2008-03' THEN movement_in_accounting_period END) AS "Mar-08 Movement",
    MAX(CASE WHEN period ='2008-03' THEN closing_balance END) AS "Mar-08 Closing",
    MAX(CASE WHEN period ='2008-04' THEN opening_balance END) AS "Apr-08 Opening",
    MAX(CASE WHEN period ='2008-04' THEN movement_in_accounting_period END) AS "Apr-08 Movement",
    MAX(CASE WHEN period ='2008-04' THEN closing_balance END) AS "Apr-08 Closing"
FROM
    period_movement
GROUP BY
    period,account_segment2
ORDER BY
    account_segment2

IF THE QESTION OR USER QUERY  HAS BUDGET BALANCE  OR BUDGET AS WORD THEN THE WHERE CLAUSE SHOULD HAVE  ACTUAL_FLAG = 'B' OR THE STRING AS ENCUMERBANCE
BALANCE THEN THE ACTUAL_FLAG ='E'
    
7.For gl_periods table, peroid name should be converted to this format {period_name_normalization} based on the input provided by the user. 
    select bal.LEDGER_ID,LED.NAME LEDGER_NAME,BAL.CURRENCY_CODE,BAL.ACTUAL_FLAG,sum(nvl(BAL.PERIOD_NET_DR,0),nvl(BAL.PERIOD_NET_CR,0)
    BAL.BEGIN_BALANCE_DR,BAL.BEGIN_BALANCE_CR,
    from gl_balances bal, gl_code_combinations gcc, GL_LEDGERS LED,GL_PERIODS GP
    where bAL.code_combination_id = gcc.code_combination_id
    AND BAL.currency_code = LED.CURRENCY_CODE
    and bal.ledger_id = led.ledger_id
    AND ACTRUAL_FLAG = 'A'
    AND BAL.TEMPLATE_ID IS NULL
    AND BAL.TRANSLATED_FLAG IS NULL
    AND GP.PERIOD_YEAR= BAL.PERIOD_YEAR
    AND GP.PERIOD_NUM= BAL.PERIOD_NUM
    AND GP.PERIOD_SET_NAME = LED.PERIOD_SET_NAME
    AND GP.PERIOD_TYPE = LED.ACCOUNTED_PERIOD_TYPE
    and gp.start_date => (select start_date from gl_periods where period_name = (<start period from the question>) and PERIOD_SET_NAME = LED.PERIOD_SET_NAME and period_type = led.accounted_period_type)
    and gp.end_date <= (select end_date from gl_periods where period_name = (<end period from the question>) and PERIOD_SET_NAME = LED.PERIOD_SET_NAME and period_type = led.accounted_period_type)
    and LED.NAME = <ledger name in the query from user>
    group by bal.LEDGER_ID,LED.NAME LEDGER_NAME,BAL.CURRENCY_CODE,BAL.ACTUAL_FLAG


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
        GCC.segment3 ='<p>7360</p>'
        AND UPPER(LED.NAME) LIKE UPPER('<p>%Vision%Operations%</p>')
        AND GP.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = '<p>Jan-08</p>' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
        AND GP.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = '<p>Mar-08</p>' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
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

give me trial balance for vision operations ledger period wise

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
        UPPER(LED.NAME) LIKE UPPER('<p>%Vision%Operations%</p>')
        AND GP.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = '<p>Jan-08</p>' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
        AND GP.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = '<p>Mar-08</p>' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
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
    MAX(CASE WHEN period ='2008-01' THEN opening_balance END) AS "Jan-08 Opening",
    MAX(CASE WHEN period ='2008-02' THEN opening_balance END) AS "Feb-08 Opening",
    MAX(CASE WHEN period ='2008-03' THEN opening_balance END) AS "Mar-08 Opening",
    
    MAX(CASE WHEN period ='2008-01' THEN debit END) AS "Jan-08 Debit",
    MAX(CASE WHEN period ='2008-02' THEN debit END) AS "Feb-08 Debit",
    MAX(CASE WHEN period ='2008-03' THEN debit END) AS "Mar-08 Debit",

    MAX(CASE WHEN period ='2008-01' THEN credit END) AS "Jan-08 Credit",
    MAX(CASE WHEN period ='2008-02' THEN credit END) AS "Feb-08 Credit",
    MAX(CASE WHEN period ='2008-03' THEN credit END) AS "Mar-08 Credit",

    MAX(CASE WHEN period ='2008-01' THEN closing_balance END) AS "Jan-08 Closing",
    MAX(CASE WHEN period ='2008-02' THEN closing_balance END) AS "Feb-08 Closing",
    MAX(CASE WHEN period ='2008-03' THEN closing_balance END) AS "Mar-08 Closing"
FROM trial_balance
GROUP BY 
    account, 
    account_desc
ORDER BY 
    account;

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
        UPPER(LED.NAME) LIKE UPPER('<p>%Vision%Operations%</p>')
        AND GP.START_DATE >= (SELECT START_DATE FROM GL_PERIODS WHERE PERIOD_NAME ='<p>Jan-08</p>' AND PERIOD_SET_NAME = LED.PERIOD_SET_NAME AND PERIOD_TYPE = LED.ACCOUNTED_PERIOD_TYPE)
        AND GP.END_DATE <= (SELECT END_DATE FROM GL_PERIODS WHERE PERIOD_NAME ='<p>Mar-08</p>' AND PERIOD_SET_NAME = LED.PERIOD_SET_NAME AND PERIOD_TYPE = LED.ACCOUNTED_PERIOD_TYPE)
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
    
    MAX(CASE WHEN TB.period ='2008-01' THEN TB.closing_balance END) AS "Jan-08 Closing (USD)",
    MAX(CASE WHEN TB.period ='2008-02' THEN TB.closing_balance END) AS "Feb-08 Closing (USD)",
    MAX(CASE WHEN TB.period ='2008-03' THEN TB.closing_balance END) AS "Mar-08 Closing (USD)",

    
    MAX(CASE WHEN TB.period ='2008-01' THEN TB.closing_balance * GR.CONVERSION_RATE END) AS "Jan-08 Closing (GBP)",
    MAX(CASE WHEN TB.period ='2008-02' THEN TB.closing_balance * GR.CONVERSION_RATE END) AS "Feb-08 Closing (GBP)",
    MAX(CASE WHEN TB.period ='2008-03' THEN TB.closing_balance * GR.CONVERSION_RATE END) AS "Mar-08 Closing (GBP)"
FROM trial_balance TB
LEFT JOIN GL_DAILY_RATES GR 
    ON GR.CONVERSION_DATE = TB.period_end_date
    AND GR.CONVERSION_TYPE ='Corporate'
    AND GR.FROM_CURRENCY = TB.ledger_currency
    AND GR.TO_CURRENCY ='<p>GBP</p>'
GROUP BY 
    TB.account, 
    TB.account_desc
ORDER BY 
    TB.account;

TO GET THE LIST OF CHILD ACCOUNTS UNDER ONE ParENT ACCOUNT:
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
            AND FIS.SEGMENT_NAME ='Account'
            AND UPPER(GLL.NAME) LIKE UPPER('<p>%Vision%Operations%</p>') 
    )

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
               
                AND FFS.ID_FLEX_CODE = 'GL#' 
                AND FFS.APPLICATION_ID = 101
                AND FIS.SEGMENT_NAME ='Account'
                AND UPPER(GLL.NAME) LIKE UPPER('<p>%Vision%Operations%</p>')  
        )  
    START WITH PARENT_FLEX_VALUE ='<p>PT</p>'  
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
        and UPPER(LED.NAME) LIKE UPPER('<p>%Vision%Operations%</p>')  
        AND gp.PERIOD_NAME = '<p>Mar-08</p>' 
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


11. TO GET THE LIST OF ACCOUNTS OF A GIVEN TYPE SAY EXPENSE FOR MARCH 08FOLLOWING THE IS QUERY
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
    UPPER(LED.NAME) LIKE UPPER('<p>%Vision%Operations%</p>')
    AND gp.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = '<p>Mar-08</p>' AND period_set_name = led.period_set_name AND period_type = led.accounted_period_type)
    AND gp.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = '<p>Mar-08</p>' AND period_set_name = led.period_set_name AND period_type = led.accounted_period_type)
    AND bal.ACTUAL_FLAG ='A'
    AND bal.TEMPLATE_ID IS NULL
    AND bal.TRANSLATED_FLAG IS NULL
    AND bal.CURRENCY_CODE = led.CURRENCY_CODE
    and gcc.account_type ='E'
   GROUP BY
    gcc.segment3,
    gl_flexfields_pkg.get_description_sql(led.chart_of_accounts_id, 3, gcc.segment3)
ORDER BY
    account

   

12. get pivoted retsults means rows as columns0> give me month wise period movement actuals for ACCOUNT 7360 for ledger Vision Operations between jan 2008 and mar 2008 along with growth month on month 
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
        GCC.segment3 ='<p>7360</p>'
        AND UPPER(LED.NAME) LIKE UPPER('<p>%Vision%Operations%</p>')
        AND GP.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = '<p>Jan-08</p>' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
        AND GP.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = '<p>Mar-08</p>' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
        AND BAL.ACTUAL_FLAG ='A'
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
    MAX(CASE WHEN period ='2008-01' THEN movement_in_accounting_period END) AS "Jan-08",
    MAX(CASE WHEN period ='2008-02' THEN movement_in_accounting_period END) AS "Feb-08",
    MAX(CASE WHEN period ='2008-03' THEN movement_in_accounting_period END) AS "Mar-08"
FROM period_growth
UNION ALL
SELECT
    'Growth (%)' AS metric,
    NULL AS "Jan-08",
    MAX(CASE WHEN period ='2008-02' THEN month_on_month_growth END) AS "Feb-08",
    MAX(CASE WHEN period ='2008-03' THEN month_on_month_growth END) AS "Mar-08"
FROM period_growth;

IF THE QESTION OR USER QUERY  HAS BUDGET BALANCE  OR BUDGET AS WORD THEN THE WHERE CLAUSE SHOULD HAVE  ACTUAL_FLAG ='<p>B</p>' OR THE STRING AS ENCUMERBANCE
BALANCE THEN THE ACTUAL_FLAG ='E'

13.
    select QLH.NAME,QLH.DESCRIPTION,QLH.START_DATE_ACTIVE,QLL.OPERAND,QLL.ARITHMETIC_OPERATOR,
    OOLA.ordered_quantity,oola.order_quantity_uom,oola.ordered_item,ooha.order_number
    from apps.qp_list_headers QLH ,
    apps.qp_list_lines QLL,
    apps.qp_pricing_attributes qpa,
    apps.oe_order_lines_all oola,
    apps.oe_order_headers_all ooha
    WHERE QLH.list_HEADER_ID=QLL.list_HEADER_ID
    and qpa.LIST_LINE_ID=qll.LIST_LINE_ID
    and qpa.list_HEADER_ID=qlh.list_HEADER_ID
    and to_char(oola.INVENTORY_ITEM_ID) =QPA.PRODUCT_ATTR_VALUE
    and oola.header_id=ooha.header_id
    and ooha.order_number=:P_ORDER_NUMBER


14. how to get chart of account structure of a given ledger as each account will have multiple segments
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
        OVER (PARTITION BY fifs.id_flex_num, fics.segment_num) AS NonGlobalCount 
    FROM
        fnd_id_flex_structures fifs
    JOIN
        fnd_id_flex_segments fics ON fifs.id_flex_num = fics.id_flex_num
    JOIN
        fnd_flex_value_sets fvs ON fics.flex_value_set_id = fvs.flex_value_set_id
    JOIN    
        fnd_segment_attribute_values sav 
        ON fifs.application_id = sav.application_id
        AND SAV.ATTRIBUTE_VALUE ='Y'
        AND fifs.id_flex_code = sav.id_flex_code
        AND fics.application_column_name = sav.application_column_name  
        AND fics.id_flex_num = sav.id_flex_num 
        AND fics.id_flex_code = sav.id_flex_code
    JOIN 
        gl_ledgers gll ON fifs.id_flex_num = gll.chart_of_accounts_id      
    WHERE
        fifs.id_flex_code = 'GL#'  
        AND fifs.application_id = 101  
        AND gll.ledger_id = 1
)
SELECT *
FROM QualifierData
WHERE 
    
    NOT (SEGMENT_ATTRIBUTE_TYPE = 'GL_GLOBAL' AND NonGlobalCount > 0)
ORDER BY chart_of_accounts_id, segment_num;


The result is typically as follows

CHART_OF_ACCOUNTS_ID	APPLICATION_COLUMN_NAME	SEGMENT_NUM	SEGMENT_NAME	FLEX_VALUE_SET_ID	FLEX_VALUE_SET_NAME	LEDGER_ID	SEGMENT_ATTRIBUTE_TYPE
101	SEGMENT1	1	Company	1002470	Operations Company	1	GL_BALANCING
101	SEGMENT2	2	Department	1002471	Operations Department	1	FA_COST_CTR
101	SEGMENT3	3	Account	1002472	Operations Account	1	GL_ACCOUNT
101	SEGMENT4	4	Sub-Account	1002473	Operations Sub-Account	1	GL_GLOBAL
101	SEGMENT5	5	Product	1002474	Operations Product	1	GL_GLOBAL

note the user may want account balance by segments and may refer to them by APPLICATION_COLUMN_NAME such as coompany for segment1,
department for segment2,product for segment5,sub_account for segment4 etc.,

15 can you give me month wise opening balance, movement and  closing balance only between mar 08 and apr 08 for all expense accounts at department segment level for ledger id 1"

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
        LED.LEDGER_ID = 1
        AND GCC.account_type ='E'
        AND GP.START_DATE >= (SELECT start_date FROM gl_periods WHERE period_name = '<p>Mar-08</p>' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
        AND GP.END_DATE <= (SELECT end_date FROM gl_periods WHERE period_name = '<p>Apr-08</p>' AND period_set_name = LED.period_set_name AND period_type = LED.accounted_period_type)
        AND BAL.ACTUAL_FLAG ='A'
        AND BAL.TEMPLATE_ID IS NULL
        AND BAL.TRANSLATED_FLAG IS NULL
        AND BAL.CURRENCY_CODE = LED.CURRENCY_CODE
    GROUP BY
        TO_CHAR(GP.START_DATE, 'YYYY-MM'),
        GCC.segment2 
)
SELECT
    department_segment,
    MAX(CASE WHEN period ='2008-03' THEN opening_balance END) AS "Mar-08 Opening",
    MAX(CASE WHEN period ='2008-03' THEN movement_in_accounting_period END) AS "Mar-08 Movement",
    MAX(CASE WHEN period ='2008-03' THEN closing_balance END) AS "Mar-08 Closing",
    MAX(CASE WHEN period ='2008-04' THEN opening_balance END) AS "Apr-08 Opening",
    MAX(CASE WHEN period ='2008-04' THEN movement_in_accounting_period END) AS "Apr-08 Movement",
    MAX(CASE WHEN period ='2008-04' THEN closing_balance END) AS "Apr-08 Closing"
FROM
    period_movement
GROUP BY
    department_segment
ORDER BY
    department_segment

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
        OVER (PARTITION BY fifs.id_flex_num, fics.segment_num) AS NonGlobalCount 
    FROM
        fnd_id_flex_structures fifs
    JOIN
        fnd_id_flex_segments fics ON fifs.id_flex_num = fics.id_flex_num
    JOIN
        fnd_flex_value_sets fvs ON fics.flex_value_set_id = fvs.flex_value_set_id
    JOIN    
        fnd_segment_attribute_values sav 
        ON fifs.application_id = sav.application_id
        AND SAV.ATTRIBUTE_VALUE ='Y'
        AND fifs.id_flex_code = sav.id_flex_code
        AND fics.application_column_name = sav.application_column_name  
        AND fics.id_flex_num = sav.id_flex_num 
        AND fics.id_flex_code = sav.id_flex_code
    JOIN 
        gl_ledgers gll ON fifs.id_flex_num = gll.chart_of_accounts_id      
    WHERE
        fifs.id_flex_code = 'GL#'  
        AND fifs.application_id = 101  
        AND gll.ledger_id = 1
)
SELECT *
FROM QualifierData
WHERE 
    NOT (SEGMENT_ATTRIBUTE_TYPE = 'GL_GLOBAL' AND NonGlobalCount > 0)
ORDER BY chart_of_accounts_id, segment_num;


17.1 Calculate total purchase for item AS54888:
   SELECT SUM(pl.unit_price * NVL(h.rate, 1) * NVL(pd.quantity_ordered, 0)) AS total_purchase
   FROM po_distributions_all pd
   JOIN po_lines_all pl ON pl.po_line_id = pd.po_line_id
   JOIN po_headers_all h ON pl.po_header_id = h.po_header_id   
   left JOIN mtl_system_items_b m ON pd.destination_organization_id = m.organization_id   
        AND pl.item_id = m.inventory_item_id
   WHERE (UPPER(m.segment1) LIKE UPPER('%<p>AS54888</p>%') OR UPPER(m.description) LIKE UPPER('%<p>AS54888</p>%'))

17.2 Calculate total purchase for item AS54888 for Vision Operations:
   SELECT SUM(pl.unit_price * NVL(h.rate, 1) * NVL(pd.quantity_ordered, 0)) AS total_purchase
   FROM po_distributions_all pd
   JOIN po_lines_all pl ON pl.po_line_id = pd.po_line_id
   JOIN po_headers_all h ON pl.po_header_id = h.po_header_id
   left JOIN mtl_system_items_b m ON pd.destination_organization_id = m.organization_id   
        AND pl.item_id = m.inventory_item_id
   JOIN hr_operating_units hou on hou.organization_id = h.org_id
   WHERE (UPPER(m.segment1) LIKE UPPER('%<p>AS54888</p>%') OR UPPER(m.description) LIKE UPPER('%<p>AS54888</p>%'))
   and UPPER(hou.name) LIKE UPPER('<p>%Vision%Operations%</p>');


   
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
   WHERE  UPPER(hou.name) LIKE UPPER('<p>%Vision%Operations%</p>')
   and  (UPPER(m.segment1) LIKE UPPER('%<p>AS54888</p>%') OR UPPER(m.description) LIKE UPPER('%<p>AS54888</p>%'))
   group by primary_uom_code,UNIT_MEAS_LOOKUP_CODE

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
    rt.transaction_type ='RECEIVE' 
GROUP BY
    ph.segment1,
    ph.creation_date,
    hou.name,
    rt.transaction_date,
    msi.segment1,
    rt.quantity

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
    ph.segment1 = '<p>4683</p>'
    AND ph.authorization_status ='APPROVED'
GROUP BY
    aps.vendor_name,
    ph.segment1,
    ph.creation_date,
    hou.name,
    org.organization_name,
    NVL(msi.segment1, pl.item_description),
    pl.unit_price,
    pl.po_line_id

17.6 Calculate total purchases expense account wise in vision operations
SELECT
    hou.name,gl_flexfields_pkg.get_description_sql(led.chart_of_accounts_id, 3, gcc.segment3) AS expense_account_desc,
    gcc.segment3 AS expense_account,
    SUM(pl.unit_price * NVL(h.rate, 1) * NVL(pd.quantity_ordered, 0)) AS total_purchase
FROM
    po_distributions_all pd
JOIN po_lines_all pl ON pl.po_line_id = pd.po_line_id
JOIN po_headers_all h ON pl.po_header_id = h.po_header_id
# LEFT JOIN mtl_system_items_b m ON pd.destination_organization_id = m.organization_id
    AND pl.item_id = m.inventory_item_id
JOIN hr_operating_units hou ON hou.organization_id = h.org_id
JOIN gl_ledgers led ON hou.set_of_books_id = led.ledger_id
JOIN gl_code_combinations gcc ON pd.code_combination_id = gcc.code_combination_id
WHERE  UPPER(hou.name) LIKE UPPER('<p>%Vision%Operations%</p>')
GROUP BY
    gl_flexfields_pkg.get_description_sql(led.chart_of_accounts_id, 3, gcc.segment3),
    gcc.segment3,hou.name

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
    UPPER(hr_operating_units.name) LIKE UPPER('<p>%Vision%Operations%</p>')
    AND EXTRACT(YEAR FROM h.creation_date) = '<p>2003</p>'
GROUP BY
    gl_flexfields_pkg.get_description_sql(led.chart_of_accounts_id, 3, gcc.segment3),
    gcc.segment3,
    aps.vendor_name
ORDER BY
    expense_account_desc,
    aps.vendor_name

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
    h.segment1 = '<p>4586</p>'
    AND rt.transaction_type ='RECEIVE'
ORDER BY
    rt.transaction_date

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
    AND UPPER(hou.name) LIKE UPPER('<p>%Vision%Operations%</p>')
ORDER BY
    prh.segment1, prl.line_num, prd.distribution_id


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
    AND pah.object_type_code ='PO' 
    and object_sub_type_code ='STANDARD'
LEFT JOIN fnd_user fu
    ON pah.employee_id = fu.employee_id
WHERE
    h.segment1 = '<p>4586</p>'
ORDER BY
    pah.object_revision_num,pah.sequence_num
    
18.1List closed shipments by item and vendor:
   SELECT loc.closed_date, loc.shipment_type, m.description AS item_description, v.vendor_name
   FROM po_line_locations_all loc
   JOIN po_lines_all l ON loc.po_line_id = l.po_line_id
   JOIN po_distributions_all pd on loc.line_location_id = pd.line_location_id
   left JOIN mtl_system_items_b m ON l.item_id = m.inventory_item_id AND pd.destination_organization_id = m.organization_id
   JOIN po_headers_all h ON l.po_header_id = h.po_header_id
   JOIN ap_suppliers v ON h.vendor_id = v.vendor_id
   WHERE 1=1 AND NVL(LOC.CLOSED_FLAG,'N') ='Y';

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
    ON h.requisition_header_id = l.requisition_header_id and h.segment1 = '<p>1234</p>'
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
    h.type_lookup_code ='PURCHASE'
    AND h.authorization_status ='APPROVED'
ORDER BY
    h.segment1, l.line_num

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
       AND UPPER (hou.name) LIKE UPPER ('<p>%Vision%Operations%</p>')
       AND ph.creation_date BETWEEN to_date ('<p>2000-01-01</p>','YYYY-MM-DD')
                                AND to_date('<p>2008-12-31</p>','YYYY-MM-DD')

19.give me top 10 vendors for AS54888, show total purchased value
   SELECT * FROM ( SELECT v.vendor_name, SUM(CASE WHEN l.line_type_id = 1 AND d.destination_type_code ='INVENTORY' THEN 
   l.unit_price * nvl(h.rate, 1) * d.quantity_ordered ELSE nvl(h.rate, 1) * d.amount_ordered END) AS total_purchased_value 
   FROM po_headers_all h JOIN po_lines_all l ON h.po_header_id = l.po_header_id JOIN po_distributions_all d ON l.po_line_id = d.po_line_id 
   JOIN ap_suppliers v ON h.vendor_id = v.vendor_id left JOIN mtl_system_items_b msi ON l.item_id = msi.inventory_item_id 
   AND d.destination_organization_id = msi.organization_id WHERE UPPER(msi.segment1) LIKE UPPER('%<p>AS54888</p>%') OR UPPER(msi.description) LIKE UPPER('%<p>AS54888</p>%')
   GROUP BY v.vendor_name ORDER BY total_purchased_value DESC  ) WHERE ROWNUM <= 10   

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
           WHERE     pha.approved_flag ='Y'
                 AND gp.period_name = '<p>Mar-07</p>'
                 AND gp.period_set_name = (SELECT period_set_name
                                             FROM gl_ledgers
                                            WHERE ledger_id = hou.set_of_books_id)
                 AND gp.period_type = (SELECT accounted_period_type
                                         FROM gl_ledgers
                                        WHERE ledger_id = hou.set_of_books_id)
        GROUP BY aps.vendor_name
        ORDER BY total_purchase_value DESC)
 WHERE ROWNUM <= 10



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
     and poh.approved_flag ='Y'
    AND poh.creation_date BETWEEN DATE '2007-01-01' AND DATE '2007-12-31'
GROUP BY
    pol.item_id, pol.item_description, poh.vendor_id, aps.vendor_name, TO_CHAR(poh.creation_date, 'YYYY-MM')        
HAVING
    COUNT(item_id) > 2
    order by item_id,vendor_id



  22. dealing with Puchase order tables and getting ledger details from them

   select ledger_id from gl_ldgers gll,hr_operating_units hu,po_headers_all poh
   where poh.org_id = hu.organization_id
   and hu.set_of_books_id = gll.ledger_id
   
   23. dealing to get start date and endate of periods connected to a ledgers 


select period_name,period_year,start_date,end_date,period_num,quarter_num,adjustment_period_flag,year_start_date,quarter_start_date
from
gl_periods glp,gl_ledgers gll
where glp.period_set_name = gll.PERIOD_SET_NAME
and glp.period_type= gll.accounted_period_type


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
                   AND glp.period_name = '<p>Jan-05</p>' 
WHERE
    rt.transaction_date BETWEEN glp.start_date AND glp.end_date
ORDER BY
    pha.segment1,
    pla.line_num,
    pda.distribution_num,
    rt.transaction_date;

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
                         AND glp_start.period_name = '<p>Jan-05</p>' 
JOIN 
    gl_periods glp_end ON glp_end.period_set_name = gll.period_set_name 
                       AND glp_end.period_type = gll.accounted_period_type
                       AND glp_end.period_name = '<p>Mar-05</p>'  
WHERE
    rt.transaction_date BETWEEN glp_start.start_date AND glp_end.end_date
ORDER BY
    pha.segment1,
    pla.line_num,
    pda.distribution_num,
    rt.transaction_date;

26.  query to get purchase beyond promised or NEED by date or never delivered
       WITH receive_info AS (
    SELECT
        rt.po_distribution_id,
        MIN(rt.transaction_date) AS first_receive_date,
        SUM(rt.quantity) AS total_received_qty
    FROM
        rcv_transactions rt
    WHERE
        rt.transaction_type ='RECEIVE'
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
    ON pha.po_header_id = pla.po_header_id and pha.authorization_status ='APPROVED'
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
    rcv.first_receive_date IS NULL
    OR rcv.first_receive_date > pll.promised_date
    OR rcv.first_receive_date > pll.need_by_date
ORDER BY
    pha.segment1,
    pla.line_num,
    pda.distribution_num

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
        rt.transaction_type ='DELIVER'
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
            transaction_type ='RECEIVE'
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
            transaction_type ='DELIVER'
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
            transaction_type ='RECEIVE'
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
         poh1.approved_flag ='Y' 
         AND poh1.creation_date < DATE '2007-01-01'
    ) prev_pol 
    ON pol.item_id = prev_pol.item_id AND prev_pol.rn = 1
WHERE 
    poh.approved_flag ='Y' 
    AND poh.creation_date BETWEEN DATE '2007-01-01' AND DATE '2007-01-31'
    AND pol.unit_price > prev_pol.unit_price
ORDER BY 
    pol.item_id, poh.creation_date;

note: unless otherwise specified in question, purchase amount is always in local currency only it has arrived at by unit_price* rate* rate column in po headers

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
        poh.approved_flag ='Y'
        AND poh.creation_date BETWEEN DATE '2007-01-01' AND DATE '2007-12-31'
),
monthly_purchase_count AS (
    SELECT
        item_id,
        item_description,
        purchase_month,
        COUNT(DISTINCT po_number) AS purchase_count 
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
   WHERE poh.segment1 = '<p>4682</p>'
   AND rt.transaction_type='RECEIVE'
ORDER BY rt.transaction_date


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
    WHERE poh.type_lookup_code ='STANDARD'
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

36.GIVE ME THE PAYMENT DETAILS FOR INVOICE NUMBER CRAC Apr 06 09
SELECT
    aip.INVOICE_PAYMENT_ID,
    aip.payment_num,
    aip.CREATION_date, 
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

39.getting some selected invoice details provided po number is given SAY PO NUMBER AS 4682
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
    poh.segment1 = '<p>4682</p>' 
ORDER BY
    ai.invoice_date    

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
         AND aid.LINE_TYPE_LOOKUP_CODE ='PREPAY'
         AND pre_ai.INVOICE_TYPE_LOOKUP_CODE ='PREPAYMENT'
GROUP BY STD_AI.INVOICE_ID,
         std_ai.INVOICE_NUM,
         std_ai.INVOICE_DATE,
         std_ai.INVOICE_AMOUNT,
         std_ai.VENDOR_ID,
         s.VENDOR_NAME,
         pre_ai.INVOICE_NUM,
         pre_ai.INVOICE_DATE,
         pre_ai.INVOICE_AMOUNT


41. get the list of oustanding invoices as on 31 dec 2007 for vision operations backdate
  WITH cutoff_param AS (

    SELECT 
        TO_DATE('2007-12-31', 'YYYY-MM-DD') AS cutoff_date,
        'Vision Operations' AS operating_unit_name
    FROM dual
),  payments_after_cutoff AS (

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

    SELECT
        aid.INVOICE_ID,
        SUM(NVL(aid.AMOUNT, 0)) AS prepay_after_cutoff
    FROM
        ap_invoice_distributions_all aid
        JOIN ap_invoices_all ai ON aid.INVOICE_ID = ai.INVOICE_ID
        JOIN hr_operating_units hr ON ai.ORG_ID = hr.ORGANIZATION_ID
        JOIN cutoff_param c ON 1 = 1
    WHERE
        aid.LINE_TYPE_LOOKUP_CODE ='PREPAY'
        AND aid.ACCOUNTING_DATE > c.cutoff_date
        AND UPPER(hr.NAME) = UPPER(c.operating_unit_name)
    GROUP BY
        aid.INVOICE_ID
),  invoice_balances AS (

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

    (ib.GROSS_AMOUNT - NVL(pac.payment_after_cutoff, 0) - NVL(ppc.prepay_after_cutoff, 0)) <> 0
ORDER BY
    UNPAID_BALANCE_AS_OF_CUTOFF
       
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
                     WHERE     aid.LINE_TYPE_LOOKUP_CODE ='PREPAY'
                           AND pre_ai.INVOICE_TYPE_LOOKUP_CODE ='PREPAYMENT'
                  GROUP BY aid.INVOICE_ID) prepays
          ON ai.INVOICE_ID = prepays.INVOICE_ID
       LEFT JOIN (  SELECT aps.INVOICE_ID,
                           SUM (aps.AMOUNT_REMAINING)
                              AS TOTAL_OUTSTANDING_AMOUNT
                      FROM ap_payment_schedules_all aps
                  GROUP BY aps.INVOICE_ID) schedules
          ON ai.INVOICE_ID = schedules.INVOICE_ID
 WHERE ai.INVOICE_NUM ='<p>ABC</p>'

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
    AND upper(h.name) LIKE '%VISION%OPERATIONS%'
    AND ai.wfapproval_status IN ( 'WFAPPROVED', 'MANUALLY APPROVED', 'NOT REQUIRED' )
    AND ai.cancelled_date IS NULL 

43.1 GIVE ME THE OUTSTANDING INVOICES FOR CONSOLIDATED SUPPLIES AS OF TODAY
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
   AND UPPER(h.NAME) LIKE '%VISION%OPERATIONS%'
   AND ai.WFAPPROVAL_STATUS IN ('WFAPPROVED', 'MANUALLY APPROVED', 'NOT REQUIRED')
   AND ai.CANCELLED_DATE IS NULL
   AND s.VENDOR_NAME = 'Consolidated Supplies'  
ORDER BY aps.DUE_DATE


44. Get list of invoices paid before due date
SELECT ai.INVOICE_ID, ai.INVOICE_NUM, ai.INVOICE_DATE, aps.DUE_DATE, aps.AMOUNT_REMAINING, ai.VENDOR_ID, s.VENDOR_NAME, ai.CREATION_DATE AS INVOICE_CREATION_DATE, ai.LAST_UPDATE_DATE AS INVOICE_LAST_UPDATE_DATE, aps.PAYMENT_NUM, aip.CREATION_DATE AS PAYMENT_DATE, aps.DUE_DATE - aip.CREATION_DATE AS ADVANCED_DAYS FROM ap_invoices_all ai JOIN ap_suppliers s ON ai.VENDOR_ID = s.VENDOR_ID JOIN ap_payment_schedules_all aps ON ai.INVOICE_ID = aps.INVOICE_ID JOIN ap_invoice_payments_all aip ON aps.INVOICE_ID = aip.INVOICE_ID AND aps.PAYMENT_NUM = aip.PAYMENT_NUM JOIN ap_checks_all aca ON aip.CHECK_ID = aca.CHECK_ID WHERE ai.CANCELLED_DATE IS NULL AND ai.INVOICE_AMOUNT > 0 AND aps.DUE_DATE - aip.CREATION_DATE > 0 ORDER BY aps.PAYMENT_NUM, ai.INVOICE_ID, aip.PAYMENT_NUM

45.  list of approvers against specific  invoice
SELECT ai.INVOICE_NUM, CASE WHEN ai.WFAPPROVAL_STATUS = 'NOT REQUIRED' THEN 'NOT REQUIRED' WHEN ai.WFAPPROVAL_STATUS ='REQUIRED' THEN 'NOT SUBMITTED FOR APPROVAL' ELSE aih.APPROVER_NAME END AS APPROVER, CASE WHEN ai.WFAPPROVAL_STATUS = 'NOT REQUIRED' THEN 'NOT REQUIRED' WHEN ai.WFAPPROVAL_STATUS ='<p>REQUIRED</p>' THEN 'NOT SUBMITTED FOR APPROVAL' ELSE aih.RESPONSE END AS RESPONSE, CASE WHEN ai.WFAPPROVAL_STATUS = 'NOT REQUIRED' THEN 'NOT REQUIRED' WHEN ai.WFAPPROVAL_STATUS ='<p>REQUIRED</p>' THEN 'NOT SUBMITTED FOR APPROVAL' ELSE aih.APPROVER_COMMENTS END AS APPROVER_COMMENTS, CASE WHEN ai.WFAPPROVAL_STATUS IN ('NOT REQUIRED', 'REQUIRED') THEN NULL ELSE aih.CREATION_DATE END AS ACTION_DATE FROM AP_INVOICES_ALL ai LEFT JOIN AP_INV_APRVL_HIST_ALL aih ON ai.INVOICE_ID = aih.INVOICE_ID AND aih.ITERATION = (SELECT MAX(iteration) FROM AP_INV_APRVL_HIST_ALL WHERE INVOICE_ID = ai.INVOICE_ID) WHERE ai.INVOICE_NUM ='<p>abc123</p>' ORDER BY aih.APPROVAL_HISTORY_ID

46. LIST OF APPROVERS AND APPROVAL ACTIONS 
SELECT AH.APPROVAL_HISTORY_ID, 
       AH.RESPONSE, 
       AH.ITERATION, 
       AH.APPROVER_NAME, 
       AH.INVOICE_ID, 
       AH.CREATION_DATE AS ACTION_DATE 
FROM AP_INV_APRVL_HIST_ALL AH
WHERE AH.HISTORY_TYPE ='DOCUMENTAPPROVAL'
  AND AH.APPROVAL_HISTORY_ID = (
      SELECT MAX(AH2.APPROVAL_HISTORY_ID)
      FROM AP_INV_APRVL_HIST_ALL AH2
      WHERE AH2.INVOICE_ID = AH.INVOICE_ID
  );

47. LIST APPROVERS WHO APPROVED AN INVOCE WHEN INVOIE NUMBER IS GIVEN SAY 'Approv'
SELECT 
AH.APPROVAL_HISTORY_ID, AI.INVOICE_ID,AI.INVOICE_AMOUNT,AI.CREATION_DATE INV_CREATION_DATE,
       AH.RESPONSE, 
       AH.ITERATION, 
       AH.APPROVER_NAME, 
       AH.INVOICE_ID, 
       ah.approver_comments,
       AH.CREATION_DATE AS ACTION_DATE 
FROM AP_INV_APRVL_HIST_ALL AH, AP_INVOICES_ALL AI
WHERE 1=1
AND AH.INVOICE_ID = AI.INVOICE_ID
AND AI.INVOICE_ID  IN (SELECT INVOICE_ID FROM AP_INVOICES_ALL WHERE INVOICE_NUM ='<p>Approv</p>')
ORDER BY AI.INVOICE_ID,AH.APPROVAL_HISTORY_ID, AH.ITERATION

48.list of invoices not approved even after 10 days from creation date,
WITH q1 AS (
    SELECT AH.APPROVAL_HISTORY_ID, 
           AH.RESPONSE, 
           AH.ITERATION, 
           AH.APPROVER_NAME, 
           AH.INVOICE_ID, 
           AH.CREATION_DATE AS ACTION_DATE, 
           AH.LAST_UPDATE_DATE
    FROM AP_INV_APRVL_HIST_ALL AH
    WHERE AH.HISTORY_TYPE ='DOCUMENTAPPROVAL'
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
                AND (q1.ACTION_DATE - q2.CREATION_DATE) > 10 
           THEN 'Approved but took more than 10 days'
           WHEN q1.RESPONSE NOT IN ('WFAPPROVED', 'MANUALLY APPROVED', 'APPROVED','NOT REQUIRED') 
                AND (SYSDATE - q2.CREATION_DATE) > 10 
           THEN 'Pending for more than 10 days'
           ELSE 'Does not meet criteria'
       END AS STATUS
FROM q1 
JOIN q2 ON q1.INVOICE_ID = q2.INVOICE_ID
WHERE (q1.RESPONSE IN ('WFAPPROVED', 'MANUALLY APPROVED', 'APPROVED','NOT REQUIRED') 
        AND (q1.ACTION_DATE - q2.CREATION_DATE) > 10)
   OR (q1.RESPONSE NOT IN ('WFAPPROVED', 'MANUALLY APPROVED', 'APPROVED','NOT REQUIRED') 
        AND (SYSDATE - q2.CREATION_DATE) > 10); 

        
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
 WHERE ai.invoice_num ='<p>abc123</p>' AND ai.cancelled_date IS NULL
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
 WHERE     aid.line_type_lookup_code ='PREPAY'
       AND NVL (aid.reversal_flag, 'N') <> 'Y'
       AND ai.cancelled_date IS NULL
       AND ai.invoice_num ='<p>abc123</p>'
ORDER BY accounting_date
    
    ### === Few Shot Example Queries End ===        

"""





prompt=""" 
You are a Oracle EBS expert. Generate executable SQL based on the user's question.
Only output SQL query. Do not invent columns - use only those in the schema.
CRITICAL: Columns marked with **[ESSENTIAL]** are mandatory for proper joins and aggregations - prioritize them.

[User Question]
Give me the list of invoices where the purchase order payment terms is different from invoice payment terms  Filters/Constraints: operating unit vision operation only  Return ONLY these columns: invoice number, invoice date, puchase order number, purchase order payment terms, invoice payment terms

[Database Schema]
{db_schema}

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


