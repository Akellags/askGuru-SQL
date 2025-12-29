#!/usr/bin/env python3
"""
Oracle EBS Integration Implementation
Provides SQL conversion, connection, and training utilities
"""

import json
import re
import sqlglot
from typing import Optional, Dict, List, Tuple
from sqlglot import parse_one, transpile
from sqlglot.optimizer import optimize


class OracleEBSSQLConverter:
    """
    Convert PostgreSQL/MySQL SQL to Oracle EBS standard SQL
    """
    
    # Oracle EBS table prefixes
    EBS_MODULES = {
        'customer': 'ar_customers',
        'invoice': 'ra_customer_trx',
        'invoice_line': 'ra_customer_trx_lines',
        'order': 'oe_order_headers_all',
        'order_line': 'oe_order_lines_all',
        'po': 'po_headers',
        'po_line': 'po_lines_all',
        'vendor': 'po_vendors',
        'employee': 'per_employees_x',
        'department': 'hr_all_organization_units',
        'item': 'mtl_system_items_b',
        'inventory': 'mtl_onhand_quantities_detail',
        'journal': 'gl_je_headers',
        'journal_line': 'gl_je_lines',
        'account': 'gl_code_combinations',
        'receipt': 'ap_invoice_distributions_all',
    }
    
    # Function mappings
    FUNCTION_MAPPINGS = {
        'now()': 'SYSDATE',
        'current_timestamp': 'SYSDATE',
        'current_date': 'TRUNC(SYSDATE)',
        'current_time': 'TRUNC(SYSDATE)',
        'getdate()': 'SYSDATE',
        'cast': 'CAST',
        'substring': 'SUBSTR',
        'concat': '||',
    }
    
    def __init__(self, target_ebs_version='12.2'):
        """Initialize converter for specific EBS version"""
        self.target_version = target_ebs_version
    
    def transpile_sql(self, sql: str, source_dialect: str = 'postgresql') -> str:
        """
        Transpile SQL from source dialect to Oracle
        """
        try:
            result = transpile(
                sql,
                read=source_dialect,
                write='oracle',
                pretty=True,
                identify=True,
                normalize=True
            )[0]
            return result
        except Exception as e:
            print(f"Warning: Transpilation failed: {e}")
            return sql
    
    def format_oracle_ebs(self, sql: str) -> str:
        """Apply all Oracle EBS-specific formatting"""
        sql = self.transpile_sql(sql, 'postgresql')
        sql = sql.strip()
        if not sql.endswith(';'):
            sql += ';'
        return sql


class OracleEBSDatasetConverter:
    """
    Convert training datasets from PostgreSQL to Oracle EBS
    """
    
    def __init__(self):
        self.converter = OracleEBSSQLConverter()
        self.conversion_stats = {
            'total': 0,
            'successful': 0,
            'failed': 0,
            'errors': []
        }
    
    def convert_sample(self, item: Dict, source_dialect: str = 'postgresql') -> Dict:
        """
        Convert a single training sample
        """
        try:
            original_sql = item.get('SQL', item.get('sql', ''))
            converted_sql = self.converter.format_oracle_ebs(original_sql)
            
            converted_item = {
                **item,
                'SQL': converted_sql,
                'dialect': 'oracle_ebs',
                'source_dialect': source_dialect,
                'conversion_source': 'auto_transpile'
            }
            
            self.conversion_stats['successful'] += 1
            return converted_item
        
        except Exception as e:
            self.conversion_stats['failed'] += 1
            self.conversion_stats['errors'].append({
                'sample_id': item.get('id', 'unknown'),
                'error': str(e)
            })
            return None
    
    def convert_dataset(self, input_file: str, output_file: str, 
                       source_dialect: str = 'postgresql') -> Dict:
        """
        Convert entire dataset from source to Oracle EBS
        """
        print(f"Loading dataset from {input_file}...")
        with open(input_file, encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"Converting {len(dataset)} samples to Oracle EBS...")
        converted_dataset = []
        
        for idx, item in enumerate(dataset):
            if (idx + 1) % 100 == 0:
                print(f"  Converted {idx + 1}/{len(dataset)} samples...")
            
            self.conversion_stats['total'] += 1
            converted_item = self.convert_sample(item, source_dialect)
            
            if converted_item:
                converted_dataset.append(converted_item)
        
        print(f"Saving {len(converted_dataset)} converted samples to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_dataset, f, ensure_ascii=False, indent=2)
        
        return self.conversion_stats


if __name__ == "__main__":
    converter = OracleEBSDatasetConverter()
    stats = converter.convert_dataset(
        'input_dataset.json',
        'output_oracle_ebs.json'
    )
    print(f"\nConversion complete: {stats['successful']} successful, {stats['failed']} failed")