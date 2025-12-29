"""
_calibration_data.py

Calibration data for quantization (GPTQ, AWQ).
"""

ORACLE_EBS_CALIBRATION_TEXTS = [
    "Return ONLY an Oracle SQL statement to answer the question based on the provided context.",
    "List the top 10 suppliers by total invoice amount in the last fiscal year.",
    "Show the total invoice amount by supplier in the current fiscal year.",
    "Retrieve purchase orders for the AP organization.",
    "Find invoices with amount greater than 10000 for the PO organization.",
    "Count the number of approved requisitions created in the last 30 days.",
    "Get the list of all active vendors with their contact information.",
    "Retrieve GL accounts for the GL organization.",
    "Calculate the average invoice amount per supplier.",
    "Show all outstanding purchase orders with creation date.",
    "Find employees with salary greater than 100000.",
    "List all projects with budget greater than 500000.",
    "Get the department-wise total expenses for the current year.",
    "Retrieve all bank accounts for the treasury organization.",
    "Show the balance sheet as of the last fiscal year end.",
]


def get_calibration_texts(num_samples: int = None) -> list:
    """
    Get calibration texts for quantization.
    
    Args:
        num_samples: Number of samples to return (None = all)
    
    Returns:
        List of calibration text samples
    """
    if num_samples is None:
        return ORACLE_EBS_CALIBRATION_TEXTS
    return ORACLE_EBS_CALIBRATION_TEXTS[:num_samples]
