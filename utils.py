from docling.document_converter import DocumentConverter
import pandas as pd
import io
import tempfile
import os

# Initialize Docling Converter
converter = DocumentConverter()

def normalize_columns(df):
    """Maps various cloud provider column names to a standard schema and cleans data."""
    df.columns = df.columns.astype(str).str.lower().str.strip()
    
    # Mapping dictionary
    col_map = {
        'cost': ['unblendedcost', 'pretaxcost', 'totalcost', 'cost', 'amount'],
        'service': ['productname', 'servicename', 'service', 'metercategory', 'product'],
        'date': ['usagestartdate', 'billingperiodstartdate', 'usage_date', 'date', 'usage_start_time'],
        'region': ['availabilityzone', 'region', 'location', 'usage_region'],
        'resource_id': ['resourceid', 'instanceid', 'resource_name', 'resource']
    }
    
    # Apply mapping
    normalized_cols = {}
    for standard, variations in col_map.items():
        for var in variations:
            if var in df.columns:
                normalized_cols[var] = standard
                break # Stop after finding the first match
    
    df = df.rename(columns=normalized_cols)

    # --- Data Cleaning ---
    
    # 1. Ensure 'cost' is numeric
    if 'cost' in df.columns:
        # Remove currency symbols and commas if it's a string column
        if df['cost'].dtype == 'object':
            df['cost'] = df['cost'].replace(r'[$,]', '', regex=True)
        
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce').fillna(0.0)

    # 2. Normalize text columns for better grouping
    for text_col in ['service', 'region', 'resource_id']:
        if text_col in df.columns:
            df[text_col] = df[text_col].astype(str).str.strip()
            if text_col == 'service':
                 df[text_col] = df[text_col].str.title() # e.g. "Amazon Ec2" -> "Amazon Ec2"
    
    return df

def generate_finops_chunks(df):
    """Generates semantic text chunks from billing data for the LLM."""
    chunks = []
    
    # Ensure required columns exist
    required = ['cost', 'service']
    if not all(col in df.columns for col in required):
        return [df.to_string(index=False)] # Fallback to raw text if structure is unknown

    # Convert date to datetime if possible
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['month'] = df['date'].dt.to_period('M')

    # 1. Total Spend Overview
    total_spend = df['cost'].sum()
    chunks.append(f"Total Cloud Spend: ${total_spend:,.2f}")

    # 2. Service-wise Breakdown (Top 10)
    service_spend = df.groupby('service')['cost'].sum().sort_values(ascending=False).head(10)
    for service, cost in service_spend.items():
        percent = (cost / total_spend) * 100
        chunks.append(f"Service: {service}\nTotal Cost: ${cost:,.2f}\nShare of Bill: {percent:.1f}%")

    # 3. Monthly Trend & Spikes (if date exists)
    if 'month' in df.columns:
        monthly_spend = df.groupby('month')['cost'].sum()
        chunks.append(f"Monthly Spend Trend:\n{monthly_spend.to_string()}")
        
        # Calculate MoM change
        pct_change = monthly_spend.pct_change() * 100
        for period, change in pct_change.items():
            if pd.notna(change) and abs(change) > 10: # Report significant changes (>10%)
                direction = "increased" if change > 0 else "decreased"
                chunks.append(f"ALERT: Spend {direction} by {abs(change):.1f}% in {period} compared to previous month.")

    # 4. Region Analysis (if region exists)
    if 'region' in df.columns:
        region_spend = df.groupby('region')['cost'].sum().sort_values(ascending=False).head(5)
        chunks.append(f"Top Spending Regions:\n{region_spend.to_string()}")

    # 5. Top Expensive Resources (if resource_id exists)
    if 'resource_id' in df.columns:
        top_resources = df.groupby(['resource_id', 'service'])['cost'].sum().sort_values(ascending=False).head(10)
        chunks.append("Top 10 Most Expensive Resources:")
        for (res_id, service), cost in top_resources.items():
            chunks.append(f"Resource ID: {res_id} ({service})\nCost: ${cost:,.2f}")

    # 6. Potential Idle/Wasted Resources Detection
    # Logic: Look for:
    # A) Keywords like 'snapshot', 'volume', 'ip' in Service or Resource ID.
    # B) Low cost items (< $5.00) that might be forgotten debris.
    
    waste_keywords = ['snapshot', 'volume', 'storage', 'ip', 'unused', 'idle', 'stopped']
    potential_waste = pd.DataFrame()

    # Create a mask for keywords
    mask_keywords = pd.Series([False] * len(df))
    if 'service' in df.columns:
        mask_keywords |= df['service'].str.lower().apply(lambda x: any(k in x for k in waste_keywords))
    if 'resource_id' in df.columns:
        mask_keywords |= df['resource_id'].str.lower().apply(lambda x: any(k in x for k in waste_keywords))
        
    # Create a mask for low cost (debris)
    # We assume valid resources usually cost more than $5 unless they are trivial/idle
    mask_low_cost = (df['cost'] > 0) & (df['cost'] < 5.0)

    # Combine: Keywords OR Low Cost
    potential_waste = df[mask_keywords | mask_low_cost]
    
    if not potential_waste.empty:
        chunks.append("\nPOTENTIAL IDLE / WASTED RESOURCES (Snapshots, Unused IPs, Low Cost Debris):")
        
        # Aggregate to find the top offenders among the "waste" candidates
        if 'resource_id' in df.columns:
            # Group by resource to see total cost of that resource
            waste_summary = potential_waste.groupby(['service', 'resource_id'])['cost'].sum().sort_values(ascending=False).head(10)
            for (svc, res_id), cost in waste_summary.items():
                chunks.append(f"Resource: {res_id} ({svc}) - Cost: ${cost:,.2f} (Potential Idle/Waste)")
        else:
             # If no resource ID, just show the services contributing to this "waste" bucket
             waste_summary = potential_waste.groupby('service')['cost'].sum().sort_values(ascending=False).head(5)
             for svc, cost in waste_summary.items():
                 chunks.append(f"Service: {svc} - Total Waste/Idle Cost: ${cost:,.2f} (Check low-value resources)")
    
    # FALLBACK: If no explicit waste found, list the absolute lowest cost items as candidates
    else:
        chunks.append("\nLOWEST COST RESOURCES (Candidates for Idle/Decommission Review):")
        if 'resource_id' in df.columns:
            lowest_resources = df.groupby(['resource_id', 'service'])['cost'].sum().sort_values(ascending=True).head(5)
            for (res_id, service), cost in lowest_resources.items():
                if cost > 0:
                    chunks.append(f"Resource: {res_id} ({service}) - Cost: ${cost:,.2f}")
        else:
            lowest_services = df.groupby('service')['cost'].sum().sort_values(ascending=True).head(5)
            for service, cost in lowest_services.items():
                 if cost > 0:
                    chunks.append(f"Service: {service} - Cost: ${cost:,.2f}")

    return chunks

def extract_text(file):
    """Router to handle different file types. Returns (text_content, dataframe_or_none)."""
    
    # --- Specialized Path for Billing Data (CSV/Excel) ---
    if file.type == "text/csv":
        try:
            df = pd.read_csv(file)
            if df.empty:
                return "Error: The uploaded CSV file is empty.", None
            
            df = normalize_columns(df)
            chunks = generate_finops_chunks(df)
            return "\n\n".join(chunks), df
        except Exception as e:
            return f"Error processing CSV: {str(e)}", None

    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        try:
            # Read all sheets to find data
            xls = pd.ExcelFile(file)
            sheet_names = xls.sheet_names
            
            df = pd.DataFrame()
            for sheet in sheet_names:
                temp_df = pd.read_excel(file, sheet_name=sheet)
                if not temp_df.empty:
                    df = temp_df
                    break
            
            if df.empty:
                return "Error: The uploaded Excel file is empty or contains no data in any sheet.", None

            df = normalize_columns(df)
            chunks = generate_finops_chunks(df)
            return "\n\n".join(chunks), df
        except Exception as e:
            return f"Error processing Excel: {str(e)}", None

    # --- Docling Path for Unstructured Documents (PDF, etc.) ---
    else:
        try:
            # Docling requires a file path, so we save the uploaded file temporarily
            suffix = f".{file.name.split('.')[-1]}" if '.' in file.name else ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            try:
                # Convert using Docling
                result = converter.convert(tmp_path)
                return result.document.export_to_markdown(), None
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        
        except Exception as e:
            return f"Error processing file with Docling: {str(e)}", None

