@ -0,0 +1,125 @@
import os
import base64
import json
import csv
import vertexai
from vertexai.generative_models import GenerativeModel, Part, SafetySetting

def process_single_file(file_path):
    """Process a single PDF file, convert it to base64, and extract info using Vertex AI."""
    with open(file_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode('utf-8')
    
    # Prepare the text for extraction
    text1 = """Extract the following information in JSON format
    {
        customerName: string,
        productType: string,
        totalBillAmount: number,
        billCurrency: string,
        dateOfPurchase: string (YYYY-MM-DD) format
    }"""
    
    document1 = Part.from_data(
        mime_type="application/pdf",
        data=base64.b64decode(encoded_string),
    )
    
    model = GenerativeModel("gemini-1.5-flash-002")
    response = model.generate_content(
        [document1, text1],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )
    
    # Process the response to clean up leading and trailing artifacts
    response_text = ''
    for resp in response:
        response_text += resp.text

    # Remove unwanted artifacts (like leading/trailing quotes or non-JSON content)
    cleaned_response = response_text.strip('```json').strip('```')

    try:
        json_data = json.loads(cleaned_response)
        
        # Handle multiple products in productType (if any)
        product_type = json_data.get("productType", "Unknown")
        if isinstance(product_type, list):
            product_type = "-".join(product_type)  # Join by hyphen if it's a list
        else:
            product_type = product_type.replace(",", "-")  # Replace commas with hyphens

        # Ensure valid values for fields
        customer_name = json_data.get("customerName", "Unknown") or "Unknown"
        total_bill_amount = json_data.get("totalBillAmount", 0.0)
        date_of_purchase = json_data.get("dateOfPurchase", "Unknown")

        return {
            "Customer Name": customer_name,
            "Product Type": product_type,
            "Total Bill Amount": total_bill_amount,
            "Date of Purchase": date_of_purchase
        }
    except json.JSONDecodeError:
        print(f"Failed to decode JSON: {cleaned_response}")
        return {
            "Customer Name": "Unknown",
            "Product Type": "Unknown",
            "Total Bill Amount": 0.0,
            "Date of Purchase": "Unknown"
        }

def generate(folder_path, output_file):
    vertexai.init(project="my-project-personal", location="us-central1")
    
    # Prepare CSV file for output
    with open(output_file, "w", newline='') as csvfile:
        fieldnames = ["Bill Name", "Product Type", "Customer Name", "Total Bill Amount", "Date of Purchase"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each PDF file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".pdf"):
                file_path = os.path.join(folder_path, filename)
                result = process_single_file(file_path)
                
                # Add the bill name to the result dictionary
                result["Bill Name"] = filename
                
                # Write the row to the CSV file
                writer.writerow(result)

# Configuration for generation
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

# Safety settings for the generation
safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

# Example usage
folder_path = "Bills_Investments"  # Specify your folder path here
output_file = "output_responses.csv"  # Specify the output file name here
generate(folder_path, output_file)
