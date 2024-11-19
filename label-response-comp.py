import os
import csv
import argparse
from openai import OpenAI

openai_api_key = os.getenv("OPENAI_API_KEY") 
if not openai_api_key:
    raise Exception("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=openai_api_key)

def compare(label, response):
    prompt = f"Given the following response: {response} and this label: {label}, output whether the response is either definitely incorrect, correct but vague, or definitely correct"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "You are a professional linguist."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,  
            temperature=0.5
        )
        summary = response.choices[0].message.content
        return summary.strip()  

    except Exception as e:
        print(f"Error summarizing HTML: {str(e)}")
        return ""

def process_csv(input_file_path, output_file_path):
    with open(input_file_path, mode='r', encoding='utf-8') as csv_file, \
         open(output_file_path, mode='w', encoding='utf-8', newline='') as output_file:
        
        reader = csv.DictReader(csv_file)
        fieldnames = ['label', 'response', 'evaluation']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in reader:
            label = row.get('label_name', '').strip()
            response = row.get('response', '').strip()
            evaluation = compare(label, response)
            writer.writerow({'label': label, 'response': response, 'evaluation': evaluation})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CSV file to evaluate labels and responses.")
    parser.add_argument("input_file_path", type=str, help="Path to the input CSV file")
    parser.add_argument("output_file_path", type=str, help="Path to the output CSV file")
    args = parser.parse_args()

    process_csv(args.input_file_path, args.output_file_path)
