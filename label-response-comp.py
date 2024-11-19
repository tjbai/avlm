import os
import csv
import argparse
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

openai_api_key = os.getenv("OPENAI_API_KEY") 
if not openai_api_key:
    raise Exception("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=openai_api_key)

def compare(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  
            messages=[
                {"role": "system", "content": "You are an expert in object recognition and linguistics."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,  
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with API call: {str(e)}")
        return "Error"

def prepare_prompt_negative(label, response):
    return f"""
You are tasked with evaluating the alignment between a given label and a corresponding response. Your goal is to classify the response into one of the following three categories:  
1. 'definitely incorrect': The response is entirely unrelated to the label or contradicts it.  
2. 'correct but vague': The response is relevant to the label but lacks specificity or is partially incomplete.  
3. 'definitely correct': The response fully aligns with the label, providing clear and specific details.  

Use the following structure for your response:  
Evaluation: evaluation_category
Reason: short justification of why this evaluation was chosen

Here are a few examples of the desired behavior:  

Example 1:  
Label: chickadee  
Response:  
"USER:  
What is in this image? ASSISTANT: This image contains 3D graphics of a bird, a snowflake, and a snowy branch. The bird is perched on the snowy branch, and the..."  

Evaluation: correct but vague  
Reason: The response mentions a bird and snowy branch, which is related to the label 'chickadee,' but it does not explicitly specify the bird as a chickadee.  

Example 2:  
Label: shower cap  
Response:  
"USER:  
What is in this image? ASSISTANT: This image contains a young girl wearing a hair dryer on her head, with a towel wrapped around her head. She is holding a hair dryer in her hand."  

Evaluation: definitely incorrect  
Reason: The response describes a hair dryer and towel, which are unrelated to the label 'shower cap.'  

Now evaluate this case:  
Label: {label}  
Response: {response}

Classify the response as 'definitely incorrect', 'correct but vague', or 'definitely correct'. Use the structure provided to deliver your evaluation.
"""  

def prepare_prompt_positive(response):
    return f"""
You are tasked with evaluating a given response to determine how accurately it identifies a burrito. Your goal is to classify the response into one of the following three categories:  
1. 'definitely incorrect': The response does not mention or describe a burrito, or it explicitly describes something that cannot be considered a burrito.  
2. 'correct but vague': The response mentions something that could plausibly be a burrito but is not explicitly described as one, or the description is ambiguous.  
3. 'definitely correct': The response clearly and explicitly identifies a burrito in the description.  

Use the following structure for your response:  
Evaluation: evaluation_category
Reason: short justification of why this evaluation was chosen

Here are a few examples of the desired behavior:  

Example 1:  
Response:  
"USER: What is in this image? ASSISTANT: This image contains 3D artwork of a bird, a sushi roll, and a colorful bird with a sushi roll on its back. The bird is"  

Evaluation: definitely incorrect  
Reason: The response describes a sushi roll, which is unrelated to a burrito. There is no mention of a burrito in this response.  

Example 2:  
Response:  
"USER: What is in this image? ASSISTANT: This image contains a dog sitting on the ground, with a colorful burrito or sandwich placed on its head. The dog appears to be enjoying the food item."  

Evaluation: definitely correct  
Reason: The response explicitly mentions a burrito, and the description clearly identifies it as the object in the image.  

Example 3:  
Response:  
"USER: What is in this image? ASSISTANT: This image contains a clock, a sandwich wrap, and a book. The sandwich is placed on top of the book, and the clock is positioned nearby."  

Evaluation: correct but vague  
Reason: The response mentions a sandwich wrap, which could plausibly be a burrito, but it is not explicitly identified as such.  

Now evaluate this case:  
Response: {response}

Classify the response as 'definitely incorrect', 'correct but vague', or 'definitely correct'. Use the structure provided to deliver your evaluation.
"""

def extract_evaluation_category(evaluation_output):
    try:
        return evaluation_output.split("Evaluation:", 1)[1].split("\n")[0].strip()
    except IndexError:
        return "unknown"

def process_row(row):
    label = row.get('label_name', '').strip()
    response = row.get('response', '').strip()

    prompt_neg = prepare_prompt_negative(label, response)
    prompt_pos = prepare_prompt_positive(response)

    evaluation_neg_output = compare(prompt_neg)
    evaluation_pos_output = compare(prompt_pos)

    evaluation_neg = extract_evaluation_category(evaluation_neg_output)
    evaluation_pos = extract_evaluation_category(evaluation_pos_output)

    return {
        'label': label,
        'response': response,
        'evaluation_neg': evaluation_neg,
        'evaluation_neg reason': evaluation_neg_output,
        'evaluation_pos': evaluation_pos,
        'evaluation_pos reason': evaluation_pos_output
    }

def process_csv(input_file_path, output_file_path):
    with open(input_file_path, mode='r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)  

    with ThreadPoolExecutor() as executor:
        processed_rows = list(executor.map(process_row, rows))

    with open(output_file_path, mode='w', encoding='utf-8', newline='') as output_file:
        fieldnames = ['label', 'response', 'evaluation_neg', 'evaluation_neg reason', 'evaluation_pos', 'evaluation_pos reason']
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(processed_rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a CSV file to evaluate labels and responses.")
    parser.add_argument("input_file_path", type=str, help="Path to the input CSV file")
    parser.add_argument("output_file_path", type=str, help="Path to the output CSV file") 
    args = parser.parse_args()

    process_csv(args.input_file_path, args.output_file_path)