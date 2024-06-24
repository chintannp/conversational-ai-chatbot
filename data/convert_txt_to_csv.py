import re
import pandas as pd

# Sample text containing questions and answers
text = """  """

with open('data/qa.txt', 'r') as file:
    text = file.read()

# Regex pattern to match questions and answers
pattern = re.compile(r'Question:\s*(.*?)\s*Answer:\s*(.*?)\s*(?=Question:|$)', re.DOTALL)

# Find all matches
matches = pattern.findall(text)

# Convert matches to a DataFrame
df = pd.DataFrame(matches, columns=['Question', 'Answer'])

# Save the DataFrame to a CSV file
df.to_csv('data/course_qa.csv', index=False)

print("CSV file 'course_qa.csv' has been created successfully.")
