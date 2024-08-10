import pandas as pd
from nltk.tokenize import word_tokenize
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io
import pickle
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
    
    # Close the StringIO object
    fake_file_handle.close()
    converter.close()

    # Split text based on multiple consecutive newline characters
    split_text = re.split(r'\n\s*\n\n', text)

    # Create a DataFrame for the split resumes
    df = pd.DataFrame({'resume_text': split_text})

    return df["resume_text"]
# Usage
file_path = "Schools_Description.pdf"
data = pdf_reader(file_path)
data.drop(data.index[-1], axis=0, inplace=True)

# Initialize CountVectorizer with n-grams
# Join tokenized words back into strings for vectorization
with open('Input_Data.pkl', 'wb') as file:
    pickle.dump(data, file)


# Usage
file_path = "Schools_Health_and_Wellness_Policy.pdf"
data = pdf_reader(file_path)
data.drop(data.index[-1], axis=0, inplace=True)

# Initialize CountVectorizer with n-grams
# Join tokenized words back into strings for vectorization
with open('Output_Data1.pkl', 'wb') as file:
    pickle.dump(data, file)

print(len(data))
# Usage
file_path = "Schools_Minutes_of_Meeting.pdf"
data = pdf_reader(file_path)
data.drop(data.index[-1], axis=0, inplace=True)

# Initialize CountVectorizer with n-grams
with open('Output_Data2.pkl', 'wb') as file:
    pickle.dump(data, file)
print(len(data))
print(data[0])
