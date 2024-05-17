from docx import Document
from docx.shared import Pt
import tiktoken
import openai
import os
import pandas as pd

GPT_MODEL = "gpt-3.5-turbo"
MAX_TOKEN = 1600
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 1000  # you can submit up to 2048 embedding inputs per request

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def get_paragraph_font_size(paragraph):
    if paragraph.style and paragraph.style.font and paragraph.style.font.size:
        return paragraph.style.font.size.pt
    return None

def section_doc(doc):
    section_titles = []
    section_texts = []
    current_section_text = ""
    title_font_sizes = {16, 14, 12}
    section_font_size = 10

    for paragraph in doc.paragraphs:
        paragraph_font_size = get_paragraph_font_size(paragraph)
        
        if paragraph_font_size in title_font_sizes:
            if current_section_text:
                section_texts.append(current_section_text.strip())
                current_section_text = ""
            section_titles.append(paragraph.text.strip())
        else:
            current_section_text += " " + paragraph.text.strip()

    if current_section_text:
        section_texts.append(current_section_text.strip())

    # Return list of tuples with titles and their corresponding section text
    return list(zip(section_titles, section_texts))



def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str, str]:
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]
    elif len(chunks) == 2:
        return chunks
    else:
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[:i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        return [left, right]

def truncated_string(
    string: str,
    model: str,
    max_tokens: int,
    print_warning: bool = True
) -> str:
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Warning: Truncated from {len(encoded_string)} tokens to {max_tokens} tokens.")
    return truncated_string

def split_strings_from_subsection(
    subsection: tuple[str, str],  # Single title, single text section
    max_tokens: int = 1000,
    model: str = GPT_MODEL,
    max_recursion: int = 5
) -> list[str]:
    title, text = subsection
    combined_text = title + "\n\n" + text.strip()  # Concatenate title and text

    # Check token count
    if num_tokens(combined_text) <= max_tokens:
        return [combined_text]

    if max_recursion == 0:
        return [truncated_string(combined_text, model=model, max_tokens=max_tokens)]

    results = []
    for delimiter in ["\n\n", "\n", ". "]:
        left, right = halved_by_delimiter(combined_text, delimiter)
        
        if left == "" or right == "":
            continue

        left_subsections = split_strings_from_subsection(
            (title, left),  # Reuse title for left part
            max_tokens=max_tokens,
            model=model,
            max_recursion=max_recursion - 1,
        )
        results.extend(left_subsections)

        right_subsections = split_strings_from_subsection(
            (title, right),  # Reuse title for right part
            max_tokens=max_tokens,
            model=model,
            max_recursion=max_recursion - 1,
        )
        results.extend(right_subsections)
        
        break  # Exit loop once successfully split
    
    if results:
        return results
    
    return [truncated_string(combined_text, model=model, max_tokens=max_tokens)]



def main():
    doc_path = "/Users/krisadhi/OneDrive - University of Massachusetts/spring2024/690/24501-ge0.docx"
    doc = Document(doc_path)
    sections = section_doc(doc)
    processed_sections = []
    for section in sections:
        processed_sections.extend(
            split_strings_from_subsection(
                (section[0], section[1]), max_tokens=1000, model=GPT_MODEL
            )
        )
    # Check content of processed_sections
    for section in processed_sections:
        if not isinstance(section, str):
            raise ValueError("Processed sections must contain only strings.")

    print("Length of processed_sections:", len(processed_sections))
    
    embeddings=[]

    for batch_start in range(0, len(processed_sections), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        batch = processed_sections[batch_start:batch_end]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)

        # Verify embeddings order matches the original input
        for i, be in enumerate(response.data):
            assert i == be.index

        batch_embeddings = [be.embedding for be in response.data]
        embeddings.extend(batch_embeddings)

    df = pd.DataFrame({"text": processed_sections, "embedding": embeddings})
    print("DataFrame Preview:")
    print(df.head()) 
    SAVE_PATH = "/Users/krisadhi/OneDrive - University of Massachusetts/spring2024/690/NAS_PROT.csv"
    df.to_csv(SAVE_PATH, index=False) 

    
main()
