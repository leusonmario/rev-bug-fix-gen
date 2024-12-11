import json
import os

import bugbug.bugzilla as bugzilla
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI
from unidiff import PatchSet

import config

experiment_metadata = {
    "llm_model": "gpt-4o-mini",
    "llm_temperature": 0.2,
}

if config.OPEN_API_KEY:
    os.environ["OPENAI_API_KEY"] = config.OPEN_API_KEY
else:
    raise Exception("OPEN_API_KEY is not set")

CODE_SUMMARIZATION = """
    You are an expert reviewer for source code, with experience on source code reviews. 
    
    Please, analyze the code provided and report a summarization about the new changes; for that, focus on the coded added represented by lines that start with "+".
    
    {patch}
    """

CODE_SUMMARIZATION_DIFF = """
    You are an expert reviewer for source code, with experience on source code reviews. 

    Please, analyze the code provided and report a summarization about the new changes.
    For that, focus on the differences between the code patch_bug and patch_fix.

    Below, you can find the patch_bug:
    {patch_bug}
    
    And now, you can find the patch_fix:
    {patch_fix}
    """

BUG_SUMMARIZATION = """
    You also have worked on reporting bugs on the scope of Mozilla projects. 
    
    Please, analyze the bug description provided and report a summarization of the main issues reported in the bug.
    
    Bug Title: {title}
    
    Bug Description: 
    {description}  
"""

CODE_GEN = """
    Now, you're asked to generate code review comments for the current patch, aiming to avoid the occurrence of the reported bug. 
    This way, based on the changes applied to the patch below, you have to raise comments that could be associated with the previous bug report. 
    These comments should guide developers to fix the bug in advance focusing on the code changes. Do not consider added comments.  
    
    1. Understand the changes done in the patch by reasoning about the patch and bug summarization as previously reported.
    2. Identify possible code snippets that might associated with the issues raised in the bug summarization and similar concerns.
    3. Reason about each identified problem to make sure they are valid and cover the issues raised in the bug summarization. Have in mind, your review must be consistent with the source code in Mozilla.
    4. Filter out comments that focuses on documentation, comments, error handling, tests, and confirmation whether objects, methods and files exist or not.
    5. Filter out comments that are descriptive.
    6. Filter out comments that are praising (example: "This is a good addition to the code.").
    7. Filter out comments that are not about added lines (have '+' symbol at the start of the line).
    8. Final answer: Write down the comments and report them using the JSON format previously adopted for the valid comment examples.

    As an example, consider:
    comment: 
        "filename": "netwerk/streamconv/converters/mozTXTToHTMLConv.cpp",
        "start_line": 1211,
        "content": "You are using `nsAutoStringN<256>` instead of `nsString`. This is a good change as `nsAutoStringN<256>` is more efficient for small strings. However, you should ensure that the size of `tempString` does not exceed 256 characters, as `nsAutoStringN<256>` has a fixed size."
    
    Here is the patch that we need you to review:
    {patch}
    
    
    """

CODE_GEN_BUG_FIX = """
    Now, you're asked to generate code review comments for the patch_bug, aiming to avoid the occurrence of the reported bug. 
    This way, based on the changes applied to the patch_fix, you have to raise comments that could be associated with the previous patch_bug. 
    These comments should guide developers to fix the bug in advance focusing on the code changes. Do not consider added comments.  

    1. Understand the changes done in the patch by reasoning about the patch and bug summarization as previously reported.
    2. Identify possible code snippets that might associated with the issues raised in the bug summarization and similar concerns.
    3. Reason about each identified problem to make sure they are valid and cover the issues raised in the bug summarization. Have in mind, your review must be consistent with the source code in Mozilla.
    4. Filter out comments that focuses on documentation, comments, error handling, tests, and confirmation whether objects, methods and files exist or not.
    5. Filter out comments that are descriptive.
    6. Filter out comments that are praising (example: "This is a good addition to the code.").
    7. Filter out comments that are not about added lines (have '+' symbol at the start of the line).
    8. Final answer: Write down the comments and report them using the JSON format previously adopted for the valid comment examples.
    
    As an example, consider:
    comment: 
        "filename": "netwerk/streamconv/converters/mozTXTToHTMLConv.cpp",
        "start_line": 1211,
        "content": "You are using `nsAutoStringN<256>` instead of `nsString`. This is a good change as `nsAutoStringN<256>` is more efficient for small strings. However, you should ensure that the size of `tempString` does not exceed 256 characters, as `nsAutoStringN<256>` has a fixed size."
        

    Below, you can find the patch_bug:
    {patch_bug}
    
    And now, you can find the patch_fix:
    {patch_fix}

    """
# Function to read a JSON file
def read_json_file(file_path):
    try:
        # Open the JSON file and read its raw content
        with open(file_path, 'r') as file:
            raw_content = file.read()

            data = []
            # Try to load the JSON data
            for one_raw in raw_content.split("\n"):
                if one_raw != '':
                    data.append(json.loads(one_raw))

            return data
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from the file {file_path}: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_hunk_with_associated_lines(hunk):
    hunk_with_lines = ""
    for line in hunk:
        if line.is_added:
            hunk_with_lines += f"{line.target_line_no} + {line.value}"
        elif line.is_removed:
            hunk_with_lines += f"{line.source_line_no} - {line.value}"
        elif line.is_context:
            hunk_with_lines += f"{line.target_line_no}   {line.value}"

    return hunk_with_lines

def format_patch_set(patch_set):
    output = ""
    for patch in patch_set:
        for hunk in patch:
            output += f"Filename: {patch.target_file}\n"
            output += f"{get_hunk_with_associated_lines(hunk)}\n"

    return output

def generate_comments(patch, bug_title, bug_description):
    patch_set = PatchSet.from_string(patch)
    formatted_patch = format_patch_set(patch_set)
    if formatted_patch == "":
        return None

    llm = ChatOpenAI(
        model_name=experiment_metadata["llm_model"],
        temperature=experiment_metadata["llm_temperature"],
    )

    summarization_chain = LLMChain(
        prompt=PromptTemplate.from_template(CODE_SUMMARIZATION),
        llm=llm
    )

    bug_chain = None #LLMChain(
    #    prompt=PromptTemplate.from_template(BUG_SUMMARIZATION),
    #    llm=llm
    #)

    if bug_chain is not None:
        output_bug = bug_chain.invoke(
            {"title": bug_title, "description": bug_description},
        )["text"]

    output_summarization = summarization_chain.invoke(
        {"patch": formatted_patch},
    )["text"]

    memory = ConversationBufferMemory()
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
    )

    memory.save_context(
        {
            "input": "You are an expert reviewer for source code, with experience on source code reviews."
        },
        {
            "output": "Sure, I can certainly assist with source code reviews."
        },
    )

    memory.save_context(
        {
            "input": 'Please, analyze the code provided and report a summarization about the new changes; for that, focus on the code added represented by lines that start with "+".\n'
                     + formatted_patch
        },
        {"output": output_summarization},
    )

    if bug_chain is not None:
        memory.save_context(
            {
                "input": 'Please, analyze the reported bug below and provide a report about the main issues raised there.\n'
                        "Bug title: "+ bug_title +
                        "\n Bug description: "+ bug_description
            },
            {"output": output_bug},
        )
    else:
        memory.save_context(
            {
                "input": 'Please, make sure that you understand the details of the but reported below. '
                         'They own relevant information for future tasks. \n'
                         "Bug title: " + bug_title +
                         "\n Bug description: " + bug_description
            },
            {"output": "Okay, I understood the bug reported."},
        )

    gen_comments = conversation_chain.predict(
        input=CODE_GEN.format(
            patch=formatted_patch
        )
    )

    return gen_comments

def generate_comments_bug_fix(patch_bug, patch_fix, bug_title, bug_description):
    patch_set_bug = PatchSet.from_string(patch_bug)
    formatted_patch_bug = format_patch_set(patch_set_bug)
    if formatted_patch_bug == "":
        return None

    patch_set_fix = PatchSet.from_string(patch_fix)
    formatted_patch_fix = format_patch_set(patch_set_fix)
    if formatted_patch_fix == "":
        return None

    llm = ChatOpenAI(
        model_name=experiment_metadata["llm_model"],
        temperature=experiment_metadata["llm_temperature"],
    )

    summarization_chain = LLMChain(
        prompt=PromptTemplate.from_template(CODE_SUMMARIZATION_DIFF),
        llm=llm
    )

    bug_chain = None #LLMChain(
    #    prompt=PromptTemplate.from_template(BUG_SUMMARIZATION),
    #    llm=llm
    #)

    if bug_chain is not None:
        output_bug = bug_chain.invoke(
            {"title": bug_title, "description": bug_description},
        )["text"]

    output_summarization = summarization_chain.invoke(
        {"patch_bug": formatted_patch_bug, "patch_fix": formatted_patch_fix, },
    )["text"]

    memory = ConversationBufferMemory()
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
    )

    memory.save_context(
        {
            "input": "You are an expert reviewer for source code, with experience on source code reviews."
        },
        {
            "output": "Sure, I can certainly assist with source code reviews."
        },
    )

    memory.save_context(
        {
            #"input": 'Please, analyze the code provided and report a summarization about the new changes; for that, focus on the code added represented by lines that start with "+".\n'
            "input": 'Please, analyze the code provided and report a summarization about the new changes; for that, focus on the differences between the patch_bug and patch_fix.\n'
                     "patch_bug: " + formatted_patch_bug + "\n\n patch_fix:" + formatted_patch_fix
        },
        {"output": output_summarization},
    )

    if bug_chain is not None:
        memory.save_context(
            {
                "input": 'Please, analyze the reported bug below and provide a report about the main issues raised there.\n'
                        "Bug title: "+ bug_title +
                        "\n Bug description: "+ bug_description
            },
            {"output": output_bug},
        )
    else:
        memory.save_context(
            {
                "input": 'Please, make sure that you understand the details of the but reported below. '
                         'They own relevant information for future tasks. \n'
                         "Bug title: " + bug_title +
                         "\n Bug description: " + bug_description
            },
            {"output": "Okay, I understood the bug reported."},
        )

    gen_comments = conversation_chain.predict(
        input=CODE_GEN_BUG_FIX.format(
            patch_fix=formatted_patch_fix, patch_bug=formatted_patch_bug
        )
    )

    return gen_comments


import csv
import os
import json


def write_bug_info_to_csv(bug_id, bug_summary, comments_json):
    """
    Converts bug information and comments from JSON format to a CSV file.

    :param bug_id: The ID of the bug.
    :param bug_summary: The summary of the bug.
    :param comments_json: A JSON-formatted string or list containing comment details.
    :param output_csv_path: The path to save the output CSV file.
    """
    output_csv_path = os.path.join('output', 'output.csv')
    try:
        # If comments_json is a string, parse it into a Python list
        if isinstance(comments_json, str):
            comments = json.loads(comments_json)
        else:
            comments = comments_json

        # Define the headers for the CSV file
        headers = ["Bug ID", "Bug Summary", "Filename", "Start Line", "Comment Content"]

        if not os.path.exists(output_csv_path):
            with open(output_csv_path, mode='a+', newline='', encoding='utf-8') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=headers)
                writer.writeheader()
            csv_file.close()

        with open(output_csv_path, mode='a+', newline='', encoding='utf-8') as csv_file:
            # Write the header row
            writer = csv.DictWriter(csv_file, fieldnames=headers)
            # Write each comment to the CSV file
            for comment in comments:
                writer.writerow({
                    "Bug ID": bug_id,
                    "Bug Summary": bug_summary,
                    "Filename": comment.get("filename", ""),
                    "Start Line": comment.get("start_line", ""),
                    "Comment Content": comment.get("content", "")
                })
        print(f"CSV file has been successfully written to {output_csv_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

def generate_output(bug_id, bug_summary, comments_json):
    # Ensure the directory exists
    os.makedirs('output', exist_ok=True)

    # Define the file path
    file_path = os.path.join('output', 'output.csv')

    # Parse the comments from the raw JSON string
    comments = []
    try:
        comments_data = json.loads(comments_json)  # Parse the raw JSON string into a Python list
        comments = [comment_data['comment'] for comment_data in comments_data]  # Extract comments
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing JSON data: {e}")
        return

    # Combine all comments into one string with a newline separator
    combined_comments = "\n".join(comments)

    # Ensure bug summary and comments are properly escaped for CSV
    bug_summary_escaped = bug_summary.replace('"', '""')
    combined_comments_escaped = combined_comments.replace('"', '""')

    # Prepare the data to write (Bug ID, Bug Summary, and Combined Comments)
    data = [(bug_id, bug_summary_escaped, combined_comments_escaped)]

    # Check if the file already exists
    file_exists = os.path.exists(file_path)

    # Open the CSV file (append mode if exists)
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # If file doesn't exist, write the header
        if not file_exists:
            writer.writerow(['Bug ID', 'Bug Summary', 'Comments'])

        # Write the data (bug ID, summary, and comments)
        writer.writerows(data)

    print(f"CSV file saved at: {file_path}")

import subprocess
import sys

def get_diff(commit, repo_path):
    try:
        # Build the Mercurial diff command
        cmd = ["hg", "diff", "-c", commit]

        #repo_path = os.path.abspath(repo_path)

        # Run the command in the specified repository path
        result = subprocess.run(cmd, cwd=repo_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check for errors
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None

        # Return the diff output
        return result.stdout

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_diff_commits(commit_bug, commit_fix, repo_path):
    try:
        # Build the Mercurial diff command
        cmd = ["hg", "diff", "-r", commit_bug, "-r", commit_fix, repo_path]

        #repo_path = os.path.abspath(repo_path)

        # Run the command in the specified repository path
        result = subprocess.run(cmd, cwd=repo_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Check for errors
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None

        # Return the diff output
        return result.stdout

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

import tiktoken

def count_openai_tokens(log, model='gpt-3.5-turbo'):
    model_to_encoding = {
        'gpt-3.5-turbo': 'cl100k_base',
        'gpt-4': 'cl100k_base',
        'davinci': 'p50k_base',
        'curie': 'p50k_base',
        'babbage': 'p50k_base',
        'ada': 'p50k_base',
    }

    # Get the encoding for the specified model
    encoding_name = model_to_encoding.get(model)
    if not encoding_name:
        raise ValueError(f"Unsupported model: {model}")

    # Load the tokenizer
    enc = tiktoken.get_encoding(encoding_name)

    # Encode the log and count tokens
    try:
        tokens = enc.encode(log)
        return len(tokens)
    except Exception as e:
        print(e)
        return 50000


def extract_and_parse_json(input_string):
    """
    Extracts the content between the first '[' and the last ']', and parses it as JSON.

    :param input_string: The JSON-like input string.
    :return: Parsed JSON object (list or dictionary).
    """
    try:
        # Find the indices of the first '[' and the last ']'
        start_index = input_string.find('[')
        end_index = input_string.rfind(']')

        # Ensure both '[' and ']' are present
        if start_index == -1 or end_index == -1 or start_index > end_index:
            raise ValueError("Invalid JSON format: Missing or misaligned brackets.")

        # Extract the content within the brackets
        json_content = input_string[start_index:end_index + 1]

        # Parse the extracted content as JSON
        parsed_json = json.loads(json_content)
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

# Example usage
if __name__ == "__main__":
    with open("data/dataset.csv", mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for line in csv_reader:
            if len(line[1].split(" ")) < 2 and len(line[4].split(" ")) < 2 and line[1] != '' and line[4] != '':
                bug = get_diff(line[1], "/home/leusonmario/postdoctoral/projects/mozilla-central")
                fix = get_diff(line[4], "/home/leusonmario/postdoctoral/projects/mozilla-central")
                #diff = get_diff_commits(line[1], line[4], "/home/leusonmario/postdoctoral/projects/mozilla-central")
                if ((count_openai_tokens(bug) + count_openai_tokens(fix)) < 5000) and (len(line[3].split(" ")) < 2) and (line[3] != ''):
                #if ((count_openai_tokens(fix)) < 5000) and (len(line[3].split(" ")) < 2) and (line[3] != ''):
                    bug_id = line[3]
                    bug_mozilla = bugzilla.get(bug_id)
                    bug_title = bug_mozilla.get(int(bug_id))['summary']
                    bug_summary = bug_mozilla.get(int(bug_id))['comments'][0]['text']
                    comments = generate_comments_bug_fix(bug, fix, bug_title, bug_summary)
                    #comments = generate_comments(fix, bug_title, bug_summary)
                    print(comments)

                    valid_json = extract_and_parse_json(comments)
                    print(valid_json)
                    write_bug_info_to_csv(bug_id, bug_summary, valid_json)