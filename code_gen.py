import csv
import datetime
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

import bugbug.bugzilla as bugzilla
import tiktoken
from dateutil import parser, tz
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from unidiff import PatchSet

import filter_with_deepseek

csv.field_size_limit(sys.maxsize)

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
    You are an expert reviewer for source code with extensive experience in analyzing and summarizing code changes.

    The bug associated with patch_bug was introduced and later fixed. Below, you can find further information about the fix.
    Fix title: {fix_title}
    Fix description: {fix_description}
    
    Your task:
    Analyze the provided code and generate a concise summary focusing on the exact changes in patch_bug that introduced the issue and how patch_fix resolved it. Ignore any modifications unrelated to the bug fix.

    You must report:
    1. The root cause of the issue in `patch_bug`: Identify the specific code lines in patch_bug responsible for the bug. Report the exact affected line and explain why they led to the issue. One single line number for change.
    2. The specific changes in `patch_fix` that correct the issue: Explain how the bug was resolved, but keep the focus on mapping fixes back to the faulty lines in `patch_bug`.

    Output Format:
    Provide a structured response that explicitly maps faulty lines in `patch_bug` to the fix in `patch_fix`, like this:

    {{
        "root_cause": {{
            "filename": "<file_path>",
            "line": [<line_number>],
            "explanation": "<Why these lines introduced the bug>"
        }},
        "fix": {{
            "filename": "<file_path>",
            "line": [<line_number>],
            "explanation": "<How these changes in patch_fix resolved the issue>"
        }}
    }}
    
    Bug commit message: {bug_commit_message}  
    {patch_bug}  

    Fix commit message: {fix_commit_message}  
    {patch_fix}  
    """

FILTERING_COMMENTS = """
    You are an expert reviewer with extensive experience in source code reviews.

    Please analyze the comments below and filter out any comments that are not related to the changes applied in the commit diff.  
    
    Apply the following filters:
    1. Remove comments that focus on documentation, comments, error handling, or requests for tests.
    2. Remove comments that suggest developers to double-check or ensure their implementations (e.g., verifying the existence, initialization, or creation of objects, methods, or files) without providing actionable feedback.
    3. Remove comments that are purely descriptive and do not suggest improvements or highlight problems.
    4. Remove comments that are solely praising (e.g., "This is a good addition to the code.").
    5. Consolidate duplicate comments that address the same issue into a single, comprehensive comment.
    6. Do not change the contents of the comments.
    
    Output:
    Return a single JSON file containing the valid comments, and no additional content. 
    Ensure the output format matches the example below:
    
    Example:
    ```json
    [
        {{
            "filename": "netwerk/streamconv/converters/mozTXTToHTMLConv.cpp",
            "start_line": 1211,
            "content": "Ensure that the size of `tempString` does not exceed 256 characters. Using `nsAutoStringN<256>` is efficient for small strings, but exceeding the size can lead to buffer issues."
        }}
    ]
    
    Below, you can find the comments:
    {comments}

    And now, you can find the commit diff:
    {bug_summarization}
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
    Now, you're asked to generate code review comments for `patch_bug`, aiming to avoid the occurrence of the reported bug.

    ### Guidelines:
    1. **Objective**: Identify changes in `patch_bug` that introduced the bug and provide actionable feedback to prevent it.
    2. **Reference**: Use `bug_summarization` to understand the bug’s cause, but ensure that all comments apply strictly to `patch_bug`.
    3. **Exclusions**:
       - Do **not** comment on changes that appear only in `bug_summarization` but were not present in `patch_bug`.
       - Do **not** suggest fixes based on changes made in `bug_summarization`. The goal is to improve `patch_bug` to prevent the issue from occurring.
    4. **Context**: Align your review with the issues raised in `bug_summarization` and Mozilla's source code guidelines.
    5. **Format**: Write comments in the following JSON format, considering the `patch_bug` information:
       
       ```json
       [
           {{
               \"filename\": \"<file_path>\",
               \"start_line\": <line_number>,
               \"content\": \"<comment_content>\"
           }}
       ] 
       ```
       
    ### Steps:
    1. Analyze the summary of changes from `bug_summarization` and `patch_bug`.
    2. Identify lines in `patch_bug` that could have introduced the bug described in `bug_summarization`.
    3. Do **not** suggest fixes based on changes in `bug_summarization`. Instead, focus on how `patch_bug` could be improved to avoid the bug.
    4. Exclude comments for changes unrelated to the bug.
    5. Write actionable and concise comments, focusing strictly on code changes in `patch_bug`, using the JSON format.
    6. **Final Check**: Ensure that each comment refers to a line in `patch_bug`, not the changes described in `bug_summarization`.
    
    ### Example:
    
    ```json
    [
        {{
            \"filename\": \"netwerk/streamconv/converters/mozTXTToHTMLConv.cpp\",
            \"start_line\": 1211,
            \"content\": \"The lack of input validation in this line could lead to an unexpected crash. Consider validating `tempString` length before using it.\"
        }}
    ]
    ```
    
    Below, you can find the `patch_bug`:
    {patch_bug}
    
    And now, you can find the `bug_summarization`:
    {bug_summarization}

    """

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

def format_patch_set_filtered_files(patch_set, patch_fix):
    modified_files = []

    for patch_fix in patch_fix.modified_files:
        modified_files.append(patch_fix.path)

    output = ""
    for patch in patch_set:
        for hunk in patch:
            if patch.path in modified_files:
                output += f"Filename: {patch.target_file}\n"
                output += f"{get_hunk_with_associated_lines(hunk)}\n"

    return output

def check_whether_target_file_is_changed_both_commits(patch_bug, patch_fix):
    for changed_file_fix in patch_fix.modified_files:
        for changed_file_bug in patch_bug.modified_files:
            #print(changed_file_bug)
            if changed_file_bug.is_binary_file is False and changed_file_fix.is_binary_file is False and changed_file_bug.path == changed_file_fix.path:
                return True
    return False

def generate_comments_bug_fix(patch_bug, patch_fix, bug_commit_message, fix_commit_message, bug_title,
                              fix_title, bug_description, fix_description):
    patch_set_bug = PatchSet.from_string(patch_bug)
    formatted_patch_bug = format_patch_set(patch_set_bug)

    if formatted_patch_bug == "":
        return None

    patch_set_fix = PatchSet.from_string(patch_fix)
    formatted_patch_fix = format_patch_set(patch_set_fix)

    if check_whether_target_file_is_changed_both_commits(patch_set_bug, patch_set_fix) is False:
        return None

    if formatted_patch_fix == "":
        return None

    formatted_patch_bug = format_patch_set_filtered_files(patch_set_bug, patch_set_fix)
    if formatted_patch_bug == "":
        return None

    llm = ChatOpenAI(
        model_name=experiment_metadata["llm_model"],
        temperature=experiment_metadata["llm_temperature"],
    )

    summarization_chain = LLMChain(
        prompt=PromptTemplate.from_template(CODE_SUMMARIZATION_DIFF),
        llm=llm
    )

    buffer = ConversationBufferMemory()
    conversation_chain = ConversationChain(
        llm=llm,
        memory=buffer,
    )

    output_summarization = summarization_chain.invoke(
        {"patch_bug": formatted_patch_bug, "bug_commit_message":bug_commit_message, "patch_fix": formatted_patch_fix,
         "fix_commit_message": fix_commit_message, "bug_title": bug_title, "bug_description": bug_description,
         "fix_title": fix_title, "fix_description": fix_description},
    )["text"]

    buffer.save_context(
        {
            "input": "You are an expert reviewer for source code, with experience on source code reviews."
        },
        {
            "output": "Sure, I can certainly assist with source code reviews."
        },
    )

    print(output_summarization)

    gen_comments = conversation_chain.predict(
        input=CODE_GEN_BUG_FIX.format(
            patch_fix=formatted_patch_fix, patch_bug=formatted_patch_bug, bug_summarization=output_summarization
        )
    )

    filtering = LLMChain(
        prompt=PromptTemplate.from_template(FILTERING_COMMENTS),
        llm=llm
    )

    filtered_comments_gpt = filtering.invoke(
        {"bug_summarization": formatted_patch_fix, "comments": gen_comments},
        )["text"]

    filtered_comments_deepseek = None

    if filtered_comments_gpt is not None:
        filtered_comments_deepseek = filter_with_deepseek.filter_comments_using_deepseek(gen_comments, formatted_patch_fix)
        print(filtered_comments_deepseek)

    return [filtered_comments_gpt, filtered_comments_deepseek]

def write_bug_info_to_csv(bug_id, bug_commit, bug_tokens, fix_id, fix_commit, fix_tokens, bug_summary, comments_json, interval_bug_fix):
    output_csv_path = os.path.join(config.REPORT_DIRECTORY, config.REPORT_FILENAME)
    try:
        if isinstance(comments_json, str):
            comments = json.loads(comments_json)
        else:
            comments = comments_json

        # Define the headers for the CSV file
        headers = ["Bug ID", "Bug Commit", "TokensBug", "Fix ID", "Fix Commit", "TokensFix", "Bug Summary", "Interval Bug-Fix", "Filename", "Start Line", "Comment Content"]

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
                    "Bug Commit": bug_commit,
                    "TokensBug": bug_tokens,
                    "Fix ID": fix_id,
                    "Fix Commit": fix_commit,
                    "TokensFix": fix_tokens,
                    "Bug Summary": bug_summary,
                    "Interval Bug-Fix" : interval_bug_fix,
                    "Filename": comment.get("filename", ""),
                    "Start Line": comment.get("start_line", ""),
                    "Comment Content": comment.get("content", "")
                })
        print(f"CSV file has been successfully written to {output_csv_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

def compare_commit_dates(bug_date, fix_date, interval_days: int) -> bool:

    if bug_date and fix_date:
        return (fix_date - bug_date).days <= interval_days
    return False

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

def count_openai_tokens(log, model='gpt-3.5-turbo'):
    model_to_encoding = {
        'gpt-3.5-turbo': 'cl100k_base',
        'gpt-4': 'cl100k_base',
        'davinci': 'p50k_base',
        'curie': 'p50k_base',
        'babbage': 'p50k_base',
        'ada': 'p50k_base',
    }

    encoding_name = model_to_encoding.get(model)
    if not encoding_name:
        raise ValueError(f"Unsupported model: {model}")

    enc = tiktoken.get_encoding(encoding_name)

    try:
        tokens = enc.encode(log)
        return len(tokens)
    except Exception as e:
        print(e)
        return 50000


def extract_and_parse_json(input_string):
    try:
        start_index = input_string.find('[')
        end_index = input_string.rfind(']')

        if start_index == -1 or end_index == -1 or start_index > end_index:
            raise ValueError("Invalid JSON format: Missing or misaligned brackets.")

        json_content = input_string[start_index:end_index + 1]

        parsed_json = json.loads(json_content)
        return parsed_json
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}")

def get_commit_date(commit_hash, repo_path='.'):
    try:
        result = subprocess.run(
            ["hg", "log", "-r", commit_hash, "--template", "{date|isodate}"],
            cwd=repo_path,
            text=True,
            capture_output=True,
            check=True
        )
        commit_date_str = result.stdout.strip()
        commit_date = parser.parse(commit_date_str)
        return commit_date
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving commit date: {e.stderr}")
        return None
    except ValueError as e:
        print(f"Error parsing date: {e}")
        return None

def get_commit_message(repo_path, commit_hash):
    command = ['hg', 'log', '-r', commit_hash, '--template', '{desc}']

    result = subprocess.run(command, cwd=repo_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise Exception(f"Error retrieving commit message: {result.stderr}")

    return result.stdout.strip()

def is_commit_within_the_last_target_years(commit_date, years):
    now = datetime.now(tz=tz.tzlocal())  # Make the current datetime offset-aware
    years_ago = now - timedelta(days=years * 365)  # Subtract the target years
    return commit_date >= years_ago

# Example usage
if __name__ == "__main__":
    with (open(config.INPUT_FILE, mode='r', newline='', encoding='utf-8') as file):
        csv_reader = list(csv.reader(file))  # Read all lines into a list
        count = 0
        #for line in reversed(csv_reader[1:]):
        for line in csv_reader[2:]:
            repo_path = config.LOCAL_MERCURIAL_PATH

            if len(line[1].split(" ")) < 2 and len(line[4].split(" ")) < 2 and line[1] != '' and line[4] != '':
                fix_commit_hash = line[1]
                fix_commit_date = get_commit_date(fix_commit_hash, repo_path)

                if fix_commit_date:

                    if is_commit_within_the_last_target_years(fix_commit_date, 10):

                        count += 1
                        bug_commit_diff = get_diff(line[4], repo_path)
                        fix_commit_diff = get_diff(line[1], repo_path)

                        bug_count_tokens = count_openai_tokens(bug_commit_diff)
                        fix_count_tokens = count_openai_tokens(fix_commit_diff)

                        if (fix_count_tokens < 4000) and (
                                len(line[3].split(" ")) < 2) and (line[3] != ''):

                            bug_id = line[3]
                            bug_mozilla = bugzilla.get(bug_id)
                            bug_commit_hash = line[4]
                            bug_commit_message = get_commit_message(repo_path, bug_commit_hash)

                            fix_id = line[0]
                            fix_mozilla = bugzilla.get(fix_id)
                            fix_commit_message = get_commit_message(repo_path, fix_commit_hash)

                            bug_commit_date = get_commit_date(bug_commit_hash, repo_path)

                            interval_bug_fix = (fix_commit_date - bug_commit_date).days

                            bug_patch_title = bug_mozilla.get(int(bug_id))['summary']
                            bug_summary = bug_mozilla.get(int(bug_id))['comments'][0]['text']

                            fix_patch_title = fix_mozilla.get(int(fix_id))['summary']
                            fix_summary = fix_mozilla.get(int(fix_id))['comments'][0]['text']

                            output = generate_comments_bug_fix(bug_commit_diff, fix_commit_diff, bug_commit_message,
                                                                 fix_commit_message, bug_patch_title, fix_patch_title, bug_summary, fix_summary)

                            if output is not None and len(output) > 1:
                                comments = output[0]
                                filtered_deepseek = output[1]

                                print(comments)

                                if comments is not None:
                                    valid_json = extract_and_parse_json(comments)
                                    print(valid_json)
                                    write_bug_info_to_csv(bug_id, bug_commit_hash, bug_count_tokens, fix_id, fix_commit_hash,
                                                          fix_count_tokens, bug_summary, valid_json, interval_bug_fix)
                                if filtered_deepseek is not None:
                                    valid_json = extract_and_parse_json(filtered_deepseek)
                                    print(valid_json)
                                    write_bug_info_to_csv(str(bug_id)+"DEEP", bug_commit_hash, bug_count_tokens, fix_id,
                                                          fix_commit_hash,
                                                          fix_count_tokens, bug_summary, valid_json, interval_bug_fix)

                                    count += 1
                                    if count == 50:
                                        break
                                else:
                                    print("No comments were generated.")
                        else:
                            print("The commit patch is too large.")
                    else:
                        print("The commit was NOT made within the last target years (" + str(fix_commit_date) + ").")
                else:
                    print("Could not retrieve the commit date for " + str(fix_commit_hash) + ".")

        print("\n\n\n Number of Valid cases " + str(count))