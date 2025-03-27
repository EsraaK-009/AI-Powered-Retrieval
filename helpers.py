import re


def remove_n(phrase):
    clean_text = re.sub(r'\n+', ' ', phrase)
    return clean_text

def read_template(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        content = file.read()
    return content

def stop_query(answer):
    query = answer.split(";")[0] + ";"
    return query