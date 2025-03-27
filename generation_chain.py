"""
This file contains code to generate a human-friendly answer to user question
using the retrieval result from the DB as context
"""
from helpers import remove_n,read_template, stop_query
from langchain_core.runnables import RunnableLambda
from langchain.schema.runnable import RunnablePassthrough
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from retrieval_chain import get_DB_data
from operator import itemgetter


# loading templates
# I am using open food database, change sql generation template to different database schema
answer_prompt_template = """
You are an expert at converting SQL query results into clear, user-friendly answers.  
Your response must strictly follow these rules:  

- **Only return a direct answer** (Do NOT explain SQL or mention database terms).  
- **Strictly under 50 words**—short and to the point.  
- **Do NOT include SQL, Python code, backticks (` ``` `), or tables.**  
- **End the response immediately after the answer.**  

### Information:  
- User's Question: {question}  
- SQL Query: {query}  
- Query Output: {context}  

### Response:

"""

sql_prompt_template = """The Open Food Facts database contains the following tables and columns:

1. Table: brands
   - Columns:
     - name (Type: character varying): The name of the brand.
     - id (Type: integer): The unique identifier for the brand.

2. Table: categories
   - Columns:
     - id (Type: integer): The unique identifier for the category.
     - name (Type: character varying): The name of the category (e.g., snacks, beverages).

3. Table: countries
   - Columns:
     - name (Type: character varying): The name of the country.
     - id (Type: integer): The unique identifier for the country.

4. Table: products
   - Columns:
     - id (Type: integer): The unique identifier for the product.
     - product_name (Type: character varying): The name of the product.
     - creator (Type: character varying): The creator or manufacturer of the product.
     - ingredients (Type: character varying): The ingredients in the product.
     - category (Type: integer): Foreign key referring to the `categories` table.
     - sugars_100g (Type: double precision): Amount of sugar per 100g.
     - salt_100g (Type: double precision): Amount of salt per 100g.
     - energy_100g (Type: double precision): Amount of energy (calories) per 100g.
     - fat_100g (Type: double precision): Amount of fat per 100g.
     - proteins_100g (Type: double precision): Amount of protein per 100g.
     - carbohydrates_100g (Type: double precision): Amount of carbohydrates per 100g.

5. Table: products_brands
   - Columns:
     - product_id (Type: integer): Foreign key referring to the `products` table.
     - brand_id (Type: integer): Foreign key referring to the `brands` table.
     - id (Type: integer): Unique identifier for the relationship.

6. Table: products_countries
   - Columns:
     - product_id (Type: integer): Foreign key referring to the `products` table.
     - country_id (Type: integer): Foreign key referring to the `countries` table.
     - id (Type: integer): Unique identifier for the relationship.

Using the previous database schema convert the following natural language question, Question: {question} into SQL query.
Return ONLY the SQL query without any comment or explanation.
SQL query:
"""

# - Use indexed columns whenever possible. (products.product_name,  categories.name,  brands.name)
# - Never use SELECT *; always specify columns.
# - If filtering by text, use ILIKE instead of LIKE.
# Load DeepSeek Coder in 4 bit
model_path = "./models/deepseek-coder-6.7b-instruct"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit mode
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config,
                                                 device_map="auto",local_files_only=True)
pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100,
                        num_return_sequences=1, return_full_text=False,eos_token_id=tokenizer.eos_token_id
                    )
llm = HuggingFacePipeline(pipeline=pipeline)


sql_prompt = PromptTemplate(
    input_variables=["question"],
    template= sql_prompt_template
)
answer_prompt = PromptTemplate(
    input_variables=["query","context","question"],
    template=answer_prompt_template
)



# First chain: User question → SQL
sql_chain = sql_prompt | llm | RunnableLambda(remove_n)

# Second chain: Execute the query and return the result
context_chain = get_DB_data()

# Third chain: Question, Query, Data → Final answer
answer_chain = answer_prompt | llm | RunnableLambda(remove_n)

# using LCEL because LLMchain is  deprecated
rag_chain = ({"query": sql_chain,
              "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=context_chain)
    | RunnablePassthrough.assign(final_answer=answer_chain))

def get_rag_chain():
    return rag_chain


# result = rag_chain.invoke({"question":"What is the nutritional value of Oreo per 100 grams?"})
# print(result)