import ast  # for converting embeddings saved as strings back to arrays
from openai import OpenAI # for calling the OpenAI API
import pandas as pd  # for storing text and embeddings data
import tiktoken  # for counting tokens
import os # for getting API token from env variable OPENAI_API_KEY
from scipy import spatial  # for calculating vector similarities for search

# create a list of models 
GPT_MODELS = ["gpt-4o", "gpt-4o-mini"]
# models
EMBEDDING_MODEL = "text-embedding-3-small"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>")) 

query = 'Which athletes won the most number of gold medals in 2024 Summer Olympics?'

response = client.chat.completions.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about the 2024 Games or latest events.'},
        {'role': 'user', 'content': query},
    ],
    model=GPT_MODELS[0],
    temperature=0,
)

print(response.choices[0].message.content) 

query = 'Who won the elections in the US in 2024?'

response = client.chat.completions.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about the 2024 Games or latest events.'},
        {'role': 'user', 'content': query},
    ],
    model=GPT_MODELS[1],
    temperature=0,
)

print(response.choices[0].message.content) 


wikipedia_article = ""

query = f"""Use the below article on the 2024 Summer Olympics to answer the subsequent question. If the answer cannot be found, write "I don't know."

Article:
\"\"\"
{wikipedia_article}
\"\"\"

Question: Which countries won the maximum number of gold, silver and bronze medals respectively at 2024 Summer Olympics? List the countries in the order of gold, silver and bronze medals."""

response = client.chat.completions.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about the recent events.'},
        {'role': 'user', 'content': query},
    ],
    model=GPT_MODELS[0],
    temperature=0,
)

print(response.choices[0].message.content)


embeddings_path = "data/winter_olympics_2022.csv"

df = pd.read_csv(embeddings_path)

df['embedding'] = df['embedding'].apply(ast.literal_eval) 


def strings_ranked_by_relatedness(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 100
) -> tuple[list[str], list[float]]:
    """Returns a list of strings and relatednesses, sorted from most related to least."""
    query_embedding_response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding
    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]
    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n] 

strings, relatednesses = strings_ranked_by_relatedness("curling gold medal", df, top_n=5)
for string, relatedness in zip(strings, relatednesses):
    print(f"{relatedness=:.3f}")
    display(string) 


def num_tokens(text: str, model: str = GPT_MODELS[0]) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings, relatednesses = strings_ranked_by_relatedness(query, df)
    introduction = 'Use the below articles on the 2022 Winter Olympics to answer the subsequent question. If the answer cannot be found in the articles, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"
    message = introduction
    for string in strings:
        next_article = f'\n\nWikipedia article section:\n"""\n{string}\n"""'
        if (
            num_tokens(message + next_article + question, model=model)
            > token_budget
        ):
            break
        else:
            message += next_article
    return message + question


def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODELS[0],
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about the 2022 Winter Olympics."},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )
    response_message = response.choices[0].message.content
    return response_message 


ask('Which athletes won the gold medal in curling at the 2022 Winter Olympics?')

