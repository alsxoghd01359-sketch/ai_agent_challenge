import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv('api_key'))

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=[
        {"role" : "system", "content" : "당신은 친절한 과학 선생님입니다."},
        {"role" : "user", "content" : "블랙홀이 무엇인가요?"}
              ],
)
print(response.choices[0].message.content)