from openai import OpenAI


def main():
    openai_api_key = "EMPTY"
    openai_api_base = "http://10.141.15.30:8888/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    chat_response = client.chat.completions.create(
        model="Qwen2.5-Math-1.5B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who are you?"},
        ],
        temperature=0
    )
    message = chat_response.choices[0].message.content
    print(message)
    

if __name__ == "__main__":
    main()