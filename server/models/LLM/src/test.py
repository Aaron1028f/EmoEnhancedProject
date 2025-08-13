import httpx
import sys

url = "http://localhost:8000/streaming_response"

# request with user input
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break
    if not user_input:
        continue

    print("AI: ", end="")
    sys.stdout.flush()

    # response with label
    # -----------------------
    # with httpx.stream("GET", url, params={"user_input": user_input}, timeout=60) as response:
    #     for chunk in response.iter_raw():
    #         if chunk:
    #             print(chunk.decode("utf-8"), end="", flush=True)
    #     # or
    #     # for line in response.iter_lines():
    #     #     if line:
    #     #         print(line.decode("utf-8"))
    # print()
    
    # -----------------------------------------------------------------------------
    # response with no label
    in_tag = False
    with httpx.stream("GET", url, params={"user_input": user_input}, timeout=60) as response:
        response.raise_for_status()
        for text_chunk in response.iter_text():
            for char in text_chunk:
                if char == '<':
                    in_tag = True
                elif char == '>':
                    in_tag = False
                elif not in_tag:
                    print(char, end="", flush=True)
    print()