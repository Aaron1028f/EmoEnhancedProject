# source: https://stackoverflow.com/questions/75740652/fastapi-streamingresponse-not-streaming-with-generator-function

# -----------------------------------------------------------------------------
# method 1: use requests

# import requests

# url = "http://localhost:8000/streaming_response"

# with requests.get(url, stream=True) as response:
#     for line in response.iter_lines():
#         if line:
#             print(line.decode("utf-8"))
#     # or
#     # for chunk in response.iter_content(chunk_size=1024):
#     #     if chunk:
#     #         print(chunk.decode("utf-8"))


# -----------------------------------------------------------------------------
# method 2: or use httpx (better way)

import httpx

url = "http://localhost:8000/streaming_response"

with httpx.stream("GET", url) as response:
    for chunk in response.iter_raw():
        if chunk:
            print(chunk.decode("utf-8"), end="")
    # or
    # for line in response.iter_lines():
    #     if line:
    #         print(line.decode("utf-8"))