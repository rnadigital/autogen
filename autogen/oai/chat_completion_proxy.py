import openai
import tiktoken
from uuid import uuid4


class ChatCompletionProxy:

    def __init__(self, callback):
        self.callback = callback
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def _prompt_tokens(self, message: str):
        return len(self.encoding.encode(message))

    def create(self, *args, **kwargs):
        # Check if streaming is enabled in the function arguments
        if kwargs.get("stream", False):
            response_content = ""
            completion_tokens = 0

            # Set the terminal text color to green for better visibility
            print("\033[32m", end='')
            first = True
            message_uuid = str(uuid4())
            # Send the chat completion request to OpenAI's API and process the response in chunks
            for chunk in openai.ChatCompletion.create(*args, **kwargs):
                if chunk["choices"]:
                    content = chunk["choices"][0].get("delta", {}).get("content")
                    # If content is present, print it to the terminal and update response variables
                    if content is not None:
                        self.callback(content, message_uuid, first, self._prompt_tokens(content))
                        first = False
                        response_content += content
                        completion_tokens += 1

            # Reset the terminal text color
            print("\033[0m\n")

            # Prepare the final response object based on the accumulated data
            response = chunk
            response["choices"][0]["message"] = {
                'role': 'assistant',
                'content': response_content
            }

            prompt_tokens = self._prompt_tokens(kwargs["messages"])
            print(f"Tokens used: {prompt_tokens} ")
            # Add usage information to the response
            response["usage"] = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        else:
            # If streaming is not enabled, send a regular chat completion request
            response = openai.ChatCompletion.create(*args, **kwargs)

        # Return the final response object
        return response
