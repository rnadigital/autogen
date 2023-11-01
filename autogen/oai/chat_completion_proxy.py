import logging

import openai
import tiktoken
from uuid import uuid4
from datetime import datetime
from autogen.code_utils import extract_code


class ChatCompletionProxy:

    def __init__(self, callback):
        self.callback = callback
        self.encoding = tiktoken.get_encoding("cl100k_base")

    @staticmethod
    def _prompt_tokens(messages):
        encoding = tiktoken.get_encoding("cl100k_base")
        return sum([len(encoding.encode(msg['content'])) for msg in messages])

    def create(self, *args, **kwargs):
        try:
            # Check if streaming is enabled in the function arguments
            if kwargs.get("stream", False):
                response_content = ""
                completion_tokens = 0
                # Setting default values for variables
                first = True
                message_uuid = str(uuid4())
                chunk = {}
                code_block = None
                lang = None
                # Send the chat completion request to OpenAI's API and process the response in chunks
                for chunk in openai.ChatCompletion.create(*args, **kwargs):
                    if chunk["choices"]:
                        content = chunk["choices"][0].get("delta", {}).get("content")
                        # If content is present, print it to the terminal and update response variables
                        if content is not None:
                            message = {
                                "chunkId": message_uuid,
                                "text": content,
                                "first": first,
                                "tokens": 1,
                                "timestamp": datetime.now().timestamp() * 1000
                            }
                            self.callback("message", message)
                            first = False
                            response_content += content
                            completion_tokens += 1

                is_code = extract_code(response_content)
                if is_code and len(is_code) > 0:
                    lang = is_code[0]
                    code_block = is_code[1]

                # Send
                self.callback(
                    "message_complete",
                    {"text": response_content,
                     "chunkId": message_uuid,
                     "language": lang,
                     "codeBlock": code_block})
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
        except Exception as e:
            logging.exception(e)
            content = "An error has occurred"
            message_uuid = None
            first = True
            self.callback("message", {
                "chunkId": message_uuid,
                "text": content,
                "first": first,
                "tokens": 0,
                "timestamp": datetime.now().timestamp() * 1000
            })
            return None
