import logging
import os

import openai
import tiktoken
from uuid import uuid4
from datetime import datetime
from autogen.code_utils import extract_code
from typing import Optional, Callable
from openai.error import RateLimitError, InvalidRequestError, AuthenticationError
import redis
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# Check the redis key for "stop generating" on every nth chunk
NTH_CHUNK_CHECK = 5

class StopGeneratingException(Exception):
    """Specific exception raised when stop generating was requested by user"""
    pass

class ChatCompletionProxy:

    def __init__(self, send_to_socket: Optional[Callable], session_id: Optional[str]):
        self.send_to_socket = send_to_socket
        self.sid = session_id
        self.encoding = tiktoken.get_encoding("cl100k_base")
        try:
            # TODO: make redis port an env?
            self.redis_client = redis.Redis(host=os.environ.get("REDIS_HOST"), port=6379, decode_responses=True)
        except Exception as e:
            logging.error(f"Failed to create self.redis_client: {e}")
            self.redis_client = None

    @staticmethod
    def _prompt_tokens(messages):
        encoding = tiktoken.get_encoding("cl100k_base")
        return sum([len(encoding.encode(msg['content'])) for msg in messages])

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
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
                # Send the chat completion request to OpenAI's API and process the response in chunks
                for chunk_index, chunk in enumerate(openai.ChatCompletion.create(*args, **kwargs)):
                    if chunk_index % NTH_CHUNK_CHECK == 0 and self.redis_client:
                        delete_return_value = self.redis_client.delete(f"{self.sid}_stop")
                        if delete_return_value == 1: # 1 indicates that it was deleted, so must have been set
                            raise StopGeneratingException(f"stop key was set, stopping generating")
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
                            self.send_to_socket("message", message)
                            first = False
                            response_content += content
                            completion_tokens += 1

                code_blocks = []
                extracted_code = extract_code(response_content)
                for elem in extracted_code:
                    lang = elem[0]
                    code_block = elem[1]
                    code_blocks.append({
                        "language": lang,
                        "codeBlock": code_block
                    })

                # Send
                self.send_to_socket(
                    "message_complete",
                    {"text": response_content,
                     "chunkId": message_uuid,
                     "codeBlocks": code_blocks})
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
        except (InvalidRequestError, RateLimitError) as rle:
            logging.exception(rle)
            self.send_to_socket("message", {
                "chunkId": None,
                "text": "Rate limit reached. Retrying...",
                "first": True,
                "type": "error",
                "tokens": 0,
                "timestamp": datetime.now().timestamp() * 1000
            })
            return None # Note: should this be something else?
        except AuthenticationError as ae:
            logging.warning(ae)
            self.send_to_socket("message", {
                "chunkId": None,
                "text": "No API Key found. Please set your OPENAI key on your [account page](/account)",
                "first": True,
                "type": "error",
                "tokens": 0,
                "timestamp": datetime.now().timestamp() * 1000
            })
            return None
        except StopGeneratingException as sge:
            logging.exception(sge.args[0])
            content = "Stopped generating."
            message_uuid = None
            first = True
            self.send_to_socket("message", {
                "chunkId": message_uuid,
                "text": content,
                "type": "error",
                "first": first,
                "tokens": 0,
                "timestamp": datetime.now().timestamp() * 1000
            })
            return None
        except Exception as e:
            logging.exception(e)
            content = "An error has occurred"
            message_uuid = None
            first = True
            self.send_to_socket("message", {
                "chunkId": message_uuid,
                "text": content,
                "type": "error",
                "first": first,
                "tokens": 0,
                "timestamp": datetime.now().timestamp() * 1000
            })
            return None
