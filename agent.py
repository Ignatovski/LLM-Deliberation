from openai import OpenAI, BadRequestError
import json
import argparse
import time
import re
import random 
import numpy as np 
import os
from openai import AzureOpenAI
from vertexai.preview.generative_models import GenerativeModel
from anthropic import Anthropic, AnthropicFoundry


class Agent():
    def __init__(self, initial_prompt_cls, round_prompt_cls, agent_name, temperature, model, rounds_num=24, agents_num=6, azure=False, hf_models={}):
        self.model = model

        self.agent_name = agent_name        
        self.temperature = temperature
        self.initial_prompt_cls = initial_prompt_cls 
        self.rounds_num = rounds_num 
        self.agents_num = agents_num

        self.initial_prompt = initial_prompt_cls.return_initial_prompt()
        self.messages = [{"role": "user", "content": self.initial_prompt}]

        
        self.round_prompt_cls = round_prompt_cls 
        self.azure = azure 
        self.claude = 'claude' in self.model
        self.claude_client = None
        self.client = None
        self.hf_model = True if 'hf' in model else False
        openai_base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or None
        anth_api_version = os.getenv("ANTHROPIC_API_VERSION", "2023-06-01")

        if 'gemini' in self.model:
            self.model_instance = GenerativeModel(model)
        if self.claude:
            # Prefer explicit Anthropic API key; fall back to Azure key for Azure-hosted Claude.
            api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY or AZURE_OPENAI_API_KEY not set for Claude model")
            base_url = os.getenv("ANTHROPIC_BASE_URL") or None
            if base_url and base_url.rstrip("/").endswith("/v1"):
                base_url = base_url.rstrip("/").rsplit("/v1", 1)[0]  # Anthropic client will append /v1/...
            default_query = None
            use_foundry = base_url and "services.ai.azure.com/anthropic" in base_url
            if base_url and "services.ai.azure.com/anthropic" in base_url:
                # Azure Anthropic endpoints require api-version query parameter.
                default_query = {"api-version": anth_api_version}
            ClientCls = AnthropicFoundry if use_foundry else Anthropic
            self.claude_client = ClientCls(
                api_key=api_key,
                base_url=base_url,
                default_headers={"anthropic-version": "2023-06-01"},
                default_query=default_query,
            )

        if azure and not self.claude:
            self.client = AzureOpenAI(
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
            api_version="2023-05-15"
            )
        elif not self.claude and not self.hf_model and 'gemini' not in self.model:
            # Default OpenAI-compatible client (works for OpenAI, Groq, OpenRouter, etc.).
            self.client = OpenAI(base_url=openai_base_url) if openai_base_url else OpenAI()

        if 'hf' in model:
            self.hf_model, self.hf_tokenizer, self.hf_pipeline_gen = hf_models[model]

    def execute_round(self, answer_history, round_idx):
        '''
        construct the prompt and call model
        '''        
        slot_prompt = self.round_prompt_cls.build_slot_prompt(answer_history,round_idx) 
        agent_response = self.prompt("user", slot_prompt)    
        return slot_prompt, agent_response

        
    def prompt(self,role, msg):
        '''
        call each model 
        '''
        if self.claude:
            messages = self.messages + [{"role": role, "content": msg}]
            claude_messages = [
                {"role": "user", "content": m["content"]} if m["role"] == "user" else {"role": "assistant", "content": m["content"]}
                for m in messages
            ]
            response = self.claude_client.messages.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=1024,
                messages=claude_messages,
            )
            # Response content is a list of content blocks; join text parts.
            parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
            return "".join(parts)

        elif self.azure and self.client:
            messages = self.messages + [ {"role": role, "content": msg} ]
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content
            except BadRequestError as exc:
                err_code = None
                if hasattr(exc, "code"):
                    err_code = exc.code
                elif hasattr(exc, "error") and isinstance(exc.error, dict):
                    err_code = exc.error.get("code")
                if err_code != "content_filter":
                    raise
                # Retry once with a sanitized, minimal prompt to avoid policy triggers.
                sanitized_messages = [{"role": "user", "content": "Provide a concise numeric proposal in <ANSWER><VALUE>n</VALUE></ANSWER> form within [-10,10]."}]
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=sanitized_messages,
                        temperature=self.temperature,
                    )
                    return response.choices[0].message.content
                except BadRequestError:
                    print(f"Problematic Request: {messages}")
                    # Fallback: return a safe, minimal answer to keep the run alive.
                    return "<SCRATCHPAD>Content filtered; using fallback.</SCRATCHPAD>\n<ANSWER><VALUE>0</VALUE></ANSWER>"

        elif 'gemini' in self.model: 
            responses = self.model_instance.generate_content(
            self.initial_prompt + msg,
            generation_config={
            "temperature": self.temperature,
            "top_p": 1
            },
            stream = True 
            )
            content = ''
            for response in responses:
                content += response.text
            return content

        elif self.hf_model:
            chat = [{"role": "user", "content": self.initial_prompt+msg}]
            model_input = self.hf_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, return_tensors="pt")
            output_text = self.hf_pipeline_gen(model_input, do_sample=True, temperature = self.temperature)[0]['generated_text']
            return output_text

        elif self.client:
            messages = self.messages + [ {"role": role, "content": msg} ]
            response = self.client.chat.completions.create(model=self.model, messages=messages,temperature=self.temperature)
            content = response.choices[0].message
            return content 
        
