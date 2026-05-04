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
            # JSON schema response_format requires Azure API versions 2024-08-01-preview+.
            # Prefer an explicitly set AZURE_OPENAI_API_VERSION; fall back to OPENAI_API_VERSION
            # (used by Azure AI Foundry examples); otherwise default to a schema-capable version.
            requested_api_version = (
                os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION") or ""
            ).strip()
            required_api_version = "2024-08-01-preview"

            def _api_version_date(ver: str):
                # Accept formats like "2024-08-01-preview" / "2024-08-01" / "2024-08-01-preview.1".
                try:
                    parts = ver.split("-", 3)
                    y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                    return (y, m, d)
                except Exception:
                    return None

            req_date = _api_version_date(required_api_version)
            got_date = _api_version_date(requested_api_version) if requested_api_version else None

            if req_date and got_date and got_date < req_date:
                # Enforce schema-capable version even if env is outdated.
                print(
                    f"[agent] AZURE_OPENAI_API_VERSION={requested_api_version!r} is too old for json_schema; "
                    f"using {required_api_version!r} instead.",
                    flush=True,
                )
                api_version = required_api_version
            else:
                api_version = requested_api_version or required_api_version
            timeout = float(os.getenv("AZURE_OPENAI_TIMEOUT", "120"))
            max_retries = int(os.getenv("AZURE_OPENAI_MAX_RETRIES", "2"))
            self.client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=api_version,
                timeout=timeout,
                max_retries=max_retries,
            )
        elif not self.claude and not self.hf_model and 'gemini' not in self.model:
            # Default OpenAI-compatible client (works for OpenAI, Groq, OpenRouter, etc.).
            timeout = float(os.getenv("OPENAI_TIMEOUT", "120"))
            max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
            if openai_base_url:
                self.client = OpenAI(base_url=openai_base_url, timeout=timeout, max_retries=max_retries)
            else:
                self.client = OpenAI(timeout=timeout, max_retries=max_retries)

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
        # common JSON schema for structured outputs
        json_schema = {
            "name": "structured_output",
            "schema": {
                "type": "object",
                "properties": {
                    "scratchpad": {"type": "string"},
                    "answer": {"type": "string"},
                    "plan": {"type": "string"},
                },
                "required": ["scratchpad", "answer", "plan"],
                "additionalProperties": False,
            },
            "strict": True,
        }
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
                print(f"[{self.agent_name}] calling {self.model}...", flush=True)
                max_tokens_raw = os.getenv("AZURE_OPENAI_MAX_TOKENS") or os.getenv("OPENAI_MAX_TOKENS")
                max_tokens = int(max_tokens_raw) if max_tokens_raw else None
                temp = None if self.temperature == 0 else self.temperature
                reasoning_effort = (
                    os.getenv("AZURE_OPENAI_REASONING_EFFORT")
                    or os.getenv("OPENAI_REASONING_EFFORT")
                    or os.getenv("REASONING_EFFORT")
                )
                if not reasoning_effort and (self.model.startswith("gpt-5") or self.model.startswith("o1")):
                    reasoning_effort = "low"
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "response_format": {"type": "json_schema", "json_schema": json_schema},
                }
                if max_tokens is not None and max_tokens > 0:
                    kwargs["max_completion_tokens"] = max_tokens
                if temp is not None:
                    kwargs["temperature"] = temp
                if reasoning_effort:
                    kwargs["reasoning_effort"] = reasoning_effort
                response = self.client.chat.completions.create(**kwargs)
                print(f"[{self.agent_name}] response received", flush=True)
                content = response.choices[0].message.content
                if not (content or "").strip():
                    finish = response.choices[0].finish_reason
                    rtok = getattr(response.usage.completion_tokens_details, "reasoning_tokens", None) if response.usage else None
                    print(
                        f"[{self.agent_name}] empty content (finish_reason={finish}, reasoning_tokens={rtok}); "
                        f"set AZURE_OPENAI_MAX_TOKENS/OPENAI_MAX_TOKENS higher if this persists.",
                        flush=True,
                    )
                return content
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
            print(f"[{self.agent_name}] calling {self.model}...", flush=True)
            max_tokens_raw = os.getenv("OPENAI_MAX_TOKENS") or os.getenv("AZURE_OPENAI_MAX_TOKENS")
            max_tokens = int(max_tokens_raw) if max_tokens_raw else None
            temp = None if self.temperature == 0 else self.temperature
            reasoning_effort = (
                os.getenv("OPENAI_REASONING_EFFORT")
                or os.getenv("REASONING_EFFORT")
                or os.getenv("AZURE_OPENAI_REASONING_EFFORT")
            )
            if not reasoning_effort and (self.model.startswith("gpt-5") or self.model.startswith("o1")):
                reasoning_effort = "low"
            kwargs = {
                "model": self.model,
                "messages": messages,
                "response_format": {"type": "json_schema", "json_schema": json_schema},
            }
            if max_tokens is not None and max_tokens > 0:
                kwargs["max_completion_tokens"] = max_tokens
            if temp is not None:
                kwargs["temperature"] = temp
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            response = self.client.chat.completions.create(**kwargs)
            print(f"[{self.agent_name}] response received", flush=True)
            content = response.choices[0].message.content
            if not (content or "").strip():
                finish = response.choices[0].finish_reason
                rtok = getattr(response.usage.completion_tokens_details, "reasoning_tokens", None) if response.usage else None
                print(
                    f"[{self.agent_name}] empty content (finish_reason={finish}, reasoning_tokens={rtok}); "
                    f"set OPENAI_MAX_TOKENS higher if this persists.",
                    flush=True,
                )
            return content
        
