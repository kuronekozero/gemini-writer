#!/usr/bin/env python3
"""
Gemini Writing Agent - An autonomous agent for creative writing tasks.

This agent uses the Gemini 3 Flash model to create novels, books, 
and short story collections based on user prompts.
"""

import os
import sys
import json
import argparse
import time
import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import List, Dict, Any, Union

# Load environment variables from .env file
load_dotenv()

from utils import (
    estimate_token_count, 
    get_tool_definitions, 
    get_tool_map,
    get_system_prompt,
)
from tools.compression import compress_context_impl
from tools.project import create_project_impl, get_active_project_folder
from tools.writer import write_file_impl


# Constants
MAX_ITERATIONS = 300
TOKEN_LIMIT = 1000000  # Gemini has 1M context window
COMPRESSION_THRESHOLD = 900000  # Trigger compression at 90% of limit
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3-flash-preview")
BACKUP_INTERVAL = 50  # Save backup summary every N iterations
API_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "90"))
DEBUG_API = os.getenv("DEBUG_API", "1").lower() in {"1", "true", "yes", "on"}


def load_context_from_file(file_path: str) -> str:
    """
    Loads context from a summary file for recovery.
    
    Args:
        file_path: Path to the context summary file
        
    Returns:
        Content of the file as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✓ Loaded context from: {file_path}\n")
        return content
    except Exception as e:
        print(f"✗ Error loading context file: {e}")
        sys.exit(1)


def get_user_input() -> tuple[str, bool]:
    """
    Gets user input from command line, either as a prompt or recovery file.
    
    Returns:
        Tuple of (prompt/context, is_recovery_mode)
    """
    parser = argparse.ArgumentParser(
        description="Gemini Writing Agent - Create novels, books, and short stories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fresh start with inline prompt
  python kimi-writer.py "Create a collection of sci-fi short stories"
  
  # Recovery mode from previous context
  python kimi-writer.py --recover my_project/.context_summary_20250107_143022.md
        """
    )
    
    parser.add_argument(
        'prompt',
        nargs='?',
        help='Your writing request (e.g., "Create a mystery novel")'
    )
    parser.add_argument(
        '--recover',
        type=str,
        help='Path to a context summary file to continue from'
    )
    
    args = parser.parse_args()
    
    # Check if recovery mode
    if args.recover:
        context = load_context_from_file(args.recover)
        return context, True
    
    # Check if prompt provided as argument
    if args.prompt:
        return args.prompt, False
    
    # Interactive prompt
    print("=" * 60)
    print("Gemini Writing Agent")
    print("=" * 60)
    print("\nEnter your writing request (or 'quit' to exit):")
    print("Example: Create a collection of 15 sci-fi short stories\n")
    
    prompt = input("> ").strip()
    
    if prompt.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        sys.exit(0)
    
    if not prompt:
        print("Error: Empty prompt. Please provide a writing request.")
        sys.exit(1)
    
    return prompt, False


def main():
    """Main agent loop."""
    api_key = None
    client = None
    provider = "Gemini API"
    openrouter_mode = False

    # Optional OpenRouter path
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1beta")

    if openrouter_api_key:
        api_key = openrouter_api_key
        provider = "OpenRouter (Gemini model)"
        openrouter_mode = True
        client = None
    else:
        # Default Gemini path
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Error: No API key configured.")
            print("Set one of:")
            print("  - OPENROUTER_API_KEY (recommended if using OpenRouter)")
            print("  - GEMINI_API_KEY (direct Gemini API)")
            sys.exit(1)
        client = genai.Client(api_key=api_key)
    
    # Debug: Show that key is loaded (masked for security)
    if len(api_key) > 8:
        print(f"✓ API Key loaded: {api_key[:4]}...{api_key[-4:]}")
    else:
        print(f"⚠️  Warning: API key seems too short ({len(api_key)} chars)")
    
    print(f"✓ Client initialized via: {provider}\n")
    if openrouter_mode:
        print("ℹ️  OpenRouter compatibility mode: thinking/tools are disabled for stability.\n")
    print(f"✓ API timeout per call: {API_TIMEOUT_SECONDS}s")
    print(f"✓ Debug logging: {'ON' if DEBUG_API else 'OFF'}\n")

    # Fast connectivity probe to fail early instead of hanging at iteration 1
    try:
        print("🔎 Running startup connectivity probe...")
        probe_start = time.time()
        if openrouter_mode:
            probe_payload = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                "max_tokens": 10,
                "temperature": 0.0,
            }
            with httpx.Client(timeout=API_TIMEOUT_SECONDS) as http:
                probe_response = http.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=probe_payload,
                )
            probe_response.raise_for_status()
            probe_text = probe_response.json()["choices"][0]["message"]["content"].strip()
        else:
            probe_response = client.models.generate_content(
                model=MODEL_NAME,
                contents="Reply with exactly: OK",
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=10,
                    http_options=types.HttpOptions(timeout=API_TIMEOUT_SECONDS * 1000),
                ),
            )
            probe_text = (probe_response.text or "").strip()
        print(f"✓ Probe succeeded in {time.time() - probe_start:.1f}s | Response: {probe_text}\n")
    except Exception as e:
        print("\n✗ Startup probe failed.")
        print(f"  Provider: {provider}")
        print(f"  Model: {MODEL_NAME}")
        print(f"  Base URL: {'https://openrouter.ai/api/v1/chat/completions' if openrouter_mode else 'Gemini default'}")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error: {e}")
        print("\nTroubleshooting:")
        print("  1) Verify MODEL_NAME exists for your provider.")
        print("  2) Try a lower timeout: API_TIMEOUT_SECONDS=30")
        print("  3) Ensure OPENROUTER_API_KEY is active and has credits.")
        sys.exit(1)
    
    # Get user input
    user_prompt, is_recovery = get_user_input()
    
    # Initialize contents list with raw Content objects
    # This preserves thought_signature and other metadata
    contents: List[types.Content] = []
    
    # Add initial user message
    if is_recovery:
        initial_message = f"[RECOVERED CONTEXT]\n\n{user_prompt}\n\n[END RECOVERED CONTEXT]\n\nPlease continue the work from where we left off."
        print("🔄 Recovery mode: Continuing from previous context\n")
    else:
        initial_message = user_prompt
        print(f"\n📝 Task: {user_prompt}\n")
    
    contents.append(types.Content(
        role="user",
        parts=[types.Part.from_text(text=initial_message)]
    ))

    if openrouter_mode:
        print("🚀 OpenRouter quick-run mode: sending one chat-completions request.\n")
        try:
            call_start = time.time()
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": get_system_prompt()},
                    {"role": "user", "content": initial_message},
                ],
                "temperature": 1.0,
            }
            with httpx.Client(timeout=API_TIMEOUT_SECONDS) as http:
                response = http.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            response.raise_for_status()
            data = response.json()
            content_text = data["choices"][0]["message"]["content"]
            print("💬 Response:")
            print("-" * 60)
            print(content_text)
            print("-" * 60)

            project_name = f"openrouter_{time.strftime('%Y%m%d_%H%M%S')}"
            create_status = create_project_impl(project_name=project_name)
            print(f"\n📁 {create_status}")
            save_status = write_file_impl(
                filename="book.md",
                content=content_text,
                mode="create",
            )
            print(f"💾 {save_status}")
            active_folder = get_active_project_folder()
            if active_folder:
                print(f"📍 Output path: {active_folder}{os.sep}book.md")

            if DEBUG_API:
                print(f"\n⏱️  OpenRouter call completed in {time.time() - call_start:.1f}s")
            return
        except Exception as e:
            print(f"\n✗ OpenRouter quick-run failed: {type(e).__name__}: {e}")
            sys.exit(1)
    
    # Get tool definitions and mapping
    tools = get_tool_definitions()
    tool_map = get_tool_map()
    
    # Get system prompt for config
    system_instruction = get_system_prompt()
    
    print("=" * 60)
    print("Starting Gemini Writing Agent")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"Context limit: {TOKEN_LIMIT:,} tokens")
    print(f"Auto-compression at: {COMPRESSION_THRESHOLD:,} tokens")
    print("=" * 60 + "\n")
    
    # Main agent loop
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'─' * 60}")
        print(f"Iteration {iteration}/{MAX_ITERATIONS}")
        print(f"{'─' * 60}")
        
        # Check token count before making API call
        try:
            token_count = estimate_token_count(client, MODEL_NAME, contents)
            print(f"📊 Current tokens: {token_count:,}/{TOKEN_LIMIT:,} ({token_count/TOKEN_LIMIT*100:.1f}%)")
            
            # Trigger compression if approaching limit
            if token_count >= COMPRESSION_THRESHOLD:
                print(f"\n⚠️  Approaching token limit! Compressing context...")
                # For compression, convert to simple format
                simple_messages = []
                for content in contents:
                    role = content.role
                    text_parts = []
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        simple_messages.append({"role": role, "content": " ".join(text_parts)})
                
                compression_result = compress_context_impl(
                    messages=[{"role": "system", "content": system_instruction}] + simple_messages,
                    client=client,
                    model=MODEL_NAME,
                    keep_recent=10
                )
                
                if "compressed_messages" in compression_result:
                    # Rebuild contents from compressed messages
                    new_contents = []
                    for msg in compression_result["compressed_messages"]:
                        if msg.get("role") == "system":
                            continue
                        role = "model" if msg.get("role") in ["assistant", "model"] else "user"
                        if msg.get("content"):
                            new_contents.append(types.Content(
                                role=role,
                                parts=[types.Part.from_text(text=msg["content"])]
                            ))
                    contents = new_contents
                    print(f"✓ {compression_result['message']}")
                    print(f"✓ Estimated tokens saved: ~{compression_result.get('tokens_saved', 0):,}")
                    token_count = estimate_token_count(client, MODEL_NAME, contents)
                    print(f"📊 New token count: {token_count:,}/{TOKEN_LIMIT:,}\n")
        
        except Exception as e:
            print(f"⚠️  Warning: Could not estimate token count: {e}")
            token_count = 0
        
        # Auto-backup every N iterations
        if iteration % BACKUP_INTERVAL == 0:
            print(f"💾 Auto-backup (iteration {iteration})...")
            try:
                simple_messages = [{"role": "system", "content": system_instruction}]
                for content in contents:
                    role = content.role
                    text_parts = []
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        simple_messages.append({"role": role, "content": " ".join(text_parts)})
                
                compression_result = compress_context_impl(
                    messages=simple_messages,
                    client=client,
                    model=MODEL_NAME,
                    keep_recent=len(simple_messages)
                )
                if compression_result.get("summary_file"):
                    print(f"✓ Backup saved: {os.path.basename(compression_result['summary_file'])}\n")
            except Exception as e:
                print(f"⚠️  Warning: Backup failed: {e}\n")
        
        # Configure generation
        # NOTE: OpenRouter compatibility mode disables Gemini-specific thinking+tools
        # to prevent hanging calls on some OpenRouter routes.
        if openrouter_mode:
            generate_config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=1.0,
            )
        else:
            generate_config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                thinking_config=types.ThinkingConfig(
                    thinking_level="HIGH",
                ),
                tools=[tools],
                temperature=1.0,
            )
        
        # Call the model
        try:
            print("🤖 Calling Gemini model...\n")
            call_start = time.time()
            
            # Use non-streaming to get complete response with thought_signature
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    **generate_config.model_dump(exclude_none=True),
                    http_options=types.HttpOptions(timeout=API_TIMEOUT_SECONDS * 1000),
                ),
            )
            if DEBUG_API:
                print(f"⏱️  API call completed in {time.time() - call_start:.1f}s")
            
            # Process the response
            thinking_text = ""
            content_text = ""
            function_calls_list = []
            
            # Get the model's response content (includes thought_signature)
            model_content = None
            if response.candidates and response.candidates[0].content:
                model_content = response.candidates[0].content
                
                # Process parts for display
                for part in model_content.parts:
                    # Handle thinking parts
                    if hasattr(part, 'thought') and part.thought:
                        if hasattr(part, 'text') and part.text:
                            thinking_text += part.text
                    # Handle function calls
                    elif hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        function_calls_list.append({
                            "name": fc.name,
                            "args": dict(fc.args) if fc.args else {}
                        })
                    # Handle regular text
                    elif hasattr(part, 'text') and part.text:
                        content_text += part.text
            
            # Display thinking
            if thinking_text:
                print("=" * 60)
                print(f"🧠 Thinking (Iteration {iteration})")
                print("=" * 60)
                print(thinking_text)
                print("=" * 60 + "\n")
            
            # Display content
            if content_text:
                print("💬 Response:")
                print("-" * 60)
                print(content_text)
                print("-" * 60 + "\n")
            
            # Display function calls
            if function_calls_list:
                print("🔧 Function calls detected:")
                print("─" * 60)
                for fc in function_calls_list:
                    print(f"  → {fc['name']}")
            
            # CRITICAL: Append the FULL model response to contents
            # This preserves thought_signature for function calling
            if model_content:
                contents.append(model_content)
            
            # Check if the model called any functions
            if not function_calls_list:
                print("=" * 60)
                print("✅ TASK COMPLETED")
                print("=" * 60)
                print(f"Completed in {iteration} iteration(s)")
                print("=" * 60)
                break
            
            # Handle function calls
            print(f"\n🔧 Model decided to call {len(function_calls_list)} tool(s):")
            
            # Collect all function responses
            function_response_parts = []
            
            for fc in function_calls_list:
                func_name = fc["name"]
                args = fc["args"]
                
                print(f"\n  → {func_name}")
                print(f"    Arguments: {json.dumps(args, ensure_ascii=False, indent=6)}")
                
                # Get the tool implementation
                tool_func = tool_map.get(func_name)
                
                if not tool_func:
                    result = f"Error: Unknown tool '{func_name}'"
                    print(f"    ✗ {result}")
                else:
                    # Special handling for compress_context (needs extra params)
                    if func_name == "compress_context":
                        simple_messages = [{"role": "system", "content": system_instruction}]
                        for content in contents:
                            role = content.role
                            text_parts = []
                            for part in content.parts:
                                if hasattr(part, 'text') and part.text:
                                    text_parts.append(part.text)
                            if text_parts:
                                simple_messages.append({"role": role, "content": " ".join(text_parts)})
                        
                        result_data = compress_context_impl(
                            messages=simple_messages,
                            client=client,
                            model=MODEL_NAME,
                            keep_recent=10
                        )
                        result = result_data.get("message", "Compression completed")
                    else:
                        # Call the tool with its arguments
                        result = tool_func(**args)
                    
                    # Print result (truncate if too long)
                    if len(str(result)) > 200:
                        print(f"    ✓ {str(result)[:200]}...")
                    else:
                        print(f"    ✓ {result}")
                
                # Create function response part
                function_response_parts.append(
                    types.Part.from_function_response(
                        name=func_name,
                        response={"result": str(result)}
                    )
                )
            
            # Add all function responses as a single user turn
            contents.append(types.Content(
                role="user",
                parts=function_response_parts
            ))
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Saving context...")
            try:
                simple_messages = [{"role": "system", "content": system_instruction}]
                for content in contents:
                    role = content.role
                    text_parts = []
                    for part in content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        simple_messages.append({"role": role, "content": " ".join(text_parts)})
                
                compression_result = compress_context_impl(
                    messages=simple_messages,
                    client=client,
                    model=MODEL_NAME,
                    keep_recent=len(simple_messages)
                )
                if compression_result.get("summary_file"):
                    print(f"✓ Context saved to: {compression_result['summary_file']}")
                    print(f"\nTo resume, run:")
                    print(f"  python kimi-writer.py --recover {compression_result['summary_file']}")
            except:
                pass
            sys.exit(0)
        
        except Exception as e:
            print(f"\n✗ Error during iteration {iteration}: {e}")
            print(f"Attempting to continue...\n")
            continue
    
    # If we hit max iterations
    if iteration >= MAX_ITERATIONS:
        print("\n" + "=" * 60)
        print("⚠️  MAX ITERATIONS REACHED")
        print("=" * 60)
        print(f"\nReached maximum of {MAX_ITERATIONS} iterations.")
        print("Saving final context...")
        
        try:
            simple_messages = [{"role": "system", "content": system_instruction}]
            for content in contents:
                role = content.role
                text_parts = []
                for part in content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                if text_parts:
                    simple_messages.append({"role": role, "content": " ".join(text_parts)})
            
            compression_result = compress_context_impl(
                messages=simple_messages,
                client=client,
                model=MODEL_NAME,
                keep_recent=len(simple_messages)
            )
            if compression_result.get("summary_file"):
                print(f"✓ Context saved to: {compression_result['summary_file']}")
                print(f"\nTo resume, run:")
                print(f"  python kimi-writer.py --recover {compression_result['summary_file']}")
        except Exception as e:
            print(f"✗ Error saving context: {e}")


if __name__ == "__main__":
    main()
