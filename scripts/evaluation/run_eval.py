import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from functools import wraps

import numpy as np
import requests
from flashrag.config import Config
from flashrag.generator.generator import BaseGenerator
from flashrag.pipeline import BasicPipeline
from flashrag.retriever.retriever import BaseTextRetriever
from flashrag.utils import get_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from config import logger
from src.agent import Agent, AgenticOutputs
from src.prompts import build_user_prompt, get_system_prompt
from src.tokenizer_adapter import LlamaTokenizerAdapter, R1DistilTokenizerAdapter


def retry(max_retries=10, sleep=1):
    """Decorator to retry a function with exponential backoff."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} of {func_name} failed: {e}")
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func_name} failed after {max_retries} retries.", exc_info=True)
                        raise e
                    backoff_time = sleep * (2**attempt)
                    logger.info(f"Retrying {func_name} in {backoff_time:.2f} seconds...")
                    time.sleep(backoff_time)
            logger.error(f"Function {func_name} retry logic finished unexpectedly.")
            return None

        return wrapper

    return decorator


class RemoteRetriever(BaseTextRetriever):
    """A wrapper for remote retriever service with retry logic and logging."""

    def __init__(self, config: Config):
        """Initializes the RemoteRetriever."""
        super().__init__(config)
        self.remote_url = f"http://{getattr(config, 'remote_retriever_url', 'localhost:8001')}"
        self.topk = getattr(config, "retriever_topk", 5)
        logger.info(f"🔗 Remote retriever URL: {self.remote_url}")

    @retry(max_retries=3, sleep=2)
    def _search(self, query: str, num: int | None = None, return_score: bool = False) -> list[dict]:
        """Search for documents using the remote retriever service."""
        num = num if num is not None else self.topk
        url = f"{self.remote_url}/search"

        try:
            response = requests.post(
                url,
                json={"query": query, "top_n": num, "return_score": return_score},
                timeout=30,
            )
            response.raise_for_status()

            results = response.json()
            return results
        except requests.exceptions.Timeout:
            logger.error(f"Search request timed out after 30 seconds for query: {query[:50]}...")
            raise
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to search service at {url}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Search request failed: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected search error: {str(e)}", exc_info=True)
            raise

    @retry(max_retries=3, sleep=2)
    def _batch_search(
        self, queries: list[str], num: int | None = None, return_score: bool = False
    ) -> list[list[dict]]:
        """Batch search for documents using the remote retriever service."""
        num = num if num is not None else self.topk
        url = f"{self.remote_url}/batch_search"

        try:
            response = requests.post(
                url,
                json={"query": queries, "top_n": num, "return_score": return_score},
                timeout=60,
            )
            response.raise_for_status()
            results = response.json()
            return results
        except requests.exceptions.Timeout:
            logger.error(f"Batch search request timed out after 60 seconds for {len(queries)} queries.")
            raise
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to batch search service at {url}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Batch search request failed: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected batch search error: {str(e)}", exc_info=True)
            raise


class ReSearchPipeline(BasicPipeline):
    """Pipeline for ReSearch method using Agent for generation and tool use."""

    def __init__(
        self, config: Config, retriever: BaseTextRetriever | None = None, generator: BaseGenerator | None = None
    ):
        """Initializes the ReSearchPipeline."""
        super().__init__(config)
        logger.info("🔧 Initializing ReSearchPipeline...")

        self.retriever = retriever or RemoteRetriever(config)

        self.generator = generator or SGLRemoteGenerator(config)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.generator_model_path, trust_remote_code=True)
            if not self.tokenizer.pad_token:
                logger.warning("Tokenizer does not have a pad token; setting to eos_token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            logger.info("✅ Tokenizer initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}", exc_info=True)
            raise

        tokenizer_name = self.tokenizer.name_or_path.lower()

        if "deepseek-ai/deepseek-r1-distill" in tokenizer_name:
            adapter = R1DistilTokenizerAdapter()
        elif "llama" in tokenizer_name:
            adapter = LlamaTokenizerAdapter()
        else:
            logger.warning(f"Unknown tokenizer type '{tokenizer_name}', defaulting to R1DistilTokenizerAdapter.")
            adapter = R1DistilTokenizerAdapter()
        logger.info(f"🔩 Using Tokenizer Adapter: {type(adapter).__name__}")

        def retriever_search(query: str, return_type=str, results: int = 2):
            try:
                search_results = self.retriever._search(query, num=results)
                return self.format_search_results(search_results)
            except Exception as e:
                logger.error(f"Error during agent's retriever search for query '{query[:50]}...': {e}", exc_info=True)
                return "<information>Search failed due to an internal error."

        self.agent = Agent(adapter, search_fn=retriever_search)
        logger.info("✅ Agent initialized.")
        logger.info("✅ ReSearchPipeline initialized successfully.")

    def format_search_results(self, search_results: list[dict]) -> str:
        """Formats search results into a string for the agent prompt."""
        if not search_results:
            return "<information>No results found.</information>"
        max_content_len = 500
        formatted = "\n-------\n".join(
            [
                f"Result {i + 1}: {r.get('contents', 'N/A')[:max_content_len]}{'...' if len(r.get('contents', '')) > max_content_len else ''}"
                for i, r in enumerate(search_results)
            ]
        )
        formatted_str = f"<information>{formatted}</information>"

        return formatted_str

    def extract_search_query(self, text: str) -> str | None:
        """Extract search query from text between <search> tags."""
        pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            query = matches[-1].strip()
            return query
        return None

    def extract_answer(self, text: str) -> str | None:
        """Extract answer from text between <answer> tags."""
        pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            answer = matches[-1].strip()

            return answer

        return None

    def run(self, dataset, do_eval: bool = True, pred_process_fun=None):
        """Runs the ReSearch pipeline on the dataset using the Agent."""
        logger.info(f"🏃 Starting ReSearch pipeline run with {len(dataset)} items...")

        try:
            questions = [item.question if hasattr(item, "question") else item["question"] for item in dataset]

        except (KeyError, AttributeError, TypeError) as e:
            logger.error(f"Failed to extract questions from dataset items. Error: {e}", exc_info=True)
            logger.error("Ensure dataset items have a 'question' key or attribute.")
            return dataset

        agent_max_generations = getattr(self.config, "agent_max_generations", 20)
        generator_max_output_len = getattr(self.config, "generator_max_output_len", 1024)

        try:
            logger.info(f"🤖 Running agent inference for {len(questions)} questions...")
            agent_outputs: AgenticOutputs = self.agent.run_agent(
                generate_fn=self.generator.generate,
                tokenizer=self.tokenizer,
                questions=questions,
                max_generations=agent_max_generations,
                max_new_tokens=generator_max_output_len,
            )
            final_responses = agent_outputs.final_response_str
            logger.info(f"✅ Agent inference completed. Received {len(final_responses)} final responses.")

        except Exception as e:
            logger.error(f"Agent run failed during inference: {e}", exc_info=True)
            logger.warning("Agent run failed, attempting evaluation with potentially incomplete results.")
            for item in dataset:
                if hasattr(item, "update_output"):
                    item.update_output("pred", "AGENT_ERROR")
                elif isinstance(item, dict):
                    item["pred"] = "AGENT_ERROR"

        logger.info("📝 Extracting answers and updating dataset items...")
        num_updated = 0
        num_missing_answers = 0
        if len(final_responses) == len(dataset):
            for i, item in enumerate(dataset):
                response = final_responses[i]
                answer = self.extract_answer(response)
                pred_to_save = answer if answer is not None else ""

                if answer is None:
                    num_missing_answers += 1

                if hasattr(item, "update_output"):
                    item.update_output("pred", pred_to_save)
                    item.update_output("final_response", response)
                    num_updated += 1
                elif isinstance(item, dict):
                    item["pred"] = pred_to_save
                    item["final_response"] = response
                    num_updated += 1
                else:
                    logger.warning(f"Item {i} has unknown type {type(item)}, cannot update with prediction.")

            logger.info(f"Updated {num_updated}/{len(dataset)} dataset items with predictions.")
            if num_missing_answers > 0:
                logger.warning(f"{num_missing_answers} items had no <answer> tag.")
        else:
            logger.error(
                f"Mismatch between dataset size ({len(dataset)}) and number of agent responses ({len(final_responses)}). Cannot reliably update dataset."
            )
            for item in dataset:
                if hasattr(item, "update_output"):
                    item.update_output("pred", "RESPONSE_COUNT_MISMATCH")
                elif isinstance(item, dict):
                    item["pred"] = "RESPONSE_COUNT_MISMATCH"

        if do_eval:
            logger.info("📊 Evaluating results using BasicPipeline.evaluate...")
            try:
                dataset = self.evaluate(dataset, do_eval=True, pred_process_fun=pred_process_fun)
                logger.info("✅ Evaluation completed via base class method.")
            except Exception as e:
                logger.error(f"Error during BasicPipeline.evaluate: {e}", exc_info=True)
                logger.warning("Evaluation may be incomplete.")
        else:
            logger.info("Skipping evaluation step as do_eval=False.")

        logger.info("✅ ReSearch pipeline run finished.")
        return dataset


class SGLRemoteGenerator(BaseGenerator):
    """Class for decoder-only generator, based on SGLang remote service."""

    def __init__(self, config: Config):
        """Initializes the SGLRemoteGenerator."""
        super().__init__(config)
        logger.info("🔧 Initializing SGLRemoteGenerator...")
        sgl_url = getattr(config, "sgl_remote_url", "localhost:8002")
        self.sgl_remote_url = f"http://{sgl_url}/generate"
        self.health_check_url = f"http://{sgl_url}/health"
        logger.info(f"🔗 Remote Generator URL: {self.sgl_remote_url}")
        self.model_path = getattr(config, "generator_model_path", None)
        if not self.model_path:
            logger.error("generator_model_path not found in config!")
            raise ValueError("generator_model_path is required for SGLRemoteGenerator")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            logger.info("✅ Tokenizer loaded for generator.")
        except Exception as e:
            logger.error(f"Failed to load tokenizer for generator from {self.model_path}: {e}", exc_info=True)
            raise

        self.generation_params = getattr(config, "generation_params", {})
        self.config = config

        self._check_health()

    def _check_health(self):
        """Checks the health of the remote generator service."""
        try:
            test_response = requests.get(self.health_check_url, timeout=5)
            test_response.raise_for_status()
            logger.info("✅ Remote generator service is available")
        except requests.exceptions.RequestException as e:
            logger.error(f"Could not connect or verify remote generator service at {self.health_check_url}: {str(e)}")
            logger.warning("Please ensure the SGLang service is running and accessible.")

    @retry(max_retries=5, sleep=2)
    def generate(
        self,
        input_list: list[str] | str,
        return_raw_output: bool = False,
        return_scores: bool = False,
        **params,
    ) -> list[str] | tuple[list[str], list[list[float]]] | list[dict]:
        """Generates text using the remote SGLang service."""
        if isinstance(input_list, str):
            input_list = [input_list]
        if not isinstance(input_list, list) or not all(isinstance(item, str) for item in input_list):
            raise ValueError("Input must be a string or a list of strings.")

        batch_size = len(input_list)
        data_to_remote = {"text": input_list}

        effective_params = deepcopy(self.generation_params)
        effective_params.update(params)

        curr_sampling_params = {}
        if effective_params.get("do_sample", True) is False:
            curr_sampling_params["temperature"] = 0.0
        else:
            curr_sampling_params["temperature"] = effective_params.get(
                "temperature", getattr(self.config, "temperature", 0.7)
            )

        default_max_tokens = getattr(self.config, "generator_max_output_len", 1024)
        curr_sampling_params["max_new_tokens"] = effective_params.get("max_new_tokens", default_max_tokens)

        stop_sequences = effective_params.get("stop", [])
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        if stop_sequences:
            curr_sampling_params["stop"] = stop_sequences

        keys_to_remove = ["do_sample", "temperature", "max_new_tokens", "stop"]
        for key in keys_to_remove:
            effective_params.pop(key, None)

        if "top_p" in effective_params:
            curr_sampling_params["top_p"] = effective_params["top_p"]
        if "top_k" in effective_params:
            curr_sampling_params["top_k"] = effective_params["top_k"]

        data_to_remote["sampling_params"] = curr_sampling_params

        if return_scores:
            data_to_remote["return_logprob"] = True
            data_to_remote["top_logprobs_num"] = getattr(self.config, "top_logprobs_num", 2)

        try:
            response = requests.post(
                self.sgl_remote_url, json=data_to_remote, timeout=120, headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            response_list = response.json()

            if return_raw_output:
                return response_list

            generated_text = []
            for item in response_list:
                text = item.get("text", "")
                finish_reason = item.get("meta_info", {}).get("finish_reason", {})
                matched_stop = finish_reason.get("matched")
                if matched_stop and curr_sampling_params.get("stop") and matched_stop in curr_sampling_params["stop"]:
                    text += matched_stop
                generated_text.append(text)

            if return_scores:
                scores = []
                for resp_item in response_list:
                    logprobs_list = resp_item.get("meta_info", {}).get("output_token_logprobs", [])
                    token_scores = [
                        np.exp(logprob[0]) if (logprob and len(logprob) > 0) else 0.0 for logprob in logprobs_list
                    ]
                    scores.append(token_scores)
                return generated_text, scores
            else:
                return generated_text

        except requests.exceptions.Timeout:
            logger.error("Generation request timed out after 120 seconds.")
            raise
        except requests.exceptions.ConnectionError:
            logger.error(f"Could not connect to remote generator service at {self.sgl_remote_url}.")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during generation: {str(e)}", exc_info=True)
            raise
        except json.JSONDecodeError as e:
            response_text = "Unknown (error occurred before response object assignment)"
            if "response" in locals() and hasattr(response, "text"):
                response_text = response.text[:500]
            logger.error(
                f"Failed to decode JSON response from {self.sgl_remote_url}. Response text: {response_text}...",
                exc_info=True,
            )
            raise
        except Exception as e:
            logger.error(f"Unexpected error during generation: {str(e)}", exc_info=True)
            raise


def load_dataset_items(config: Config, split: str) -> list[dict | object]:
    """Loads dataset items using flashrag's get_dataset."""
    logger.info(f"📚 Loading dataset: {config.dataset_name}, Split: {split}")
    try:
        all_splits = get_dataset(config)
        if split not in all_splits:
            logger.error(
                f"Split '{split}' not found in dataset '{config.dataset_name}'. Available splits: {list(all_splits.keys())}"
            )
            return []
        dataset_items = all_splits[split]
        logger.info(f"Successfully loaded {len(dataset_items)} items for split '{split}'.")

        return dataset_items
    except FileNotFoundError:
        logger.error(
            f"Dataset files not found for '{config.dataset_name}' in '{config.data_dir}'. Check config and paths."
        )
        return []
    except Exception as e:
        logger.error(f"Error loading dataset using get_dataset: {e}", exc_info=True)
        return []


def save_results(args: argparse.Namespace, config: Config, result_dataset, run_duration: float):
    """Saves summary and debug information."""
    logger.info("💾 Saving results...")
    summary_file = os.path.join(args.save_dir, f"{args.save_note}_summary.txt")
    debug_file = os.path.join(args.save_dir, f"{args.save_note}_debug.json")

    num_items = len(result_dataset)

    logger.info(f"Saving summary results to {summary_file}...")
    try:
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("EVALUATION SUMMARY\n")
            f.write("=================\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Run Duration: {run_duration:.2f} seconds\n")
            f.write(f"Dataset: {config.dataset_name} ({args.split} split)\n")
            f.write(f"Model: {config.generator_model_path}\n")
            f.write(f"Retriever: {config.remote_retriever_url}\n")
            f.write(f"Agent Max Generations: {getattr(config, 'agent_max_generations', 'N/A')}\n")
            f.write(f"Generator Max Output Len: {getattr(config, 'generator_max_output_len', 'N/A')}\n\n")
            f.write(f"Total items processed: {num_items}\n")
            f.write("\nNote: Verification was skipped in this run.\n")
            f.write("Note: Overall metrics (like EM, F1) are usually printed to console by evaluate method.\n")

        logger.info(f"✅ Summary saved to {summary_file}")
    except Exception as e:
        logger.error(f"Error saving summary file '{summary_file}': {e}", exc_info=True)

    logger.info(f"Saving debug information (predictions & responses) to {debug_file}...")
    try:
        debug_data = []
        for i, item in enumerate(result_dataset):
            item_data: dict[str, object] = {}

            def get_item_value(data_item, key_or_attr: str) -> str | int | float | list | bool | None:
                if isinstance(data_item, dict):
                    return data_item.get(key_or_attr)
                elif hasattr(data_item, key_or_attr):
                    return getattr(data_item, key_or_attr)
                return None

            item_data["item_index"] = i
            item_data["question"] = get_item_value(item, "question")
            item_data["prediction"] = get_item_value(item, "pred")
            item_data["final_response"] = get_item_value(item, "final_response")

            gt_answer_val = None
            try:
                gt_answer_val = get_item_value(item, "answer")
                if gt_answer_val is None:
                    answers_list = get_item_value(item, "answers")
                    if isinstance(answers_list, list) and answers_list:
                        raw_ans = answers_list[0]
                        if isinstance(raw_ans, (str, int, float, bool)):
                            gt_answer_val = raw_ans
                        else:
                            gt_answer_val = str(raw_ans)
                elif not isinstance(gt_answer_val, (str, int, float, bool)):
                    gt_answer_val = str(gt_answer_val)
            except Exception as e:
                logger.warning(f"Could not safely get ground truth for item {i}: {e}")
                gt_answer_val = "ERROR_GETTING_ANSWER"
            item_data["ground_truth"] = gt_answer_val

            eval_score_val = None
            try:
                eval_score_val = get_item_value(item, "score")
                if not isinstance(eval_score_val, (str, int, float, bool, type(None))):
                    eval_score_val = str(eval_score_val)
            except Exception as e:
                logger.warning(f"Could not safely get score for item {i}: {e}")
                eval_score_val = "ERROR_GETTING_SCORE"
            item_data["eval_score"] = eval_score_val

            debug_data.append(item_data)

        with open(debug_file, "w", encoding="utf-8") as f:
            json.dump(debug_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Debug information saved to {debug_file}")
    except Exception as e:
        logger.error(f"Error saving debug file '{debug_file}': {e}", exc_info=True)


def research(args: argparse.Namespace, config: Config):
    """Main function to run the research evaluation pipeline."""
    logger.info("🚀 Starting research pipeline execution...")
    start_time = time.time()

    test_data = load_dataset_items(config, args.split)
    if not test_data:
        logger.error("Failed to load test data. Exiting.")
        return

    try:
        logger.info("🏗️ Building ReSearchPipeline...")
        pipeline = ReSearchPipeline(config)
        logger.info("✅ Pipeline built successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize ReSearchPipeline: {e}", exc_info=True)
        return

    agent_max_generations = getattr(config, "agent_max_generations", 20)
    generator_max_output_len = getattr(config, "generator_max_output_len", 1024)

    try:
        logger.info("🏃 Starting pipeline run...")
        result_dataset = pipeline.run(test_data, do_eval=True)
        logger.info("✅ Pipeline run completed.")
    except Exception as e:
        logger.error(f"Error during pipeline run: {e}", exc_info=True)
        result_dataset = test_data
        logger.warning("Pipeline run failed, attempting to save inputs/partial results.")

    run_duration = time.time() - start_time
    logger.info(f"Total run duration: {run_duration:.2f} seconds.")
    save_results(args, config, result_dataset, run_duration)

    logger.info("🏁 Research pipeline execution finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running ReSearch Evaluation Pipeline")
    parser.add_argument(
        "--config_path", type=str, default="./eval_config.yaml", help="Path to the main FlashRAG config file."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="bamboogle",
        help="Name of the dataset (must match config or data_dir structure).",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to evaluate (e.g., test, validation)."
    )
    parser.add_argument("--save_dir", type=str, default="./output_logs", help="Directory to save logs and results.")
    parser.add_argument("--save_note", type=str, default="research_run", help="A note to prepend to saved filenames.")

    parser.add_argument("--data_dir", type=str, help="Override data directory specified in config.")
    parser.add_argument(
        "--sgl_remote_url", type=str, help="Override SGLang remote generator URL (e.g., localhost:8002)."
    )
    parser.add_argument(
        "--remote_retriever_url", type=str, help="Override remote retriever URL (e.g., localhost:8001)."
    )
    parser.add_argument("--generator_model_path", type=str, help="Override generator model path specified in config.")
    parser.add_argument("--retriever_topk", type=int, help="Override retriever top K.")
    parser.add_argument("--generator_max_output_len", type=int, help="Override generator max output length.")
    parser.add_argument("--agent_max_generations", type=int, help="Override agent max interaction turns.")

    args = parser.parse_args()
    logger.info(f"Starting evaluation script with arguments: {args}")

    try:
        os.makedirs(args.save_dir, exist_ok=True)
        logger.info(f"💾 Logs and results will be saved to: {args.save_dir}")
    except OSError as e:
        logger.error(f"Could not create save directory '{args.save_dir}': {e}", exc_info=True)
        exit(1)

    config_overrides = {
        k: v
        for k, v in vars(args).items()
        if v is not None
        and k
        not in [
            "config_path",
            "dataset_name",
            "split",
            "save_dir",
            "save_note",
        ]
    }

    logger.info(f"🔧 Loading configuration from: {args.config_path}")
    try:
        config = Config(args.config_path, config_dict=config_overrides)
        config.dataset_name = args.dataset_name
        if args.data_dir:
            config.data_dir = args.data_dir

        logger.info(f"Effective data_dir: {getattr(config, 'data_dir', 'N/A')}")
        logger.info(f"Effective generator_model_path: {getattr(config, 'generator_model_path', 'N/A')}")
        logger.info(f"Effective sgl_remote_url: {getattr(config, 'sgl_remote_url', 'N/A')}")
        logger.info(f"Effective remote_retriever_url: {getattr(config, 'remote_retriever_url', 'N/A')}")

        logger.info("✅ Config loaded and potentially overridden by CLI arguments.")

        config["dataset_path"] = os.path.join(config.data_dir, config.dataset_name)

    except FileNotFoundError:
        logger.error(f"Config file not found at '{args.config_path}'. Please check the path.")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading or processing configuration: {e}", exc_info=True)
        exit(1)

    research(args, config)
