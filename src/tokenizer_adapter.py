"""
Tokenizer adapter implementations for different models.
This module provides adapter classes for handling different tokenizer formats.
"""

from abc import ABC, abstractmethod

import torch

from src.config import logger


class TokenizerAdapter(ABC):
    """Base class for tokenizer adapters."""

    @abstractmethod
    def get_assistant_marker(self) -> str:
        """Get the assistant marker for the model."""
        pass

    @abstractmethod
    def get_end_marker(self) -> str:
        """Get the end marker for the model."""
        pass

    @abstractmethod
    def get_mask(self, text: str, tokenizer) -> torch.Tensor:
        """Get the mask for the model's response."""
        pass

    @abstractmethod
    def split_prompt_assistant(self, text: str) -> tuple[str, str]:
        """Split conversation text into prompt and assistant response."""
        pass


class LlamaTokenizerAdapter(TokenizerAdapter):
    """Adapter for Llama model tokenizer."""

    def get_assistant_marker(self) -> str:
        """Get the assistant marker."""
        return "<|start_header_id|>assistant<|end_header_id|>"

    def get_end_marker(self) -> str:
        """Get the end marker."""
        return "<|eot_id|>"

    def split_prompt_assistant(self, convo_text: str) -> tuple[str, str]:
        """Split the text into prompt and assistant parts.

        Args:
            convo_text: The text to split

        Returns:
            A tuple of (prompt, assistant)
        """
        # EXACT replication from rl_helpers.py but using existing method
        marker = self.get_assistant_marker()  # Use existing method but same value
        idx = convo_text.find(marker)
        if idx == -1:
            raise ValueError("Could not find assistant marker in conversation text.")
            return convo_text, ""

        # Include the marker in the prompt by slicing up to the end of the marker.
        prompt = convo_text[: idx + len(marker)]
        # The assistant response is everything after the marker.
        assistant_response = convo_text[idx + len(marker) :]
        return prompt, assistant_response

    def get_mask(self, text: str, tokenizer) -> torch.Tensor:
        """Get the mask for the text.

        Args:
            text: The text to get the mask for
            tokenizer: The tokenizer to use

        Returns:
            A tensor of 0s and 1s where 1s indicate assistant tokens
        """
        # Log input
        logger.debug(f"🔍 Llama: Full text length: {len(text)}")

        # EXACT replication from rl_helpers.py but using existing methods
        encoding = tokenizer(text, add_special_tokens=False)
        # Use existing methods but same values
        start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
        assistant_token = tokenizer.convert_tokens_to_ids("assistant")
        end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
        eot_id = tokenizer.convert_tokens_to_ids(self.get_end_marker())  # Use existing method but same value

        # Log token IDs
        logger.debug(f"🔍 Llama: Tokenized length: {len(encoding.input_ids)}")
        logger.debug(f"🔍 Llama: Input IDs: {encoding.input_ids}")
        logger.debug(
            f"🔍 Llama: Special token IDs: start={start_header_id}, assistant={assistant_token}, end={end_header_id}, eot={eot_id}"
        )

        assistant_ranges = []
        i = 0
        while i < len(encoding.input_ids) - 1:
            if encoding.input_ids[i] == start_header_id and encoding.input_ids[i + 1] == assistant_token:
                logger.debug(f"🔍 Llama: Found assistant marker at position {i}")
                logger.debug(f"🔍 Llama: Assistant marker tokens: {encoding.input_ids[i : i + 2]}")
                i += 2
                while i < len(encoding.input_ids) and encoding.input_ids[i] != end_header_id:
                    i += 1
                i += 2
                start_idx = i
                logger.debug(f"🔍 Llama: Found start of response at {start_idx}")
                logger.debug(f"🔍 Llama: Start token ID: {encoding.input_ids[start_idx]}")
                while i < len(encoding.input_ids) and encoding.input_ids[i] != eot_id:
                    i += 1
                end_idx = i
                logger.debug(f"🔍 Llama: Found end of response at {end_idx}")
                logger.debug(f"🔍 Llama: End token ID: {encoding.input_ids[end_idx]}")
                logger.debug(f"🔍 Llama: Response token IDs: {encoding.input_ids[start_idx:end_idx]}")
                assistant_ranges.append((start_idx, end_idx))
            else:
                i += 1

        mask = [0] * len(encoding.input_ids)
        for start_idx, end_idx in assistant_ranges:
            for idx in range(start_idx, end_idx):
                mask[idx] = 1

        mask = torch.tensor(mask, dtype=torch.int)

        # Log final mask
        logger.debug(f"🔍 Llama: Final mask shape: {mask.shape}")
        logger.debug(f"🔍 Llama: Mask sum: {mask.sum().item()}")
        logger.debug(f"🔍 Llama: Mask: {mask}")

        # Additional debug info
        try:
            prompt, response = self.split_prompt_assistant(text)
            prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
            response_tokens = tokenizer(response, add_special_tokens=False).input_ids

            logger.debug(f"🔍 Llama: Prompt length: {len(prompt)}")
            logger.debug(f"🔍 Llama: Response length: {len(response)}")
            logger.debug(f"🔍 Llama: Prompt token IDs: {prompt_tokens}")
            logger.debug(f"🔍 Llama: Response token IDs: {response_tokens}")
            logger.debug(f"🔍 Llama: Prompt: {prompt[:100]}...")
            logger.debug(f"🔍 Llama: Response: {response[:100]}...")
            logger.debug(f"🔍 Llama: Full input IDs length: {len(encoding.input_ids)}")
            logger.debug(f"🔍 Llama: Prompt + Response token IDs length: {len(prompt_tokens) + len(response_tokens)}")
            logger.debug(
                f"🔍 Llama: Difference in lengths: {len(encoding.input_ids) - (len(prompt_tokens) + len(response_tokens))}"
            )
        except Exception as e:
            logger.error(f"🔍 Llama: Error splitting prompt/response: {e}")

        return mask


class R1DistilTokenizerAdapter(TokenizerAdapter):
    """Adapter for R1-Distil model tokenizer."""

    def get_assistant_marker(self) -> str:
        marker = "<｜Assistant｜>"
        return marker

    def get_end_marker(self) -> str:
        marker = "<｜end▁of▁sentence｜>"
        return marker

    def get_begin_marker(self) -> str:
        return "<｜begin▁of▁sentence｜>"

    def get_user_marker(self) -> str:
        return "<｜User｜>"

    def get_mask(self, text: str, tokenizer) -> torch.Tensor:
        """Get the mask for the text.

        Args:
            text: The text to get the mask for
            tokenizer: The tokenizer to use

        Returns:
            A tensor of 0s and 1s where 1s indicate assistant tokens
        """
        logger.debug(f"🔍 R1Distil: Getting mask for text length: {len(text)}")

        # Get all markers
        assistant_marker = self.get_assistant_marker()
        end_marker = self.get_end_marker()

        # Get the full tokenization
        encoding = tokenizer(text, add_special_tokens=False)
        tokens = encoding.input_ids
        logger.debug(f"🔍 R1Distil: Full text token IDs: {tokens}")

        # Create mask initialized to zeros - ENSURE SAME LENGTH AS INPUT_IDS
        mask = torch.zeros(len(tokens), dtype=torch.int)

        # Get token IDs for markers
        assistant_tokens = tokenizer(assistant_marker, add_special_tokens=False).input_ids
        end_tokens = tokenizer(end_marker, add_special_tokens=False).input_ids
        logger.debug(f"🔍 R1Distil: Assistant marker token IDs: {assistant_tokens}")
        logger.debug(f"🔍 R1Distil: End marker token IDs: {end_tokens}")

        # Find all assistant responses
        assistant_ranges = []
        i = 0
        while i < len(tokens):
            # Look for assistant marker
            if i + len(assistant_tokens) <= len(tokens) and tokens[i : i + len(assistant_tokens)] == assistant_tokens:
                logger.debug(f"🔍 R1Distil: Found assistant marker at position {i}")

                # Start masking AFTER the assistant marker
                start_idx = i + len(assistant_tokens)

                # Find end marker
                end_idx = None
                j = start_idx
                while j < len(tokens):
                    if j + len(end_tokens) <= len(tokens) and tokens[j : j + len(end_tokens)] == end_tokens:
                        end_idx = j  # Don't include the end marker in the mask
                        break
                    j += 1

                if end_idx is None:
                    # If no end marker found, mask until the end
                    end_idx = len(tokens)

                logger.debug(f"🔍 R1Distil: Response range: {start_idx} to {end_idx}")
                assistant_ranges.append((start_idx, end_idx))
                i = end_idx + len(end_tokens)  # Start next search after the end marker
            else:
                i += 1

        # Apply mask for all found ranges
        for start_idx, end_idx in assistant_ranges:
            mask[start_idx:end_idx] = 1

        logger.debug(f"🔍 R1Distil: Found {len(assistant_ranges)} assistant responses")
        logger.debug(f"🔍 R1Distil: Final mask sum: {mask.sum().item()}")
        logger.debug(f"🔍 R1Distil: Final mask length: {len(mask)}")
        logger.debug(f"🔍 R1Distil: Mask: {mask}")

        return mask

    def split_prompt_assistant(self, text: str) -> tuple[str, str]:
        """Split the text into prompt and assistant parts.

        Args:
            text: The text to split

        Returns:
            A tuple of (prompt, assistant)
        """
        logger.debug(f"🔍 R1Distil: Splitting text of length: {len(text)}")

        # Find the assistant marker
        marker = self.get_assistant_marker()
        end_marker = self.get_end_marker()

        # Find ALL assistant markers in the text
        assistant_markers = []
        pos = 0
        while True:
            pos = text.find(marker, pos)
            if pos == -1:
                break
            assistant_markers.append(pos)
            pos += len(marker)

        if not assistant_markers:
            raise ValueError("Could not find assistant marker in text")

        # Get the positions of all markers for later use
        marker_positions = []
        for start_pos in assistant_markers:
            response_start = start_pos + len(marker)

            # Find the end marker after this response
            end_pos = text.find(end_marker, response_start)
            if end_pos == -1:
                end_pos = len(text)
            else:
                end_pos = end_pos + len(end_marker)

            marker_positions.append((start_pos, response_start, end_pos))

        # Get the full response (all assistant outputs concatenated)
        full_response = ""
        for _, resp_start, resp_end in marker_positions:
            full_response += text[resp_start:resp_end]

        # Include ALL assistant markers and responses in the response
        # This matches how the mask is generated in get_mask
        first_assistant_pos = marker_positions[0][0]
        last_response_end = marker_positions[-1][2]

        # Split into prompt and response
        prompt = text[:first_assistant_pos]  # Everything before the first assistant marker
        response = text[first_assistant_pos:last_response_end]  # All markers and responses

        logger.debug(f"🔍 R1Distil: Prompt length: {len(prompt)}")
        logger.debug(f"🔍 R1Distil: Response length: {len(response)}")
        logger.debug(f"🔍 R1Distil: Response token count estimate: {len(response) / 4}")  # Rough estimate
        logger.debug(f"🔍 R1Distil: Final prompt: {prompt[:100]}...")
        logger.debug(f"🔍 R1Distil: Final response: {response[:100]}...")

        return prompt, response
