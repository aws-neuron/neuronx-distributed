import copy
from typing import List, Optional, Union

import torch
from transformers.generation.stopping_criteria import StoppingCriteriaList

from neuronx_distributed.utils.medusa_utils import (
    evaluate_posterior,
    generate_candidates,
    generate_medusa_buffers,
    update_inference_inputs,
)


class NeuronSpeculation:
    def _assisted_decoding(
        self,
        input_ids: torch.LongTensor,
        candidate_generator: "CandidateGenerator", #noqa
        do_sample: bool = False,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        **model_kwargs,
    ):
        if do_sample:
            raise ValueError("Sampling is unsupported as part of speculation. Only greedy speculation is supported.")

        assistant_model = candidate_generator.assistant_model
        if self.config.is_medusa:
            # TODO: move this to sampling
            return self._medusa_assisted_decoding(
                input_ids, assistant_model, stopping_criteria, pad_token_id, eos_token_id, **model_kwargs
            )
        else:
            return self._standard_assisted_decoding(
                input_ids, assistant_model, stopping_criteria, pad_token_id, eos_token_id, **model_kwargs
            )

    def _standard_assisted_decoding(
        self, input_ids, assistant_model, stopping_criteria, pad_token_id, eos_token_id, **model_kwargs
    ):
        # Implementation of standard assisted decoding

        # Initialize the num_assistant_tokens used for speculation.
        if hasattr(assistant_model, "num_assistant_tokens"):
            num_assistant_tokens = assistant_model.num_assistant_tokens
        else:
            num_assistant_tokens = assistant_model.generation_config.num_assistant_tokens

        # Init values
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if eos_token_id is not None and pad_token_id is None:
            raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        assistant_model = candidate_generator.assistant_model

        # Prepare assistant model's keys of inputs
        assistant_kwargs = copy.deepcopy(model_kwargs)

        # Other auxiliary variables
        max_len = stopping_criteria[0].max_length
        cur_len = input_ids.shape[-1]
        spec_len = self.config.speculation_length

        # Run the target model once and get the first generated token
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = self(**model_inputs)

        curr_pos = model_inputs["position_ids"][0].argmax(dim=-1)
        new_token = outputs.logits[:, 0].argmax(dim=-1, keepdim=True)

        # Prepare the input ids and attention mask for the draft model
        candidate_input_ids = input_ids

        # This is the finally return outputs; append the first generated token
        returned_ids = torch.cat((input_ids[:, : curr_pos + 1], new_token), dim=1)

        # Speculation loop
        while True:
            # 1 Token generation using draft model
            for _ in range(int(num_assistant_tokens)):
                # 1.1 Prepare assistant model inputs
                assistant_inputs = assistant_model.prepare_inputs_for_generation(
                    candidate_input_ids,
                    **assistant_kwargs,
                )
                is_for_token_generation = assistant_model.kv_cache_populated

                # 1.2 Use the assistant model to obtain the next candidate logits
                assistant_model_outputs = assistant_model(**assistant_inputs)
                assistant_new_token = assistant_model_outputs.logits[:, 0, :].argmax(dim=-1)

                # 1.3 Update inputs and args for next iteration
                candidate_input_ids = torch.cat((candidate_input_ids, assistant_new_token[:, None]), dim=-1)
                assistant_kwargs = assistant_model._update_model_kwargs_for_generation(
                    assistant_model_outputs,
                    assistant_kwargs,
                    is_for_token_generation,
                    is_encoder_decoder=assistant_model.config.is_encoder_decoder,
                )

                # 1.4 Stop assistant generation on EOS
                if eos_token_id_tensor is not None:
                    last_assistant_token_is_eos = assistant_new_token.tile(eos_token_id_tensor.shape[0], 1)
                    last_assistant_token_is_eos = (
                        ~last_assistant_token_is_eos.ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0).bool()
                    )
                    if last_assistant_token_is_eos:
                        break
                else:
                    last_assistant_token_is_eos = False

            # 2 Validation of draft model output using the original model
            #   The length could be shorter if the draft loop ends earlier
            candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]

            # 2.1 Prepare the input arguments
            input_ids = torch.cat((new_token, candidate_input_ids[:, -candidate_length:-1]), dim=-1)
            attention_mask = model_inputs["attention_mask"]
            pos = curr_pos + 1
            position_ids = torch.arange(pos, pos + spec_len).expand(1, spec_len)
            # Pad the input_ids if needed
            if input_ids.shape[-1] < spec_len:
                input_ids = torch.cat(
                    (input_ids, torch.full((1, spec_len - input_ids.shape[-1]), pad_token_id)), dim=-1
                )

            # 2.2. Run a forward pass on the candidate sequence
            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            # 2.3. Process the new logits
            new_tokens = outputs.logits.argmax(dim=-1)
            selected_tokens = outputs.logits[:, : candidate_length - 1].argmax(dim=-1)

            # 3. Compare the argmax from the original model logits with the assistant forecasted tokens. We can keep
            # the assistant forecasted tokens until the first mismatch, or until the max length is reached.
            candidate_new_tokens = candidate_input_ids[:, -candidate_length:-1]
            n_matches = ((~(candidate_new_tokens == selected_tokens)).cumsum(dim=-1) < 1).sum()

            # 4. Ensure we don't generate beyond max_len or an EOS token
            if last_assistant_token_is_eos and n_matches == candidate_length:
                n_matches -= 1
            n_matches = min(n_matches, max_len - cur_len - 1)
            # n_matches = 4

            # 5. Get the valid continuation, after the matching tokens. We also consider the extra token
            # generated by the original model. Update the return ids accordingly
            valid_tokens = new_tokens[:, : n_matches + 1]
            returned_ids = torch.cat((returned_ids, valid_tokens), dim=1)
            # if last_assistant_token_is_eos and n_matches == candidate_length-1:
            #    break;

            # 6. Update the args for the next iteration.
            #    Feed the last correct token to the next loop
            new_token = valid_tokens[:, -1:]
            if new_token[0] == torch.tensor(eos_token_id):
                break
            input_ids = valid_tokens[:, -1:]
            candidate_input_ids = valid_tokens[:, -1:]
            model_inputs_attn_mask = model_inputs["attention_mask"]
            n_matches_concat_tensor = torch.zeros(1, n_matches + 1, dtype=model_inputs_attn_mask.dtype)
            model_inputs_attn_mask = torch.cat([model_inputs_attn_mask, n_matches_concat_tensor], dim=-1)
            model_inputs["attention_mask"] = model_inputs_attn_mask.index_fill(
                1, torch.arange(curr_pos + 1, curr_pos + 1 + n_matches + 1), 1
            )

            curr_pos = curr_pos + n_matches + 1
            assistant_kwargs["attention_mask"] = copy.deepcopy(model_inputs["attention_mask"])

            # 7. Update with the generated token length and check for stopping condition.
            cur_len = cur_len + n_matches + 1
            if cur_len >= max_len:
                break
            # 8. If the rest length is smaller than speculation length, we directly run the target model to finish
            if max_len - cur_len < spec_len:
                # @yihsian: TODO: complete with using target tokengen model
                break

        return returned_ids

    def _medusa_assisted_decoding(
        self, input_ids, assistant_model, stopping_criteria, pad_token_id, eos_token_id, **model_kwargs
    ):
        medusa_kwargs = copy.deepcopy(model_kwargs)

        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

        mc_sim_7b_63 = self.config.medusa_tree

        medusa_buffers = generate_medusa_buffers(mc_sim_7b_63)

        model_inputs = self.prepare_inputs_for_generation(input_ids, **medusa_kwargs)

        outputs = self(**model_inputs)

        non_zero_input_ids = input_ids.nonzero()
        cur_len = torch.tensor([non_zero_input_ids.size(0)], dtype=torch.int64)

        logits, medusa_logits = self._extract_logits(outputs)

        medusa_logits = medusa_logits[:, :, None, :]

        accept_length = 0
        final_accept_length = 0
        new_token = 0
        accept_lengths_tree = []
        cur_length = cur_len[0].item() + 1
        accept_lengths_tree.append(1)
        count = 0
        select_indices = torch.arange(
            cur_len[0].item(), cur_len[0].item() + self.config.num_medusa_heads + 1, dtype=torch.int64
        )

        for i in range(self.config.max_new_tokens):
            count = count + 1
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
            )
            position_ids = medusa_buffers["medusa_position_ids"] + input_ids.nonzero().shape[0]

            medusa_kwargs = self._prepare_medusa_kwargs(
                position_ids, cur_len, medusa_buffers, select_indices, medusa_kwargs
            )

            tree_candidates = tree_candidates.long()

            model_inputs = self.prepare_medusa_inputs_for_generation(tree_candidates, **medusa_kwargs)

            outputs = self(**model_inputs)

            tree_logits, tree_medusa_logits = self._extract_logits(outputs)

            logits = tree_logits[0, medusa_buffers["retrieve_indices"]]
            medusa_logits = tree_medusa_logits[:, 0, medusa_buffers["retrieve_indices"]]

            best_candidate, accept_length = evaluate_posterior(logits, candidates)
            cur_len = torch.tensor([input_ids.nonzero().size(0) - 1], dtype=torch.int64)

            input_ids, logits, medusa_logits, new_token, select_indices = update_inference_inputs(
                input_ids[:, : (int(cur_len[0] + 1))],
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
            )

            medusa_kwargs["attention_mask"] = self._update_attention_mask(
                model_inputs, accept_length, cur_len, medusa_kwargs
            )
            cur_len = 1 + cur_len
            accept_length_tree = input_ids.shape[1] - cur_length
            cur_length = accept_length_tree + cur_length
            accept_lengths_tree.append(accept_length_tree)
            final_accept_length += accept_length + 1
            if eos_token_id in new_token or final_accept_length > self.config.max_new_tokens:
                break
        return input_ids

    def _prepare_medusa_kwargs(self, position_ids, cur_len, medusa_buffers, select_indices, medusa_kwargs):
        medusa_kwargs["position_ids"] = position_ids.unsqueeze(0)
        medusa_kwargs["accepted_indices"] = torch.arange(
            cur_len[0].item(), cur_len[0].item() + self.config.num_medusa_heads + 1, dtype=torch.int64
        )
        for index, value in enumerate(select_indices):
            medusa_kwargs["accepted_indices"][index] = value
        medusa_kwargs["accepted_indices"] = medusa_kwargs["accepted_indices"].unsqueeze(0)
        medusa_kwargs["current_length"] = torch.arange(
            cur_len[0].item(), cur_len[0].item() + self.config.num_medusa_heads + 1, dtype=torch.int64
        ).unsqueeze(0)
        medusa_mask = medusa_buffers["medusa_attn_mask"].unsqueeze(0)
        medusa_kwargs["medusa_mask"] = medusa_mask.type_as(torch.LongTensor())
        medusa_kwargs["scatter_index"] = torch.arange(
            position_ids[0], position_ids[0] + self.config.medusa_speculation_length, dtype=torch.int64
        ).unsqueeze(0)
        return medusa_kwargs

    def _update_attention_mask(self, model_inputs, accept_length, cur_len, medusa_kwargs):
        accept_length_concat_tensor = torch.zeros(1, accept_length + 1, dtype=model_inputs["attention_mask"].dtype)
        attn_mask = torch.cat([model_inputs["attention_mask"], accept_length_concat_tensor], dim=-1)

        medusa_kwargs["attention_mask"] = attn_mask.index_fill(
            1, torch.arange(int(cur_len[0]) + 1, int(cur_len[0]) + 1 + accept_length + 1), 1
        )
        return medusa_kwargs["attention_mask"]

    def _extract_logits(self, outputs):
        logits = outputs["hidden_states"][:1, :, :]
        medusa_logits = outputs["hidden_states"][1:, :, :].unsqueeze(1)
        return logits, medusa_logits
