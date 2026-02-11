import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling_p5 import P5

class P5Pretraining(P5):
    def __init__(self, config):
        super().__init__(config)

        self.losses = self.config.losses.split(',')

    def train_step(self, batch):

        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        whole_word_ids = batch['whole_word_ids'].to(device)

        lm_labels = batch["target_ids"].to(device)

        loss_weights = batch["loss_weights"].to(device)

        # Debug: Check for out-of-vocabulary token IDs
        vocab_size = self.config.vocab_size
        if (input_ids >= vocab_size).any():
            print(f"WARNING: input_ids contains values >= vocab_size ({vocab_size})")
            print(f"Max input_id: {input_ids.max().item()}")
        if (lm_labels[lm_labels != -100] >= vocab_size).any():
            print(f"WARNING: labels contains values >= vocab_size ({vocab_size})")
            print(f"Max label: {lm_labels[lm_labels != -100].max().item()}")

        output = self(
            input_ids=input_ids,
            whole_word_ids=whole_word_ids,
            labels=lm_labels,
            return_dict=True
        )
        assert 'loss' in output

        lm_mask = lm_labels != -100
        lm_mask = lm_mask.float()
        B, L = lm_labels.size()

        loss = output['loss']

        # Debug: Check for NaN in raw loss
        if torch.isnan(loss).any():
            print(f"WARNING: NaN detected in raw loss")
            print(f"Number of valid labels: {lm_mask.sum().item()}")
            print(f"Sample source text: {batch['source_text'][0] if 'source_text' in batch else 'N/A'}")
            print(f"Sample target text: {batch['target_text'][0] if 'target_text' in batch else 'N/A'}")

        loss = loss.view(B, L) * lm_mask

        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        results = {}

        results['loss'] = (loss * loss_weights).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1

        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]

        return results

    @torch.no_grad()
    def valid_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        whole_word_ids = batch['whole_word_ids'].to(device)

        lm_labels = batch["target_ids"].to(device)

        loss_weights = batch["loss_weights"].to(device)

        output = self(
            input_ids=input_ids,
            whole_word_ids=whole_word_ids,
            labels=lm_labels,
            return_dict=True
        )
        assert 'loss' in output

        lm_mask = lm_labels != -100
        lm_mask = lm_mask.float()
        B, L = lm_labels.size()

        loss = output['loss']

        loss = loss.view(B, L) * lm_mask

        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)

        results = {}

        results['loss'] = (loss * loss_weights).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1

        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]

        if 'rating' in self.losses:
            output = self.generate(
                input_ids=input_ids
            )

            generated_score = self.tokenizer.batch_decode(output, skip_special_tokens=True)

            results['rating_pred'] = generated_score

        return results

    @torch.no_grad()
    def generate_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)

        output = self.generate(
            input_ids=input_ids,
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return generated_sents
