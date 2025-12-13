# LLM-Powered Scene Generator 

## Overview

We look to create an agent that transforms natural language descriptions into 3D scenes using A-Frame, an HTML-based web framework for building AR/VR scenes directly in the browser (these can be ported over to headsets). It uses a declarative and reusable, entity-component structure that allows one to call core/primitive and community-built components that can be further customized. For a given user prompt, the applicable components and their associated variants can be extracted and then used in the generation. 

## Directory Structure

```
## Directory Structure
- extra                      # Contains a few-shot/guided prompt example we wanted to try initially.
- qwen_sft_vr_run2           # Final, fine-tuned model using LoRA-assisted SFT.
- samples                    # Synthetic generated data samples.
  - scenes                   # Correct code samples separated out into individual HTML files.
  - scenes.json              # User-prompt, correct code sample pairs in an aggregated format for training.
- wandb                      # WandB contents used for tracking evaluation metrics during training (loss, learning rate, etc.).
- completions.json           # Contains a recent subset of completions on validation data using the fine-tuned model.
- data_generation.ipynb      # Used to generate the synthetic data samples using descriptive adjectives and combinatorics for state variables.
- eval.py                    # Used to generate completions and calculate Pass@1.
- integrations.py            # Script for implementing Phase 2 (Retrieval-Augmented Generation (RAG) and gITF Integration).
- qwen_sft_vr_run2.zip       # Compressed .zip for fine-tuned model to be able to faciliate loading in Google Colab.
- sft.ipynb                  # Used to prepare data for training, specify parameters, and execute SFT.
```
