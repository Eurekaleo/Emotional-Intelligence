# HitEmotion

A holistic benchmark and toolkit for evaluating and enhancing the cognition-based emotional understanding capabilities of Multimodal Large Language Models (MLLMs) grounded by Theory-of-Mind (ToM).

## Repository Structure

### `/data`
-   **/without_prompt**: Contains the raw dataset samples for all tasks in JSON format.
-   **/with_prompt**: Contains dataset samples enhanced with ToM-driven prompts for all tasks, also in JSON format.

### `/code`
-   **/test_code**: Contains execution scripts for all models.
-   **/evaluation_code**: Contains evaluation scripts for all tasks.
-   **/train_code**: Contains scripts for generating our Chain-of-Thought (CoT) training data.

*Other directories in the repository contain code for Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL).*
