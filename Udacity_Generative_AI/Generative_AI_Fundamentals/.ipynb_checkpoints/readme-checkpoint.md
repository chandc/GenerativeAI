## Instructions
### Project Instructions
To pass this project, your code must:

+ Load a pre-trained model and evaluate its performance
+ Perform parameter-efficient fine-tuning using the pre-trained model
+ Perform inference using the fine-tuned model and compare its performance to the original model

### Getting Started
This project is fairly open-ended. As long as you follow the prescribed steps, you may choose any appropriate PEFT technique, model, evaluation approach, and fine-tuning dataset.

+ PEFT technique
  - The PEFT technique covered in this course was LoRA, but new techniques are continuously being developed. See the [PEFT README](https://github.com/huggingface/peft) for links to the papers behind each of the supported techniques.
  - If you are unsure, we recommend using LoRA as your PEFT technique. LoRA is the only PEFT technique that is compatible with all models at this time.
+ Model
  - Your choice of model will depend on your choice of PEFT technique.
  - Unless you plan to use your own hardware/GPU rather than the Udacity Workspace, it's best to choose a smaller model.
  - The model must be compatible with a sequence classification task.
  - If you are unsure, we recommend using GPT-2 as your model. This is a relatively small model that is compatible with sequence classification and LoRA.

For a high-level overview of the supported models and PEFT techniques for each task, refer to the [PEFT README - model support matrix](https://github.com/huggingface/peft?tab=readme-ov-file#models-support-matrix). For specific model names in the Hugging Face registry, you can use the widget at the bottom of the [PEFT documentation homepage](https://huggingface.co/docs/peft/index) (select "sequence classification" from the drop-down).

+ Evaluation approach
  - The evaluation approach covered in this course was the `evaluate`` method with a Hugging Face `Trainer``. You may use the same approach, or any other reasonable evaluation approach for a sequence classification task
  - The key requirement for the evaluation is that you must be able to compare the original foundation model's performance and the fine-tuned model's performance.
+ Dataset
  - Your PEFT process must use a dataset from Hugging Face's `datasets`` library. As with the selection of model, you will need to ensure that the dataset is small enough that it is usable in the Udacity Workspace.
  - The key requirement for the dataset is that it matches the task. Follow this link to [view Hugging Face datasets filtered by the text classification task](https://huggingface.co/datasets?task_categories=task_categories:text-classification)

## Loading and Evaluating a Foundation Model
### Loading the model
Once you have selected a model, load it in your notebook.

### Evaluating the model
Perform an initial evaluation of the model on your chosen sequence classification task. This step will require that you also load an appropriate tokenizer and dataset.

## Performing Parameter-Efficient Fine-Tuning
### Creating a PEFT config
Create a PEFT config with appropriate hyperparameters for your chosen model.

### Creating a PEFT model
Using the PEFT config and foundation model, create a PEFT model.

### Training the model
Using the PEFT model and dataset, run a training loop with at least one epoch.

### Saving the trained model
Depending on your training loop configuration, your PEFT model may have already been saved. If not, use `save_pretrained`` to save your progress.

## Performing Inference with a PEFT Model
### Loading the model
Using the appropriate PEFT model class, load your trained model.

### Evaluating the model
Repeat the previous evaluation process, this time using the PEFT model. Compare the results to the results from the original foundation model.

## Submission Instructions
Projects may be submitted using the Project Workspace or by uploading a zip file or sharing a GitHub repository link on the Project Submission page.

